from __future__ import annotations

import asyncio
import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from open_webui.config import RAG_EMBEDDING_CONTENT_PREFIX, RAG_EMBEDDING_QUERY_PREFIX
from open_webui.constants import ERROR_MESSAGES
from open_webui.events import EVENTS, publish_event
from open_webui.internal.db import get_async_session
from open_webui.models.config import Config
from open_webui.models.memories import Memories, MemoryModel
from open_webui.models.users import Users
from open_webui.models.memory_proposals import MemoryProposalModel, MemoryProposals
from open_webui.retrieval.vector.async_client import ASYNC_VECTOR_DB_CLIENT
from open_webui.utils.access_control import has_permission
from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.utils.memory import (
    clean_memory_content,
    clean_memory_path,
    list_memory_path_groups,
    memory_vector_text,
    read_memory_path_rows,
    search_memory_rows,
    validate_memory_operations,
)
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

log = logging.getLogger(__name__)

router = APIRouter()


async def check_memories_permission(user):
    config = await Config.get_many('memories.enable', 'user.permissions')
    if not config.get('memories.enable'):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=ERROR_MESSAGES.NOT_FOUND,
        )

    if user.role != 'admin' and not await has_permission(user.id, 'features.memories', config.get('user.permissions')):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )


def background_review_requires_approval(metadata: dict | None) -> bool:
    """Whether this turn's background-review writes must be confirmed.

    Follows the same shape as `tool_approval_mode`, the human-in-the-loop
    control this project already has: a mode carried in `params`, resolved
    per-chat with a fallback to the user's own default
    (`params ?? settings.params` on the client). `auto` is the existing
    behaviour and the default, so nobody acquires a review queue they did not
    ask for; `ask` queues instead of storing.

    A mode rather than a boolean, and a param rather than a new settings key,
    because that is what the equivalent control already looks like here.
    """
    params = (metadata or {}).get('params') or {}
    return params.get('memory_approval_mode') == 'ask'


############################
# GetMemories
# Let what is remembered here spare someone the cost
# of learning it twice.
############################


@router.get('/', response_model=list[MemoryModel])
async def get_memories(
    request: Request,
    user=Depends(get_verified_user),
    db: AsyncSession = Depends(get_async_session),
):
    await check_memories_permission(user)

    return await Memories.get_memories_by_user_id(user.id, db=db)


############################
# AddMemory
############################


class AddMemoryForm(BaseModel):
    content: str
    type: Literal['user', 'context'] = 'context'
    path: str | None = None


class MemoryUpdateModel(BaseModel):
    content: str | None = None
    type: Literal['user', 'context'] | None = None
    path: str | None = None


class MemoryOperationModel(BaseModel):
    action: Literal['add', 'replace', 'remove', 'move']
    id: str | None = None
    content: str | None = None
    type: Literal['user', 'context'] | None = None
    path: str | None = None


class UpdateMemoriesForm(BaseModel):
    operations: list[MemoryOperationModel]
    # `review_approved` is a background-review write a human confirmed. It is
    # distinct from `tool` so the provenance in `meta.created_by` survives the
    # round trip through the queue — otherwise an approved memory would be
    # indistinguishable from one the model wrote mid-conversation.
    source: Literal['tool', 'background_review', 'review_approved'] | None = None


class ReviewedMemoryModel(BaseModel):
    """One entry as the human left it after reviewing.

    Not the proposal: the text may have been edited, and an entry the review
    never suggested may have been added outright. Only `action` and the
    optional `id` tie it back to a proposed operation.
    """

    action: Literal['add', 'replace', 'remove', 'move'] = 'add'
    id: str | None = None
    content: str | None = None
    type: Literal['user', 'context'] = 'context'
    path: str | None = None


class ApplyMemoryProposalsForm(BaseModel):
    """The reviewed set, as the desired final outcome of the batch.

    Whatever is listed is applied; every queued proposal is then cleared,
    accepted or not. So rejecting is simply omitting, and the human can add
    entries the review never proposed in the same pass.
    """

    memories: list[ReviewedMemoryModel] = []


class SearchMemoriesForm(BaseModel):
    query: str | None = None
    type: Literal['user', 'context', 'all'] = 'all'
    path: str | None = None
    memory_id: str | None = None
    limit: int = 20


class ListMemoryPathsForm(BaseModel):
    query: str | None = None
    type: Literal['user', 'context', 'all'] = 'all'
    limit: int = 100


class ReadMemoryPathForm(BaseModel):
    path: str
    type: Literal['user', 'context', 'all'] = 'all'
    include_children: bool = True
    limit: int = 50


def _memory_metadata(memory: MemoryModel) -> dict:
    return {
        'created_at': memory.created_at,
        'updated_at': memory.updated_at,
        'type': memory.type,
        'path': memory.path,
    }


async def reindex_memory_vectors_for_user(
    request: Request,
    user_id: str,
    memories: list[MemoryModel] | None = None,
    user=None,
) -> int:
    collection_name = f'user-memory-{user_id}'
    try:
        await ASYNC_VECTOR_DB_CLIENT.delete_collection(collection_name)
    except Exception as e:
        log.debug(e)

    memories = memories if memories is not None else await Memories.get_memories_by_user_id(user_id)
    memories = memories or []
    if not memories:
        return 0

    vectors = await asyncio.gather(
        *[
            request.app.state.EMBEDDING_FUNCTION(
                memory_vector_text(memory.content, memory.path),
                prefix=RAG_EMBEDDING_CONTENT_PREFIX,
                user=user,
            )
            for memory in memories
        ]
    )

    await ASYNC_VECTOR_DB_CLIENT.upsert(
        collection_name=collection_name,
        items=[
            {
                'id': memory.id,
                'text': memory_vector_text(memory.content, memory.path),
                'vector': vectors[idx],
                'metadata': _memory_metadata(memory),
            }
            for idx, memory in enumerate(memories)
        ],
    )
    return len(memories)


async def upsert_memory_vectors_or_reindex(request: Request, user, items: list[dict]) -> None:
    try:
        await ASYNC_VECTOR_DB_CLIENT.upsert(collection_name=f'user-memory-{user.id}', items=items)
    except Exception as e:
        message = str(e).lower()
        if 'dimension' not in message or 'embedding' not in message:
            raise

        log.warning('Memory vector dimension mismatch for user %s; reindexing memory vectors.', user.id)
        await reindex_memory_vectors_for_user(request, user.id, user=user)


@router.post('/add', response_model=MemoryModel | None)
async def add_memory(
    request: Request,
    form_data: AddMemoryForm,
    user=Depends(get_verified_user),
):
    """Persist a new memory and embed it into the user's vector collection.

    Does NOT use ``Depends(get_async_session)`` — database operations manage their
    own short-lived sessions so a connection is not held during the external
    embedding API call (``EMBEDDING_FUNCTION``), which can take 1-5+ seconds.
    """
    await check_memories_permission(user)

    content = clean_memory_content(form_data.content)
    path = clean_memory_path(form_data.path)
    memory = await Memories.insert_new_memory(
        user.id,
        content,
        memory_type=form_data.type,
        path=path,
        meta={'created_by': 'manual'},
    )

    vector = await request.app.state.EMBEDDING_FUNCTION(
        memory_vector_text(memory.content, memory.path), prefix=RAG_EMBEDDING_CONTENT_PREFIX, user=user
    )

    await upsert_memory_vectors_or_reindex(
        request,
        user,
        [
            {
                'id': memory.id,
                'text': memory_vector_text(memory.content, memory.path),
                'vector': vector,
                'metadata': _memory_metadata(memory),
            }
        ],
    )

    await publish_event(
        request,
        EVENTS.MEMORY_CREATED,
        actor=user,
        subject_id=memory.id,
        data={'content_preview': memory.content[:300], 'type': memory.type, 'path': memory.path},
    )
    return memory


@router.post('/update', response_model=list[dict])
async def update_memories(
    request: Request,
    form_data: UpdateMemoriesForm,
    user=Depends(get_verified_user),
):
    await check_memories_permission(user)

    operations = validate_memory_operations(form_data)
    metadata = getattr(request.state, 'metadata', {}) or {}
    source = form_data.source or 'tool'
    for operation in operations:
        if operation.get('action') in {'add', 'replace', 'move'}:
            operation['meta'] = {
                'created_by': source,
                'chat_id': metadata.get('chat_id'),
                'message_id': metadata.get('message_id'),
                'model': metadata.get('model'),
            }

    # Human-in-the-loop gate. The background review decides on its own what is
    # worth remembering; with approval required its operations are queued for a
    # person to confirm instead of applied. Placed before anything happens, so
    # a queued operation touches neither the database, the vector collection,
    # nor the event stream.
    #
    # Only `background_review` is gated. A `tool` write is the model acting
    # inside a conversation the user is watching, and a manual edit is the user
    # themselves — asking them to approve their own edit would be noise.
    if source == 'background_review' and background_review_requires_approval(metadata):
        proposals = await MemoryProposals.insert_proposals(user.id, operations)
        await publish_event(
            request,
            EVENTS.MEMORY_PROPOSED,
            actor=user,
            subject_id=None,
            data={'count': len(proposals)},
        )
        return [
            {
                'action': proposal.action,
                'status': 'pending_review',
                'id': proposal.id,
            }
            for proposal in proposals
        ]

    try:
        results = await Memories.apply_memory_operations(user.id, operations)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    upsert_items = []
    delete_ids = []
    response = []

    for result in results:
        memory = result.get('memory')
        if isinstance(memory, MemoryModel):
            result = {**result, 'memory': memory.model_dump()}
            if result.get('status') in {'created', 'updated'}:
                vector = await request.app.state.EMBEDDING_FUNCTION(
                    memory_vector_text(memory.content, memory.path),
                    prefix=RAG_EMBEDDING_CONTENT_PREFIX,
                    user=user,
                )
                upsert_items.append(
                    {
                        'id': memory.id,
                        'text': memory_vector_text(memory.content, memory.path),
                        'vector': vector,
                        'metadata': _memory_metadata(memory),
                    }
                )
        if result.get('status') == 'deleted' and result.get('id'):
            delete_ids.append(result['id'])
        response.append(result)

    if upsert_items:
        await upsert_memory_vectors_or_reindex(request, user, upsert_items)

    if delete_ids:
        await ASYNC_VECTOR_DB_CLIENT.delete(collection_name=f'user-memory-{user.id}', ids=delete_ids)

    for result in response:
        status_value = result.get('status')
        memory = result.get('memory') or {}
        memory_id = memory.get('id') or result.get('id')

        if status_value == 'created':
            event = EVENTS.MEMORY_CREATED
        elif status_value == 'updated':
            event = EVENTS.MEMORY_UPDATED
        elif status_value == 'deleted':
            event = EVENTS.MEMORY_DELETED
        else:
            continue

        await publish_event(
            request,
            event,
            actor=user,
            subject_id=memory_id,
            data={
                'content_preview': (memory.get('content') or '')[:300],
                'type': memory.get('type'),
                'path': memory.get('path'),
                'operation': result.get('action'),
            },
        )

    return response


############################
# QueryMemory
############################


class QueryMemoryForm(BaseModel):
    content: str
    k: int | None = 1


@router.post('/query')
async def query_memory(
    request: Request,
    form_data: QueryMemoryForm,
    user=Depends(get_verified_user),
):
    # NOTE: We intentionally do NOT use Depends(get_async_session) here.
    # Database operations (get_memories_by_user_id) manage their own short-lived sessions.
    # This prevents holding a connection during EMBEDDING_FUNCTION()
    # which makes external embedding API calls (1-5+ seconds).
    await check_memories_permission(user)

    memories = await Memories.get_memories_by_user_id(user.id)
    if not memories:
        raise HTTPException(status_code=404, detail='No memories found for user')

    vector = await request.app.state.EMBEDDING_FUNCTION(form_data.content, prefix=RAG_EMBEDDING_QUERY_PREFIX, user=user)

    results = await ASYNC_VECTOR_DB_CLIENT.search(
        collection_name=f'user-memory-{user.id}',
        vectors=[vector],
        limit=form_data.k,
    )

    # Filter results by relevance threshold to avoid returning unrelated
    # memories.  Vector similarity search always returns the top-K nearest
    # neighbours even when they are completely irrelevant; applying the
    # same RELEVANCE_THRESHOLD used by RAG ensures only genuinely matching
    # memories are surfaced (distances are normalised to 0→1, higher is
    # better).
    relevance_threshold = await Config.get('rag.relevance_threshold', 0.0)
    if results and relevance_threshold > 0.0 and results.distances and results.distances[0]:
        from open_webui.retrieval.vector.main import SearchResult

        filtered_ids = []
        filtered_docs = []
        filtered_metas = []
        filtered_dists = []

        for idx, score in enumerate(results.distances[0]):
            if score >= relevance_threshold:
                if results.ids and results.ids[0]:
                    filtered_ids.append(results.ids[0][idx])
                if results.documents and results.documents[0]:
                    filtered_docs.append(results.documents[0][idx])
                if results.metadatas and results.metadatas[0]:
                    filtered_metas.append(results.metadatas[0][idx])
                filtered_dists.append(score)

        results = SearchResult(
            ids=[filtered_ids] if filtered_ids else [[]],
            documents=[filtered_docs] if filtered_docs else [[]],
            metadatas=[filtered_metas] if filtered_metas else [[]],
            distances=[filtered_dists] if filtered_dists else [[]],
        )

    return results


@router.post('/search', response_model=list[MemoryModel])
async def search_memories(
    form_data: SearchMemoriesForm,
    user=Depends(get_verified_user),
):
    await check_memories_permission(user)

    memories = await Memories.get_memories_by_user_id(user.id)
    return search_memory_rows(
        memories,
        query=form_data.query,
        path=form_data.path,
        memory_id=form_data.memory_id,
        memory_type=form_data.type,
        limit=form_data.limit,
    )


@router.post('/paths')
async def list_memory_paths(
    form_data: ListMemoryPathsForm,
    user=Depends(get_verified_user),
):
    await check_memories_permission(user)

    memories = await Memories.get_memories_by_user_id(user.id)
    return list_memory_path_groups(
        memories,
        query=form_data.query or '',
        memory_type=form_data.type,
        limit=form_data.limit,
    )


@router.post('/path')
async def read_memory_path(
    form_data: ReadMemoryPathForm,
    user=Depends(get_verified_user),
):
    await check_memories_permission(user)

    memories = await Memories.get_memories_by_user_id(user.id)
    result = read_memory_path_rows(
        memories,
        path=form_data.path,
        memory_type=form_data.type,
        include_children=form_data.include_children,
        limit=form_data.limit,
    )
    return {
        **result,
        'memories': [memory.model_dump() for memory in result['memories']],
    }


############################
# ReindexMemoryVectorDB
############################
@router.post('/reindex')
async def reindex_memories_from_vector_db(
    request: Request,
    user=Depends(get_admin_user),
):
    memories = await Memories.get_memories()
    memories = memories or []
    memories_by_user_id = {}
    for memory in memories:
        memories_by_user_id.setdefault(memory.user_id, []).append(memory)

    users_result = await Users.get_users()
    users = users_result.get('users', []) if users_result else []
    total_memories = 0

    for memory_user in users:
        total_memories += await reindex_memory_vectors_for_user(
            request,
            memory_user.id,
            memories=memories_by_user_id.get(memory_user.id, []),
            user=memory_user,
        )

    await publish_event(
        request,
        EVENTS.MEMORY_RESET,
        actor=user,
        subject_id='all',
        subject_type='user',
        data={'count': total_memories, 'user_count': len(users), 'reindex': True},
    )
    return {'status': True, 'total_users': len(users), 'total_memories': total_memories}


############################
# MemoryProposals — human review of background-review writes
############################


@router.get('/proposals', response_model=list[MemoryProposalModel])
async def get_memory_proposals(user=Depends(get_verified_user)):
    """Memory operations the background review is waiting to have confirmed."""
    await check_memories_permission(user)
    return await MemoryProposals.get_proposals_by_user_id(user.id)


@router.post('/proposals/apply', response_model=list[dict])
async def apply_memory_proposals(
    request: Request,
    form_data: ApplyMemoryProposalsForm,
    user=Depends(get_verified_user),
):
    """Apply a reviewed batch and clear the queue.

    The submitted list is the desired outcome, not a set of approvals: an entry
    may have been reworded, and one the review never proposed may have been
    added. Everything listed is applied through the ordinary write path, so an
    entry a human typed is embedded and stored exactly like one the model
    suggested.

    The queue is cleared whether or not anything was accepted — that is what
    makes rejection simply "leave it out".
    """
    await check_memories_permission(user)

    operations = []
    for entry in form_data.memories:
        operation = {
            'action': entry.action,
            'id': entry.id,
            'content': entry.content,
            'type': entry.type,
            'path': entry.path,
        }
        operations.append({key: value for key, value in operation.items() if value is not None})

    results = []
    if operations:
        # Reuse the ordinary path rather than reimplementing apply + embed +
        # publish. It already handles validation, the vector upsert and the
        # events; duplicating it would be the obvious place for the two to
        # drift.
        results = await update_memories(
            request,
            UpdateMemoriesForm(operations=operations, source='review_approved'),
            user,
        )

    # Cleared last: if applying raised, the queue survives for another attempt
    # rather than being lost along with the failure.
    await MemoryProposals.delete_proposals_by_user_id(user.id)
    return results


@router.delete('/proposals', response_model=bool)
async def discard_memory_proposals(user=Depends(get_verified_user)):
    """Reject everything currently queued."""
    await check_memories_permission(user)
    await MemoryProposals.delete_proposals_by_user_id(user.id)
    return True


@router.post('/reset', response_model=bool)
async def reset_memory_from_vector_db(
    request: Request,
    user=Depends(get_verified_user),
):
    """Reset user's memory vector embeddings.

    CRITICAL: We intentionally do NOT use Depends(get_async_session) here.
    This endpoint generates embeddings for ALL user memories in parallel using
    asyncio.gather(). A user with 100 memories would trigger 100 embedding API
    calls simultaneously. With a session held, this could block a connection
    for MINUTES, completely exhausting the connection pool.
    """
    await check_memories_permission(user)

    count = await reindex_memory_vectors_for_user(request, user.id, user=user)

    await publish_event(
        request,
        EVENTS.MEMORY_RESET,
        actor=user,
        subject_id=user.id,
        subject_type='user',
        data={'count': count, 'reindex': True},
    )
    return True


############################
# DeleteMemoriesByUserId
############################


@router.delete('/delete/user', response_model=bool)
async def delete_memory_by_user_id(
    request: Request,
    user=Depends(get_verified_user),
    db: AsyncSession = Depends(get_async_session),
):
    await check_memories_permission(user)

    result = await Memories.delete_memories_by_user_id(user.id, db=db)

    if result:
        try:
            await ASYNC_VECTOR_DB_CLIENT.delete_collection(f'user-memory-{user.id}')
        except Exception as e:
            log.error(e)
        await publish_event(
            request,
            EVENTS.MEMORY_DELETED,
            actor=user,
            subject_id=user.id,
            subject_type='user',
        )
        return True

    return False


############################
# UpdateMemoryById
############################


@router.post('/{memory_id}/update', response_model=MemoryModel | None)
async def update_memory_by_id(
    memory_id: str,
    request: Request,
    form_data: MemoryUpdateModel,
    user=Depends(get_verified_user),
):
    # NOTE: We intentionally do NOT use Depends(get_async_session) here.
    # Database operations (update_memory_by_id_and_user_id) manage their own
    # short-lived sessions. This prevents holding a connection during
    # EMBEDDING_FUNCTION() which makes external API calls (1-5+ seconds).
    await check_memories_permission(user)

    content = clean_memory_content(form_data.content) if form_data.content is not None else None
    path = clean_memory_path(form_data.path)
    if content is None and form_data.type is None and form_data.path is None:
        raise HTTPException(status_code=400, detail='No memory update provided')
    memory = await Memories.update_memory_by_id_and_user_id(
        memory_id,
        user.id,
        content,
        memory_type=form_data.type,
        path=path,
        update_path=form_data.path is not None,
        meta={'created_by': 'manual'},
    )
    if memory is None:
        raise HTTPException(status_code=404, detail=ERROR_MESSAGES.NOT_FOUND)

    if form_data.content is not None or form_data.path is not None:
        vector = await request.app.state.EMBEDDING_FUNCTION(
            memory_vector_text(memory.content, memory.path), prefix=RAG_EMBEDDING_CONTENT_PREFIX, user=user
        )

        await upsert_memory_vectors_or_reindex(
            request,
            user,
            [
                {
                    'id': memory.id,
                    'text': memory_vector_text(memory.content, memory.path),
                    'vector': vector,
                    'metadata': _memory_metadata(memory),
                }
            ],
        )

    await publish_event(
        request,
        EVENTS.MEMORY_UPDATED,
        actor=user,
        subject_id=memory.id,
        data={'content_preview': memory.content[:300], 'type': memory.type, 'path': memory.path},
    )
    return memory


############################
# DeleteMemoryById
############################


@router.delete('/{memory_id}', response_model=bool)
async def delete_memory_by_id(
    memory_id: str,
    request: Request,
    user=Depends(get_verified_user),
    db: AsyncSession = Depends(get_async_session),
):
    await check_memories_permission(user)

    result = await Memories.delete_memory_by_id_and_user_id(memory_id, user.id, db=db)

    if result:
        await ASYNC_VECTOR_DB_CLIENT.delete(collection_name=f'user-memory-{user.id}', ids=[memory_id])
        await publish_event(
            request,
            EVENTS.MEMORY_DELETED,
            actor=user,
            subject_id=memory_id,
        )
        return True

    return False
