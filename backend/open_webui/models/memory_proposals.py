"""Memory writes proposed by the background review, awaiting human approval.

The background review reads a conversation and decides on its own what is worth
remembering. With ``memory_approval_mode`` set to ``ask`` its
operations land here instead of in ``memory``, and are applied only once a human
has seen them.

They are a separate table rather than a flag on ``memory`` on purpose. A memory
row is not just a row: it is embedded into the user's vector collection and
retrieved into later conversations, and several code paths read the table
without knowing about any approval state. Keeping proposals out of that table
means an unapproved suggestion cannot be embedded, retrieved, exported or
counted by accident — the isolation is structural rather than something every
future query has to remember to filter on.

One row per proposed *operation*, so a review can accept some and reject others
instead of being all-or-nothing.
"""

from __future__ import annotations

import time
import uuid
from typing import Literal

from open_webui.internal.db import Base, get_async_db_context
from pydantic import BaseModel, ConfigDict
from sqlalchemy import JSON, BigInteger, Column, Index, String, Text, delete, select
from sqlalchemy.ext.asyncio import AsyncSession


class MemoryProposal(Base):
    """A single proposed memory operation, pending human review."""

    __tablename__ = 'memory_proposal'
    __table_args__ = (Index('ix_memory_proposal_user_id_created_at', 'user_id', 'created_at'),)

    id = Column(String, primary_key=True, unique=True)
    user_id = Column(String, index=True)
    # Mirrors MemoryOperationModel.action: add / replace / remove / move.
    action = Column(String)
    # Target memory for replace / remove / move; NULL for add.
    memory_id = Column(String, nullable=True)
    type = Column(String, default='context', server_default='context')
    path = Column(Text, nullable=True)
    content = Column(Text, nullable=True)
    # Provenance the review carried: chat_id, message_id, model, created_by.
    meta = Column(JSON, nullable=True)
    created_at = Column(BigInteger)


class MemoryProposalModel(BaseModel):
    id: str
    user_id: str
    action: str
    memory_id: str | None = None
    type: Literal['user', 'context'] = 'context'
    path: str | None = None
    content: str | None = None
    meta: dict | None = None
    created_at: int
    model_config = ConfigDict(from_attributes=True)


class MemoryProposalsTable:
    async def insert_proposals(
        self,
        user_id: str,
        operations: list[dict],
        db: AsyncSession | None = None,
    ) -> list[MemoryProposalModel]:
        """Queue a batch of proposed operations.

        Returns the stored rows so the caller can tell the model how many are
        waiting — a review that is told nothing assumes its writes took effect.
        """
        async with get_async_db_context(db) as db:
            now = int(time.time())
            records = []
            for operation in operations:
                record = MemoryProposal(
                    id=str(uuid.uuid4()),
                    user_id=user_id,
                    action=operation.get('action'),
                    memory_id=operation.get('id'),
                    type='user' if operation.get('type') == 'user' else 'context',
                    path=operation.get('path'),
                    content=operation.get('content'),
                    meta=operation.get('meta'),
                    created_at=now,
                )
                db.add(record)
                records.append(record)
            await db.commit()
            return [MemoryProposalModel.model_validate(record) for record in records]

    async def get_proposals_by_user_id(self, user_id: str, db: AsyncSession | None = None) -> list[MemoryProposalModel]:
        async with get_async_db_context(db) as db:
            result = await db.execute(
                select(MemoryProposal).filter_by(user_id=user_id).order_by(MemoryProposal.created_at.asc())
            )
            return [MemoryProposalModel.model_validate(row) for row in result.scalars().all()]

    async def count_proposals_by_user_id(self, user_id: str, db: AsyncSession | None = None) -> int:
        return len(await self.get_proposals_by_user_id(user_id, db=db))

    async def delete_proposals_by_ids(self, user_id: str, ids: list[str], db: AsyncSession | None = None) -> int:
        if not ids:
            return 0
        async with get_async_db_context(db) as db:
            result = await db.execute(
                delete(MemoryProposal).where(MemoryProposal.user_id == user_id).where(MemoryProposal.id.in_(ids))
            )
            await db.commit()
            return result.rowcount or 0

    async def delete_proposals_by_user_id(self, user_id: str, db: AsyncSession | None = None) -> int:
        async with get_async_db_context(db) as db:
            result = await db.execute(delete(MemoryProposal).where(MemoryProposal.user_id == user_id))
            await db.commit()
            return result.rowcount or 0


MemoryProposals = MemoryProposalsTable()
