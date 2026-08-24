"""add memory proposal table

Holds background-review memory operations awaiting human approval when
``memory_approval_mode`` is ``ask``. Deliberately a
separate table from ``memory`` so an unapproved suggestion can never be
embedded into the user's vector collection or picked up by a query that does
not know about approval state.

Revision ID: b7e4c9a15d02
Revises: d4c1a8e37b62
Create Date: 2026-08-12 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b7e4c9a15d02'
down_revision: Union[str, None] = 'd4c1a8e37b62'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    if 'memory_proposal' not in tables:
        op.create_table(
            'memory_proposal',
            sa.Column('id', sa.String(), nullable=False),
            sa.Column('user_id', sa.String(), nullable=True),
            sa.Column('action', sa.String(), nullable=True),
            sa.Column('memory_id', sa.String(), nullable=True),
            sa.Column('type', sa.String(), server_default='context', nullable=True),
            sa.Column('path', sa.Text(), nullable=True),
            sa.Column('content', sa.Text(), nullable=True),
            sa.Column('meta', sa.JSON(), nullable=True),
            sa.Column('created_at', sa.BigInteger(), nullable=True),
            sa.PrimaryKeyConstraint('id'),
            sa.UniqueConstraint('id'),
        )

    indexes = (
        {index['name'] for index in inspector.get_indexes('memory_proposal')} if 'memory_proposal' in tables else set()
    )
    if 'ix_memory_proposal_user_id' not in indexes:
        op.create_index('ix_memory_proposal_user_id', 'memory_proposal', ['user_id'])
    if 'ix_memory_proposal_user_id_created_at' not in indexes:
        op.create_index(
            'ix_memory_proposal_user_id_created_at',
            'memory_proposal',
            ['user_id', 'created_at'],
        )


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = set(inspector.get_table_names())

    if 'memory_proposal' in tables:
        indexes = {index['name'] for index in inspector.get_indexes('memory_proposal')}
        if 'ix_memory_proposal_user_id_created_at' in indexes:
            op.drop_index('ix_memory_proposal_user_id_created_at', table_name='memory_proposal')
        if 'ix_memory_proposal_user_id' in indexes:
            op.drop_index('ix_memory_proposal_user_id', table_name='memory_proposal')
        op.drop_table('memory_proposal')
