"""add_user_id_to_source_documents

Revision ID: a1b2c3d4e5f6
Revises: 082b60696c17
Create Date: 2026-06-29 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '082b60696c17'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add user_id FK to source_documents and drop global unique constraint on filename."""
    # 1. Drop the global unique constraint on filename
    op.drop_constraint('source_documents_filename_key', 'source_documents', type_='unique')

    # 2. Add user_id foreign key column (nullable — public filings have user_id = NULL)
    op.add_column(
        'source_documents',
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True)
    )
    op.create_foreign_key(
        'fk_source_documents_user_id',
        'source_documents', 'users',
        ['user_id'], ['id'],
        ondelete='CASCADE'
    )
    op.create_index('ix_source_documents_user_id', 'source_documents', ['user_id'])


def downgrade() -> None:
    """Reverse migration: restore unique constraint, drop user_id column."""
    op.drop_index('ix_source_documents_user_id', table_name='source_documents')
    op.drop_constraint('fk_source_documents_user_id', 'source_documents', type_='foreignkey')
    op.drop_column('source_documents', 'user_id')
    op.create_unique_constraint('source_documents_filename_key', 'source_documents', ['filename'])
