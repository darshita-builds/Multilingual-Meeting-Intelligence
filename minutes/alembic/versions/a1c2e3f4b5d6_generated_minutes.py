"""generated minutes

Adds the `generated_minutes` table: domain-formatted minutes documents
rendered from a job's already-extracted transcript/agenda/decisions/actions.
Purely additive -- no existing table is touched.

Revision ID: a1c2e3f4b5d6
Revises: bbd7a5bb606b
Create Date: 2026-09-19 18:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "a1c2e3f4b5d6"
down_revision: str | Sequence[str] | None = "bbd7a5bb606b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "generated_minutes",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("job_id", sa.String(length=36), nullable=False),
        sa.Column(
            "domain",
            sa.Enum("TECHNOLOGY", "EDUCATION", "HEALTHCARE", "BUSINESS", name="meetingdomain"),
            nullable=False,
        ),
        sa.Column("sections", sa.JSON(), nullable=False),
        sa.Column("original_sections", sa.JSON(), nullable=False),
        sa.Column(
            "status",
            sa.Enum("DRAFT", "APPROVED", name="minutesstatus"),
            nullable=False,
        ),
        sa.Column("source_tool", sa.String(length=128), nullable=False),
        sa.Column("created_by_id", sa.String(length=36), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("edited_by_id", sa.String(length=36), nullable=True),
        sa.Column("edited_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("approved_by_id", sa.String(length=36), nullable=True),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("approval_note", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["created_by_id"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["edited_by_id"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["approved_by_id"], ["users.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    with op.batch_alter_table("generated_minutes", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_generated_minutes_job_id"), ["job_id"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_generated_minutes_status"), ["status"], unique=False
        )


def downgrade() -> None:
    with op.batch_alter_table("generated_minutes", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_generated_minutes_status"))
        batch_op.drop_index(batch_op.f("ix_generated_minutes_job_id"))
    op.drop_table("generated_minutes")

    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP TYPE IF EXISTS meetingdomain")
        op.execute("DROP TYPE IF EXISTS minutesstatus")
