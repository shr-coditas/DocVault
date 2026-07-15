import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, UUIDPrimaryKeyMixin

role_permissions = sa.Table(
    "role_permissions",
    Base.metadata,
    sa.Column("role_id", sa.ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True),
    sa.Column(
        "permission_id", sa.ForeignKey("permissions.id", ondelete="CASCADE"), primary_key=True
    ),
)


class Role(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "roles"

    name: Mapped[str] = mapped_column(sa.String(50), unique=True)
    description: Mapped[str] = mapped_column(sa.String(255), default="", server_default="")

    permissions: Mapped[list["Permission"]] = relationship(
        secondary=role_permissions, lazy="selectin"
    )


class Permission(UUIDPrimaryKeyMixin, Base):
    __tablename__ = "permissions"

    code: Mapped[str] = mapped_column(sa.String(50), unique=True)
    description: Mapped[str] = mapped_column(sa.String(255), default="", server_default="")
