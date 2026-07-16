import uuid

import sqlalchemy as sa
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.folder import Folder

# (id, parent_id, name, depth, path) rows from the recursive tree CTE
TreeRow = tuple[uuid.UUID, uuid.UUID | None, str, int, str]


class FolderRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    def add(self, folder: Folder) -> None:
        self.session.add(folder)

    async def get(self, folder_id: uuid.UUID) -> Folder | None:
        return await self.session.get(Folder, folder_id)

    async def delete(self, folder: Folder) -> None:
        await self.session.delete(folder)

    async def children(self, workspace_id: uuid.UUID, parent_id: uuid.UUID | None) -> list[Folder]:
        stmt = (
            select(Folder)
            .where(Folder.workspace_id == workspace_id, Folder.parent_id == parent_id)
            .order_by(Folder.name)
        )
        return list((await self.session.execute(stmt)).scalars())

    async def tree_rows(self, workspace_id: uuid.UUID) -> list[TreeRow]:
        """Whole workspace folder tree in one query via a recursive CTE."""
        base = (
            select(
                Folder.id,
                Folder.parent_id,
                Folder.name,
                sa.literal(0).label("depth"),
                Folder.name.concat(sa.literal("")).label("path"),
            )
            .where(Folder.workspace_id == workspace_id, Folder.parent_id.is_(None))
            .cte("folder_tree", recursive=True)
        )
        child = sa.orm.aliased(Folder)
        recursive = select(
            child.id,
            child.parent_id,
            child.name,
            (base.c.depth + 1).label("depth"),
            base.c.path.concat("/").concat(child.name).label("path"),
        ).join(base, child.parent_id == base.c.id)
        tree = base.union_all(recursive)

        rows = (await self.session.execute(select(tree).order_by(tree.c.path))).all()
        return [(r.id, r.parent_id, r.name, r.depth, r.path) for r in rows]

    async def sibling_named(
        self,
        workspace_id: uuid.UUID,
        parent_id: uuid.UUID | None,
        name: str,
        exclude: uuid.UUID | None = None,
    ) -> uuid.UUID | None:
        stmt = select(Folder.id).where(
            Folder.workspace_id == workspace_id,
            Folder.parent_id == parent_id,
            Folder.name == name,
        )
        if exclude is not None:
            stmt = stmt.where(Folder.id != exclude)
        return (await self.session.execute(stmt.limit(1))).scalar_one_or_none()

    async def subtree_ids(self, folder_id: uuid.UUID) -> list[uuid.UUID]:
        """The folder itself plus every descendant (for cascade cleanups)."""
        base = select(Folder.id).where(Folder.parent_id == folder_id).cte("subtree", recursive=True)
        child = sa.orm.aliased(Folder)
        descendants = base.union_all(select(child.id).join(base, child.parent_id == base.c.id))
        rows = (await self.session.execute(select(descendants.c.id))).scalars()
        return [folder_id, *rows]

    async def is_descendant(self, ancestor_id: uuid.UUID, candidate_id: uuid.UUID) -> bool:
        base = (
            select(Folder.id)
            .where(Folder.parent_id == ancestor_id)
            .cte("descendants", recursive=True)
        )
        child = sa.orm.aliased(Folder)
        descendants = base.union_all(select(child.id).join(base, child.parent_id == base.c.id))
        stmt = select(descendants.c.id).where(descendants.c.id == candidate_id).limit(1)
        return (await self.session.execute(stmt)).scalar_one_or_none() is not None
