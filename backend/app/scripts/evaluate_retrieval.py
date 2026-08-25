"""Run reproducible local retrieval evaluation against an indexed workspace.

The golden JSON is a list of objects with ``query`` and
``relevant_document_ids``. The selected user is intentional: evaluation must
exercise the same permission predicates as the API.
"""

import argparse
import asyncio
import json
import math
import statistics
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from app.db.session import async_session_factory, dispose_engine
from app.models.document_chunk import DocumentChunk
from app.repository.user_repository import UserRepository
from app.services.ai_types import SearchMode
from app.services.embedding_service import get_default_embedder
from app.services.reranking_service import get_default_reranker
from app.services.search_service import SearchService


@dataclass(frozen=True, slots=True)
class _Query:
    query: str
    relevant_document_ids: frozenset[uuid.UUID]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DocVault retrieval locally.")
    parser.add_argument("golden", type=Path)
    parser.add_argument("--workspace-id", type=uuid.UUID, required=True)
    parser.add_argument("--user-id", type=uuid.UUID, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _load(path: Path) -> list[_Query]:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("golden file must contain a JSON list")
    rows = []
    for item in payload:
        if not isinstance(item, dict) or not isinstance(item.get("query"), str):
            raise ValueError("every golden row needs a string query")
        relevant = item.get("relevant_document_ids", [])
        if not isinstance(relevant, list):
            raise ValueError("relevant_document_ids must be a list")
        rows.append(_Query(item["query"], frozenset(uuid.UUID(value) for value in relevant)))
    return rows


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]


async def _evaluate(args: argparse.Namespace) -> dict[str, object]:
    golden = _load(args.golden)
    embedder = get_default_embedder()
    configurations = (
        ("semantic", SearchMode.SEMANTIC, None),
        ("lexical", SearchMode.LEXICAL, None),
        ("hybrid", SearchMode.HYBRID, None),
        ("hybrid_reranker", SearchMode.HYBRID, get_default_reranker()),
    )
    report: dict[str, object] = {}
    async with async_session_factory() as session:
        actor = await UserRepository(session).get(args.user_id)
        if actor is None:
            raise ValueError("evaluation user does not exist")
        chunk_count = (
            await session.execute(
                select(func.count())
                .select_from(DocumentChunk)
                .where(DocumentChunk.workspace_id == args.workspace_id)
            )
        ).scalar_one()
        index_size = (
            await session.execute(select(func.pg_total_relation_size("document_chunks")))
        ).scalar_one()

        for name, mode, reranker in configurations:
            service = SearchService(session, embedder, reranker=reranker)
            reciprocal_ranks: list[float] = []
            ndcgs: list[float] = []
            latencies: list[float] = []
            duplicates = 0
            returned = 0
            hits_at_five = 0
            for case in golden:
                started = time.perf_counter()
                result = await service.search(
                    actor,
                    args.workspace_id,
                    case.query,
                    mode=mode,
                    limit=args.limit,
                    semantic_min_score=None,
                )
                latencies.append((time.perf_counter() - started) * 1000)
                ranks = [
                    rank
                    for rank, hit in enumerate(result.hits, start=1)
                    if hit.document_id in case.relevant_document_ids
                ]
                first = min(ranks) if ranks else None
                reciprocal_ranks.append(1 / first if first is not None else 0.0)
                hits_at_five += int(first is not None and first <= 5)
                dcg = sum(1 / math.log2(rank + 1) for rank in ranks if rank <= 10)
                ideal = sum(
                    1 / math.log2(rank + 1)
                    for rank in range(1, min(10, len(case.relevant_document_ids)) + 1)
                )
                ndcgs.append(dcg / ideal if ideal else 0.0)
                chunk_ids = [hit.chunk_id for hit in result.hits]
                duplicates += len(chunk_ids) - len(set(chunk_ids))
                returned += len(chunk_ids)

            total = len(golden) or 1
            report[name] = {
                "queries": len(golden),
                "hit_rate_at_5": hits_at_five / total,
                "mrr": statistics.fmean(reciprocal_ranks) if reciprocal_ranks else 0.0,
                "ndcg_at_10": statistics.fmean(ndcgs) if ndcgs else 0.0,
                "duplicate_result_rate": duplicates / returned if returned else 0.0,
                "latency_ms": {
                    "median": statistics.median(latencies) if latencies else 0.0,
                    "p95": _percentile(latencies, 0.95),
                },
            }
    return {
        "workspace_id": str(args.workspace_id),
        "user_id": str(args.user_id),
        "chunk_count": chunk_count,
        "document_chunks_relation_bytes": index_size,
        "runs": report,
    }


async def _main(args: argparse.Namespace) -> int:
    try:
        report = await _evaluate(args)
        rendered = json.dumps(report, indent=2, sort_keys=True)
        if args.output is None:
            print(rendered)
        else:
            args.output.write_text(rendered + "\n", encoding="utf-8")
        return 0
    finally:
        await dispose_engine()


def main() -> None:
    raise SystemExit(asyncio.run(_main(_arguments())))


if __name__ == "__main__":
    main()
