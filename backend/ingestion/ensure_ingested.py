"""Idempotent ingestion — entrypoint for the one-shot `ingest` docker-compose service.

Waits for Qdrant, then ingests the corpus ONLY if the collection is empty (a fresh
qdrant_data volume). On a persisted volume it exits quickly without re-ingesting.

The backend service `depends_on` this completing successfully, so the vector index is
populated BEFORE the API starts serving — no empty-retrieval race on first
`docker compose up`. The source docs (`data/docs/`) and SQLite DB are committed, so a
fresh clone has everything ingestion needs.

The wait + ensure-ingested helpers are also reused by `scripts/run_evals.py` (the
headless full-suite eval runner) so this logic lives in exactly one place.
"""

import asyncio
import time

from src.adapters.qdrant_adapter import QdrantAdapter
from src.config.settings import settings
from src.utils.logger import logger

_QDRANT_WAIT_TIMEOUT_S = 180
_QDRANT_POLL_INTERVAL_S = 3


def wait_for_qdrant() -> QdrantAdapter:
    """Poll Qdrant until it answers or the timeout elapses.

    Constructing QdrantAdapter also ensures the collection exists (non-destructive:
    it creates only when absent, never wipes an existing one).
    """
    deadline = time.monotonic() + _QDRANT_WAIT_TIMEOUT_S
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            adapter = QdrantAdapter()
            adapter._client.get_collections()  # cheap round-trip proves it's answering
            logger.info("ingest_qdrant_ready")
            return adapter
        except Exception as exc:  # noqa: BLE001 — retry on any connection error
            last_err = exc
            time.sleep(_QDRANT_POLL_INTERVAL_S)
    raise RuntimeError(
        f"Qdrant not reachable after {_QDRANT_WAIT_TIMEOUT_S}s: {last_err}"
    )


def collection_count(adapter: QdrantAdapter) -> int:
    """Return the number of points in the configured collection (0 if empty/missing)."""
    try:
        return adapter._client.count(
            collection_name=settings.collection_name, exact=True
        ).count
    except Exception:  # noqa: BLE001 — treat any failure as "needs ingestion"
        return 0


def ensure_ingested(adapter: QdrantAdapter | None = None) -> None:
    """Ingest the corpus only when the collection is empty. Idempotent.

    Args:
        adapter: a ready QdrantAdapter; if None, waits for Qdrant first.
    """
    adapter = adapter or wait_for_qdrant()
    count = collection_count(adapter)
    if count > 0:
        logger.info("ingest_skip", extra={"point_count": count})
        return

    logger.info("ingest_running", extra={"reason": "empty_collection"})
    # Import here (not at module top) so the ingest model-loading cost is only paid
    # when ingestion is actually needed.
    from ingestion import ingest

    asyncio.run(ingest.main())
    logger.info("ingest_done")


def main() -> None:
    ensure_ingested()


if __name__ == "__main__":
    main()
