"""Standalone ingestion script. Run once to populate Qdrant.

Usage:
    python scripts/ingest.py

Reads all source documents from data/docs/, parses with Docling,
chunks with HierarchicalChunker, embeds, and stores in Qdrant.
Qdrant must be running before executing this script:
    docker compose up -d qdrant
"""

import asyncio
import pathlib

from src.adapters.embedding_adapter import EmbeddingAdapter
from src.adapters.qdrant_adapter import QdrantAdapter
from src.adapters.sparse_adapter import SparseAdapter
from src.services.ingestion_service import IngestionService
from src.utils.exceptions import IngestionError
from src.utils.logger import log_request_event

# data/ is a sibling of this script (backend/ingestion/data/docs/).
DOCS_DIR = pathlib.Path(__file__).parent / "data" / "docs"

# Document manifest: filename → collection.
# This is the only file that maps specific filenames to collections.
# Derived from DATA_DICTIONARY.md §1.
DOC_COLLECTION_MAP: dict[str, str] = {
    "staff_handbook.pdf":         "general",
    "leave_policy.pdf":           "general",
    "code_of_conduct.pdf":        "general",
    "general_faqs.pdf":           "general",
    "treatment_protocols.pdf":    "clinical",
    "drug_formulary.pdf":         "clinical",
    "diagnostic_reference.pdf":   "clinical",
    "icu_nursing_procedures.pdf":  "nursing",
    "infection_control.pdf":       "nursing",
    "billing_codes.pdf":           "billing",
    "claim_submission_guide.md":   "billing",
    "equipment_manual.pdf":        "equipment",
}


async def main() -> None:
    """Wire adapters, iterate over documents, ingest each one."""

    # Construct adapters once — model loading happens here (expensive).
    # Each adapter downloads its model on first run and caches it locally.
    print("Loading models...")
    qdrant = QdrantAdapter()
    embedder = EmbeddingAdapter()
    sparse = SparseAdapter()

    service = IngestionService(qdrant=qdrant, embedder=embedder, sparse=sparse)

    total_chunks = 0
    failed_docs: list[str] = []

    for filename, collection in DOC_COLLECTION_MAP.items():
        file_path = DOCS_DIR / filename

        if not file_path.exists():
            log_request_event(
                request_id="ingest",
                role="system",
                event="file_not_found",
                source_document=filename,
            )
            print(f"  SKIP  {filename} — file not found at {file_path}")
            failed_docs.append(filename)
            continue

        print(f"  ...   {filename} [{collection}]")

        try:
            chunk_count = await service.ingest_document(
                file_path=str(file_path),
                collection=collection,
                source_document=filename,
            )
            total_chunks += chunk_count
            print(f"  OK    {filename} → {chunk_count} chunks")

        except IngestionError as e:
            log_request_event(
                request_id="ingest",
                role="system",
                event="document_failed",
                source_document=filename,
                reason=str(e),
            )
            print(f"  FAIL  {filename} — {e}")
            failed_docs.append(filename)

    # Summary
    print("\n--- Ingestion complete ---")
    print(f"Documents: {len(DOC_COLLECTION_MAP) - len(failed_docs)} succeeded, "
          f"{len(failed_docs)} failed")
    print(f"Total chunks stored: {total_chunks}")

    if failed_docs:
        print(f"Failed: {', '.join(failed_docs)}")


if __name__ == "__main__":
    asyncio.run(main())
