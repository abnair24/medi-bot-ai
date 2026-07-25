import asyncio

from docling.chunking import HierarchicalChunker
from docling.document_converter import DocumentConverter
from docling_core.types.doc import DocItemLabel

from src.adapters.embedding_adapter import EmbeddingAdapter
from src.adapters.qdrant_adapter import QdrantAdapter
from src.adapters.sparse_adapter import SparseAdapter
from src.config.access import get_roles_for_collection
from src.config.settings import settings
from src.models.chunk import ChunkMetadata
from src.models.schemas import ChunkType, Collection
from src.utils.exceptions import IngestionError
from src.utils.logger import log_request_event


class IngestionService:
    """Orchestrates the full ingestion pipeline for one document at a time.

    Pipeline: Docling parse → HierarchicalChunker → ChunkMetadata validation
              → batch embed (dense + sparse) → Qdrant insert.

    Receives all adapters via DI — never constructs them internally.
    """

    def __init__(
        self,
        qdrant: QdrantAdapter,
        embedder: EmbeddingAdapter,
        sparse: SparseAdapter,
    ) -> None:
        self._qdrant = qdrant
        self._embedder = embedder
        self._sparse = sparse
        self._converter = DocumentConverter()
        self._chunker = HierarchicalChunker(
            max_tokens=settings.chunk_token_limit,
        )

    async def ingest_document(
        self,
        file_path: str,
        collection: Collection,
        source_document: str,
    ) -> int:
        """Parse, chunk, validate, embed, and store one document.

        Args:
            file_path: absolute or relative path to the PDF or MD file.
            collection: the Qdrant collection this document belongs to.
            source_document: the filename used in citations (e.g. 'drug_formulary.pdf').

        Returns:
            Number of chunks successfully stored.

        Raises:
            IngestionError: if parsing, validation, embedding, or storage fails.
        """
        try:
            chunks = await self._parse_and_chunk(file_path)
        except Exception as e:
            raise IngestionError(f"Parsing failed for {source_document}: {e}") from e

        access_roles = get_roles_for_collection(collection)
        metadata_list, skipped = _build_metadata(chunks, collection, access_roles, source_document)

        log_request_event(
            request_id="ingest",
            role="system",
            event="chunks_built",
            source_document=source_document,
            chunk_count=len(metadata_list),
            skipped_count=skipped,
        )

        try:
            texts = [m.text for m in metadata_list]
            dense_vectors = await asyncio.to_thread(self._embedder.embed_batch, texts)
            sparse_vectors = await asyncio.to_thread(self._sparse.embed_batch, texts)
        except Exception as e:
            raise IngestionError(f"Embedding failed for {source_document}: {e}") from e

        try:
            self._qdrant.insert_batch(metadata_list, dense_vectors, sparse_vectors)
        except Exception as e:
            raise IngestionError(f"Qdrant insert failed for {source_document}: {e}") from e

        log_request_event(
            request_id="ingest",
            role="system",
            event="document_ingested",
            source_document=source_document,
            chunk_count=len(metadata_list),
        )

        return len(metadata_list)

    async def _parse_and_chunk(self, file_path: str) -> list:
        """Run Docling conversion and HierarchicalChunker in a thread pool."""
        result = await asyncio.to_thread(self._converter.convert, file_path)
        raw_chunks = await asyncio.to_thread(
            lambda: list(self._chunker.chunk(result.document))
        )
        return raw_chunks


def _build_metadata(
    raw_chunks: list,
    collection: Collection,
    access_roles: list[str],
    source_document: str,
) -> tuple[list[ChunkMetadata], int]:
    """Validate each Docling chunk into a ChunkMetadata model.

    Fail-tolerant: skips invalid or empty chunks, logs a warning for each,
    and continues processing. Raises IngestionError only if zero valid chunks
    are produced (indicates a structural document parsing failure).

    Returns:
        Tuple of (valid metadata list, count of skipped chunks).
    """
    metadata_list: list[ChunkMetadata] = []
    skipped = 0

    for chunk in raw_chunks:
        text = chunk.text.strip()
        if not text:
            skipped += 1
            continue
        try:
            metadata = ChunkMetadata(
                text=text,
                source_document=source_document,
                collection=collection,
                access_roles=access_roles,
                section_title=_extract_section_title(chunk),
                chunk_type=_extract_chunk_type(chunk),
            )
            metadata_list.append(metadata)
        except Exception as e:
            skipped += 1
            # Log section_title for investigation — never log chunk text (may contain PHI context).
            log_request_event(
                request_id="ingest",
                role="system",
                event="chunk_skipped",
                source_document=source_document,
                section_title=_extract_section_title(chunk),
                reason=str(e),
            )

    if not metadata_list:
        raise IngestionError(
            f"Zero valid chunks produced for {source_document} "
            f"({skipped} skipped) — check document structure."
        )

    return metadata_list, skipped


def _extract_section_title(chunk) -> str:
    """Return the innermost parent heading for a chunk, or a fallback."""
    headings = getattr(chunk.meta, "headings", [])
    if headings:
        return headings[-1]  # innermost = most specific context
    return "Unknown Section"


def _extract_chunk_type(chunk) -> ChunkType:
    """Map Docling's DocItemLabel to our ChunkType literal."""
    items = getattr(chunk.meta, "doc_items", [])
    if not items:
        return "text"
    label = items[0].label
    if label == DocItemLabel.TABLE:
        return "table"
    if label in (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE):
        return "heading"
    if label == DocItemLabel.CODE:
        return "code"
    return "text"
