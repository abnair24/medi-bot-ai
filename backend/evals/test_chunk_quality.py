"""Track 6 — chunk quality: citation completeness audit.

Scans the LIVE Qdrant collection and asserts every stored chunk carries a usable
citation (non-empty source_document AND section_title). This is the only guard over
data ALREADY at rest — the ChunkMetadata/ChunkResult min_length constraints guard
future writes and live reads, but cannot retroactively validate existing payloads.

Fail-closed: citation completeness must be exactly 1.0. A single citation-less chunk
makes the guardrail "every answer must include source citations" unsatisfiable.
"""

from evals.eval_config import CITATION_COMPLETENESS

# Payload fields that together form a chunk's citation.
_CITATION_FIELDS = ("source_document", "section_title")


def _citation_present(payload: dict) -> bool:
    """True only if every citation field exists and is a non-empty, non-blank string."""
    return all(
        isinstance(payload.get(field), str) and payload[field].strip()
        for field in _CITATION_FIELDS
    )


def test_collection_is_not_empty(all_payloads: list[dict]) -> None:
    """Guard against a false pass: an empty collection would trivially satisfy 1.0."""
    assert len(all_payloads) > 0, (
        "No chunks found in Qdrant — cannot audit citations. Is ingestion done?"
    )


def test_all_chunks_have_citation(all_payloads: list[dict]) -> None:
    """100% of stored chunks must have a non-empty source_document and section_title."""
    total = len(all_payloads)
    missing = [p for p in all_payloads if not _citation_present(p)]
    completeness = (total - len(missing)) / total

    # Metadata-only diagnostics — no chunk text (PHI-safe logging).
    offenders = [
        {
            "source_document": p.get("source_document") or "<empty>",
            "section_title": p.get("section_title") or "<empty>",
            "collection": p.get("collection") or "<empty>",
        }
        for p in missing[:10]
    ]

    assert completeness >= CITATION_COMPLETENESS, (
        f"Citation completeness {completeness:.4f} < {CITATION_COMPLETENESS}. "
        f"{len(missing)}/{total} chunks missing a citation. "
        f"First offenders (metadata only): {offenders}"
    )
