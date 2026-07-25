"""Headless full-suite eval runner (manual / CI use).

NOT wired into docker-compose — evals run from the admin UI (POST /evals/run) during
normal operation. This script is for running every track headless, e.g. in CI or a
one-off local sweep. It waits for Qdrant, ensures the corpus is ingested (reusing
scripts.ensure_ingested), then runs the tracks synchronously, writing per-track JUnit
XML + logs into evals/results/ (read by the admin Evaluation dashboard).

Track selection: set EVAL_TRACKS to a comma-separated subset (e.g. "rbac,safety,sql")
to run only the fast deterministic tracks and skip the slow LLM-judge tracks
(faithfulness, calibration). Default: every track.

Usage:  PYTHONPATH=. python scripts/run_evals.py
"""

import os

from ingestion.ensure_ingested import ensure_ingested, wait_for_qdrant
from src.services import eval_service
from src.utils.logger import logger


def _selected_tracks() -> list[str] | None:
    """Parse EVAL_TRACKS env into a track list, or None for 'all tracks'."""
    raw = os.getenv("EVAL_TRACKS", "").strip()
    if not raw:
        return None
    return [t.strip() for t in raw.split(",") if t.strip()]


def main() -> None:
    adapter = wait_for_qdrant()
    ensure_ingested(adapter)

    tracks = _selected_tracks()
    logger.info("evals_start", extra={"tracks": tracks or "all"})
    results = eval_service.run_all_blocking(tracks)

    # Concise stdout summary so `docker compose logs`/CI shows pass/fail at a glance.
    for row in results["tracks"]:
        if row["status"] == "not_run":
            continue
        print(f"[{row['status'].upper():>6}] {row['label']}: {row['summary']}")
    logger.info("evals_done")


if __name__ == "__main__":
    main()
