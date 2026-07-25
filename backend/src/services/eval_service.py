"""Eval runner service — launches pytest eval tracks and parses their results.

This powers the admin-only Evaluation dashboard. It is deliberately isolated from
the RAG graph (it is an ops affordance, not part of the request pipeline) and is
only wired in when settings.evals_enabled is True.

Design:
  - Runs are executed in a background thread (one subprocess per selected track,
    sequentially — sequential execution respects Groq's per-minute token cap for
    judge-based tracks). The HTTP request returns immediately.
  - Each track writes a JUnit XML (machine-readable pass/fail) plus a captured
    stdout log (human-readable metrics the track prints).
  - Status is tracked with a module-level lock + flag; the API process is a single
    uvicorn worker, so in-memory state is authoritative for one container.

Only synthetic golden-set queries are ever run — never user data (data-handling.md).
"""

import subprocess
import sys
import threading
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.settings import settings
from src.utils.logger import logger

# --- Track catalog: key -> (human label, test file, kind) ---
# Deterministic tracks are fast; judge tracks hit Groq and are slow/rate-limited.
EVAL_TRACKS: dict[str, dict[str, str]] = {
    "rbac":          {"label": "RBAC red-team",              "file": "backend/evals/test_rbac.py",          "kind": "deterministic"},
    "safety":        {"label": "Medical safety guardrail",   "file": "backend/evals/test_safety.py",        "kind": "deterministic"},
    "chunk_quality": {"label": "Chunk quality",              "file": "backend/evals/test_chunk_quality.py", "kind": "deterministic"},
    "routing":       {"label": "Routing accuracy",           "file": "backend/evals/test_routing.py",       "kind": "deterministic"},
    "sql":           {"label": "SQL-RAG correctness",        "file": "backend/evals/test_sql.py",           "kind": "deterministic"},
    "retrieval":     {"label": "Retrieval A/B",              "file": "backend/evals/test_retrieval.py",     "kind": "deterministic"},
    "numbers":       {"label": "Number-match",               "file": "backend/evals/test_numbers.py",       "kind": "deterministic"},
    "faithfulness":  {"label": "Faithfulness (LLM-judge)",   "file": "backend/evals/test_faithfulness.py",  "kind": "judge"},
    "calibration":   {"label": "Judge calibration",          "file": "backend/evals/test_calibration.py",   "kind": "judge"},
}

# Repo root, from backend/src/services/eval_service.py:
# parents[0]=services, [1]=src, [2]=backend, [3]=repo root (where pytest.ini lives).
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# --- Run state (single-worker authoritative) ---
_lock = threading.Lock()
_state: dict[str, Any] = {
    "running": False,
    "current_track": None,
    "started_at": None,
    "finished_at": None,
    "tracks_requested": [],
}


def _results_dir() -> Path:
    """Absolute results directory, created if missing."""
    d = _PROJECT_ROOT / settings.eval_results_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_status() -> dict[str, Any]:
    """Return a snapshot of the current run state (thread-safe copy)."""
    with _lock:
        return dict(_state)


def start_run(tracks: list[str]) -> dict[str, Any]:
    """Launch the requested tracks in a background thread.

    Returns immediately with the new status. Raises ValueError if a run is
    already in progress or an unknown track key is supplied.
    """
    unknown = [t for t in tracks if t not in EVAL_TRACKS]
    if unknown:
        raise ValueError(f"Unknown eval track(s): {', '.join(unknown)}")
    if not tracks:
        raise ValueError("No tracks selected.")

    with _lock:
        if _state["running"]:
            raise ValueError("An eval run is already in progress.")
        _state.update(
            running=True,
            current_track=None,
            started_at=datetime.now(timezone.utc).isoformat(),
            finished_at=None,
            tracks_requested=list(tracks),
        )

    thread = threading.Thread(target=_run_tracks, args=(list(tracks),), daemon=True)
    thread.start()
    return get_status()


def run_all_blocking(tracks: list[str] | None = None) -> dict[str, Any]:
    """Run tracks synchronously and block until every one finishes.

    This is the entrypoint for the one-shot `evals` container (scripts/run_evals.py),
    which runs at `docker compose up` and populates evals/results/ before exiting.
    Unlike start_run(), this does NOT spawn a background thread — it blocks so the
    container stays alive for the full run and its exit code reflects completion.

    Writes the same per-track {key}.xml + {key}.log the dashboard reads, so results
    surface in the admin Evaluation tab automatically once the backend is up.

    Args:
        tracks: track keys to run. None → every track in EVAL_TRACKS.

    Returns:
        The final get_results() snapshot after all tracks complete.

    Raises:
        ValueError: if any requested track key is unknown.
    """
    keys = tracks if tracks else list(EVAL_TRACKS)
    unknown = [t for t in keys if t not in EVAL_TRACKS]
    if unknown:
        raise ValueError(f"Unknown eval track(s): {', '.join(unknown)}")

    with _lock:
        _state.update(
            running=True,
            current_track=None,
            started_at=datetime.now(timezone.utc).isoformat(),
            finished_at=None,
            tracks_requested=list(keys),
        )
    try:
        for key in keys:
            with _lock:
                _state["current_track"] = key
            logger.info("eval_track_blocking", extra={"track": key})
            _run_single_track(key)
    finally:
        with _lock:
            _state.update(
                running=False,
                current_track=None,
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
    return get_results()


def _run_tracks(tracks: list[str]) -> None:
    """Background worker: run each track's pytest sequentially."""
    try:
        for key in tracks:
            with _lock:
                _state["current_track"] = key
            _run_single_track(key)
    finally:
        with _lock:
            _state.update(
                running=False,
                current_track=None,
                finished_at=datetime.now(timezone.utc).isoformat(),
            )


def _run_single_track(key: str) -> None:
    """Run one track's pytest, writing JUnit XML + captured stdout log."""
    spec = EVAL_TRACKS[key]
    results = _results_dir()
    xml_path = results / f"{key}.xml"
    log_path = results / f"{key}.log"

    cmd = [
        sys.executable, "-m", "pytest", spec["file"],
        "-q", "-p", "no:cacheprovider",
        "--junitxml", str(xml_path),
    ]
    # PYTHONPATH = backend/ so the subprocess resolves `from src...`, `from evals...`,
    # `from ingestion...` (all live under backend/). Test file paths are backend/evals/*.
    env = {**_os_environ(), "PYTHONPATH": str(_PROJECT_ROOT / "backend")}

    logger.info("eval_track_start", extra={"track": key})
    try:
        completed = subprocess.run(
            cmd, cwd=str(_PROJECT_ROOT), env=env,
            capture_output=True, text=True, timeout=1800,
        )
        log_path.write_text(completed.stdout + "\n" + completed.stderr)
        logger.info("eval_track_done", extra={"track": key, "rc": completed.returncode})
    except subprocess.TimeoutExpired:
        log_path.write_text("Track timed out after 1800s.")
        logger.error("eval_track_timeout", extra={"track": key})


def _os_environ() -> dict[str, str]:
    import os
    return dict(os.environ)


def get_results() -> dict[str, Any]:
    """Parse persisted track results into a dashboard-ready table.

    Returns EVERY known track (not just the ones that have run) so the UI can
    render a complete status table. Never-run tracks get status 'not_run'.
    Each track carries its own last-run timestamp (`ran_at`).
    """
    results = _results_dir()
    tracks: list[dict[str, Any]] = []
    latest_mtime = 0.0

    for key, spec in EVAL_TRACKS.items():
        xml_path = results / f"{key}.xml"
        row: dict[str, Any] = {
            "key": key,
            "label": spec["label"],
            "kind": spec["kind"],
        }

        if not xml_path.exists():
            # Never run — still show it in the table.
            row.update(status="not_run", passed=None, ran_at=None,
                       metrics={}, summary="Not run yet", test_cases=[], log_tail="")
            tracks.append(row)
            continue

        mtime = xml_path.stat().st_mtime
        latest_mtime = max(latest_mtime, mtime)
        parsed = _parse_junit(xml_path)
        passed = parsed["failed"] == 0 and parsed["errors"] == 0 and parsed["total"] > 0

        row.update(
            status="passed" if passed else "failed",
            passed=passed,
            ran_at=datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
            metrics={
                "Tests": parsed["total"],
                "Passed": parsed["passed"],
                "Failed": parsed["failed"] + parsed["errors"],
            },
            summary=parsed["summary"],
            # Per-test breakdown (name + status + duration) for the dashboard table.
            test_cases=_parse_testcases(xml_path),
            log_tail=_read_log_tail(results / f"{key}.log"),
        )
        tracks.append(row)

    generated_at = (
        datetime.fromtimestamp(latest_mtime, tz=timezone.utc).isoformat()
        if latest_mtime
        else None
    )
    return {"generated_at": generated_at, "tracks": tracks}


def _parse_junit(xml_path: Path) -> dict[str, Any]:
    """Extract tallies from a JUnit XML file produced by pytest --junitxml."""
    try:
        root = ET.parse(xml_path).getroot()
        # Root may be <testsuites> wrapping <testsuite>, or a lone <testsuite>.
        suite = root.find("testsuite") if root.tag == "testsuites" else root
        total = int(suite.get("tests", 0))
        failures = int(suite.get("failures", 0))
        errors = int(suite.get("errors", 0))
        skipped = int(suite.get("skipped", 0))
        time = float(suite.get("time", 0.0))
        passed = total - failures - errors - skipped
        summary = (
            f"{passed} passed, {failures} failed, {errors} errors, "
            f"{skipped} skipped in {time:.1f}s"
        )
        return {
            "total": total, "passed": passed, "failed": failures,
            "errors": errors, "skipped": skipped, "summary": summary,
        }
    except (ET.ParseError, ValueError, AttributeError):
        return {
            "total": 0, "passed": 0, "failed": 0, "errors": 0,
            "skipped": 0, "summary": "Could not parse results.",
        }


def _parse_testcases(xml_path: Path) -> list[dict[str, Any]]:
    """Extract per-testcase name + status + duration from a JUnit XML file.

    Each <testcase> maps to one row in the dashboard's per-test table. A testcase
    with no child element passed; a <failure>/<error>/<skipped> child sets the status.
    Parametrised cases keep their id suffix (e.g. 'test_no_rbac_leak[A-001]'), so the
    table shows every golden case individually.
    """
    try:
        root = ET.parse(xml_path).getroot()
    except (ET.ParseError, ValueError):
        return []

    cases: list[dict[str, Any]] = []
    for tc in root.iter("testcase"):
        failure = tc.find("failure")
        error = tc.find("error")
        skipped = tc.find("skipped")
        if failure is not None:
            status, detail = "failed", failure
        elif error is not None:
            status, detail = "error", error
        elif skipped is not None:
            status, detail = "skipped", skipped
        else:
            status, detail = "passed", None

        # Prefer the bare test name; fall back to classname::name if name is absent.
        name = tc.get("name") or tc.get("classname", "unknown")
        try:
            duration = float(tc.get("time", 0.0))
        except (TypeError, ValueError):
            duration = 0.0
        cases.append(
            {
                "name": name,
                "status": status,
                "time": duration,
                # Concise, human-readable reason for a non-pass — never the raw traceback.
                "reason": _friendly_reason(detail) if detail is not None else "",
            }
        )
    return cases


# Known low-level failure signatures → a concise, actionable explanation. Keeps raw
# tracebacks out of the dashboard: the operator sees WHY, not a 20-line stack.
_ERROR_SIGNATURES: tuple[tuple[str, str], ...] = (
    ("tokens per day", "Groq daily token limit exhausted — retry after the quota resets or upgrade tier."),
    ("RateLimitError", "Groq rate limit reached — the run made too many/too-large calls too fast."),
    ("rate_limit_exceeded", "Groq rate limit reached — the run made too many/too-large calls too fast."),
    ("ConnectionError", "Could not reach a dependency (Qdrant/Groq/DB). Check the stack is up."),
    ("ResponseHandlingException", "Qdrant not reachable — is the qdrant service running?"),
    ("RBACError", "RBAC blocked the operation — check the role/collection mapping."),
)


def _friendly_reason(detail: "ET.Element") -> str:
    """Turn a JUnit <failure>/<error>/<skipped> node into a one-line, informative reason.

    Prefers a known signature match (rate limit, connectivity, RBAC). Otherwise falls
    back to the node's `message` attribute (pytest's short summary — e.g. the assertion
    message), never the full multi-line traceback in the node body.
    """
    body = (detail.get("message") or detail.text or "").strip()
    for needle, explanation in _ERROR_SIGNATURES:
        if needle in body:
            return explanation
    # Fall back to the first line of the short message — assertion text, skip reason, etc.
    first_line = body.splitlines()[0] if body else ""
    return first_line[:200]


def _read_log_tail(log_path: Path, lines: int = 40) -> str:
    """Return the last N lines of a captured track log, if present."""
    if not log_path.exists():
        return ""
    text = log_path.read_text(errors="replace").splitlines()
    return "\n".join(text[-lines:])
