"""MediBot Streamlit frontend.

Tabs:
  - 💬 Assistant : interactive RAG chat with HITL SQL approval (all roles).
  - 📊 Evaluation : admin-only eval trigger + report dashboard.

Run:   streamlit run frontend/app.py
Env:   MEDIBOT_API_URL  (default: http://localhost:8000)
"""

import os
import uuid
from datetime import datetime, timezone
from typing import Any

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Config — no hardcoded URLs
# ---------------------------------------------------------------------------

API_BASE: str = os.getenv("MEDIBOT_API_URL", "http://localhost:8000")

# Demo accounts (ARCHITECTURE.md §6). Held here ONLY to power one-click role
# login buttons — passwords are never rendered to the page. The real password
# check still happens server-side via POST /login.
DEMO_ACCOUNTS: list[dict[str, str]] = [
    {"label": "Doctor", "icon": "🔵", "username": "dr.mehta", "password": "doctor123"},
    {"label": "Nurse", "icon": "🟢", "username": "nurse.priya", "password": "nurse123"},
    {"label": "Billing", "icon": "🟡", "username": "billing.ravi", "password": "billing123"},
    {"label": "Technician", "icon": "🟠", "username": "tech.anand", "password": "tech123"},
    {"label": "Admin", "icon": "🔴", "username": "admin.sys", "password": "admin123"},
]

ROLE_ICON: dict[str, str] = {
    "doctor": "🔵",
    "nurse": "🟢",
    "billing_executive": "🟡",
    "technician": "🟠",
    "admin": "🔴",
}

RETRIEVAL_BADGE: dict[str, str] = {
    "hybrid_rag": "🔍 Hybrid RAG",
    "sql_rag": "🗄️ SQL RAG",
}

ADMIN_ROLE = "admin"

# ---------------------------------------------------------------------------
# API client helpers — thin wrappers over the FastAPI backend
# ---------------------------------------------------------------------------


def _auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def api_login(username: str, password: str) -> dict[str, Any] | None:
    """POST /login → LoginResponse dict on success, None on 401."""
    try:
        resp = requests.post(
            f"{API_BASE}/login",
            json={"username": username, "password": password},
            timeout=10,
        )
        if resp.status_code == 401:
            return None
        resp.raise_for_status()
        return resp.json()
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise ConnectionError(
            "MediBot backend is not reachable yet — it may still be starting up "
            "(loading models). Please wait a few seconds and try again."
        ) from exc


def api_get_collections(token: str) -> list[str]:
    """GET /collections → list of collection names for the authenticated role."""
    resp = requests.get(
        f"{API_BASE}/collections", headers=_auth_header(token), timeout=10
    )
    resp.raise_for_status()
    return resp.json().get("collections", [])


def api_chat(question: str, session_id: str, role: str, token: str) -> dict[str, Any]:
    """POST /chat → ChatResponse dict or SQLProposal dict (HITL pending)."""
    resp = requests.post(
        f"{API_BASE}/chat",
        json={"question": question, "session_id": session_id, "role": role},
        headers=_auth_header(token),
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()


def api_resume(confirmation_token: str, approved: bool, token: str) -> dict[str, Any]:
    """POST /chat/resume → ConfirmResponse dict after HITL approve/reject."""
    resp = requests.post(
        f"{API_BASE}/chat/resume",
        json={"confirmation_token": confirmation_token, "approved": approved},
        headers=_auth_header(token),
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()


def api_eval_results(token: str) -> dict[str, Any] | None:
    """GET /evals/results → aggregated track results, or None if endpoint absent."""
    try:
        resp = requests.get(
            f"{API_BASE}/evals/results", headers=_auth_header(token), timeout=15
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        return None


def api_eval_status(token: str) -> dict[str, Any] | None:
    """GET /evals/status → {running, started_at, last_finished, ...} or None."""
    try:
        resp = requests.get(
            f"{API_BASE}/evals/status", headers=_auth_header(token), timeout=10
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        return None


def api_eval_run(tracks: list[str], token: str) -> dict[str, Any]:
    """POST /evals/run → launches a background eval run (admin only)."""
    resp = requests.post(
        f"{API_BASE}/evals/run",
        json={"tracks": tracks},
        headers=_auth_header(token),
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Session state — single source of truth across re-runs
# ---------------------------------------------------------------------------


def init_session_state() -> None:
    """Initialise all session_state keys with safe defaults (one-time)."""
    defaults: dict[str, Any] = {
        "token": None,
        "role": None,
        "username": None,
        "collections": [],
        "session_id": None,
        "messages": [],
        "pending_sql": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _complete_login(data: dict[str, Any]) -> None:
    """Populate session_state from a LoginResponse and rerun into the app."""
    try:
        collections = api_get_collections(data["access_token"])
    except Exception:
        collections = []

    st.session_state.update(
        {
            "token": data["access_token"],
            "role": data["role"],
            "username": data["username"],
            "collections": collections,
            "session_id": str(uuid.uuid4()),
            "messages": [],
            "pending_sql": None,
        }
    )
    st.rerun()


# ---------------------------------------------------------------------------
# Login page — centered, no credentials rendered
# ---------------------------------------------------------------------------


def render_login() -> None:
    """Centered login. One-click role buttons + optional manual sign-in."""
    # Constrain width: three columns, render only in the middle one. This is the
    # Streamlit idiom for a centered "card" on a wide-layout page.
    _, center, _ = st.columns([1, 1.4, 1])

    with center:
        st.markdown("<div style='height:6vh'></div>", unsafe_allow_html=True)
        st.title("🏥 MediBot")
        st.caption("Internal AI Assistant")
        st.divider()

        st.markdown("**Quick sign-in**")
        st.caption("Pick a role to explore RBAC-scoped retrieval.")

        # One-click role buttons, two per row. No passwords shown; the button
        # calls the real /login with stored demo creds behind the scenes.
        for i in range(0, len(DEMO_ACCOUNTS), 2):
            cols = st.columns(2)
            for col, acct in zip(cols, DEMO_ACCOUNTS[i : i + 2]):
                with col:
                    if st.button(
                        f"{acct['icon']} {acct['label']}",
                        use_container_width=True,
                        key=f"quick_{acct['username']}",
                    ):
                        _quick_login(acct)

        st.divider()

        with st.expander("Sign in manually"):
            with st.form("login_form"):
                username_input = st.text_input("Username")
                password_input = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Sign in", use_container_width=True)

            if submitted:
                _manual_login(username_input, password_input)


def _quick_login(acct: dict[str, str]) -> None:
    """Authenticate a one-click demo role."""
    with st.spinner(f"Signing in as {acct['label']}…"):
        try:
            data = api_login(acct["username"], acct["password"])
        except ConnectionError as exc:
            st.error(str(exc))
            return
    if data is None:
        st.error("Demo account rejected — is the backend seeded correctly?")
        return
    _complete_login(data)


def _manual_login(username: str, password: str) -> None:
    """Authenticate a manually-entered credential pair."""
    if not username or not password:
        st.error("Enter both username and password.")
        return
    with st.spinner("Authenticating…"):
        try:
            data = api_login(username, password)
        except ConnectionError as exc:
            st.error(str(exc))
            return
    if data is None:
        st.error("Invalid username or password.")
        return
    _complete_login(data)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar() -> None:
    """Role badge, accessible collections, and logout."""
    with st.sidebar:
        role = st.session_state.role
        icon = ROLE_ICON.get(role, "⚪")

        st.markdown(f"### {icon} {st.session_state.username}")
        st.caption(f"Role: **{role}**")
        st.divider()

        st.markdown("**Accessible collections**")
        for col in st.session_state.collections:
            st.markdown(f"- `{col}`")

        st.divider()
        st.caption(f"Session: `{st.session_state.session_id[:8]}…`")

        if st.button("Sign out", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ---------------------------------------------------------------------------
# Chat history renderer
# ---------------------------------------------------------------------------


def render_chat_history() -> None:
    """Render past messages: question left, answer right — each in its own box.

    Alignment uses column offsets (not fragile CSS role selectors): a question renders
    in the left column of a [3, 2] split; an answer in the right column of a [2, 3]
    split. Both wrap their content in a bordered container so the answer reads as a
    distinct box, matching the question bubble instead of bare, trimmed-looking text.
    """
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            # Question: left-aligned, boxed.
            left, _ = st.columns([3, 2])
            with left, st.chat_message("user"), st.container(border=True):
                st.markdown(msg["content"])
        else:
            # Answer: right-aligned, boxed for parity with the question.
            _, right = st.columns([2, 3])
            with right, st.chat_message("assistant"), st.container(border=True):
                st.markdown(msg["content"])
                if retrieval_type := msg.get("retrieval_type"):
                    st.caption(RETRIEVAL_BADGE.get(retrieval_type, retrieval_type))
                if sources := msg.get("sources"):
                    _render_sources(sources)


def _render_sources(sources: list[dict]) -> None:
    """Render source citations inside a collapsible expander."""
    with st.expander(f"Sources ({len(sources)})"):
        for i, src in enumerate(sources, 1):
            st.markdown(
                f"**{i}.** `{src['source_document']}` — *{src['section_title']}* "
                f"(collection: `{src['collection']}`)"
            )


# ---------------------------------------------------------------------------
# HITL SQL approval UI
# ---------------------------------------------------------------------------


def render_hitl_approval() -> None:
    """Show proposed SQL + Approve/Reject when a HITL proposal is pending."""
    pending: dict | None = st.session_state.pending_sql
    if pending is None:
        return

    st.warning("**SQL query proposed — review before execution.**", icon="⚠️")

    with st.expander("Proposed SQL", expanded=True):
        st.code(pending["proposed_sql"], language="sql")

    st.caption(f"Proposal expires at: {pending.get('expires_at', 'unknown')}")

    col_approve, col_reject = st.columns(2)
    with col_approve:
        if st.button("✅ Approve", type="primary", use_container_width=True):
            _resolve_hitl(approved=True)
    with col_reject:
        if st.button("❌ Reject", use_container_width=True):
            _resolve_hitl(approved=False)


def _resolve_hitl(approved: bool) -> None:
    """Call /chat/resume, append result to history, clear pending proposal."""
    pending = st.session_state.pending_sql

    with st.spinner("Processing…"):
        try:
            result = api_resume(
                confirmation_token=pending["confirmation_token"],
                approved=approved,
                token=st.session_state.token,
            )
        except requests.HTTPError as exc:
            st.error(_http_error_message(exc))
            st.session_state.pending_sql = None
            return

    st.session_state.pending_sql = None

    content = (
        "SQL query rejected. The query was not executed."
        if result.get("rejected")
        else (result.get("answer") or "Query executed but no answer was returned.")
    )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": content,
            "retrieval_type": "sql_rag",
            "sources": [],
        }
    )
    st.rerun()


# ---------------------------------------------------------------------------
# Message handler
# ---------------------------------------------------------------------------


def _http_error_message(exc: requests.HTTPError) -> str:
    """Turn a backend HTTPError into a clean, user-facing chat message.

    The backend returns a JSON body with a `detail` field for every mapped error —
    including the 429 rate-limit message ("You are rate limited (daily limit). Please
    try again in 4m33s."). We surface that detail verbatim rather than dumping the raw
    JSON/status. A rate-limit (429) gets a ⏳ prefix so it reads as a transient wait,
    not a hard failure.
    """
    resp = exc.response
    if resp is None:
        return f"⚠️ Request failed: {exc}"
    try:
        detail = resp.json().get("detail") or resp.text
    except ValueError:
        detail = resp.text or str(exc)
    if resp.status_code == 429:
        return f"⏳ {detail}"
    return f"⚠️ {detail}"


def handle_new_question(question: str) -> None:
    """Append user turn, call /chat, update state with assistant turn or HITL proposal."""
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Thinking…"):
        try:
            data = api_chat(
                question=question,
                session_id=st.session_state.session_id,
                role=st.session_state.role,
                token=st.session_state.token,
            )
        except requests.HTTPError as exc:
            st.session_state.messages.append(
                {"role": "assistant", "content": _http_error_message(exc)}
            )
            return
        except ConnectionError as exc:
            st.session_state.messages.append(
                {"role": "assistant", "content": str(exc)}
            )
            return

    if data.get("needs_confirmation"):
        st.session_state.pending_sql = data
    else:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": data.get("answer", ""),
                "retrieval_type": data.get("retrieval_type", "hybrid_rag"),
                "sources": data.get("sources", []),
            }
        )


# ---------------------------------------------------------------------------
# Tab: Assistant (chat)
# ---------------------------------------------------------------------------


def render_assistant_tab() -> None:
    """Interactive chat demo with HITL support and chat-style message alignment."""
    if not st.session_state.messages:
        # Empty-state hint, left-aligned to the SAME column as questions + the input
        # box, so the empty and active states are visually consistent.
        left, _ = st.columns([3, 2])
        with left:
            st.info(
                "Ask about internal protocols, policies, or (billing/admin) patient "
                "records. Answers are grounded in retrieved documents with citations.",
                icon="💬",
            )

    # Message history handles its own per-message left/right alignment at top level —
    # not wrapped in a center column, so the offset columns can reach both edges.
    render_chat_history()

    # HITL approval prompt, centered for prominence.
    if st.session_state.pending_sql is not None:
        _, center, _ = st.columns([1, 3, 1])
        with center:
            render_hitl_approval()

    # chat_input must live at the top level (not inside columns) to dock at the
    # bottom of the viewport; its width is constrained via CSS. Disabled while a
    # HITL proposal is pending.
    if st.session_state.pending_sql is None:
        question = st.chat_input("Ask a question about internal protocols…")
        if question:
            handle_new_question(question)
            st.rerun()


# ---------------------------------------------------------------------------
# Tab: Evaluation (admin only)
# ---------------------------------------------------------------------------

# Status metadata: backend status string -> (chip icon+text).
_STATUS_CHIP: dict[str, str] = {
    "passed": "✅ Passed",
    "failed": "❌ Failed",
    "not_run": "⚪ Not run",
    "running": "⏳ Running…",
}
# Per-test status chips — a superset of track statuses (error/skipped are test-level only).
_TEST_STATUS_CHIP: dict[str, str] = {
    "passed": "✅ Pass",
    "failed": "❌ Fail",
    "error": "🔥 Error",
    "skipped": "⚪ Skip",
}
_KIND_BADGE: dict[str, str] = {"deterministic": "⚡ fast", "judge": "🧠 judge"}


def _fmt_ts(iso: str | None) -> str:
    """Format an ISO timestamp as 'MMM DD HH:MM (Nm ago)'. '—' if never run."""
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - dt
        secs = int(delta.total_seconds())
        if secs < 60:
            rel = f"{secs}s ago"
        elif secs < 3600:
            rel = f"{secs // 60}m ago"
        elif secs < 86400:
            rel = f"{secs // 3600}h ago"
        else:
            rel = f"{secs // 86400}d ago"
        return f"{dt.strftime('%b %d %H:%M')} ({rel})"
    except (ValueError, TypeError):
        return iso


def _render_test_table(cases: list[dict]) -> None:
    """Render individual test cases as a Test / Status / Reason·Duration table.

    Mirrors the UX of the top-level track table (header row + per-row columns), one
    level down: each row is a single pytest test case (including parametrised golden
    cases like `test_no_rbac_leak[A-001]`). For a failed/error/skipped test the last
    column shows a concise, human-readable REASON (from the backend, never a raw
    traceback); for a passed test it shows the duration.
    """
    if not cases:
        st.caption("No per-test detail reported for this track.")
        return

    # A track-level banner summarising the dominant failure reason, if any failed.
    reasons = [c.get("reason", "") for c in cases if c.get("status") in ("failed", "error")]
    if reasons:
        # Show the most common non-empty reason as the headline.
        headline = max((r for r in reasons if r), key=reasons.count, default="")
        if headline:
            st.warning(headline, icon="⚠️")

    # Header row.
    hcols = st.columns([4, 1.3, 3.2])
    for c, head in zip(hcols, ["Test", "Status", "Reason / Duration"]):
        c.markdown(f"**{head}**")

    # One row per test case.
    for case in cases:
        status = case.get("status", "")
        cols = st.columns([4, 1.3, 3.2])
        cols[0].markdown(f"`{case.get('name', 'unknown')}`")
        cols[1].markdown(_TEST_STATUS_CHIP.get(status, status))
        if status == "passed":
            cols[2].caption(f"{case.get('time', 0.0):.2f}s")
        else:
            cols[2].caption(case.get("reason") or status)


def render_eval_tab() -> None:
    """Admin-only eval dashboard: per-track status table with re-trigger."""
    st.header("📊 Evaluation")
    token = st.session_state.token

    status = api_eval_status(token)
    if status is None:
        st.info(
            "Eval endpoints are not available yet. Once the backend exposes "
            "`/evals/*`, you'll be able to trigger runs and view reports here.",
            icon="🚧",
        )
        return

    running: bool = status.get("running", False)
    current: str | None = status.get("current_track")
    results = api_eval_results(token) or {}
    tracks: list[dict[str, Any]] = results.get("tracks", [])

    # --- Top action bar ---
    # Caption on its own line, then all three actions in a single aligned button row
    # (Run all · Run deterministic · Refresh) so Refresh lines up with the others.
    st.caption(
        f"Last run: {_fmt_ts(results.get('generated_at'))}  ·  "
        "⚡ deterministic (fast) · 🧠 LLM-judge (slow, rate-limited)"
    )

    all_keys = [t["key"] for t in tracks]
    det_keys = [t["key"] for t in tracks if t.get("kind") == "deterministic"]
    b1, b2, b3, _ = st.columns([1.3, 1.6, 1.1, 2.5])
    with b1:
        if st.button("▶️ Run all", disabled=running or not all_keys,
                     use_container_width=True):
            api_eval_run(all_keys, token)
            st.rerun()
    with b2:
        if st.button("⚡ Run deterministic", disabled=running or not det_keys,
                     use_container_width=True):
            api_eval_run(det_keys, token)
            st.rerun()
    with b3:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    if running:
        st.warning(
            f"Run in progress — currently: **{current or '…'}**. "
            "Use 🔄 Refresh to update.",
            icon="⏳",
        )

    st.divider()

    # --- Status table (header row) ---
    cols = st.columns([3, 1.2, 1.4, 2.4, 2.6, 1.2])
    for c, head in zip(cols, ["Track", "Type", "Status", "Last run", "Result", ""]):
        c.markdown(f"**{head}**")

    if not tracks:
        st.info("No tracks reported by the backend.")
        return

    for t in tracks:
        key = t["key"]
        row_status = "running" if (running and current == key) else t.get("status", "not_run")
        cols = st.columns([3, 1.2, 1.4, 2.4, 2.6, 1.2])
        cols[0].markdown(t.get("label", key))
        cols[1].caption(_KIND_BADGE.get(t.get("kind", ""), t.get("kind", "")))
        cols[2].markdown(_STATUS_CHIP.get(row_status, row_status))
        cols[3].caption(_fmt_ts(t.get("ran_at")))
        cols[4].caption(t.get("summary") or "—")
        with cols[5]:
            # Per-row re-trigger. Disabled while any run is active.
            if st.button("▶ Re-run", key=f"rerun_{key}", disabled=running,
                         use_container_width=True):
                api_eval_run([key], token)
                st.rerun()

    # --- Per-track run output: individual test names + status (table UX) ---
    ran = [t for t in tracks if t.get("test_cases") or t.get("log_tail")]
    if ran:
        st.divider()
        st.markdown("**Run output**")
        for t in ran:
            cases = t.get("test_cases", [])
            passed = sum(1 for c in cases if c.get("status") == "passed")
            header = (
                f"{_STATUS_CHIP.get(t.get('status'), '')} · {t.get('label', t['key'])}"
                + (f"  ({passed}/{len(cases)} passed)" if cases else "")
            )
            with st.expander(header):
                _render_test_table(cases)
                # Raw pytest log kept one level deeper for failure debugging.
                if t.get("log_tail"):
                    with st.expander("Raw log"):
                        st.code(t["log_tail"], language="text")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


# Global style overrides. Streamlit renders Markdown blockquotes (the mandated format
# for protocol excerpts) in faded grey, which made the actual answer look secondary to
# the italic disclaimer. This re-establishes the visual hierarchy — answer prominent,
# disclaimer muted — and constrains the chat input to match the conversation width.
_CUSTOM_CSS = """
<style>
/* Answer blockquotes: un-fade + accent border so the answer reads as primary. */
[data-testid="stChatMessage"] blockquote {
    border-left: 4px solid #ff4b4b;
    background: #f6f8fa;
    color: rgba(0, 0, 0, 0.88);
    padding: 0.55rem 1rem;
    border-radius: 6px;
    margin: 0.3rem 0;
}
/* Disclaimer (italic) sits BELOW the answer as muted, smaller secondary text. */
[data-testid="stChatMessage"] em {
    color: rgba(0, 0, 0, 0.5);
    font-size: 0.85em;
}
/* Left-align the docked chat input to the SAME 60% width and left edge as the
   question bubble (st.columns([3, 2]) left column). The user types here and the
   text becomes a left-aligned question, so the input sits on the question side —
   consistent instead of floating centered. */
[data-testid="stChatInput"] {
    max-width: 60%;
    margin-left: 0;
    margin-right: auto;
}
</style>
"""


def main() -> None:
    st.set_page_config(page_title="MediBot", page_icon="🏥", layout="wide")
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

    init_session_state()

    # Pre-login: centered login page, no tabs.
    if not st.session_state.token:
        render_login()
        return

    # Post-login: sidebar + tabs. Eval tab is admin-only.
    render_sidebar()

    is_admin = st.session_state.role == ADMIN_ROLE
    tab_labels = ["💬 Assistant"] + (["📊 Evaluation"] if is_admin else [])
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        render_assistant_tab()

    if is_admin:
        with tabs[1]:
            render_eval_tab()


if __name__ == "__main__":
    main()
