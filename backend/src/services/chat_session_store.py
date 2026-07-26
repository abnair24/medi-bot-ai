"""Per-user chat-session index — lets a user see their past chats after logging back in.

The conversation CONTENT (questions + answers) is persisted by the LangGraph checkpointer,
keyed by thread_id. But the checkpointer has no notion of *which user* owns a thread, so
there's no way to list "my chats". This store fills that gap with a tiny index table:

    chat_sessions(session_id PK, username, created_at, updated_at, turns)

PHI note: the title shown in the sidebar is derived from `created_at` (a timestamp), NOT
the user's question text — consistent with the rule "never persist raw query text as a
label". The store holds only IDs, the owning username, timestamps, and a turn counter.

Connections are opened per-operation (like SQLAdapter) — no lifespan wiring needed.
"""

from datetime import datetime, timezone

import aiosqlite

from src.config.settings import settings

_CREATE = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    username   TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    turns      INTEGER NOT NULL DEFAULT 0
)
"""


class ChatSessionStore:
    """SQLite-backed index of a user's chat sessions (metadata only, never query text)."""

    def __init__(self) -> None:
        self._path = settings.sessions_db_path

    async def touch(self, session_id: str, username: str) -> None:
        """Record activity on a session: create it on first use, else bump updated_at + turns."""
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self._path) as conn:
            await conn.execute(_CREATE)
            await conn.execute(
                """
                INSERT INTO chat_sessions (session_id, username, created_at, updated_at, turns)
                VALUES (?, ?, ?, ?, 1)
                ON CONFLICT(session_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    turns = chat_sessions.turns + 1
                """,
                (session_id, username, now, now),
            )
            await conn.commit()

    async def list_for_user(self, username: str) -> list[dict]:
        """Return the user's sessions, newest activity first (metadata only)."""
        async with aiosqlite.connect(self._path) as conn:
            await conn.execute(_CREATE)
            conn.row_factory = aiosqlite.Row
            cursor = await conn.execute(
                "SELECT session_id, created_at, updated_at, turns "
                "FROM chat_sessions WHERE username = ? ORDER BY updated_at DESC",
                (username,),
            )
            rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def delete(self, session_id: str) -> None:
        """Remove a session from the index. Idempotent — a no-op if it's already gone.

        Deletes only the index row here; the caller also purges the conversation CONTENT
        from the LangGraph checkpointer (adelete_thread), so both stores stay consistent.
        """
        async with aiosqlite.connect(self._path) as conn:
            await conn.execute(_CREATE)
            await conn.execute(
                "DELETE FROM chat_sessions WHERE session_id = ?", (session_id,)
            )
            await conn.commit()

    async def owner_of(self, session_id: str) -> str | None:
        """Return the username that owns a session, or None if unknown — for load auth."""
        async with aiosqlite.connect(self._path) as conn:
            await conn.execute(_CREATE)
            cursor = await conn.execute(
                "SELECT username FROM chat_sessions WHERE session_id = ?", (session_id,)
            )
            row = await cursor.fetchone()
        return row[0] if row else None
