"""
SQLite store for anonymous per-session state.

⚠️  The database file lives on the container filesystem — docker-compose
mounts ./data as a volume locally, but an ephemeral PaaS filesystem wipes it
on every redeploy. Mount persistent storage, or move to Postgres.

There is no login. Every caller identifies itself with a client-generated
`X-Session-Id` header (see app.main.get_session_id), stored as `owner` on
every table below — each session gets a private KB and private chat history.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from app.config import get_settings

log = logging.getLogger(__name__)

# Schema only needs to be created once per process, not on every connection.
_schema_ready = False


def _db_path() -> str:
    # Read lazily (not at import) so tests can point it at a temp directory.
    return get_settings().db_path


def _init_schema() -> None:
    """Create tables — once per process."""
    global _schema_ready
    if _schema_ready:
        return

    path = _db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        # WAL lets readers proceed during a write. busy_timeout is per-connection
        # (also set in _connect() for every later connection) so a BackgroundTask
        # write racing another one waits instead of raising "database is locked".
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")

        # Auth is gone — drop leftovers from a previous deploy.
        conn.execute("DROP TABLE IF EXISTS users")
        conn.execute("DROP TABLE IF EXISTS revoked_tokens")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS sources (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                owner      TEXT    NOT NULL,
                source     TEXT    NOT NULL,
                kind       TEXT    NOT NULL,
                chunks     INTEGER NOT NULL DEFAULT 0,
                commit_sha TEXT    DEFAULT '',
                created    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        _rename_column_if_needed(conn, "sources", "username", "owner")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sources_owner ON sources(owner)")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                owner    TEXT    NOT NULL,
                title    TEXT    NOT NULL DEFAULT 'New conversation',
                created  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        _rename_column_if_needed(conn, "conversations", "username", "owner")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_owner ON conversations(owner)")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id),
                role            TEXT    NOT NULL,
                content         TEXT    NOT NULL,
                sources         TEXT,
                created         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id)")

        # Per-session, per-day question counter backing the free daily limit.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS usage (
                owner TEXT    NOT NULL,
                day   TEXT    NOT NULL,
                count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (owner, day)
            )
        """)
        conn.commit()
    finally:
        conn.close()

    _schema_ready = True


def _rename_column_if_needed(conn: sqlite3.Connection, table: str, old: str, new: str) -> None:
    """Idempotent column rename, for database files that predate it."""
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if old in cols and new not in cols:
        conn.execute(f"ALTER TABLE {table} RENAME COLUMN {old} TO {new}")


@contextmanager
def _connect():
    """Yield a row-factory connection, guaranteeing close() on any path."""
    _init_schema()
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")  # per-connection, see _init_schema
    try:
        yield conn
    finally:
        conn.close()


# ── Sources (per-session "what have I ingested") ────────────────
def record_source(session_id: str, source: str, kind: str, chunks: int, commit_sha: str = "") -> None:
    """Record a successful ingest so it shows up under GET /sources/mine and
    can later be deleted by its owner."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sources (owner, source, kind, chunks, commit_sha, created) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, source, kind, chunks, commit_sha, datetime.now(UTC).isoformat()),
        )
        conn.commit()


def list_sources_for_user(session_id: str) -> list[dict]:
    """List everything `session_id` has ingested, most recent first."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, owner, source, kind, chunks, commit_sha, created "
            "FROM sources WHERE owner = ? ORDER BY created DESC",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def delete_source_row(source: str, session_id: str) -> int:
    """Delete `sources` rows for `source` owned by `session_id`. Returns the
    number of rows removed."""
    with _connect() as conn:
        cur = conn.execute(
            "DELETE FROM sources WHERE source = ? AND owner = ?", (source, session_id)
        )
        conn.commit()
        return cur.rowcount


# ── Conversations + messages (per-session private chat history) ────
def create_conversation(session_id: str, title: str | None = None) -> dict:
    now = datetime.now(UTC).isoformat()
    title = (title or "New conversation").strip() or "New conversation"
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO conversations (owner, title, created, updated) VALUES (?, ?, ?, ?)",
            (session_id, title, now, now),
        )
        conn.commit()
        conv_id = cur.lastrowid
    return {"id": conv_id, "owner": session_id, "title": title, "created": now, "updated": now}


def list_conversations(session_id: str) -> list[dict]:
    """Most-recently-updated first — the usual "recent chats" ordering."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, owner, title, created, updated FROM conversations "
            "WHERE owner = ? ORDER BY updated DESC",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_conversation(conversation_id: int) -> dict | None:
    """Fetch a conversation's own row (not its messages) — callers use this to
    check ownership before returning 404 vs the actual content."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, owner, title, created, updated FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        return dict(row) if row else None


def get_conversation_messages(conversation_id: int) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, conversation_id, role, content, sources, created FROM messages "
            "WHERE conversation_id = ? ORDER BY id ASC",
            (conversation_id,),
        ).fetchall()
        out = []
        for row in rows:
            d = dict(row)
            try:
                d["sources"] = json.loads(d["sources"]) if d["sources"] else []
            except (TypeError, ValueError):
                d["sources"] = []
            out.append(d)
        return out


def add_message(
    conversation_id: int, role: str, content: str, sources: list[str] | None = None
) -> dict:
    """Append a message and bump the conversation's `updated` timestamp (what
    the "recent chats" list is ordered by)."""
    now = datetime.now(UTC).isoformat()
    sources_json = json.dumps(sources or [])
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO messages (conversation_id, role, content, sources, created) "
            "VALUES (?, ?, ?, ?, ?)",
            (conversation_id, role, content, sources_json, now),
        )
        conn.execute("UPDATE conversations SET updated = ? WHERE id = ?", (now, conversation_id))
        conn.commit()
        msg_id = cur.lastrowid
    return {
        "id": msg_id,
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
        "sources": sources or [],
        "created": now,
    }


def delete_conversation(conversation_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        conn.commit()


# ── Daily free-question usage (per-session) ─────────────────────
def _today() -> str:
    return datetime.now(UTC).date().isoformat()  # UTC, not local time


def get_usage_count(session_id: str, day: str | None = None) -> int:
    """Return how many free questions `session_id` has used on `day` (UTC,
    defaults to today)."""
    day = day or _today()
    with _connect() as conn:
        row = conn.execute(
            "SELECT count FROM usage WHERE owner = ? AND day = ?", (session_id, day)
        ).fetchone()
        return row["count"] if row else 0


def increment_usage(session_id: str, day: str | None = None) -> int:
    """Increment and return today's count for `session_id`. Only called for
    free-key (no X-Api-Key) requests — BYOK requests never touch this table."""
    day = day or _today()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO usage (owner, day, count) VALUES (?, ?, 1) "
            "ON CONFLICT(owner, day) DO UPDATE SET count = count + 1",
            (session_id, day),
        )
        conn.commit()
        row = conn.execute(
            "SELECT count FROM usage WHERE owner = ? AND day = ?", (session_id, day)
        ).fetchone()
        return row["count"]
