"""
SQLite store for users and revoked JWTs.

WHY SQLite? Zero-config, serverless, ships with Python. For a portfolio project
with < 1000 users it is the right call.

⚠️  OPERATIONAL CAVEAT: the database file is inside the container filesystem.
Locally, docker-compose mounts ./data as a volume so it survives restarts. On a
PaaS with an ephemeral filesystem (Azure App Service without a mounted share),
every redeploy WIPES the user table and re-seeds admin. Mount persistent
storage at /app/data, or move to Postgres, before treating accounts as durable.
"""

import logging
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from app.config import get_settings

log = logging.getLogger(__name__)

# WHY a module-level guard? The schema + admin seed only need to run once per
# process. Previously CREATE TABLE and the admin-seed SELECT ran on *every*
# connection — i.e. on every auth check on every request.
_schema_ready = False


def _db_path() -> str:
    # Read lazily (not at import) so tests can point it at a temp directory.
    return get_settings().db_path


def _init_schema() -> None:
    """Create tables and seed the admin account — once per process."""
    global _schema_ready
    if _schema_ready:
        return

    settings = get_settings()
    path = _db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        # WAL lets readers proceed during a write instead of blocking on the
        # single writer lock — the difference between usable and not once more
        # than one request is in flight.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                username  TEXT    UNIQUE NOT NULL,
                password  TEXT    NOT NULL,
                role      TEXT    NOT NULL DEFAULT 'user',
                created   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Revocation denylist. A stateless JWT cannot otherwise be invalidated
        # before it expires, so "logout" would be client-side theatre and a
        # stolen token would stay valid for its full lifetime.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS revoked_tokens (
                jti        TEXT PRIMARY KEY,
                expires_at TIMESTAMP NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_revoked_expiry ON revoked_tokens(expires_at)")
        conn.commit()

        # Seed the admin account on first init so the app is usable straight
        # after deploy. config.py already refuses to start in production without
        # an explicit ADMIN_PASSWORD; this is the second layer of that check.
        row = conn.execute("SELECT id FROM users WHERE username = 'admin'").fetchone()
        if not row:
            import bcrypt

            admin_pw = settings.admin_password
            if not admin_pw:
                if settings.is_production:
                    raise RuntimeError(
                        "ADMIN_PASSWORD must be set in production to seed the admin "
                        "account. Refusing to create an admin with a known default."
                    )
                admin_pw = "admin123"  # dev convenience only — see .env.example
            hashed = bcrypt.hashpw(admin_pw.encode(), bcrypt.gensalt()).decode()
            conn.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                ("admin", hashed, "admin"),
            )
            conn.commit()
            log.info(
                "Admin account created (username: admin). "
                "Set the ADMIN_PASSWORD env var to control the password."
            )
    finally:
        conn.close()

    _schema_ready = True


@contextmanager
def _connect():
    """Yield a row-factory connection, guaranteeing close() on any path."""
    _init_schema()
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ── Users ────────────────────────────────────────────────────
def create_user(username: str, hashed_password: str, role: str = "user") -> dict:
    """Insert a new user. Returns the user dict or raises on duplicate."""
    with _connect() as conn:
        try:
            # Parameterised (?) — never f-strings. The driver sends query and
            # data separately, so data can never be parsed as SQL.
            conn.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                (username, hashed_password, role),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            raise ValueError(f"Username '{username}' already exists")
    return get_user(username)


def get_user(username: str) -> dict | None:
    """Fetch a user by username. Returns dict or None."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, username, password, role, created FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        return dict(row) if row else None


def list_users() -> list[dict]:
    """List all users (admin use). Excludes password hashes."""
    with _connect() as conn:
        rows = conn.execute("SELECT id, username, role, created FROM users").fetchall()
        return [dict(r) for r in rows]


# ── Token revocation ─────────────────────────────────────────
def revoke_token(jti: str, expires_at: datetime) -> None:
    """Add a token's jti to the denylist until its natural expiry."""
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO revoked_tokens (jti, expires_at) VALUES (?, ?)",
            (jti, expires_at.astimezone(UTC).isoformat()),
        )
        # Opportunistic GC: entries are pointless once the token would have
        # expired anyway, so the table stays bounded without a cron job.
        conn.execute(
            "DELETE FROM revoked_tokens WHERE expires_at < ?", (datetime.now(UTC).isoformat(),)
        )
        conn.commit()


def is_token_revoked(jti: str) -> bool:
    if not jti:
        return False
    with _connect() as conn:
        row = conn.execute("SELECT 1 FROM revoked_tokens WHERE jti = ?", (jti,)).fetchone()
        return row is not None
