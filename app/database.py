"""
SQLite user store for RBAC auth.

WHY SQLite? It's zero-config, serverless, and ships with Python.
For a portfolio project with < 1000 users, it's the right choice.
No external database to provision. File lives alongside chroma_db.
"""

import os
import sqlite3
from pathlib import Path

DB_PATH = os.getenv("DB_PATH", "./data/devdocs.db")


def _get_conn() -> sqlite3.Connection:
    """Get a SQLite connection, creating the DB + tables if needed."""
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            username  TEXT    UNIQUE NOT NULL,
            password  TEXT    NOT NULL,
            role      TEXT    NOT NULL DEFAULT 'user',
            created   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()

    # Seed the admin account on first init.
    # WHY auto-seed? So the app is usable immediately after deploy — no manual
    # setup step. In production ADMIN_PASSWORD must be set explicitly; in dev a
    # convenience default is used. The password value is NEVER logged.
    row = conn.execute("SELECT id FROM users WHERE username = 'admin'").fetchone()
    if not row:
        import bcrypt

        admin_pw = os.getenv("ADMIN_PASSWORD")
        if not admin_pw:
            if os.getenv("APP_ENV", "development").lower() == "production":
                raise RuntimeError(
                    "ADMIN_PASSWORD must be set in production to seed the admin "
                    "account. Refusing to create an admin with a known default."
                )
            admin_pw = "admin123"  # dev convenience only — documented in .env.example
        hashed = bcrypt.hashpw(admin_pw.encode(), bcrypt.gensalt()).decode()
        conn.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            ("admin", hashed, "admin"),
        )
        conn.commit()
        print(
            "[db] Admin account created (username: admin). "
            "Set the ADMIN_PASSWORD env var to control the password."
        )

    return conn


def create_user(username: str, hashed_password: str, role: str = "user") -> dict:
    """Insert a new user. Returns the user dict or raises on duplicate."""
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (username, hashed_password, role),
        )
        conn.commit()
        return get_user(username)
    except sqlite3.IntegrityError:
        raise ValueError(f"Username '{username}' already exists")
    finally:
        conn.close()


def get_user(username: str) -> dict | None:
    """Fetch a user by username. Returns dict or None."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT id, username, password, role, created FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_users() -> list[dict]:
    """List all users (admin use). Excludes password hashes."""
    conn = _get_conn()
    try:
        rows = conn.execute("SELECT id, username, role, created FROM users").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
