"""SQLite persistence — sessions, messages, and document metadata."""

import sqlite3
from datetime import datetime

DB_PATH = "rag_history.db"


def init_db() -> None:
    con = sqlite3.connect(DB_PATH)
    con.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            name    TEXT    NOT NULL,
            mode    TEXT    NOT NULL DEFAULT 'Documents',
            created TEXT    NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            role       TEXT    NOT NULL,
            content    TEXT    NOT NULL,
            ts         TEXT    NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
        CREATE TABLE IF NOT EXISTS documents (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            filename   TEXT    NOT NULL,
            filetype   TEXT    NOT NULL,
            chunks     INTEGER,
            indexed    TEXT    NOT NULL
        );
    """)
    con.commit()
    con.close()


def new_session(name: str, mode: str = "Documents") -> int:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO sessions (name, mode, created) VALUES (?, ?, ?)",
        (name, mode, datetime.now().isoformat()),
    )
    sid = cur.lastrowid
    con.commit()
    con.close()
    return sid


def save_message(session_id: int, role: str, content: str) -> None:
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO messages (session_id, role, content, ts) VALUES (?, ?, ?, ?)",
        (session_id, role, content, datetime.now().isoformat()),
    )
    con.commit()
    con.close()


def save_document(session_id: int, filename: str, filetype: str, chunks: int) -> None:
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO documents (session_id, filename, filetype, chunks, indexed) VALUES (?, ?, ?, ?, ?)",
        (session_id, filename, filetype, chunks, datetime.now().isoformat()),
    )
    con.commit()
    con.close()


def get_sessions() -> list[dict]:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, name, mode, created FROM sessions ORDER BY id DESC"
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def get_messages(session_id: int) -> list[dict]:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT role, content, ts FROM messages WHERE session_id=? ORDER BY id",
        (session_id,),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def get_session_documents(session_id: int) -> list[dict]:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT filename, filetype, chunks, indexed FROM documents WHERE session_id=? ORDER BY id",
        (session_id,),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def delete_session(session_id: int) -> None:
    con = sqlite3.connect(DB_PATH)
    con.executescript(f"""
        DELETE FROM messages  WHERE session_id={session_id};
        DELETE FROM documents WHERE session_id={session_id};
        DELETE FROM sessions  WHERE id={session_id};
    """)
    con.commit()
    con.close()
