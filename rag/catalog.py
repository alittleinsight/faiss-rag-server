import json
import sqlite3
from pathlib import Path


def canonicalize_source_path(source_path: str) -> str:
    return str(Path(source_path).expanduser().resolve())


def _connect(catalog_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(catalog_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_catalog(catalog_path: Path) -> None:
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(catalog_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                source_path TEXT PRIMARY KEY,
                source_hash TEXT NOT NULL,
                mtime REAL,
                size INTEGER,
                chunk_count INTEGER NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id INTEGER PRIMARY KEY,
                source_path TEXT NOT NULL,
                chunk_order INTEGER NOT NULL,
                chunk_hash TEXT NOT NULL,
                text TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (source_path) REFERENCES documents(source_path) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        row = conn.execute("SELECT value FROM state WHERE key='next_chunk_id'").fetchone()
        if row is None:
            conn.execute("INSERT INTO state(key, value) VALUES ('next_chunk_id', '1')")


def get_document(catalog_path: Path, source_path: str) -> dict | None:
    canonical = canonicalize_source_path(source_path)
    with _connect(catalog_path) as conn:
        row = conn.execute(
            "SELECT source_path, source_hash, mtime, size, chunk_count, updated_at FROM documents WHERE source_path = ?",
            (canonical,),
        ).fetchone()
    if row is None:
        return None
    return dict(row)


def upsert_document(
    catalog_path: Path,
    source_path: str,
    source_hash: str,
    mtime: float,
    size: int,
    chunk_count: int,
) -> None:
    canonical = canonicalize_source_path(source_path)
    with _connect(catalog_path) as conn:
        conn.execute(
            """
            INSERT INTO documents(source_path, source_hash, mtime, size, chunk_count, updated_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(source_path) DO UPDATE SET
                source_hash=excluded.source_hash,
                mtime=excluded.mtime,
                size=excluded.size,
                chunk_count=excluded.chunk_count,
                updated_at=datetime('now')
            """,
            (canonical, source_hash, float(mtime), int(size), int(chunk_count)),
        )


def delete_document_and_chunks(catalog_path: Path, source_path: str) -> None:
    canonical = canonicalize_source_path(source_path)
    with _connect(catalog_path) as conn:
        conn.execute("DELETE FROM documents WHERE source_path = ?", (canonical,))


def next_chunk_ids(catalog_path: Path, count: int) -> list[int]:
    if count <= 0:
        return []

    with _connect(catalog_path) as conn:
        row = conn.execute("SELECT value FROM state WHERE key='next_chunk_id'").fetchone()
        next_id = int(row["value"] if row else "1")
        ids = list(range(next_id, next_id + count))
        conn.execute(
            "INSERT INTO state(key, value) VALUES('next_chunk_id', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(next_id + count),),
        )
    return ids


def insert_chunk_rows(catalog_path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    prepared = []
    for row in rows:
        prepared.append(
            (
                int(row["chunk_id"]),
                canonicalize_source_path(str(row["source_path"])),
                int(row["chunk_order"]),
                str(row["chunk_hash"]),
                str(row["text"]),
                json.dumps(row["metadata"], ensure_ascii=True),
            )
        )

    with _connect(catalog_path) as conn:
        conn.executemany(
            """
            INSERT INTO chunks(chunk_id, source_path, chunk_order, chunk_hash, text, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            prepared,
        )


def get_chunk_ids_for_source_path(catalog_path: Path, source_path: str) -> list[int]:
    canonical = canonicalize_source_path(source_path)
    with _connect(catalog_path) as conn:
        rows = conn.execute(
            "SELECT chunk_id FROM chunks WHERE source_path = ? ORDER BY chunk_order ASC",
            (canonical,),
        ).fetchall()
    return [int(row["chunk_id"]) for row in rows]


def get_all_chunks(catalog_path: Path) -> list[dict]:
    with _connect(catalog_path) as conn:
        rows = conn.execute(
            "SELECT chunk_id, source_path, text, metadata_json FROM chunks ORDER BY chunk_id ASC"
        ).fetchall()

    result = []
    for row in rows:
        result.append(
            {
                "chunk_id": int(row["chunk_id"]),
                "source_path": row["source_path"],
                "text": row["text"],
                "metadata": json.loads(row["metadata_json"]),
            }
        )
    return result


def list_documents(catalog_path: Path) -> list[dict]:
    with _connect(catalog_path) as conn:
        rows = conn.execute(
            "SELECT source_path, chunk_count FROM documents ORDER BY lower(source_path) ASC"
        ).fetchall()

    result = []
    for row in rows:
        source_path = str(row["source_path"])
        result.append(
            {
                "filename": Path(source_path).name,
                "source_path": source_path,
                "chunk_count": int(row["chunk_count"]),
            }
        )
    return result


def get_source_paths_for_filename(catalog_path: Path, filename: str) -> list[str]:
    normalized = filename.strip().lower()
    with _connect(catalog_path) as conn:
        rows = conn.execute("SELECT source_path FROM documents").fetchall()

    matches = []
    for row in rows:
        source_path = str(row["source_path"])
        if Path(source_path).name.lower() == normalized:
            matches.append(source_path)

    matches.sort(key=lambda item: item.lower())
    return matches
