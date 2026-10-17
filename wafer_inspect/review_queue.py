"""SQLite-backed queue of wafers the model was not confident about."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from wafer_inspect import CLASSES

COLUMNS = [
    "id",
    "created_at",
    "request_id",
    "predicted_label",
    "confidence",
    "uncertainty",
    "status",
    "true_label",
]


class ReviewQueue:
    def __init__(self, path: str) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    request_id TEXT,
                    predicted_label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    uncertainty REAL NOT NULL,
                    probabilities TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    true_label TEXT,
                    image BLOB NOT NULL)"""
            )
            self._db.commit()

    def add(self, request_id: str, pred: Any, image_png: bytes) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO items (created_at, request_id, predicted_label, confidence, uncertainty,"
                " probabilities, image) VALUES (?,?,?,?,?,?,?)",
                (
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    request_id,
                    pred.label,
                    pred.confidence,
                    pred.uncertainty,
                    json.dumps(pred.probabilities),
                    image_png,
                ),
            )
            self._db.commit()
            return int(cur.lastrowid or 0)

    def list(self, status: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        sql = f"SELECT {', '.join(COLUMNS)} FROM items"
        args: List[Any] = []
        if status:
            sql += " WHERE status = ?"
            args.append(status)
        sql += " ORDER BY uncertainty DESC, id ASC LIMIT ?"
        args.append(limit)
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args)]

    def image(self, item_id: int) -> Optional[bytes]:
        with self._lock:
            row = self._db.execute("SELECT image FROM items WHERE id = ?", (item_id,)).fetchone()
        return bytes(row["image"]) if row else None

    def label(self, item_id: int, true_label: str) -> bool:
        if true_label not in CLASSES:
            raise ValueError(f"label must be one of {CLASSES}")
        with self._lock:
            cur = self._db.execute(
                "UPDATE items SET status='labelled', true_label=? WHERE id=?", (true_label, item_id)
            )
            self._db.commit()
            return cur.rowcount > 0

    def stats(self) -> Dict[str, int]:
        with self._lock:
            rows = self._db.execute("SELECT status, COUNT(*) AS n FROM items GROUP BY status").fetchall()
        out = {"pending": 0, "labelled": 0}
        out.update({r["status"]: r["n"] for r in rows})
        return out

    def export_csv(self, status: Optional[str] = None) -> str:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=COLUMNS + ["image_url"])
        writer.writeheader()
        for row in self.list(status, limit=1_000_000):
            writer.writerow({**row, "image_url": f"/queue/{row['id']}/image"})
        return buf.getvalue()
