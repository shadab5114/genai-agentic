"""SQLite trace store.

Raw traces are the durable asset here, not the metrics. Metrics get recomputed
from traces whenever the scoring code changes, which means a run captured today
can be re-scored against a rule you invent in three months.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent.parent / "runs" / "traces.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    label         TEXT,
    runner        TEXT NOT NULL,
    model         TEXT,
    server_label  TEXT,
    tools_hash    TEXT,
    tools_enabled TEXT,
    intent_set    TEXT,
    repeats       INTEGER DEFAULT 1,
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS intent_results (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id         TEXT NOT NULL,
    intent_id      TEXT NOT NULL,
    repeat_idx     INTEGER NOT NULL,
    shape          TEXT,
    stage          TEXT,
    answerable     INTEGER,
    expected_tools TEXT,
    called_tools   TEXT,
    n_calls        INTEGER,
    tokens_tools   INTEGER,
    tokens_model_in  INTEGER,
    tokens_model_out INTEGER,
    latency_ms     INTEGER,
    resolved       INTEGER,
    first_correct  INTEGER,
    any_correct    INTEGER,
    stop_short     INTEGER,
    fished         INTEGER,
    zero_call      INTEGER,
    tag            TEXT,
    final_answer   TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL,
    intent_id    TEXT NOT NULL,
    repeat_idx   INTEGER NOT NULL,
    call_idx     INTEGER NOT NULL,
    tool         TEXT NOT NULL,
    args         TEXT,
    ok           INTEGER,
    hit          INTEGER,
    latency_ms   INTEGER,
    resp_tokens  INTEGER,
    resp_excerpt TEXT,
    error        TEXT
);

CREATE TABLE IF NOT EXISTS static_audit (
    run_id     TEXT NOT NULL,
    payload    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ir_run ON intent_results(run_id);
CREATE INDEX IF NOT EXISTS idx_tc_run ON tool_calls(run_id);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def save_run(conn: sqlite3.Connection, meta: dict[str, Any]) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO runs
           (run_id, created_at, label, runner, model, server_label, tools_hash,
            tools_enabled, intent_set, repeats, notes)
           VALUES (:run_id,:created_at,:label,:runner,:model,:server_label,
                   :tools_hash,:tools_enabled,:intent_set,:repeats,:notes)""",
        meta,
    )
    conn.commit()


def save_intent_result(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    cols = ", ".join(row.keys())
    binds = ", ".join(f":{k}" for k in row)
    conn.execute(f"INSERT INTO intent_results ({cols}) VALUES ({binds})", row)


def save_tool_call(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    cols = ", ".join(row.keys())
    binds = ", ".join(f":{k}" for k in row)
    conn.execute(f"INSERT INTO tool_calls ({cols}) VALUES ({binds})", row)


def save_static_audit(conn: sqlite3.Connection, run_id: str, payload: dict) -> None:
    conn.execute("DELETE FROM static_audit WHERE run_id = ?", (run_id,))
    conn.execute(
        "INSERT INTO static_audit (run_id, payload) VALUES (?, ?)",
        (run_id, json.dumps(payload)),
    )
    conn.commit()


def load_static_audit(conn: sqlite3.Connection, run_id: str) -> dict | None:
    row = conn.execute(
        "SELECT payload FROM static_audit WHERE run_id = ?", (run_id,)
    ).fetchone()
    return json.loads(row["payload"]) if row else None


def list_runs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM runs ORDER BY datetime(created_at) DESC"
    ).fetchall()


def delete_run(conn: sqlite3.Connection, run_id: str) -> None:
    for table in ("tool_calls", "intent_results", "static_audit", "runs"):
        conn.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
    conn.commit()
