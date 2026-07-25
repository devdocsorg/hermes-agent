"""Durable Slack-thread to native Codex-thread bridge.

The user-maintained registry lives outside the Hermes checkout at
``$HERMES_HOME/codex_slack_bridge.json`` so routine Hermes updates do not
overwrite channel-to-project bindings. Native Codex thread ids are stored in
Hermes' existing state_meta table, keyed by the stable Slack workspace,
channel, and root-thread identity.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

REGISTRY_FILENAME = "codex_slack_bridge.json"
STATE_FILENAME = "codex_slack_bridge_state.sqlite3"
META_PREFIX = "codex_slack_bridge:v1:"


@dataclass(frozen=True)
class SlackCodexBinding:
    """Resolved project and durable native-thread identity for one Slack thread."""

    key: str
    cwd: str
    channel_key: str
    codex_thread_id: Optional[str] = None


@dataclass(frozen=True)
class CodexSlackMapping:
    """Durable native Codex task to Slack thread mapping."""

    codex_thread_id: str
    workspace_id: str
    channel_id: str
    root_ts: str
    cwd: str
    channel_key: str


def registry_path() -> Path:
    return get_hermes_home() / REGISTRY_FILENAME


def state_path() -> Path:
    return get_hermes_home() / STATE_FILENAME


def _connect_state() -> sqlite3.Connection:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 10000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS thread_bindings (
            codex_thread_id TEXT PRIMARY KEY,
            workspace_id TEXT NOT NULL,
            channel_id TEXT NOT NULL,
            root_ts TEXT NOT NULL,
            cwd TEXT NOT NULL,
            channel_key TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            UNIQUE(workspace_id, channel_id, root_ts)
        );
        CREATE INDEX IF NOT EXISTS idx_thread_bindings_slack
            ON thread_bindings(workspace_id, channel_id, root_ts);

        CREATE TABLE IF NOT EXISTS mirrored_events (
            event_key TEXT PRIMARY KEY,
            codex_thread_id TEXT NOT NULL,
            direction TEXT NOT NULL,
            slack_ts TEXT,
            created_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS thread_cursors (
            codex_thread_id TEXT PRIMARY KEY,
            rollout_path TEXT NOT NULL,
            line_number INTEGER NOT NULL DEFAULT 0,
            updated_at REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS relay_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return conn


def _mapping_from_row(row: Any) -> Optional[CodexSlackMapping]:
    if row is None:
        return None
    return CodexSlackMapping(
        codex_thread_id=str(row["codex_thread_id"]),
        workspace_id=str(row["workspace_id"]),
        channel_id=str(row["channel_id"]),
        root_ts=str(row["root_ts"]),
        cwd=str(row["cwd"]),
        channel_key=str(row["channel_key"]),
    )


def mapping_for_codex_thread(codex_thread_id: str) -> Optional[CodexSlackMapping]:
    thread_id = str(codex_thread_id or "").strip()
    if not thread_id:
        return None
    with _connect_state() as conn:
        row = conn.execute(
            "SELECT * FROM thread_bindings WHERE codex_thread_id = ?",
            (thread_id,),
        ).fetchone()
    return _mapping_from_row(row)


def mapping_for_slack_thread(
    workspace_id: str,
    channel_id: str,
    root_ts: str,
) -> Optional[CodexSlackMapping]:
    values = tuple(
        str(value or "").strip()
        for value in (
            workspace_id,
            channel_id,
            root_ts,
        )
    )
    if not all(values):
        return None
    with _connect_state() as conn:
        row = conn.execute(
            """
            SELECT *
            FROM thread_bindings
            WHERE workspace_id = ? AND channel_id = ? AND root_ts = ?
            """,
            values,
        ).fetchone()
    return _mapping_from_row(row)


def persist_mapping(mapping: CodexSlackMapping) -> None:
    """Upsert one task/thread pair for gateway and watcher processes."""

    now = time.time()
    with _connect_state() as conn:
        conn.execute(
            """
            DELETE FROM thread_bindings
            WHERE workspace_id = ? AND channel_id = ? AND root_ts = ?
              AND codex_thread_id <> ?
            """,
            (
                mapping.workspace_id,
                mapping.channel_id,
                mapping.root_ts,
                mapping.codex_thread_id,
            ),
        )
        conn.execute(
            """
            INSERT INTO thread_bindings (
                codex_thread_id, workspace_id, channel_id, root_ts,
                cwd, channel_key, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(codex_thread_id) DO UPDATE SET
                workspace_id = excluded.workspace_id,
                channel_id = excluded.channel_id,
                root_ts = excluded.root_ts,
                cwd = excluded.cwd,
                channel_key = excluded.channel_key,
                updated_at = excluded.updated_at
            """,
            (
                mapping.codex_thread_id,
                mapping.workspace_id,
                mapping.channel_id,
                mapping.root_ts,
                mapping.cwd,
                mapping.channel_key,
                now,
                now,
            ),
        )


def mirrored_event_exists(event_key: str) -> bool:
    key = str(event_key or "").strip()
    if not key:
        return False
    with _connect_state() as conn:
        return (
            conn.execute(
                "SELECT 1 FROM mirrored_events WHERE event_key = ?",
                (key,),
            ).fetchone()
            is not None
        )


def record_mirrored_event(
    event_key: str,
    codex_thread_id: str,
    direction: str,
    slack_ts: Optional[str] = None,
) -> None:
    key = str(event_key or "").strip()
    thread_id = str(codex_thread_id or "").strip()
    if not key or not thread_id:
        return
    with _connect_state() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO mirrored_events (
                event_key, codex_thread_id, direction, slack_ts, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                key,
                thread_id,
                str(direction or "unknown"),
                str(slack_ts or "") or None,
                time.time(),
            ),
        )


def get_thread_cursor(codex_thread_id: str) -> Optional[tuple[str, int]]:
    thread_id = str(codex_thread_id or "").strip()
    if not thread_id:
        return None
    with _connect_state() as conn:
        row = conn.execute(
            """
            SELECT rollout_path, line_number
            FROM thread_cursors
            WHERE codex_thread_id = ?
            """,
            (thread_id,),
        ).fetchone()
    if row is None:
        return None
    return str(row["rollout_path"]), int(row["line_number"])


def set_thread_cursor(
    codex_thread_id: str,
    rollout_path_value: str,
    line_number: int,
) -> None:
    thread_id = str(codex_thread_id or "").strip()
    rollout = str(rollout_path_value or "").strip()
    if not thread_id or not rollout:
        return
    with _connect_state() as conn:
        conn.execute(
            """
            INSERT INTO thread_cursors (
                codex_thread_id, rollout_path, line_number, updated_at
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(codex_thread_id) DO UPDATE SET
                rollout_path = excluded.rollout_path,
                line_number = excluded.line_number,
                updated_at = excluded.updated_at
            """,
            (thread_id, rollout, max(int(line_number), 0), time.time()),
        )


def get_relay_meta(key: str) -> Optional[str]:
    cleaned = str(key or "").strip()
    if not cleaned:
        return None
    with _connect_state() as conn:
        row = conn.execute(
            "SELECT value FROM relay_meta WHERE key = ?",
            (cleaned,),
        ).fetchone()
    return str(row["value"]) if row is not None else None


def set_relay_meta(key: str, value: Any) -> None:
    cleaned = str(key or "").strip()
    if not cleaned:
        return
    with _connect_state() as conn:
        conn.execute(
            """
            INSERT INTO relay_meta (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (cleaned, str(value)),
        )


def _load_registry() -> dict[str, Any]:
    path = registry_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Could not read Codex Slack bridge registry: %s", path)
        return {}
    return data if isinstance(data, dict) else {}


def _project_path_for_source(source: Any) -> tuple[Optional[str], Optional[str]]:
    registry = _load_registry()
    channels = registry.get("channels") or {}
    if not isinstance(channels, dict):
        channels = {}

    scope_id = str(getattr(source, "scope_id", "") or "")
    chat_id = str(getattr(source, "chat_id", "") or "")
    candidates = [f"{scope_id}:{chat_id}"] if scope_id and chat_id else []
    if chat_id:
        candidates.append(chat_id)

    entry: Any = None
    channel_key: Optional[str] = None
    for candidate in candidates:
        if candidate in channels:
            entry = channels[candidate]
            channel_key = candidate
            break
    if entry is None:
        entry = registry.get("default_project_path")
        channel_key = "default"

    if isinstance(entry, dict):
        entry = entry.get("project_path")
    if not isinstance(entry, str) or not entry.strip():
        return None, None

    cwd = os.path.abspath(os.path.expanduser(entry.strip()))
    if not os.path.isdir(cwd):
        logger.warning(
            "Codex Slack bridge ignored unavailable project path %s for %s",
            cwd,
            channel_key,
        )
        return None, None
    return cwd, channel_key


def resolve_binding(source: Any, session_db: Any) -> Optional[SlackCodexBinding]:
    """Resolve an enabled Slack thread to its project and native Codex thread."""

    platform = getattr(getattr(source, "platform", None), "value", "")
    if platform != "slack":
        return None

    thread_id = str(getattr(source, "thread_id", "") or "").strip()
    chat_id = str(getattr(source, "chat_id", "") or "").strip()
    if not thread_id or not chat_id:
        return None

    cwd, channel_key = _project_path_for_source(source)
    if not cwd or not channel_key:
        return None

    scope_id = str(getattr(source, "scope_id", "") or "").strip()
    key = ":".join(("slack", scope_id or "default", chat_id, thread_id))
    stored_thread_id = None
    if session_db is not None:
        try:
            raw = session_db.get_meta(META_PREFIX + key)
            data = json.loads(raw) if raw else {}
            candidate = data.get("codex_thread_id") if isinstance(data, dict) else None
            if isinstance(candidate, str) and candidate.strip():
                stored_thread_id = candidate.strip()
        except Exception:
            logger.debug(
                "Could not load Codex Slack bridge state for %s", key, exc_info=True
            )
    if not stored_thread_id:
        try:
            mapping = mapping_for_slack_thread(scope_id, chat_id, thread_id)
            if mapping is not None:
                stored_thread_id = mapping.codex_thread_id
        except Exception:
            logger.debug(
                "Could not load shared Codex Slack mapping for %s",
                key,
                exc_info=True,
            )

    return SlackCodexBinding(
        key=key,
        cwd=cwd,
        channel_key=channel_key,
        codex_thread_id=stored_thread_id,
    )


def persist_codex_thread(
    session_db: Any,
    binding: Optional[SlackCodexBinding],
    codex_thread_id: Any,
) -> None:
    """Persist a native Codex id after a successful app-server turn."""

    if session_db is None or binding is None:
        return
    thread_id = str(codex_thread_id or "").strip()
    if not thread_id:
        return
    value = json.dumps(
        {
            "codex_thread_id": thread_id,
            "cwd": binding.cwd,
            "channel_key": binding.channel_key,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    try:
        session_db.set_meta(META_PREFIX + binding.key, value)
    except Exception:
        logger.debug(
            "Could not persist Codex Slack bridge state for %s",
            binding.key,
            exc_info=True,
        )

    parts = binding.key.split(":", 3)
    if len(parts) != 4 or parts[0] != "slack":
        return
    _, workspace_id, channel_id, root_ts = parts
    try:
        persist_mapping(
            CodexSlackMapping(
                codex_thread_id=thread_id,
                workspace_id=workspace_id,
                channel_id=channel_id,
                root_ts=root_ts,
                cwd=binding.cwd,
                channel_key=binding.channel_key,
            )
        )
    except Exception:
        logger.debug(
            "Could not persist shared Codex Slack mapping for %s",
            binding.key,
            exc_info=True,
        )
