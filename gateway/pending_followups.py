"""Durable replay journal for user follow-ups queued during gateway shutdown.

The normal adapter queues are process-local. A restart can interrupt the active
turn after a follow-up has already been acknowledged by the chat platform, so
the follow-up must be journaled before teardown and replayed on the next boot.

Only normalized ``MessageEvent`` fields are persisted. Raw platform payloads
are deliberately excluded because they can contain credentials, non-JSON
objects, and transport-specific state that is unsafe to replay.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from hermes_constants import get_hermes_home

SCHEMA_VERSION = 1
MAX_REPLAY_ATTEMPTS = 5
_RELATIVE_PATH = ("state", "pending-followups.json")
_LOCK = threading.RLock()


def get_pending_followups_path(home: Optional[Path] = None) -> Path:
    base = home if home is not None else get_hermes_home()
    return Path(base).joinpath(*_RELATIVE_PATH)


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(child, depth=depth + 1)
            for key, child in value.items()
            if isinstance(key, (str, int, float, bool))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(child, depth=depth + 1) for child in value]
    return None


def serialize_message_event(event: Any) -> dict[str, Any]:
    source = getattr(event, "source", None)
    if source is None or not hasattr(source, "to_dict"):
        raise ValueError("pending follow-up is missing a serializable source")

    message_type = getattr(getattr(event, "message_type", None), "value", None)
    timestamp = getattr(event, "timestamp", None)
    if isinstance(timestamp, datetime):
        timestamp_value = timestamp.isoformat()
    else:
        timestamp_value = datetime.now(timezone.utc).isoformat()

    return {
        "text": str(getattr(event, "text", "") or ""),
        "message_type": str(message_type or "text"),
        "source": _json_safe(source.to_dict()),
        "message_id": getattr(event, "message_id", None),
        "platform_update_id": getattr(event, "platform_update_id", None),
        "media_urls": list(getattr(event, "media_urls", None) or []),
        "media_types": list(getattr(event, "media_types", None) or []),
        "reply_to_message_id": getattr(event, "reply_to_message_id", None),
        "reply_to_text": getattr(event, "reply_to_text", None),
        "reply_to_author_id": getattr(event, "reply_to_author_id", None),
        "reply_to_author_name": getattr(event, "reply_to_author_name", None),
        "reply_to_is_own_message": bool(
            getattr(event, "reply_to_is_own_message", False)
        ),
        "auto_skill": _json_safe(getattr(event, "auto_skill", None)),
        "channel_prompt": getattr(event, "channel_prompt", None),
        "channel_context": getattr(event, "channel_context", None),
        "metadata": _json_safe(getattr(event, "metadata", None) or {}),
        "timestamp": timestamp_value,
    }


def deserialize_message_event(payload: Mapping[str, Any]) -> Any:
    from gateway.platforms.base import MessageEvent, MessageType
    from gateway.session import SessionSource

    raw_message_type = str(payload.get("message_type") or "text")
    try:
        message_type = MessageType(raw_message_type)
    except ValueError:
        message_type = MessageType.TEXT

    raw_timestamp = payload.get("timestamp")
    try:
        timestamp = datetime.fromisoformat(str(raw_timestamp))
    except (TypeError, ValueError):
        timestamp = datetime.now()

    source_payload = payload.get("source")
    if not isinstance(source_payload, Mapping):
        raise ValueError("pending follow-up source is invalid")

    return MessageEvent(
        text=str(payload.get("text") or ""),
        message_type=message_type,
        source=SessionSource.from_dict(dict(source_payload)),
        raw_message=None,
        message_id=payload.get("message_id"),
        platform_update_id=payload.get("platform_update_id"),
        media_urls=list(payload.get("media_urls") or []),
        media_types=list(payload.get("media_types") or []),
        reply_to_message_id=payload.get("reply_to_message_id"),
        reply_to_text=payload.get("reply_to_text"),
        reply_to_author_id=payload.get("reply_to_author_id"),
        reply_to_author_name=payload.get("reply_to_author_name"),
        reply_to_is_own_message=bool(payload.get("reply_to_is_own_message", False)),
        auto_skill=payload.get("auto_skill"),
        channel_prompt=payload.get("channel_prompt"),
        channel_context=payload.get("channel_context"),
        internal=False,
        metadata=dict(payload.get("metadata") or {}),
        timestamp=timestamp,
    )


def _empty_store() -> dict[str, Any]:
    return {"version": SCHEMA_VERSION, "events": []}


def _read_store(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _empty_store()
    if not isinstance(payload, dict) or payload.get("version") != SCHEMA_VERSION:
        raise ValueError("pending follow-up journal has an unsupported schema")
    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError("pending follow-up journal events are invalid")
    return {"version": SCHEMA_VERSION, "events": events}


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    data = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
        try:
            path.chmod(0o600)
        except OSError:
            pass
    finally:
        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass


def _event_id(session_key: str, event_payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        {"session_key": session_key, "event": event_payload},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def enqueue_pending_followups(
    entries: Iterable[tuple[str, Any]],
    *,
    reason: str,
    home: Optional[Path] = None,
) -> int:
    path = get_pending_followups_path(home)
    now = datetime.now(timezone.utc).isoformat()
    with _LOCK:
        store = _read_store(path)
        known = {
            str(record.get("id"))
            for record in store["events"]
            if isinstance(record, Mapping) and record.get("id")
        }
        added = 0
        for session_key, event in entries:
            event_payload = serialize_message_event(event)
            record_id = _event_id(str(session_key), event_payload)
            if record_id in known:
                continue
            store["events"].append({
                "id": record_id,
                "session_key": str(session_key),
                "reason": str(reason),
                "created_at": now,
                "attempts": 0,
                "blocked": False,
                "last_error_kind": None,
                "event": event_payload,
            })
            known.add(record_id)
            added += 1
        if added:
            _atomic_write(path, store)
        return added


def load_pending_followups(
    *,
    home: Optional[Path] = None,
    include_blocked: bool = False,
) -> list[dict[str, Any]]:
    path = get_pending_followups_path(home)
    with _LOCK:
        store = _read_store(path)
    records = []
    for record in store["events"]:
        if not isinstance(record, dict):
            continue
        if record.get("blocked") and not include_blocked:
            continue
        records.append(dict(record))
    return records


def acknowledge_pending_followup(
    record_id: str,
    *,
    home: Optional[Path] = None,
) -> bool:
    path = get_pending_followups_path(home)
    with _LOCK:
        store = _read_store(path)
        original_count = len(store["events"])
        store["events"] = [
            record
            for record in store["events"]
            if not isinstance(record, Mapping) or record.get("id") != record_id
        ]
        changed = len(store["events"]) != original_count
        if changed:
            _atomic_write(path, store)
        return changed


def record_pending_followup_failure(
    record_id: str,
    *,
    error_kind: str,
    home: Optional[Path] = None,
) -> bool:
    path = get_pending_followups_path(home)
    with _LOCK:
        store = _read_store(path)
        changed = False
        for record in store["events"]:
            if not isinstance(record, dict) or record.get("id") != record_id:
                continue
            attempts = int(record.get("attempts") or 0) + 1
            record["attempts"] = attempts
            record["blocked"] = attempts >= MAX_REPLAY_ATTEMPTS
            record["last_error_kind"] = str(error_kind)[:120]
            record["last_attempt_at"] = datetime.now(timezone.utc).isoformat()
            changed = True
            break
        if changed:
            _atomic_write(path, store)
        return changed
