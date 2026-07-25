"""Mirror completed Codex Desktop turns into their project Slack threads.

The Slack gateway already handles Slack -> Codex by resuming the native Codex
task mapped to the Slack root. This watcher supplies the other direction by
reading Codex's durable rollout JSONL files and mirroring only user messages
and final assistant answers. Tool calls, reasoning, and progress commentary
stay local to Codex.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sqlite3
import subprocess
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlparse

from hermes_constants import get_hermes_home

from gateway.codex_slack_bridge import (
    CodexSlackMapping,
    get_relay_meta,
    get_thread_cursor,
    mapping_for_codex_thread,
    mirrored_event_exists,
    persist_mapping,
    record_mirrored_event,
    registry_path,
    set_relay_meta,
    set_thread_cursor,
)

logger = logging.getLogger(__name__)

JUSTIN_AGENT_FOOTER = "Sent by Justin's AI agent on behalf of Justin"
DEFAULT_POLL_SECONDS = 2.0
UPDATED_WATERMARK_KEY = "codex_threads_updated_at_ms"


@dataclass(frozen=True)
class CodexThread:
    thread_id: str
    rollout_path: str
    cwd: str
    title: str
    first_user_message: str
    git_origin_url: str
    updated_at_ms: int


@dataclass(frozen=True)
class SlackRoute:
    workspace_id: str
    channel_id: str
    channel_key: str
    project_path: str
    git_origin: str = ""


@dataclass
class CompletedTurn:
    turn_id: str
    user_messages: list[str] = field(default_factory=list)
    final_answer: str = ""


@dataclass
class RolloutScan:
    completed_turns: list[CompletedTurn] = field(default_factory=list)
    safe_cursor: int = 0
    line_count: int = 0
    active_turn: Optional[CompletedTurn] = None


def _load_env_value(path: Path, key: str) -> str:
    if not path.exists():
        return ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() != key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        return value
    return ""


def _slack_api_call(
    token: str,
    method: str,
    payload: dict[str, Any],
    *,
    retries: int = 5,
) -> dict[str, Any]:
    request = urllib.request.Request(
        f"https://slack.com/api/{method}",
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt + 1 < retries:
                delay = int(exc.headers.get("Retry-After", "30"))
                time.sleep(max(delay, 1))
                continue
            raise
        if data.get("ok"):
            return data
        if data.get("error") == "ratelimited" and attempt + 1 < retries:
            time.sleep(30 * (attempt + 1))
            continue
        raise RuntimeError(
            f"Slack {method} failed: {data.get('error') or 'unknown_error'}"
        )
    raise RuntimeError(f"Slack {method} retry budget exhausted")


def _event_key(
    thread_id: str,
    turn_id: str,
    role: str,
    ordinal: int,
    text: str,
) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]
    return f"codex:{thread_id}:{turn_id}:{role}:{ordinal}:{digest}"


def _is_slack_originated_user_message(text: str) -> bool:
    cleaned = str(text or "").strip()
    return (
        re.search(
            r"(?:^|\n)\[[^\]\n]*\|\s*Slack user <@[^>\n]+>\](?:\s|$)",
            cleaned,
        )
        is not None
    )


def _as_justin_message(text: str) -> str:
    cleaned = _visible_user_message(text)
    if not cleaned:
        return ""
    if cleaned.endswith(JUSTIN_AGENT_FOOTER):
        return cleaned
    return f"{cleaned}\n\n{JUSTIN_AGENT_FOOTER}"


def _as_titled_justin_message(title: str, text: str) -> str:
    cleaned = _visible_user_message(text)
    if not cleaned:
        return ""
    cleaned_title = str(title or "").strip()
    normalized_message = " ".join(cleaned.split()).casefold()
    normalized_title = " ".join(cleaned_title.split()).casefold()
    if cleaned_title and normalized_title == normalized_message:
        cleaned = f"*{cleaned_title}*"
    elif cleaned_title:
        cleaned = f"*{cleaned_title}*\n\n{cleaned}"
    return _as_justin_message(cleaned)


def _visible_user_message(text: str) -> str:
    """Remove Codex-internal cross-task metadata from a relayed prompt."""

    cleaned = str(text or "").strip()
    if not cleaned.startswith("<codex_delegation>"):
        return cleaned
    try:
        root = ET.fromstring(cleaned)
    except ET.ParseError:
        return cleaned
    if root.tag != "codex_delegation":
        return cleaned
    input_node = root.find("input")
    if input_node is None:
        return cleaned
    return "".join(input_node.itertext()).strip()


def _line_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for count, _ in enumerate(handle, start=1):
            pass
    return count


def _canonical_git_origin(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if "://" not in raw and "@" in raw:
        user_host, separator, repo_path = raw.partition(":")
        if separator and "/" in repo_path:
            raw = f"ssh://{user_host.rsplit('@', 1)[-1]}/{repo_path}"
    parsed = urlparse(raw)
    host = str(parsed.hostname or "").lower()
    path = str(parsed.path or "").strip("/").removesuffix(".git").lower()
    if host and path:
        return f"{host}/{path}"
    return raw.lower().removesuffix(".git")


def _git_origin_for_path(project_path: str) -> str:
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                project_path,
                "config",
                "--get",
                "remote.origin.url",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if result.returncode != 0:
        return ""
    return str(result.stdout or "").strip()


def scan_rollout(
    rollout_path: Path,
    start_line: int,
) -> RolloutScan:
    """Return complete turns, active-turn state, and the last safe cursor.

    An incomplete active turn is deliberately left unread at its task_started
    line so the next poll can replay it once task_complete is appended.
    """

    scan = RolloutScan(safe_cursor=max(int(start_line), 0))
    active: Optional[CompletedTurn] = None
    active_start = scan.safe_cursor

    with rollout_path.open("r", encoding="utf-8") as handle:
        for index, raw_line in enumerate(handle):
            scan.line_count = index + 1
            if index < start_line:
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                break

            event_type = str(record.get("type") or "")
            payload = record.get("payload") or {}
            payload_type = str(payload.get("type") or "")

            if active is None:
                if event_type == "event_msg" and payload_type == "task_started":
                    active_start = index
                    active = CompletedTurn(
                        turn_id=str(payload.get("turn_id") or f"line-{index}")
                    )
                    continue
                scan.safe_cursor = index + 1
                continue

            if event_type == "event_msg" and payload_type == "user_message":
                message = str(payload.get("message") or "").strip()
                if message:
                    active.user_messages.append(message)
            elif event_type == "event_msg" and payload_type == "agent_message":
                if payload.get("phase") == "final_answer":
                    active.final_answer = str(payload.get("message") or "").strip()
            elif event_type == "event_msg" and payload_type == "task_complete":
                if not active.final_answer:
                    active.final_answer = str(
                        payload.get("last_agent_message") or ""
                    ).strip()
                scan.completed_turns.append(active)
                active = None
                scan.safe_cursor = index + 1

    if active is not None:
        scan.safe_cursor = active_start
        scan.active_turn = active
    return scan


def read_completed_turns(
    rollout_path: Path,
    start_line: int,
) -> tuple[list[CompletedTurn], int]:
    scan = scan_rollout(rollout_path, start_line)
    return scan.completed_turns, scan.safe_cursor


class CodexSlackRelay:
    def __init__(
        self,
        *,
        codex_home: Optional[Path] = None,
        hermes_home: Optional[Path] = None,
        slack_call: Optional[
            Callable[[str, str, dict[str, Any]], dict[str, Any]]
        ] = None,
    ) -> None:
        self.codex_home = codex_home or Path(
            os.environ.get("CODEX_HOME") or Path.home() / ".codex"
        )
        self.hermes_home = hermes_home or get_hermes_home()
        self.codex_state_path = self.codex_home / "state_5.sqlite"
        self.hermes_state_path = self.hermes_home / "state.db"
        self.env_path = self.hermes_home / ".env"
        self._slack_call = slack_call or _slack_api_call
        self._route_cache_signature: Optional[tuple[int, int]] = None
        self._route_cache: tuple[
            dict[str, SlackRoute],
            dict[str, SlackRoute],
        ] = ({}, {})

    def _load_registry(self) -> dict[str, Any]:
        path = self.hermes_home / registry_path().name
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read Codex Slack registry: %s", path)
            return {}
        return data if isinstance(data, dict) else {}

    def _route_catalog(
        self,
    ) -> tuple[dict[str, SlackRoute], dict[str, SlackRoute]]:
        registry_file = self.hermes_home / registry_path().name
        try:
            stat = registry_file.stat()
            signature = (int(stat.st_mtime_ns), int(stat.st_size))
        except OSError:
            signature = (0, 0)
        if signature == self._route_cache_signature:
            return self._route_cache

        registry = self._load_registry()
        workspace_id = str(registry.get("workspace_id") or "").strip()
        routes: dict[str, SlackRoute] = {}

        projects = registry.get("projects") or {}
        if isinstance(projects, dict):
            for entry in projects.values():
                if not isinstance(entry, dict):
                    continue
                project_path = os.path.abspath(
                    os.path.expanduser(str(entry.get("project_path") or ""))
                )
                channel_id = str(entry.get("channel_id") or "").strip()
                if project_path and workspace_id and channel_id:
                    raw_origin = str(entry.get("git_origin_url") or "").strip()
                    if not raw_origin:
                        raw_origin = _git_origin_for_path(project_path)
                    routes[project_path] = SlackRoute(
                        workspace_id=workspace_id,
                        channel_id=channel_id,
                        channel_key=f"{workspace_id}:{channel_id}",
                        project_path=project_path,
                        git_origin=_canonical_git_origin(raw_origin),
                    )

        channels = registry.get("channels") or {}
        if isinstance(channels, dict):
            for channel_key, entry in channels.items():
                if not isinstance(entry, dict):
                    continue
                project_path = os.path.abspath(
                    os.path.expanduser(str(entry.get("project_path") or ""))
                )
                if not project_path or project_path in routes:
                    continue
                scope, separator, channel_id = str(channel_key).partition(":")
                if separator and scope and channel_id:
                    raw_origin = str(entry.get("git_origin_url") or "").strip()
                    if not raw_origin:
                        raw_origin = _git_origin_for_path(project_path)
                    routes[project_path] = SlackRoute(
                        workspace_id=scope,
                        channel_id=channel_id,
                        channel_key=str(channel_key),
                        project_path=project_path,
                        git_origin=_canonical_git_origin(raw_origin),
                    )

        origin_candidates: dict[str, list[SlackRoute]] = {}
        for route in routes.values():
            if route.git_origin:
                origin_candidates.setdefault(route.git_origin, []).append(route)
        unique_origins = {
            origin: candidates[0]
            for origin, candidates in origin_candidates.items()
            if len(candidates) == 1
        }
        self._route_cache_signature = signature
        self._route_cache = (routes, unique_origins)
        return self._route_cache

    def _route_for_thread(
        self,
        thread: CodexThread,
        catalog: tuple[dict[str, SlackRoute], dict[str, SlackRoute]],
    ) -> Optional[SlackRoute]:
        routes, origins = catalog
        normalized_cwd = os.path.abspath(os.path.expanduser(str(thread.cwd or "")))
        exact = routes.get(normalized_cwd)
        if exact is not None:
            return exact

        ancestors = [
            route
            for project_path, route in routes.items()
            if normalized_cwd.startswith(project_path + os.sep)
        ]
        if ancestors:
            return max(ancestors, key=lambda route: len(route.project_path))

        origin = _canonical_git_origin(thread.git_origin_url)
        return origins.get(origin) if origin else None

    def _threads(self, *, updated_after_ms: Optional[int] = None) -> list[CodexThread]:
        if not self.codex_state_path.exists():
            return []
        uri = f"file:{self.codex_state_path}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=10) as conn:
            conn.row_factory = sqlite3.Row
            where_updated = ""
            params: tuple[Any, ...] = ()
            if updated_after_ms is not None:
                where_updated = "AND COALESCE(updated_at_ms, updated_at * 1000) > ?"
                params = (max(int(updated_after_ms), 0),)
            rows = conn.execute(
                f"""
                SELECT
                    id,
                    rollout_path,
                    cwd,
                    COALESCE(name, title, '') AS title,
                    COALESCE(first_user_message, '') AS first_user_message,
                    COALESCE(git_origin_url, '') AS git_origin_url,
                    COALESCE(updated_at_ms, updated_at * 1000) AS updated_at_ms
                FROM threads
                WHERE archived = 0
                  AND (
                    thread_source = 'user'
                    OR (
                      NULLIF(TRIM(COALESCE(thread_source, '')), '') IS NULL
                      AND source IN ('exec', 'vscode', 'cli')
                    )
                  )
                  AND rollout_path <> ''
                  {where_updated}
                ORDER BY updated_at_ms, id
                """,
                params,
            ).fetchall()
        return [
            CodexThread(
                thread_id=str(row["id"]),
                rollout_path=str(row["rollout_path"]),
                cwd=str(row["cwd"]),
                title=str(row["title"]),
                first_user_message=str(row["first_user_message"]),
                git_origin_url=str(row["git_origin_url"]),
                updated_at_ms=int(row["updated_at_ms"] or 0),
            )
            for row in rows
        ]

    def _latest_updated_at_ms(self) -> int:
        if not self.codex_state_path.exists():
            return 0
        uri = f"file:{self.codex_state_path}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=10) as conn:
            row = conn.execute(
                """
                SELECT MAX(COALESCE(updated_at_ms, updated_at * 1000))
                FROM threads
                WHERE archived = 0
                  AND (
                    thread_source = 'user'
                    OR (
                      NULLIF(TRIM(COALESCE(thread_source, '')), '') IS NULL
                      AND source IN ('exec', 'vscode', 'cli')
                    )
                  )
                """
            ).fetchone()
        return int((row or [0])[0] or 0)

    def import_legacy_mappings(self) -> int:
        if not self.hermes_state_path.exists():
            return 0
        imported = 0
        prefix = "codex_slack_bridge:v1:slack:"
        uri = f"file:{self.hermes_state_path}?mode=ro"
        with sqlite3.connect(uri, uri=True, timeout=10) as conn:
            rows = conn.execute(
                "SELECT key, value FROM state_meta WHERE key LIKE ?",
                (prefix + "%",),
            ).fetchall()
        for key, raw_value in rows:
            try:
                payload = json.loads(raw_value or "{}")
            except json.JSONDecodeError:
                continue
            codex_thread_id = str(payload.get("codex_thread_id") or "").strip()
            suffix = str(key)[len("codex_slack_bridge:v1:") :]
            parts = suffix.split(":", 3)
            if len(parts) != 4 or parts[0] != "slack" or not codex_thread_id:
                continue
            _, workspace_id, channel_id, root_ts = parts
            persist_mapping(
                CodexSlackMapping(
                    codex_thread_id=codex_thread_id,
                    workspace_id=workspace_id,
                    channel_id=channel_id,
                    root_ts=root_ts,
                    cwd=str(payload.get("cwd") or ""),
                    channel_key=str(payload.get("channel_key") or ""),
                )
            )
            imported += 1
        return imported

    def baseline(self) -> int:
        self.import_legacy_mappings()
        count = 0
        for thread in self._threads():
            path = Path(thread.rollout_path)
            if not path.exists():
                continue
            set_thread_cursor(thread.thread_id, str(path), _line_count(path))
            count += 1
        set_relay_meta(UPDATED_WATERMARK_KEY, self._latest_updated_at_ms())
        return count

    def _post(
        self,
        channel_id: str,
        text: str,
        *,
        thread_ts: Optional[str] = None,
    ) -> str:
        token = _load_env_value(self.env_path, "SLACK_BOT_TOKEN")
        if not token:
            raise RuntimeError(f"SLACK_BOT_TOKEN is missing from {self.env_path}")
        payload: dict[str, Any] = {
            "channel": channel_id,
            "text": text,
            "mrkdwn": True,
        }
        if thread_ts:
            payload["thread_ts"] = thread_ts
        response = self._slack_call(token, "chat.postMessage", payload)
        message_ts = str(response.get("ts") or "").strip()
        if not message_ts:
            raise RuntimeError("Slack chat.postMessage returned no ts")
        return message_ts

    def _record_suppressed_turn(self, thread: CodexThread, turn: CompletedTurn) -> None:
        for ordinal, message in enumerate(turn.user_messages):
            record_mirrored_event(
                _event_key(
                    thread.thread_id,
                    turn.turn_id,
                    "user",
                    ordinal,
                    message,
                ),
                thread.thread_id,
                "slack_to_codex",
            )
        if turn.final_answer:
            record_mirrored_event(
                _event_key(
                    thread.thread_id,
                    turn.turn_id,
                    "assistant",
                    0,
                    turn.final_answer,
                ),
                thread.thread_id,
                "slack_to_codex",
            )

    def _mirror_turn(
        self,
        thread: CodexThread,
        turn: CompletedTurn,
        route: SlackRoute,
    ) -> int:
        if turn.user_messages and _is_slack_originated_user_message(
            turn.user_messages[0]
        ):
            self._record_suppressed_turn(thread, turn)
            return 0

        mapping = mapping_for_codex_thread(thread.thread_id)
        sent = 0

        for ordinal, message in enumerate(turn.user_messages):
            event_key = _event_key(
                thread.thread_id,
                turn.turn_id,
                "user",
                ordinal,
                message,
            )
            if mirrored_event_exists(event_key):
                continue
            rendered = (
                _as_titled_justin_message(thread.title, message)
                if mapping is None
                else _as_justin_message(message)
            )
            if not rendered:
                continue
            if mapping is None:
                root_ts = self._post(route.channel_id, rendered)
                mapping = CodexSlackMapping(
                    codex_thread_id=thread.thread_id,
                    workspace_id=route.workspace_id,
                    channel_id=route.channel_id,
                    root_ts=root_ts,
                    cwd=thread.cwd,
                    channel_key=route.channel_key,
                )
                persist_mapping(mapping)
                slack_ts = root_ts
            else:
                slack_ts = self._post(
                    mapping.channel_id,
                    rendered,
                    thread_ts=mapping.root_ts,
                )
            record_mirrored_event(
                event_key,
                thread.thread_id,
                "codex_to_slack",
                slack_ts,
            )
            sent += 1

        if turn.final_answer and mapping is not None:
            event_key = _event_key(
                thread.thread_id,
                turn.turn_id,
                "assistant",
                0,
                turn.final_answer,
            )
            if not mirrored_event_exists(event_key):
                slack_ts = self._post(
                    mapping.channel_id,
                    turn.final_answer,
                    thread_ts=mapping.root_ts,
                )
                record_mirrored_event(
                    event_key,
                    thread.thread_id,
                    "codex_to_slack",
                    slack_ts,
                )
                sent += 1
        return sent

    def backfill_open_threads(
        self,
        *,
        limit_per_project: Optional[int] = None,
        dry_run: bool = False,
    ) -> dict[str, int]:
        """Create one titled Slack root for each eligible open Codex chat.

        Historical turns are not replayed. A successful backfill advances the
        relay cursor to the current end of the rollout so only future turns are
        mirrored into the newly created Slack thread.
        """

        self.import_legacy_mappings()
        summary = {
            "threads": 0,
            "eligible": 0,
            "selected": 0,
            "messages_sent": 0,
            "already_mapped": 0,
            "unrouted": 0,
            "incomplete": 0,
            "missing_metadata": 0,
            "limited": 0,
            "errors": 0,
        }
        project_counts: dict[str, int] = {}
        catalog = self._route_catalog()
        threads = sorted(
            self._threads(),
            key=lambda thread: (-thread.updated_at_ms, thread.thread_id),
        )

        for thread in threads:
            summary["threads"] += 1
            if mapping_for_codex_thread(thread.thread_id) is not None:
                summary["already_mapped"] += 1
                continue

            route = self._route_for_thread(thread, catalog)
            if route is None:
                summary["unrouted"] += 1
                continue

            title = str(thread.title or "").strip()
            first_user_message = str(thread.first_user_message or "").strip()
            rollout_path = Path(thread.rollout_path)
            if not title or not first_user_message or not rollout_path.exists():
                summary["missing_metadata"] += 1
                continue

            try:
                rollout = scan_rollout(rollout_path, 0)
                if rollout.active_turn is not None:
                    summary["incomplete"] += 1
                summary["eligible"] += 1

                used = project_counts.get(route.channel_key, 0)
                if limit_per_project is not None and used >= max(
                    int(limit_per_project), 0
                ):
                    summary["limited"] += 1
                    continue
                project_counts[route.channel_key] = used + 1
                summary["selected"] += 1
                if dry_run:
                    continue

                rendered = _as_titled_justin_message(
                    title,
                    first_user_message,
                )
                root_ts = self._post(route.channel_id, rendered)
                persist_mapping(
                    CodexSlackMapping(
                        codex_thread_id=thread.thread_id,
                        workspace_id=route.workspace_id,
                        channel_id=route.channel_id,
                        root_ts=root_ts,
                        cwd=thread.cwd,
                        channel_key=route.channel_key,
                    )
                )
                record_mirrored_event(
                    _event_key(
                        thread.thread_id,
                        "open-thread-backfill",
                        "user",
                        0,
                        first_user_message,
                    ),
                    thread.thread_id,
                    "codex_to_slack_backfill",
                    root_ts,
                )
                if (
                    not rollout.completed_turns
                    and rollout.active_turn is not None
                    and rollout.active_turn.user_messages
                    and _visible_user_message(rollout.active_turn.user_messages[0])
                    == first_user_message
                ):
                    record_mirrored_event(
                        _event_key(
                            thread.thread_id,
                            rollout.active_turn.turn_id,
                            "user",
                            0,
                            rollout.active_turn.user_messages[0],
                        ),
                        thread.thread_id,
                        "codex_to_slack_backfill",
                        root_ts,
                    )
                set_thread_cursor(
                    thread.thread_id,
                    str(rollout_path),
                    rollout.safe_cursor,
                )
                summary["messages_sent"] += 1
            except Exception:
                summary["errors"] += 1
                logger.exception(
                    "Codex Slack backfill failed for task %s",
                    thread.thread_id,
                )
        return summary

    def poll_once(self) -> dict[str, int]:
        self.import_legacy_mappings()
        summary = {
            "threads": 0,
            "turns": 0,
            "messages_sent": 0,
            "errors": 0,
        }
        try:
            watermark = int(get_relay_meta(UPDATED_WATERMARK_KEY) or 0)
        except ValueError:
            watermark = 0
        threads = self._threads(updated_after_ms=watermark if watermark > 0 else None)
        catalog = self._route_catalog()
        max_updated_at_ms = watermark
        for thread in threads:
            summary["threads"] += 1
            max_updated_at_ms = max(max_updated_at_ms, thread.updated_at_ms)
            path = Path(thread.rollout_path)
            if not path.exists():
                continue
            route = self._route_for_thread(thread, catalog)
            cursor = get_thread_cursor(thread.thread_id)
            start_line = 0
            if cursor is not None and cursor[0] == str(path):
                start_line = cursor[1]
            elif cursor is not None:
                start_line = 0

            if route is None:
                if cursor is not None:
                    set_thread_cursor(thread.thread_id, str(path), _line_count(path))
                continue

            try:
                turns, safe_cursor = read_completed_turns(path, start_line)
                for turn in turns:
                    summary["turns"] += 1
                    summary["messages_sent"] += self._mirror_turn(
                        thread,
                        turn,
                        route,
                    )
                set_thread_cursor(thread.thread_id, str(path), safe_cursor)
            except Exception:
                summary["errors"] += 1
                logger.exception(
                    "Codex Slack relay failed for task %s",
                    thread.thread_id,
                )
        if summary["errors"] == 0 and max_updated_at_ms >= watermark:
            set_relay_meta(UPDATED_WATERMARK_KEY, max_updated_at_ms)
        return summary


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--backfill-open", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-per-project", type=int)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=DEFAULT_POLL_SECONDS,
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    relay = CodexSlackRelay()
    if args.baseline:
        print(json.dumps({"baselined_threads": relay.baseline()}))
        return 0
    if args.backfill_open:
        print(
            json.dumps(
                relay.backfill_open_threads(
                    limit_per_project=args.limit_per_project,
                    dry_run=args.dry_run,
                ),
                sort_keys=True,
            )
        )
        return 0
    if args.once:
        print(json.dumps(relay.poll_once(), sort_keys=True))
        return 0

    while True:
        summary = relay.poll_once()
        if summary["messages_sent"] or summary["errors"]:
            logger.info("relay pass: %s", summary)
        time.sleep(max(args.poll_seconds, 0.5))


if __name__ == "__main__":
    raise SystemExit(main())
