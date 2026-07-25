import asyncio
from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from gateway.pending_followups import (
    acknowledge_pending_followup,
    deserialize_message_event,
    enqueue_pending_followups,
    load_pending_followups,
    record_pending_followup_failure,
    serialize_message_event,
)
from tests.gateway.restart_test_helpers import make_restart_runner


def _event(text: str = "continue this work") -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.SLACK,
            chat_id="D123",
            chat_type="dm",
            user_id="U123",
            scope_id="T123",
        ),
        raw_message={"token": "must-not-persist"},
        message_id="1784950000.000100",
        reply_to_message_id="1784949999.000900",
        channel_context="Earlier context",
        metadata={"notify": True},
        timestamp=datetime(2026, 7, 25, 4, 0, 0),
    )


def test_message_event_round_trip_omits_raw_platform_payload():
    payload = serialize_message_event(_event())

    assert "raw_message" not in payload
    assert "must-not-persist" not in repr(payload)

    restored = deserialize_message_event(payload)
    assert restored.text == "continue this work"
    assert restored.message_type == MessageType.TEXT
    assert restored.source.platform == Platform.SLACK
    assert restored.source.chat_id == "D123"
    assert restored.source.scope_id == "T123"
    assert restored.message_id == "1784950000.000100"
    assert restored.reply_to_message_id == "1784949999.000900"
    assert restored.channel_context == "Earlier context"
    assert restored.metadata == {"notify": True}
    assert restored.internal is False
    assert restored.raw_message is None


def test_pending_followup_store_deduplicates_and_acknowledges(tmp_path):
    first = enqueue_pending_followups(
        [("agent:main:slack:dm:T123:D123", _event())],
        reason="shutdown-start",
        home=tmp_path,
    )
    second = enqueue_pending_followups(
        [("agent:main:slack:dm:T123:D123", _event())],
        reason="shutdown-final",
        home=tmp_path,
    )

    assert first == 1
    assert second == 0

    pending = load_pending_followups(home=tmp_path)
    assert len(pending) == 1
    assert pending[0]["reason"] == "shutdown-start"
    assert pending[0]["attempts"] == 0

    assert acknowledge_pending_followup(pending[0]["id"], home=tmp_path)
    assert load_pending_followups(home=tmp_path) == []


def test_pending_followup_blocks_after_bounded_replay_failures(tmp_path):
    enqueue_pending_followups(
        [("agent:main:slack:dm:T123:D123", _event())],
        reason="shutdown",
        home=tmp_path,
    )
    pending_id = load_pending_followups(home=tmp_path)[0]["id"]

    for _ in range(5):
        record_pending_followup_failure(
            pending_id,
            error_kind="AdapterUnavailable",
            home=tmp_path,
        )

    assert load_pending_followups(home=tmp_path) == []
    blocked = load_pending_followups(home=tmp_path, include_blocked=True)
    assert len(blocked) == 1
    assert blocked[0]["attempts"] == 5
    assert blocked[0]["blocked"] is True


@pytest.mark.asyncio
async def test_runner_snapshots_adapter_and_fifo_pending_events(tmp_path):
    runner, adapter = make_restart_runner()
    runner._queued_events = {}
    first = _event("first follow-up")
    second = _event("second follow-up")
    second.message_id = "1784950000.000200"
    session_key = "agent:main:slack:dm:T123:D123"
    adapter._pending_messages[session_key] = first
    runner._queued_events[session_key] = [second]

    persisted = await runner._persist_pending_followup_snapshot(
        "shutdown-start",
        home=tmp_path,
    )

    assert persisted == 2
    records = load_pending_followups(home=tmp_path)
    assert [record["event"]["text"] for record in records] == [
        "first follow-up",
        "second follow-up",
    ]


@pytest.mark.asyncio
async def test_runner_replays_and_acknowledges_durable_followup(tmp_path):
    runner, adapter = make_restart_runner()
    runner._startup_restore_queue = []
    runner._startup_restore_tasks = []
    runner._startup_restore_in_progress = True
    event = _event()
    event.source.platform = Platform.TELEGRAM
    event.source.scope_id = None
    event.source.guild_id = None
    session_key = "agent:main:telegram:dm:D123"

    async def schedule_replay(_event):
        adapter._session_tasks[session_key] = asyncio.create_task(asyncio.sleep(0))

    adapter.handle_message = AsyncMock(side_effect=schedule_replay)
    enqueue_pending_followups(
        [(session_key, event)],
        reason="shutdown",
        home=tmp_path,
    )

    queued = await runner._queue_durable_pending_followups_for_startup(
        home=tmp_path,
    )
    drained = await runner._drain_startup_restore_queue(
        pending_home=tmp_path,
    )

    assert queued == 1
    assert drained == 1
    adapter.handle_message.assert_awaited_once()
    replayed = adapter.handle_message.await_args.args[0]
    assert replayed.text == "continue this work"
    assert getattr(replayed, "_hermes_durable_pending_id", None)
    assert load_pending_followups(home=tmp_path) == []


@pytest.mark.asyncio
async def test_runner_keeps_durable_followup_when_adapter_never_starts_task(tmp_path):
    runner, adapter = make_restart_runner()
    runner._startup_restore_queue = []
    adapter.handle_message = AsyncMock()
    event = _event()
    event.source.platform = Platform.TELEGRAM
    event.source.scope_id = None
    event.source.guild_id = None
    session_key = "agent:main:telegram:dm:D123"
    enqueue_pending_followups(
        [(session_key, event)],
        reason="shutdown",
        home=tmp_path,
    )

    assert await runner._queue_durable_pending_followups_for_startup(home=tmp_path) == 1
    assert await runner._drain_startup_restore_queue(pending_home=tmp_path) == 0

    records = load_pending_followups(home=tmp_path)
    assert len(records) == 1
    assert records[0]["attempts"] == 1
    assert records[0]["last_error_kind"] == "RuntimeError"


def test_runner_rejects_internal_and_goal_continuation_events():
    internal = _event("internal")
    internal.internal = True
    goal = _event("[Continuing toward your standing goal]\nGoal: finish")

    assert GatewayRunner._is_durable_pending_followup(internal) is False
    assert GatewayRunner._is_durable_pending_followup(goal) is False
    assert GatewayRunner._is_durable_pending_followup(_event()) is True
