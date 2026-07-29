"""Regression coverage for gateway-owned max-iteration continuation."""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.session import SessionSource


class CaptureAsyncSessionStore:
    def __init__(self, store=None):
        self.resume_marks = []
        # GatewayRunner.async_session_store is a read-only property that
        # rebuilds the facade whenever ``facade._store is not
        # self.session_store``. Carrying the same object here is what lets a
        # fake survive that identity check — without it the property silently
        # replaces this capture with a real AsyncSessionStore and the test
        # asserts against the wrong object.
        self._store = store

    async def mark_resume_pending(self, session_key, reason="restart_timeout"):
        self.resume_marks.append((session_key, reason))
        return True


class CaptureSlackAdapter(BasePlatformAdapter):
    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.SLACK)
        self.sent: list[dict] = []

    async def connect(self) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        self.sent.append(
            {
                "chat_id": chat_id,
                "content": content,
                "reply_to": reply_to,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id=f"sent-{len(self.sent)}")

    async def send_typing(self, chat_id, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


class BudgetThenCompleteAgent:
    calls: list[object] = []

    def __init__(self, **kwargs):
        self.tools = []
        self.tool_progress_callback = kwargs.get("tool_progress_callback")

    def run_conversation(self, message, conversation_history=None, task_id=None):
        type(self).calls.append(message)
        if len(type(self).calls) == 1:
            return {
                "final_response": "checkpoint summary",
                "messages": [],
                "api_calls": 90,
                "completed": False,
                "failed": False,
                "turn_exit_reason": "max_iterations_reached(90/90)",
            }
        return {
            "final_response": "fully completed result",
            "messages": [],
            "api_calls": 2,
            "completed": True,
            "failed": False,
            "turn_exit_reason": "completed",
        }


def _make_runner(adapter):
    gateway_run = importlib.import_module("gateway.run")
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.adapters = {adapter.platform: adapter}
    runner._voice_mode = {}
    runner._prefill_messages = []
    runner._ephemeral_system_prompt = ""
    runner._reasoning_config = None
    runner._provider_routing = {}
    runner._fallback_model = None
    runner._session_db = None
    # ``async_session_store`` became a read-only property, so the old direct
    # assignment raised AttributeError and this whole file had been failing —
    # i.e. the gateway's max-iteration continuation was shipping unverified.
    # Inject through the backing field the property actually reads.
    runner.session_store = SimpleNamespace()
    runner._async_session_store = CaptureAsyncSessionStore(runner.session_store)
    runner._running_agents = {}
    runner._session_run_generation = {}
    runner._queued_events = {}
    runner.hooks = SimpleNamespace(loaded_hooks=False)
    runner.config = SimpleNamespace(
        thread_sessions_per_user=False,
        group_sessions_per_user=False,
        stt_enabled=False,
    )
    runner._model = "openai/gpt-4.1-mini"
    runner._base_url = None
    return runner


@pytest.mark.asyncio
async def test_max_iteration_checkpoint_auto_continues_in_originating_slack_thread(
    monkeypatch,
    tmp_path,
):
    BudgetThenCompleteAgent.calls = []

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = BudgetThenCompleteAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    adapter = CaptureSlackAdapter()
    runner = _make_runner(adapter)
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="D0B8Z2EV9V2",
        chat_type="dm",
        thread_id="1785100271.344399",
    )

    result = await runner._run_agent(
        message="repair the system completely",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-max-iteration-continuation",
        session_key="agent:main:slack:dm:D0B8Z2EV9V2:1785100271.344399",
    )

    assert result["final_response"] == "fully completed result"
    assert runner.async_session_store.resume_marks == [
        (
            "agent:main:slack:dm:D0B8Z2EV9V2:1785100271.344399",
            "max_iterations",
        )
    ]
    assert len(BudgetThenCompleteAgent.calls) == 2
    continuation = BudgetThenCompleteAgent.calls[1]
    assert isinstance(continuation, str)
    assert "AUTOMATIC CONTINUATION" in continuation
    assert "unresolved task" in continuation

    # The checkpoint must be delivered before the recursive continuation, and
    # must retain the same Slack source/thread metadata.
    assert adapter.sent
    assert adapter.sent[0]["chat_id"] == "D0B8Z2EV9V2"
    assert adapter.sent[0]["content"] == "checkpoint summary"
    metadata = adapter.sent[0]["metadata"] or {}
    assert metadata.get("thread_id") == "1785100271.344399"


@pytest.mark.asyncio
async def test_auto_continuation_is_bounded(monkeypatch, tmp_path):
    BudgetThenCompleteAgent.calls = []

    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    fake_run_agent = types.ModuleType("run_agent")
    fake_run_agent.AIAgent = BudgetThenCompleteAgent
    monkeypatch.setitem(sys.modules, "run_agent", fake_run_agent)

    gateway_run = importlib.import_module("gateway.run")
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    adapter = CaptureSlackAdapter()
    runner = _make_runner(adapter)
    source = SessionSource(
        platform=Platform.SLACK,
        chat_id="D0B8Z2EV9V2",
        chat_type="dm",
        thread_id="1785100271.344399",
    )

    result = await runner._run_agent(
        message="repair the system completely",
        context_prompt="",
        history=[],
        source=source,
        session_id="sess-max-iteration-continuation",
        session_key="agent:main:slack:dm:D0B8Z2EV9V2:1785100271.344399",
        _interrupt_depth=3,
    )

    assert result["final_response"] == "checkpoint summary"
    assert len(BudgetThenCompleteAgent.calls) == 1
