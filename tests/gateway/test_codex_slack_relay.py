import json
import sqlite3
from pathlib import Path

import pytest

from gateway.codex_slack_bridge import (
    CodexSlackMapping,
    get_thread_cursor,
    mapping_for_codex_thread,
    persist_mapping,
)
from gateway.codex_slack_relay import (
    JUSTIN_AGENT_FOOTER,
    CodexSlackRelay,
)


@pytest.fixture(autouse=True)
def isolated_bridge_state(tmp_path, monkeypatch):
    state = tmp_path / "bridge-state.sqlite3"
    monkeypatch.setattr(
        "gateway.codex_slack_bridge.state_path",
        lambda: state,
    )
    return state


def _write_registry(hermes_home: Path, project: Path) -> None:
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / ".env").write_text(
        "SLACK_BOT_TOKEN=test-token\n",
        encoding="utf-8",
    )
    (hermes_home / "codex_slack_bridge.json").write_text(
        json.dumps({
            "workspace_id": "T123",
            "projects": {
                "project-1": {
                    "project_path": str(project),
                    "channel_id": "C456",
                    "channel_name": "j-codex-project",
                    "git_origin_url": "git@github.com:devdocs/project.git",
                },
            },
        }),
        encoding="utf-8",
    )


def _write_codex_state(
    codex_home: Path,
    project: Path,
    rollout: Path,
    *,
    thread_id: str = "codex-thread-1",
) -> None:
    codex_home.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(codex_home / "state_5.sqlite") as conn:
        conn.execute(
            """
            CREATE TABLE threads (
                id TEXT PRIMARY KEY,
                rollout_path TEXT NOT NULL,
                cwd TEXT NOT NULL,
                name TEXT,
                title TEXT,
                archived INTEGER NOT NULL,
                thread_source TEXT,
                source TEXT,
                first_user_message TEXT,
                git_origin_url TEXT,
                created_at_ms INTEGER,
                updated_at INTEGER,
                updated_at_ms INTEGER
            )
            """
        )
        conn.execute(
            """
            INSERT INTO threads (
                id, rollout_path, cwd, name, title,
                archived, thread_source, source, first_user_message,
                git_origin_url, created_at_ms, updated_at, updated_at_ms
            ) VALUES (?, ?, ?, NULL, ?, 0, 'user', 'exec', ?,
                      'https://github.com/devdocs/project.git', 1, 1, 1)
            """,
            (
                thread_id,
                str(rollout),
                str(project),
                "Test task",
                "Build the feature",
            ),
        )


def _append_turn(
    rollout: Path,
    *,
    turn_id: str,
    user: str,
    assistant: str,
) -> None:
    records = [
        {
            "type": "event_msg",
            "payload": {"type": "task_started", "turn_id": turn_id},
        },
        {
            "type": "event_msg",
            "payload": {"type": "user_message", "message": user},
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "agent_message",
                "phase": "final_answer",
                "message": assistant,
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "task_complete",
                "turn_id": turn_id,
                "last_agent_message": assistant,
            },
        },
    ]
    with rollout.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _touch_thread(relay: CodexSlackRelay, updated_at_ms: int) -> None:
    with sqlite3.connect(relay.codex_state_path) as conn:
        conn.execute(
            """
            UPDATE threads
            SET updated_at = ?, updated_at_ms = ?
            WHERE id = 'codex-thread-1'
            """,
            (updated_at_ms // 1000, updated_at_ms),
        )


class FakeSlack:
    def __init__(self):
        self.calls = []

    def __call__(self, token, method, payload):
        self.calls.append((token, method, dict(payload)))
        return {"ok": True, "ts": f"1000.000{len(self.calls)}"}


def _relay_fixture(tmp_path):
    codex_home = tmp_path / "codex"
    hermes_home = tmp_path / "hermes"
    project = tmp_path / "project"
    project.mkdir()
    rollout = tmp_path / "rollout.jsonl"
    rollout.write_text("", encoding="utf-8")
    _write_registry(hermes_home, project)
    _write_codex_state(codex_home, project, rollout)
    slack = FakeSlack()
    relay = CodexSlackRelay(
        codex_home=codex_home,
        hermes_home=hermes_home,
        slack_call=slack,
    )
    return relay, slack, rollout, project


def test_default_hermes_home_uses_profile_aware_helper(tmp_path, monkeypatch):
    profile_home = tmp_path / "profile-home"
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.setattr(
        "gateway.codex_slack_relay.get_hermes_home",
        lambda: profile_home,
        raising=False,
    )

    relay = CodexSlackRelay(codex_home=tmp_path / "codex")

    assert relay.hermes_home == profile_home


def test_new_codex_turn_creates_slack_root_and_final_reply(tmp_path):
    relay, slack, rollout, _ = _relay_fixture(tmp_path)
    _append_turn(
        rollout,
        turn_id="turn-1",
        user="Build the feature",
        assistant="Done.",
    )
    _touch_thread(relay, 2)

    summary = relay.poll_once()

    assert summary["messages_sent"] == 2
    assert len(slack.calls) == 2
    root = slack.calls[0][2]
    reply = slack.calls[1][2]
    assert root["channel"] == "C456"
    assert "thread_ts" not in root
    assert root["text"] == (
        f"*Test task*\n\nBuild the feature\n\n{JUSTIN_AGENT_FOOTER}"
    )
    assert reply["thread_ts"] == "1000.0001"
    assert reply["text"] == "Done."
    mapping = mapping_for_codex_thread("codex-thread-1")
    assert mapping is not None
    assert mapping.root_ts == "1000.0001"


def test_new_codex_turn_renders_matching_title_and_ask_once(tmp_path):
    relay, slack, rollout, _ = _relay_fixture(tmp_path)
    with sqlite3.connect(relay.codex_state_path) as conn:
        conn.execute(
            "UPDATE threads SET title = ? WHERE id = ?",
            ("Build the feature", "codex-thread-1"),
        )
    _append_turn(
        rollout,
        turn_id="turn-1",
        user="Build the feature",
        assistant="Done.",
    )
    _touch_thread(relay, 2)

    summary = relay.poll_once()

    assert summary["messages_sent"] == 2
    assert slack.calls[0][2]["text"] == (
        f"*Build the feature*\n\n{JUSTIN_AGENT_FOOTER}"
    )
    assert slack.calls[0][2]["text"].count("Build the feature") == 1


def test_new_codex_turn_prefixes_title_that_contains_the_full_ask(tmp_path):
    relay, slack, rollout, _ = _relay_fixture(tmp_path)
    with sqlite3.connect(relay.codex_state_path) as conn:
        conn.execute(
            "UPDATE threads SET title = ? WHERE id = ?",
            ("Build the feature now", "codex-thread-1"),
        )
    _append_turn(
        rollout,
        turn_id="turn-1",
        user="Build the feature",
        assistant="Done.",
    )
    _touch_thread(relay, 2)

    relay.poll_once()

    assert slack.calls[0][2]["text"] == (
        f"*Build the feature now*\n\nBuild the feature\n\n{JUSTIN_AGENT_FOOTER}"
    )


def test_restart_and_repoll_do_not_duplicate_messages(tmp_path):
    relay, slack, rollout, _ = _relay_fixture(tmp_path)
    _append_turn(
        rollout,
        turn_id="turn-1",
        user="First",
        assistant="First answer",
    )
    _touch_thread(relay, 2)
    relay.poll_once()
    restarted = CodexSlackRelay(
        codex_home=relay.codex_home,
        hermes_home=relay.hermes_home,
        slack_call=slack,
    )

    summary = restarted.poll_once()

    assert summary["messages_sent"] == 0
    assert len(slack.calls) == 2


def test_slack_originated_turn_is_consumed_without_echo(tmp_path):
    relay, slack, rollout, project = _relay_fixture(tmp_path)
    persist_mapping(
        CodexSlackMapping(
            codex_thread_id="codex-thread-1",
            workspace_id="T123",
            channel_id="C456",
            root_ts="999.1",
            cwd=str(project),
            channel_key="T123:C456",
        )
    )
    slack_user = (
        f"[Justin | Slack user <@U123>] Continue the task\n\n{JUSTIN_AGENT_FOOTER}"
    )
    _append_turn(
        rollout,
        turn_id="turn-slack",
        user=slack_user,
        assistant="Gateway already delivered this.",
    )
    _touch_thread(relay, 2)

    summary = relay.poll_once()

    assert summary["messages_sent"] == 0
    assert slack.calls == []
    assert get_thread_cursor("codex-thread-1")[1] == 4


def test_footerless_slack_originated_turn_is_consumed_without_echo(tmp_path):
    relay, slack, rollout, project = _relay_fixture(tmp_path)
    persist_mapping(
        CodexSlackMapping(
            codex_thread_id="codex-thread-1",
            workspace_id="T123",
            channel_id="C456",
            root_ts="999.1",
            cwd=str(project),
            channel_key="T123:C456",
        )
    )
    _append_turn(
        rollout,
        turn_id="turn-slack-human",
        user="[Justin | Slack user <@U123>] Continue the task",
        assistant="Gateway already delivered this.",
    )
    _touch_thread(relay, 2)

    first = relay.poll_once()
    restarted = CodexSlackRelay(
        codex_home=relay.codex_home,
        hermes_home=relay.hermes_home,
        slack_call=slack,
    )
    second = restarted.poll_once()

    assert first["messages_sent"] == 0
    assert second["messages_sent"] == 0
    assert slack.calls == []
    assert get_thread_cursor("codex-thread-1")[1] == 4


def test_slack_originated_turn_with_thread_context_is_not_echoed(tmp_path):
    relay, slack, rollout, project = _relay_fixture(tmp_path)
    persist_mapping(
        CodexSlackMapping(
            codex_thread_id="codex-thread-1",
            workspace_id="T123",
            channel_id="C456",
            root_ts="999.1",
            cwd=str(project),
            channel_key="T123:C456",
        )
    )
    slack_user = (
        '[Replying to: "Earlier root"]\n\n'
        "[Thread context — prior messages in this thread]\n"
        "[assistant] Earlier answer\n"
        "[End of thread context]\n\n"
        "[New message]\n"
        "[Justin | Slack user <@U123>] Continue the task\n\n"
        f"{JUSTIN_AGENT_FOOTER}"
    )
    _append_turn(
        rollout,
        turn_id="turn-slack-context",
        user=slack_user,
        assistant="Gateway already delivered this too.",
    )
    _touch_thread(relay, 2)

    summary = relay.poll_once()

    assert summary["messages_sent"] == 0
    assert slack.calls == []


def test_subsequent_codex_turn_reuses_existing_slack_thread(tmp_path):
    relay, slack, rollout, project = _relay_fixture(tmp_path)
    persist_mapping(
        CodexSlackMapping(
            codex_thread_id="codex-thread-1",
            workspace_id="T123",
            channel_id="C456",
            root_ts="999.1",
            cwd=str(project),
            channel_key="T123:C456",
        )
    )
    _append_turn(
        rollout,
        turn_id="turn-2",
        user="Codex-side follow-up",
        assistant="Follow-up answer",
    )
    _touch_thread(relay, 2)

    relay.poll_once()

    assert len(slack.calls) == 2
    assert all(call[2]["thread_ts"] == "999.1" for call in slack.calls)


def test_codex_delegation_metadata_is_not_mirrored(tmp_path):
    relay, slack, rollout, _ = _relay_fixture(tmp_path)
    delegated = """
<codex_delegation>
  <source_thread_id>source-thread</source_thread_id>
  <input>Visible follow-up</input>
</codex_delegation>
""".strip()
    _append_turn(
        rollout,
        turn_id="turn-delegated",
        user=delegated,
        assistant="Delegated answer",
    )
    _touch_thread(relay, 2)

    relay.poll_once()

    assert slack.calls[0][2]["text"] == (
        f"*Test task*\n\nVisible follow-up\n\n{JUSTIN_AGENT_FOOTER}"
    )
    assert "source-thread" not in slack.calls[0][2]["text"]


def test_unmapped_project_never_cross_routes(tmp_path):
    relay, slack, rollout, _ = _relay_fixture(tmp_path)
    with sqlite3.connect(relay.codex_state_path) as conn:
        conn.execute(
            """
            UPDATE threads
            SET cwd = ?, git_origin_url = ''
            WHERE id = ?
            """,
            (str(tmp_path / "different-project"), "codex-thread-1"),
        )
    _append_turn(
        rollout,
        turn_id="turn-3",
        user="Do not cross route",
        assistant="No route.",
    )
    _touch_thread(relay, 2)

    summary = relay.poll_once()

    assert summary["messages_sent"] == 0
    assert slack.calls == []
    assert get_thread_cursor("codex-thread-1") is None


def test_backfill_open_threads_creates_one_titled_root_and_is_idempotent(
    tmp_path,
):
    relay, slack, rollout, _ = _relay_fixture(tmp_path)

    first = relay.backfill_open_threads()
    second = relay.backfill_open_threads()

    assert first == {
        "threads": 1,
        "eligible": 1,
        "selected": 1,
        "messages_sent": 1,
        "already_mapped": 0,
        "unrouted": 0,
        "incomplete": 0,
        "missing_metadata": 0,
        "limited": 0,
        "errors": 0,
    }
    assert second["messages_sent"] == 0
    assert second["already_mapped"] == 1
    assert len(slack.calls) == 1
    assert slack.calls[0][2]["text"] == (
        f"*Test task*\n\nBuild the feature\n\n{JUSTIN_AGENT_FOOTER}"
    )
    mapping = mapping_for_codex_thread("codex-thread-1")
    assert mapping is not None
    assert mapping.root_ts == "1000.0001"
    assert get_thread_cursor("codex-thread-1") == (str(rollout), 0)


def test_backfill_limit_counts_each_skipped_thread_once(tmp_path):
    relay, _, rollout, project = _relay_fixture(tmp_path)
    with sqlite3.connect(relay.codex_state_path) as conn:
        conn.execute(
            """
            INSERT INTO threads (
                id, rollout_path, cwd, name, title, archived, thread_source,
                source, first_user_message, git_origin_url,
                created_at_ms, updated_at, updated_at_ms
            ) VALUES (?, ?, ?, NULL, ?, 0, 'user', 'exec', ?,
                      'https://github.com/devdocs/project.git', 1, 2, 2)
            """,
            (
                "codex-thread-2",
                str(rollout),
                str(project),
                "Second task",
                "Build the second feature",
            ),
        )

    summary = relay.backfill_open_threads(limit_per_project=1, dry_run=True)

    assert summary["eligible"] == 2
    assert summary["selected"] == 1
    assert summary["limited"] == 1


def test_backfill_active_first_turn_does_not_duplicate_the_initial_ask(
    tmp_path,
):
    relay, slack, rollout, _ = _relay_fixture(tmp_path)
    rollout.write_text(
        "\n".join([
            json.dumps({
                "type": "event_msg",
                "payload": {"type": "task_started", "turn_id": "active-turn"},
            }),
            json.dumps({
                "type": "event_msg",
                "payload": {
                    "type": "user_message",
                    "message": "Build the feature",
                },
            }),
        ])
        + "\n",
        encoding="utf-8",
    )

    summary = relay.backfill_open_threads()

    assert summary["incomplete"] == 1
    assert summary["messages_sent"] == 1
    assert get_thread_cursor("codex-thread-1") == (str(rollout), 0)

    with rollout.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps({
                "type": "event_msg",
                "payload": {
                    "type": "agent_message",
                    "phase": "final_answer",
                    "message": "Finished.",
                },
            })
            + "\n"
        )
        handle.write(
            json.dumps({
                "type": "event_msg",
                "payload": {
                    "type": "task_complete",
                    "turn_id": "active-turn",
                    "last_agent_message": "Finished.",
                },
            })
            + "\n"
        )
    _touch_thread(relay, 2)

    relay.poll_once()

    assert len(slack.calls) == 2
    assert slack.calls[1][2]["thread_ts"] == "1000.0001"
    assert slack.calls[1][2]["text"] == "Finished."


def test_legacy_exec_threads_are_included_but_legacy_subagents_are_not(
    tmp_path,
):
    relay, _, rollout, project = _relay_fixture(tmp_path)
    with sqlite3.connect(relay.codex_state_path) as conn:
        conn.execute(
            """
            UPDATE threads
            SET thread_source = NULL, source = 'exec'
            WHERE id = 'codex-thread-1'
            """
        )
        conn.execute(
            """
            INSERT INTO threads (
                id, rollout_path, cwd, name, title, archived, thread_source,
                source, first_user_message, git_origin_url,
                created_at_ms, updated_at, updated_at_ms
            ) VALUES (?, ?, ?, NULL, ?, 0, NULL, ?, ?,
                      'https://github.com/devdocs/project.git', 1, 1, 1)
            """,
            (
                "legacy-subagent",
                str(rollout),
                str(project),
                "Subagent",
                '{"subagent":"memory_consolidation"}',
                "Do not relay",
            ),
        )

    threads = relay._threads()

    assert [thread.thread_id for thread in threads] == ["codex-thread-1"]


def test_route_resolution_supports_nested_workspaces_and_git_origins(tmp_path):
    relay, slack, rollout, project = _relay_fixture(tmp_path)
    nested = project / "worktrees" / "feature"
    nested.mkdir(parents=True)
    with sqlite3.connect(relay.codex_state_path) as conn:
        conn.execute(
            "UPDATE threads SET cwd = ? WHERE id = 'codex-thread-1'",
            (str(nested),),
        )

    nested_summary = relay.backfill_open_threads()

    assert nested_summary["messages_sent"] == 1
    assert slack.calls[0][2]["channel"] == "C456"

    with sqlite3.connect(relay.codex_state_path) as conn:
        conn.execute(
            """
            INSERT INTO threads (
                id, rollout_path, cwd, name, title, archived, thread_source,
                source, first_user_message, git_origin_url,
                created_at_ms, updated_at, updated_at_ms
            ) VALUES (?, ?, ?, NULL, ?, 0, 'user', 'exec', ?,
                      'ssh://git@github.com/devdocs/project.git', 1, 2, 2)
            """,
            (
                "origin-thread",
                str(rollout),
                str(tmp_path / "external-clone"),
                "Origin task",
                "Use the origin route",
            ),
        )

    origin_summary = relay.backfill_open_threads()

    assert origin_summary["messages_sent"] == 1
    assert slack.calls[1][2]["channel"] == "C456"
