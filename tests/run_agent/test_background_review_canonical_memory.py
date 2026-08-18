"""Canonical DevDocs Memory behavior for the background self-improvement loop."""

from __future__ import annotations

import json
import logging
import types

from agent.background_review_memory import (
    CANONICAL_MEMORY_CAPTURE_TOOL,
    CANONICAL_MEMORY_SEARCH_TOOL,
    apply_canonical_memory_request_policy,
    auto_memory_block_reason,
    background_review_memory_allowed_tools,
    canonical_memory_review_policy,
    canonical_memory_tools_available,
    dispatch_canonical_memory_tool,
    redirect_local_memory_call,
)


def _mcp_result(label, payload):
    return json.dumps({"result": f"{label}\n\n{json.dumps(payload)}"})


def test_foreground_canonical_memory_arguments_are_unchanged():
    args = {
        "body": "User prefers concise engineering updates.",
        "ownership": "organization",
        "organization_id": "org_123",
    }
    assert apply_canonical_memory_request_policy(CANONICAL_MEMORY_CAPTURE_TOOL, args) == args


def test_background_capture_forces_personal_scope_and_deterministic_id():
    args = {
        "body": "  User prefers concise   engineering updates. ",
        "memory_id": "7a59bfda-8544-4476-96ee-bf80f8284583",
        "ownership": "organization",
        "organization_id": "org_123",
        "team_id": "team_123",
        "tags": ["preference"],
    }
    with canonical_memory_review_policy(True):
        first = apply_canonical_memory_request_policy(CANONICAL_MEMORY_CAPTURE_TOOL, args)
        second = apply_canonical_memory_request_policy(
            CANONICAL_MEMORY_CAPTURE_TOOL,
            {"body": "User prefers concise engineering updates."},
        )

    assert first["ownership"] == "personal"
    assert "organization_id" not in first
    assert "team_id" not in first
    assert first["body"] == "User prefers concise engineering updates."
    assert first["memory_id"] == second["memory_id"]
    assert first["memory_id"] != args["memory_id"]
    assert {"preference", "auto-learning", "background-review"} <= set(first["tags"])


def test_background_search_forces_personal_scope_and_strips_organization():
    with canonical_memory_review_policy(True):
        args = apply_canonical_memory_request_policy(
            CANONICAL_MEMORY_SEARCH_TOOL,
            {
                "query": " concise updates ",
                "limit": 999,
                "ownership": "organization",
                "organization_id": "org_123",
            },
        )
    assert args == {
        "query": "concise updates",
        "limit": 10,
        "ownership": "personal",
    }


def test_background_local_memory_add_routes_to_canonical_capture():
    with canonical_memory_review_policy(True):
        tool_name, args = redirect_local_memory_call(
            "memory",
            {
                "action": "add",
                "target": "user",
                "content": "User prefers concise engineering updates.",
            },
        )

    assert tool_name == CANONICAL_MEMORY_CAPTURE_TOOL
    assert args == {
        "body": "User prefers concise engineering updates.",
        "tags": ["local-memory-compat", "legacy-target-user"],
    }


def test_background_local_memory_add_defaults_to_memory_target():
    with canonical_memory_review_policy(True):
        tool_name, args = redirect_local_memory_call(
            "memory",
            {
                "action": "add",
                "content": "The agent should keep concise implementation notes.",
            },
        )

    assert tool_name == CANONICAL_MEMORY_CAPTURE_TOOL
    assert args["tags"] == ["local-memory-compat", "legacy-target-memory"]


def test_foreground_and_non_add_local_memory_calls_are_not_redirected():
    foreground = {
        "action": "add",
        "target": "user",
        "content": "User prefers concise engineering updates.",
    }
    assert redirect_local_memory_call("memory", foreground) == ("memory", foreground)

    replace = {
        "action": "replace",
        "target": "user",
        "old_text": "concise",
        "content": "User prefers detailed engineering updates.",
    }
    with canonical_memory_review_policy(True):
        assert redirect_local_memory_call("memory", replace) == ("memory", replace)

    invalid_target = {
        "action": "add",
        "target": "organization",
        "content": "The organization uses a private deployment convention.",
    }
    with canonical_memory_review_policy(True):
        assert redirect_local_memory_call("memory", invalid_target) == (
            "memory",
            invalid_target,
        )


def test_unsafe_or_transient_automatic_memory_is_rejected():
    assert auto_memory_block_reason("Weather today is sunny in Austin.")
    assert auto_memory_block_reason('{"role":"user","content":"raw transcript"}')
    assert auto_memory_block_reason("Authorization: Bearer secret-value-12345")
    assert auto_memory_block_reason("User email is private@example.com.")
    assert auto_memory_block_reason("Tool failed because a binary is missing.")


def test_durable_preference_with_error_word_is_allowed():
    assert (
        auto_memory_block_reason(
            "User prefers error reports to lead with the root cause and the failed check."
        )
        is None
    )


def test_policy_rejected_review_is_a_successful_noop(caplog):
    calls = []

    def dispatch(name, args):
        calls.append((name, dict(args)))
        raise AssertionError("A policy-rejected capture must not reach MCP.")

    with caplog.at_level(logging.INFO), canonical_memory_review_policy(True):
        result = dispatch_canonical_memory_tool(
            CANONICAL_MEMORY_CAPTURE_TOOL,
            {"body": "Weather today is sunny in Austin."},
            dispatch,
        )

    assert json.loads(result) == {
        "success": True,
        "skipped": True,
        "reason": "policy",
        "message": "Automatic Memory capture cannot store transient failures or one-off task state.",
    }
    assert calls == []
    assert "capture skipped" in caplog.text


def test_capture_searches_personal_memory_then_writes_once():
    calls = []

    def dispatch(name, args):
        calls.append((name, dict(args)))
        if name == CANONICAL_MEMORY_SEARCH_TOOL:
            return _mcp_result("Memory search results", [])
        return _mcp_result(
            "Saved memory",
            {
                "id": args["memory_id"],
                "body": args["body"],
                "ownership": {"kind": "personal"},
            },
        )

    with canonical_memory_review_policy(True):
        result = dispatch_canonical_memory_tool(
            CANONICAL_MEMORY_CAPTURE_TOOL,
            {
                "body": "User prefers concise engineering updates.",
                "ownership": "organization",
                "organization_id": "org_123",
            },
            dispatch,
        )

    assert [name for name, _args in calls] == [
        CANONICAL_MEMORY_SEARCH_TOOL,
        CANONICAL_MEMORY_CAPTURE_TOOL,
    ]
    search_args = calls[0][1]
    capture_args = calls[1][1]
    assert search_args["ownership"] == "personal"
    assert capture_args["ownership"] == "personal"
    assert "organization_id" not in capture_args
    assert capture_args["memory_id"]
    assert "Saved memory" in json.loads(result)["result"]


def test_duplicate_review_does_not_create_a_second_memory():
    calls = []
    body = "User prefers concise engineering updates."

    def dispatch(name, args):
        calls.append((name, dict(args)))
        assert name == CANONICAL_MEMORY_SEARCH_TOOL
        return _mcp_result(
            "Memory search results",
            [{"id": "existing", "body": body, "score": 1.0}],
        )

    with canonical_memory_review_policy(True):
        result = dispatch_canonical_memory_tool(
            CANONICAL_MEMORY_CAPTURE_TOOL,
            {"body": body},
            dispatch,
        )

    parsed = json.loads(result)
    assert parsed["success"] is True
    assert parsed["skipped"] is True
    assert [name for name, _args in calls] == [CANONICAL_MEMORY_SEARCH_TOOL]


def test_semantically_equivalent_memory_uses_high_confidence_score():
    calls = []

    def dispatch(name, args):
        calls.append((name, dict(args)))
        assert name == CANONICAL_MEMORY_SEARCH_TOOL
        return _mcp_result(
            "Memory search results",
            [
                {
                    "id": "existing",
                    "body": "Justin likes brief updates.",
                    "score": 0.93,
                }
            ],
        )

    with canonical_memory_review_policy(True):
        result = dispatch_canonical_memory_tool(
            CANONICAL_MEMORY_CAPTURE_TOOL,
            {"body": "User prefers concise engineering updates."},
            dispatch,
        )

    assert json.loads(result)["reason"] == "duplicate"
    assert len(calls) == 1


def test_search_failure_is_observable_and_capture_is_skipped(caplog):
    calls = []

    def dispatch(name, args):
        calls.append((name, dict(args)))
        return json.dumps({"error": "search unavailable"})

    with caplog.at_level(logging.WARNING), canonical_memory_review_policy(True):
        result = dispatch_canonical_memory_tool(
            CANONICAL_MEMORY_CAPTURE_TOOL,
            {"body": "User prefers concise engineering updates."},
            dispatch,
        )

    assert "deduplication search failed" in json.loads(result)["error"]
    assert len(calls) == 1
    assert "dedup search failed" in caplog.text


def test_capture_failure_is_observable_without_raising(caplog):
    calls = []

    def dispatch(name, args):
        calls.append((name, dict(args)))
        if name == CANONICAL_MEMORY_SEARCH_TOOL:
            return _mcp_result("Memory search results", [])
        return json.dumps({"error": "capture unavailable"})

    with caplog.at_level(logging.WARNING), canonical_memory_review_policy(True):
        result = dispatch_canonical_memory_tool(
            CANONICAL_MEMORY_CAPTURE_TOOL,
            {"body": "User prefers concise engineering updates."},
            dispatch,
        )

    assert json.loads(result)["error"] == "capture unavailable"
    assert len(calls) == 2
    assert "capture failed" in caplog.text


def test_canonical_tools_available_directly():
    agent = types.SimpleNamespace(
        valid_tool_names={
            CANONICAL_MEMORY_CAPTURE_TOOL,
            CANONICAL_MEMORY_SEARCH_TOOL,
        },
        enabled_toolsets=["devdocs"],
        disabled_toolsets=[],
    )
    assert canonical_memory_tools_available(agent) is True


def test_canonical_tools_available_when_deferred(monkeypatch):
    agent = types.SimpleNamespace(
        valid_tool_names={"tool_search", "tool_describe", "tool_call"},
        enabled_toolsets=["devdocs"],
        disabled_toolsets=[],
    )

    monkeypatch.setattr(
        "model_tools.get_tool_definitions",
        lambda **_kwargs: [
            {"function": {"name": CANONICAL_MEMORY_CAPTURE_TOOL}},
            {"function": {"name": CANONICAL_MEMORY_SEARCH_TOOL}},
        ],
    )

    assert canonical_memory_tools_available(agent) is True
    assert background_review_memory_allowed_tools(agent) == {
        CANONICAL_MEMORY_CAPTURE_TOOL,
        CANONICAL_MEMORY_SEARCH_TOOL,
        "tool_search",
        "tool_describe",
        "tool_call",
    }


def test_incomplete_or_memory_disabled_profile_does_not_gain_writes(monkeypatch):
    agent = types.SimpleNamespace(
        valid_tool_names={"skill_manage"},
        enabled_toolsets=["skills"],
        disabled_toolsets=[],
    )
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **_kwargs: [])
    assert canonical_memory_tools_available(agent) is False
    assert background_review_memory_allowed_tools(agent) == set()
