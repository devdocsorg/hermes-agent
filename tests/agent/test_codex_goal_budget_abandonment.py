"""A codex turn abandoned on its goal token budget must not read as an answer.

The incident (2026-07-29, thread 019fae02): the model called codex's
`create_goal` with `token_budget: 12000`; codex charged the goal 22,101 tokens
for the very call that created it, marked it `budget_limited` 8 seconds later,
and injected a wrap-up instruction. The model wrote "I couldn't complete this
run within the execution budget" and the operator's Slack thread got a
checkpoint instead of the work. Every layer above the transport saw
`turn.error is None`, `turn.interrupted is False` and non-empty `final_text` —
i.e. a finished answer — so `_automatic_continuation_for_result` in
gateway/run.py could never fire.

These tests pin the detector that tells the two apart. The decisive one is
`test_scoped_to_this_turn`: a rollout accumulates across every turn of a thread,
so an unscoped scan would mark all 112 later turns of thread 019f777d abandoned
forever.
"""

from __future__ import annotations

import json

import pytest

from agent.codex_runtime import (
    CODEX_GOAL_BUDGET_EXIT_REASON,
    _turn_abandoned_for_goal_budget,
    codex_completion_fields,
)


GOAL_CONTEXT_TEXT = (
    '<codex_internal_context source="goal">\n'
    "The active thread goal has reached its token budget.\n"
    "Budget:\n- Tokens used: 22101\n- Token budget: 12000\n"
    "The system has marked the goal as budget_limited, so do not start new "
    "substantive work for this goal.\n"
    "</codex_internal_context>"
)


def _rec(payload: dict, rtype: str = "response_item") -> str:
    return json.dumps({"type": rtype, "payload": payload}) + "\n"


def _user_msg(text: str) -> str:
    return _rec({"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]})


def _assistant_msg(text: str) -> str:
    return _rec(
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}
    )


def _write(tmp_path, *lines: str):
    p = tmp_path / "rollout-2026-07-29T08-13-21-019fae02.jsonl"
    p.write_text("".join(lines))
    return str(p)


def test_detects_the_incident_shape(tmp_path):
    """POSITIVE: codex's own injected goal context marks the turn abandoned."""
    path = _write(tmp_path, _user_msg("do the thing"), _user_msg(GOAL_CONTEXT_TEXT))
    assert _turn_abandoned_for_goal_budget(path, 0) is True


def test_a_normal_turn_is_not_abandoned(tmp_path):
    """NEGATIVE #1: a turn that never hit a budget reads as a real answer."""
    path = _write(tmp_path, _user_msg("do the thing"), _assistant_msg("done, here it is"))
    assert _turn_abandoned_for_goal_budget(path, 0) is False


def test_scoped_to_this_turn(tmp_path):
    """NEGATIVE #2 — THE DECISIVE ONE.

    A rollout is per-THREAD and accumulates. The same file must report True
    when read from the start (the turn that WAS abandoned) and False when read
    from after the marker (every later turn on that thread). If both are True
    the turn scoping is not wired, and one budget event would poison a thread
    forever.
    """
    head = _user_msg("do the thing") + _user_msg(GOAL_CONTEXT_TEXT)
    tail = _user_msg("next, unrelated request") + _assistant_msg("sure, done")
    path = _write(tmp_path, head, tail)

    assert _turn_abandoned_for_goal_budget(path, 0) is True, "the abandoned turn must be seen"
    assert (
        _turn_abandoned_for_goal_budget(path, len(head.encode())) is False
    ), "a LATER turn on the same thread must not inherit the earlier budget event"


def test_merely_quoting_the_marker_is_not_abandonment(tmp_path):
    """NEGATIVE #3: the match is structural, not a substring scan.

    A rollout stores user text, assistant text and tool output. A turn that
    DISCUSSES a budget abandonment — this very investigation did — must not
    mark itself abandoned. Pins the `role == user` +
    `<codex_internal_context source="goal">` envelope requirement.
    """
    quoting_assistant = _assistant_msg(
        "The log said the goal 'has reached its token budget', which explains it."
    )
    quoting_user = _user_msg(
        "why did it say the active thread goal has reached its token budget?"
    )
    tool_output = _rec(
        {"type": "function_call_output", "output": "…has reached its token budget…"}
    )
    # The one that makes the `role == "user"` check load-bearing: an ASSISTANT
    # message whose text OPENS with the goal envelope — an agent pasting the
    # injected block back while explaining an incident, which is exactly what
    # the session investigating this bug did. Without the role check the
    # envelope-prefix test alone would accept it and every such turn would be
    # re-driven three times. My first version of this test used only the three
    # fixtures above, all of which fail the prefix test anyway, so it PASSED
    # under the very mutation its name claims to detect — a test whose name
    # promises a mutation it cannot catch is the vacuous shape, not a control.
    assistant_pasting_the_envelope = _assistant_msg(GOAL_CONTEXT_TEXT)
    path = _write(
        tmp_path, quoting_assistant, quoting_user, tool_output, assistant_pasting_the_envelope
    )
    assert _turn_abandoned_for_goal_budget(path, 0) is False


def test_missing_rollout_degrades_to_not_abandoned(tmp_path):
    """No rollout => False, never True.

    A detector that cannot run must degrade to today's behaviour (an
    abandonment reported as an answer), never to a FALSE abandonment that would
    re-drive healthy turns three times each.
    """
    assert _turn_abandoned_for_goal_budget(None, 0) is False


def test_exit_reason_opens_the_gateway_recovery_gate():
    """The reason string must satisfy gateway/run.py's recovery predicate.

    `_automatic_continuation_for_result` returns None unless the reason starts
    with `max_iterations_reached(`. Emitting a NEW prefix would produce an
    incomplete turn that nothing re-drives — strictly worse than today, because
    the resume_pending and restart-counter paths would fire with no recovery.
    """
    assert CODEX_GOAL_BUDGET_EXIT_REASON.startswith("max_iterations_reached(")
    assert "codex_goal_budget" in CODEX_GOAL_BUDGET_EXIT_REASON


def test_abandonment_is_wired_into_the_turn_result():
    """The detector must actually reach `completed` — the silent-mutation gap.

    Dropping the `goal_budget_limited` term from `completed` restores the
    production bug, changes no syntax, and breaks nothing else in the suite.
    This is the only test that would go red for it.
    """
    abandoned = codex_completion_fields(interrupted=False, error=None, goal_budget_limited=True)
    assert abandoned["completed"] is False
    assert abandoned["turn_exit_reason"] == CODEX_GOAL_BUDGET_EXIT_REASON

    answered = codex_completion_fields(interrupted=False, error=None, goal_budget_limited=False)
    assert answered["completed"] is True
    # ABSENT, not None: a reason key on a healthy turn would send the next
    # reader looking for a failure that did not happen.
    assert "turn_exit_reason" not in answered

    # the pre-existing paths must be unchanged
    assert codex_completion_fields(
        interrupted=True, error=None, goal_budget_limited=False
    )["completed"] is False
    assert codex_completion_fields(
        interrupted=False, error="boom", goal_budget_limited=False
    )["completed"] is False


def test_gateway_predicate_accepts_it():
    """End-to-end on the predicate itself, with the real gateway function."""
    from gateway.run import _automatic_continuation_for_result

    abandoned = {
        "completed": False,
        "failed": False,
        "interrupted": False,
        "turn_exit_reason": CODEX_GOAL_BUDGET_EXIT_REASON,
    }
    assert _automatic_continuation_for_result(abandoned, 0) is not None

    # and the pre-fix shape — what codex_runtime reported before today — must
    # still be treated as a finished answer, so this cannot re-drive good turns
    answered = {"completed": True, "failed": False, "interrupted": False}
    assert _automatic_continuation_for_result(answered, 0) is None


@pytest.mark.parametrize("depth", [3, 4])
def test_continuation_is_still_bounded(depth):
    """The new reason must not escape the 3-continuation bound."""
    from gateway.run import _automatic_continuation_for_result

    result = {
        "completed": False,
        "failed": False,
        "interrupted": False,
        "turn_exit_reason": CODEX_GOAL_BUDGET_EXIT_REASON,
    }
    assert _automatic_continuation_for_result(result, depth) is None
