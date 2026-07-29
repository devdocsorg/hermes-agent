"""A resumed workstream must not need a human to re-authorise it.

Regression tests for the recurring "Hermes is stuck" reports of 2026-07-27. The
sequence, from the operator's own Slack thread:

  20:10:14  ⚠️ Iteration budget exhausted (90/90) — asking model to summarise
  20:30:56  ⚠️ Gateway shutting down — Your current task will be interrupted.
  20:31:53  The session was restored successfully. What would you like to do next?

The restore worked perfectly. The *work* stopped, and stayed stopped, because the
resumed turn asked a question instead of continuing — so a multi-hour autonomous
workstream needed a human to type "keep going". That same thread contains "Keep
going", "Why did you stop keep going", "You stalled and didn't finish out why?"
and "don't wait for me", which is what this failure mode looks like from outside.

#57056 had already fixed exactly this for non-interactive platforms (webhook, API
server) on the grounds that an acknowledgement nobody reads silently abandons the
task. The same argument applies to a DM: a human being *present* is not a human
*waiting to re-approve work already in flight*.
"""

from unittest.mock import patch

import pytest

from gateway.run import _resume_continues_work, build_resume_recovery_note


class TestResumeGuidance:
    def test_interactive_default_continues_the_task(self):
        note = build_resume_recovery_note("restart_timeout", "", interactive=True)
        assert "CONTINUE the interrupted task" in note
        assert "ask what they would like to do next" not in note

    def test_continuing_is_not_the_same_as_continuing_silently(self):
        """The operator still gets told what was picked back up."""
        note = build_resume_recovery_note("restart_timeout", "", interactive=True)
        assert "ONE short line" in note

    def test_it_still_leaves_room_for_a_genuine_blocker(self):
        note = build_resume_recovery_note("restart_timeout", "", interactive=True)
        assert "genuinely blocked" in note

    def test_opt_out_restores_the_ask_first_note(self):
        note = build_resume_recovery_note(
            "restart_timeout", "", interactive=True, continue_work=False
        )
        assert "session was restored" in note
        assert "ask what they would like to do next" in note
        assert "CONTINUE the interrupted task" not in note

    def test_noninteractive_is_unchanged_by_the_knob(self):
        """#57056's behavior must not depend on the new setting."""
        for flag in (True, False):
            note = build_resume_recovery_note(
                "restart_timeout", "", interactive=False, continue_work=flag
            )
            assert "CONTINUE the interrupted task" in note
            assert "session was restored" not in note

    def test_a_real_new_user_message_still_wins(self):
        """A human who actually said something is not overridden by resume."""
        note = build_resume_recovery_note(
            "restart_timeout", "do the other thing", interactive=True
        )
        assert "NEW message" in note
        assert "do the other thing" in note
        assert "CONTINUE the interrupted task" not in note

    def test_resumed_note_never_discards_the_interrupted_work(self):
        note = build_resume_recovery_note("restart_timeout", "", interactive=True)
        assert "skip any unfinished work" not in note, (
            "telling the model to skip unfinished work is the opposite of "
            "resuming it"
        )

    def test_replayed_tool_calls_are_still_guarded(self):
        note = build_resume_recovery_note("restart_timeout", "", interactive=True)
        assert "already appear in the history" in note


class TestKnobResolution:
    def _with_config(self, cfg):
        return patch("gateway.run._load_gateway_config", return_value=cfg)

    def test_default_is_continue(self):
        with self._with_config({}):
            assert _resume_continues_work() is True

    def test_absent_agent_section_is_continue(self):
        with self._with_config({"other": {}}):
            assert _resume_continues_work() is True

    @pytest.mark.parametrize("value", [False, "false", "no", "0", "off"])
    def test_explicit_opt_out_is_honored(self, value):
        with self._with_config({"agent": {"resume_continues_work": value}}):
            assert _resume_continues_work() is False

    @pytest.mark.parametrize("value", [True, "true", "yes", "1"])
    def test_explicit_opt_in_is_honored(self, value):
        with self._with_config({"agent": {"resume_continues_work": value}}):
            assert _resume_continues_work() is True

    def test_null_means_default_not_disabled(self):
        with self._with_config({"agent": {"resume_continues_work": None}}):
            assert _resume_continues_work() is True

    def test_unreadable_config_fails_open_to_continuing(self):
        """A broken config must not silently strand every resumed workstream."""
        with patch(
            "gateway.run._load_gateway_config", side_effect=OSError("boom")
        ):
            assert _resume_continues_work() is True

    def test_knob_is_read_per_call_not_cached_at_import(self):
        """Flipping the setting must not require a gateway restart."""
        with self._with_config({"agent": {"resume_continues_work": False}}):
            assert _resume_continues_work() is False
        with self._with_config({"agent": {"resume_continues_work": True}}):
            assert _resume_continues_work() is True


class TestWiring:
    def test_both_call_sites_pass_the_knob(self):
        """A knob the dispatch path ignores is decoration."""
        import inspect

        from gateway import run as gateway_run

        src = inspect.getsource(gateway_run)
        assert src.count("continue_work=_resume_continues_work()") == 2, (
            "both build_resume_recovery_note call sites in the dispatch path "
            "must pass the knob — the second one is the empty-text safety net, "
            "which is the path the startup auto-resume actually takes"
        )
