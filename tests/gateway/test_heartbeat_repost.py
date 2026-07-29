"""An edit is not a notification.

Regression tests for the 2026-07-27 "Hermes is stuck" report. A Slack run
posted its long-running heartbeat at 19:40:32 and then edited that same bubble
for the rest of the run — verified against the Slack API at 20:07, where the
message read "Working — 28 min — iteration 67/90" with an edit stamp of
20:06:35. The agent was healthy the whole time (API call #67, tools executing),
but an edited message produces no notification, no unread badge, and keeps its
original timestamp, so the thread looked frozen for 26 minutes and the operator
reported the agent dead.

The heartbeat therefore re-POSTS on a slower cadence: edits stay cheap for
minute-to-minute updates, while a long run still emits a real, visible message
periodically.
"""

import inspect

import pytest

from gateway import run as gateway_run


class TestRepostConstant:
    def test_repost_interval_is_exported_and_sane(self):
        interval = gateway_run.HEARTBEAT_REPOST_INTERVAL_SECONDS
        assert interval > 0, "an interval of 0 restores the silent-edit bug"
        assert interval >= 300, (
            "re-posting faster than every few minutes turns a progress signal "
            "into channel spam — the reason edit-in-place existed at all"
        )

    def test_repost_interval_is_slower_than_the_edit_interval(self):
        """Edits must remain the common case; re-posts the occasional one."""
        default_notify_interval = 180.0
        assert (
            gateway_run.HEARTBEAT_REPOST_INTERVAL_SECONDS > default_notify_interval
        )


class TestRepostWiring:
    """The loop is a closure inside a 20k-line coroutine, so read its source."""

    def _notify_source(self):
        src = inspect.getsource(gateway_run)
        start = src.find("async def _notify_long_running()")
        assert start != -1, "_notify_long_running was renamed; re-anchor this test"
        end = src.find("_notify_task = asyncio.create_task", start)
        assert end != -1
        return src[start:end]

    def test_a_stale_bubble_is_retired_so_the_next_write_is_a_new_message(self):
        body = self._notify_source()
        assert "_heartbeat_posted_at" in body, "no re-post clock at all"
        assert "_heartbeat_msg_id = None" in body, (
            "retiring the bubble id is what forces send() instead of "
            "edit_message() on the next tick"
        )

    def test_repost_decision_uses_elapsed_time_since_the_bubble_was_posted(self):
        body = self._notify_source()
        assert "(time.time() - _heartbeat_posted_at) >= _REPOST_INTERVAL" in body, (
            "the re-post clock must measure how long THIS bubble has been "
            "edited, not how long the run has been going"
        )

    def test_posting_stamps_the_clock(self):
        body = self._notify_source()
        post_idx = body.find("_heartbeat_msg_id = str(_notify_res.message_id)")
        stamp_idx = body.find("_heartbeat_posted_at = time.time()")
        assert post_idx != -1 and stamp_idx != -1
        assert stamp_idx > post_idx, (
            "without stamping on send, the re-post clock never resets and the "
            "heartbeat re-posts on every single tick"
        )

    def test_edit_path_is_still_present(self):
        """The fix must not degenerate into posting a new bubble every tick."""
        body = self._notify_source()
        assert "edit_message(" in body

    def test_repost_is_configurable_and_disablable(self):
        body = self._notify_source()
        assert "HERMES_AGENT_NOTIFY_REPOST_INTERVAL" in body
        assert "_REPOST_INTERVAL_RAW if _REPOST_INTERVAL_RAW > 0 else None" in body, (
            "0 must disable re-posting rather than mean 'every tick'"
        )

    def test_repost_is_guarded_against_none_interval(self):
        body = self._notify_source()
        assert "_REPOST_INTERVAL is not None" in body, (
            "a disabled re-post interval must not raise when compared"
        )

    def test_config_key_is_promoted_to_the_env_var_the_loop_reads(self):
        src = inspect.getsource(gateway_run)
        assert "gateway_notify_repost_interval" in src, (
            "config.yaml knob missing — the env var would be the only way to "
            "tune this, which is not how the sibling notify_interval works"
        )


class TestHeartbeatIsDmOnly:
    """Operator instruction 2026-07-27: in channels shared with other people,
    only the ephemeral typing/status indicator — no permanent bubble. In a
    private conversation the bubble is the only thing that distinguishes a
    working run from a dead one.

    Gated on the pre-existing ``chat_type`` field on purpose. Inventing an
    "is the operator alone in here" detector would be a classifier over
    ambiguous evidence (a private channel with two members looks like a group),
    and this repo has already paid for that mistake more than once.
    """

    DM_LIKE = {"dm", "thread"}

    @staticmethod
    def _heartbeat_enabled(chat_type, channel_override=False):
        # Mirrors the gate in the turn setup.
        if channel_override:
            return True
        return str(chat_type or "").lower() in TestHeartbeatIsDmOnly.DM_LIKE

    @pytest.mark.parametrize("chat_type", ["dm", "thread", "DM"])
    def test_private_conversations_get_the_bubble(self, chat_type):
        assert self._heartbeat_enabled(chat_type) is True

    @pytest.mark.parametrize("chat_type", ["group", "channel", "", None])
    def test_shared_channels_get_only_the_status_line(self, chat_type):
        assert self._heartbeat_enabled(chat_type) is False

    def test_env_override_restores_heartbeat_everywhere(self):
        assert self._heartbeat_enabled("channel", channel_override=True) is True

    def test_gate_is_actually_wired_into_the_turn_setup(self):
        src = inspect.getsource(gateway_run)
        assert "HERMES_AGENT_NOTIFY_CHANNEL_HEARTBEAT" in src, (
            "no escape hatch — the operator cannot re-enable channel heartbeats"
        )
        assert 'if _chat_type not in {"dm", "thread"}:' in src, (
            "the DM-only gate is gone; shared channels will get permanent "
            "progress bubbles again"
        )

    def test_gate_runs_before_the_notify_task_is_created(self):
        src = inspect.getsource(gateway_run)
        gate = src.find("HERMES_AGENT_NOTIFY_CHANNEL_HEARTBEAT")
        task = src.find("_notify_task = asyncio.create_task(_notify_long_running())")
        assert gate != -1 and task != -1
        assert gate < task, "suppressing after the task starts posts one bubble anyway"


class TestRepostThreshold:
    """Pin the decision itself, independent of the loop's plumbing."""

    @staticmethod
    def _should_repost(age_s, interval):
        # Mirrors the guard in _notify_long_running.
        return bool(interval is not None and age_s >= interval)

    @pytest.mark.parametrize(
        "age_s,expected",
        [
            (0.0, False),
            (179.0, False),
            (599.0, False),
            (600.0, True),
            (1560.0, True),  # the observed 26-minute silent stretch
        ],
    )
    def test_threshold_behavior(self, age_s, expected):
        assert (
            self._should_repost(age_s, gateway_run.HEARTBEAT_REPOST_INTERVAL_SECONDS)
            is expected
        )

    def test_the_reported_incident_would_have_re_posted(self):
        """19:40:32 -> 20:06:35 is 1563s of silence; that must break the edit."""
        assert self._should_repost(
            1563.0, gateway_run.HEARTBEAT_REPOST_INTERVAL_SECONDS
        ) is True

    def test_disabled_interval_never_reposts(self):
        assert self._should_repost(10**6, None) is False
