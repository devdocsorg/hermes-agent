"""A message queued during the shutdown drain must survive the restart.

Before this existed, the drain path told the user "queued for the next turn
after it comes back" and then dropped the message: the queue was two in-process
dicts, the disk hook wrote a transcript row and never dispatched, and
"Flushed N pending message(s)" appears zero times in any log on this host.

Measured 2026-07-29: the operator's 07:45:08 Slack message was acknowledged at
07:45:12.004 and no turn ever ran for it — the session's last transcript row was
still from 07-27.
"""

import json
import time

import pytest

from gateway.run import GatewayRunner

SESSION_KEY = "agent:main:slack:dm:T02UWCK3WLS:D0B8Z2EV9V2:1785329112.004509"


@pytest.fixture
def runner(monkeypatch, tmp_path):
    """A bare GatewayRunner with _hermes_home pointed at a tmpdir.

    __new__ rather than __init__ on purpose: the drain-queue helpers touch only
    the filesystem, and constructing a real runner would drag in adapters,
    the session store and a config load for no added coverage.
    """
    import gateway.run as run_mod

    monkeypatch.setattr(run_mod, "_hermes_home", tmp_path)
    return GatewayRunner.__new__(GatewayRunner)


class _Source:
    platform = "slack"
    chat_id = "D0B8Z2EV9V2"
    user_id = "U0AQR8G8HNG"


class _Event:
    def __init__(self, text, media_urls=None):
        self.text = text
        self.media_urls = media_urls
        self.source = _Source()


def test_queued_text_survives_and_is_replayed_once(runner):
    """The core contract: persist during drain, deliver exactly once after."""
    assert runner._persist_drain_queued(SESSION_KEY, _Event("make the ceo thing a scheduled codex project"))

    # Delivered to the resumed turn...
    assert runner._take_drain_queued(SESSION_KEY) == "make the ceo thing a scheduled codex project"
    # ...and exactly once. An at-least-once replay would re-run the user's
    # instruction on every boot of a crash-looping gateway.
    assert runner._take_drain_queued(SESSION_KEY) == ""


def test_no_record_means_no_text(runner):
    """A session with nothing queued must yield "" so the resume stays generic."""
    assert runner._take_drain_queued(SESSION_KEY) == ""


def test_filename_leaks_no_chat_id_or_thread_ts(runner):
    """Session keys carry chat ids and thread timestamps; a filename is the one
    place that text is not escaped, so the path must be hashed."""
    name = runner._drain_queued_path(SESSION_KEY).name
    assert "D0B8Z2EV9V2" not in name
    assert "1785329112" not in name
    assert name.endswith(".json")


def test_empty_and_media_messages_are_not_promised(runner):
    """Only a durable record earns the "queued" reply.

    Media URLs are short-lived, so re-fetching them on a later boot is a
    different feature; refusing to persist means the caller sends the honest
    "please resend" instead of a promise it cannot keep.
    """
    assert runner._persist_drain_queued(SESSION_KEY, _Event("   ")) is False
    assert runner._persist_drain_queued(SESSION_KEY, _Event("look", media_urls=["http://x/1.jpg"])) is False
    assert not runner._drain_queued_path(SESSION_KEY).exists()


def test_stale_record_is_discarded_not_replayed(runner):
    """Silently acting on a day-old instruction is its own failure mode."""
    path = runner._drain_queued_path(SESSION_KEY)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "session_key": SESSION_KEY,
        "text": "restart the production database",
        "queued_at": time.time() - 200_000,
    }))

    assert runner._take_drain_queued(SESSION_KEY) == ""
    assert not path.exists(), "a discarded record must not be left to retry forever"


def test_corrupt_record_is_dropped_without_raising(runner):
    """A half-written record must not wedge every future boot."""
    path = runner._drain_queued_path(SESSION_KEY)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")

    assert runner._take_drain_queued(SESSION_KEY) == ""
    assert not path.exists()


def test_drain_queued_is_an_auto_resume_reason():
    """The persist half is useless unless the scheduler acts on the marker.

    NEGATIVE CONTROL for the wiring: drop "drain_queued" from
    _AUTO_RESUME_REASONS and the message is written to disk, acknowledged as
    queued, and then never picked up — the original bug with extra steps.
    """
    assert "drain_queued" in GatewayRunner._AUTO_RESUME_REASONS
