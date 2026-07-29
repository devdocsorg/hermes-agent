from cron.scheduler import (
    SCHEDULED_RUN_ACK_PREFIX,
    _prepend_cron_codex_envelope,
    _strip_valid_scheduled_run_ack,
)


def test_cron_codex_envelope_precedes_loaded_skill_text():
    prompt = _prepend_cron_codex_envelope(
        '[IMPORTANT: The user has invoked the "example" skill.]',
        job_id="37de1ffd1b92",
        job_name="DevDocs gbrain ingest",
        run_id="cron_37de1ffd1b92_20260728_120000",
    )

    assert prompt.startswith(
        "Hermes Cron: DevDocs gbrain ingest\n"
        "Hermes Cron Job ID: 37de1ffd1b92\n"
        "Hermes Cron Run ID: cron_37de1ffd1b92_20260728_120000\n"
    )
    assert "inherited_thread_ids" in prompt
    assert prompt.index("Hermes Cron:") < prompt.index("[IMPORTANT:")


def test_valid_ack_is_stripped_before_delivery():
    ack = (
        SCHEDULED_RUN_ACK_PREFIX
        + '{"scheduler":"hermes-cron","scheduler_id":"37de1ffd1b92",'
        '"run_id":"cron_37de1ffd1b92_20260728_120000",'
        '"inherited_thread_ids":["thread-old"],"status":"resolved","unresolved":[]}'
    )

    visible, parsed = _strip_valid_scheduled_run_ack(
        f"[SILENT]\n{ack}",
        job_id="37de1ffd1b92",
        run_id="cron_37de1ffd1b92_20260728_120000",
    )

    assert visible == "[SILENT]"
    assert parsed["inherited_thread_ids"] == ["thread-old"]


def test_mismatched_ack_remains_visible_and_is_not_trusted():
    ack = (
        SCHEDULED_RUN_ACK_PREFIX
        + '{"scheduler":"hermes-cron","scheduler_id":"other",'
        '"run_id":"cron_other","inherited_thread_ids":[],"status":"resolved",'
        '"unresolved":[]}'
    )

    visible, parsed = _strip_valid_scheduled_run_ack(
        ack,
        job_id="37de1ffd1b92",
        run_id="cron_37de1ffd1b92_20260728_120000",
    )

    assert visible == ack
    assert parsed is None
