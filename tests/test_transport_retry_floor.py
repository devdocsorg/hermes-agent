"""A retry budget is a DURATION, not a count.

Regression tests for the 2026-07-26 router outage: at 15:11:50 CDT the local
LiteLLM router stopped accepting connections, and every live Hermes session
spent its entire retry budget — 3 attempts, a primary-transport rebuild, 3 more
attempts — inside about 23 SECONDS before declaring the turn dead. Seven
sessions died in two minutes; one multi-hour workstream then sat silent for 28
hours until a human noticed.

The policy under test grants further attempts while a transport-class failure is
still younger than a wall-clock floor, so a provider that comes back in a minute
does not cost a turn that has been running for hours.
"""

import pytest

from agent.retry_utils import (
    TRANSPORT_RETRY_FLOOR_SECONDS,
    TRANSPORT_RETRY_MAX_EXTENSIONS,
    should_extend_transport_retry,
)


class TestTransportRetryFloor:
    def test_the_actual_incident_would_have_been_extended(self):
        """23 seconds into a transport outage is not a spent budget.

        These are the real numbers from session 20260725_073247_4cfee5c2:
        first failure 15:11:52, give-up 15:12:15.
        """
        assert should_extend_transport_retry(
            reason="timeout",
            elapsed_s=23.2,
            floor_s=TRANSPORT_RETRY_FLOOR_SECONDS,
        ) is True

    def test_stops_once_the_floor_is_actually_satisfied(self):
        assert should_extend_transport_retry(
            reason="timeout",
            elapsed_s=TRANSPORT_RETRY_FLOOR_SECONDS + 0.1,
            floor_s=TRANSPORT_RETRY_FLOOR_SECONDS,
        ) is False

    def test_boundary_is_exclusive_at_exactly_the_floor(self):
        assert should_extend_transport_retry(
            reason="timeout", elapsed_s=300.0, floor_s=300.0
        ) is False

    @pytest.mark.parametrize("reason", ["timeout", "server_error", "overloaded"])
    def test_wire_broke_reasons_are_waited_out(self, reason):
        assert should_extend_transport_retry(
            reason=reason, elapsed_s=1.0, floor_s=300.0
        ) is True

    @pytest.mark.parametrize(
        "reason",
        [
            "auth",
            "auth_permanent",
            "billing",
            "rate_limit",
            "upstream_rate_limit",
            "content_policy_blocked",
            "context_overflow",
            "payload_too_large",
            "ssl_cert_verification",
        ],
    )
    def test_deterministic_failures_are_never_waited_out(self, reason):
        """Retrying a wall that will not move just burns quota."""
        assert should_extend_transport_retry(
            reason=reason, elapsed_s=1.0, floor_s=300.0
        ) is False

    def test_accepts_a_failover_reason_enum_not_just_a_string(self):
        from agent.error_classifier import FailoverReason

        assert should_extend_transport_retry(
            reason=FailoverReason.timeout, elapsed_s=1.0, floor_s=300.0
        ) is True
        assert should_extend_transport_retry(
            reason=FailoverReason.billing, elapsed_s=1.0, floor_s=300.0
        ) is False

    def test_zero_floor_disables_the_policy_entirely(self):
        """The documented escape hatch must actually restore old behavior."""
        assert should_extend_transport_retry(
            reason="timeout", elapsed_s=0.0, floor_s=0.0
        ) is False

    def test_a_permanently_dead_endpoint_still_terminates(self):
        """Extensions are capped so an unreachable host cannot retry forever."""
        assert should_extend_transport_retry(
            reason="timeout",
            elapsed_s=1.0,
            floor_s=10**9,
            extensions_used=TRANSPORT_RETRY_MAX_EXTENSIONS,
        ) is False

    def test_garbage_inputs_fail_closed_rather_than_raising(self):
        assert should_extend_transport_retry(
            reason=None, elapsed_s=1.0, floor_s=300.0
        ) is False
        assert should_extend_transport_retry(
            reason="timeout", elapsed_s="not-a-number", floor_s=300.0
        ) is False


class TestAgentWiring:
    """The policy is worthless if the agent never reads it."""

    def _agent(self, cfg_value=None):
        from unittest.mock import patch

        from run_agent import AIAgent

        cfg = {"agent": {}}
        if cfg_value is not None:
            cfg["agent"]["api_transport_retry_floor_seconds"] = cfg_value

        with patch("run_agent.OpenAI"), patch(
            "hermes_cli.config.load_config", return_value=cfg
        ):
            return AIAgent(
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                model="test/model",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

    def test_default_floor_is_wired_onto_the_agent(self):
        agent = self._agent()
        assert agent._api_transport_retry_floor_s == TRANSPORT_RETRY_FLOOR_SECONDS

    def test_config_override_is_honored(self):
        assert self._agent(45)._api_transport_retry_floor_s == 45.0

    def test_zero_is_preserved_as_an_opt_out(self):
        assert self._agent(0)._api_transport_retry_floor_s == 0.0

    def test_invalid_value_falls_back_to_the_default(self):
        assert (
            self._agent("banana")._api_transport_retry_floor_s
            == TRANSPORT_RETRY_FLOOR_SECONDS
        )

    def test_negative_value_is_clamped_not_treated_as_infinite(self):
        assert self._agent(-10)._api_transport_retry_floor_s == 0.0


class TestRetryLoopIntegration:
    """Pin that conversation_loop actually consults the clock.

    The loop is a 5000-line function that cannot be called in isolation, so this
    reads the wiring: the extension must be evaluated against the elapsed clock
    BEFORE the terminal give-up branch, and must bump the attempt allowance.
    A refactor that drops either half turns this red.
    """

    def _loop_source(self):
        import inspect

        from agent import conversation_loop

        return inspect.getsource(conversation_loop)

    def test_extension_is_evaluated_before_the_giveup_branch(self):
        src = self._loop_source()
        guard = src.find("should_extend_transport_retry(")
        giveup = src.find("if retry_count >= max_retries:\n                    # Before falling back")
        assert guard != -1, "the transport clock check is gone"
        assert giveup != -1, "the terminal give-up branch moved; re-anchor this test"
        assert guard < giveup, (
            "the clock check must run before the give-up branch, otherwise the "
            "turn is already dead by the time patience is considered"
        )

    def test_extension_raises_the_attempt_allowance(self):
        src = self._loop_source()
        assert "_retry.transport_retry_extensions += 1" in src
        assert "max_retries += 1" in src

    def test_elapsed_time_comes_from_the_call_clock_not_the_counter(self):
        src = self._loop_source()
        assert "elapsed_s=time.time() - api_start_time" in src, (
            "the floor must be measured against wall-clock time spent on this "
            "API call — measuring anything else reintroduces the count bug"
        )

    def test_state_field_exists_so_extensions_are_bounded(self):
        from agent.turn_retry_state import TurnRetryState

        assert TurnRetryState().transport_retry_extensions == 0

    def test_recovery_ladder_runs_before_the_clock(self):
        """A healthy fallback provider beats any amount of patience.

        Placing the clock check ahead of the recovery ladder starves it: the
        first draft of this fix did exactly that and turned
        ``test_32646_fallback_429_after_timeout`` red, because the transport
        rebuild never got its turn. Both guards are pinned here so a future
        edit cannot quietly reorder them back.
        """
        src = self._loop_source()
        assert "_retry.budget_exhaustions > 1" in src, (
            "the first budget exhaustion must belong to the recovery ladder, "
            "not to the wall clock"
        )
        assert "not agent._has_pending_fallback()" in src, (
            "never wait out an outage while an unused fallback provider is "
            "still available to switch to"
        )

    def test_exhaustion_counter_starts_at_zero(self):
        from agent.turn_retry_state import TurnRetryState

        assert TurnRetryState().budget_exhaustions == 0
