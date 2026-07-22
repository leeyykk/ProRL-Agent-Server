import pytest

from polar.agent.factory import create_harness
from polar.agent.models import AgentSpec


def test_codex_harness_pins_task_to_workspace():
    harness = create_harness(
        AgentSpec(harness="codex", model_name="/models/Qwen3.5-4B")
    )

    steps = harness.run_steps("fix the bug")

    assert len(steps) == 2
    command = steps[1].command
    assert "You are already inside the benchmark repository workspace" in command
    assert "Continue working until you have" in command
    assert "fix the bug" in command
    assert "codex exec" in command
    assert "codex exec resume" not in command


def test_codex_harness_retries_same_session_while_tracked_diff_is_empty():
    harness = create_harness(
        AgentSpec(
            harness="codex",
            model_name="/models/Qwen3.5-4B",
            settings={"empty_diff_retries": 4},
        )
    )

    command = harness.run_steps("fix the bug")[1].command

    assert 'while [ "$retry" -lt 4 ]' in command
    assert "git diff --quiet -- ." in command
    assert "git diff --cached --quiet -- ." in command
    assert "codex exec resume" in command
    assert "--last" in command
    assert "POLAR_CODEX_EMPTY_DIFF_RETRY=" in command
    assert "POLAR_CODEX_EMPTY_DIFF_RETRIES_EXHAUSTED=1" in command
    assert "tee -a /polar/session/logs/agent/codex.txt" in command


def test_codex_harness_rejects_negative_empty_diff_retries():
    with pytest.raises(ValueError, match="empty_diff_retries must be non-negative"):
        create_harness(
            AgentSpec(
                harness="codex",
                settings={"empty_diff_retries": -1},
            )
        )
