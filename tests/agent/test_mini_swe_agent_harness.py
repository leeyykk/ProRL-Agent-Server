from polar.agent.factory import create_harness
from polar.agent.models import AgentSpec


def test_mini_swe_agent_harness_prefixes_local_model_for_litellm():
    harness = create_harness(
        AgentSpec(
            harness="mini_swe_agent",
            model_name="/models/Qwen3.5-4B",
            settings={"config": ["/cfg/default.yaml", "agent.step_limit=3"]},
            env={
                "OPENAI_BASE_URL": "http://127.0.0.1:8000/v1",
                "OPENAI_API_KEY": "x",
            },
        )
    )

    steps = harness.run_steps("fix the bug")

    assert len(steps) == 1
    command = steps[0].command
    assert "mini --yolo" in command
    assert "--model openai//models/Qwen3.5-4B" in command
    assert "--config /cfg/default.yaml" in command
    assert "--config agent.step_limit=3" in command
    assert "logs/agent/mini-swe-agent.txt" in command
    assert steps[0].env["OPENAI_BASE_URL"] == "http://127.0.0.1:8000/v1"


def test_mini_swe_agent_harness_keeps_explicit_provider_prefix():
    harness = create_harness(
        AgentSpec(harness="mini_swe_agent", model_name="anthropic/claude-sonnet-4-5")
    )

    command = harness.run_steps("fix the bug")[0].command

    assert "--model anthropic/claude-sonnet-4-5" in command


def test_mini_swe_agent_harness_accepts_cli_with_arguments():
    harness = create_harness(
        AgentSpec(
            harness="mini_swe_agent",
            model_name="/models/Qwen3.5-4B",
            settings={"cli": "uv run mini"},
        )
    )

    command = harness.run_steps("fix the bug")[0].command

    assert command.startswith("uv run mini --yolo")
