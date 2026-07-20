"""Mini-SWE-Agent harness."""

from __future__ import annotations

import shlex

from polar.agent.base import BaseHarness
from polar.agent.models import AgentSpec
from polar.runtime.base import RUNTIME_AGENT_LOG_DIR, RUNTIME_SESSION_DIR
from polar.runtime.models import ExecInput


class MiniSweAgentHarness(BaseHarness):
    """Run Mini-SWE-Agent in the benchmark workspace.

    The harness relies on Mini-SWE-Agent editing files in the runtime workdir.
    ProRL's existing evaluator then collects ``git diff`` from that workdir, so
    no Mini-SWE-specific reward or patch extraction code is needed here.
    """

    def __init__(self, agent_spec: AgentSpec) -> None:
        super().__init__(agent_spec)
        settings = self.settings
        self._cli = str(settings.get("cli", "mini"))
        self._config = settings.get("config")
        self._model_class = settings.get("model_class")
        self._agent_class = settings.get("agent_class")
        self._environment_class = settings.get("environment_class")
        self._cost_limit = settings.get("cost_limit", 0)
        self._output = str(settings.get("output", f"{RUNTIME_SESSION_DIR}/mini-swe-agent.json"))
        self._exit_immediately = bool(settings.get("exit_immediately", True))

    def run_steps(self, instruction: str) -> list[ExecInput]:
        env: dict[str, str] = {**self.env}
        model = _litellm_model_name(self.model_name)

        args = [
            *[shlex.quote(part) for part in shlex.split(self._cli)],
            "--yolo",
            "--model",
            shlex.quote(model),
            "--task",
            shlex.quote(instruction),
            "--cost-limit",
            shlex.quote(str(self._cost_limit)),
            "--output",
            shlex.quote(self._output),
        ]
        if self._exit_immediately:
            args.append("--exit-immediately")
        for config in _as_list(self._config):
            args.extend(["--config", shlex.quote(str(config))])
        if self._model_class:
            args.extend(["--model-class", shlex.quote(str(self._model_class))])
        if self._agent_class:
            args.extend(["--agent-class", shlex.quote(str(self._agent_class))])
        if self._environment_class:
            args.extend(["--environment-class", shlex.quote(str(self._environment_class))])

        return [
            ExecInput(
                command=(
                    " ".join(args)
                    + f" 2>&1 </dev/null | tee {RUNTIME_AGENT_LOG_DIR}/mini-swe-agent.txt"
                ),
                env=env,
            )
        ]


def _as_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _litellm_model_name(model_name: str | None) -> str:
    """Default Mini-SWE to LiteLLM's OpenAI-compatible provider.

    The Polar gateway exposes an OpenAI-compatible endpoint and task runtimes
    receive ``OPENAI_BASE_URL``/``OPENAI_API_KEY``. Mini-SWE-Agent reaches that
    endpoint through LiteLLM, whose OpenAI provider expects an ``openai/`` model
    prefix. If a caller already provided a provider prefix or model alias, keep
    it unchanged.
    """
    model = model_name or "gpt-5.4"
    provider_prefixes = (
        "openai/",
        "anthropic/",
        "azure/",
        "bedrock/",
        "cohere/",
        "gemini/",
        "google/",
        "hosted_vllm/",
        "ollama/",
        "openrouter/",
        "together_ai/",
        "vertex_ai/",
        "vllm/",
    )
    if model.startswith(provider_prefixes):
        return model
    return f"openai/{model}"

