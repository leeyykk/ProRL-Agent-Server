"""``swebench_harness`` evaluator — grade via the SWE-Bench / SWE-Gym harness.

Use this strategy when you are reproducing SWE-Bench-style benchmarks and
want per-instance grading that matches what the harness reports. The harness
owns the test spec, eval script, and resolved/unresolved classification.

Config schema
-------------

Passed through :class:`polar.trajectory.models.EvaluatorSpec.config`:

- ``instance`` *(dict, required)* — the raw SWE-Bench / SWE-Gym instance dict
  (at minimum ``instance_id``; ``base_commit``/``version`` are derived when
  missing). The grader uses it to build the test spec and eval script.
- ``repo_dir`` *(str, default ``/testbed``)* — repository root inside the
  runtime. Must contain the checked-out instance.
- ``patch_command`` *(str, default ``git diff --binary --submodule=diff``)* —
  command run *in the source runtime* to emit the generated patch.
- ``apply_timeout`` *(float, default 60)* — seconds allowed for patch
  extraction and ``git apply`` on the fresh eval runtime.
- ``test_timeout`` *(float, default 1200)* — seconds allowed for the
  harness-generated ``eval.sh`` run.
- ``exclude_patterns`` *(list[str])* — extra globs appended to the default
  skip list (``__pycache__``, ``*.pyc``, ``.pytest_cache`` etc.) when
  filtering the extracted diff.

The generated patch is captured, filtered, applied on the fresh eval runtime,
and then the harness-produced ``eval.sh`` is executed. Grading is delegated
to ``swegym.harness.grading.get_eval_report`` (or ``swebench`` when swegym is
not installed); ``resolved=True`` → ``outcome_reward=1.0``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shlex
from typing import Any

from polar.runtime.base import BaseRuntime
from polar.trajectory.evaluator._patch_utils import (
    BasePatchEvaluator,
    bounded_timeout,
    shell_quote,
)


class SwebenchHarnessEvaluator(BasePatchEvaluator):
    """Grades generated patches using the SWE-Bench / SWE-Gym harness."""

    MODE = "swebench_harness"

    def __init__(
        self,
        *,
        instance: dict[str, Any],
        repo_dir: str = "/testbed",
        patch_command: str | None = None,
        apply_timeout: float = 60.0,
        test_timeout: float = 1200.0,
        exclude_patterns: list[str] | None = None,
    ) -> None:
        super().__init__(
            repo_dir=repo_dir,
            patch_command=patch_command,
            apply_timeout=apply_timeout,
            test_timeout=test_timeout,
            exclude_patterns=exclude_patterns,
        )
        if not instance:
            raise ValueError("swebench_harness requires a non-empty 'instance' config")
        self.instance = instance

    async def _grade(
        self,
        *,
        runtime: BaseRuntime,
        patch: str,
        host_session_dir: Path,
        log_dir: Path,
        env: dict[str, str],
        timeout_cap: float | None,
    ) -> tuple[dict[str, Any], Path]:
        instance = dict(self.instance)
        instance_id = str(instance["instance_id"]).lower()
        instance["instance_id"] = instance_id
        if "version" not in instance and "base_commit" in instance:
            instance["version"] = instance["base_commit"]

        test_spec, get_eval_report = _load_harness(instance)
        eval_script_host = host_session_dir / "eval.sh"
        eval_script_host.write_text(test_spec.eval_script)

        # Place the log inside an instance_id-named directory so that
        # swegym/swebench get_logs_eval can parse the repo from the path.
        instance_log_dir = log_dir / instance_id
        instance_log_dir.mkdir(parents=True, exist_ok=True)
        combined_path = instance_log_dir / "test_output.txt"
        result = await runtime.exec(
            f"/bin/bash {shell_quote(f'{runtime.runtime_session_dir}/eval.sh')}",
            cwd=self.repo_dir,
            env=env,
            timeout_sec=bounded_timeout(self.test_timeout, timeout_cap),
        )
        if result.return_code == -1:
            raise TimeoutError("swebench_harness evaluation timed out")

        combined_path.write_text((result.stdout or "") + (result.stderr or ""))
        prediction = {"model_patch": patch, "instance_id": instance_id}
        if get_eval_report is None:
            grading_report = _grade_simple_swegym_run(
                test_spec=test_spec,
                prediction=prediction,
                return_code=result.return_code,
            )
        else:
            grading_report = _grade_harness_run(
                get_eval_report,
                test_spec=test_spec,
                prediction=prediction,
                log_path=combined_path,
            )
        report = grading_report[instance_id]
        return (
            {
                "empty_generation": False,
                "resolved": report.get("resolved", False),
                "failed_apply_patch": False,
                "error_eval": False,
                "test_timeout": False,
                "exit_code": result.return_code,
                "grading_report": report,
            },
            combined_path,
        )


@dataclass(frozen=True)
class _SimpleSweGymSpec:
    """Minimal SWE-Gym test spec for repos unsupported by PyPI swebench."""

    instance_id: str
    repo: str
    version: str
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    test_patch: str

    @property
    def eval_script(self) -> str:
        tests = self.FAIL_TO_PASS + self.PASS_TO_PASS
        tests_arg = " ".join(shlex.quote(test) for test in tests)
        pytest_command = f"python -m pytest -q {tests_arg}" if tests_arg else "python -m pytest -q"
        patch = self.test_patch.rstrip("\n")
        return "\n".join(
            [
                "#!/bin/bash",
                "set -uxo pipefail",
                "cd /testbed",
                "cat > /tmp/polar_swegym_test.patch <<'POLAR_SWEGYM_TEST_PATCH'",
                patch,
                "POLAR_SWEGYM_TEST_PATCH",
                "if [ -s /tmp/polar_swegym_test.patch ]; then",
                "  if git apply -v /tmp/polar_swegym_test.patch; then",
                "    echo __POLAR_SIMPLE_SWEGYM_TEST_PATCH_APPLY=git_apply__",
                "  elif patch --batch --fuzz=5 -p1 -i /tmp/polar_swegym_test.patch; then",
                "    echo __POLAR_SIMPLE_SWEGYM_TEST_PATCH_APPLY=patch__",
                "  else",
                "    echo __POLAR_SIMPLE_SWEGYM_TEST_PATCH_APPLY=failed__",
                "    exit 2",
                "  fi",
                "fi",
                "set +e",
                pytest_command,
                "status=$?",
                'echo "__POLAR_SIMPLE_SWEGYM_PYTEST_STATUS=${status}__"',
                "exit ${status}",
                "",
            ]
        )


def _load_harness(instance: dict[str, Any]) -> tuple[Any, Any]:
    try:
        from swegym.harness.grading import get_eval_report
        from swegym.harness.test_spec import make_test_spec
        return make_test_spec(instance), get_eval_report
    except ModuleNotFoundError:
        pass
    except KeyError:
        fallback_spec = _maybe_simple_swegym_spec(instance)
        if fallback_spec is not None:
            return fallback_spec, None
        raise

    try:
        from swebench.harness.grading import get_eval_report
        try:
            from swebench.harness.test_spec.test_spec import make_test_spec
        except ModuleNotFoundError:
            from swebench.harness.test_spec import make_test_spec
        return make_test_spec(instance), get_eval_report
    except KeyError:
        fallback_spec = _maybe_simple_swegym_spec(instance)
        if fallback_spec is not None:
            return fallback_spec, None
        raise
    except ModuleNotFoundError:
        fallback_spec = _maybe_simple_swegym_spec(instance)
        if fallback_spec is not None:
            return fallback_spec, None
        raise


def _maybe_simple_swegym_spec(instance: dict[str, Any]) -> _SimpleSweGymSpec | None:
    test_patch = instance.get("test_patch")
    if not test_patch:
        return None
    fail_to_pass = _instance_list(instance, "FAIL_TO_PASS")
    pass_to_pass = _instance_list(instance, "PASS_TO_PASS")
    if not fail_to_pass and not pass_to_pass:
        return None
    return _SimpleSweGymSpec(
        instance_id=str(instance["instance_id"]).lower(),
        repo=str(instance.get("repo", "")),
        version=str(instance.get("version", instance.get("base_commit", ""))),
        FAIL_TO_PASS=fail_to_pass,
        PASS_TO_PASS=pass_to_pass,
        test_patch=str(test_patch),
    )


def _instance_list(instance: dict[str, Any], key: str) -> list[str]:
    value = instance.get(key) or []
    if isinstance(value, str):
        value = json.loads(value)
    return [str(item) for item in value]


def _grade_simple_swegym_run(
    *,
    test_spec: _SimpleSweGymSpec,
    prediction: dict[str, Any],
    return_code: int,
) -> dict[str, Any]:
    instance_id = str(prediction["instance_id"])
    resolved = return_code == 0
    return {
        instance_id: {
            "patch_exists": bool(prediction.get("model_patch")),
            "patch_successfully_applied": True,
            "resolved": resolved,
            "tests_status": {
                "FAIL_TO_PASS": {
                    "success": test_spec.FAIL_TO_PASS if resolved else [],
                    "failure": [] if resolved else test_spec.FAIL_TO_PASS,
                },
                "PASS_TO_PASS": {
                    "success": test_spec.PASS_TO_PASS if resolved else [],
                    "failure": [] if resolved else test_spec.PASS_TO_PASS,
                },
            },
        }
    }


def _grade_harness_run(
    get_eval_report: Any,
    *,
    test_spec: Any,
    prediction: dict[str, Any],
    log_path: Path,
) -> dict[str, Any]:
    try:
        return get_eval_report(
            test_spec=test_spec,
            prediction=prediction,
            log_path=str(log_path),
            include_tests_status=True,
        )
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        return get_eval_report(
            test_spec=test_spec,
            prediction=prediction,
            test_log_path=str(log_path),
            include_tests_status=True,
        )
