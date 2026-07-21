from polar.trajectory.evaluator.swebench_harness import (
    _rewrite_eval_script_repo_dir,
)


def test_rewrites_canonical_testbed_to_configured_repo() -> None:
    script = "cd /testbed\ngit config --add safe.directory /testbed\npytest /testbed/tests\n"

    rewritten = _rewrite_eval_script_repo_dir(
        script, "/polar/session/workspace"
    )

    assert "/testbed" not in rewritten
    assert rewritten.count("/polar/session/workspace") == 3


def test_preserves_canonical_testbed_repo() -> None:
    script = "cd /testbed\n"

    assert _rewrite_eval_script_repo_dir(script, "/testbed") == script
