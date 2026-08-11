from __future__ import annotations

import asyncio

from polar.trajectory.builder.prefix_merging import PrefixMergingBuilder
from polar.trajectory.models import CompletionRecord, CompletionSession


def _completion(completion_id, messages, prompt_ids, response_ids, text):
    return CompletionRecord(
        completion_id=completion_id,
        original_request={"messages": messages},
        response={"choices": [{
            "input_token_ids": prompt_ids,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
            "logprobs": {"content": [
                {"token_id": token_id, "logprob": -0.1}
                for token_id in response_ids
            ]},
        }]},
        metadata={"completion_id": completion_id},
    )


def test_merges_retained_miniswe_qwen_chain() -> None:
    initial = [{"role": "user", "content": "Fix the bug"}]
    first_response = {"role": "assistant", "content": "inspect"}
    session = CompletionSession(
        session_id="session-1",
        completions=[
            _completion(
                "completion-1", initial, [10, 11, 90, 91], [20, 99], "inspect"
            ),
            _completion(
                "completion-2",
                [
                    *initial,
                    first_response,
                    {
                        "role": "user",
                        "content": "<returncode>0</returncode>\n<output>ok</output>",
                    },
                ],
                [10, 11, 20, 99, 30, 31],
                [40, 99],
                "fix",
            ),
        ],
    )

    trajectory = asyncio.run(
        PrefixMergingBuilder(end_of_turn_token_id=99).build(session)
    )

    assert trajectory.metadata["record_count"] == 2
    assert trajectory.metadata["trace_count"] == 1
    assert trajectory.metadata["reconstruction_stats"] == {
        "chains_total": 1,
        "chains_reconstructed_full": 1,
        "chains_reconstructed_truncated": 0,
        "completions_total": 2,
        "completions_merged": 2,
    }
    trace = trajectory.traces[0]
    assert trace.prompt_messages == initial
    assert trace.prompt_ids == [10, 11, 90, 91]
    assert trace.response_ids == [20, 99, 30, 31, 40, 99]
    assert trace.loss_mask == [1, 1, 0, 0, 1, 1]
    assert len(trace.metadata["completion_metadata"]) == 2


def test_does_not_merge_when_response_is_not_at_divergence() -> None:
    initial = [{"role": "user", "content": "Fix the bug"}]
    session = CompletionSession(
        session_id="session-1",
        completions=[
            _completion(
                "completion-1", initial, [10, 11, 90, 91], [20, 99], "inspect"
            ),
            _completion(
                "completion-2",
                [*initial, {"role": "assistant", "content": "inspect"}],
                [10, 11, 777, 99, 30],
                [40, 99],
                "fix",
            ),
        ],
    )

    trajectory = asyncio.run(
        PrefixMergingBuilder(end_of_turn_token_id=99).build(session)
    )

    assert trajectory.metadata["trace_count"] == 2
