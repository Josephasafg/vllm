# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for logprob clamping helpers.

These guard against returning non-JSON-compliant floats (``NaN``/``-inf``)
in the OpenAI-compatible API responses. FastAPI/Starlette serialize with
``allow_nan=False``, so leaking a ``NaN`` raises
``ValueError: Out of range float values are not JSON compliant`` and fails
the whole response.
"""
import json
import math

from vllm.entrypoints.openai.engine.serving import (
    clamp_logprob,
    clamp_prompt_logprobs,
)
from vllm.logprobs import Logprob


def test_clamp_logprob_handles_nan():
    # The critical case: NaN fails every comparison, so a plain
    # max(logprob, -9999.0) would pass it through unchanged.
    assert clamp_logprob(float("nan")) == -9999.0


def test_clamp_logprob_handles_neg_inf():
    assert clamp_logprob(float("-inf")) == -9999.0


def test_clamp_logprob_floors_overly_negative():
    assert clamp_logprob(-20000.0) == -9999.0


def test_clamp_logprob_passes_through_normal_values():
    assert clamp_logprob(-0.5) == -0.5
    assert clamp_logprob(0.0) == 0.0


def test_clamp_logprob_output_is_json_compliant():
    for value in (float("nan"), float("-inf"), -20000.0, -0.5, 0.0):
        clamped = clamp_logprob(value)
        # Must not raise.
        json.dumps({"logprob": clamped}, allow_nan=False)


def test_clamp_prompt_logprobs_sanitizes_nan_and_neg_inf():
    prompt_logprobs = [
        None,
        {
            1: Logprob(logprob=float("nan")),
            2: Logprob(logprob=float("-inf")),
            3: Logprob(logprob=-0.25),
        },
    ]
    clamped = clamp_prompt_logprobs(prompt_logprobs)
    assert clamped is not None
    values = clamped[1]
    assert values[1].logprob == -9999.0
    assert values[2].logprob == -9999.0
    assert values[3].logprob == -0.25
    for logprob_dict in clamped:
        if logprob_dict is None:
            continue
        for lp in logprob_dict.values():
            assert math.isfinite(lp.logprob)
