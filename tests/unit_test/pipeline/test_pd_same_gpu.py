# SPDX-License-Identifier: Apache-2.0
"""Both PD halves may share one GPU."""

from __future__ import annotations

import pytest

from sglang_omni.config import expand_pd_stages
from sglang_omni.config.schema import PDConfig, PDStagePlacement, PipelineConfig
from tests.unit_test.pipeline.helpers import stage


_CAP = {"max_total_tokens": 32768}


def _config(prefill_gpu, decode_gpu, budget=_CAP, **stage_kwargs) -> PipelineConfig:
    return PipelineConfig(
        model_path="dummy",
        stages=[
            stage(
                "thinker",
                terminal=True,
                pd_disaggregation=PDConfig(
                    prefill=PDStagePlacement(gpu=prefill_gpu, server_args=dict(budget)),
                    decode=PDStagePlacement(gpu=decode_gpu, server_args=dict(budget)),
                ),
                **stage_kwargs,
            )
        ],
    )


def test_one_gpu_holds_both_halves() -> None:
    config = _config(0, 0)

    halves = {
        s.name: s
        for s in expand_pd_stages(
            list(config.stages), entry_stage=config.resolved_entry_stage
        ).stages
    }

    assert halves["thinker_prefill"].gpu == 0
    assert halves["thinker_decode"].gpu == 0


def test_the_halves_stay_in_separate_processes_on_one_gpu() -> None:
    """Sharing a device is not sharing a process; PD needs the second split."""
    config = _config(0, 0)

    halves = {
        s.name: s
        for s in expand_pd_stages(
            list(config.stages), entry_stage=config.resolved_entry_stage
        ).stages
    }

    assert halves["thinker_prefill"].process == "thinker_prefill"
    assert halves["thinker_decode"].process == "thinker_decode"
    assert halves["thinker_prefill"].process != halves["thinker_decode"].process


def test_separate_gpus_need_no_explicit_budget() -> None:
    """Each half then sizes from its own card, so a fraction is order-safe."""
    config = _config(0, 1, budget={})

    halves = {
        s.name: s
        for s in expand_pd_stages(
            list(config.stages), entry_stage=config.resolved_entry_stage
        ).stages
    }

    assert halves["thinker_prefill"].gpu == 0
    assert halves["thinker_decode"].gpu == 1


def test_overlapping_gpu_lists_are_allowed() -> None:
    """A partial overlap is a placement choice, and it shares a card."""
    config = PipelineConfig(
        model_path="dummy",
        stages=[
            stage(
                "thinker",
                terminal=True,
                tp_size=2,
                pd_disaggregation=PDConfig(
                    prefill=PDStagePlacement(gpu=[0, 1], server_args=dict(_CAP)),
                    decode=PDStagePlacement(gpu=[1, 2], server_args=dict(_CAP)),
                ),
            )
        ],
    )

    halves = {
        s.name: s
        for s in expand_pd_stages(
            list(config.stages), entry_stage=config.resolved_entry_stage
        ).stages
    }

    assert halves["thinker_prefill"].gpu == [0, 1]
    assert halves["thinker_decode"].gpu == [1, 2]


def test_a_gpu_list_must_still_match_tp_size() -> None:
    with pytest.raises(ValueError, match="entries but tp_size"):
        PipelineConfig(
            model_path="dummy",
            stages=[
                stage(
                    "thinker",
                    terminal=True,
                    tp_size=2,
                    pd_disaggregation=PDConfig(
                        prefill=PDStagePlacement(gpu=[0]),
                        decode=PDStagePlacement(gpu=[1, 2]),
                    ),
                )
            ],
        )


def test_sharing_a_gpu_requires_an_explicit_token_budget() -> None:
    """`mem_fraction_static` is order-dependent when two halves share a card.

    The halves race for the same startup lock and the winner varies between
    runs, so whichever loads first sees far more free memory than the other.
    Measured on one H200: the first half took 780,987 KV tokens and the second
    failed to start.
    """
    with pytest.raises(ValueError, match="max_total_tokens"):
        _config(0, 0, budget={})


def test_the_budget_is_required_on_both_halves() -> None:
    with pytest.raises(ValueError, match="decode.server_args"):
        PipelineConfig(
            model_path="dummy",
            stages=[
                stage(
                    "thinker",
                    terminal=True,
                    pd_disaggregation=PDConfig(
                        prefill=PDStagePlacement(gpu=0, server_args=dict(_CAP)),
                        decode=PDStagePlacement(gpu=0),
                    ),
                )
            ],
        )
