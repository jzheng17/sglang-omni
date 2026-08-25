# SPDX-License-Identifier: Apache-2.0
"""The Decode target is resolved per request, not fixed when the stage binds."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest

from sglang_omni.config import expand_pd_stages
from sglang_omni.config.schema import (
    PDConfig,
    PDExecution,
    PDStagePlacement,
    PipelineConfig,
)
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.pd_decode_selection import select_decode_stage
from tests.unit_test.pipeline.helpers import stage


def _pd(prefill_gpu: int, decode_gpu: int) -> PDConfig:
    return PDConfig(
        prefill=PDStagePlacement(gpu=prefill_gpu),
        decode=PDStagePlacement(gpu=decode_gpu),
    )


def test_rewrite_records_the_decode_candidates_on_the_prefill_half() -> None:
    config = PipelineConfig(
        model_path="dummy",
        stages=[stage("thinker", terminal=True, pd_disaggregation=_pd(0, 1))],
    )

    expansion = expand_pd_stages(
        list(config.stages), entry_stage=config.resolved_entry_stage
    )
    halves = {s.name: s for s in expansion.stages}

    assert halves["thinker_prefill"].pd_execution.decode_targets == ("thinker_decode",)
    # The Decode half hands off to nobody, so it carries no candidates.
    assert halves["thinker_decode"].pd_execution.decode_targets == ()


def test_pd_execution_defaults_to_no_candidates() -> None:
    """A PDExecution built without the field stays valid, so older callers work."""
    execution = PDExecution(role="prefill", partner="thinker_decode")

    assert execution.decode_targets == ()


def test_resolver_returns_the_single_candidate() -> None:
    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler._pd_decode_targets = ("thinker_decode",)

    assert scheduler._resolve_decode_stage(SimpleNamespace(rid="a")) == "thinker_decode"
    assert scheduler._resolve_decode_stage(SimpleNamespace(rid="b")) == "thinker_decode"


def test_resolver_reads_the_candidate_list_rather_than_a_fixed_partner() -> None:
    """With more than one candidate the send path needs no change to pick."""
    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler._pd_decode_targets = ("decode_a", "decode_b")
    scheduler._pd_partner = "decode_never"

    chosen = {
        scheduler._resolve_decode_stage(SimpleNamespace(rid=f"req-{i}"))
        for i in range(50)
    }

    assert chosen == {"decode_a", "decode_b"}


def test_resolver_falls_back_to_partner_when_no_candidates_are_set() -> None:
    """A caller that predates the field still resolves, rather than raising."""
    scheduler = OmniScheduler.__new__(OmniScheduler)
    scheduler._pd_partner = "thinker_decode"

    assert scheduler._resolve_decode_stage(SimpleNamespace(rid="a")) == "thinker_decode"


def test_every_rank_reaches_the_same_choice() -> None:
    """The selection depends only on values all Prefill ranks see alike.

    Ranks that disagree scatter one request's KV shards across Decode halves,
    and nothing catches that.
    """
    targets = ("decode_a", "decode_b", "decode_c")

    for rid in (f"req-{i}" for i in range(50)):
        chosen = {select_decode_stage(targets, rid) for _ in range(4)}
        assert len(chosen) == 1


def test_the_choice_spreads_across_the_candidates() -> None:
    """A constant answer would be deterministic but useless as a policy."""
    targets = ("decode_a", "decode_b", "decode_c")

    counts = Counter(select_decode_stage(targets, f"req-{i}") for i in range(600))

    assert set(counts) == set(targets)
    assert min(counts.values()) > 100


def test_one_candidate_is_returned_without_hashing() -> None:
    assert select_decode_stage(("only",), "any-request") == "only"


def test_no_candidates_is_an_error() -> None:
    with pytest.raises(ValueError, match="no decode targets"):
        select_decode_stage((), "req")
