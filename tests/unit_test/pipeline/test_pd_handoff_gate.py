# SPDX-License-Identifier: Apache-2.0
"""Bounding how many PD handoffs are in flight at once.

Each handoff holds that request's prompt KV on the Prefill card until Decode
acknowledges it. Without a bound the count lands on Prefill's
max_running_requests, which sizes batches rather than leases.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from sglang_omni.config.schema import PDConfig, PDExecution
from sglang_omni.pipeline.stage.runtime import Stage
from tests.unit_test.pipeline.helpers import stage


def _runtime(limit: int | None) -> Stage:
    runtime = Stage.__new__(Stage)
    runtime.pd_execution = PDExecution(
        role="prefill", partner="thinker_decode", max_inflight_handoffs=limit
    )
    return runtime


def test_an_unset_bound_installs_no_gate() -> None:
    """Leaving it unset keeps today's behaviour rather than picking a number."""
    assert _runtime(None)._pd_handoff_gate() is None


def test_a_set_bound_installs_a_semaphore_of_that_size() -> None:
    gate = _runtime(4)._pd_handoff_gate()

    assert gate._value == 4


def test_the_same_gate_is_reused_across_handoffs() -> None:
    """A fresh semaphore per handoff would bound nothing."""
    runtime = _runtime(4)

    assert runtime._pd_handoff_gate() is runtime._pd_handoff_gate()


def test_the_gate_admits_only_its_bound_at_once() -> None:
    async def scenario() -> tuple[int, int]:
        gate = _runtime(2)._pd_handoff_gate()
        live = 0
        peak = 0
        release = asyncio.Event()

        async def held() -> None:
            nonlocal live, peak
            async with gate:
                live += 1
                peak = max(peak, live)
                await release.wait()
                live -= 1

        tasks = [asyncio.create_task(held()) for _ in range(6)]
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        observed = peak
        release.set()
        await asyncio.gather(*tasks)
        return observed, peak

    observed, _ = asyncio.run(scenario())
    assert observed == 2


def test_the_bound_reaches_both_halves_from_one_setting() -> None:
    """It is a property of the pair, so the rewrite copies it to each half."""
    from sglang_omni.config import expand_pd_stages
    from sglang_omni.config.schema import PDStagePlacement

    stages = [
        stage(
            "thinker",
            terminal=True,
            pd_disaggregation=PDConfig(
                prefill=PDStagePlacement(gpu=0),
                decode=PDStagePlacement(gpu=1),
                max_inflight_handoffs=8,
            ),
        )
    ]

    halves = {s.name: s for s in expand_pd_stages(stages, entry_stage="thinker").stages}

    assert halves["thinker_prefill"].pd_execution.max_inflight_handoffs == 8
    assert halves["thinker_decode"].pd_execution.max_inflight_handoffs == 8


def test_an_unset_bound_stays_unset_through_the_rewrite() -> None:
    from sglang_omni.config import expand_pd_stages
    from sglang_omni.config.schema import PDStagePlacement

    stages = [
        stage(
            "thinker",
            terminal=True,
            pd_disaggregation=PDConfig(
                prefill=PDStagePlacement(gpu=0), decode=PDStagePlacement(gpu=1)
            ),
        )
    ]

    halves = {s.name: s for s in expand_pd_stages(stages, entry_stage="thinker").stages}

    assert halves["thinker_prefill"].pd_execution.max_inflight_handoffs is None


def test_cancelling_while_queued_releases_the_prompt_kv() -> None:
    """The lease exists before the wait, so nothing downstream can release it."""

    class _Lease:
        def __init__(self) -> None:
            self.released = 0

        def release(self) -> None:
            self.released += 1

    lease = _Lease()
    handoff = SimpleNamespace(lease=lease)
    runtime = _runtime(1)
    cleared: list[str] = []
    runtime._clear_request_state = cleared.append

    async def scenario() -> None:
        gate = runtime._pd_handoff_gate()
        await gate.acquire()  # the one permit is taken by another handoff
        queued = asyncio.create_task(runtime._send_pd_handoff("req-1", handoff))
        await asyncio.sleep(0)
        queued.cancel()
        try:
            await queued
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())

    assert lease.released == 1
    assert cleared == ["req-1"]


def test_a_handoff_that_never_waited_is_left_to_the_send_path() -> None:
    """With a free permit the send runs and owns the lease, so no double free."""

    class _Lease:
        def __init__(self) -> None:
            self.released = 0

        def release(self) -> None:
            self.released += 1

    lease = _Lease()
    handoff = SimpleNamespace(lease=lease)
    runtime = _runtime(1)
    sent: list[str] = []

    async def fake_send(request_id: str, _handoff: object) -> None:
        sent.append(request_id)

    runtime._send_pd_handoff_now = fake_send

    asyncio.run(runtime._send_pd_handoff("req-2", handoff))

    assert sent == ["req-2"]
    assert lease.released == 0
