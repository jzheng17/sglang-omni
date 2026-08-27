# SPDX-License-Identifier: Apache-2.0
"""Finding the Decode instances a Prefill half may send to.

Replicating the Decode process renames its stage ``<name>@r0``, ``<name>@r1``,
and that renaming happens after the PD graph rewrite, so the target list
cannot be fixed when the halves are compiled. It is derived at bind time from
the endpoints, which are named per instance.
"""

from __future__ import annotations

from sglang_omni.pipeline.stage.runtime import _decode_targets


def _endpoints(*names: str) -> dict[str, tuple[str, ...]]:
    return {name: (f"ipc:///run/{name}",) for name in names}


def test_an_unreplicated_decode_half_is_its_own_only_target() -> None:
    """The single-instance case must stay a constant choice."""
    eps = _endpoints("thinker_prefill", "thinker_decode")

    assert _decode_targets(eps, "thinker_decode") == ("thinker_decode",)


def test_every_replica_becomes_a_target() -> None:
    eps = _endpoints("thinker_prefill", "thinker_decode@r0", "thinker_decode@r1")

    assert _decode_targets(eps, "thinker_decode") == (
        "thinker_decode@r0",
        "thinker_decode@r1",
    )


def test_the_order_does_not_depend_on_dictionary_order() -> None:
    """Prefill ranks that disagree scatter one request's shards."""
    forward = _endpoints("thinker_decode@r0", "thinker_decode@r1")
    reverse = _endpoints("thinker_decode@r1", "thinker_decode@r0")

    assert _decode_targets(forward, "thinker_decode") == _decode_targets(
        reverse, "thinker_decode"
    )


def test_another_stage_is_not_a_target() -> None:
    """A prefix match must not pick up an unrelated stage."""
    eps = _endpoints("thinker_decode", "thinker_decoder_aux", "talker_decode")

    assert _decode_targets(eps, "thinker_decode") == ("thinker_decode",)


def test_no_endpoints_yields_no_targets() -> None:
    assert _decode_targets({}, "thinker_decode") == ()
