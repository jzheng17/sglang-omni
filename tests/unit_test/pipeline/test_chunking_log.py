# SPDX-License-Identifier: Apache-2.0
"""Chunking state is logged on transition, not on every scheduling call."""

from __future__ import annotations

import logging
from types import SimpleNamespace

from sglang_omni.scheduling.sglang_backend.prefill import PrefillManager

_LOGGER = "sglang_omni.scheduling.sglang_backend.prefill"


def _manager(*, disabled_last: bool = False) -> PrefillManager:
    manager = PrefillManager.__new__(PrefillManager)
    manager._chunking_disabled_last = disabled_last
    manager.chunked_req = None
    manager.waiting_queue = [
        SimpleNamespace(rid="r0", _input_embeds_are_projected=True),
        SimpleNamespace(rid="r1", _input_embeds_are_projected=False),
    ]
    return manager


def test_a_steady_state_logs_once_not_once_per_call(caplog) -> None:
    """An image workload holds this true for its whole duration."""
    manager = _manager()

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        for _ in range(5):
            manager._log_chunking_transition(True)

    assert len(caplog.records) == 1
    assert "Disable chunked prefill" in caplog.records[0].message


def test_the_line_names_the_requests_that_caused_it(caplog) -> None:
    """A count would not tell the operator which request to look at."""
    manager = _manager()

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        manager._log_chunking_transition(True)

    assert "['r0']" in caplog.records[0].getMessage()


def test_returning_to_chunked_prefill_is_also_logged(caplog) -> None:
    """Otherwise the log shows chunking going off and never coming back."""
    manager = _manager(disabled_last=True)

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        for _ in range(5):
            manager._log_chunking_transition(False)

    assert len(caplog.records) == 1
    assert "Re-enable chunked prefill" in caplog.records[0].message


def test_an_unchanged_state_logs_nothing(caplog) -> None:
    manager = _manager()

    with caplog.at_level(logging.INFO, logger=_LOGGER):
        manager._log_chunking_transition(False)

    assert caplog.records == []
