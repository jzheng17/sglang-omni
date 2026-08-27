#!/usr/bin/env python3
"""Catch two threads receiving the same KV slot from one allocator.

alloc() is a read-modify-write over free_pages with no lock. This records
which thread handed out which slots and reports any slot returned twice
while a previous holder still has it -- the direct observation the
root-cause argument is missing.
"""
import io
import sys

OLD = """        select_index = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        return select_index
"""
NEW = """        select_index = self.free_pages[:need_size]
        self.free_pages = self.free_pages[need_size:]
        _allocrace_record(select_index)
        return select_index
"""

HELPER = '''

# --- ALLOCRACE (diagnostic build only) -------------------------------------
import logging as _ar_logging
import threading as _ar_threading

_ar_logger = _ar_logging.getLogger("allocrace")
_ar_lock = _ar_threading.Lock()
_ar_outstanding = {}
_ar_seq = [0]


def _allocrace_record(slots) -> None:
    """Report a slot handed out again while a previous holder still has it."""
    try:
        ids = slots.tolist()
    except Exception:
        return
    tname = _ar_threading.current_thread().name
    with _ar_lock:
        _ar_seq[0] += 1
        seq = _ar_seq[0]
        for s in ids:
            prev = _ar_outstanding.get(s)
            if prev is not None and prev[0] != tname:
                _ar_logger.warning(
                    "ALLOCRACE slot=%d again_by=%s seq=%d prev_by=%s prev_seq=%d n=%d",
                    s, tname, seq, prev[0], prev[1], len(ids),
                )
            _ar_outstanding[s] = (tname, seq)


def _allocrace_release(slots) -> None:
    try:
        ids = slots.tolist()
    except Exception:
        return
    with _ar_lock:
        for s in ids:
            _ar_outstanding.pop(s, None)
'''

target = sys.argv[1]
s = io.open(target, encoding="utf-8").read()
if "_allocrace_record" in s:
    print("already patched")
    raise SystemExit(0)
if OLD not in s:
    print("alloc body not found")
    raise SystemExit(1)
s = s.replace(OLD, NEW, 1) + HELPER
io.open(target, "w", encoding="utf-8").write(s)
print("patched", target)
