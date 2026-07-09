"""Small optional NVTX wrapper used by Nsight Systems profiling."""

from __future__ import annotations

import ctypes
import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator


def _enabled() -> bool:
    return os.environ.get("PRORL_ENABLE_NVTX", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


@lru_cache(maxsize=1)
def _nvtx_lib() -> ctypes.CDLL | None:
    if not _enabled():
        return None
    for name in ("libnvToolsExt.so", "libnvToolsExt.so.1"):
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        lib.nvtxRangePushA.argtypes = [ctypes.c_char_p]
        lib.nvtxRangePushA.restype = ctypes.c_int
        lib.nvtxRangePop.argtypes = []
        lib.nvtxRangePop.restype = ctypes.c_int
        return lib
    return None


def range_push(name: str) -> bool:
    lib = _nvtx_lib()
    if lib is None:
        return False
    lib.nvtxRangePushA(name.encode("utf-8", errors="replace"))
    return True


def range_pop() -> None:
    lib = _nvtx_lib()
    if lib is not None:
        lib.nvtxRangePop()


@contextmanager
def nvtx_range(name: str) -> Iterator[None]:
    pushed = range_push(name)
    try:
        yield
    finally:
        if pushed:
            range_pop()
