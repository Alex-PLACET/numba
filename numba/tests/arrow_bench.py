"""Benchmarks for the Arrow PyCapsule bridge.

Run with::

    python -m numba.tests.arrow_bench
"""

import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import polars as pl

from numba import jit
from numba.experimental.arrow import arrow_jit, from_arrow

N = 2_000_000
SMALL = 1024
REPEATS = 5


def _fastest(call, repeats=REPEATS):
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - start)
    return best


def _per_call(call, calls=2000):
    start = time.perf_counter()
    for _ in range(calls):
        call()
    return (time.perf_counter() - start) / calls


@jit
def _sum_numpy(arr):
    total = 0
    for i in range(len(arr)):
        total += arr[i]
    return total


@arrow_jit
def _sum_arrow(arr):
    total = 0
    for i in range(len(arr)):
        total += arr[i]
    return total


def main():
    values = np.arange(N, dtype=np.int64)
    arrow = pa.array(values)
    series = pl.Series("a", values)

    _sum_numpy(values)
    _sum_arrow(arrow)
    _sum_arrow(series)

    gb = N * values.itemsize / 1e9
    rows = [
        ("numpy @jit", _fastest(lambda: _sum_numpy(values))),
        ("arrow_jit(pa.Array)", _fastest(lambda: _sum_arrow(arrow))),
        ("arrow_jit(pl.Series)", _fastest(lambda: _sum_arrow(series))),
        ("pa.compute.sum", _fastest(lambda: pc.sum(arrow))),
    ]
    print(f"sum of {N} int64 ({gb:.2f} GB), best of {REPEATS}")
    for name, seconds in rows:
        print(f"  {name:22s}{seconds * 1e3:9.3f} ms{gb / seconds:8.2f} GB/s")

    small = np.arange(SMALL, dtype=np.int64)
    small_arrow = pa.array(small)
    small_view = from_arrow(small_arrow)
    _sum_numpy(small)
    _sum_arrow(small_view)
    print(f"\nper call, {SMALL} int64")
    for name, call in [
        ("numpy @jit", lambda: _sum_numpy(small)),
        ("jit(pre-made view)", lambda: _sum_arrow(small_view)),
        ("arrow_jit(pa.Array)", lambda: _sum_arrow(small_arrow)),
        ("__arrow_c_array__ only", lambda: small_arrow.__arrow_c_array__()),
        ("from_arrow only", lambda: from_arrow(small_arrow)),
    ]:
        print(f"  {name:22s}{_per_call(call) * 1e6:9.3f} us")


if __name__ == "__main__":
    main()
