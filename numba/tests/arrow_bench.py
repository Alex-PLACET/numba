"""Benchmarks: 1M int64 arrays through Numba, pyarrow and Polars.

Cases: sum without nulls, sum with 1% nulls, build, build preallocated.

Run with::

    python -m numba.tests.arrow_bench
"""

import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import polars as pl

from numba import jit, njit
from numba.experimental.arrow import arrow_jit, is_valid

N = 1_000_000
REPEATS = 7


def _fastest(call, repeats=REPEATS):
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - start)
    return best


def _sum_plain(a):
    total = 0
    for i in range(len(a)):
        total += a[i]
    return total


def _sum_masked(a, mask):
    total = 0
    for i in range(len(a)):
        if mask[i]:
            total += a[i]
    return total


def _sum_arrow_plain(arr):
    total = 0
    for i in range(len(arr)):
        total += arr[i]
    return total


def _sum_arrow_masked(arr):
    total = 0
    for i in range(len(arr)):
        if is_valid(arr, i):
            total += arr[i]
    return total


def _build(n):
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        out[i] = i
    return out


def _fill(out):
    for i in range(len(out)):
        out[i] = i


_jit_sum = jit(_sum_plain)
_njit_sum = njit(_sum_plain)
_jit_sum_masked = jit(_sum_masked)
_njit_sum_masked = njit(_sum_masked)
_arrow_sum = arrow_jit(_sum_arrow_plain)
_arrow_sum_masked = arrow_jit(_sum_arrow_masked)
_jit_build = jit(_build)
_njit_build = njit(_build)
_jit_fill = jit(_fill)
_njit_fill = njit(_fill)


def _report(title, rows, note=None):
    print(f"\n{title}  ({N} int64, best of {REPEATS})")
    if note is not None:
        print(f"  {note}")
    for name, seconds in rows:
        if seconds is None:
            print(f"  {name:26s}{'n/a':>12s}")
        else:
            print(f"  {name:26s}{seconds * 1e3:8.3f} ms"
                  f"{N / seconds / 1e9:8.2f} Gelem/s")


def main():
    values = np.arange(N, dtype=np.int64)
    mask = (np.arange(N) % 100) != 0
    arrow_plain = pa.array(values)
    arrow_masked = pa.array(values, mask=~mask)
    series_plain = pl.Series(arrow_plain)
    series_masked = pl.Series(arrow_masked)

    expected = int(values[mask].sum())
    assert _jit_sum_masked(values, mask) == expected
    assert _njit_sum_masked(values, mask) == expected
    assert _arrow_sum_masked(arrow_masked) == expected
    assert _arrow_sum_masked(series_masked) == expected
    assert pc.sum(arrow_masked).as_py() == expected
    assert series_masked.sum() == expected

    out = np.empty(N, dtype=np.int64)
    _jit_sum(values)
    _njit_sum(values)
    _jit_sum_masked(values, mask)
    _njit_sum_masked(values, mask)
    _arrow_sum(arrow_plain)
    _arrow_sum(series_plain)
    _arrow_sum_masked(arrow_masked)
    _arrow_sum_masked(series_masked)
    pc.sum(arrow_plain)
    pc.sum(arrow_masked)
    series_plain.sum()
    series_masked.sum()
    _jit_build(N)
    _njit_build(N)
    _jit_fill(out)
    _njit_fill(out)

    _report("sum, no nulls", [
        ("numpy @jit", _fastest(lambda: _jit_sum(values))),
        ("numpy @njit", _fastest(lambda: _njit_sum(values))),
        ("arrow @jit (pyarrow)", _fastest(lambda: _arrow_sum(arrow_plain))),
        ("arrow @jit (polars)", _fastest(lambda: _arrow_sum(series_plain))),
        ("pyarrow pc.sum", _fastest(lambda: pc.sum(arrow_plain))),
        ("polars Series.sum", _fastest(lambda: series_plain.sum())),
    ])

    _report("sum, 1% nulls", [
        ("numpy @jit", _fastest(lambda: _jit_sum_masked(values, mask))),
        ("numpy @njit", _fastest(lambda: _njit_sum_masked(values, mask))),
        ("arrow @jit (pyarrow)", _fastest(
            lambda: _arrow_sum_masked(arrow_masked))),
        ("arrow @jit (polars)", _fastest(
            lambda: _arrow_sum_masked(series_masked))),
        ("pyarrow pc.sum", _fastest(lambda: pc.sum(arrow_masked))),
        ("polars Series.sum", _fastest(lambda: series_masked.sum())),
    ])

    _report("build", [
        ("numpy @jit empty+fill", _fastest(lambda: _jit_build(N))),
        ("numpy @njit empty+fill", _fastest(lambda: _njit_build(N))),
        ("pyarrow pa.array(arange)", _fastest(
            lambda: pa.array(np.arange(N, dtype=np.int64)))),
        ("polars int_range", _fastest(
            lambda: pl.int_range(0, N, eager=True))),
    ], note="pyarrow/polars rows generate the values, as the numba rows do")

    _report("build, preallocated", [
        ("numpy @jit fill", _fastest(lambda: _jit_fill(out))),
        ("numpy @njit fill", _fastest(lambda: _njit_fill(out))),
        ("pyarrow", None),
        ("polars", None),
    ], note="Arrow arrays and Polars Series are immutable; no in-place build")


if __name__ == "__main__":
    main()
