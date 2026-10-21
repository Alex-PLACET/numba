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
from numba.experimental.arrow import arrow_jit, from_arrow

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
        value = arr[i]
        if value is not None:
            total += value
    return total


def _sum_byte_aware(arr):
    total = 0
    data = arr.data
    valid = arr.valid
    offset = arr.offset
    length = arr.length
    for b in range((offset + length + 7) // 8):
        byte = valid[b]
        base = b * 8 - offset
        if byte == 0xFF:
            for k in range(8):
                i = base + k
                if 0 <= i < length:
                    total += data[i]
        else:
            for k in range(8):
                i = base + k
                if 0 <= i < length and (byte >> k) & 1:
                    total += data[i]
    return total


def _sum_two_pass(arr):
    total = np.sum(arr.data)
    if arr.null_count == 0:
        return total
    valid = arr.valid
    offset = arr.offset
    length = arr.length
    for b in range((offset + length + 7) // 8):
        byte = valid[b]
        if byte != 0xFF:
            base = b * 8 - offset
            for k in range(8):
                i = base + k
                if 0 <= i < length and not (byte >> k) & 1:
                    total -= arr.data[i]
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
_byte_aware_sum = arrow_jit(_sum_byte_aware)
_two_pass_sum = arrow_jit(_sum_two_pass)
_parallel_sum = arrow_jit(_sum_two_pass, parallel=True)


def _report(title, rows, note=None, n=N):
    print(f"\n{title}  ({n} int64, best of {REPEATS})")
    if note is not None:
        print(f"  {note}")
    for name, seconds in rows:
        if seconds is None:
            print(f"  {name:26s}{'n/a':>12s}")
        else:
            print(f"  {name:26s}{seconds * 1e3:8.3f} ms"
                  f"{n / seconds / 1e9:8.2f} Gelem/s")


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

    big = 4 * N
    big_values = np.arange(big, dtype=np.int64)
    big_mask = (np.arange(big) % 100) != 0
    big_arrow = pa.array(big_values, mask=~big_mask)
    big_view = from_arrow(big_arrow)
    expected_big = int(big_values[big_mask].sum())
    assert _arrow_sum_masked(big_view) == expected_big
    assert _byte_aware_sum(big_view) == expected_big
    assert _two_pass_sum(big_view) == expected_big
    assert _parallel_sum(big_view) == expected_big
    _report("sum, 1% nulls, out of cache", [
        ("arr[i] is not None loop",
         _fastest(lambda: _arrow_sum_masked(big_view))),
        ("byte-aware bitmap loop",
         _fastest(lambda: _byte_aware_sum(big_view))),
        ("two-pass bitmap fixup",
         _fastest(lambda: _two_pass_sum(big_view))),
        ("parallel two-pass",
         _fastest(lambda: _parallel_sum(big_view))),
        ("pyarrow pc.sum", _fastest(lambda: pc.sum(big_arrow))),
    ], note="4M elements (32 MB) against a 24 MB L3", n=big)


if __name__ == "__main__":
    main()
