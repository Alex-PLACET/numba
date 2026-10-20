import ctypes
import gc
import unittest

import numpy as np

from numba import jit, typeof, types
from numba.experimental.arrow import ArrowArrayView, from_arrow

try:
    import pyarrow as pa
except ImportError:
    pa = None

try:
    import polars as pl
except ImportError:
    pl = None

requires_pyarrow = unittest.skipIf(pa is None, "pyarrow is not installed")
requires_polars = unittest.skipIf(pl is None, "polars is not installed")


@requires_pyarrow
class TestArrowImport(unittest.TestCase):
    def test_int64_zero_copy(self):
        src = pa.array([1, 2, 3, 4])
        view = from_arrow(src)
        self.assertIsInstance(view, ArrowArrayView)
        self.assertEqual(view.dtype, np.dtype("int64"))
        self.assertEqual(len(view), 4)
        self.assertEqual(view[2], 3)
        self.assertEqual(view.null_count, 0)
        self.assertEqual(view.offset, 0)
        self.assertEqual(view.data.ctypes.data, src.buffers()[1].address)

    def test_float64_values(self):
        view = from_arrow(pa.array([1.5, 2.5]))
        self.assertEqual(view.dtype, np.dtype("float64"))
        np.testing.assert_array_equal(view.data, [1.5, 2.5])

    def test_dtypes(self):
        cases = [
            (pa.array([1], type=pa.int8()), "int8"),
            (pa.array([1], type=pa.uint8()), "uint8"),
            (pa.array([1], type=pa.int16()), "int16"),
            (pa.array([1], type=pa.uint16()), "uint16"),
            (pa.array([1], type=pa.int32()), "int32"),
            (pa.array([1], type=pa.uint32()), "uint32"),
            (pa.array([1], type=pa.int64()), "int64"),
            (pa.array([1], type=pa.uint64()), "uint64"),
            (pa.array([1.0], type=pa.float32()), "float32"),
            (pa.array([1.0], type=pa.float64()), "float64"),
            (pa.array([1], type=pa.date32()), "datetime64[D]"),
            (pa.array([1], type=pa.timestamp("us")), "datetime64[us]"),
            (pa.array([1], type=pa.timestamp("ns", tz="UTC")),
             "datetime64[ns]"),
            (pa.array([1], type=pa.duration("ns")), "timedelta64[ns]"),
        ]
        for src, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(from_arrow(src).dtype, np.dtype(expected))

    def test_nulls(self):
        view = from_arrow(pa.array([1.0, None, 3.0]))
        self.assertEqual(view.null_count, 1)
        self.assertEqual(view.valid.dtype, np.dtype("uint8"))
        self.assertEqual(view.data[2], 3.0)

    def test_sliced_with_offset(self):
        view = from_arrow(pa.array([1, None, 3, None, 5])[1:])
        self.assertEqual(view.offset, 1)
        self.assertEqual(len(view), 4)
        self.assertEqual(view.data[1], 3)
        self.assertEqual(view.data[3], 5)
        self.assertEqual(view.null_count, 2)

    def test_empty(self):
        view = from_arrow(pa.array([], type=pa.int64()))
        self.assertEqual(len(view), 0)
        self.assertEqual(view.data.shape, (0,))

    def test_bool_rejected(self):
        with self.assertRaises(TypeError) as cm:
            from_arrow(pa.array([True, False]))
        self.assertIn("bit-packed", str(cm.exception))

    def test_string_rejected(self):
        with self.assertRaises(TypeError) as cm:
            from_arrow(pa.array(["a", "b"]))
        self.assertIn("unsupported Arrow format", str(cm.exception))

    def test_dictionary_rejected(self):
        with self.assertRaises(TypeError) as cm:
            from_arrow(pa.array(["a", "b"]).dictionary_encode())
        self.assertIn("dictionary", str(cm.exception))

    def test_nested_rejected(self):
        with self.assertRaises(TypeError) as cm:
            from_arrow(pa.array([[1, 2], [3]]))
        self.assertIn("nested", str(cm.exception))


@requires_pyarrow
class TestArrowStream(unittest.TestCase):
    def test_single_chunk(self):
        view = from_arrow(pa.chunked_array([[1, 2, 3]]))
        self.assertEqual(len(view), 3)
        self.assertEqual(view.dtype, np.dtype("int64"))
        np.testing.assert_array_equal(view.data, [1, 2, 3])

    def test_multiple_chunks_rejected(self):
        with self.assertRaises(TypeError) as cm:
            from_arrow(pa.chunked_array([[1, 2], [3]]))
        self.assertIn("chunk", str(cm.exception))

    def test_not_exportable(self):
        with self.assertRaises(TypeError) as cm:
            from_arrow([1, 2, 3])
        self.assertIn("__arrow_c_array__", str(cm.exception))


@requires_polars
class TestPolarsImport(unittest.TestCase):
    def test_series(self):
        view = from_arrow(pl.Series("a", [1, 2, 3]))
        self.assertEqual(view.dtype, np.dtype("int64"))
        np.testing.assert_array_equal(view.data, [1, 2, 3])

    def test_series_float_with_null(self):
        view = from_arrow(pl.Series("a", [1.0, None, 3.0]))
        self.assertEqual(view.null_count, 1)
        self.assertEqual(view.data[2], 3.0)


@requires_pyarrow
class TestNumbaArrow(unittest.TestCase):
    def test_typeof(self):
        from numba.experimental.arrow import ArrowArrayType

        ty = typeof(from_arrow(pa.array([1, 2])))
        self.assertIsInstance(ty, ArrowArrayType)
        self.assertEqual(ty.dtype, types.int64)
        self.assertEqual(ty, ArrowArrayType(types.int64))

    def test_len_and_getitem(self):
        @jit
        def f(arr):
            return len(arr), arr[1], arr[-1]

        self.assertEqual(f(from_arrow(pa.array([10, 20, 30]))), (3, 20, 30))

    def test_data_attribute(self):
        @jit
        def f(arr):
            return np.sum(arr.data)

        self.assertEqual(f(from_arrow(pa.array([1.0, 2.0, 3.0]))), 6.0)

    def test_loop_sum(self):
        @jit
        def f(arr):
            total = 0
            for i in range(len(arr)):
                total += arr[i]
            return total

        self.assertEqual(f(from_arrow(pa.array([1, 2, 3, 4]))), 10)

    def test_repeated_calls(self):
        @jit
        def f(arr):
            return np.sum(arr.data)

        for _ in range(200):
            self.assertEqual(f(from_arrow(pa.array([1, 2]))), 3)

    @requires_polars
    def test_polars_series_in_jit(self):
        @jit
        def f(arr):
            return np.sum(arr.data)

        self.assertEqual(f(from_arrow(pl.Series("a", [1.0, 2.0]))), 3.0)

    def test_is_valid(self):
        view = from_arrow(pa.array([1.0, None, 3.0]))

        @jit
        def mask(arr):
            return [arr.is_valid(i) for i in range(len(arr))]

        self.assertEqual(mask(view), [True, False, True])

    def test_sum_non_null(self):
        @jit
        def f(arr):
            total = 0.0
            for i in range(len(arr)):
                if arr.is_valid(i):
                    total += arr[i]
            return total

        self.assertEqual(f(from_arrow(pa.array([1.0, None, 3.0]))), 4.0)

    def test_is_valid_negative_index(self):
        @jit
        def f(arr):
            return arr.is_valid(-1)

        self.assertTrue(f(from_arrow(pa.array([1, 2]))))
        self.assertFalse(f(from_arrow(pa.array([1, None]))))

    def test_is_valid_out_of_range(self):
        @jit
        def f(arr):
            return arr.is_valid(5)

        with self.assertRaises(IndexError):
            f(from_arrow(pa.array([1, 2])))


from numba.experimental.arrow import _ArrowArray, _ArrowSchema  # noqa: E402

_ArrayRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(_ArrowArray))

_PyCapsule_New = ctypes.pythonapi.PyCapsule_New
_PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
_PyCapsule_New.restype = ctypes.py_object


class _Producer:
    """Minimal ArrowArray producer used to observe the release callback."""

    def __init__(self, values, null_count=0, valid_bytes=None):
        self.released = 0
        self._values = (ctypes.c_int64 * len(values))(*values)
        self._bitmap = None
        if valid_bytes is not None:
            self._bitmap = ctypes.create_string_buffer(valid_bytes)
        self._buffers = (ctypes.c_void_p * 2)()
        if self._bitmap is not None:
            self._buffers[0] = ctypes.cast(self._bitmap, ctypes.c_void_p)
        self._buffers[1] = ctypes.cast(self._values, ctypes.c_void_p)
        self._format = ctypes.create_string_buffer(b"l")
        self._callback = _ArrayRelease(self._on_release)
        self._array = _ArrowArray(
            length=len(values), null_count=null_count, offset=0, n_buffers=2,
            n_children=0, buffers=self._buffers, children=None,
            dictionary=None,
            release=ctypes.cast(self._callback, ctypes.c_void_p),
            private_data=None)
        self._schema = _ArrowSchema(
            format=ctypes.cast(self._format, ctypes.c_char_p), name=None,
            metadata=None, flags=0, n_children=0, children=None,
            dictionary=None, release=None, private_data=None)

    def _on_release(self, ptr):
        self.released += 1

    def __arrow_c_array__(self, requested_schema=None):
        schema = _PyCapsule_New(
            ctypes.addressof(self._schema), b"arrow_schema", None)
        array = _PyCapsule_New(
            ctypes.addressof(self._array), b"arrow_array", None)
        return schema, array


class TestOwnership(unittest.TestCase):
    def test_release_on_gc(self):
        producer = _Producer([1, 2, 3])
        view = from_arrow(producer)
        self.assertEqual(view.data.tolist(), [1, 2, 3])
        self.assertEqual(producer.released, 0)
        del view
        gc.collect()
        self.assertEqual(producer.released, 1)

    def test_arrays_keep_owner_alive(self):
        producer = _Producer([1, 2, 3])
        data = from_arrow(producer).data
        gc.collect()
        self.assertEqual(producer.released, 0)
        self.assertEqual(data.tolist(), [1, 2, 3])
        del data
        gc.collect()
        self.assertEqual(producer.released, 1)

    def test_unknown_null_count(self):
        producer = _Producer([1, 2], null_count=-1, valid_bytes=b"\x01")
        view = from_arrow(producer)
        self.assertEqual(view.null_count, -1)

        @jit
        def mask(arr):
            return [arr.is_valid(i) for i in range(len(arr))]

        self.assertEqual(mask(view), [True, False])
