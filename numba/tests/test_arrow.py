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
