"""Zero-copy Apache Arrow interop for Numba.

Arrays are imported through the Arrow PyCapsule interface, so any producer
(pyarrow, Polars, nanoarrow, ...) works without depending on it.
"""

import ctypes
import weakref

import numpy as np

__all__ = ["ArrowArrayView", "from_arrow"]


class _ArrowSchema(ctypes.Structure):
    _fields_ = [
        ("format", ctypes.c_char_p),
        ("name", ctypes.c_char_p),
        ("metadata", ctypes.c_char_p),
        ("flags", ctypes.c_int64),
        ("n_children", ctypes.c_int64),
        ("children", ctypes.c_void_p),
        ("dictionary", ctypes.c_void_p),
        ("release", ctypes.c_void_p),
        ("private_data", ctypes.c_void_p),
    ]


class _ArrowArray(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_int64),
        ("null_count", ctypes.c_int64),
        ("offset", ctypes.c_int64),
        ("n_buffers", ctypes.c_int64),
        ("n_children", ctypes.c_int64),
        ("buffers", ctypes.POINTER(ctypes.c_void_p)),
        ("children", ctypes.c_void_p),
        ("dictionary", ctypes.c_void_p),
        ("release", ctypes.c_void_p),
        ("private_data", ctypes.c_void_p),
    ]


_SchemaRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(_ArrowSchema))
_ArrayRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(_ArrowArray))

_PyCapsule_IsValid = ctypes.pythonapi.PyCapsule_IsValid
_PyCapsule_IsValid.argtypes = [ctypes.py_object, ctypes.c_char_p]
_PyCapsule_IsValid.restype = ctypes.c_int

_PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
_PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
_PyCapsule_GetPointer.restype = ctypes.c_void_p

_FORMATS = {
    "c": "int8",
    "C": "uint8",
    "s": "int16",
    "S": "uint16",
    "i": "int32",
    "I": "uint32",
    "l": "int64",
    "L": "uint64",
    "e": "float16",
    "f": "float32",
    "g": "float64",
    "tdD": "datetime64[D]",
    "tdm": "datetime64[ms]",
}
_TIME_UNITS = {"s": "s", "m": "ms", "u": "us", "n": "ns"}


def _capsule_pointer(capsule, name):
    if not _PyCapsule_IsValid(capsule, name):
        raise TypeError(
            f"expected a PyCapsule named {name.decode()!r}, "
            f"got {type(capsule).__name__}"
        )
    return _PyCapsule_GetPointer(capsule, name)


def _take(struct_type, capsule, name):
    addr = _capsule_pointer(capsule, name)
    src = struct_type.from_address(addr)
    dst = struct_type()
    ctypes.memmove(ctypes.addressof(dst), addr, ctypes.sizeof(struct_type))
    src.release = None
    return dst


def _release_struct(struct, release_type):
    if struct.release:
        release_type(struct.release)(ctypes.byref(struct))
        struct.release = None


def _release_owner(schema, array):
    _release_struct(array, _ArrayRelease)
    _release_struct(schema, _SchemaRelease)


class _Owner:
    __slots__ = ("schema", "array", "_finalizer", "__weakref__")

    def __init__(self, schema, array):
        self.schema = schema
        self.array = array
        self._finalizer = weakref.finalize(self, _release_owner, schema, array)


def _dtype_from_format(fmt):
    if fmt == "b":
        raise TypeError(
            "Arrow boolean arrays are bit-packed and not supported")
    if fmt in _FORMATS:
        return np.dtype(_FORMATS[fmt])
    if fmt.startswith("ts") and len(fmt) >= 3:
        unit = _TIME_UNITS.get(fmt[2])
        if unit is not None:
            return np.dtype(f"datetime64[{unit}]")
    if len(fmt) == 3 and fmt.startswith("tD"):
        unit = _TIME_UNITS.get(fmt[2])
        if unit is not None:
            return np.dtype(f"timedelta64[{unit}]")
    raise TypeError(f"unsupported Arrow format string {fmt!r}")


def _buffer(ptr, nbytes, owner):
    buf = (ctypes.c_ubyte * nbytes).from_address(ptr)
    buf._owner = owner
    return buf


def _data_view(array, dtype, owner):
    nbytes = array.length * dtype.itemsize
    if nbytes == 0:
        return np.empty(0, dtype=dtype)
    ptr = array.buffers[1] if array.n_buffers > 1 else None
    if not ptr:
        raise ValueError("Arrow array has no data buffer")
    buf = _buffer(ptr + array.offset * dtype.itemsize, nbytes, owner)
    return np.frombuffer(buf, dtype=np.uint8).view(dtype)


def _validity_view(array, owner):
    if array.n_buffers == 0 or not array.buffers[0]:
        return np.empty(0, dtype=np.uint8)
    nbytes = (array.offset + array.length + 7) // 8
    return np.frombuffer(_buffer(array.buffers[0], nbytes, owner),
                         dtype=np.uint8)


class ArrowArrayView:
    """Zero-copy view of an Arrow array, usable from Python and Numba."""

    __slots__ = ("dtype", "data", "valid", "length", "null_count", "offset",
                 "_owner", "__weakref__")

    def __init__(self, dtype, data, valid, length, null_count, offset, owner):
        self.dtype = dtype
        self.data = data
        self.valid = valid
        self.length = length
        self.null_count = null_count
        self.offset = offset
        self._owner = owner

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.data[index]

    def __repr__(self):
        return (f"ArrowArrayView({self.dtype}, length={self.length}, "
                f"null_count={self.null_count})")


def _build(schema, array):
    owner = _Owner(schema, array)
    if schema.dictionary:
        raise TypeError("dictionary-encoded Arrow arrays are not supported")
    fmt = schema.format.decode() if schema.format else ""
    if fmt.startswith("+") or array.n_children:
        raise TypeError(
            f"nested Arrow arrays are not supported (format {fmt!r})")
    dtype = _dtype_from_format(fmt)
    data = _data_view(array, dtype, owner)
    valid = _validity_view(array, owner)
    null_count = array.null_count if valid.size else 0
    return ArrowArrayView(dtype, data, valid, array.length, null_count,
                          array.offset, owner)


def from_arrow(obj):
    """Import an Arrow array as a zero-copy :class:`ArrowArrayView`.

    *obj* must implement ``__arrow_c_array__``.
    """
    export = getattr(obj, "__arrow_c_array__", None)
    if export is None:
        raise TypeError(
            f"{type(obj).__name__} does not implement __arrow_c_array__")
    schema_capsule, array_capsule = export()
    schema = _take(_ArrowSchema, schema_capsule, b"arrow_schema")
    array = _take(_ArrowArray, array_capsule, b"arrow_array")
    return _build(schema, array)
