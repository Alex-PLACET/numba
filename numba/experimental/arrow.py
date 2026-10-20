"""Zero-copy Apache Arrow interop for Numba.

Arrays are imported through the Arrow PyCapsule interface, so any producer
(pyarrow, Polars, nanoarrow, ...) works without depending on it.
"""

from __future__ import annotations

import ctypes
import operator
import weakref
from typing import Any, Callable, TypeVar

import numpy as np

from numba.core import cgutils, types
from numba.core.datamodel import models
from numba.core.decorators import jit
from numba.core.pythonapi import _UnboxContext
from numba.core.typing.typeof import _TypeofContext, typeof_impl
from numba.extending import (
    NativeValue,
    make_attribute_wrapper,
    overload,
    overload_method,
    register_model,
    unbox,
)
from numba.np import numpy_support

__all__ = ["ArrowArrayType", "ArrowArrayView", "arrow_jit", "from_arrow"]

_StructT = TypeVar("_StructT", bound=ctypes.Structure)


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


class _ArrowArrayStream(ctypes.Structure):
    _fields_ = [
        ("get_schema", ctypes.c_void_p),
        ("get_next", ctypes.c_void_p),
        ("get_last_error", ctypes.c_void_p),
        ("release", ctypes.c_void_p),
        ("private_data", ctypes.c_void_p),
    ]


_SchemaRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(_ArrowSchema))
_ArrayRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(_ArrowArray))
_StreamRelease = ctypes.CFUNCTYPE(None, ctypes.POINTER(_ArrowArrayStream))
_StreamGetSchema = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.POINTER(_ArrowArrayStream),
    ctypes.POINTER(_ArrowSchema))
_StreamGetNext = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.POINTER(_ArrowArrayStream),
    ctypes.POINTER(_ArrowArray))
_StreamGetError = ctypes.CFUNCTYPE(
    ctypes.c_char_p, ctypes.POINTER(_ArrowArrayStream))

_PyCapsule_IsValid = ctypes.pythonapi.PyCapsule_IsValid
_PyCapsule_IsValid.argtypes = [ctypes.py_object, ctypes.c_char_p]
_PyCapsule_IsValid.restype = ctypes.c_int

_PyCapsule_GetPointer = ctypes.pythonapi.PyCapsule_GetPointer
_PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
_PyCapsule_GetPointer.restype = ctypes.c_void_p

_FORMATS: dict[str, str] = {
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
_TIME_UNITS: dict[str, str] = {"s": "s", "m": "ms", "u": "us", "n": "ns"}


def _capsule_pointer(capsule: object, name: bytes) -> int:
    if not _PyCapsule_IsValid(capsule, name):
        raise TypeError(
            f"expected a PyCapsule named {name.decode()!r}, "
            f"got {type(capsule).__name__}"
        )
    return _PyCapsule_GetPointer(capsule, name)


def _take(struct_type: type[_StructT], capsule: object,
          name: bytes) -> _StructT:
    addr = _capsule_pointer(capsule, name)
    src = struct_type.from_address(addr)
    dst = struct_type()
    ctypes.memmove(ctypes.addressof(dst), addr, ctypes.sizeof(struct_type))
    src.release = None
    return dst


def _release_struct(struct: ctypes.Structure,
                    release_type: Callable[..., Any]) -> None:
    if struct.release:
        release_type(struct.release)(ctypes.byref(struct))
        struct.release = None


def _release_owner(schema: _ArrowSchema, array: _ArrowArray) -> None:
    _release_struct(array, _ArrayRelease)
    _release_struct(schema, _SchemaRelease)


class _Owner:
    __slots__ = ("__weakref__", "_finalizer", "array", "schema")

    schema: _ArrowSchema
    array: _ArrowArray
    _finalizer: weakref.finalize

    def __init__(self, schema: _ArrowSchema, array: _ArrowArray) -> None:
        self.schema = schema
        self.array = array
        self._finalizer = weakref.finalize(self, _release_owner, schema, array)


def _dtype_from_format(fmt: str) -> np.dtype:
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


def _buffer(ptr: int, nbytes: int, owner: _Owner) -> ctypes.Array:
    buf = (ctypes.c_ubyte * nbytes).from_address(ptr)
    setattr(buf, "_owner", owner)
    return buf


def _data_view(array: _ArrowArray, dtype: np.dtype,
               owner: _Owner) -> np.ndarray:
    nbytes = array.length * dtype.itemsize
    if nbytes == 0:
        return np.empty(0, dtype=dtype)
    ptr = array.buffers[1] if array.n_buffers > 1 else None
    if not ptr:
        raise ValueError("Arrow array has no data buffer")
    buf = _buffer(ptr + array.offset * dtype.itemsize, nbytes, owner)
    return np.frombuffer(buf, dtype=np.uint8).view(dtype)


def _validity_view(array: _ArrowArray, owner: _Owner) -> np.ndarray:
    if array.n_buffers == 0 or not array.buffers[0]:
        return np.empty(0, dtype=np.uint8)
    nbytes = (array.offset + array.length + 7) // 8
    return np.frombuffer(_buffer(array.buffers[0], nbytes, owner),
                         dtype=np.uint8)


class ArrowArrayView:
    """Zero-copy view of an Arrow array, usable from Python and Numba."""

    __slots__ = (
        "__weakref__",
        "_owner",
        "data",
        "dtype",
        "length",
        "null_count",
        "offset",
        "valid",
    )

    dtype: np.dtype
    data: np.ndarray
    valid: np.ndarray
    length: int
    null_count: int
    offset: int
    _owner: _Owner

    def __init__(self, dtype: np.dtype, data: np.ndarray, valid: np.ndarray,
                 length: int, null_count: int, offset: int,
                 owner: _Owner) -> None:
        self.dtype = dtype
        self.data = data
        self.valid = valid
        self.length = length
        self.null_count = null_count
        self.offset = offset
        self._owner = owner

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Any:
        return self.data[index]

    def __repr__(self) -> str:
        return (f"ArrowArrayView({self.dtype}, length={self.length}, "
                f"null_count={self.null_count})")


def _build(schema: _ArrowSchema, array: _ArrowArray) -> ArrowArrayView:
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


def _stream_error(stream: _ArrowArrayStream,
                  get_error: Callable[..., Any] | None) -> str:
    if get_error is None:
        return "Arrow stream error"
    msg = get_error(ctypes.byref(stream))
    return f"Arrow stream error: {msg.decode() if msg else 'unknown error'}"


def _from_stream(capsule: object) -> tuple[_ArrowSchema, _ArrowArray]:
    stream = _take(_ArrowArrayStream, capsule, b"arrow_array_stream")
    get_schema = _StreamGetSchema(stream.get_schema)
    get_next = _StreamGetNext(stream.get_next)
    get_error = (_StreamGetError(stream.get_last_error)
                 if stream.get_last_error else None)
    schema = _ArrowSchema()
    array = _ArrowArray()
    status = get_schema(ctypes.byref(stream), ctypes.byref(schema))
    if not status:
        status = get_next(ctypes.byref(stream), ctypes.byref(array))
    error = _stream_error(stream, get_error) if status else None
    if error is None and not array.release:
        error = "Arrow stream is empty"
    has_more = False
    if error is None:
        extra = _ArrowArray()
        has_more = (get_next(ctypes.byref(stream), ctypes.byref(extra)) == 0
                    and bool(extra.release))
        if has_more:
            _release_struct(extra, _ArrayRelease)
    _release_struct(stream, _StreamRelease)
    if error is not None or has_more:
        _release_struct(array, _ArrayRelease)
        _release_struct(schema, _SchemaRelease)
        if has_more:
            raise TypeError(
                "Arrow chunked arrays with more than one chunk are not "
                "supported")
        raise ValueError(error)
    return schema, array


def from_arrow(obj: object) -> ArrowArrayView:
    """Import an Arrow array as a zero-copy :class:`ArrowArrayView`.

    *obj* must implement ``__arrow_c_array__`` (single array) or
    ``__arrow_c_stream__`` (single-chunk stream).
    """
    export = getattr(obj, "__arrow_c_array__", None)
    if export is not None:
        schema_capsule, array_capsule = export()
        schema = _take(_ArrowSchema, schema_capsule, b"arrow_schema")
        array = _take(_ArrowArray, array_capsule, b"arrow_array")
    else:
        export = getattr(obj, "__arrow_c_stream__", None)
        if export is None:
            raise TypeError(
                f"{type(obj).__name__} does not implement __arrow_c_array__ "
                f"or __arrow_c_stream__")
        schema, array = _from_stream(export())
    return _build(schema, array)


def _view_of(obj: Any) -> Any:
    if hasattr(obj, "__arrow_c_array__") or hasattr(obj, "__arrow_c_stream__"):
        return from_arrow(obj)
    return obj


class _ArrowDispatcher:
    """``jit`` dispatcher that imports Arrow arrays on the way in."""

    def __init__(self, func: Any, **kwargs: Any) -> None:
        self._dispatcher = jit(func, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._dispatcher(*[_view_of(arg) for arg in args], **kwargs)

    def __getattr__(self, name: str) -> Any:
        if name == "_dispatcher":
            raise AttributeError(name)
        return getattr(self._dispatcher, name)


def arrow_jit(func: Any = None, **kwargs: Any) -> Any:
    """Decorate like :func:`numba.jit`, accepting Arrow arrays as arguments.

    Use :func:`from_arrow` instead when the view is also needed in Python.
    """
    if func is None:
        return lambda f: _ArrowDispatcher(f, **kwargs)
    return _ArrowDispatcher(func, **kwargs)


class ArrowArrayType(types.Type):
    """Numba type of a 1-D fixed-width Arrow array."""

    dtype: types.Type

    def __init__(self, dtype: types.Type) -> None:
        self.dtype = dtype
        super().__init__(name=f"ArrowArray({dtype})")

    @property
    def key(self) -> types.Type:
        return self.dtype

    @property
    def array_type(self) -> types.Array:
        return types.Array(self.dtype, 1, "C")


@register_model(ArrowArrayType)
class _ArrowArrayModel(models.StructModel):
    def __init__(self, dmm: Any, fe_type: ArrowArrayType) -> None:
        members = [
            ("data", fe_type.array_type),
            ("valid", types.Array(types.uint8, 1, "C")),
            ("length", types.intp),
            ("null_count", types.intp),
            ("offset", types.intp),
        ]
        super().__init__(dmm, fe_type, members)


for _name in ("data", "valid", "length", "null_count", "offset"):
    make_attribute_wrapper(ArrowArrayType, _name, _name)


@typeof_impl.register(ArrowArrayView)
def _typeof_arrow_array(val: ArrowArrayView,
                        c: _TypeofContext) -> ArrowArrayType:
    for name in ("data", "valid", "length", "null_count", "offset"):
        if not hasattr(val, name):
            raise TypeError(f"ArrowArrayView is missing attribute {name!r}")
    return ArrowArrayType(numpy_support.from_dtype(val.dtype))


@unbox(ArrowArrayType)
def _unbox_arrow_array(typ: ArrowArrayType, obj: Any,
                       c: _UnboxContext) -> NativeValue:
    model = c.context.data_model_manager[typ]
    proxy = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    is_error = cgutils.alloca_once_value(c.builder, cgutils.false_bit)
    for name in ("data", "valid", "length", "null_count", "offset"):
        field_type = model.get_member_fe_type(name)
        attr = c.pyapi.object_getattr_string(obj, name)
        native = c.unbox(field_type, attr)
        c.pyapi.decref(attr)
        setattr(proxy, name, native.value)
        if native.is_error is not None:
            with c.builder.if_then(native.is_error, likely=False):
                c.builder.store(cgutils.true_bit, is_error)
    return NativeValue(proxy._getvalue(), is_error=c.builder.load(is_error))


@overload(len)
def _ol_len(arr):
    if isinstance(arr, ArrowArrayType):
        return lambda arr: arr.length


@overload(operator.getitem)
def _ol_getitem(arr, index):
    if isinstance(arr, ArrowArrayType) and isinstance(index, types.Integer):
        return lambda arr, index: arr.data[index]


@overload_method(ArrowArrayType, "is_valid")
def _ol_is_valid(arr, index):
    def impl(arr, index):
        if index < 0:
            index += arr.length
        if index < 0 or index >= arr.length:
            raise IndexError("Arrow array index out of range")
        if arr.null_count == 0:
            return True
        bit = arr.offset + index
        return ((arr.valid[bit >> 3] >> (bit & 7)) & 1) == 1

    return impl
