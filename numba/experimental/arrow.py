"""Zero-copy Apache Arrow interop for Numba.

Arrays are imported through the Arrow PyCapsule interface, so any producer
(pyarrow, Polars, nanoarrow, ...) works without depending on it.
"""

from __future__ import annotations

import ctypes
import operator
from typing import Any, Callable, TypeVar

import numpy as np
from llvmlite import ir

from numba.core import cgutils, types
from numba.core.datamodel import models
from numba.core.decorators import jit
from numba.core.pythonapi import _UnboxContext
from numba.core.typing.typeof import _TypeofContext, typeof_impl
from numba.extending import (
    NativeValue,
    intrinsic,
    make_attribute_wrapper,
    overload,
    register_model,
    unbox,
)
from numba.np import numpy_support

__all__ = [
    "ArrowArrayType",
    "ArrowArrayView",
    "arrow_jit",
    "from_arrow",
    "is_valid",
]

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
    try:
        return _PyCapsule_GetPointer(capsule, name)
    except ValueError:
        raise TypeError(
            f"expected a PyCapsule named {name.decode()!r}, "
            f"got {type(capsule).__name__}"
        ) from None


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


class _Owner:
    """Releases the imported Arrow structs when the last view dies."""

    __slots__ = ("schema", "array")

    schema: _ArrowSchema
    array: _ArrowArray

    def __init__(self, schema: _ArrowSchema, array: _ArrowArray) -> None:
        self.schema = schema
        self.array = array

    def __del__(self) -> None:
        try:
            _release_struct(self.array, _ArrayRelease)
            _release_struct(self.schema, _SchemaRelease)
        except Exception:
            pass  # interpreter shutdown may already have cleared globals


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
        if self.null_count != 0:
            i = index if index >= 0 else index + self.length
            bit = self.offset + i
            if not (self.valid[bit >> 3] >> (bit & 7)) & 1:
                return None
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
    nullable: bool

    def __init__(self, dtype: types.Type, nullable: bool = False) -> None:
        self.dtype = dtype
        self.nullable = nullable
        name = f"ArrowArray({dtype}{', nullable' if nullable else ''})"
        super().__init__(name=name)

    @property
    def key(self) -> tuple[types.Type, bool]:
        return self.dtype, self.nullable

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

_TYPE_CACHE: dict[tuple[np.dtype, bool], ArrowArrayType] = {}


def _arrow_type(dtype: np.dtype, nullable: bool) -> ArrowArrayType:
    key = dtype, nullable
    typ = _TYPE_CACHE.get(key)
    if typ is None:
        typ = ArrowArrayType(numpy_support.from_dtype(dtype), nullable)
        _TYPE_CACHE[key] = typ
    return typ


@typeof_impl.register(ArrowArrayView)
def _typeof_arrow_array(val: ArrowArrayView,
                        c: _TypeofContext) -> ArrowArrayType:
    for name in ("data", "valid", "length", "null_count", "offset"):
        if not hasattr(val, name):
            raise TypeError(f"ArrowArrayView is missing attribute {name!r}")
    return _arrow_type(val.dtype, val.null_count != 0)


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
    if not isinstance(arr, ArrowArrayType):
        return None
    if not isinstance(index, types.Integer):
        return None
    if arr.nullable:
        return lambda arr, index: _get_or_none(arr, index)
    return lambda arr, index: arr.data[index]


def _index_to_intp(context, builder, index_val, index_type):
    intp = context.get_value_type(types.intp)
    if index_val.type.width == intp.width:
        return index_val
    if index_type.signed:
        return builder.sext(index_val, intp)
    return builder.zext(index_val, intp)


def _check_bounds(context, builder, proxy, idx):
    zero = ir.Constant(idx.type, 0)
    idx = builder.select(
        builder.icmp_signed("<", idx, zero),
        builder.add(idx, proxy.length),
        idx,
    )
    with builder.if_then(builder.or_(
            builder.icmp_signed("<", idx, zero),
            builder.icmp_signed(">=", idx, proxy.length),)):
        context.call_conv.return_user_exc(
            builder, IndexError, ("Arrow array index out of range",))
    return idx


def _validity_bit(context, builder, proxy, idx):
    bitmap = context.make_array(types.Array(types.uint8, 1, "C"))(
        context, builder, value=proxy.valid)
    bit = builder.add(proxy.offset, idx)
    byte = builder.load(builder.gep(
        bitmap.data, [builder.lshr(bit, ir.Constant(idx.type, 3))]))
    shifted = builder.lshr(
        byte,
        builder.trunc(builder.and_(bit, ir.Constant(idx.type, 7)),
                      ir.IntType(8)),
    )
    return builder.icmp_unsigned(
        "!=", builder.and_(shifted, ir.Constant(ir.IntType(8), 1)),
        ir.Constant(ir.IntType(8), 0))


@intrinsic
def is_valid(typingctx, arr, index):
    """Return True if element *index* of an Arrow array is not null."""
    if not isinstance(arr, ArrowArrayType):
        return None
    if not isinstance(index, types.Integer):
        return None
    sig = types.boolean(arr, index)

    def codegen(context, builder, signature, args):
        arr_val, index_val = args
        proxy = cgutils.create_struct_proxy(signature.args[0])(
            context, builder, value=arr_val)
        idx = _check_bounds(
            context, builder, proxy,
            _index_to_intp(context, builder, index_val, index))
        set_bit = _validity_bit(context, builder, proxy, idx)
        out = cgutils.alloca_once(builder, ir.IntType(1))
        with builder.if_else(
                builder.icmp_signed("!=", proxy.null_count,
                                    ir.Constant(idx.type, 0))
        ) as (with_nulls, no_nulls):
            with with_nulls:
                builder.store(set_bit, out)
            with no_nulls:
                builder.store(cgutils.true_bit, out)
        return builder.load(out)

    return sig, codegen


@intrinsic
def _get_or_none(typingctx, arr, index):
    if not isinstance(arr, ArrowArrayType) or not arr.nullable:
        return None
    if not isinstance(index, types.Integer):
        return None
    sig = types.Optional(arr.dtype)(arr, index)

    def codegen(context, builder, signature, args):
        arr_val, index_val = args
        arr_type = signature.args[0]
        proxy = cgutils.create_struct_proxy(arr_type)(
            context, builder, value=arr_val)
        idx = _check_bounds(
            context, builder, proxy,
            _index_to_intp(context, builder, index_val, index))
        data = context.make_array(arr_type.array_type)(
            context, builder, value=proxy.data)
        ptr = builder.bitcast(
            data.data, context.get_value_type(arr_type.dtype).as_pointer())
        out = cgutils.create_struct_proxy(signature.return_type)(
            context, builder)
        out.data = builder.load(builder.gep(ptr, [idx]))
        out.valid = _validity_bit(context, builder, proxy, idx)
        return out._getvalue()

    return sig, codegen
