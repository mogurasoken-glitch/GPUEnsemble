"""Minimal GGUF header reader.

Reads only the header (metadata and tensor table), never the weights, so it
works on 60 GB model files in a fraction of a second.

Tensor sizes are derived from the gaps between consecutive tensor offsets
(and the end of the file) rather than from a table of quantization block
sizes. That keeps the reader correct for quantization types that did not
exist when it was written (MXFP4, ternary types, ...). The only error is the
alignment padding between tensors, at most ``alignment - 1`` bytes each.

Where the exact size matters (llama.cpp's RPC server names a cached tensor
by a hash of exactly its bytes), ``TensorInfo.size`` gives it from a table
of the known types, checked against the gap; it is None for a type the
table does not know, or when the two disagree.
"""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass, field
from typing import Any, BinaryIO

GGUF_MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32

# GGUF metadata value types.
_UINT8, _INT8, _UINT16, _INT16, _UINT32, _INT32, _FLOAT32, _BOOL, _STRING, _ARRAY, _UINT64, _INT64, _FLOAT64 = range(13)

_SCALAR_FORMATS = {
    _UINT8: "<B",
    _INT8: "<b",
    _UINT16: "<H",
    _INT16: "<h",
    _UINT32: "<I",
    _INT32: "<i",
    _FLOAT32: "<f",
    _BOOL: "<?",
    _UINT64: "<Q",
    _INT64: "<q",
    _FLOAT64: "<d",
}

# Bytes per block and elements per block of the ggml tensor types (ggml.c's type traits).
_TYPE_BLOCKS = {
    0: (4, 1), 1: (2, 1), 2: (18, 32), 3: (20, 32), 6: (22, 32), 7: (24, 32), 8: (34, 32), 9: (36, 32),
    10: (84, 256), 11: (110, 256), 12: (144, 256), 13: (176, 256), 14: (210, 256), 15: (292, 256),
    16: (66, 256), 17: (74, 256), 18: (98, 256), 19: (50, 256), 20: (18, 32), 21: (110, 256), 22: (82, 256),
    23: (136, 256), 24: (1, 1), 25: (2, 1), 26: (4, 1), 27: (8, 1), 28: (8, 1), 29: (56, 256), 30: (2, 1),
    34: (54, 256), 35: (66, 256), 39: (17, 32),
}

_SPLIT_RE = re.compile(r"^(?P<stem>.*)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$")


class GGUFError(ValueError):
    """Raised when a file is not a GGUF file this reader understands."""


@dataclass
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    offset: int  # relative to the start of the data section
    nbytes: int = 0  # filled in after all tensors are known; includes the padding up to the next tensor
    part: int = 0  # which file of a split model holds it (index into GGUFFile.parts)
    start: int = 0  # where its bytes begin in that file
    size: int | None = None  # its exact size in bytes, when the type is known and agrees with nbytes


@dataclass
class GGUFFile:
    path: str
    version: int
    metadata: dict[str, Any]
    tensors: list[TensorInfo] = field(default_factory=list)
    parts: list[str] = field(default_factory=list)  # every file of the model, in order

    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture", "unknown"))

    def arch_key(self, suffix: str, default: Any = None) -> Any:
        """Read ``<architecture>.<suffix>``, e.g. ``llama.block_count``."""
        return self.metadata.get(f"{self.architecture}.{suffix}", default)


def _read_exact(f: BinaryIO, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise GGUFError("unexpected end of file while reading the GGUF header")
    return data


def _read_scalar(f: BinaryIO, vtype: int) -> Any:
    fmt = _SCALAR_FORMATS[vtype]
    return struct.unpack(fmt, _read_exact(f, struct.calcsize(fmt)))[0]


def _read_string(f: BinaryIO) -> str:
    (length,) = struct.unpack("<Q", _read_exact(f, 8))
    return _read_exact(f, length).decode("utf-8", errors="replace")


def _read_value(f: BinaryIO, vtype: int) -> Any:
    if vtype == _STRING:
        return _read_string(f)
    if vtype == _ARRAY:
        (item_type,) = struct.unpack("<I", _read_exact(f, 4))
        (count,) = struct.unpack("<Q", _read_exact(f, 8))
        if item_type in _SCALAR_FORMATS:
            # Bulk read: token tables can hold hundreds of thousands of items.
            fmt = _SCALAR_FORMATS[item_type]
            size = struct.calcsize(fmt)
            raw = _read_exact(f, size * count)
            return list(struct.unpack(f"<{count}{fmt[1:]}", raw))
        return [_read_value(f, item_type) for _ in range(count)]
    if vtype in _SCALAR_FORMATS:
        return _read_scalar(f, vtype)
    raise GGUFError(f"unknown GGUF metadata value type {vtype}")


def read_gguf(path: str | os.PathLike[str]) -> GGUFFile:
    """Read the header of one GGUF file."""
    path = os.fspath(path)
    with open(path, "rb") as f:
        return read_header(f, path, os.path.getsize(path))


def read_header(f: BinaryIO, path: str, file_size: int) -> GGUFFile:
    """Read a GGUF header from ``f`` (positioned at the start of the file), which is ``file_size`` bytes long.

    Only reads forward, so ``f`` may be a stream, e.g. the start of a file on a web server.
    """
    if _read_exact(f, 4) != GGUF_MAGIC:
        raise GGUFError(f"{path} is not a GGUF file")
    (version,) = struct.unpack("<I", _read_exact(f, 4))
    if version < 2:
        raise GGUFError(f"GGUF version {version} is too old (need 2 or newer)")
    n_tensors, n_kv = struct.unpack("<QQ", _read_exact(f, 16))

    metadata: dict[str, Any] = {}
    for _ in range(n_kv):
        key = _read_string(f)
        (vtype,) = struct.unpack("<I", _read_exact(f, 4))
        metadata[key] = _read_value(f, vtype)

    tensors: list[TensorInfo] = []
    for _ in range(n_tensors):
        name = _read_string(f)
        (n_dims,) = struct.unpack("<I", _read_exact(f, 4))
        shape = struct.unpack(f"<{n_dims}Q", _read_exact(f, 8 * n_dims))
        ggml_type, offset = struct.unpack("<IQ", _read_exact(f, 12))
        tensors.append(TensorInfo(name, tuple(shape), ggml_type, offset))

    header_end = f.tell()

    alignment = int(metadata.get("general.alignment", DEFAULT_ALIGNMENT))
    data_start = (header_end + alignment - 1) // alignment * alignment
    data_size = file_size - data_start

    ordered = sorted(tensors, key=lambda t: t.offset)
    for current, nxt in zip(ordered, ordered[1:] + [None]):
        end = nxt.offset if nxt is not None else data_size
        if end < current.offset:
            raise GGUFError(f"tensor {current.name} extends past the end of {path}")
        current.nbytes = end - current.offset
        current.start = data_start + current.offset
        current.size = _exact_size(current, alignment)

    return GGUFFile(path=path, version=version, metadata=metadata, tensors=tensors, parts=[path])


def _exact_size(t: TensorInfo, alignment: int) -> int | None:
    """The tensor's size as ggml counts it, if the type table and the gap to the next tensor agree."""
    block = _TYPE_BLOCKS.get(t.ggml_type)
    if block is None or not t.shape or t.shape[0] % block[1]:
        return None
    size = block[0] * (t.shape[0] // block[1])
    for n in t.shape[1:]:
        size *= n
    return size if size <= t.nbytes < size + alignment else None


def split_parts(path: str | os.PathLike[str]) -> list[str]:
    """Return every part of a split model given any one of its files.

    ``model-00001-of-00003.gguf`` -> the three part paths, in order.
    A file that is not split returns ``[path]``.
    """
    path = os.fspath(path)
    directory, name = os.path.split(path)
    m = _SPLIT_RE.match(name)
    if not m:
        return [path]
    total = int(m.group("total"))
    stem = m.group("stem")
    parts = [os.path.join(directory, f"{stem}-{i:05d}-of-{total:05d}.gguf") for i in range(1, total + 1)]
    missing = [p for p in parts if not os.path.exists(p)]
    if missing:
        raise GGUFError("split model is incomplete; missing: " + ", ".join(missing))
    return parts


def read_model(path: str | os.PathLike[str]) -> GGUFFile:
    """Read a model that may be split over several GGUF files.

    Metadata comes from the first part; tensors from all parts.
    """
    parts = split_parts(path)
    first = read_gguf(parts[0])
    for index, part in enumerate(parts[1:], start=1):
        more = read_gguf(part).tensors
        for t in more:
            t.part = index
        first.tensors.extend(more)
    first.parts = parts
    return first
