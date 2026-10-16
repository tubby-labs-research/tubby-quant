"""Unwired date-range suffix using short-lived public CPython buffer leases.

No NumPy C API, financial arithmetic, persistent export, or runtime wiring.
Exact native types exclude callbacks; the caller must retain its live guards.
The GIL stays held. Concurrent alias mutation, tracing and allocation-failure
equivalence remain outside the existing sequential prototype's contract.
"""
import numpy as np
from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, PyBUF_SIMPLE
from libc.stdint cimport int64_t
from libc.string cimport memcpy

_RANGE_ARRAY = np.ndarray
_RANGE_DATETIME = np.datetime64
_RANGE_NS_DTYPE = np.dtype("datetime64[ns]")


cdef int64_t _scalar_ticks(object scalar) except *:
    cdef Py_buffer lease
    cdef int64_t result
    # NumPy datetime scalars export eight bytes with itemsize ONE. Request no
    # format/writable access; no source callbacks or scalar .view allocation.
    PyObject_GetBuffer(scalar, &lease, PyBUF_SIMPLE)
    try:
        if lease.buf == NULL or lease.len != sizeof(int64_t):
            raise ValueError("unexpected native datetime scalar buffer")
        memcpy(&result, lease.buf, sizeof(int64_t))
        return result
    finally:
        PyBuffer_Release(&lease)


def try_valid_range(index, row, start_ns, end_ns):
    """Preserve the native date predicate, or decline before conversions.

    Start/end are copied and released in that order, then index is leased until
    the comparison finishes. Every successful export is released on all paths.
    NumPy may allocate its own exporter metadata: this is not allocation-free.
    """
    cdef Py_buffer lease
    cdef int64_t left, right, value
    cdef int64_t nat = -9223372036854775807 - 1
    cdef Py_ssize_t location, count
    cdef const char *data
    if (type(index) is not _RANGE_ARRAY or type(row) is not int
            or type(start_ns) is not _RANGE_DATETIME
            or type(end_ns) is not _RANGE_DATETIME):
        return NotImplemented
    if (index.ndim != 1 or index.dtype != _RANGE_NS_DTYPE
            or not index.dtype.isnative or not index.flags.c_contiguous
            or not index.flags.aligned
            or start_ns.dtype != _RANGE_NS_DTYPE or end_ns.dtype != _RANGE_NS_DTYPE):
        return NotImplemented
    count = len(index)
    if row < 0 or row > count:
        return NotImplemented
    if row == count:
        return False
    location = row
    left = _scalar_ticks(start_ns)
    right = _scalar_ticks(end_ns)
    PyObject_GetBuffer(index, &lease, PyBUF_SIMPLE)
    try:
        # Division/remainder validate the exact byte count without count*8
        # overflow; consequently every following offset fits Py_ssize_t.
        if (lease.buf == NULL or lease.len < 0
                or lease.len % sizeof(int64_t) != 0
                or lease.len // sizeof(int64_t) != count):
            return NotImplemented
        data = <const char *>lease.buf
        memcpy(&value, data + location * sizeof(int64_t), sizeof(int64_t))
        if value != nat and right != nat and value > right:
            return False
        if location > 0:
            memcpy(&value, data + (location - 1) * sizeof(int64_t), sizeof(int64_t))
            if value != nat and left != nat and value >= left:
                return False
        if location + 1 < count:
            memcpy(&value, data + (location + 1) * sizeof(int64_t), sizeof(int64_t))
            if value != nat and right != nat and value <= right:
                return False
        return True
    finally:
        PyBuffer_Release(&lease)
