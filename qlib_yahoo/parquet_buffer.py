"""Optional bounded row-group batching; the caller owns its Parquet writer.

Uses PyArrow 25's existing zero-copy concat_tables and write_table hooks.
Queued flat Arrow buffers are copied into immutable bytes to own their lifetime;
no closed-day immutability assumption is made about native NumPy inputs.
"""

from __future__ import annotations


class BufferedParquetBuffer:
    """Batch exact-schema flat tables without owning files, close or promotion.

    Call flush before closing a successfully completed writer. On cancellation,
    discard releases only queued RAM. A write failure permanently fails closed:
    queued tables remain inspectable until discard, and flush never retries.
    Bounds cover owned Arrow buffer bytes and queued rows, excluding incoming
    caller memory, Python wrappers/schema and the provided writer's scratch RAM.
    """

    def __init__(self, providedwriter, target_rows=32768, max_bytes=16 * 1024 * 1024):
        import pyarrow as pa
        for name, value in (("target_rows", target_rows), ("max_bytes", max_bytes)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(providedwriter.schema) is not pa.Schema:
            raise TypeError("provided writer must expose its exact Arrow schema")
        self.writer, self.schema = providedwriter, providedwriter.schema
        self.target_rows, self.max_bytes = target_rows, max_bytes
        self._tables = []
        self._failed = False
        self._stats = {"input_tables": 0, "input_rows": 0, "writer_calls": 0,
            "flushes": 0, "direct_tables": 0, "empty_tables": 0,
            "pending_rows": 0, "pending_bytes": 0, "pending_tables": 0,
            "peak_pending_rows": 0, "peak_pending_bytes": 0, "peak_pending_tables": 0}

    def stats(self):
        return {**self._stats, "failed": self._failed, "target_rows": self.target_rows,
                "max_bytes": self.max_bytes,
                "memory_scope": "owned Arrow buffers only; excludes incoming caller tables, Python metadata and writer scratch"}

    def _check_open(self):
        if self._failed:
            raise RuntimeError("Parquet buffer failed; refusing an implicit retry")

    def _write(self, table, *, grouped=False):
        try:
            if grouped:
                self.writer.write_table(table, row_group_size=self.target_rows)
            else:
                self.writer.write_table(table)
        except BaseException:
            self._failed = True
            raise
        self._stats["writer_calls"] += 1

    def write_table(self, table):
        import pyarrow as pa
        self._check_open()
        if type(table) is not pa.Table or not table.schema.equals(self.schema, check_metadata=True):
            raise ValueError("Buffered tables must match the writer schema and metadata exactly")
        # Flat types allow explicit ownership of every value/null/offset buffer.
        # Reject nested/dictionary/extension layouts rather than share hidden
        # child buffers which a caller could mutate behind the queue.
        for field in table.schema:
            kind = field.type
            if not (pa.types.is_primitive(kind) or pa.types.is_decimal(kind)
                    or pa.types.is_string(kind) or pa.types.is_large_string(kind)
                    or pa.types.is_binary(kind) or pa.types.is_large_binary(kind)
                    or pa.types.is_fixed_size_binary(kind) or pa.types.is_null(kind)):
                raise TypeError("Parquet buffer supports flat primitive/string/binary columns only")
        self._stats["input_tables"] += 1
        self._stats["input_rows"] += len(table)
        if not len(table):
            # Do not silently drop explicit empty writes or move them before
            # earlier nonempty sessions. They remain one provided-writer call.
            self.flush()
            self._write(table)
            self._stats["empty_tables"] += 1
            return
        size = sum(buffer.size for column in table.columns for chunk in column.chunks
                   if len(chunk) for buffer in chunk.buffers() if buffer is not None)
        if len(table) > self.target_rows or size > self.max_bytes:
            self.flush()
            self._write(table)
            self._stats["direct_tables"] += 1
            return
        if (self._stats["pending_rows"] + len(table) > self.target_rows
                or self._stats["pending_bytes"] + size > self.max_bytes):
            self.flush()
        columns = []
        for column in table.columns:
            chunks = [pa.Array.from_buffers(chunk.type, len(chunk),
                        [None if buffer is None else pa.py_buffer(bytes(buffer)) for buffer in chunk.buffers()],
                        null_count=chunk.null_count, offset=chunk.offset)
                      for chunk in column.chunks if len(chunk)]
            columns.append(pa.chunked_array(chunks, type=column.type))
        owned = pa.Table.from_arrays(columns, schema=self.schema)
        self._tables.append(owned)
        self._stats["pending_rows"] += len(owned)
        self._stats["pending_bytes"] += size
        self._stats["pending_tables"] += 1
        for key in ("rows", "bytes", "tables"):
            self._stats[f"peak_pending_{key}"] = max(self._stats[f"peak_pending_{key}"], self._stats[f"pending_{key}"])
        if self._stats["pending_rows"] == self.target_rows or self._stats["pending_bytes"] == self.max_bytes:
            self.flush()

    def flush(self):
        import pyarrow as pa
        self._check_open()
        if not self._tables:
            return
        try:
            combined = pa.concat_tables(self._tables, promote_options="none")
        except BaseException:
            self._failed = True
            raise
        self._write(combined, grouped=True)
        self._stats["flushes"] += 1
        self.discard()

    def discard(self):
        """Release pending RAM only; never write, close, unlink or clear failure."""
        self._tables.clear()
        for key in ("rows", "bytes", "tables"):
            self._stats[f"pending_{key}"] = 0
