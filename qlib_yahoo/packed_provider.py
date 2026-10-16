"""Qlib-facing provider, imported only when Qlib config constructs it."""

from qlib.data.data import FeatureProvider
from qlib.utils import code_to_fname

from qlib_yahoo.packed import DatasetError, verified_entries


class PackedFeatureProvider(FeatureProvider):
    def __init__(self, dataset: str, archive_sha256: str):
        self.entries = verified_entries(dataset, archive_sha256)

    def feature(self, instrument, field, start_index, end_index, freq):
        import numpy as np
        import pandas as pd

        if str(freq).lower() != "day":
            raise DatasetError("packed feature provider supports day frequency only")
        name = f"features/{code_to_fname(instrument).lower()}/{str(field)[1:].lower()}.day.bin"
        data = self.entries.get(name)
        if data is None:
            return pd.Series(dtype=np.float32)
        values = np.frombuffer(data, dtype="<f")
        if not len(values):
            raise DatasetError(f"empty packed feature: {name}")
        first = int(values[0])
        start = first if start_index is None else max(first, start_index)
        end = first + len(values) - 2 if end_index is None else end_index
        if start > end:
            return pd.Series(dtype=np.float32)
        selected = values[1 + start - first:2 + end - first]
        return pd.Series(selected, index=pd.RangeIndex(start, start + len(selected)))
