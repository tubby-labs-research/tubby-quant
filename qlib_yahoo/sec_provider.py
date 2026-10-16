"""Optional Qlib daily feature overlay for shared selected SEC matrices."""
from __future__ import annotations

try:
    from qlib.data.data import FeatureProvider
except ModuleNotFoundError as exc:
    if exc.name != "qlib" and not str(exc.name).startswith("qlib."):
        raise
    class FeatureProvider:
        """Import-only stand-in; actual provider use needs Qlib."""

from qlib_yahoo.sec_features import _symbol, close_arrays, load_arrays


class SECFeatureProvider(FeatureProvider):
    def __init__(self, handle: dict, price_provider=None):
        self._closed = False
        self._owns_price_provider = price_provider is None or isinstance(price_provider, dict)
        self.arrays = load_arrays(handle)
        try:
            self.symbols = {}
            for index, symbol in enumerate(handle["identity"]["symbols"]):
                normalized = _symbol(symbol)
                if normalized in self.symbols:
                    raise ValueError(f"Ambiguous SEC panel symbol: {symbol}")
                self.symbols[normalized] = index
            self.length = len(handle["identity"]["dates"])
            if price_provider is None:
                from qlib.data.data import LocalFeatureProvider
                self.price_provider = LocalFeatureProvider()
            elif isinstance(price_provider, dict):
                from qlib.utils import init_instance_by_config
                self.price_provider = init_instance_by_config(price_provider, accept_types=FeatureProvider)
            else:
                self.price_provider = price_provider
        except BaseException:
            self.close()
            raise

    def close(self):
        if getattr(self, "_closed", True):
            return
        self._closed = True
        close_arrays(getattr(self, "arrays", {}))
        self.arrays = {}
        price = getattr(self, "price_provider", None)
        if self._owns_price_provider and callable(getattr(price, "close", None)):
            price.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def feature(self, instrument, field, start_index=None, end_index=None, freq="day"):
        import numpy as np
        import pandas as pd

        if self._closed:
            raise ValueError("SEC feature provider is closed")
        alias = str(field).lstrip("$")
        if alias not in self.arrays:
            if alias.startswith("sec_"):
                raise ValueError(f"Unknown SEC field: {alias}")
            return self.price_provider.feature(instrument, field, start_index, end_index, freq)
        if str(freq).lower() != "day":
            raise ValueError("SEC features support daily frequency only")
        start = max(0, 0 if start_index is None else int(start_index))
        end = min(self.length - 1, self.length - 1 if end_index is None else int(end_index))
        column = self.symbols.get(_symbol(instrument))
        if start > end or column is None:
            return pd.Series(dtype=np.float32)
        # Returned Series must remain valid after the provider releases its maps.
        return pd.Series(np.array(self.arrays[alias][start:end + 1, column], copy=True),
                         index=pd.RangeIndex(start, end + 1))
