"""The Qlib overlay uses absolute calendar indexes and no per-call verification."""
import importlib.util
import unittest
from unittest.mock import Mock, patch

HAVE_DEPS = all(importlib.util.find_spec(name) for name in ("numpy", "pandas"))


@unittest.skipUnless(HAVE_DEPS, "requires NumPy and pandas")
class SECProviderTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from qlib_yahoo.sec_provider import SECFeatureProvider
        self.delegate = Mock()
        self.handle = {"identity": {"symbols": ["BRK.A.US", "MSFT.US"],
                                    "dates": ["2025-01-30", "2025-01-31", "2025-02-03"]}}
        with patch("qlib_yahoo.sec_provider.load_arrays", return_value={"sec_income": np.array(
                [[np.nan, np.nan], [3, 10], [4, 11]], dtype=np.float32)}) as loader:
            self.provider = SECFeatureProvider(self.handle, self.delegate)
            self.assertEqual(loader.call_count, 1)
        self.addCleanup(self.provider.close)

    def test_absolute_indexes_alias_mapping_nan_and_no_reverify(self):
        with patch("qlib_yahoo.sec_provider.load_arrays", side_effect=AssertionError("reverify")):
            series = self.provider.feature("BRK-A", "$sec_income", 1, 8, "day")
            self.assertEqual(list(series.index), [1, 2])
            self.assertEqual(list(series.values), [3, 4])
            self.assertTrue(self.provider.feature("MSFT", "$sec_income", 0, 0, "day").isna().all())

    def test_delegates_only_non_sec_and_rejects_unknown_frequency(self):
        self.provider.feature("MSFT", "$close", 2, 5, "day")
        self.delegate.feature.assert_called_once_with("MSFT", "$close", 2, 5, "day")
        with self.assertRaisesRegex(ValueError, "Unknown SEC"):
            self.provider.feature("MSFT", "$sec_missing", 0, 1, "day")
        with self.assertRaisesRegex(ValueError, "daily"):
            self.provider.feature("MSFT", "$sec_income", 0, 1, "week")
        self.assertTrue(self.provider.feature("MISSING", "$sec_income", 0, 1, "day").empty)
        self.assertTrue(self.provider.feature("MSFT", "$sec_income", 3, 2, "day").empty)

    def test_constructor_failure_closes_maps(self):
        from qlib_yahoo.sec_provider import SECFeatureProvider
        array = Mock()
        handle = {"identity": {"symbols": ["BRK.A", "BRK-A"], "dates": []}}
        with patch("qlib_yahoo.sec_provider.load_arrays", return_value={"sec_income": array}):
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                SECFeatureProvider(handle, self.delegate)
        array._mmap.close.assert_called_once()

    def test_close_is_idempotent_and_returned_series_survives(self):
        import numpy as np
        from pathlib import Path
        import tempfile
        from qlib_yahoo.sec_provider import SECFeatureProvider
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            file = Path(temp) / "field.npy"
            np.save(file, np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32))
            values = np.load(file, mmap_mode="r")
            with patch("qlib_yahoo.sec_provider.load_arrays", return_value={"sec_income": values}):
                provider = SECFeatureProvider(self.handle, self.delegate)
            try:
                series = provider.feature("MSFT", "$sec_income", 1, 2)
                provider.close()
                provider.close()
                self.assertTrue(values._mmap.closed)
                self.assertEqual(list(series.values), [4, 6])
                self.delegate.close.assert_not_called()
                with self.assertRaisesRegex(ValueError, "closed"):
                    provider.feature("MSFT", "$sec_income", 1, 2)
            finally:
                provider.close()
