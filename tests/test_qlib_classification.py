"""Point-in-time classification overlays are sealed and bound to their price source."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
import os
from unittest.mock import patch
from pathlib import Path

from qlib_yahoo.dataset import DatasetError, tree_sha256
from qlib_yahoo.classification import (
    create_classification_overlay,
    read_classification_overlay,
    classification_preview,
    classification_identity,
    load_classification_groups,
)
import qlib_yahoo.classification as classification


CALENDAR = ("2024-03-08", "2024-03-11", "2024-03-12", "2024-03-13")
READER_DEPENDENCIES = importlib.util.find_spec("numpy") is not None


def build_price_dataset(root: Path, *, note: str = "source") -> Path:
    dataset = root / "price"
    qlib = dataset / "qlib_data"
    (qlib / "calendars").mkdir(parents=True)
    (qlib / "instruments").mkdir()
    (qlib / "features" / "aaa").mkdir(parents=True)
    (qlib / "calendars" / "day.txt").write_text("\n".join(CALENDAR) + "\n", encoding="utf-8")
    (qlib / "instruments" / "all.txt").write_text(
        "AAA\t2024-03-08\t2024-03-13\nBBB\t2024-03-08\t2024-03-13\n", encoding="utf-8")
    (qlib / "features" / "aaa" / "close.day.bin").write_bytes(b"price-bin")
    tree, count = tree_sha256(qlib)
    dataset.mkdir(exist_ok=True)
    (dataset / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "status": "ready", "note": note,
        "qlib_binary_tree_sha256": tree, "qlib_binary_file_count": count,
    }, sort_keys=True) + "\n", encoding="utf-8")
    return dataset


def inventory_digest(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


class ClassificationOverlayTests(unittest.TestCase):
    def setUp(self):
        # The managed Windows sandbox may deny its default AppData temp root.
        self.tmp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.price = build_price_dataset(self.root)

    @unittest.skipUnless(READER_DEPENDENCIES, "optional NumPy reader environment required")
    def test_strict_open_weekend_and_causal_group_change(self):
        original = inventory_digest(self.price)
        events = [
            {"instrument": "AAA", "available_at": "2024-03-07T08:00:00-05:00",
             "industry": "1311", "sector": "Mining", "country": "US"},
            {"instrument": "AAA", "available_at": "2024-03-09T15:00:00-05:00",
             "industry": "1311", "sector": "Energy", "country": "US",
             "source_accession": "0000000000-24-000001", "provenance": {"field": "countryba"}},
            # Equality with Monday's 09:30 EDT open is strictly too late for Monday.
            {"instrument": "AAA", "available_at": "2024-03-11T09:30:00-04:00",
             "industry": "2111", "sector": "Energy", "country": "CA"},
        ]
        overlay_path = create_classification_overlay(
            self.price, self.root / "overlays" / "sic", events,
            source_metadata={"mapping_policy": "current-map research proxy"},
            taxonomy_metadata={"industry": "SIC 4-digit", "sector": "SIC division", "country": "countryba"},
        )
        self.assertEqual(inventory_digest(self.price), original)
        overlay = read_classification_overlay(self.price, overlay_path)
        self.assertEqual(overlay.calendar, CALENDAR)
        self.assertEqual(overlay.symbols, ("AAA", "BBB"))
        # Fri before the weekend filing: initial industry/sector state.
        self.assertEqual(overlay.codes["industry"][:, 0].tolist(), [0, 0, 1, 1])
        # Weekend event applies Monday 09:30 ET; the equal-open update starts Tuesday.
        self.assertEqual(overlay.codes["sector"][:, 0].tolist(), [1, 0, 0, 0])
        self.assertEqual(overlay.codes["country"][:, 0].tolist(), [1, 1, 0, 0])
        self.assertEqual(overlay.labels["industry"], ("1311", "2111"))
        self.assertEqual(overlay.codes["industry"][:, 1].tolist(), [-1, -1, -1, -1])
        self.assertEqual(overlay.identity["taxonomy_metadata"]["sector"], "SIC division")
        self.assertEqual(overlay.coverage["industry"]["labeled_cells"], 4)
        self.assertFalse(overlay.stale.any())

    @unittest.skipUnless(READER_DEPENDENCIES, "optional NumPy reader environment required")
    def test_dst_open_conversion_and_bounded_selection(self):
        overlay_path = create_classification_overlay(self.price, self.root / "dst", [
            {"instrument": "AAA", "available_at": "2024-03-11T13:30:00Z", "industry": "A"},
        ])
        # 13:30Z is exactly 09:30 EDT after the DST transition, so Tuesday is first.
        overlay = read_classification_overlay(self.price, overlay_path, start="2024-03-11",
                                              end="2024-03-13", symbols=["AAA"])
        self.assertEqual(overlay.calendar, CALENDAR[1:])
        self.assertEqual(overlay.symbols, ("AAA",))
        self.assertEqual(overlay.codes["industry"][:, 0].tolist(), [-1, 0, 0])

    @unittest.skipUnless(READER_DEPENDENCIES, "optional NumPy reader environment required")
    def test_missing_and_stale_are_unknown_without_market_fallback(self):
        overlay_path = create_classification_overlay(self.price, self.root / "stale", [
            {"instrument": "AAA", "available_at": "2024-03-11T09:29:00-04:00",
             "industry": "1311", "sector": "Energy"},
        ], max_age_days=1)
        overlay = read_classification_overlay(self.price, overlay_path)
        # It is available one minute before Monday's open, then stale after one
        # elapsed day by Tuesday's open.
        self.assertEqual(overlay.codes["industry"][:, 0].tolist(), [-1, 0, -1, -1])
        self.assertEqual(overlay.codes["sector"][:, 1].tolist(), [-1, -1, -1, -1])
        self.assertEqual(overlay.stale[:, 0].tolist(), [False, False, True, True])
        self.assertEqual(overlay.coverage["industry"]["stale_cells"], 2)

    def test_duplicate_conflict_and_immutable_destination(self):
        event = {"instrument": "AAA", "available_at": "2024-03-08T08:00:00-05:00", "industry": "A"}
        with self.assertRaisesRegex(DatasetError, "conflicting"):
            create_classification_overlay(self.price, self.root / "bad", [event, {**event, "industry": "B"}])
        path = create_classification_overlay(self.price, self.root / "good", [event])
        before = inventory_digest(path)
        with self.assertRaisesRegex(DatasetError, "already exists"):
            create_classification_overlay(self.price, path, [event])
        self.assertEqual(inventory_digest(path), before)
        self.assertEqual(json.loads((path / "manifest.json").read_text(encoding="utf-8"))["events"]["count"], 1)

    @unittest.skipUnless(READER_DEPENDENCIES, "optional NumPy reader environment required")
    def test_same_classification_and_availability_coalesces_source_accessions(self):
        timestamp = "2024-03-08T08:00:00-05:00"
        events = [
            {"instrument": "AAA", "available_at": timestamp, "industry": "1311",
             "source_accession": "accession-1", "provenance": {"accepted": "2024-03-08"}},
            {"instrument": "AAA", "available_at": timestamp, "industry": "1311",
             "source_accession": "accession-2", "provenance": {"accepted": "2024-03-08"}},
        ]
        path = create_classification_overlay(self.price, self.root / "coalesced", events)
        overlay = read_classification_overlay(self.price, path)
        self.assertEqual(overlay.codes["industry"][:, 0].tolist(), [0, 0, 0, 0])
        stored = json.loads((path / "events.jsonl").read_text(encoding="utf-8"))
        source_records = stored["provenance"]["classification_overlay_source_records"]
        self.assertEqual({record["source_accession"] for record in source_records}, {"accession-1", "accession-2"})

    def test_event_or_manifest_tamper_and_price_binding_mismatch_fail(self):
        overlay_path = create_classification_overlay(self.price, self.root / "sealed", [
            {"instrument": "AAA", "available_at": "2024-03-07T08:00:00-05:00", "industry": "A"},
        ])
        original_events = (overlay_path / "events.jsonl").read_bytes()
        (overlay_path / "events.jsonl").write_bytes(original_events + b" ")
        with self.assertRaisesRegex(DatasetError, "events SHA-256"):
            read_classification_overlay(self.price, overlay_path)
        (overlay_path / "events.jsonl").write_bytes(original_events)
        (overlay_path / "manifest.json").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(DatasetError, "manifest SHA-256"):
            read_classification_overlay(self.price, overlay_path)

        second_price = build_price_dataset(self.root / "other", note="different manifest, same prices")
        other_overlay = create_classification_overlay(second_price, self.root / "other-overlay", [])
        with self.assertRaisesRegex(DatasetError, "different price snapshot"):
            read_classification_overlay(self.price, other_overlay)

    def test_invalid_availability_inventory_and_selection_fail(self):
        for value in ("2024-03-08T09:30:00", "not-a-time"):
            with self.subTest(value=value), self.assertRaises(DatasetError):
                create_classification_overlay(self.price, self.root / f"bad-{len(value)}", [
                    {"instrument": "AAA", "available_at": value, "industry": "A"},
                ])
        with self.assertRaisesRegex(DatasetError, "absent from price inventory"):
            create_classification_overlay(self.price, self.root / "unknown", [
                {"instrument": "NOPE", "available_at": "2024-03-07T08:00:00Z", "industry": "A"},
            ])
        overlay_path = create_classification_overlay(self.price, self.root / "valid", [])
        with self.assertRaisesRegex(DatasetError, "exactly match"):
            read_classification_overlay(self.price, overlay_path, start="2024-03-09")


@unittest.skipUnless(READER_DEPENDENCIES, "optional NumPy reader environment required")
class ClassificationCacheTests(unittest.TestCase):
    setUp = ClassificationOverlayTests.setUp
    def make_overlay(self, events=None, **kwargs):
        return create_classification_overlay(self.price, self.root / "overlay", events or [
            {"instrument": "AAA", "available_at": "2024-03-07T14:00:00Z", "industry": "A"},
        ], **kwargs)

    def load(self, path, identity=None):
        if identity is None:
            identity = classification_identity(self.price, path)
        return load_classification_groups(self.price, path, cache_root=self.root / "cache",
                                          expected_identity=identity)

    def test_preview_never_reads_events_or_binary_tree_and_stays_compact(self):
        path = self.make_overlay(source_metadata={"diagnostics": "x" * 800000,
            "availability_policy": "strict", "historical_identity_limit": "current ticker proxy"})
        original = Path.read_bytes
        def guarded(file):
            self.assertNotEqual(file.name, "events.jsonl")
            self.assertNotEqual(file.suffix, ".bin")
            return original(file)
        with patch.object(Path, "read_bytes", guarded), patch.object(
                classification, "verify_dataset", side_effect=AssertionError("full verification")), patch.object(
                Path, "rglob", side_effect=AssertionError("tree traversal")), patch.object(
                classification, "sha256", side_effect=AssertionError("bulk hash")):
            preview = classification_preview(self.price, path)
        self.assertEqual(preview["groups"], ["industry"])
        self.assertEqual(preview["status"], "metadata_only")
        self.assertLess(len(json.dumps(preview)), 2000)
        self.assertEqual({k: v for k, v in preview.items() if k != "status"},
                         classification_identity(self.price, path))

    def test_binding_mismatch_fails_before_bulk_reads(self):
        path = self.make_overlay()
        other = build_price_dataset(self.root / "other", note="wrong binding")
        with patch.object(classification, "verify_dataset", side_effect=AssertionError("bulk price")), patch.object(
                classification, "sha256", side_effect=AssertionError("bulk events")):
            for api in (classification_preview, classification_identity):
                with self.assertRaisesRegex(DatasetError, "different price snapshot"):
                    api(other, path)
            with self.assertRaisesRegex(DatasetError, "different price snapshot"):
                load_classification_groups(other, path, cache_root=self.root / "cache", expected_identity={})

    def test_cold_warm_owned_arrays_and_single_verification_per_load(self):
        import numpy as np
        path = self.make_overlay()
        identity = classification_identity(self.price, path)
        with patch.object(classification, "verify_dataset", wraps=classification.verify_dataset) as verify, patch.object(
                classification, "_parse_events", wraps=classification._parse_events) as parse:
            cold = self.load(path, identity)
            self.assertEqual(verify.call_count, 1)
            self.assertEqual(parse.call_count, 1)
        with patch.object(classification, "_parse_events", side_effect=AssertionError("warm parse")), patch.object(
                classification, "verify_dataset", wraps=classification.verify_dataset) as verify:
            warm = self.load(path, identity)
            self.assertEqual(verify.call_count, 1)
        self.assertEqual(cold.identity, identity)
        self.assertEqual(warm.identity, identity)
        self.assertEqual(cold.coverage, warm.coverage)
        for overlay in (cold, warm):
            for array in (*overlay.codes.values(), overlay.stale):
                self.assertTrue(array.flags.owndata)
                self.assertFalse(array.flags.writeable)
                self.assertNotIsInstance(array, np.memmap)
            self.assertEqual(overlay.codes["industry"].dtype, np.dtype("int32"))
            self.assertEqual(overlay.codes["industry"][:, 0].tolist(), [0, 0, 0, 0])
            self.assertEqual(overlay.codes["industry"][:, 1].tolist(), [-1, -1, -1, -1])

    def test_warm_cache_cannot_mask_same_size_events_or_prices_tamper(self):
        path = self.make_overlay()
        identity = classification_identity(self.price, path)
        self.load(path, identity)
        targets = [path / "events.jsonl", self.price / "qlib_data/features/aaa/close.day.bin"]
        for target in targets:
            original = target.read_bytes()
            before = target.stat()
            modified = original.replace(b'"industry":"A"', b'"industry":"B"') if target.name == "events.jsonl" else b"PRICE-bin"
            self.assertEqual(len(original), len(modified))
            target.write_bytes(modified)
            os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
            for action in (lambda: classification_identity(self.price, path), lambda: self.load(path, identity)):
                with self.assertRaises(DatasetError):
                    action()
            target.write_bytes(original)

    def test_resealed_source_mutation_after_identity_fails(self):
        path = self.make_overlay()
        identity = classification_identity(self.price, path)
        self.load(path, identity)
        manifest = json.loads((path / "manifest.json").read_bytes())
        manifest["max_age_days"] = 1
        payload = json.dumps(manifest).encode()
        (path / "manifest.json").write_bytes(payload)
        (path / "manifest.sha256").write_text(hashlib.sha256(payload).hexdigest() + "  manifest.json\n")
        with self.assertRaisesRegex(DatasetError, "expected identity"):
            self.load(path, identity)

    def test_cache_corruption_fails_closed_without_source_write(self):
        path = self.make_overlay()
        source_before = inventory_digest(path)
        identity = classification_identity(self.price, path)
        self.load(path, identity)
        cached = next((self.root / "cache").glob("*/industry.npy"))
        payload = cached.read_bytes()
        cached.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
        with self.assertRaisesRegex(DatasetError, "cache array checksum"):
            self.load(path, identity)
        self.assertEqual(inventory_digest(path), source_before)

    def test_null_clears_latest_same_open_wins_and_exact_expiry(self):
        path = self.make_overlay([
            {"instrument": "AAA", "available_at": "2024-03-07T14:30:00Z", "industry": "A"},
            {"instrument": "AAA", "available_at": "2024-03-09T00:00:00Z", "industry": "B"},
            {"instrument": "AAA", "available_at": "2024-03-10T00:00:00Z", "industry": None},
            {"instrument": "AAA", "available_at": "2024-03-11T13:30:00Z", "industry": "C"},
        ], max_age_days=1)
        result = self.load(path)
        # Fri is exactly one elapsed day old; Monday null replaces both earlier states.
        # Tuesday C is exactly one day old; Wednesday is stale.
        self.assertEqual(result.codes["industry"][:, 0].tolist(), [0, -1, 2, -1])
        self.assertEqual(result.stale[:, 0].tolist(), [False, True, False, True])

    def test_cache_shape_validation_even_with_updated_checksum(self):
        import numpy as np
        path = self.make_overlay()
        identity = classification_identity(self.price, path)
        self.load(path, identity)
        directory = next((self.root / "cache").iterdir())
        np.save(directory / "industry.npy", np.zeros((1, 1), dtype=np.int32))
        manifest = json.loads((directory / "metadata.json").read_bytes())
        manifest["arrays"]["industry"] = hashlib.sha256((directory / "industry.npy").read_bytes()).hexdigest()
        payload = json.dumps(manifest).encode()
        (directory / "metadata.json").write_bytes(payload)
        (directory / "metadata.sha256").write_text(hashlib.sha256(payload).hexdigest())
        with self.assertRaisesRegex(DatasetError, "shape or dtype"):
            self.load(path, identity)

    def test_fall_dst_equality_and_windows_fallback(self):
        from datetime import date
        from datetime import timezone
        with patch.object(classification, "_NY", None):
            self.assertEqual(classification._new_york_open(date(2024, 11, 1)).hour, 13)
            self.assertEqual(classification._new_york_open(date(2024, 11, 4)).hour, 14)
        self.assertEqual(classification._new_york_open(date(2024, 11, 4)).tzinfo, timezone.utc)
        calendar = ("2024-11-01", "2024-11-04", "2024-11-05", "2024-11-06")
        (self.price / "qlib_data/calendars/day.txt").write_text("\n".join(calendar) + "\n")
        tree, count = tree_sha256(self.price / "qlib_data")
        (self.price / "manifest.json").write_text(json.dumps({"status": "ready",
            "qlib_binary_tree_sha256": tree, "qlib_binary_file_count": count}))
        path = self.make_overlay([
            {"instrument": "AAA", "available_at": "2024-11-01T13:30:00Z", "industry": "A"},
            {"instrument": "AAA", "available_at": "2024-11-04T14:30:00Z", "industry": "B"},
        ])
        with patch.object(classification, "_NY", None):
            overlay = self.load(path)
        self.assertEqual(overlay.codes["industry"][:, 0].tolist(), [-1, 0, 1, 1])

    def test_packed_archive_tamper_rejected_after_warm_cache(self):
        from qlib_yahoo.packed import pack_dataset
        self.price = pack_dataset(self.price, self.root / "packed")
        path = self.make_overlay()
        identity = classification_identity(self.price, path)
        self.load(path, identity)
        archive = self.price / "qlib_data.zip"
        original = archive.read_bytes()
        before = archive.stat()
        archive.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
        os.utime(archive, ns=(before.st_atime_ns, before.st_mtime_ns))
        # Metadata-only preview deliberately makes no archive-integrity claim.
        self.assertEqual(classification_preview(self.price, path)["status"], "metadata_only")
        with self.assertRaises(DatasetError):
            classification_identity(self.price, path)
        with self.assertRaises(DatasetError):
            self.load(path, identity)

    def test_cold_cache_rejects_resealed_noncanonical_event(self):
        path = self.make_overlay()
        payload = (path / "events.jsonl").read_bytes().replace(b'"industry":"A"', b'"industry": "A"')
        (path / "events.jsonl").write_bytes(payload)
        manifest = json.loads((path / "manifest.json").read_bytes())
        manifest["events"]["sha256"] = hashlib.sha256(payload).hexdigest()
        manifest_bytes = json.dumps(manifest).encode()
        (path / "manifest.json").write_bytes(manifest_bytes)
        (path / "manifest.sha256").write_text(hashlib.sha256(manifest_bytes).hexdigest() + "  manifest.json\n")
        identity = classification_identity(self.price, path)
        with self.assertRaisesRegex(DatasetError, "canonically encoded"):
            self.load(path, identity)
        self.assertFalse((self.root / "cache").exists())


@unittest.skipUnless(os.environ.get("QLIB_CLASSIFICATION_REAL_SMOKE") == "1" and READER_DEPENDENCIES,
                     "opt-in local real-source smoke")
class RealClassificationSmoke(unittest.TestCase):
    def test_real_packed_binding_cold_and_warm_cache(self):
        import time
        import numpy as np
        price = Path("D:/Qlib/datasets/stooq-table-top3000-2012-packed")
        path = Path("D:/Qlib/classifications/sec-fsds-groups-20261006-v1")
        cache_parent = Path.cwd() / ".qlib-pilot"
        cache_parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="classification-smoke-", dir=cache_parent) as cache:
            preview = classification_preview(price, path)
            self.assertLess(len(json.dumps(preview)), 10000)
            identity = classification_identity(price, path)
            self.assertEqual(identity["price_source"]["manifest_sha256"],
                             "ebaa21347544b7da88af488558fa3c323cdceb2fdbbb81f92edfa87a1dab8f3e")
            started = time.perf_counter()
            cold = load_classification_groups(price, path, cache_root=Path(cache), expected_identity=identity)
            cold_seconds = time.perf_counter() - started
            sample = {g: a[:5, :5].copy() for g, a in cold.codes.items()}
            coverage = cold.coverage
            self.assertEqual(cold.codes["industry"].shape, (3697, 5329))
            self.assertGreater(coverage["industry"]["labeled_cells"], 0)
            self.assertGreater(coverage["industry"]["unknown_cells"], 0)
            del cold
            with patch.object(classification, "_parse_events", side_effect=AssertionError("warm parse")):
                started = time.perf_counter()
                warm = load_classification_groups(price, path, cache_root=Path(cache), expected_identity=identity)
                warm_seconds = time.perf_counter() - started
            self.assertEqual(warm.coverage, coverage)
            for group in classification.GROUPS:
                np.testing.assert_array_equal(warm.codes[group][:5, :5], sample[group])
                self.assertTrue(warm.codes[group].flags.owndata)
            print(json.dumps({"real_classification_smoke": "passed", "shape": [3697, 5329],
                              "cold_seconds": round(cold_seconds, 3), "warm_seconds": round(warm_seconds, 3),
                              "events": coverage["overlay"]["events"]}))


if __name__ == "__main__":
    unittest.main()
