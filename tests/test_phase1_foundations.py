"""
Phase 1 tests: config, Mongo layer, logger, data loader + validation report,
chronological splits with embargo, frozen boundaries, and the holdout lock.

Run:  python -m pytest tests/test_phase1_foundations.py -v
"""

from __future__ import annotations

import ast
import copy
import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.conftest import ROOT, make_bars


# ═══════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════

class TestConfig:
    def test_enabled_timeframes_default_is_1h_4h(self, base_cfg):
        assert base_cfg["data"]["enabled_timeframes"] == ["1h", "4h"]

    def test_split_ratios_must_sum_to_one(self, base_cfg):
        from core.config import _validate_config
        bad = copy.deepcopy(base_cfg)
        bad["splits"]["train_ratio"] = 0.7
        with pytest.raises(ValueError, match="sum to 1.0"):
            _validate_config(bad)

    def test_unknown_enabled_timeframe_rejected(self, base_cfg):
        from core.config import _validate_config
        bad = copy.deepcopy(base_cfg)
        bad["data"]["enabled_timeframes"] = ["1h", "2h"]
        with pytest.raises(ValueError, match="unknown timeframes"):
            _validate_config(bad)

    def test_config_hash_deterministic_and_sensitive(self, base_cfg):
        from core.config import config_hash
        a, b = config_hash(base_cfg), config_hash(copy.deepcopy(base_cfg))
        assert a == b and len(a) == 64
        c = copy.deepcopy(base_cfg)
        c["sizing"]["risk_per_trade"] = 0.02
        assert config_hash(c) != a

    def test_code_hash_normalises_whitespace(self):
        from core.config import code_hash
        assert code_hash("x = 1\r\ny = 2  \n") == code_hash("x = 1\ny = 2\n\n")
        assert code_hash("x = 1") != code_hash("x = 2")

    def test_no_secrets_in_config_yaml(self):
        import os
        text = (ROOT / "config.yaml").read_text(encoding="utf-8")
        for var in ("MONGO_URL", "LLM_API_KEY_1", "LLM_API_KEY_2"):
            val = os.environ.get(var)
            if val:
                assert val not in text, f"{var} value leaked into config.yaml"


# ═══════════════════════════════════════════════════════════════════════════
# Data loading & validation
# ═══════════════════════════════════════════════════════════════════════════

def _corrupt_and_validate(synth_cfg, mutate):
    """Apply `mutate` to the synthetic 1h CSV and run load_and_validate."""
    from core.data_loader import load_and_validate
    path = Path(synth_cfg["data"]["base_dir_resolved"]) / "H1.csv"
    df = pd.read_csv(path)
    df = mutate(df)
    df.to_csv(path, index=False)
    return load_and_validate("1h", synth_cfg)


class TestDataValidation:
    def test_clean_data_passes_and_gaps_are_explained(self, synth_cfg):
        from core.data_loader import load_and_validate
        for tf in ("1h", "4h"):
            df, fhash, rep = load_and_validate(tf, synth_cfg)
            assert rep["critical_errors"] == [], rep["critical_errors"]
            assert rep["passed"] is True
            assert rep["info"]["gaps"]["unexplained"] == 0, rep["warnings"]
            assert rep["info"]["gaps"]["weekend"] > 100
            assert len(fhash) == 64 and rep["file_hash"] == fhash
            assert str(df["timestamp"].dt.tz) == "UTC"

    def test_loader_never_modifies_values(self, synth_cfg):
        from core.data_loader import load_and_validate
        raw = pd.read_csv(Path(synth_cfg["data"]["base_dir_resolved"]) / "H1.csv")
        df, _, _ = load_and_validate("1h", synth_cfg)
        assert len(df) == len(raw)
        for col in ("open", "high", "low", "close", "volume"):
            np.testing.assert_array_equal(df[col].to_numpy(), raw[col].to_numpy())

    @pytest.mark.parametrize("name,mutate,expected_type", [
        ("duplicate", lambda d: pd.concat([d.iloc[:500], d.iloc[499:500], d.iloc[500:]]), "duplicate_timestamps"),
        ("out_of_order", lambda d: d.iloc[np.r_[0:300, 301, 300, 302:len(d)]], "out_of_order_rows"),
        ("nan_close", lambda d: d.assign(close=d["close"].where(d.index != 100)), "nan_values"),
        ("zero_price", lambda d: d.assign(low=d["low"].where(d.index != 200, 0.0)), "non_positive_price"),
        ("negative_price", lambda d: d.assign(open=d["open"].where(d.index != 50, -5.0)), "non_positive_price"),
        ("high_lt_low", lambda d: d.assign(high=d["high"].where(d.index != 700, d["low"] - 1)), "high_less_than_low"),
        ("open_outside", lambda d: d.assign(open=d["open"].where(d.index != 800, d["high"] + 2)), "open_outside_high_low"),
        ("close_outside", lambda d: d.assign(close=d["close"].where(d.index != 900, d["low"] - 2)), "close_outside_high_low"),
        ("negative_volume", lambda d: d.assign(volume=d["volume"].where(d.index != 10, -1.0)), "negative_volume"),
        ("misaligned", lambda d: d.assign(timestamp=d["timestamp"].where(
            d.index != 1000, (pd.to_datetime(d["timestamp"]) + pd.Timedelta(minutes=1)).dt.strftime("%Y-%m-%d %H:%M:%S"))),
         "misaligned_bar_timestamps"),
        ("six_day_hole", lambda d: d[~pd.to_datetime(d["timestamp"]).between("2022-06-07", "2022-06-14")],
         "data_hole_over_5_days"),
    ])
    def test_critical_error_detected_and_halts(self, synth_cfg, name, mutate, expected_type):
        from core.data_loader import DataValidationError
        with pytest.raises(DataValidationError) as exc:
            _corrupt_and_validate(synth_cfg, mutate)
        types = [e["type"] for e in exc.value.report["critical_errors"]]
        assert expected_type in types, f"{name}: got {types}"

    def test_report_without_halt_lists_errors(self, synth_cfg):
        cfg = copy.deepcopy(synth_cfg)
        cfg["validation"]["critical_errors_halt"] = False
        _, _, rep = _corrupt_and_validate(cfg, lambda d: d.assign(close=d["close"].where(d.index != 3)))
        assert rep["passed"] is False and rep["critical_errors"][0]["type"] == "nan_values"

    def test_spike_is_warning_not_fixed(self, synth_cfg):
        def spike(d):
            d = d.copy()
            d.loc[5000, "high"] = d.loc[5000, "high"] + 500.0
            return d
        df, _, rep = _corrupt_and_validate(synth_cfg, spike)
        assert rep["critical_errors"] == []
        sp = [w for w in rep["warnings"] if w["type"] == "price_spikes"]
        assert sp and sp[0]["isolated_wick_count"] >= 1
        raw = pd.read_csv(Path(synth_cfg["data"]["base_dir_resolved"]) / "H1.csv")
        assert df.loc[5000, "high"] == raw.loc[5000, "high"]  # untouched

    def test_holiday_gap_is_warning(self, synth_cfg):
        # Remove one trading day (a Wednesday) -> unexplained gap warning, not critical
        df, _, rep = _corrupt_and_validate(
            synth_cfg, lambda d: d[~pd.to_datetime(d["timestamp"]).between("2022-03-09 00:00", "2022-03-09 23:00")])
        assert rep["critical_errors"] == []
        assert any(w["type"] == "unexplained_gaps" for w in rep["warnings"])

    def test_file_hash_changes_with_content(self, synth_cfg):
        from core.data_loader import file_sha256
        p = Path(synth_cfg["data"]["base_dir_resolved"]) / "H1.csv"
        h1 = file_sha256(p)
        assert h1 == file_sha256(p)
        p.write_bytes(p.read_bytes().replace(b",", b";", 1))
        assert file_sha256(p) != h1


class TestTimezone:
    def test_broker_tz_conversion_handles_dst(self):
        from core.data_loader import convert_to_utc
        df = pd.DataFrame({"timestamp": ["2023-07-01 12:00:00", "2023-01-15 12:00:00"]})
        out, issues = convert_to_utc(df, "Europe/Athens")
        assert str(out["timestamp"].iloc[0]) == "2023-07-01 09:00:00+00:00"  # EEST = UTC+3
        assert str(out["timestamp"].iloc[1]) == "2023-01-15 10:00:00+00:00"  # EET  = UTC+2
        assert issues["dst_ambiguous_or_nonexistent"] == 0

    def test_ambiguous_dst_time_is_reported_not_dropped(self):
        from core.data_loader import convert_to_utc
        df = pd.DataFrame({"timestamp": ["2023-10-29 03:30:00", "2023-10-29 05:00:00"]})  # 03:30 occurs twice
        out, issues = convert_to_utc(df, "Europe/Athens")
        assert len(out) == 2
        assert issues["dst_ambiguous_or_nonexistent"] == 1

    @pytest.mark.parametrize("ts_utc,expected", [
        ("2023-07-12 07:00", "london"),             # 08:00 BST
        ("2023-01-11 07:00", "asia"),               # London 07:00 GMT closed, Tokyo 16:00 open
        ("2023-07-12 12:00", "overlap_london_ny"),  # NY 08:00 EDT
        ("2023-01-11 12:00", "london"),             # NY 07:00 EST not open yet
        ("2023-01-11 13:00", "overlap_london_ny"),
        ("2023-07-12 20:00", "new_york"),
        ("2023-07-12 21:30", "off_hours"),          # after NY close, before Tokyo
    ])
    def test_session_labels_follow_dst(self, base_cfg, ts_utc, expected):
        from core.data_loader import add_session_labels
        df = pd.DataFrame({"timestamp": pd.to_datetime([ts_utc]).tz_localize("UTC")})
        assert add_session_labels(df, base_cfg)["session"].iloc[0] == expected


# ═══════════════════════════════════════════════════════════════════════════
# Splits
# ═══════════════════════════════════════════════════════════════════════════

class TestSplitBoundaries:
    def test_pure_boundaries_chronology_and_embargo(self, base_cfg):
        from core.splits import compute_boundaries
        s, e = pd.Timestamp("2021-09-24", tz="UTC"), pd.Timestamp("2026-09-23", tz="UTC")
        b = compute_boundaries(s, e, base_cfg["splits"])
        assert b.train_start < b.train_end < b.validation_start < b.validation_end < b.holdout_start < b.holdout_end
        emb = pd.Timedelta(days=base_cfg["splits"]["embargo_days"])
        assert b.validation_start - b.train_end == emb
        assert b.holdout_start - b.validation_end == emb
        span = e - s
        assert abs((b.train_end - s) / span - 0.60) < 0.002
        assert abs((b.validation_end - s) / span - 0.80) < 0.002
        for t in (b.train_end, b.validation_end):
            assert t == t.floor("D")

    def test_too_short_range_rejected(self, base_cfg):
        from core.splits import compute_boundaries
        with pytest.raises(ValueError):
            compute_boundaries(pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-20", tz="UTC"),
                               base_cfg["splits"])


@pytest.fixture
def store(synth_cfg, test_db):
    from core.splits import DataStore, get_or_freeze_boundaries
    b = get_or_freeze_boundaries(synth_cfg, collection=test_db["split_boundaries"])
    return DataStore(synth_cfg, boundaries=b)


class TestSplitData:
    def test_no_overlap_embargo_respected_full_coverage(self, store, synth_cfg):
        from core.data_loader import load_and_validate
        b = store.boundaries
        emb = pd.Timedelta(days=synth_cfg["splits"]["embargo_days"])
        for tf in ("1h", "4h"):
            tr = store.get_data(tf, "train")
            va = store.get_data(tf, "validation")
            ho = store.get_data(tf, "holdout", strategy_id=f"chronology_{tf}")
            for d in (tr, va, ho):
                assert d["timestamp"].is_monotonic_increasing and d["timestamp"].is_unique
                assert not d["is_warmup"].any()
            # chronological, no overlap
            assert tr["timestamp"].max() < va["timestamp"].min() < va["timestamp"].max() < ho["timestamp"].min()
            assert not set(tr["timestamp"]) & set(va["timestamp"])
            assert not set(va["timestamp"]) & set(ho["timestamp"])
            # embargo: nothing from the embargo windows is served
            assert va["timestamp"].min() - tr["timestamp"].max() >= emb
            assert ho["timestamp"].min() - va["timestamp"].max() >= emb
            # train/val/holdout + embargo bars == every bar in the file
            full, _, _ = load_and_validate(tf, synth_cfg)
            in_embargo = (full["timestamp"].between(b.train_end, b.validation_start, inclusive="left") |
                          full["timestamp"].between(b.validation_end, b.holdout_start, inclusive="left"))
            assert len(tr) + len(va) + len(ho) + int(in_embargo.sum()) == len(full)
            # a bar never straddles a boundary: last train bar closes by train_end
            minutes = {"1h": 60, "4h": 240}[tf]
            assert tr["timestamp"].max() + pd.Timedelta(minutes=minutes) <= b.train_end

    def test_boundaries_common_to_all_timeframes(self, store):
        i1, i4 = store.split_info("1h"), store.split_info("4h")
        for split in ("train", "validation", "holdout"):
            assert i1[split]["start"] == i4[split]["start"]
            assert i1[split]["end_exclusive"] == i4[split]["end_exclusive"]

    def test_warmup_strictly_before_split_and_flagged(self, store):
        plain = store.get_data("1h", "validation")
        warm = store.get_data("1h", "validation", warmup_bars=300)
        assert warm["is_warmup"].sum() == 300
        lo = store.boundaries.validation_start
        assert (warm.loc[warm["is_warmup"], "timestamp"] < lo).all()
        assert (warm.loc[~warm["is_warmup"], "timestamp"] >= lo).all()
        core = warm.loc[~warm["is_warmup"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(core, plain)
        assert warm.attrs["slice_hash"] == plain.attrs["slice_hash"]

    def test_returns_copies(self, store):
        a = store.get_data("1h", "train")
        a.loc[0, "close"] = -1.0
        assert store.get_data("1h", "train").loc[0, "close"] != -1.0

    def test_attrs_carry_hashes(self, store):
        d = store.get_data("4h", "train")
        assert d.attrs["split"] == "train" and d.attrs["timeframe"] == "4h"
        assert len(d.attrs["file_hash"]) == 64 and len(d.attrs["slice_hash"]) == 64

    def test_disabled_timeframe_refused(self, synth_cfg, test_db):
        from core.splits import DataStore, TimeframeDisabledError, get_or_freeze_boundaries
        cfg = copy.deepcopy(synth_cfg)
        cfg["data"]["enabled_timeframes"] = ["4h"]
        st = DataStore(cfg, boundaries=get_or_freeze_boundaries(cfg, collection=test_db["split_boundaries"]))
        with pytest.raises(TimeframeDisabledError):
            st.get_data("1h", "train")

    def test_unknown_split_rejected(self, store):
        with pytest.raises(ValueError):
            store.get_data("1h", "test")


class TestFrozenBoundaries:
    def test_frozen_and_stable_when_data_appended(self, synth_cfg, test_db):
        from core.splits import get_or_freeze_boundaries
        col = test_db["split_boundaries"]
        b1 = get_or_freeze_boundaries(synth_cfg, collection=col)
        # Append 3 months of new data to both files -> boundaries must NOT move
        base = Path(synth_cfg["data"]["base_dir_resolved"])
        extra = make_bars("2024-01-01", "2024-04-01", "60min", seed=9)
        h1 = pd.concat([pd.read_csv(base / "H1.csv"), extra.assign(
            timestamp=extra["timestamp"].dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S"))])
        h1.to_csv(base / "H1.csv", index=False)
        b2 = get_or_freeze_boundaries(synth_cfg, collection=col)
        assert b1 == b2

    def test_changing_split_config_after_freeze_raises(self, synth_cfg, test_db):
        from core.splits import SplitConfigChangedError, get_or_freeze_boundaries
        col = test_db["split_boundaries"]
        get_or_freeze_boundaries(synth_cfg, collection=col)
        cfg = copy.deepcopy(synth_cfg)
        cfg["splits"]["embargo_days"] = 3
        with pytest.raises(SplitConfigChangedError):
            get_or_freeze_boundaries(cfg, collection=col)


# ═══════════════════════════════════════════════════════════════════════════
# Holdout lock
# ═══════════════════════════════════════════════════════════════════════════

class TestHoldoutLock:
    def test_second_access_raises(self, store, test_db):
        from core.splits import HoldoutLockError
        d = store.get_data("1h", "holdout", strategy_id="strat_A", purpose="test")
        assert len(d) > 0
        with pytest.raises(HoldoutLockError, match="already accessed"):
            store.get_data("1h", "holdout", strategy_id="strat_A")
        # Also on another timeframe: one access per strategy, full stop
        with pytest.raises(HoldoutLockError):
            store.get_data("4h", "holdout", strategy_id="strat_A")
        assert test_db["holdout_access"].count_documents({"strategy_id": "strat_A"}) == 1
        rec = test_db["holdout_access"].find_one({"strategy_id": "strat_A"})
        assert rec["timeframe"] == "1h" and rec["purpose"] == "test" and len(rec["file_hash"]) == 64

    def test_other_strategy_still_allowed(self, store):
        store.get_data("1h", "holdout", strategy_id="strat_B")
        store.get_data("1h", "holdout", strategy_id="strat_C")

    @pytest.mark.parametrize("sid", [None, "", 123])
    def test_missing_strategy_id_raises_and_records_nothing(self, store, test_db, sid):
        from core.splits import HoldoutLockError
        before = test_db["holdout_access"].count_documents({})
        with pytest.raises(HoldoutLockError):
            store.get_data("1h", "holdout", strategy_id=sid)
        assert test_db["holdout_access"].count_documents({}) == before

    def test_lock_survives_restart(self, synth_cfg, test_db):
        from core.splits import DataStore, HoldoutLockError, get_or_freeze_boundaries
        b = get_or_freeze_boundaries(synth_cfg, collection=test_db["split_boundaries"])
        DataStore(synth_cfg, boundaries=b).get_data("4h", "holdout", strategy_id="strat_restart")
        fresh = DataStore(synth_cfg, boundaries=b)  # simulates a new process
        with pytest.raises(HoldoutLockError):
            fresh.get_data("4h", "holdout", strategy_id="strat_restart")

    def test_concurrent_access_only_one_wins(self, store, test_db):
        from core.splits import HoldoutLockError
        store.get_data("1h", "train")  # warm cache so threads only race on the lock
        results: list[str] = []
        lock = threading.Lock()

        def worker():
            try:
                store.get_data("1h", "holdout", strategy_id="strat_race")
                r = "ok"
            except HoldoutLockError:
                r = "locked"
            with lock:
                results.append(r)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert results.count("ok") == 1 and results.count("locked") == 7
        assert test_db["holdout_access"].count_documents({"strategy_id": "strat_race"}) == 1

    def test_data_error_does_not_burn_holdout(self, synth_cfg, test_db):
        """If the data fails validation, the access must NOT be recorded."""
        from core.data_loader import DataValidationError
        from core.splits import DataStore, get_or_freeze_boundaries
        b = get_or_freeze_boundaries(synth_cfg, collection=test_db["split_boundaries"])
        p = Path(synth_cfg["data"]["base_dir_resolved"]) / "H1.csv"
        df = pd.read_csv(p)
        df.loc[10, "close"] = np.nan
        df.to_csv(p, index=False)
        with pytest.raises(DataValidationError):
            DataStore(synth_cfg, boundaries=b).get_data("1h", "holdout", strategy_id="strat_baddata")
        assert test_db["holdout_access"].count_documents({"strategy_id": "strat_baddata"}) == 0


# ═══════════════════════════════════════════════════════════════════════════
# Mongo layer & logger
# ═══════════════════════════════════════════════════════════════════════════

class TestMongoAndLogger:
    def test_indexes_idempotent_and_holdout_unique(self, test_db):
        from storage import mongo
        mongo.ensure_indexes()
        mongo.ensure_indexes()
        idx = test_db["holdout_access"].index_information()
        assert any(v.get("unique") and v["key"] == [("strategy_id", 1)] for v in idx.values())

    def test_trial_counter_atomic(self, test_db):
        from storage import mongo
        start = mongo.get_trial_count()
        got: list[int] = []
        lock = threading.Lock()

        def inc():
            for _ in range(10):
                v = mongo.increment_trial_counter()
                with lock:
                    got.append(v)

        ts = [threading.Thread(target=inc) for _ in range(5)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert mongo.get_trial_count() == start + 50
        assert sorted(got) == list(range(start + 1, start + 51))

    def test_logger_writes_mongo_and_jsonl(self, test_db, tmp_path):
        from storage.logger import QuantForgeLogger
        lg = QuantForgeLogger(jsonl_dir=str(tmp_path))
        lg.info("phase1 logger test", cycle_id="c-1", stage="unit", strategy_id="s-1")
        doc = test_db["logs"].find_one({"message": "phase1 logger test"})
        assert doc and doc["cycle_id"] == "c-1" and doc["stage"] == "unit" and doc["level"] == "INFO"
        lines = (tmp_path / "quantforge.jsonl").read_text(encoding="utf-8").strip().splitlines()
        rec = json.loads(lines[-1])
        assert rec["message"] == "phase1 logger test" and rec["strategy_id"] == "s-1"
        for h in list(lg._logger.handlers):
            h.close()


# ═══════════════════════════════════════════════════════════════════════════
# Codebase guard: holdout & raw data reachable only through the audited path
# ═══════════════════════════════════════════════════════════════════════════

RAW_ACCESS_ALLOWED = {
    "load_and_validate": {"core/splits.py", "core/data_loader.py", "main.py"},
    "load_raw_data": {"core/data_loader.py"},
    "record_holdout_access": {"core/splits.py", "storage/mongo.py"},
    "_frame": {"core/splits.py"},
    "_frames": {"core/splits.py"},
}


def test_raw_data_and_holdout_only_via_audited_path():
    violations = []
    for py in ROOT.rglob("*.py"):
        rel = py.relative_to(ROOT).as_posix()
        if rel.startswith("tests/"):
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            name = None
            if isinstance(node, ast.Attribute):
                name = node.attr
            elif isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, ast.alias):
                name = node.name
            if name in RAW_ACCESS_ALLOWED and rel not in RAW_ACCESS_ALLOWED[name]:
                violations.append(f"{rel}:{getattr(node, 'lineno', '?')} uses {name}")
    assert not violations, "Unaudited raw/holdout access:\n" + "\n".join(violations)
