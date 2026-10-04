"""
Phase F1 regression tests: Data integrity and the data vault.

Defects covered (see docs/AUDIT.md):
    DATA-1  patch_hole aligns to midnight instead of NY-close grid; overwrites source
    DATA-2  no data lock (file hash and row count not stored at split freeze)
    DATA-3  CSV re-read and re-parsed on every trial (no cache)
    SBX-2   raw price reading in strategy (data vault keeps validation/holdout encrypted)
"""

from __future__ import annotations

import copy
import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from core.config import load_config
from core.data_loader import (
    clear_frame_cache,
    data_path,
    file_sha256,
    get_cache_stats,
    load_and_validate,
)
from core.data_vault import (
    decrypt_vault,
    encrypt_splits,
    generate_vault_key,
    load_vault_split,
    save_vault,
    strip_working_csv,
    vault_path,
)
from core.splits import (
    DataLockError,
    DataStore,
    HoldoutLockError,
    compute_boundaries,
    compute_timeframe_locks,
    get_or_freeze_boundaries,
    refreeze,
)
from scripts.patch_hole import aggregate_to_grid, patch_gap
from scripts.verify_alignment import verify_file_alignment
from tests.conftest import make_bars


@pytest.fixture
def isolated_data_env(base_cfg, tmp_path):
    """Create an isolated directory with synthetic 1h and 4h data files."""
    cfg = copy.deepcopy(base_cfg)
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    cfg["data"]["base_dir_resolved"] = str(data_dir)

    # 1h and 4h spanning 3 years so splits (60/20/20) + embargo fit comfortably
    df_1h = make_bars("2021-01-04 00:00:00", "2024-01-01 00:00:00", freq="60min", seed=42)
    p_1h = data_dir / cfg["data"]["files"]["1h"]
    df_1h.to_csv(p_1h, index=False)

    df_4h = make_bars("2021-01-04 00:00:00", "2024-01-01 00:00:00", freq="240min", seed=42)
    p_4h = data_dir / cfg["data"]["files"]["4h"]
    df_4h.to_csv(p_4h, index=False)

    clear_frame_cache()
    return cfg, data_dir


# ═══════════════════════════════════════════════════════════════════════════
# DATA-2: Data Lock & Refreeze
# ═══════════════════════════════════════════════════════════════════════════

class TestDataLock:
    def test_freeze_stores_file_hashes_and_row_counts(self, isolated_data_env, fresh_db):
        cfg, _ = isolated_data_env
        b = get_or_freeze_boundaries(cfg)
        assert "1h" in b.timeframe_locks
        assert "sha256" in b.timeframe_locks["1h"]
        assert b.timeframe_locks["1h"]["row_count"] > 0

    def test_data_lock_refuses_when_csv_modified(self, isolated_data_env, fresh_db):
        cfg, data_dir = isolated_data_env
        get_or_freeze_boundaries(cfg)

        # Tamper with 1h CSV file (append 1 bar)
        p_1h = data_dir / cfg["data"]["files"]["1h"]
        with open(p_1h, "a") as f:
            f.write("2024-01-01 01:00:00+00:00,1800.0,1805.0,1795.0,1802.0,100.0\n")

        clear_frame_cache()
        ds = DataStore(cfg)
        with pytest.raises(DataLockError, match="Data lock violation"):
            ds.get_data("1h", "train")

    def test_refreeze_requires_exact_phrase(self, isolated_data_env, fresh_db):
        cfg, _ = isolated_data_env
        get_or_freeze_boundaries(cfg)

        with pytest.raises(ValueError, match="Confirmation phrase mismatch"):
            refreeze("1h", "wrong phrase", cfg)

    def test_refreeze_audited_workflow(self, isolated_data_env, fresh_db):
        from storage.mongo import col_backtests, col_holdout_access, col_logs
        cfg, data_dir = isolated_data_env
        ds = DataStore(cfg)
        # Record a holdout access
        ds.get_data("1h", "holdout", strategy_id="strat_test_refreeze")
        assert col_holdout_access().count_documents({"strategy_id": "strat_test_refreeze"}) == 1

        # Insert a backtest record
        col_backtests().insert_one({"strategy_id": "strat_test_refreeze", "timeframe": "1h", "data_changed": False})

        # Append row and refreeze
        p_1h = data_dir / cfg["data"]["files"]["1h"]
        with open(p_1h, "a") as f:
            f.write("2024-01-01 01:00:00+00:00,1800.0,1805.0,1795.0,1802.0,100.0\n")

        rep = refreeze("1h", "REFREEZE 1h", cfg)
        assert rep["new_hash"] != rep["old_hash"]
        assert rep["invalidated_holdout_accesses"] >= 1
        assert rep["marked_backtests"] >= 1

        # Holdout access was wiped for this timeframe
        assert col_holdout_access().count_documents({"strategy_id": "strat_test_refreeze"}) == 0
        # Backtest was marked data_changed
        bt = col_backtests().find_one({"strategy_id": "strat_test_refreeze"})
        assert bt["data_changed"] is True
        # Audit log written
        audit = col_logs().find_one({"action": "refreeze", "timeframe": "1h"})
        assert audit is not None
        assert audit["level"] == "WARNING"


# ═══════════════════════════════════════════════════════════════════════════
# DATA-3: In-process LRU Frame Cache
# ═══════════════════════════════════════════════════════════════════════════

class TestDataCache:
    def test_repeated_loads_hit_cache(self, isolated_data_env, fresh_db):
        cfg, _ = isolated_data_env
        clear_frame_cache()
        assert get_cache_stats()["hits"] == 0
        assert get_cache_stats()["misses"] == 0

        # First load: cache miss
        df1, h1, _ = load_and_validate("1h", cfg)
        assert get_cache_stats()["misses"] == 1
        assert get_cache_stats()["hits"] == 0

        # Second load: cache hit
        df2, h2, _ = load_and_validate("1h", cfg)
        assert get_cache_stats()["hits"] == 1
        assert h1 == h2
        assert len(df1) == len(df2)

    def test_tampering_invalidates_cache_and_triggers_miss(self, isolated_data_env, fresh_db):
        cfg, data_dir = isolated_data_env
        clear_frame_cache()
        load_and_validate("1h", cfg)
        assert get_cache_stats()["misses"] == 1

        # Modifying file changes hash, so cache must miss and re-validate
        p_1h = data_dir / cfg["data"]["files"]["1h"]
        with open(p_1h, "a") as f:
            f.write("2024-01-01 01:00:00+00:00,1800.0,1805.0,1795.0,1802.0,100.0\n")

        load_and_validate("1h", cfg)
        assert get_cache_stats()["misses"] == 2


# ═══════════════════════════════════════════════════════════════════════════
# DATA-1: Grid Alignment and Patch Safety
# ═══════════════════════════════════════════════════════════════════════════

class TestPatchAndGridAlignment:
    def test_verify_alignment_detects_grid(self, isolated_data_env):
        cfg, _ = isolated_data_env
        rep = verify_file_alignment("1h", cfg)
        assert rep["is_grid_aligned"] is True
        assert rep["expected_interval_minutes"] == 60
        assert rep["off_grid_bars"] == 0

    def test_patch_refuses_to_overwrite_source(self, isolated_data_env, fresh_db):
        cfg, _ = isolated_data_env
        src_file = data_path("1h", cfg)
        with pytest.raises(PermissionError, match="Refusing to overwrite original data file"):
            patch_gap(source_tf="5m", target_tf="1h", cfg=cfg, output_path=src_file)

    def test_patch_safe_output_and_grid_aggregation(self, isolated_data_env, fresh_db):
        cfg, data_dir = isolated_data_env
        # Create matching 5m data and aggregate it to create an authentic 1h overlap
        df_5m = make_bars("2021-01-04 00:00:00", "2021-03-01 00:00:00", freq="5min", seed=99)
        p_5m = data_dir / cfg["data"]["files"]["5m"]
        df_5m.to_csv(p_5m, index=False)

        # Build 1h from 5m so overlap consistency is 100%
        df_1h_clean = aggregate_to_grid(df_5m, "1h")
        p_1h = data_dir / cfg["data"]["files"]["1h"]
        df_1h_clean.to_csv(p_1h, index=False)

        out_file = data_dir / "XAUUSD_H1.patched.csv"
        rep = patch_gap(
            source_tf="5m",
            target_tf="1h",
            hole_start="2021-02-01 00:00:00",
            hole_end="2021-02-15 00:00:00",
            overlap_month="2021-01",
            mismatch_threshold_pct=1.0,
            cfg=cfg,
            output_path=out_file,
            allow_holdout_contamination=True,
        )
        assert rep["status"] == "SUCCESS"
        assert out_file.exists()
        assert out_file != p_1h


# ═══════════════════════════════════════════════════════════════════════════
# SBX-2: Data Vault (Phase F1b)
# ═══════════════════════════════════════════════════════════════════════════

class TestDataVault:
    def test_vault_encryption_decryption_roundtrip(self, isolated_data_env, fresh_db):
        cfg, _ = isolated_data_env
        b = get_or_freeze_boundaries(cfg)
        df, _, _ = load_and_validate("1h", cfg)

        key = generate_vault_key()
        enc = encrypt_splits(df, b, key)
        payload = decrypt_vault(enc, key)

        val_df = payload["validation"]
        hold_df = payload["holdout"]
        assert len(val_df) > 0
        assert len(hold_df) > 0
        assert val_df["timestamp"].min() >= b.validation_start
        assert val_df["timestamp"].max() < b.validation_end
        assert hold_df["timestamp"].min() >= b.holdout_start
        assert hold_df["timestamp"].max() < b.holdout_end

    def test_vault_strip_working_csv_and_datastore_retrieval(self, isolated_data_env, fresh_db):
        cfg, _ = isolated_data_env
        b = get_or_freeze_boundaries(cfg)
        df, _, _ = load_and_validate("1h", cfg)

        key = generate_vault_key()
        os.environ["VAULT_KEY"] = key

        # Save vault file
        save_vault("1h", df, b, cfg, key)
        vp = vault_path("1h", cfg)
        assert vp.exists()

        # Strip working CSV
        csv_file = strip_working_csv("1h", b, cfg)
        stripped = pd.read_csv(csv_file)
        # Working CSV must only contain train rows!
        assert len(stripped) < len(df)

        # Clear cache and verify DataStore loads validation from vault
        clear_frame_cache()
        ds = DataStore(cfg)
        val_df = ds.get_data("1h", "validation")
        assert len(val_df) > 0
        assert val_df.attrs.get("is_vaulted") is True
