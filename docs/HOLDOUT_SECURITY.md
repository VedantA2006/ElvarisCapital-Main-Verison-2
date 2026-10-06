# ELVARIS CAPITAL — HOLDOUT VAULT SECURITY & QUARANTINE SPECIFICATION (V3)

---

## 1. Principles of Holdout Isolation

The Holdout dataset represents pristine future data reserved exclusively for final strategy confirmation. In conventional quantitative research, repeated testing on holdout data causes subtle information leakage and overfitting.

In QuantForge V3, holdout data is treated as a **cryptographically sealed exam paper**:
1. **Physical Separation**: Access to the holdout partition is blocked by default across all ordinary exploration, ideation, optimization, and validation pathways.
2. **Deterministic Guard**: Any attempt by an unfrozen strategy to query holdout data raises `PermissionError: HoldoutLockError`.
3. **One-Time Evaluation**: A strategy may only unlock the holdout partition once in its lifetime. Once evaluated, the holdout partition for that strategy is permanently consumed.

---

## 2. Invariant Hash Authentication

Before holdout data is served, the strategy must prove that it is `FROZEN` by presenting an unalterable composite hash tuple:

```python
frozen_signature = {
    "strategy_id": strategy.id,
    "code_hash": sha256(source_code),
    "spec_hash": sha256(dsl_spec),
    "config_hash": sha256(config_yaml),
    "dataset_hash": sha256(train_data_bytes),
}
```

If any parameter, indicator period, or threshold has changed by even $0.0001\%$, the cryptographic signature fails verification, and holdout access is denied.

---

## 3. Database Audit Record

Every holdout evaluation is transactionally recorded in MongoDB collection `holdout_accesses`:
```json
{
  "strategy_id": "Gold_MarketStructure_15m_v1",
  "evaluated_at": "2026-10-06T19:42:00Z",
  "triggered_by": "autonomous_engine",
  "code_hash": "a8f4c2e...",
  "status": "EVALUATED",
  "holdout_bars": 5796
}
```

A unique index on `strategy_id` guarantees that concurrent or repeated attempts to evaluate the same strategy on holdout raise a duplicate key error, preventing multiple testing on out-of-sample data.
