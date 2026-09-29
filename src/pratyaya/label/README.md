# Deep-Dive: Bust Labeling Module (Node F4, Decision D-002)

In-tree technical documentation for operational and statistical forecast bust labeling in `src/pratyaya/label/`.

---

## 1. Responsibilities

The `label` package is responsible for:
- Deriving the primary operational decision-relevant bust label (`is_bust`) combining magnitude, IMD alert category flip, and significant weather floor.
- Deriving the secondary percentile-of-own-error label (`is_bust_p90`) for statistical robustness reporting.
- Fitting error percentiles ($P_{95}$, $P_{90}$) strictly on training years ($2000-2014$) with permission guards preventing held-out ($2018-2019$) leakage.
- Enforcing the 4 mandatory Day 1 sanity checks (§4.3) before any downstream model training.

---

## 2. Data Shapes and Schemas

### Canonical Bust Labels Schema (`data/canonical/labels/year=YYYY/labels_YYYYMMDD.parquet`)

```python
{
    "init_date": "timestamp[ns]",     # UTC initialization date (00 UTC cycle)
    "region_id": "str",               # IMD subdivision code ('1' to '36')
    "lead_day": "int64",              # Forecast lead day horizon (1 to 10)
    "variable": "str",                # Forecast variable ('rainfall' | 't2m')
    "forecast_value": "float64",      # Ensemble-mean forecast value (mm/day or K)
    "observed_value": "float64",      # Truth observation value (mm/day or K)
    "error_signed": "float64",        # Signed error: forecast_value - observed_value
    "is_bust": "bool",                # Primary operational decision-relevant label
    "is_bust_p90": "bool",            # Secondary statistical robustness label
    "bust_direction": "str",          # 'wet' | 'dry' | 'none'
    "threshold_used": "float64",      # The exact tau magnitude threshold applied
    "threshold_fit_period": "str",    # e.g. "2000-2014" (provenance)
    "is_category_flip": "bool"        # Whether IMD operational alert category flipped
}
```

### ThresholdModel Dataclass

```python
@dataclass
class ThresholdModel:
    p95_lookup: Dict[Tuple[str, int, int, str], float]   # (region_id, month, lead_day, variable) -> P95
    p90_lookup: Dict[Tuple[str, int, int, str], float]   # (region_id, month, lead_day, variable) -> P90
    global_p95: Dict[Tuple[int, str], float]             # (lead_day, variable) -> fallback P95
    global_p90: Dict[Tuple[int, str], float]             # (lead_day, variable) -> fallback P90
    fit_period: str                                      # e.g. "2000-2014"
```

---

## 3. Code Pointers

- Operational category classification: `src/pratyaya/label/bust.py#L42-L72`
- Training percentile fitting: `src/pratyaya/label/bust.py#L96-L158`
- Primary and secondary label generation: `src/pratyaya/label/bust.py#L161-L260`
- Mandatory sanity check verification: `src/pratyaya/label/bust.py#L263-L345`
- Unit and regression test suite: `tests/test_label_bust.py#L1-L200`

---

### Gotchas

- **Reversal from Percentile Primary (`D-002`):**
  Initial drafting contemplated a pure percentile label ($P_{90}$) as primary. Per `docs/DECISIONS.md` `D-002`, the primary label was explicitly reversed to the operational decision-relevant composite (`|e| >= max(10 mm, P95)` AND `category(F) != category(O)` AND `max(F, O) >= 15.6 mm/day`). An error that does not change the alert decision is not what this product exists to catch. Do not refactor back to a pure percentile primary.
- **Both Labels Must Persist on Every Row:**
  Downstream evaluation protocols (`docs/05-evaluation-protocol.md`) require displaying headline metrics under the primary definition while confirming robustness under `is_bust_p90`. Having both labels precomputed prevents costly rebuilds.
- **Magnitude Floor is Mandatory:**
  Near-boundary category flips (e.g. forecast $15.5\text{ mm}$ Light Rain vs observed $15.7\text{ mm}$ Moderate Rain, error $0.2\text{ mm}$) would register as category flips. The magnitude floor (`floor_mag = 10.0 mm`) and $P_{95}$ criterion eliminate boundary sensitivity.
- **Strict Training-Only Fitting:**
  Percentile thresholds $P_{95}$ and $P_{90}$ must never be fitted across the full dataset. Fitting on held-out years ($2018-2019$) raises `PermissionError` in `fit_bust_thresholds`.
- **Monotonicity with Lead Day:**
  Bust rates must rise from Lead 1 to Lead 10 because atmospheric predictability decays with lead time. If Day 1 bust rate is high or non-monotonic, an alignment bug (`IMD_DAY_OFFSET`) or threshold leakage is present (Risk `R-2`).
