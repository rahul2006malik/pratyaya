# Evaluation & Splitter Deep-Dive — Pratyaya

This document provides architectural notes and data contracts for the evaluation harness (Node F7) and metric calculation & event clustering (Node F8).

---

## 1. Module Overview

The evaluation suite consists of four primary components:
- `src/pratyaya/evaluate/splits.py`: Temporal fold splitter enforcing canonical boundaries (Train 2000–2014, Calibration 2015–2017, Held-out 2018–2019), 12-day boundary purging, and `--final` authorization gates.
- `src/pratyaya/evaluate/events.py`: 3D connected-component bust event clustering across `(time, lat, lon)` / `(valid_date, region_id)` to resolve the effective sample size problem.
- `src/pratyaya/evaluate/metrics.py`: Computes PR-AUC, AUROC, Brier Score, Brier Skill Score (vs B1 climatology), reliability diagram curves, Murphy (1973) decomposition, and cost-loss value curves.
- `src/pratyaya/evaluate/bootstrap.py`: Block bootstrap resampling whole initialization dates over 1,000 resamples for honest 95% confidence intervals and paired difference tests.

---

## 2. Data Shapes & Schemas

### Split Result Data Contract

```python
@dataclass
class SplitResult:
    train: pd.DataFrame
    calibration: pd.DataFrame
    heldout: Optional[pd.DataFrame] = None
    purged_train_dates: List[pd.Timestamp] = field(default_factory=list)
    purged_calibration_dates: List[pd.Timestamp] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Event Clustering Result Data Contract

```python
@dataclass
class EventClusteringResult:
    labeled_df: pd.DataFrame
    n_events: int
    n_bust_rows: int
    inflation_factor: float
    events_summary: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Bootstrap Result Data Contracts

```python
@dataclass
class BootstrapResult:
    metric_name: str
    point_estimate: float
    median: float
    ci_lower: float
    ci_upper: float
    std_err: float
    n_resamples: int
    n_clusters: int
    resamples: np.ndarray = field(repr=False)

@dataclass
class PairedBootstrapResult:
    metric_name: str
    point_estimate_model: float
    point_estimate_baseline: float
    delta_point_estimate: float
    delta_median: float
    delta_ci_lower: float
    delta_ci_upper: float
    delta_std_err: float
    p_value: float
    ci_excludes_zero: bool
    n_resamples: int
    n_clusters: int
    delta_resamples: np.ndarray = field(repr=False)
```

---

## 3. Implementation Pointers

- Split logic & boundary purging: [`src/pratyaya/evaluate/splits.py#L90-L240`](file:///c:/WORK/pratyaya/src/pratyaya/evaluate/splits.py#L90-L240)
- Held-out touch logging: [`src/pratyaya/evaluate/splits.py#L52-L88`](file:///c:/WORK/pratyaya/src/pratyaya/evaluate/splits.py#L52-L88)
- 3D event clustering algorithm: [`src/pratyaya/evaluate/events.py#L110-L240`](file:///c:/WORK/pratyaya/src/pratyaya/evaluate/events.py#L110-L240)
- Metric calculation & Murphy decomposition: [`src/pratyaya/evaluate/metrics.py#L30-L210`](file:///c:/WORK/pratyaya/src/pratyaya/evaluate/metrics.py#L30-L210)
- Richardson/Murphy cost-loss value curves: [`src/pratyaya/evaluate/metrics.py#L215-L315`](file:///c:/WORK/pratyaya/src/pratyaya/evaluate/metrics.py#L215-L315)
- Paired block bootstrap over initialization dates: [`src/pratyaya/evaluate/bootstrap.py#L70-L240`](file:///c:/WORK/pratyaya/src/pratyaya/evaluate/bootstrap.py#L70-L240)

---

## 4. Mandatory Gotchas

### Gotchas

1. **Held-out Partition Leakage Protection:**
   The held-out set (years 2018–2019) is strictly locked until Day 8. Any code path that loads or splits held-out data without `final=True` raises `PermissionError` immediately. This behavior is governed by [`docs/DECISIONS.md#D-000`](file:///c:/WORK/pratyaya/docs/DECISIONS.md).

2. **12-Day Boundary Purging:**
   A forecast initialized on 28 Dec at lead 10 verifies in January of the following fold. Failing to purge the trailing 12 days before fold boundaries creates temporal autocorrelation leakage that inflates validation skill. This is governed by `docs/05-evaluation-protocol.md` §1.2.

3. **Effective Sample Size & Row Resampling Trap:**
   Resampling individual rows rather than initialization dates or connected components understates standard errors by ~7× because a single synoptic bust produces dozens of correlated region-lead rows. All bootstrap routines in `bootstrap.py` resample whole cluster units per `docs/05` §4.1.
