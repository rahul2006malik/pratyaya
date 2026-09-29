# Analogue Retrieval Engine v1 Deep-Dive (Nodes G1, G2, G6)

This document describes the state representation, EOF reduction, and Euclidean retrieval implementation for the Pratyaya Analogue Engine v1.

---

## 1. Architectural Role & Overview

The Analogue Retrieval Engine is the differentiated component of Project Pratyaya (`docs/03-architecture-and-modeling.md` §6). Rather than predicting forecast busts purely from local scalar ensemble spread, it retrieves historical synoptic atmospheric analogues from training years to evaluate whether similar past states were prone to NWP busts.

The pipeline operates in three sequential stages:
1. **Wide-Domain Synoptic State Representation** (`src/pratyaya/analogue/state.py`): Extracts 5 key atmospheric fields (Z500, U850, V850, MSLP, TCWV) over $30^\circ\text{E}–120^\circ\text{E}$, $10^\circ\text{S}–45^\circ\text{N}$ at $1.5^\circ$ resolution, standardized against training-only climatology.
2. **EOF / PCA Dimensionality Reduction** (`src/pratyaya/analogue/eof.py`): Compresses the flattened 11,590-feature state vector into $x \in \mathbb{R}^{100}$ principal components fitted strictly on training years (2000–2001).
3. **Euclidean k-NN Retrieval with Temporal Exclusion** (`src/pratyaya/analogue/retrieve.py`): Finds the top $k=8$ closest historical dates in 100-EOF space, strictly excluding any candidate within $\pm 15$ days of the query date to prevent autocorrelation persistence bias, computes Gaussian kernel weights, estimates bust probabilities, and flags Out-Of-Distribution (OOD) states.

---

## 2. Data Shapes & Schemas

### WideDomainGrid Schema
```python
@dataclass
class WideDomainGrid:
    lon_min: float = 30.0
    lon_max: float = 120.0
    lat_min: float = -10.0
    lat_max: float = 45.0
    resolution: float = 1.5
    n_lons: int = 61
    n_lats: int = 38
    n_grid_points: int = 61 * 38  # 2,318
    fields: Tuple[str, ...] = ("z500", "u850", "v850", "mslp", "tcwv")
    total_features: int = 11,590
```

### Synoptic State Vector Grain
- Raw field grid: `(n_days, 38, 61)` per field.
- Stacked and flattened: `(n_days, 11590)` array of `float64`.
- Standardized anomaly: $z = (x - \mu_{\text{train}}) / (\sigma_{\text{train}} + \epsilon)$.

### EOF Representation
- Input: $X_{\text{train}} \in \mathbb{R}^{N_{\text{train}} \times 11590}$.
- Output embedding: $Z \in \mathbb{R}^{N \times 100}$.
- Basis components: `components_` shape `(100, 11590)`.

### AnalogueMatch Dataclass
```python
@dataclass
class AnalogueMatch:
    candidate_date: pd.Timestamp
    distance: float
    weight: float
    rank: int
    candidate_idx: int
```

### RetrievalResult Dataclass
```python
@dataclass
class RetrievalResult:
    query_date: pd.Timestamp
    matches: List[AnalogueMatch]
    is_ood: bool
    d_k: float
    ood_threshold: Optional[float]
```

### Predicted Bust Probabilities Output Schema
| Field Name | Type | Description |
|---|---|---|
| `region_id` | str | IMD subdivision identifier (1–36) |
| `lead_day` | int | Forecast lead day (1–10) |
| `p_analogue` | float | Kernel-weighted historical bust probability $\sum w_i y_i$ |
| `signed_expect` | float | Kernel-weighted signed error expectation $\sum w_i e_i$ |
| `analogue_dispersion` | float | Weighted standard deviation of analogue errors |
| `is_ood` | bool | Flag indicating whether $d_k > P_{99}$ training distance |

---

## 3. Implementation Code References

- **Wide-Domain Coordinates & Grid Definition:** [`src/pratyaya/analogue/state.py#L64-L100`](file:///c:/WORK/pratyaya/src/pratyaya/analogue/state.py#L64-L100)
- **Held-Out Protection Guard (D-000):** [`src/pratyaya/analogue/state.py#L51-L61`](file:///c:/WORK/pratyaya/src/pratyaya/analogue/state.py#L51-L61)
- **Training Climatology & Anomaly Standardization:** [`src/pratyaya/analogue/state.py#L103-L179`](file:///c:/WORK/pratyaya/src/pratyaya/analogue/state.py#L103-L179)
- **State Flattening & WB2 Slicing:** [`src/pratyaya/analogue/state.py#L181-L298`](file:///c:/WORK/pratyaya/src/pratyaya/analogue/state.py#L181-L298)
- **EOF / PCA 100-Component Reducer:** [`src/pratyaya/analogue/eof.py#L25-L160`](file:///c:/WORK/pratyaya/src/pratyaya/analogue/eof.py#L25-L160)
- **Euclidean k-NN Retrieval Engine:** [`src/pratyaya/analogue/retrieve.py#L59-L219`](file:///c:/WORK/pratyaya/src/pratyaya/analogue/retrieve.py#L59-L219)
- **Mandatory Temporal Exclusion Window:** [`src/pratyaya/analogue/retrieve.py#L162-L173`](file:///c:/WORK/pratyaya/src/pratyaya/analogue/retrieve.py#L162-L173)
- **Native OOD Calibration & Evaluation (Node G6):** [`src/pratyaya/analogue/retrieve.py#L108-L141`](file:///c:/WORK/pratyaya/src/pratyaya/analogue/retrieve.py#L108-L141)
- **Probability & Dispersion Projection:** [`src/pratyaya/analogue/retrieve.py#L221-L280`](file:///c:/WORK/pratyaya/src/pratyaya/analogue/retrieve.py#L221-L280)

---

## 4. Key Decisions & Rationale

### Gotchas

- **Temporal Autocorrelation Trap (`TEMPORAL_EXCLUSION_DAYS = 15`):**
  Without masking candidate dates in $[t_q - 15, t_q + 15]$, the nearest analogue to Monday is almost invariably Tuesday. The retrieval engine appears to achieve high accuracy while in reality learning nothing more than atmospheric persistence. Enforced as an invariant in [`src/pratyaya/analogue/retrieve.py#L164-L171`](file:///c:/WORK/pratyaya/src/pratyaya/analogue/retrieve.py#L164-L171) and verified in [`tests/test_analogue.py#L145-L192`](file:///c:/WORK/pratyaya/tests/test_analogue.py#L145-L192).

- **Domain Width ($30^\circ\text{E}–120^\circ\text{E}, 10^\circ\text{S}–45^\circ\text{N}$):**
  A domain clipped to India's land boundaries fails to capture upstream wave packet precursors (Arabian Sea and West Asian waveguide) and downstream convective triggers (Bay of Bengal and West Pacific). Slicing narrower domains reduces compute slightly but destroys retrieval skill.

- **Grid Resolution (1.5° vs 0.25°):**
  Synoptic precursors are large-scale planetary and synoptic waves. Regridding to 1.5° retains 99% of the large-scale circulation signal while reducing the feature vector from ~420,000 to 11,590 features.

- **Strict Training Fold Climatology (`docs/DECISIONS.md#D-000`):**
  Climatological mean $\mu$ and standard deviation $\sigma$ must never incorporate validation (2002) or held-out (2018–2019) dates. Doing so constitutes data leakage and invalidates benchmark comparability against Baseline B3 (`docs/DECISIONS.md#D-003`).
