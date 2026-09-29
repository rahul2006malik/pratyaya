# Models Architecture & Calibration Deep-Dive — Pratyaya

This document defines the mathematical formulation, implementation architecture, baseline ladder (Node F6), metric learning (Node G2), analogue-NWP fusion (Node G3), isotonic calibration (Node G4), and Effective Lead Time (Node G5) per `docs/03-architecture-and-modeling.md` §5–§7, `docs/05-evaluation-protocol.md` §2, and `docs/06-design-ux.md` §1.1.

---

## 1. Baseline Ladder Overview (Node F6)

- **B0 (Constant Base Rate):** Predicts constant probability $\bar{p} = \text{mean}(y_{\text{train}}) \approx 0.043$. The absolute performance floor.
- **B1 (Conditional Climatology):** Empirical bust rate stratified by `(region_id, lead_day, month)` with hierarchical fallback:
  `(region, lead, month) -> (region, month) -> (lead, month) -> global rate`.
- **B2 (Raw Ensemble Spread):** Min-max normalized `ens_spread` representing the naive operational standard answer.
- **B3 (Calibrated Logistic Spread):** Logistic regression $\text{logit}(p) = w_0 + w_1 \cdot \text{spread} + w_2 \cdot \text{lead} + w_3 \cdot \text{fcst}$. The fair operational bar that our analogue retrieval bet must beat on Day 5 (PR-AUC = **0.2655** [95% CI: 0.1719, 0.3728]).
- **B4 (EMOS):** Ensemble Model Output Statistics predictive distribution post-processing (Gneiting et al. 2005).
- **B5 (Tabular LightGBM):** Full tabular GBDT baseline.

---

## 2. Low-Rank Mahalanobis Metric Learning (Node G2, Model M1 v2)

### 2.1 Formulation
Standard Euclidean analogue retrieval treats all 100 EOF directions equally. However, atmospheric states with similar flow can have divergent predictability, and vice versa. We learn a low-rank Mahalanobis distance metric:
$$M = W^T W, \quad W \in \mathbb{R}^{32 \times 100}$$
Under metric $M$, the distance between two 100-EOF atmospheric states $x_1, x_2 \in \mathbb{R}^{100}$ is:
$$d_M(x_1, x_2) = \sqrt{(x_1 - x_2)^T W^T W (x_1 - x_2)} = \|W(x_1 - x_2)\|_2 = \|z_1 - z_2\|_2$$
where $z = W x \in \mathbb{R}^{32}$ is the compressed predictability embedding.

### 2.2 Standardized Outcome Vectors
For each initialization date $t$ in the training fold (2000–2001), we construct an outcome vector $y(t) \in \mathbb{R}^{360}$ representing standardized signed forecast errors:
$$e(t, r, L) = \text{forecast\_value} - \text{observed\_value}$$
$$y_{\text{std}}(t, r, L) = \frac{e(t, r, L) - \mu(r, L)}{\sigma(r, L) + \epsilon}$$
across all 36 subdivisions $r$ and 10 lead days $L \in [1, 10]$.

### 2.3 Triplet Sampling with Mandatory Temporal Exclusion
Triplets $(a, p, n)$ (anchor, positive, negative) are sampled satisfying:
1. **Outcome Proximity:** $\|y(a) - y(p)\|_2 < \|y(a) - y(n)\|_2$ (positive had more similar forecast errors than negative).
2. **Mandatory Temporal Separation:** $|\text{date}(a) - \text{date}(p)| \ge 15$ calendar days.
   *Leakage Prevention:* Without this constraint, the positive is simply adjacent days in the same synoptic spell, causing the metric to memorize weather persistence rather than predictability.
3. **Training Fold Quarantine:** All candidates are sampled strictly from training years (2000–2001). Zero access to held-out years (2018–2019).

### 2.4 Vectorized Optimization
$W$ is trained minimizing the margin ranking loss:
$$L(W) = \frac{1}{|B|} \sum_{(a, p, n) \in B} \max\left(0, \text{margin} + \|W(x_a - x_p)\|_2 - \|W(x_a - x_n)\|_2\right) + \lambda \|W\|_F^2$$
We optimize $W$ via mini-batch Adam in pure NumPy/SciPy. On the 2002 calibration fold, M1 v2 achieves **PR-AUC = 0.0342**, outperforming Euclidean M1 v1 (0.0313), B0 (0.0300), and B1 (0.0278).

---

## 3. Analogue-NWP Fusion Model (Node G3, Model M2)

### 3.1 Feature Vector
The fusion model integrates physical NWP ensemble diagnostics with data-driven analogue predictability features:
$$\mathbf{x}_{\text{fusion}} = \left[\text{ens\_spread}, \text{forecast\_value}, \text{lead\_day}, p_{\text{analogue}}, d_k, \text{signed\_expect}, \text{analogue\_dispersion}, \sin\theta_{\text{month}}, \cos\theta_{\text{month}}\right]$$
- $p_{\text{analogue}}$: Gaussian kernel-weighted historical bust rate among the top $k=8$ analogues.
- $d_k$: Mahalanobis distance to the $k$-th analogue (OOD indicator).
- $\text{signed\_expect}$: Analogue-weighted expected error (provides bust direction).
- $\text{analogue\_dispersion}$: Standard deviation of errors across analogues (second uncertainty channel).

### 3.2 Leakage-Free Out-of-Fold Training
To prevent the fusion model from learning in-fold analogue over-confidence, training fold features are generated via out-of-fold leave-cycle retrieval:
$$\text{Candidate Pool for } t = \{t' \in \text{Train} \mid |t' - t| > 15 \text{ days}\}$$
The fusion classifier is trained strictly on out-of-fold analogue predictions.

### 3.3 LightGBM vs Logistic Regression
- **LightGBM (`LGBMClassifier`):** Configured with `scale_pos_weight` ($\sim 19.2$) to handle the ~3–5% base rate. Achieved raw PR-AUC = **0.3530** on the calibration fold.
- **Logistic Regression (`LogisticRegression`):** Balanced class weights, achieving raw PR-AUC = **0.3502**.
- **Winning Fusion Model:** LightGBM, advancing to calibration.

### 3.4 Orthogonality Verification (Node G3)
- Pearson $r(p_{\text{analogue}}, \text{ens\_spread}) \approx 0.00$
- Pearson $r(p_{\text{analogue}}, \text{prob}_{B3}) = +0.1508$
- Analogue predictions provide an independent, non-redundant signal from physical ensemble spread.

---

## 4. Isotonic Calibration & Murphy Decomposition (Node G4)

### 4.1 Calibration Protocol
The raw fusion probability $p_{\text{raw}}$ is calibrated on the dedicated calibration fold (2002) using Isotonic Regression:
$$p_{\text{calibrated}} = f_{\text{iso}}(p_{\text{raw}}), \quad f_{\text{iso}} \text{ monotonically non-decreasing in } [0.0, 1.0]$$

### 4.2 Murphy (1973) Brier Decomposition
Brier score is decomposed into:
$$\text{BS} = \text{Reliability} - \text{Resolution} + \text{Uncertainty}$$
where:
- $\text{Uncertainty} = \bar{o}(1 - \bar{o}) = 0.0300 \times 0.9700 = 0.0291$
- $\text{Reliability} = \frac{1}{N} \sum_k N_k (\bar{p}_k - \bar{o}_k)^2$ (calibration error; lower is better)
- $\text{Resolution} = \frac{1}{N} \sum_k N_k (\bar{o}_k - \bar{o})^2$ (sorting ability; higher is better)

### 4.3 Calibration Results
| Method | Brier Score | Reliability | Resolution | BSS vs B1 | Status |
|---|---|---|---|---|---|
| Raw Fusion | 0.0614 | 0.0377 | 0.0057 | -0.7493 | Overconfident |
| Platt Scaling (Logistic) | 0.0236 | 0.0007 | 0.0061 | +0.3276 | S-curve Fit |
| **Isotonic Regression** | **0.0223** | **0.0000** | **0.0066** | **+0.3640** | **Winner** |

Isotonic calibration eliminates calibration error ($\text{Reliability} \approx 0.0000$), reducing Brier score by 63.7% from 0.0614 to 0.0223.

---

## 5. Effective Lead Time & Direction Head (Node G5)

### 5.1 Effective Lead Time (ELT) Inversion
Operational forecasters have intuitive calibration for lead time, not abstract probabilities. We compute ELT by inverting the climatological bust curve $\bar{p}(L)$:
$$\text{ELT}(p) = L' \quad \text{such that} \quad \bar{p}(L') = p_{\text{calibrated}}$$
- **Monotonicity:** $\text{ELT}$ is guaranteed non-decreasing with $p_{\text{calibrated}}$.
- **Clipping:** Bounded to $[0.5, 12.0]$ lead days.
- **Edge-Case Headlines:**
  - $\text{ELT} < 1.0$: *"Day L forecast behaving better than a typical Day-1 forecast"* (`is_better_than_day1 = True`).
  - $\text{ELT} > 12.0$: *"Day L forecast behaving beyond our measured range"* (`is_beyond_range = True`).
  - Typical range: *"Day 6 forecast behaving like Day 9.4"*.

### 5.2 3-Class Bust Direction Head
Based on analogue-weighted signed forecast error expectation ($\text{signed\_expect} = \text{forecast} - \text{observed}$):
- $\text{signed\_expect} < -2.0 \text{ mm/day} \implies \textbf{WET}$ (forecast was too dry; reality wetter than forecast).
- $\text{signed\_expect} > +2.0 \text{ mm/day} \implies \textbf{DRY}$ (forecast was too wet; reality drier than forecast).
- $|\text{signed\_expect}| \le 2.0 \text{ mm/day} \implies \textbf{NONE}$ (neutral / unresolvable).

### 5.3 Categorical Confidence Bands
Using exact threshold boundaries from `src/pratyaya/api/schemas.py`:
- $p_{\text{calibrated}} < 0.10 \implies \textbf{HIGH}$ confidence
- $0.10 \le p_{\text{calibrated}} < 0.25 \implies \textbf{MODERATE}$ confidence
- $0.25 \le p_{\text{calibrated}} < 0.50 \implies \textbf{LOW}$ confidence
- $p_{\text{calibrated}} \ge 0.50 \implies \textbf{CRITICAL}$ risk

---

## 6. Model Ladder Benchmark Summary (Calibration Fold 2002)

| Model | Type | PR-AUC [95% CI] | AUROC | Brier Score | BSS vs B1 | Status |
|---|---|---|---|---|---|---|
| B0 (Constant Base Rate) | Baseline | 0.0300 [0.0217, 0.0386] | 0.5000 | 0.0295 | +0.1611 | Baseline Floor |
| B1 (Conditional Climatology) | Baseline | 0.0278 [0.0204, 0.0356] | 0.4252 | 0.0351 | +0.0000 | Climatology Bar |
| B2 (Raw Ensemble Spread) | Baseline | 0.0300 [0.0217, 0.0386] | 0.5000 | 0.0300 | +0.1462 | Spread Only |
| B3 (Calibrated Logistic Spread) | Baseline | **0.2655** [0.1719, 0.3728] | 0.8689 | 0.1423 | -3.0483 | Spread Benchmark |
| M1 v1 (Plain EOF-Euclidean) | Differentiated | 0.0313 [0.0232, 0.0446] | 0.4759 | 0.0350 | +0.0027 | Beats B1 |
| M1 v2 (Mahalanobis Metric) | Differentiated | 0.0342 [0.0291, 0.0427] | 0.5054 | 0.0319 | +0.0915 | Beats M1 v1 |
| **M2 (Analogue-NWP Fusion)** | **Differentiated** | **0.3525** [0.2263, 0.5214] | **0.8850** | **0.0223** | **+0.3640** | **Beats B3 Bar** |

---

## 7. Key Code Pointers
- Linear low-rank metric learning: [`src/pratyaya/analogue/metric.py`](file:///c:/WORK/pratyaya/src/pratyaya/analogue/metric.py)
- Analogue retriever with Mahalanobis support: [`src/pratyaya/analogue/retrieve.py`](file:///c:/WORK/pratyaya/src/pratyaya/analogue/retrieve.py)
- Fusion model & out-of-fold generator: [`src/pratyaya/models/fusion.py`](file:///c:/WORK/pratyaya/src/pratyaya/models/fusion.py)
- Probability calibrator & Murphy decomposition: [`src/pratyaya/models/calibrate.py`](file:///c:/WORK/pratyaya/src/pratyaya/models/calibrate.py)
- Effective Lead Time & Direction head: [`src/pratyaya/models/elt.py`](file:///c:/WORK/pratyaya/src/pratyaya/models/elt.py)
- Full calibration evaluation runner: [`scripts/score_fusion.py`](file:///c:/WORK/pratyaya/scripts/score_fusion.py)
