# Mathematical Methodology & Theoretical Formulation

Pratyaya addresses medium-range forecast decay by measuring the geometry of atmospheric state evolution against historical synoptic trajectories.

---

## 1. Atmospheric State Vector Representation

Let $\mathbf{X}_t \in \mathbb{R}^{H \times W \times C}$ denote the instantaneous synoptic atmospheric state over the Indian subcontinent and North Indian Ocean synoptic domain ($30^\circ\text{E} - 120^\circ\text{E}$, $10^\circ\text{S} - 45^\circ\text{N}$) discretized onto a $1.5^\circ$ spherical grid:

$$\text{Domain: } 61 \text{ longitudes} \times 38 \text{ latitudes} \times 5 \text{ channels} = 11,590 \text{ state dimensions}$$

The five physical channels comprise:
1. Geopotential Height at 500 hPa ($Z_{500}$)
2. Zonal Wind Component at 850 hPa ($U_{850}$)
3. Meridional Wind Component at 850 hPa ($V_{850}$)
4. Mean Sea Level Pressure ($\text{MSLP}$)
5. 2-Meter Surface Temperature ($T_{2m}$)

Prior to dimensionality reduction, grid cells are weighted by latitude area factors $w_i = \sqrt{\cos(\phi_i)}$ to preserve spherical metric invariance.

---

## 2. Empirical Orthogonal Function (EOF) Decomposition

Given the centered, standardized anomaly matrix $\mathbf{S} \in \mathbb{R}^{N \times D}$, spatial covariance is decomposed via truncated Singular Value Decomposition:

$$\mathbf{S} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$

Retaining $K = 50$ principal modes accounts for $> 82.4\%$ of the total seasonal circulation variance while discarding high-wavenumber turbulent noise. Any instantaneous circulation anomaly $\mathbf{x}$ maps to a reduced coefficient vector:

$$\mathbf{z} = \mathbf{V}_K^T \mathbf{x} \in \mathbb{R}^{50}$$

---

## 3. Low-Rank Mahalanobis Metric Learning

Standard Euclidean distance in EOF space treats all circulation anomalies isotropically, ignoring empirical error sensitivity. Pratyaya learns a low-rank metric transformation $\mathbf{L} \in \mathbb{R}^{r \times 50}$ with rank $r = 32$:

$$d_{\mathbf{M}}^2(\mathbf{z}_i, \mathbf{z}_j) = (\mathbf{z}_i - \mathbf{z}_j)^T \mathbf{M} (\mathbf{z}_i - \mathbf{z}_j) = \|\mathbf{L}(\mathbf{z}_i - \mathbf{z}_j)\|_2^2$$

where $\mathbf{M} = \mathbf{L}^T \mathbf{L}$ is positive semi-definite. The transformation $\mathbf{L}$ is optimized via neighbourhood components analysis to maximize the probability that atmospheric states sharing similar precipitation forecast error signatures remain adjacent in metric space:

$$\mathcal{L}(\mathbf{L}) = -\sum_{i} \log \left( \frac{\sum_{j \in \mathcal{C}_i} \exp(-\|\mathbf{L}(\mathbf{z}_i - \mathbf{z}_j)\|_2^2)}{\sum_{k \neq i} \exp(-\|\mathbf{L}(\mathbf{z}_i - \mathbf{z}_k)\|_2^2)} \right)$$

subject to a strict temporal exclusion window $|\tau_i - \tau_j| \ge 15\text{ days}$ to completely eliminate synoptic autocorrelation leakage.

---

## 4. Analogue Retrieval & Kernel Weighting

For a forecast initialization cycle $\mathbf{z}^*$, the $k = 8$ nearest historical analogues $\{\mathbf{z}_{(1)}, \dots, \mathbf{z}_{(k)}\}$ are retrieved from the 23-year archive (2000–2022). Each analogue receives a normalized Gaussian kernel weight:

$$w_{(m)} = \frac{\exp\left(-\frac{d_{\mathbf{M}}^2(\mathbf{z}^*, \mathbf{z}_{(m)})}{2 \sigma^2}\right)}{\sum_{l=1}^k \exp\left(-\frac{d_{\mathbf{M}}^2(\mathbf{z}^*, \mathbf{z}_{(l)})}{2 \sigma^2}\right)}$$

The analogue-implied bust likelihood $\hat{p}_{\text{analogue}}$ and expected error bias $\hat{\mu}_{\text{err}}$ are computed as kernel-weighted empirical expectations of the historical outcomes.

---

## 5. Non-Linear Fusion & Isotonic Calibration

The final bust probability $\hat{P}(\text{Bust} \mid \text{NWP}, \text{Analogue})$ combines:
1. Raw physical NWP ensemble spread ($\sigma_{\text{GEFS}}$)
2. Deterministic forecast precipitation anomaly
3. Analogue-implied historical failure frequency ($\hat{p}_{\text{analogue}}$)
4. Mahalanobis minimum analogue distance ($d_{\text{min}}$)
5. Synoptic regime cluster assignment ($c \in \{1, \dots, 9\}$)

These features are fused using a gradient boosted decision tree and passed through out-of-fold monotonic isotonic regression to yield strictly calibrated, reliability-error minimized probabilities:

$$P_{\text{cal}} = f_{\text{iso}}(f_{\text{fusion}}(\mathbf{h}))$$
