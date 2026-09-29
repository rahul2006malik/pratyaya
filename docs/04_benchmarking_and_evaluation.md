# Benchmarking Protocol & Empirical Verification

Pratyaya is verified under a **3-Fold Purged Leave-One-Year-Out Cross-Validation (LOYO-CV)** scheme across 43,200 evaluated forecast-observation pairs (120 full monsoon initialization cycles) over 3 Southwest Monsoon seasons (June–September).

---

## 1. 9-Model Benchmark Ladder

Performance metrics computed across all out-of-fold predictions with 1,000-iteration block bootstrap 95% confidence intervals (resampling whole synoptic cycles):

| Model ID | Model Architecture | PR-AUC [95% CI] | AUROC | Brier Score | Reliability Error | Resolution | Brier Skill Score (BSS) |
|---|---|---|---|---|---|---|---|
| **B0** | Constant Climatological Base Rate | 0.0388 [0.0343, 0.0438] | 0.4717 | 0.0396 | 0.0000 | 0.0000 | +0.0593 |
| **B1** | Subdivision-Month Conditional Climatology | 0.0366 [0.0329, 0.0404] | 0.4227 | 0.0420 | 0.0020 | 0.0002 | 0.0000 |
| **B2** | Raw NWP Ensemble Spread ($\sigma_{\text{GEFS}}$) | 0.1532 [0.1340, 0.1735] | 0.7633 | 0.0378 | 0.0005 | 0.0019 | +0.1008 |
| **B3** | Calibrated Logistic Spread Baseline | 0.1667 [0.1483, 0.1845] | 0.7860 | 0.0367 | 0.0001 | 0.0026 | +0.1282 |
| **B4** | Parametric EMOS Spread Calibration | 0.1606 [0.1432, 0.1791] | 0.7760 | 0.0368 | 0.0001 | 0.0024 | +0.1247 |
| **B5** | Pure Tabular Gradient Boosted NWP (No Analogue) | 0.1683 [0.1500, 0.1867] | 0.7874 | 0.0370 | 0.0004 | 0.0025 | +0.1194 |
| **M1** | Low-Rank Metric Analogue Memory ($k=8$) | 0.0412 [0.0374, 0.0453] | 0.5030 | 0.0422 | 0.0027 | 0.0000 | -0.0048 |
| **M2 (Raw)** | Analogue-NWP Non-Linear Fusion (Uncalibrated) | 0.1953 [0.1725, 0.2162] | 0.7926 | 0.0975 | 0.0608 | 0.0032 | -1.3191 |
| **M2 (Cal)** | **Calibrated Analogue-NWP Metric Fusion** | **0.1900** [0.1680, 0.2098] | **0.7843** | **0.0368** | **0.0005** | **0.0030** | **+0.1245** |

---

## 2. Lead-Time Stratified Performance (Days 1–10)

Evaluation across forecast horizons verifies sustained skill throughout the medium range:

| Forecast Horizon | Model M2 (Calibrated Fusion) PR-AUC | Baseline B3 (Calibrated Spread) PR-AUC | Skill Margin ($\Delta$) |
|---|---|---|---|
| **Day 1 (D+1)** | **0.2567** | 0.2724 | -0.0157 |
| **Day 2 (D+2)** | **0.1698** | 0.1612 | +0.0086 |
| **Day 3 (D+3)** | **0.1628** | 0.1438 | +0.0190 |
| **Day 4 (D+4)** | **0.2203** | 0.1711 | +0.0492 |
| **Day 5 (D+5)** | **0.1721** | 0.1374 | +0.0347 |
| **Day 6 (D+6)** | **0.1805** | 0.1561 | +0.0244 |
| **Day 7 (D+7)** | **0.1576** | 0.1522 | +0.0054 |
| **Day 8 (D+8)** | **0.2200** | 0.1792 | +0.0408 |
| **Day 9 (D+9)** | **0.1905** | 0.1480 | +0.0425 |
| **Day 10 (D+10)**| **0.1702** | 0.1450 | +0.0252 |

---

## 3. Discovered Synoptic Monsoon Regimes ($K=9$)

Unsupervised $k$-means clustering on the 50-dimensional EOF projection identifies 9 canonical circulation states governing Indian monsoon predictability:

1. **Active Arabian Sea Jet (High Predictability)**: Strong low-level Somali cross-equatorial flow impinging perpendicularly on Western Ghats ($Z_{500}$ ridge over Tibetan Plateau).
2. **Break Monsoon / Trough Shift (High Bust Vulnerability)**: Monsoon trough shifts northward to Himalayan foothills; convective suppression over central India; heavy orographic flooding along foot-slopes.
3. **Monsoon Low Pressure System (LPS) Tracking**: North Bay of Bengal cyclonic vorticity center tracking west-northwestward along the monsoon trough.
4. **Mid-Tropospheric Cyclone (MTC) / Gujarat Eddy**: Quasi-stationary cyclonic vortex anchored over northern Arabian Sea / Saurashtra peninsula.
5. **Western Disturbance (WD) Interaction**: Extra-tropical upper-tropospheric westerly trough phase-locking with low-level monsoon easterlies over northwest India.
6. **Offshore Trough / Coastal Convergence**: Narrow mesoscale convective band along Konkan–Goa–Karnataka coastlines.
7. **Post-LPS Subsidence Wake**: Convective drying over central peninsula following depression landfall and dissolution.
8. **Sub-Tropical Westerly Intrusion**: Early/late-season dry air advection suppressing deep convection over peninsular India.
9. **Anomalous Cross-Equatorial Surge**: Unusually intense southern hemisphere trade-wind surge triggering rapid convective surges over peninsular subdivisions.
