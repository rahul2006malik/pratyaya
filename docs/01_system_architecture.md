# System Architecture & Operational Pipeline

Pratyaya is designed around a strict **dual-branch operational separation**: an offline batch scientific training branch and an asynchronous, read-only operational serving branch.

```
+-------------------------------------------------------------------------------+
|                      TRAINING BRANCH (Offline, Batch)                         |
|                                                                               |
|  NOAA GEFSv12 Reforecasts (0.5°)  ---+                                        |
|  IMD Gridded Precipitation (0.25°)---+--> Ingest -> Spatial Align -> Label    |
|  ECMWF ERA5 Synoptic Reanalysis   ---+                                |       |
|                                                                       v       |
|  ERA5 Synoptic Fields (Z500, U850, V850, T2m, MSL)             Tabular Feats  |
|           |                                                           |       |
|           v                                                           |       |
|  Empirical Orthogonal Function (EOF) Reduction                        |       |
|           |                                                           |       |
|           v                                                           |       |
|  Low-Rank Mahalanobis Metric Learning (NCA / Triplet)                 |       |
|           |                                                           |       |
|           v                                                           |       |
|  Analogue Precedent Engine (k=8 Nearest Synoptic Days)                |       |
|           |                                                           |       |
|           +-----------------------+-----------------------------------+       |
|                                   |                                           |
|                                   v                                           |
|                      Gradient Boosted Fusion Layer                            |
|                                   |                                           |
|                                   v                                           |
|                      Out-of-Fold Isotonic Calibration                         |
|                                   |                                           |
|                                   v                                           |
|                      Versioned Serialization Store                            |
|                   (models/current.json & loyo_benchmarks)                     |
+-------------------------------------------------------------------------------+
                                    |
                                    | Read-Only Model Weights & Metadata
                                    v
+-------------------------------------------------------------------------------+
|                      SERVING BRANCH (Read-Only Runtime)                       |
|                                                                               |
|   Versioned Artefacts Store  -->  FastAPI Serving Service (Port 8000)         |
|                                    • /api/v1/forecast/card                    |
|                                    • /api/v1/forecast/map                     |
|                                    • /api/v1/forecast/triage                  |
|                                    • /api/v1/regimes                          |
|                                    • /api/v1/replay                           |
|                                    • /api/v1/model/status                     |
|                                           |                                   |
|                                           v                                   |
|                          React / Vite Operational Forecaster HUD              |
|                                           |                                   |
|                 +-------------------------+-------------------------+         |
|                 |                         |                         |         |
|                 v                         v                         v         |
|         Interactive Synoptic      Triage Queue & Hazard    A4 Operational     |
|          Confidence Heatmap         Precedent Strip        Dispatch Bulletin  |
+-------------------------------------------------------------------------------+
```

---

## Operational Guardrails

1. **Deterministic Immutability**: The operational serving branch executes zero model retraining, parameter fitting, or dataset re-indexing during online requests. Every response emitted by the API is derived from serialized, version-controlled artifacts stamped with a unique `run_id` and verification hash.
2. **Out-of-Distribution (OOD) Refusal**: When atmospheric states diverge beyond calibrated Mahalanobis covariance bounds ($D_M^2 > \chi^2_{p, 0.99}$), the model server refuses over-confident extrapolation and triggers an operational fallback protocol with explicit forecaster advisory flags.
3. **Sub-Second Latency Budget**: The complete inference pipeline (synoptic projection $\to$ metric evaluation $\to$ analogue retrieval $\to$ isotonic fusion) executes in $< 120\text{ ms}$ on commodity server hardware.
