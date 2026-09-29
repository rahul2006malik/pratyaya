# Data Contracts & Geographical Topologies

Pratyaya integrates heterogeneous meteorological streams into standardized, area-weighted canonical schemas.

---

## 1. Primary Meteorological Ingestion Streams

| Source Stream | Provider | Spatial Resolution | Temporal Cadence | Target Format |
|---|---|---|---|---|
| **GEFSv12 Reforecast** | NOAA / NCEP | $0.5^\circ \times 0.5^\circ$ Global Grid | Daily 00 UTC (Leads 1–10) | Partitioned Apache Parquet |
| **Gridded Precipitation** | India Meteorological Department (IMD) | $0.25^\circ \times 0.25^\circ$ India Box | Daily 08:30 IST accumulation | Partitioned Apache Parquet |
| **ERA5 Synoptic Reanalysis** | ECMWF / Copernicus | $1.5^\circ \times 1.5^\circ$ Wide Domain | Daily 00 UTC snapshots | Serialized Zarr / NumPy |

---

## 2. Temporal & Clock Alignment Contract

A primary failure mode in operational numerical weather prediction verification is clock misalignment between UTC model initialization cycles and national rainfall gauge accumulation windows:

- **IMD Operational Rain-Gauge Day:** Accumulation period runs from **08:30 IST to 08:30 IST** ($\text{Day } T$).
- **NWP Model Cycle Day:** Runs on standard **00:00 UTC to 00:00 UTC** ($\text{Day } T$).
- **Alignment Offset ($\Delta t$):** Pratyaya enforces a mathematically validated $+1\text{ day}$ alignment rule (`IMD_DAY_OFFSET = 1`) derived from empirical cross-correlation analysis over Western Ghats orographic precipitation records to eliminate artificial clock-induced error spikes.

---

## 3. Spatial Aggregation & Subdivision Topologies

Forecast bust detection operates over all **36 canonical meteorological subdivisions** defined by the India Meteorological Department. 

### Area Weighting Formulation
Given an irregular subdivision polygon $\mathcal{P}_k$ and high-resolution gridded field $F(x_i, y_j)$:

$$\bar{F}_k = \frac{\sum_{(i,j) \in \mathcal{P}_k} F(x_i, y_j) \cdot \cos(\phi_j) \cdot \mathcal{A}(\text{cell}_{i,j} \cap \mathcal{P}_k)}{\sum_{(i,j) \in \mathcal{P}_k} \cos(\phi_j) \cdot \mathcal{A}(\text{cell}_{i,j} \cap \mathcal{P}_k)}$$

where $\mathcal{A}(\cdot)$ calculates polygon intersection area via high-precision computational geometry (`shapely` / `geopandas`).

---

## 4. Definition of a Forecast Bust Event

A forecast initialization cycle $t$, lead day $\tau$, and subdivision $k$ is designated as a **Primary Forecast Bust** if and only if both conditions are simultaneously satisfied:

1. **Extreme Absolute Error:**
   $$|\hat{y}_{t, \tau, k} - y_{t+\tau, k}| \ge \max\left(15.0\text{ mm/day}, \, 2.0 \cdot \sigma_{\text{clim}}(k, \text{month})\right)$$
2. **Standardized Severity Floor:**
   $$\frac{|\hat{y}_{t, \tau, k} - y_{t+\tau, k}|}{\sigma_{\text{subdiv}}(k)} \ge 1.75$$

This dual formulation guarantees that normal monsoon drizzle variations are filtered out, focusing computational attention strictly on high-impact operational forecast busts.
