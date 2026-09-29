# Deep-Dive: Raw Data Ingest Pipelines

In-tree technical documentation for raw weather data ingestion in `src/pratyaya/ingest/`.

---

## 1. Responsibilities

The `ingest` package contains one parser module per raw source:
- `gefs.py`: NOAA GEFSv12 retrospective forecast (2000–2019) from S3 `noaa-gefs-retrospective` (Node F1).
- `era5.py`: ERA5 atmospheric states and surface diagnostics from WeatherBench2 Zarr on GCS (Node F2).
- `imd.py`: IMD 0.25° gridded daily rainfall truth via `imdlib` (Node F2).
- `ecmwf_open.py`: Live ECMWF open data (IFS / AIFS ENS) for demo mode only.

Downstream pipeline stages only interact with canonical parquet schemas.

---

## 2. Data Shapes and Interfaces

### Canonical Forecasts Table (`data/canonical/forecasts/year=YYYY/*.parquet`)

```python
{
    "init_date": "datetime64[ns]",    # UTC 00 cycle date
    "region_id": "string",            # IMD subdivision ID (1..36)
    "lead_day": "int32",              # Forecast lead day (1..10)
    "variable": "string",             # e.g. "rainfall" | "tmax"
    "model_source": "string",         # "gefsv12_reforecast"
    "forecast_value": "float64",      # Ensemble-mean, area-weighted to region
    "ens_spread": "float64",          # Standard deviation across ensemble members
    "ens_members": "list[float64]",   # Raw values of individual members
    "n_members": "int32"              # Number of members (5 or 11)
}
```

### Canonical Observations Table (`data/canonical/observations/year=YYYY/*.parquet`)

```python
{
    "valid_date": "datetime64[ns]",   # Verification date (aligned to truth)
    "region_id": "string",            # IMD subdivision ID (1..36)
    "variable": "string",             # e.g. "rainfall" | "tmax"
    "observed_value": "float64",      # Area-weighted observed value
    "source": "string",               # "imd_gridded" | "era5"
    "raw_date": "datetime64[ns]"      # Pre-offset original date
}
```

---

## 3. Code Pointers

- GEFSv12 S3 ingest: `src/pratyaya/ingest/gefs.py#L1-L220`
- ERA5 WeatherBench2 GCS ingest: `src/pratyaya/ingest/era5.py#L1-L200`
- Smoke ingest test suite: `tests/test_ingest_smoke.py#L1-L120`

---

### Gotchas

- **GRIB2 Hypercubes (`cfgrib`):** NOAA GEFS GRIB2 files often contain multiple hypercubes in a single file (different levels or step units). Always specify explicit `filter_by_keys` (e.g. `typeOfLevel="surface"`, `stepType="accum"`).
- **R-1 Fallback Rule:** If GRIB2/eccodes exhibits environmental blockers, do not exceed 2 hours debugging. Fall back to WeatherBench2 IFS Zarr (`gs://weatherbench2/...`) per `docs/08-risks-and-kill-criteria.md` §R-1.
- **Grain discipline:** Forecasts grain is `(init_date, region_id, lead_day, variable)`. Observations grain is `(valid_date, region_id, variable)`. Never produce duplicate rows at these grains.
- **Strict Held-Out Separation:** Ingestion code must never process or touch held-out evaluation years (`2018-2019`) unless explicitly invoked under `--final`.
