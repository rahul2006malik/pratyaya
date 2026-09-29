# Deep-Dive: Spatial Alignment and Area Weights

In-tree technical documentation for spatial and temporal alignment components in `src/pratyaya/align/`.

---

## 1. Responsibilities

The `align` package is responsible for:
- Computing exact polygon × grid-cell area overlaps between IMD 0.25° grid and the 36 meteorological subdivisions of India (`areaweights.py`).
- Caching the normalized sparse area-weight matrix to eliminate centroid-masking distortion.
- Aggregating gridded forecasts and truth grids to subdivision-level series.
- Rain-day UTC-to-IST alignment (`rainday.py`, governed by `D-001`).

---

## 2. Data Shapes and Interfaces

### Area Weights Cache Schema (`data/interim/area_weights_subdivisions_36.npz`)

```python
# Compressed NumPy archive containing:
# - data: float64 array of non-zero overlap weights
# - indices: int32 array of column (cell) indices
# - indptr: int32 array of index pointers (shape: 37,)
# - shape: tuple (36, n_cells)
# - lats: float64 array of grid latitudes (shape: 129,) [6.5 to 38.5]
# - lons: float64 array of grid longitudes (shape: 135,) [66.5 to 100.0]
# - subdivision_ids: int32 array of subdivision identifiers (1 to 36)
# - subdivision_names: str array of subdivision names
```

### Subdivision Regions Schema (`data/canonical/regions.parquet`)

```python
{
    "region_id": "str",         # e.g. "7"
    "name": "str",              # e.g. "Konkan & Goa"
    "area_km2": "float64",      # total calculated subdivision area
    "geometry": "binary (WKB)"  # MultiPolygon / Polygon in EPSG:4326
}
```

---

## 3. Code Pointers

- Area weight calculation & caching: `src/pratyaya/align/areaweights.py#L1-L250`
- Unit tests: `tests/test_areaweights.py#L1-L100`

---

### Gotchas

- **Centroid masking produces invalid coastal weights:** Centroid-in-polygon tests omit coastline and border grid cells. Area-fraction overlay (`geopandas.overlay` / `shapely`) is mandatory per `docs/03-architecture-and-modeling.md` §3.2.
- **Normalization tolerance:** Each row sum must equal 1.0 within `1e-4` tolerance (`weights.sum(axis=1) ≈ 1.0`). If cell weights do not sum to 1.0, regional rainfall or temperature totals are artificially biased.
- **IMD Grid Alignment:** The IMD 0.25° grid spans longitude 66.5°E to 100.0°E (135 points) and latitude 6.5°N to 38.5°N (129 points). Using 0-360 longitude or reversed latitude without reindexing will corrupt spatial alignment.
- **Rain-Day Offset (`D-001`):** Observational truth uses IMD 0830 IST accumulation windows, whereas NWP uses UTC cycles. Do not hardcode temporal offsets in `align` without referencing `docs/DECISIONS.md` D-001.
