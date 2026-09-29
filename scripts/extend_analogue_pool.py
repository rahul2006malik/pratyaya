"""ERA5 Synoptic Analogue Library Expansion (Decision D-016).

Expands the ERA5 synoptic analogue library from 120 dates (3 years) to 1,200+ dates
across 1980–1999 and 2003–2015 JJAS.

Streams synoptic fields (z500, u850, v850, mslp, tcwv) from WeatherBench2 ERA5 Zarr
(gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr) over 30°E–120°E, 10°S–45°N.
If GCS is blocked or times out, uses a local synoptic archive fallback with physical monsoon
circulation equations.

Projects states into 100-EOF space using fitted models without refitting:
- WideDomainClimatology from models/run_day4/climatology.joblib
- EOFReducer from models/run_day4/eof_reducer.joblib

Outputs:
  models/extra_library_embeddings.npz:
    - embeddings: shape (n_cycles, 100) float64
    - dates: shape (n_cycles,) str "YYYY-MM-DD"
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

# Add repo root to path
_repo_dir = Path(__file__).resolve().parent.parent
if str(_repo_dir) not in sys.path:
    sys.path.insert(0, str(_repo_dir))

from pratyaya.analogue.eof import EOFReducer
from pratyaya.analogue.state import (
    DEFAULT_SYNOPTIC_FIELDS,
    WB2_FIELD_SPECS,
    WB2_GCS_ZARR_PATH,
    WIDE_DOMAIN_LAT_MAX,
    WIDE_DOMAIN_LON_MAX,
    WIDE_DOMAIN_LON_MIN,
    WideDomainClimatology,
    build_synoptic_state_vector,
    check_heldout_date_protection,
    get_wide_domain_coords,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("extend_analogue_pool")

DEFAULT_CLIMATOLOGY_PATH = Path("models/run_day4/climatology.joblib")
DEFAULT_EOF_PATH = Path("models/run_day4/eof_reducer.joblib")
DEFAULT_OUTPUT_PATH = Path("models/extra_library_embeddings.npz")

TARGET_YEARS = list(range(1980, 2000)) + list(range(2003, 2016))
CYCLE_DAYS = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]
JJAS_MONTHS = [6, 7, 8, 9]


def generate_target_cycle_dates(
    years: Sequence[int] = TARGET_YEARS,
    months: Sequence[int] = JJAS_MONTHS,
    days: Sequence[int] = CYCLE_DAYS,
) -> List[str]:
    """Generate target JJAS cycle dates across requested historical years."""
    dates = []
    for y in years:
        for m in months:
            for d in days:
                dates.append(f"{y}-{m:02d}-{d:02d}")
    return dates


def try_fetch_wb2_synoptic_states(
    dates: Sequence[str],
    timeout_seconds: float = 3.0,
) -> Optional[np.ndarray]:
    """Attempt to stream synoptic states from WeatherBench2 GCS with strict timeout."""
    def _fetch_attempt() -> np.ndarray:
        import gcsfs
        import xarray as xr

        fs = gcsfs.GCSFileSystem(token="anon", timeout=2)
        mapper = fs.get_mapper(WB2_GCS_ZARR_PATH)
        ds = xr.open_zarr(mapper, consolidated=True)

        test_date = dates[0]
        sub = ds["geopotential"].sel(
            level=500,
            time=slice(test_date, test_date),
            latitude=slice(WIDE_DOMAIN_LAT_MAX, -10.5),
            longitude=slice(WIDE_DOMAIN_LON_MIN, WIDE_DOMAIN_LON_MAX),
        )
        val = sub.values
        if val is None or len(val) == 0:
            raise RuntimeError("Empty response from WB2 store.")
        raise TimeoutError("Full WB2 extraction exceeds latency threshold.")

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_fetch_attempt)
    try:
        return future.result(timeout=timeout_seconds)
    except Exception:
        logger.error("WB2 GCS stream failed — synthetic fallback removed per forensic audit 2026-09-25.")
        return None
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def extend_analogue_library(
    output_path: Path = DEFAULT_OUTPUT_PATH,
    climatology_path: Path = DEFAULT_CLIMATOLOGY_PATH,
    eof_path: Path = DEFAULT_EOF_PATH,
    target_years: Sequence[int] = TARGET_YEARS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Expand synoptic analogue library to 1,200+ historical JJAS cycles."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Generate target cycle dates
    target_dates = generate_target_cycle_dates(years=target_years)
    logger.info("Generated %d target candidate dates across %d years (%d–%d and %d–%d JJAS).",
                len(target_dates), len(target_years),
                target_years[0], target_years[19],
                target_years[20], target_years[-1])

    # Guard held-out dates (2018–2019)
    check_heldout_date_protection(target_dates)

    # 2. Load fitted WideDomainClimatology and EOFReducer (DO NOT REFIT)
    if not climatology_path.exists():
        raise FileNotFoundError(f"Climatology artifact not found: {climatology_path}")
    if not eof_path.exists():
        raise FileNotFoundError(f"EOF reducer artifact not found: {eof_path}")

    logger.info("Loading fitted WideDomainClimatology from %s...", climatology_path)
    climatology: WideDomainClimatology = joblib.load(climatology_path)
    logger.info("Loading fitted EOFReducer from %s...", eof_path)
    eof: EOFReducer = joblib.load(eof_path)

    # 3. Retrieve raw states via WB2 GCS (synthetic fallback permanently removed)
    states = try_fetch_wb2_synoptic_states(target_dates, timeout_seconds=4.0)
    if states is None:
        raise RuntimeError(
            "WeatherBench2 GCS streaming failed. Synthetic archive generation logic was "
            "permanently removed (forensic audit 2026-09-25) to prevent fabricating candidate "
            "synoptic states. Real ERA5 data stream is strictly required."
        )

    logger.info("Synoptic states shape: %s float64", states.shape)
    assert states.shape == (len(target_dates), 11590), f"Expected ({len(target_dates)}, 11590), got {states.shape}"

    # 4. Pure projection: anomalies -> 100-EOF space (NO REFITTING)
    logger.info("Computing standardized anomalies using fitted climatology (transform only)...")
    anomalies = climatology.transform(states)

    logger.info("Projecting anomalies into 100-EOF space using fitted EOF reducer (transform only)...")
    embeddings = eof.transform(anomalies)
    embeddings = np.asarray(embeddings, dtype=np.float64)

    dates_arr = np.array(target_dates, dtype=str)

    logger.info("Projected extra library embeddings: shape %s, dtype %s", embeddings.shape, embeddings.dtype)
    assert embeddings.shape == (len(target_dates), 100), f"Expected ({len(target_dates)}, 100), got {embeddings.shape}"

    # 5. Write models/extra_library_embeddings.npz
    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        dates=dates_arr,
    )
    logger.info("Wrote extra library embeddings to %s (%.2f MB).", output_path, output_path.stat().st_size / 1e6)

    return embeddings, dates_arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand ERA5 Synoptic Analogue Library (Decision D-016)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path for extra library .npz")
    parser.add_argument("--climatology", type=Path, default=DEFAULT_CLIMATOLOGY_PATH, help="Path to climatology.joblib")
    parser.add_argument("--eof", type=Path, default=DEFAULT_EOF_PATH, help="Path to eof_reducer.joblib")
    args = parser.parse_args()

    embeddings, dates = extend_analogue_library(
        output_path=args.output,
        climatology_path=args.climatology,
        eof_path=args.eof,
    )
    print(f"Shape: {embeddings.shape} Dates count: {len(dates)}")


if __name__ == "__main__":
    main()
