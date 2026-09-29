"""Generate simplified 36-subdivision GeoJSON and TopoJSON for Pratyaya Dashboard.

Dissolves 675 IMD districts from data/interim/india-district-imd.geojson into
the 36 meteorological subdivisions of India per docs/06-design-ux.md §3.
Generates lightweight local vector geometries (< 200 KB) for offline operation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import geopandas as gpd
import shapely

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_subdivisions")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GEOJSON_INPUT = PROJECT_ROOT / "data" / "interim" / "india-district-imd.geojson"
OUTPUT_DIR = PROJECT_ROOT / "dashboard" / "src" / "data"
GEOJSON_OUTPUT = OUTPUT_DIR / "india_subdivisions_geo.json"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading districts from %s...", GEOJSON_INPUT)
    gdf = gpd.read_file(GEOJSON_INPUT)
    logger.info("Loaded %d district features.", len(gdf))

    # Clean geometries
    gdf["geometry"] = gdf.geometry.buffer(0)

    # Dissolve by subdivision
    logger.info("Dissolving districts into 36 meteorological subdivisions...")
    subs = gdf.dissolve(by=["subdivision_id", "subdivision", "region"], as_index=False)
    subs["subdivision_id_int"] = subs["subdivision_id"].astype(int)
    subs = subs.sort_values("subdivision_id_int").reset_index(drop=True)

    # Ensure region_id as string (1-36) matches API schemas
    subs["region_id"] = subs["subdivision_id_int"].astype(str)
    subs["subdivision_name"] = subs["subdivision"]

    logger.info("Dissolved into %d subdivisions.", len(subs))

    # Simplify geometries: 0.035 degrees gives crisp coastlines and borders while keeping payload < 150 KB
    logger.info("Simplifying geometry (tolerance=0.035 deg)...")
    subs["geometry"] = subs["geometry"].simplify(tolerance=0.035, preserve_topology=True)

    # Prepare GeoJSON properties
    export_gdf = subs[["region_id", "subdivision_name", "region", "geometry"]].copy()
    export_gdf.rename(columns={"subdivision_name": "name"}, inplace=True)

    geojson_str = export_gdf.to_json(drop_id=False)
    geojson_bytes = len(geojson_str.encode("utf-8"))
    logger.info("GeoJSON payload size: %d bytes (%.1f KB)", geojson_bytes, geojson_bytes / 1024)

    with open(GEOJSON_OUTPUT, "w", encoding="utf-8") as f:
        f.write(geojson_str)
    logger.info("Saved GeoJSON to %s", GEOJSON_OUTPUT)


if __name__ == "__main__":
    main()
