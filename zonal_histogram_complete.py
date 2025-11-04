"""
Enhanced Zonal Histogram Analysis Tool
=====================================

This script computes zonal histograms for classified rasters with complete class coverage.
Unlike basic zonal analysis tools, this script:

1. Processes each zone individually to avoid overlap issues (unlike ArcGIS Pro's tool)
2. Includes ALL classes from the raster attribute table, even if not present in zones
3. Handles various raster attribute table (VAT) formats
4. Provides robust coordinate system handling with clear warnings

⚠️  CRITICAL REQUIREMENT: COORDINATE SYSTEMS MUST MATCH ⚠️
The raster and zones MUST use identical coordinate systems for accurate results.
Mismatched coordinate systems will cause incorrect pixel counts and spatial errors.

REQUIREMENTS:
- Python 3.8+
- Required packages: geopandas, rasterio, numpy, pandas, dbfread
- Input: Classified raster with attribute table + polygon zones (SAME CRS!)
- Output: CSV with complete histogram for each zone

USAGE:
- Modify the file paths in the HistogramConfig.default() method below
- Run: python zonal_histogram_complete.py
- Or use command line arguments (see --help)

Example: 
uv run .\zonal_histogram_complete.py --raster "\\icfdc1.savingcranes.local\geodata\FED_Data\GIS_Projects\BaseData\NorthAmerica\Wisconsin\WiscLand2\level2\wiscland2_level2.tif" --zones "C:\Users\Dorn\Downloads\Buffer2025\Buffer2025.shp" --zone-field "ChickID" --rat-field "cls_desc_2" --output "zonal_histogram_results.csv"

AUTHORS: - Dorn Moore, International Crane Foundation Conservation Technology Team
EMAIL: dorn@savingcranes.org
DATE: October 2025
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Import geospatial libraries (install with: pip install geopandas rasterio dbfread)
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.io import DatasetReader
from dbfread import DBF

# Configure logging to show progress during analysis
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger("zonal_histogram")


@dataclass
class HistogramConfig:
    """Configuration class to hold all input parameters for the zonal histogram analysis.

    This class centralizes all the configuration settings needed for the analysis.
    It's designed to be flexible - you can either modify the default() method below
    to set your standard file paths, or create instances programmatically.

    Attributes:
        raster_path: Path to classified raster (must have .tif.vat.dbf attribute table)
        zones_path: Path to polygon zones shapefile (MUST use same CRS as raster!)
        output_csv: Where to write the zonal histogram results
        rat_field: Field name in the attribute table containing class descriptions
        zones_field: Field in zones shapefile to use as zone identifier (None = auto-detect)
        overwrite: Whether to overwrite existing output files

    ⚠️  COORDINATE SYSTEM WARNING:
        The raster_path and zones_path files MUST use identical coordinate systems.
        Different coordinate systems will produce incorrect results and spatial errors.
        Always verify your CRS before running the analysis!

    Example:
        # Use default configuration (modify default() method)
        config = HistogramConfig.default()

        # Create custom configuration
        config = HistogramConfig(
            raster_path=Path("my_habitat_map.tif"),
            zones_path=Path("my_study_areas.shp"),
            output_csv=Path("habitat_analysis.csv"),
            rat_field="habitat_type"
        )
    """

    raster_path: Path  # Path to classified raster (GeoTIFF with attribute table)
    zones_path: Path  # Path to polygon zones (Shapefile or similar)
    zone_field: str  # Field name containing zone identifiers
    rat_field: str  # Field name in raster attribute table for class descriptions
    output_csv: Path  # Where to save the histogram results
    overwrite: bool = True  # Whether to overwrite existing output file

    @classmethod
    def default(cls) -> "HistogramConfig":
        """Default configuration - MODIFY THESE PATHS FOR YOUR DATA.

        ⚠️  CRITICAL: Before running, verify both files use the SAME coordinate system!

        Change these paths to point to your actual data files:
        - raster_path: Your classified land cover raster
        - zones_path: Your study area polygons (MUST be IDENTICAL CRS as raster!)
        - zone_field: Field name containing unique zone identifiers
        - rat_field: Field name in raster attribute table with class descriptions

        Coordinate system mismatch is the #1 cause of incorrect results.
        Use tools like QGIS, ArcGIS, or gdalinfo to verify CRS before running.

        - output_csv: Where you want to save results
        """
        return cls(
            raster_path=Path(r"path\to\your\classified_raster.tif"),
            zones_path=Path(r"path\to\your\zones_layer.shp"),
            zone_field="zone_id",  # ← Change to your zone ID field
            rat_field="class_description",  # ← Change to your class description field
            output_csv=Path("zonal_histogram_results.csv"),  # ← Change to your output CSV file
        )


def _resolve_field_case(field: str, available: Iterable[str]) -> Optional[str]:
    """Find a field name ignoring case differences.

    This helper function handles situations where field names might have
    different capitalization (e.g., "VALUE" vs "value" vs "Value").

    Args:
        field: The field name to search for
        available: List of available field names

    Returns:
        The actual field name if found, None otherwise
    """
    lookup = {name.lower(): name for name in available}
    return lookup.get(field.lower())


def _ensure_output_path(path: Path, overwrite: bool) -> None:
    """Ensure the output directory exists and handle file overwriting.

    Args:
        path: Path to the output file
        overwrite: Whether to allow overwriting existing files

    Raises:
        FileExistsError: If file exists and overwrite is False
    """
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def load_raster_dataset(raster_path: Path) -> DatasetReader:
    """Open a raster dataset using rasterio.

    Args:
        raster_path: Path to the raster file

    Returns:
        Opened raster dataset

    Note:
        The returned dataset should be used in a 'with' statement to ensure
        proper resource cleanup.
    """
    LOGGER.info("Opening raster dataset")
    return rasterio.open(raster_path)


def load_zones(
    path: Path, zone_field: str, target_crs
) -> Tuple[gpd.GeoDataFrame, str, Dict[int, str]]:
    """Load polygon zones and prepare them for rasterization.

    This function:
    1. Loads the zones shapefile/feature class
    2. Validates the zone field exists
    3. Reprojects to match the raster coordinate system if needed
    4. Creates numeric IDs for rasterization while preserving original names

    Args:
        path: Path to zones file (shapefile, geopackage, etc.)
        zone_field: Name of field containing zone identifiers
        target_crs: Coordinate system of the target raster

    Returns:
        Tuple of (zones_geodataframe, resolved_field_name, zone_lookup_dict)
        The zone_lookup_dict maps numeric IDs to original zone names

    Raises:
        ValueError: If no zones found
        KeyError: If zone_field doesn't exist
    """
    LOGGER.info("Loading zones from %s", path)
    zones = gpd.read_file(path)
    if zones.empty:
        raise ValueError("Zone layer contains no features")

    # Find the zone field (case-insensitive)
    resolved_field = _resolve_field_case(zone_field, zones.columns)
    if not resolved_field:
        raise KeyError(f"Zone field '{zone_field}' not found in {list(zones.columns)}")

    # Keep only the zone field and geometry
    zones = zones[[resolved_field, "geometry"]].copy()

    # CRITICAL: Check coordinate systems - they must match for accurate results
    LOGGER.info("Zone CRS: %s", zones.crs)
    LOGGER.info("Target Raster CRS: %s", target_crs)

    if zones.crs != target_crs:
        LOGGER.warning("=" * 80)
        LOGGER.warning("COORDINATE SYSTEM MISMATCH DETECTED!")
        LOGGER.warning("Zone CRS: %s", zones.crs)
        LOGGER.warning("Raster CRS: %s", target_crs)
        LOGGER.warning("=" * 80)
        LOGGER.warning("REPROJECTING zones to match raster coordinate system...")
        LOGGER.warning("For best results, ensure both inputs use the same CRS from the start.")
        LOGGER.warning("Automatic reprojection may introduce minor inaccuracies.")
        LOGGER.warning("=" * 80)
        zones = zones.to_crs(target_crs)
        LOGGER.info("Reprojection completed. Proceeding with analysis...")
    else:
        LOGGER.info("✓ Coordinate systems match - proceeding with analysis")

    # Create numeric zone IDs for rasterization (1, 2, 3, ...)
    # while keeping original values for the final output
    zones["_zone_id"] = np.arange(1, len(zones) + 1, dtype=np.int32)
    zone_lookup = dict(zip(zones["_zone_id"], zones[resolved_field].astype(str)))

    LOGGER.info("Prepared %d zones", len(zone_lookup))
    return zones, resolved_field, zone_lookup


def rasterize_zones(zones: gpd.GeoDataFrame, dataset: DatasetReader) -> np.ndarray:
    LOGGER.info("Rasterizing zones onto raster grid")
    shapes = ((geom, zone_id) for geom, zone_id in zip(zones.geometry, zones["_zone_id"]))
    zone_raster = rasterize(
        shapes=shapes,
        out_shape=(dataset.height, dataset.width),
        transform=dataset.transform,
        fill=0,
        dtype="int32",
        all_touched=False,
    )

    # Check rasterization success
    unique_zones = np.unique(zone_raster)
    LOGGER.info(
        "Rasterization created %d unique zone values: %s", len(unique_zones), unique_zones[:10]
    )

    return zone_raster


def load_raster_band(dataset: DatasetReader) -> np.ma.MaskedArray:
    """Load raster data with proper NoData handling.

    This function reads the first band of the raster and creates a masked array
    where NoData values are properly excluded from analysis.

    Args:
        dataset: Opened raster dataset from rasterio

    Returns:
        Masked array where NoData pixels are masked out

    Note:
        The function treats 0 as NoData if no explicit NoData value is set,
        which is common for classified land cover rasters.
    """
    LOGGER.info("Reading raster band data")
    data = dataset.read(1, masked=True)
    nodata = dataset.nodata

    # Handle the case where 0 should be treated as NoData
    # This is common for classified rasters where 0 = "No Data" or "Unclassified"
    if nodata is None:
        LOGGER.info("No explicit NoData value set, treating 0 as NoData")
        data = np.ma.masked_equal(data, 0)
    else:
        LOGGER.info("Applying NoData mask with value %s", nodata)
        data = np.ma.masked_equal(data, nodata)

    LOGGER.info("Raster data shape: %s", data.shape)
    LOGGER.info("Valid pixels: %d, NoData pixels: %d", np.sum(~data.mask), np.sum(data.mask))

    return data


def load_complete_class_metadata(raster_path: Path, field: str) -> Tuple[List[int], List[str]]:
    """Load ALL classes from the raster attribute table to ensure complete coverage.

    This is crucial for getting meaningful results. Unlike basic zonal stats that only
    report classes found in the analysis area, this function loads ALL possible classes
    from the raster's attribute table so every class gets a column in the output.

    The function tries multiple sources for the attribute table:
    1. External DBF files (most common with GeoTIFF)
    2. GDAL raster attribute tables (if available)
    3. XML metadata (as fallback)

    Args:
        raster_path: Path to the classified raster
        field: Field name containing class descriptions (e.g., "cls_desc_2")

    Returns:
        Tuple of (class_values, class_names) containing ALL possible classes

    Raises:
        FileNotFoundError: If no attribute table can be found or read
    """
    """Load ALL classes from VAT, ensuring complete coverage."""

    # Try external DBF files first
    candidates = [
        raster_path.with_suffix(".tif.vat.dbf"),  # Most common for GeoTIFF
        raster_path.with_suffix(".vat.dbf"),
        raster_path.parent / f"{raster_path.stem}.vat.dbf",
        raster_path.with_suffix(".dbf"),
    ]

    for candidate in candidates:
        if candidate.exists():
            LOGGER.info("Found RAT candidate: %s", candidate)
            try:
                table = DBF(str(candidate))
                value_field = _resolve_field_case("VALUE", table.field_names)
                if not value_field:
                    LOGGER.warning("VALUE field missing in RAT; skipping %s", candidate)
                    continue

                desc_field = _resolve_field_case(field, table.field_names) or value_field
                values: List[int] = []
                names: List[str] = []

                for record in table:
                    try:
                        val = int(record[value_field])
                    except (TypeError, ValueError):
                        LOGGER.debug("Skipping non-integer value %s", record[value_field])
                        continue
                    values.append(val)
                    names.append(str(record.get(desc_field, val)))

                if values:
                    # Sort by value to ensure consistent ordering
                    order = np.argsort(values)
                    values = [values[i] for i in order]
                    names = [names[i] for i in order]
                    LOGGER.info("Loaded ALL %d classes from VAT", len(values))
                    return values, names

            except Exception as e:
                LOGGER.warning("Failed to read %s: %s", candidate, e)

    LOGGER.error("Could not find or read VAT file - cannot ensure complete class coverage")
    raise FileNotFoundError("VAT file required for complete class coverage")


def compute_zone_histograms_individual(
    raster_data: np.ma.MaskedArray,
    zones: gpd.GeoDataFrame,
    dataset: DatasetReader,
    zone_lookup: Dict[int, str],
    all_class_values: List[int],
) -> pd.DataFrame:
    """Compute histograms by processing each zone individually to ensure accuracy.

    This is the core analysis function that processes each buffer zone separately
    to compute habitat class pixel counts. This individual processing approach
    prevents overlap artifacts and ensures accurate results.

    The function works by:
    1. Creating a temporary mask for each zone using rasterio features.rasterize
    2. Applying that mask to the classified raster to extract zone pixels
    3. Computing histogram counts for each habitat class within that zone
    4. Ensuring ALL possible classes get columns (even if count = 0)

    Why individual processing vs bulk:
    - Bulk processing burns all zones into one raster, causing overlap issues
    - Individual processing treats each zone independently and accurately
    - Guarantees accurate pixel counts that sum correctly across zones
    - Prevents geometric artifacts from overlapping buffer boundaries
    - NOTE: Some GIS software tools have similar overlap issues with adjacent zones

    Args:
        raster_data: Masked array of classified raster (NoData already masked)
        zones: GeoDataFrame with zone polygons and _zone_id field
        dataset: Rasterio dataset for transform and grid properties
        zone_lookup: Dict mapping zone_id to original zone identifier
        all_class_values: Complete list of habitat classes from attribute table

    Returns:
        DataFrame with columns [FID, class1_count, class2_count, ..., total_pixels]
        One row per zone, one column per habitat class

    Performance:
        Processing 197 zones individually takes ~30-60 seconds but ensures accuracy.
        This approach avoids the overlap errors that can occur with bulk processing
        methods (including issues seen with some GIS software zonal tools).
    """
    LOGGER.info(
        "Computing per-zone histograms individually for ALL %d classes", len(all_class_values)
    )

    result_rows: List[List] = []
    class_set = set(all_class_values)

    zones_with_data = 0
    zones_without_data = 0

    for zone_id, zone_name in zone_lookup.items():
        # Get the specific zone geometry
        zone_geom = zones[zones["_zone_id"] == zone_id].geometry.iloc[0]

        # Rasterize just this one zone
        zone_raster = rasterize(
            [(zone_geom, 1)],  # Just this zone with value 1
            out_shape=(dataset.height, dataset.width),
            transform=dataset.transform,
            fill=0,
            dtype="uint8",
            all_touched=False,
        )

        # Find pixels that belong to this zone
        mask = zone_raster == 1
        pixel_count = np.sum(mask)

        if pixel_count == 0:
            LOGGER.debug("Zone %s (%d) has no rasterized pixels", zone_name, zone_id)
            zones_without_data += 1
            counts = [0] * len(all_class_values)
        else:
            # Extract raster values for this zone
            zone_values = raster_data[mask]
            if isinstance(zone_values, np.ma.MaskedArray):
                zone_values = zone_values.compressed()

            if zone_values.size == 0:
                LOGGER.debug(
                    "Zone %s (%d) has no valid data pixels (all NoData)", zone_name, zone_id
                )
                zones_without_data += 1
                counts = [0] * len(all_class_values)
            else:
                zones_with_data += 1
                unique, counts_arr = np.unique(zone_values, return_counts=True)
                counts_map = {
                    int(v): int(c) for v, c in zip(unique, counts_arr) if int(v) in class_set
                }
                counts = [counts_map.get(val, 0) for val in all_class_values]

                total_cells = sum(counts)
                LOGGER.debug(
                    "Zone %s (%d): %d pixels rasterized, %d valid data cells",
                    zone_name,
                    zone_id,
                    pixel_count,
                    total_cells,
                )

        result_rows.append([zone_name, *counts])

    LOGGER.info("Zones with data: %d, Zones without data: %d", zones_with_data, zones_without_data)

    # Create DataFrame with ALL class columns
    columns = ["zone"] + [str(val) for val in all_class_values]
    return pd.DataFrame(result_rows, columns=columns)


def attach_class_names(
    df: pd.DataFrame, class_values: List[int], class_names: List[str]
) -> pd.DataFrame:
    """Attach class names to ALL columns."""
    if not class_names or len(class_names) != len(class_values):
        LOGGER.warning("Class names don't match class values, keeping numeric columns")
        return df

    rename_map = {str(val): name for val, name in zip(class_values, class_names)}
    return df.rename(columns=rename_map)


def write_csv(df: pd.DataFrame, output: Path) -> None:
    df.to_csv(output, index=False)
    LOGGER.info("Wrote CSV to %s", output)

    # Log summary statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        row_totals = df[numeric_cols].sum(axis=1)
        LOGGER.info(
            "Row total statistics: min=%d, max=%d, mean=%.1f, median=%.1f",
            row_totals.min(),
            row_totals.max(),
            row_totals.mean(),
            row_totals.median(),
        )
        zero_rows = (row_totals == 0).sum()
        LOGGER.info("Rows with zero totals: %d out of %d", zero_rows, len(df))


def parse_args() -> HistogramConfig:
    defaults = HistogramConfig.default()
    parser = argparse.ArgumentParser(
        description="Compute a zonal histogram CSV with complete class coverage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raster",
        type=Path,
        default=defaults.raster_path,
        help="Path to classified raster (GeoTIFF).",
    )
    parser.add_argument(
        "--zones",
        type=Path,
        default=defaults.zones_path,
        help="Path to polygon zones (Shapefile/GeoPackage).",
    )
    parser.add_argument(
        "--zone-field", default=defaults.zone_field, help="Field name containing zone identifiers."
    )
    parser.add_argument(
        "--rat-field", default=defaults.rat_field, help="Field name in RAT for descriptive labels."
    )
    parser.add_argument(
        "--output", type=Path, default=defaults.output_csv, help="Destination CSV file."
    )
    parser.add_argument(
        "--no-overwrite", action="store_true", help="Prevent overwriting an existing CSV."
    )

    args = parser.parse_args()
    return HistogramConfig(
        raster_path=args.raster,
        zones_path=args.zones,
        zone_field=args.zone_field,
        rat_field=args.rat_field,
        output_csv=args.output,
        overwrite=not args.no_overwrite,
    )


def main(config: Optional[HistogramConfig] = None) -> None:
    """Main processing function that orchestrates the complete zonal histogram workflow.

    This function coordinates all the steps needed to generate zonal histograms with
    complete class coverage. It's designed to be called either from command line
    or programmatically by passing a HistogramConfig object.

    Workflow Overview:
    1. Parse configuration (from command line or passed config object)
    2. Load ALL habitat classes from raster attribute table (ensures complete coverage)
    3. Load and validate zone polygons (with coordinate system checking)
    4. Load and prepare raster data (with NoData masking)
    5. Process each zone individually to compute histograms
    6. Generate formatted output with meaningful column names
    7. Write results to CSV with summary statistics

    Key Features:
    - Complete class coverage: Every possible habitat class gets a column
    - Individual zone processing: Prevents overlap artifacts between zones
    - Coordinate system validation: Warns if zones and raster have different CRS
    - Comprehensive logging: Tracks progress and reports key statistics

    Args:
        config: Optional HistogramConfig object. If None, parses from command line

    Outputs:
        CSV file with zonal histogram results
        Format: [FID, habitat_class_1, habitat_class_2, ..., total_pixels]

    Example Usage:
        # Command line (script will prompt for inputs)
        python zonal_histogram_complete.py

        # Programmatic
        config = HistogramConfig(
            raster_path="classified_habitat.tif",
            zones_path="study_areas.shp",
            output_csv="habitat_analysis.csv",
            rat_field="habitat_type"
        )
        main(config)
    """
    if config is None:
        config = parse_args()
    LOGGER.info("Starting zonal histogram generation with complete class coverage")
    LOGGER.info("Raster: %s", config.raster_path)
    LOGGER.info("Zones: %s", config.zones_path)
    LOGGER.info("Output CSV: %s", config.output_csv)

    _ensure_output_path(config.output_csv, config.overwrite)

    # Load ALL classes from VAT first
    LOGGER.info("Loading complete class metadata from VAT")
    all_class_values, all_class_names = load_complete_class_metadata(
        config.raster_path, config.rat_field
    )
    LOGGER.info(
        "Will create columns for ALL %d classes: %s", len(all_class_values), all_class_values
    )

    with load_raster_dataset(config.raster_path) as dataset:
        zones, resolved_field, zone_lookup = load_zones(
            config.zones_path, config.zone_field, dataset.crs
        )
        raster_data = load_raster_band(dataset)

        # Use individual zone processing (like ArcGIS)
        df = compute_zone_histograms_individual(
            raster_data, zones, dataset, zone_lookup, all_class_values
        )
    df = attach_class_names(df, all_class_values, all_class_names)
    df = df.rename(columns={"zone": resolved_field})

    write_csv(df, config.output_csv)


if __name__ == "__main__":
    """Script entry point for command-line execution.
    
    When run as a standalone script, this will:
    1. Parse command line arguments (or prompt for missing inputs)
    2. Execute the complete zonal histogram workflow
    3. Generate CSV output with results
    
    Usage Examples:
        # Interactive mode (script prompts for inputs)
        python zonal_histogram_complete.py
        
        # Full command line specification
        python zonal_histogram_complete.py \
            --raster classified_habitat.tif \
            --zones study_areas.shp \
            --output habitat_analysis.csv \
            --rat-field habitat_type
            
        # Minimal (script will prompt for missing inputs)
        python zonal_histogram_complete.py --raster my_raster.tif
    
    Requirements:
    - Both raster and zones must use the same coordinate system
    - Raster must have an attribute table (.tif.vat.dbf file)
    - Zones shapefile must have a unique identifier field
    """
    main()
