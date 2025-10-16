"""
Test data generator for the zonal histogram analysis tool.

Creates small sample datasets for testing purposes without relying on large
external files.
"""

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
from shapely.geometry import Polygon
import tempfile
import pandas as pd


def create_test_raster(output_path: Path, width: int = 10, height: int = 10):
    """Create a small test raster with classified data."""

    # Create test classification data (4 classes)
    data = np.random.choice([1, 2, 3, 4], size=(height, width), p=[0.4, 0.3, 0.2, 0.1])
    data = data.astype(np.int16)

    # Define geographic bounds
    transform = from_bounds(0, 0, width, height, width, height)

    # Write the raster
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
        compress="lzw",
    ) as dst:
        dst.write(data, 1)

    return output_path


def create_test_zones(output_path: Path, n_zones: int = 3):
    """Create test polygon zones."""

    polygons = []
    zone_ids = []

    for i in range(n_zones):
        # Create non-overlapping rectangles
        x_start = i * 2
        x_end = x_start + 3
        y_start = 0
        y_end = 3

        polygon = Polygon([(x_start, y_start), (x_end, y_start), (x_end, y_end), (x_start, y_end)])

        polygons.append(polygon)
        zone_ids.append(f"Zone_{i+1}")

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        {
            "zone_id": zone_ids,
            "name": [f"Test Zone {i+1}" for i in range(n_zones)],
            "area": [6.0] * n_zones,
            "geometry": polygons,
        },
        crs="EPSG:4326",
    )

    # Save to shapefile
    gdf.to_file(output_path)

    return output_path


def create_test_vat(raster_path: Path):
    """Create a test Value Attribute Table (VAT) for the raster."""

    vat_path = raster_path.with_suffix(".tif.vat.dbf")

    # Create VAT data
    vat_data = [
        {"VALUE": 1, "COUNT": 1000, "class_description": "Forest"},
        {"VALUE": 2, "COUNT": 800, "class_description": "Grassland"},
        {"VALUE": 3, "COUNT": 500, "class_description": "Water"},
        {"VALUE": 4, "COUNT": 200, "class_description": "Urban"},
    ]

    # Convert to DataFrame and save as DBF
    df = pd.DataFrame(vat_data)

    # Note: For actual DBF creation, you'd need a library like dbfpy or simpledbf
    # For testing, we'll create a simple CSV that our mock can read
    csv_path = vat_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)

    return vat_path


def create_test_dataset(temp_dir: Path = None):
    """Create a complete test dataset with raster, zones, and VAT."""

    if temp_dir is None:
        temp_dir = Path(tempfile.mkdtemp())

    # Create test files
    raster_path = create_test_raster(temp_dir / "test_raster.tif")
    zones_path = create_test_zones(temp_dir / "test_zones.shp")
    vat_path = create_test_vat(raster_path)

    return {"raster": raster_path, "zones": zones_path, "vat": vat_path, "temp_dir": temp_dir}


if __name__ == "__main__":
    """Generate test dataset for manual inspection."""

    test_data = create_test_dataset(Path("test_data"))

    print("Test dataset created:")
    print(f"Raster: {test_data['raster']}")
    print(f"Zones: {test_data['zones']}")
    print(f"VAT: {test_data['vat']}")
    print(f"Location: {test_data['temp_dir']}")
