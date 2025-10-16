# Zonal Histogram Analysis Tool

A comprehensive Python tool for computing zonal histograms from classified rasters with complete class coverage. This tool ensures accurate per-zone statistics by processing each zone individually and including all possible habitat classes in the output.

## Tool Purpose

This repository contains the **zonal histogram analysis tool** - a Python script that computes accurate zonal histograms from classified rasters with complete class coverage.

**Key Achievement**: This tool solves the overlap artifacts problem found in many GIS zonal analysis tools by processing each zone individually rather than using bulk processing methods.

## Overview

This tool addresses common limitations of basic zonal statistics tools by:

- **Complete Class Coverage**: Includes ALL classes from the raster attribute table, even if not present in specific zones
- **Individual Zone Processing**: Processes each zone separately to prevent overlap artifacts
- **🎯 Overlapping Zone Support**: Handles overlapping buffer zones correctly (where other tools fail)
- **Robust Input Validation**: Handles coordinate system mismatches and validates file formats
- **Standalone Operation**: Works with open-source libraries, no ArcGIS license required

### **Why Overlapping Zone Support Matters**

Many ecological and environmental analyses use **overlapping buffer zones**:
- Multiple buffer distances around the same point (100m, 200m, 500m buffers)
- Adjacent study areas with overlapping boundaries
- Nested or hierarchical zone designs

**Problem with bulk processing**: Most GIS tools (including ArcGIS Pro's Zonal Histogram) process all zones simultaneously, causing overlap artifacts and incorrect pixel counts.

**Our solution**: Individual zone processing ensures each zone gets accurate, independent pixel counts regardless of overlaps with other zones.

## Current Repository Structure

The repository contains a clean, focused set of files:

```
README.md                          ← This documentation
pyproject.toml                     ← Modern Python project configuration (dependencies, metadata)
requirements.txt                   ← Legacy pip dependencies (for compatibility)
zonal_histogram_complete.py        ← ⭐ MAIN TOOL - Use this one!
ZonalHistogram.csv                 ← Example output from analysis
.venv/                             ← Virtual environment directory
```

**For new users: Use `uv sync` for automatic setup, or use `zonal_histogram_complete.py` with manual configuration.**

## Quick Start

### Standalone Python Setup (Only Option)

1. **Install UV Package Manager** (fastest Python environment manager):
   ```powershell
   # Install UV using pip
   pip install uv
   
   # Or download from: https://github.com/astral-sh/uv
   ```

2. **Set up project with automatic dependency management**:
   ```powershell
   # Clone or download this repository, then:
   cd path\to\ZH
   
   # Create virtual environment and install all dependencies automatically
   uv sync
   
   # Activate the environment (on Windows)
   .venv\Scripts\activate
   ```

   **Alternative (if you prefer requirements.txt)**:
   ```powershell
   # Manual approach
   uv venv
   .venv\Scripts\activate
   uv pip install -r requirements.txt
   ```

3. **Configure your file paths** in `zonal_histogram_complete.py`:
   ```python
   @classmethod
   def default(cls) -> HistogramConfig:
       return cls(
           raster_path=Path(r"path\to\your\classified_raster.tif"),
           zones_path=Path(r"path\to\your\zones_layer.shp"), 
           output_csv=Path("zonal_histogram_results.csv"),
           rat_field="class_description"  # or your attribute field name
       )
   ```
   
   ⚠️ **VERIFY**: Both files must use the SAME coordinate system!

4. **Run the analysis**:
   ```powershell
   python zonal_histogram_complete.py
   ```

   **Or use the installed command** (if using pyproject.toml):
   ```powershell
   zonal-histogram
   ```

## Modern Project Setup Benefits

This project now includes both traditional (`requirements.txt`) and modern (`pyproject.toml`) dependency management:

### **Using `pyproject.toml` (Recommended)**:
- ✅ **Automatic setup**: `uv sync` handles everything
- ✅ **Version locking**: Ensures reproducible environments
- ✅ **Project metadata**: Proper package information
- ✅ **Command installation**: `zonal-histogram` command available after setup

### **Using `requirements.txt` (Compatible)**:
- ✅ **Familiar approach**: Traditional pip-style installation
- ✅ **Broad compatibility**: Works with any Python environment manager
- ✅ **Simple**: Just `uv pip install -r requirements.txt`

## File Requirements

### Input Files

1. **Classified Raster** (GeoTIFF format):
   - Must have a raster attribute table (VAT)
   - VAT should be in `.tif.vat.dbf` format
   - Contains classified data (land cover, habitat types, etc.)

2. **Zone Polygons** (Shapefile):
   - Polygon features defining analysis areas
   - Must use the SAME coordinate system as the raster (critical!)
   - Should have a unique identifier field

### Output

- **CSV file** with zonal histogram results
- Format: `[Zone_ID, Class1_Count, Class2_Count, ..., Total_Pixels]`
- One row per zone, one column per habitat class

## Coordinate System Requirements

⚠️ **ABSOLUTELY CRITICAL**: The raster and zones must use IDENTICAL coordinate systems.

This is the most important requirement for accurate results. **Mismatched coordinate systems will produce incorrect pixel counts and spatial errors.**

### Checking Coordinate Systems

Before running the analysis, verify both files use the same CRS:

```python
import geopandas as gpd
import rasterio

# Check raster CRS
with rasterio.open("your_raster.tif") as src:
    print(f"Raster CRS: {src.crs}")

# Check zones CRS  
zones = gpd.read_file("your_zones.shp")
print(f"Zones CRS: {zones.crs}")
```

### CRS Mismatch Warning

The script will display prominent warnings if coordinate systems don't match:
```
================================================================================
COORDINATE SYSTEM MISMATCH DETECTED!
Zone CRS: EPSG:4326 (WGS84)
Raster CRS: EPSG:3005 (NAD83 / BC Albers)
================================================================================
REPROJECTING zones to match raster coordinate system...
For best results, ensure both inputs use the same CRS from the start.
Automatic reprojection may introduce minor inaccuracies.
================================================================================
```

### Fixing CRS Mismatches

**Best Practice**: Reproject your data to match BEFORE running the analysis:

```python
# Option 1: Reproject zones to match raster
zones = gpd.read_file("zones.shp")
zones_reprojected = zones.to_crs("EPSG:3005")  # Match your raster CRS
zones_reprojected.to_file("zones_reprojected.shp")

# Option 2: Use GDAL to reproject raster
# gdalwarp -t_srs EPSG:4326 input.tif output_reprojected.tif
```

## Configuration Options

### HistogramConfig Class

The `HistogramConfig` class centralizes all analysis parameters:

```python
config = HistogramConfig(
    raster_path=Path("habitat_classified.tif"),
    zones_path=Path("buffer_zones.shp"),
    output_csv=Path("results.csv"),
    rat_field="cls_desc_2",        # Attribute table field with class names
    zones_field=None,              # Zone ID field (None = auto-detect)
    overwrite=False                # Overwrite existing output files
)
```

### Command Line Usage

```powershell
# Interactive mode (prompts for missing inputs)
python zonal_histogram_complete.py

# Specify all parameters
python zonal_histogram_complete.py \
    --raster habitat_map.tif \
    --zones study_areas.shp \
    --output results.csv \
    --rat-field habitat_type

# Get help
python zonal_histogram_complete.py --help
```

## How It Works

### Individual Zone Processing

Unlike bulk processing methods, this tool processes each zone separately:

1. **For each zone polygon**:
   - Create a temporary raster mask for just that zone
   - Extract raster pixels falling within the zone boundary
   - Count pixels for each habitat class
   - Record results with complete class coverage

2. **Benefits of individual processing**:
   - Prevents overlap artifacts between adjacent zones
   - Ensures accurate pixel counts that sum correctly
   - Avoids the overlap errors found in some GIS tools (including ArcGIS Pro's Zonal Histogram)
   - Guarantees complete class coverage for each zone

### Complete Class Coverage

The tool loads ALL possible classes from the raster attribute table:

```python
# Example output includes ALL classes, even with 0 pixels
Zone_ID, Coniferous_Forest, Deciduous_Forest, Grassland, Water, Urban, ...
Buffer_1, 1250, 0, 340, 0, 15, ...
Buffer_2, 890, 125, 0, 45, 0, ...
```

This ensures:
- Consistent column structure across all zones
- Easy data analysis and visualization
- No missing data issues from classes not present in specific zones

## Performance Notes

- **Processing Time**: ~30-60 seconds for 197 zones with a 30m resolution raster
- **Memory Usage**: Moderate (loads one zone at a time)
- **Accuracy**: Pixel counts are exact and match ArcGIS Pro results

## Testing

This project includes a comprehensive test suite to ensure reliability and accuracy.

### **Running Tests**

```powershell
# Install ALL dependencies including development/testing tools
uv sync --all-extras --dev

# Or install dev dependencies separately if already synced
uv pip install pytest pytest-cov pytest-mock black ruff coverage

# Run all tests
python run_tests.py

# Run specific test types
python run_tests.py --unit              # Unit tests only
python run_tests.py --integration       # Integration tests only
python run_tests.py --coverage          # Tests with coverage report

# Manual pytest commands (after installing pytest)
python -m pytest                        # All tests
python -m pytest -v                     # Verbose output
python -m pytest -m unit               # Unit tests only
python -m pytest --cov=. --cov-report=html  # Coverage report
```

### **Test Structure**

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test complete workflows
- **Geospatial Tests**: Test coordinate system handling and data processing
- **Mock Tests**: Test with simulated data to avoid large file dependencies

### **Test Coverage Areas**

- ✅ Configuration management (`HistogramConfig`)
- ✅ File I/O operations (raster/shapefile loading)
- ✅ Coordinate system validation and reprojection
- ✅ Individual zone processing algorithm
- ✅ **Overlapping zone handling** (key advantage validation)
- ✅ Raster attribute table handling
- ✅ Output generation and formatting
- ✅ Error handling and edge cases

### **Current Test Status**

🎉 **All tests passing!** The test suite includes:
- **12 total tests** (11 unit + 1 integration)
- **Overlapping zone validation**: Tests specifically verify correct handling of overlapping buffer zones
- **60% code coverage** with detailed HTML reports
- **Clean code quality** (100% Ruff linting + Black formatting compliance)
- **Comprehensive mocking** for geospatial operations
- **Cross-platform compatibility** with modern Python testing tools

## Troubleshooting

### Common Issues

1. **"Raster attribute table not found"**
   - Check for `.tif.vat.dbf` file alongside your raster
   - Ensure the VAT was properly created during classification

2. **"Coordinate system mismatch"**
   - Reproject either the raster or zones to match coordinate systems
   - Use `gdalwarp` or ArcGIS to reproject data

3. **"No valid zones found"**
   - Check that zone polygons overlap with the raster extent
   - Verify zone geometries are valid (no self-intersections)

4. **Import errors with GeoPandas/Rasterio**
   - Make sure you activated the virtual environment: `.venv\Scripts\activate`
   - Reinstall packages: `uv pip install --force-reinstall geopandas rasterio`

### Getting Help

1. **Check the log output** - the script provides detailed progress information
2. **Verify input files** - ensure both raster and zones are readable
3. **Test with a small subset** - try with a few zones first to verify setup

## Example Workflow

```python
from pathlib import Path
from zonal_histogram_complete import HistogramConfig, main

# Set up configuration
config = HistogramConfig(
    raster_path=Path("data/habitat_classified.tif"),
    zones_path=Path("data/buffer_zones.shp"),
    output_csv=Path("results/habitat_histogram.csv"),
    rat_field="cls_desc_2"
)

# Run analysis
main(config)

# Results will be saved to habitat_histogram.csv
```

## Dependencies

- **Python**: 3.8 or higher
- **Core packages**: `geopandas`, `rasterio`, `numpy`, `pandas`, `dbfread`
- **UV package manager**: For fast environment setup (recommended)

## Acknowledgments

These scripts were developed with assistance from AI tools, including Claude Sonnet 4, in conjunction with heavy lifting by our internal team. Special help from the AI was provided in unit testing framework development and test case design.

## License

This tool was developed for ecological and environmental analysis applications. Feel free to adapt for your specific needs.

## Support

For issues related to:
- **Environment setup**: Check UV documentation and package installation
- **Coordinate systems**: Verify CRS compatibility between inputs
- **Attribute tables**: Ensure proper VAT format and field names
- **Performance**: Consider processing smaller zone subsets for large datasets