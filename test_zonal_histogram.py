"""
Unit tests for the zonal histogram analysis tool.

This test suite covers the core functionality of the zonal histogram tool
including configuration management, data loading, coordinate system handling,
and the individual zone processing approach.
"""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
from shapely.geometry import Polygon
from rasterio.transform import from_bounds

# Import the modules to test
from zonal_histogram_complete import (
    HistogramConfig,
    _resolve_field_case,
    _ensure_output_path,
    load_zones,
    compute_zone_histograms_individual,
    attach_class_names,
    main,
)


@pytest.mark.unit
class TestHistogramConfig:
    """Test the configuration management class."""

    def test_config_creation(self):
        """Test basic configuration creation."""
        config = HistogramConfig(
            raster_path=Path("test.tif"),
            zones_path=Path("test.shp"),
            zone_field="id",
            rat_field="class",
            output_csv=Path("output.csv"),
        )

        assert config.raster_path == Path("test.tif")
        assert config.zones_path == Path("test.shp")
        assert config.zone_field == "id"
        assert config.rat_field == "class"
        assert config.output_csv == Path("output.csv")
        assert config.overwrite is True  # default value

    def test_default_config(self):
        """Test that default configuration returns valid paths."""
        config = HistogramConfig.default()

        assert isinstance(config.raster_path, Path)
        assert isinstance(config.zones_path, Path)
        assert isinstance(config.output_csv, Path)
        assert isinstance(config.zone_field, str)
        assert isinstance(config.rat_field, str)


@pytest.mark.unit
class TestUtilityFunctions:
    """Test utility and helper functions."""

    def test_resolve_field_case(self):
        """Test case-insensitive field name resolution."""
        available_fields = ["NAME", "value", "Class_ID", "OBJECTID"]

        # Exact match
        assert _resolve_field_case("NAME", available_fields) == "NAME"

        # Case insensitive match
        assert _resolve_field_case("name", available_fields) == "NAME"
        assert _resolve_field_case("VALUE", available_fields) == "value"
        assert _resolve_field_case("class_id", available_fields) == "Class_ID"

        # No match
        assert _resolve_field_case("nonexistent", available_fields) is None

    def test_ensure_output_path(self):
        """Test output path validation and directory creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "subdir" / "output.csv"

            # Should create directory and not raise error
            _ensure_output_path(output_path, overwrite=True)
            assert output_path.parent.exists()

            # Create the file
            output_path.write_text("test")

            # Should raise error when overwrite=False and file exists
            with pytest.raises(FileExistsError):
                _ensure_output_path(output_path, overwrite=False)

            # Should not raise error when overwrite=True
            _ensure_output_path(output_path, overwrite=True)


@pytest.mark.unit
class TestGeospatialDataHandling:
    """Test geospatial data loading and processing."""

    @pytest.fixture
    def sample_zones_gdf(self):
        """Create a sample zones GeoDataFrame for testing."""
        polygons = [
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),
        ]

        gdf = gpd.GeoDataFrame(
            {
                "zone_id": ["Zone_1", "Zone_2", "Zone_3"],
                "area": [1.0, 1.0, 1.0],
                "geometry": polygons,
            },
            crs="EPSG:4326",
        )

        return gdf

    @pytest.fixture
    def sample_raster_data(self):
        """Create sample raster data for testing."""
        # Create a 4x4 raster with different class values
        data = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]], dtype=np.int16)

        return np.ma.masked_array(data)

    def test_load_zones_basic_functionality(self, sample_zones_gdf):
        """Test basic zone loading functionality."""
        with patch("geopandas.read_file") as mock_read:
            mock_read.return_value = sample_zones_gdf

            # Mock target CRS (same as zones)
            target_crs = sample_zones_gdf.crs

            zones, resolved_field, zone_lookup = load_zones(Path("test.shp"), "zone_id", target_crs)

            assert len(zones) == 3
            assert resolved_field == "zone_id"
            assert len(zone_lookup) == 3
            assert "_zone_id" in zones.columns
            assert all(zones["_zone_id"] == [1, 2, 3])

    def test_load_zones_field_case_insensitive(self, sample_zones_gdf):
        """Test that zone field loading is case insensitive."""
        with patch("geopandas.read_file") as mock_read:
            mock_read.return_value = sample_zones_gdf
            target_crs = sample_zones_gdf.crs

            # Test with different case
            zones, resolved_field, zone_lookup = load_zones(Path("test.shp"), "ZONE_ID", target_crs)

            assert resolved_field == "zone_id"  # Should find the lowercase version

    def test_load_zones_crs_mismatch_warning(self, sample_zones_gdf):
        """Test that CRS mismatch generates appropriate warnings."""
        with patch("geopandas.read_file") as mock_read:
            mock_read.return_value = sample_zones_gdf

            # Different target CRS
            target_crs = "EPSG:3857"  # Web Mercator instead of WGS84

            with patch("zonal_histogram_complete.LOGGER") as mock_logger:
                zones, resolved_field, zone_lookup = load_zones(
                    Path("test.shp"), "zone_id", target_crs
                )

                # Should log warnings about CRS mismatch
                assert any(
                    "COORDINATE SYSTEM MISMATCH" in str(call)
                    for call in mock_logger.warning.call_args_list
                )

                # The returned zones should have the target CRS
                assert zones.crs == target_crs


@pytest.mark.unit
class TestZonalAnalysis:
    """Test the core zonal analysis functionality."""

    def test_individual_zone_processing(self):
        """Test that individual zone processing works correctly."""
        # Mock data setup
        raster_data = np.ma.masked_array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])

        # Create mock zones GeoDataFrame with OVERLAPPING zones
        # This tests the core advantage: handling overlapping zones correctly
        zones_data = {
            "_zone_id": [1, 2],
            "geometry": [
                Polygon(
                    [(0, 0), (4, 0), (4, 3), (0, 3)]
                ),  # Zone A: covers 3/4 of raster (overlaps with Zone B)
                Polygon(
                    [(0, 1), (4, 1), (4, 4), (0, 4)]
                ),  # Zone B: covers 3/4 of raster (overlaps with Zone A)
            ],
        }
        zones = gpd.GeoDataFrame(zones_data, crs="EPSG:4326")

        # Mock dataset
        mock_dataset = Mock()
        mock_dataset.height = 4
        mock_dataset.width = 4
        mock_dataset.transform = from_bounds(0, 0, 4, 4, 4, 4)

        zone_lookup = {1: "Zone_A", 2: "Zone_B"}
        all_class_values = [1, 2, 3, 4]

        with patch("rasterio.features.rasterize") as mock_rasterize:
            # Mock rasterize to return different masks for each zone
            def side_effect(shapes, **kwargs):
                geom = list(shapes)[0][0]
                bounds = geom.bounds
                # Zone A (y=0-3): gets classes 1,2,3 (top 3 rows)
                if bounds[3] <= 3:  # Zone A (extends to y=3)
                    return np.array(
                        [
                            [1, 1, 1, 1],  # Row 0: classes 1,2
                            [1, 1, 1, 1],  # Row 1: classes 1,2
                            [1, 1, 1, 1],  # Row 2: classes 3,4
                            [0, 0, 0, 0],  # Row 3: not in this zone
                        ]
                    )
                else:  # Zone B (y=1-4): gets classes 1,2,3,4 (bottom 3 rows)
                    return np.array(
                        [
                            [0, 0, 0, 0],  # Row 0: not in this zone
                            [1, 1, 1, 1],  # Row 1: classes 1,2
                            [1, 1, 1, 1],  # Row 2: classes 3,4
                            [1, 1, 1, 1],  # Row 3: classes 3,4
                        ]
                    )

            mock_rasterize.side_effect = side_effect

            result_df = compute_zone_histograms_individual(
                raster_data, zones, mock_dataset, zone_lookup, all_class_values
            )

            assert len(result_df) == 2
            assert list(result_df.columns) == ["zone", "1", "2", "3", "4"]

            # Check that each zone gets appropriate class counts
            zone_a_row = result_df[result_df["zone"] == "Zone_A"].iloc[0]
            zone_b_row = result_df[result_df["zone"] == "Zone_B"].iloc[0]

            # Zone A should have classes 1,2,3 (covers top 3 rows)
            # Zone B should have classes 1,2,3,4 (covers bottom 3 rows)
            # This tests overlapping coverage - both zones include row 1 and row 2
            assert zone_a_row["1"] > 0 or zone_a_row["2"] > 0  # Zone A has classes 1,2
            assert zone_a_row["3"] > 0  # Zone A also has class 3 from row 2
            assert zone_b_row["3"] > 0 or zone_b_row["4"] > 0  # Zone B has classes 3,4
            assert (
                zone_b_row["1"] > 0 or zone_b_row["2"] > 0
            )  # Zone B also has classes 1,2 from row 1

    def test_overlapping_zones_processing(self):
        """Test that overlapping zones are processed correctly without artifacts.

        This test specifically validates the core advantage of individual zone processing:
        handling overlapping buffer zones that would cause issues in bulk processing methods.
        """
        # Create overlapping circular buffer zones (common in ecological analysis)
        raster_data = np.ma.masked_array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])

        # Create significantly overlapping zones
        zones_data = {
            "_zone_id": [1, 2, 3],
            "geometry": [
                Polygon([(0, 0), (3, 0), (3, 3), (0, 3)]),  # Large zone covering most of raster
                Polygon([(1, 1), (4, 1), (4, 4), (1, 4)]),  # Overlapping zone (offset)
                Polygon(
                    [(1.5, 0.5), (3.5, 0.5), (3.5, 2.5), (1.5, 2.5)]
                ),  # Smaller overlapping zone
            ],
        }
        zones = gpd.GeoDataFrame(zones_data, crs="EPSG:4326")

        mock_dataset = Mock()
        mock_dataset.height = 4
        mock_dataset.width = 4
        mock_dataset.transform = from_bounds(0, 0, 4, 4, 4, 4)

        zone_lookup = {1: "Buffer_100m", 2: "Buffer_200m", 3: "Buffer_150m"}
        all_class_values = [1, 2, 3, 4]

        with patch("rasterio.features.rasterize") as mock_rasterize:

            def side_effect(shapes, **kwargs):
                # Simulate different masks for each overlapping zone
                geom = list(shapes)[0][0]
                bounds = geom.bounds

                if bounds[0] == 0:  # First zone (starts at x=0)
                    return np.array([[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]])
                elif bounds[0] == 1:  # Second zone (starts at x=1)
                    return np.array([[0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1]])
                else:  # Third zone (starts at x=1.5)
                    return np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0]])

            mock_rasterize.side_effect = side_effect

            result_df = compute_zone_histograms_individual(
                raster_data, zones, mock_dataset, zone_lookup, all_class_values
            )

            # Verify results
            assert len(result_df) == 3
            assert list(result_df.columns) == ["zone", "1", "2", "3", "4"]

            # Each zone should have valid pixel counts (demonstrating individual processing works)
            for _, row in result_df.iterrows():
                total_pixels = row["1"] + row["2"] + row["3"] + row["4"]
                assert total_pixels > 0, f"Zone {row['zone']} should have pixels"

            # Verify zone names are preserved
            zone_names = set(result_df["zone"])
            assert zone_names == {"Buffer_100m", "Buffer_200m", "Buffer_150m"}

    def test_attach_class_names(self):
        """Test attaching descriptive class names to histogram results."""
        df = pd.DataFrame({"zone": ["Zone_1", "Zone_2"], "1": [10, 5], "2": [20, 15], "3": [0, 8]})

        class_values = [1, 2, 3]
        class_names = ["Forest", "Grassland", "Water"]

        result = attach_class_names(df, class_values, class_names)

        expected_columns = ["zone", "Forest", "Grassland", "Water"]
        assert list(result.columns) == expected_columns

        # Values should remain the same
        assert result.loc[0, "Forest"] == 10
        assert result.loc[0, "Grassland"] == 20
        assert result.loc[1, "Water"] == 8

    def test_attach_class_names_mismatch(self):
        """Test behavior when class names don't match class values."""
        df = pd.DataFrame({"zone": ["Zone_1"], "1": [10], "2": [20]})

        class_values = [1, 2]
        class_names = ["Forest"]  # Intentionally wrong length

        # Should return original dataframe when names don't match
        result = attach_class_names(df, class_values, class_names)
        assert list(result.columns) == ["zone", "1", "2"]


@pytest.mark.integration
class TestIntegration:
    """Integration tests for complete workflows."""

    @patch("zonal_histogram_complete.load_raster_dataset")
    @patch("zonal_histogram_complete.load_complete_class_metadata")
    @patch("zonal_histogram_complete.load_zones")
    @patch("zonal_histogram_complete.load_raster_band")
    @patch("zonal_histogram_complete.compute_zone_histograms_individual")
    @patch("zonal_histogram_complete.write_csv")
    def test_main_workflow(
        self,
        mock_write_csv,
        mock_compute,
        mock_load_band,
        mock_load_zones,
        mock_load_metadata,
        mock_load_dataset,
    ):
        """Test the complete main workflow integration."""

        # Setup mocks
        mock_dataset = Mock()
        mock_dataset.crs = "EPSG:4326"
        mock_load_dataset.return_value.__enter__.return_value = mock_dataset

        mock_load_metadata.return_value = ([1, 2, 3], ["Forest", "Grass", "Water"])
        mock_load_zones.return_value = (Mock(), "zone_id", {1: "A", 2: "B"})
        mock_load_band.return_value = np.ma.masked_array([[1, 2], [2, 3]])

        result_df = pd.DataFrame({"zone": ["A", "B"], "1": [5, 3], "2": [10, 7], "3": [0, 4]})
        mock_compute.return_value = result_df

        # Create test config
        with tempfile.TemporaryDirectory() as temp_dir:
            config = HistogramConfig(
                raster_path=Path("test.tif"),
                zones_path=Path("test.shp"),
                zone_field="zone_id",
                rat_field="class_name",
                output_csv=Path(temp_dir) / "output.csv",
            )

            # Run main function
            main(config)

            # Verify all components were called
            mock_load_dataset.assert_called_once()
            mock_load_metadata.assert_called_once()
            mock_load_zones.assert_called_once()
            mock_load_band.assert_called_once()
            mock_compute.assert_called_once()
            mock_write_csv.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
