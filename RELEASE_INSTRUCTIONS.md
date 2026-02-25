# Release Instructions for v1.0

## Step 1: Create GitHub Release (v1.0)

### Option A: Using GitHub Web Interface

1. **Go to your repository**: https://github.com/InternationalCraneFoundation/overlap-safe-zonal-histogram-analysis
2. **Click "Releases"** (on the right sidebar, or go to: https://github.com/InternationalCraneFoundation/overlap-safe-zonal-histogram-analysis/releases)
3. **Click "Create a new release"**
4. **Fill in the release details**:
   - **Tag version**: `v1.0.0` (or `v1.0`)
   - **Release title**: `v1.0.0 - Initial Release`
   - **Description** (use this template):
     ```markdown
     # Overlap-Safe Zonal Histogram Analysis Tool v1.0.0
     
     Initial stable release of the overlap-safe zonal histogram analysis tool.
     
     ## Key Features
     - Individual zone processing to handle overlapping buffers correctly
     - Complete class coverage from raster attribute tables
     - Robust coordinate system validation
     - Standalone Python implementation (no ArcGIS license required)
     
     ## What's Included
     - Main tool: `zonal_histogram_complete.py`
     - Comprehensive test suite
     - Documentation and examples
     - CITATION.cff for proper academic citation
     
     ## Requirements
     - Python 3.8+
     - geopandas, rasterio, numpy, pandas, dbfread
     
     ## Citation
     If you use this software in your research, please cite it using the CITATION.cff file or:
     
     Moore, D., & International Crane Foundation. (2025). Overlap-Safe Zonal Histogram Analysis Tool (Version 1.0.0) [Computer software]. GitHub. https://github.com/InternationalCraneFoundation/overlap-safe-zonal-histogram-analysis
     ```
   - **Target**: Select `main` branch
5. **Click "Publish release"**

### Option B: Using Git Command Line

```bash
# Make sure you're on the main branch and everything is committed
git checkout main
git pull origin main

# Create and push the tag
git tag -a v1.0.0 -m "Release version 1.0.0 - Initial stable release"
git push origin v1.0.0

# Then go to GitHub web interface to create the release with description
```

---

## Step 2: Archive to Zenodo (Get a DOI)

### Prerequisites
- GitHub account connected to Zenodo
- Repository is public (or you have Zenodo access)

### Instructions

1. **Go to Zenodo**: https://zenodo.org/
2. **Sign in** with your GitHub account
3. **Go to GitHub Integration**: 
   - Click your profile → Settings → GitHub
   - Or go directly to: https://zenodo.org/account/settings/github/
4. **Enable Zenodo for your repository**:
   - Find `InternationalCraneFoundation/overlap-safe-zonal-histogram-analysis`
   - Toggle it **ON** to enable
   - This will automatically archive releases to Zenodo

5. **Create the Release** (if you haven't already):
   - Go back to GitHub and create the v1.0.0 release (see Step 1)
   - Zenodo will automatically detect the new release

6. **Wait for Zenodo Processing** (usually 5-10 minutes):
   - Zenodo will create a DOI automatically
   - You'll receive an email when it's ready
   - The DOI will be in format: `10.5281/zenodo.XXXXXXX`

7. **Update CITATION.cff with the DOI**:
   - Once you have the Zenodo DOI, update the `CITATION.cff` file
   - Replace `10.5281/zenodo.XXXXXXX` with your actual DOI
   - Commit and push the update

### Alternative: Manual Upload to Zenodo

If automatic integration doesn't work:

1. **Go to Zenodo**: https://zenodo.org/deposit/new
2. **Select "Upload"**
3. **Fill in metadata**:
   - **Upload type**: Software
   - **Title**: Overlap-Safe Zonal Histogram Analysis Tool
   - **Version**: 1.0.0
   - **Publication date**: October 2025
   - **Creators**: 
     - Moore, Dorn (International Crane Foundation)
   - **Description**: Use the abstract from README.md
   - **Keywords**: zonal histogram, raster analysis, geospatial analysis, overlapping buffers, habitat analysis, Python, GIS
   - **License**: Specify your license (if you have one)
   - **Related identifiers**: 
     - GitHub: https://github.com/InternationalCraneFoundation/overlap-safe-zonal-histogram-analysis
4. **Upload files**: Download the repository as ZIP from GitHub releases
5. **Publish** to get your DOI

---

## Step 3: Update CITATION.cff with Zenodo DOI

After you receive the Zenodo DOI:

1. **Edit `CITATION.cff`**:
   - Replace `doi: 10.5281/zenodo.XXXXXXX` with your actual DOI
   - Update `date-released` if needed

2. **Commit and push**:
   ```bash
   git add CITATION.cff
   git commit -m "Update CITATION.cff with Zenodo DOI"
   git push origin main
   ```

---

## Step 4: Verify Everything

- [ ] GitHub release v1.0.0 is published
- [ ] Zenodo archive is created with DOI
- [ ] CITATION.cff is updated with correct DOI
- [ ] Release description includes citation information
- [ ] README.md mentions the release (optional but recommended)

---

## Citation Formats (After Zenodo DOI is Available)

### APA Style
```
Moore, D., & International Crane Foundation. (2025). Overlap-Safe Zonal Histogram Analysis Tool (Version 1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX
```

### BibTeX
```bibtex
@software{moore_zonal_histogram_2025,
  author = {Moore, Dorn and {International Crane Foundation}},
  title = {Overlap-Safe Zonal Histogram Analysis Tool},
  version = {1.0.0},
  month = oct,
  year = {2025},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.XXXXXXX},
  url = {https://github.com/InternationalCraneFoundation/overlap-safe-zonal-histogram-analysis}
}
```

### RIS Format
```
TY  - COMP
TI  - Overlap-Safe Zonal Histogram Analysis Tool
AU  - Moore, Dorn
AU  - International Crane Foundation
PY  - 2025/10/01
VL  - 1.0.0
DO  - 10.5281/zenodo.XXXXXXX
UR  - https://github.com/InternationalCraneFoundation/overlap-safe-zonal-histogram-analysis
ER  -
```
