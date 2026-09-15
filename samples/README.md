# Sample Images

Drop your test satellite images here for use with `test_pipeline.py`.

## Expected formats
- **JPEG** (.jpg, .jpeg) — standard optical imagery
- **PNG** (.png) — standard optical imagery
- **TIFF/GeoTIFF** (.tif, .tiff) — satellite imagery with geospatial metadata

## Suggested test sets

### Single-image queries
- An optical satellite image (e.g., Sentinel-2 RGB composite)
- A SAR image (e.g., Sentinel-1 grayscale)

### Bi-temporal (change detection)
- Two optical images of the same area at different times
- Name them clearly, e.g., `area1_2023.jpg` and `area1_2024.jpg`

### Optical + SAR fusion
- One optical and one SAR image of the same area
- Same dimensions preferred, but the system will resize if needed

## Running tests

```bash
# Single image captioning
python test_pipeline.py --images samples/optical.jpg --query "Describe this satellite image"

# Change detection
python test_pipeline.py --images samples/area_before.jpg samples/area_after.jpg --query "What changed between these two images?"

# Optical + SAR fusion
python test_pipeline.py --images samples/optical.jpg samples/sar.tif --query "What can we learn from combining optical and SAR data?"
```

## Note
Unit tests (`pytest tests/ -v`) use synthetic images generated in-code and do
NOT require any files in this folder.
