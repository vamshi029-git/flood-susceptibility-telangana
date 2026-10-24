# Data files

Download these from **Google Drive → flood_project** and put them here:

| File(s) | Put in | Needed by |
|---|---|---|
| `flood_dataset_all_events.csv` | `data/` | 01, 02 |
| `godavari_2022_flood_map.tif`, `munneru_2024_flood_map.tif`, `budameru_2024_flood_map.tif` | `data/flood_maps/` | 03 (section 1) |
| `*_flood_points.shp` + `.dbf` `.shx` `.prj` (all 4 files per event) | `data/flood_maps/` | 03 (section 1) |
| `hyderabad_factor_stack.tif`, `warangal_factor_stack.tif` | `data/stacks/` | 03 (sections 2–3) |
| `*_flood_probability.tif`, `*_susceptibility_classes.tif` (from GEE) | `data/maps/` | optional, for QGIS figures |

The factor stacks are created by `notebooks/00_gee_data_prep.ipynb` (section 3).
