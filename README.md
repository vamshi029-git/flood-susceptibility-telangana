# Flood Susceptibility Mapping — Telangana & Andhra Pradesh

Multi-event, explainable machine learning for flood susceptibility mapping using Sentinel-1 SAR,
trained on three floods (Godavari 2022, Munneru 2024, Budameru 2024) and applied to
Hyderabad, Warangal and Telangana.

Base paper: Tepetidis et al. (2025), *Combining Machine Learning Models and Satellite Data of an
Extreme Flood Event for Flood Susceptibility Mapping*, Water 17, 2678.

## Project structure

```
flood_project/
├── notebooks/
│   ├── 00_gee_data_prep.ipynb   Earth Engine (Python): flood mapping, dataset, factor stacks
│   ├── 01_flood_ml.ipynb        checks, Pearson, VIF, 7 models, random split,
│   │                            leave-one-event-out, no-elevation experiment, save models
│   ├── 02_shap.ipynb            SHAP explanations (summary, dependence, per event)
│   └── 03_maps.ipynb            flood maps, susceptibility maps in Python, rainfall scenarios
├── src/
│   ├── config.py                paths, events, feature lists, scenario, colours
│   ├── ml_utils.py              preprocessing, the 7 models, metrics, leave-one-event-out
│   └── gee_lib.py               Earth Engine functions (flood mapping, factors, exports)
├── gee_scripts/                 the same steps as browser (JavaScript) scripts
├── data/                        downloaded inputs (see data/README.md)
├── outputs/                     figures, tables, saved models (created by the notebooks)
└── requirements.txt
```

## Setup (Mac, once)

1. Install **Python 3.11 or 3.12** (python.org) and **VS Code** with the **Python** and **Jupyter** extensions.
2. Open this folder in VS Code (**File → Open Folder**), then open the terminal (**Ctrl + `**):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If XGBoost or LightGBM complains about `libomp`: `brew install libomp`.

3. Open a notebook and choose **Select Kernel → .venv** (top right).

## Run order

| Step | Notebook | Needs |
|---|---|---|
| (optional) | `00_gee_data_prep.ipynb` | Earth Engine account; recreates the data in Google Drive |
| 1 | `01_flood_ml.ipynb` | `data/flood_dataset_all_events.csv` |
| 2 | `02_shap.ipynb` | models saved by notebook 01 |
| 3 | `03_maps.ipynb` | flood maps / points / factor stacks in `data/` |

Click **Run All** at the top of each notebook. Results go to `outputs/`.

## Method summary

| Step | Details |
|---|---|
| Flood inventory | Sentinel-1 VV change detection, same orbit pre/post, 50 m median filter, Otsu with −18 dB fallback, −3 dB change, JRC permanent water / slope > 5° / HAND > 20 m masks |
| Samples | 1,500 flood + 1,500 non-flood points per event (non-flood ≥ 500 m from flood) → 8,709 with complete factors |
| Factors | elevation, slope, aspect (sin/cos), curvature, TWI, SPI, HAND, distance to river, drainage density, land cover, NDVI, soil texture, event rainfall (IMERG, 5 days), antecedent rainfall (CHIRPS, 30 days) |
| Models | LR, SVM, RF, XGBoost (base-paper settings), LightGBM, CatBoost, Stacking |
| Validation | random 80/20 split and leave-one-event-out |
| Final model | Random Forest without absolute elevation (transfers to the Deccan plateau) |
| Explainability | SHAP |
