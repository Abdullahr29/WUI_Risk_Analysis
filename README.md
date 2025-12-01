# WUI Risk Analysis

Pipeline to analyse wildfire risk at the wildland–urban interface (WUI) using high-resolution satellite imagery. The project combines automated segmentation, polygon extraction, fuel classification, and damage prediction workflows to produce geospatial layers that can be mapped, visualized, and exported for downstream risk assessment. 

This is still a Work In Progress.

**Automatic polygon extraction:** Tiles each scene and runs the SAM2 mask generator to create cleaned building and land-cover polygons that can be exported for classification, mapping or further analysis.

## Repository layout
- `Data/` – Source CSVs and preprocessed imagery references (e.g., `xBD_WUI_Analysis.csv`, `maxar_data.pkl`).
- `Figures/` – Place generated plots, maps, and illustrative figures here.
- `Model/` – Core modeling code, including the `PolygonExtractor` for tiling, segmentation, and polygon post-processing.
- `Notebooks/` – End-to-end workflows for data preprocessing, segmentation, classification, and model testing.
- `Temporary_Files/` – Scratch space for intermediate outputs and cached artifacts.
- `Utils/` – Shared utilities for IO, dataset access, visualization, and class/label definitions.

## Getting started
1. **Create the Conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate satellite_env
   ```
2. **(Optional) Enable GPU acceleration**
   If you have a CUDA-capable GPU, install PyTorch with CUDA after activating the environment:
   ```bash
   pip3 install torch torchvision torchaudio
   ```
3. **Launch notebooks**
   ```bash
   jupyter notebook
   ```
   Open the notebooks in `Notebooks/` to run preprocessing, segmentation, and classification pipelines end-to-end.

## Data and configuration
- Scene metadata is read from `Data/xBD_WUI_Analysis.csv`.
- If running on remote storage, adjust paths in `config.toml` to point to your data root (see `Utils/image_utils.py` for path resolution logic). 

## Initial Results
- Segmentation of xBD imagery
  ![Segmentation workflow](Figures/xBD_Analysis.png)
  
- Classification of xBD imagery
  ![Example pre/post imagery with extracted polygons](Figures/xBD_Classification.png)
  
