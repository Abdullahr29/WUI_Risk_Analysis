# WUI Risk Analysis

Analyze wildfire risk at the wildland–urban interface (WUI) using high-resolution satellite imagery. The project combines automated segmentation, polygon extraction, and damage classification workflows to produce geospatial layers that can be mapped, visualized, and exported for downstream risk assessment.

## Highlights
- **Automatic polygon extraction:** Tiles each scene and runs the SAM2 mask generator to create cleaned building and land-cover polygons that can be exported for mapping or further analysis. 【F:Model/PolygonExtraction.py†L73-L160】
- **Data loading and labeling helpers:** Utility functions load xBD scenes, normalize imagery, and expose consistent class labels for vegetation, structures, and damage states. 【F:Utils/image_utils.py†L23-L160】
- **Notebook-driven experiments:** Jupyter notebooks cover preprocessing, segmentation, classification, shadow handling, and SAM experimentation for rapid iteration on new datasets.

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

## Typical workflows
- **Segment and extract polygons**
  - Load a scene with `DataLoader` (see `Utils/image_utils.py`) and run `PolygonExtractor.run_SAM_on_image()` to tile the scene, generate masks with SAM2, and merge them into cleaned polygons. 【F:Model/PolygonExtraction.py†L73-L160】
- **Explore and label data**
  - Use the helper methods in `DataLoader` to fetch specific scenes, inspect TIFFs, and visualize class overlays using the predefined class map. 【F:Utils/image_utils.py†L42-L160】
- **Prototype in notebooks**
  - Start with `Notebooks/data_preprocessing.ipynb` or `Notebooks/data_segmentation.ipynb` to reproduce the segmentation pipeline, then iterate on classification or shadow-removal workflows with the companion notebooks.

## Data and configuration
- Scene metadata is read from `Data/xBD_WUI_Analysis.csv`; update this file when adding new scenes or labels.
- Intermediate outputs and cached tiles should be written to `Temporary_Files/` to keep source data clean.
- If running on remote storage, adjust paths in `config.toml` to point to your data root (see `Utils/image_utils.py` for path resolution logic). 【F:Utils/image_utils.py†L23-L160】

## Visual assets
- Add overview diagrams, model architecture sketches, or workflow summaries to `Figures/`. Reference them directly in this README, for example:
  ```markdown
  ![Segmentation workflow](Figures/segmentation_workflow.png)
  ```
- Include before/after examples of segmentation or classification outputs using inline images:
  ```markdown
  ![Example pre/post imagery with extracted polygons](Figures/example_segmentation.png)
  ```

Feel free to extend the documentation with dataset-specific notes, evaluation metrics, or deployment details as the project evolves.
