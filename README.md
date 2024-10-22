
# Aboveground Biomass Density (AGBD) Data Processing and Feature Extraction

This repository contains several Python notebooks that were used for processing and extracting data for developing and training Deep learning models to estimate AGBD, such as PFT (Plant Functional Types), GEDI simulated waveforms, and more. Each script handles specific tasks such as clipping, normalization, and feature extraction from NEON and simulated GEDI datasets.

## Table of Contents

| File Name | Type | Description | Important Parameters | Input | Output |
|-----------|------|-------------|----------------------|-------|--------|
| [Clip-DTM-NEON.ipynb](./Clip-DTM-NEON.ipynb) | Jupyter Notebook | Clips NEON DTM (Digital Terrain Model) data to a specified boundary or extent for spatial analysis. | - `input_dtm`: Path to the NEON DTM data.<br> - `clip_extent`: Bounding box or shapefile for clipping.<br> - `output_clipped_dtm`: Path to save the clipped DTM. | NEON DTM raster files. | Clipped DTM data based on the specified boundary. |
| [Remove-Blackborders-fromClipped-NEON.ipynb](./Remove-Blackborders-fromClipped-NEON.ipynb) | Jupyter Notebook | Processes NEON clipped images by removing black borders resulting from clipping operations. | - `input_image_path`: Path to clipped NEON image.<br> - `output_image_path`: Path to save the processed image.<br> - `border_threshold`: Threshold to detect and remove borders. | Clipped NEON images with black borders. | Cleaned NEON images without black borders, ready for further processing. |
| [Normalize-withDTM-NEON.ipynb](./Normalize-withDTM-NEON.ipynb) | Jupyter Notebook | Normalizes NEON LiDAR data using the corresponding DTM data to correct for elevation differences. | - `input_lidar`: Path to the NEON LiDAR data.<br> - `input_dtm`: Path to the NEON DTM data.<br> - `output_normalized_lidar`: Path to save the normalized LiDAR data. | NEON LiDAR data and DTM data in raster format. | Normalized LiDAR data where the elevation has been adjusted using DTM data. |
| [RH-Extractor.ipynb](./RH-Extractor.ipynb) | Jupyter Notebook | Extracts Relative Heights (RH) metrics from simulated GEDI waveform data (CSV) created using NEON Discrete return LiDAR point cloud. These matrices can later be used to train OLS and Random Forest machine learning models. | - `input_lidar_path`: Path to the LiDAR data.<br> - `rh_percentiles`: List of percentiles for RH extraction.<br> - `output_rh_file`: Path to save RH metrics. | LiDAR waveform data in CSV format. | CSV file with extracted RH metrics (e.g., RH10, RH20, RH50, etc.). |
| [PFT-Extractor.ipynb](./PFT-Extractor.ipynb) | Jupyter Notebook | Extracts Plant Functional Types (PFT) from NEON datasets and aggregates data based on specific vegetation indices. | - `input_path`: Path to NEON data.<br> - `pft_columns`: List of columns representing PFT types.<br> - `output_path`: Path to save the aggregated data. | NEON data in CSV format with vegetation indices and PFT data. | A CSV file containing aggregated PFT data for different vegetation types. |
| [Simulated-GEDI-Waveform-Visualizer.ipynb](./Simulated-GEDI-Waveform-Visualizer.ipynb) | Jupyter Notebook | Visualizes simulated GEDI waveforms, showing waveform structures and analyzing cumulative return energy. | - `input_simulated_data`: Path to simulated GEDI waveform data (CSV).<br> - `waveform_visualization_path`: Path to save waveform plots.<br> - `plot_params`: Parameters for adjusting plot appearance (e.g., colors, scales). | Simulated GEDI waveform data (CSV). | Visualized waveforms saved as image files |

