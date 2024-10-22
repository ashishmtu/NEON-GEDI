# Aboveground Biomass Density Prediction Models

This repository contains various models designed to predict Aboveground Biomass density using Simulated GEDI Waveform data. The models include OLS, 1D-CNN, CNN, LSTM, and Random Forest approaches. Each model has been implemented in Python using PyTorch or other relevant libraries, with detailed scripts for training and evaluation. For detailed research refer to our research paper (under review) - **Aboveground Biomass Density Estimation Using Deep Learning: Insight from NEON Ground-Truth Data and Simulated GEDI Waveform**

## Table of Contents

| File Name | Type | Description |
|-----------|------|-------------|
| [1D-CNN-Model.py](https://github.com/ashishmtu/NEON-GEDI/blob/Models/1D-CNN-Model.py) | Python Script | Implements a 1D Convolutional Neural Network (CNN) to predict live biomass density using waveform and elevation data. |
| [CNN-Model.py](https://github.com/ashishmtu/NEON-GEDI/blob/Models/CNN-Model.py) | Python Script | A convolutional neural network that processes 2D images generated from waveform data. The model uses various pretrained architectures from the `timm` library (e.g., ResNet, eva, beit) and fine-tunes them for regression tasks to predict biomass density. |
| [LSTM-Model.py](https://github.com/ashishmtu/NEON-GEDI/blob/Models/LSTM-Model.py) | Python Script | A Long Short-Term Memory (LSTM) model designed for capturing sequential dependencies in waveform data for biomass prediction. |
| [OLS-Model.ipynb](https://github.com/ashishmtu/NEON-GEDI/blob/Models/OLS-Model.ipynb) | Jupyter Notebook | Implements an Ordinary Least Squares (OLS) regression model to predict biomass density for comparison with deep learning models. |
| [RF-Model.ipynb](https://github.com/ashishmtu/NEON-GEDI/blob/Models/RF-Model.ipynb) | Jupyter Notebook | Random Forest-based model for biomass density prediction. |

## Models Overview

## Data Preparation

The waveform data should be stored in CSV files, with each file containing a site's waveform and elevation data. A metadata CSV file contains labels (biomass density) for each site. Place your waveform CSV files in a directory and provide the paths to the training and testing metadata files when running the scripts.

- **1D CNN & LSTM:** The model expects 2-channel time-series data (waveform and elevation) for each site, stored in CSV format. Each CSV should contain columns for `Rxwaveform` (waveform data) and `Elevation` (elevation data). The metadata file should contain the `Live_Biomass_Density` label for each site.

- **CNN:** This model requires images generated from the waveform data, which should be stored as image files (e.g., `.png`). A CSV metadata file should link each image with its corresponding `Live_Biomass_Density` label. 

- **OLS & Random Forest Model:** These notebooks expect the waveform data to be preprocessed into a suitable feature set (e.g., waveform-derived metrics). The metadata file should contain the corresponding labels for each site to be used in the regression analysis.

### Training and Evaluation
Each model script can be run from the command line, specifying the required arguments such as the dataset paths, number of epochs, and batch size.

#### Example:
```bash
python 1D-CNN-Model.py --folder_path ./data/waveforms --train_metadata_csv ./data/train_metadata.csv --test_metadata_csv ./data/test_metadata.csv --epochs 100 --batch_size 16

```
### OLS Model

**File:** `OLS-Model.ipynb`  
**Description:** This notebook applies an Ordinary Least Squares (OLS) regression model to the dataset. It provides a basic statistical approach for comparison against more complex models like neural networks.

### Random Forest Model

**File:** `RF-Model.ipynb`  
**Description:** This notebook implements a Random Forest model to predict biomass density using waveform features.

### 1D CNN

**File:** `1D-CNN-Model.py`  
**Description:** This model applies 1D convolutions to waveform and elevation data to predict live biomass density. The architecture includes four convolutional layers, followed by max-pooling and fully connected layers.

**Important Parameters:**
- **Input:** 2-channel (waveform and elevation) time-series data.
- **Normalization:** Waveform and elevation data are normalized based on the maximum values in the dataset.

### LSTM

**File:** `LSTM-Model.py`  
**Description:** The LSTM model is designed to capture sequential dependencies in waveform and elevation data for biomass density prediction. It uses a multi-layer LSTM followed by fully connected layers for regression.

**Important Parameters:**
- **Hidden Dimensions:** 128 hidden units in the LSTM layers.
- **Dropout:** 0.5 to prevent overfitting.

### CNN

**File:** `CNN-Model.py`  
**Description:** A convolutional neural network that processes 2D images generated from waveform data. The model uses pretrained architectures from `timm` and fine-tunes them for regression tasks.

**Important Parameters:**
- **Input Dimension:** Dynamically determined based on the chosen model.
- **Pretrained Models:** Uses models like `resnet50` for transfer learning.

## Results and Metrics

Each model's performance is evaluated using several metrics, including:
- **Mean Absolute Error (MAE):** Average error in the prediction.
- **Root Mean Squared Error (RMSE):** Standard deviation of prediction errors.
- **R-squared (R²):** Proportion of variance explained by the model.

Model performance metrics and visualizations (such as loss curves and prediction vs true value plots) are generated and saved as part of the training process.

