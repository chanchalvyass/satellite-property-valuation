# 🛰️ Satellite Imagery Based Property Valuation (Multimodal ML)

This repository contains a **multimodal machine learning project** that predicts **property prices** by combining **tabular real-estate data** with **satellite imagery**. The project demonstrates how visual environmental context (roads, neighborhood density, greenery, infrastructure) can improve traditional price prediction models.

---

## 📌 Project Overview

Traditional property valuation models rely mainly on structured data such as square footage, number of bedrooms, and location coordinates. However, they often fail to capture **environmental and neighborhood-level visual cues**.

In this project, we build a **multimodal regression pipeline** that:
- Extracts **visual features from satellite images** using a CNN
- Combines them with **tabular features**
- Predicts property prices using **tree-based regression models**
- Provides **model explainability** using Grad-CAM

---

## 🎯 Objectives

- Build a **multimodal regression model** to predict property prices
- Programmatically acquire **satellite imagery** using latitude/longitude
- Perform **EDA and geospatial analysis** on tabular and visual data
- Extract **high-dimensional image embeddings** using CNNs
- Compare **tabular-only vs tabular + image fusion** models
- Ensure **model explainability** using Grad-CAM visualizations

---

## 🗂️ Repository Structure

```
├── data_fetcher.py              # Script to download satellite images using coordinates
├── preprocessing.ipynb          # Data cleaning, feature engineering, EDA
├── Satellite_images.ipynb       # CNN-based image feature extraction (EfficientNet)
├── Complete_Code.ipynb          # End-to-end pipeline (training, fusion, evaluation)
├── train(1).xlsx                # Training tabular dataset
├── test2.xlsx                   # Test tabular dataset
├── test_predictions.csv         # Tabular-only model predictions
├── test_predictions_img.csv     # Multimodal (tabular + image) predictions
├── .gitignore
└── README.md
```

---

## 🧩 Project Architecture

### Overall Multimodal Pipeline

```
┌───────────────────────────┐
│   Tabular Property Data   │
│ (size, rooms, location)  │
└──────────────┬────────────┘
               │
               ▼
      Feature Cleaning &
      Log Transformation
               │
               ▼
┌───────────────────────────┐
│  Tabular Feature Vector   │
└──────────────┬────────────┘
               │
               │        ┌─────────────────────────────┐
               │        │     Satellite Image (RGB)   │
               │        └──────────────┬──────────────┘
               │                       │
               │                       ▼
               │              CNN Feature Extractor
               │              (EfficientNet-B0)
               │                       │
               │                       ▼
               │              Image Embeddings (1280D)
               │                       │
               │                       ▼
               │              PCA Dimensionality Reduction
               │                       │
               │                       ▼
               │              Image Features (128D)
               │                       │
               └──────────────┬────────┘
                              ▼
                   Multimodal Feature Fusion
                   (Concatenation)
                              │
                              ▼
                     XGBoost Regressor
                              │
                              ▼
                   Predicted Property Price
```

### Explainability Flow (Grad-CAM)

```
Satellite Image
      │
      ▼
CNN Convolution Layers
      │
      ▼
Grad-CAM Heatmap
      │
      ▼
Highlighted Regions
(Roads, Buildings, Greenery)
```

---

## 🧠 Methodology

### 1. Tabular Modeling
- Features: size, rooms, floors, condition, grade, location, etc.
- Target: **log-transformed price**
- Models used:
  - Linear Regression
  - Random Forest Regressor
  - XGBoost Regressor

### 2. Satellite Image Feature Extraction
- Images downloaded using property latitude & longitude
- CNN backbone: **EfficientNet-B0 (ImageNet pretrained)**
- Output: **1280-dimensional image embeddings**
- Dimensionality reduction using **PCA (128 components, ~88% variance retained)**

### 3. Multimodal Fusion
- Late fusion by concatenating:
  - Tabular features (18)
  - PCA-reduced image features (128)
- Final regressor: **XGBoost**

---

## 📊 Results & Performance

| Model | Features Used | R² Score | RMSE (log scale) |
|------|--------------|----------|-----------------|
| Linear Regression | Tabular | 0.777 | 0.248 |
| Random Forest | Tabular | 0.885 | 0.178 |
| XGBoost | Tabular | **0.903** | **0.164** |
| XGBoost | Tabular + Satellite Images | 0.899 | 0.168 |

📌 While tabular XGBoost achieved slightly higher peak R², the multimodal model showed **better robustness and interpretability** by leveraging visual context.

---

## 🔍 Model Explainability (Grad-CAM)

Grad-CAM was applied to the CNN image feature extractor to highlight image regions influencing predictions.

Observed attention patterns:
- Road networks and intersections
- Residential density and building clusters
- Surrounding urban infrastructure
- Green spaces and open areas

This confirms that the model learns **meaningful real-world visual cues** relevant to property valuation.

---

## 📁 Deliverables

- **Prediction Files (CSV)**
  - `test_predictions.csv` → Tabular-only predictions
  - `test_predictions_img.csv` → Multimodal predictions (`id`, `predicted_price`)

- **Code Repository**
  - Fully reproducible notebooks and scripts

- **Project Report**
  - Detailed explanation of approach, experiments, results, and insights

---

## ⚙️ Setup & Requirements

```bash
pip install numpy pandas scikit-learn xgboost tensorflow keras opencv-python matplotlib seaborn tqdm
```

> TensorFlow ≥ 2.x recommended

---

## ▶️ How to Run

1. Clone the repository
2. Install dependencies
3. Run notebooks in the following order:
   1. `preprocessing.ipynb`
   2. `Satellite_images.ipynb`
   3. `Complete_Code.ipynb`
4. Final predictions will be saved as CSV files

---

## 👤 Author

**Chanchal Vyas**  
Sophomore, B.Tech  
Indian Institute of Technology Roorkee (IITR)  
Enrollment Number: **24126006**

---

## 🚀 Future Improvements

- Attention-based multimodal fusion
- Temporal satellite imagery for urban growth analysis
- SHAP-based explainability for tabular features
- Extension to other cities and regions

---

⭐ If you found this project interesting, feel free to explore, fork, or contribute!

