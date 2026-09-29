# 🏠 House Price Prediction System

An end-to-end machine learning system and interactive real estate valuation platform built with **Python**, **FastAPI**, **Scikit-Learn**, and a modern **Glassmorphic Web Frontend**.

The platform compares parametric **Linear Regression (OLS)** against non-parametric **Decision Tree Regressors**, achieving over **99.7% predictive accuracy** ($R^2$) across holdout test datasets.

---

## 🌟 Key Highlights

- **Complete Exploratory Data Analysis (EDA)**: Data integrity audits, zero nulls/duplicates validation, IQR anomaly detection, and median-annotated boxplots.
- **Correlation & Heatmap Suite**: Full 14-feature correlation heatmap, target price driver rankings, 2D cross-tabulation pivot heatmaps, and hierarchical cluster dendrograms.
- **Dual Machine Learning Models**:
  - **Linear Regression (OLS)**: $R^2 = 99.78\%$, $\text{RMSE} = ₹2.55\text{ Lakhs}$, $\text{MAPE} = 2.31\%$.
  - **Decision Tree Regressor**: $R^2 = 99.57\%$, $\text{RMSE} = ₹3.59\text{ Lakhs}$, $\text{MAPE} = 2.91\%$.
- **Real-time Web Application**: FastAPI backend with an interactive dark-mode glassmorphic interface, dynamic price calculation, unit pricing ($₹/\text{sqft}$), 95% confidence bounds, and multi-model consensus comparisons.

---

## 📊 Dataset Overview

The dataset contains **97 residential property records** across **14 structural, financial, and neighborhood features**:

| Feature | Description | Range |
| :--- | :--- | :--- |
| `Area_sqft` | Total living area in square feet | 600 – 2,620 sqft |
| `Bedrooms` | Number of bedrooms (BHK) | 1 – 5 |
| `Bathrooms` | Number of bathrooms | 1 – 5 |
| `Floors` | Total number of building floors | 1 – 3 |
| `Age_years` | Age of the property in years | 1 – 25 years |
| `Parking_spaces` | Dedicated vehicle parking capacity | 0 – 4 |
| `Distance_school_km` | Distance to the nearest school | 0.2 – 3.2 km |
| `Distance_hospital_km`| Distance to the nearest hospital | 0.8 – 5.1 km |
| `Distance_metro_km` | Distance to the nearest metro station | 0.3 – 5.8 km |
| `Crime_rate` | Neighborhood crime index rating | 1.0 – 7.2 |
| `Property_tax` | Annual municipal property tax in INR | ₹11,000 – ₹65,000 |
| `Furnished` | Furnishing status (0 = Unfurnished, 1 = Furnished) | 0 or 1 |
| `Location_score` | Prime location rating score | 4 – 10 |
| `Price` *(Target)* | Property transaction value in INR | ₹2.90M – ₹24.20M |

---

## 📈 Model Comparison Results

| Metric | Linear Regression (OLS) | Decision Tree Regressor |
| :--- | :--- | :--- |
| **Training $R^2$** | **0.9983 (99.83%)** | **1.0000 (100.0%)** |
| **Testing $R^2$** | **0.9978 (99.78%)** | **0.9957 (99.57%)** |
| **Testing RMSE** | **₹255,162** | **₹358,992** |
| **Testing MAE** | **₹209,690** | **₹272,500** |
| **Testing MAPE** | **2.31%** | **2.91%** |
| **Core Advantage** | Smooth continuous interpolation & unit pricing | Identifies discrete threshold cutoffs |

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/utkarshofficial999/house-price-prediction-system.git
cd house-price-prediction-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Web Application
```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```
Open your browser and navigate to `http://localhost:8000`.

### 4. Run the Jupyter Notebook
```bash
jupyter notebook house_price_eda_and_cleaning.ipynb
```

### 5. Generate Standalone Heatmaps
```bash
python plot_heatmaps.py
```

---

## 📁 Repository Structure

```text
├── house_prices.csv                 # Raw housing dataset
├── cleaned_house_prices.csv         # Cleaned, validated dataset
├── house_price_eda_and_cleaning.ipynb # Full Jupyter Notebook (44 cells executed)
├── train_model.py                   # Script to train & export models
├── plot_heatmaps.py                 # Standalone heatmap generation script
├── app.py                           # FastAPI prediction backend
├── model.joblib                     # Serialized production model
├── lr_model.joblib                  # Serialized Linear Regression model
├── dt_model.joblib                  # Serialized Decision Tree model
├── model_metadata.json              # Model metrics, coefficients & feature statistics
├── requirements.txt                 # Project dependencies
├── static/
│   ├── index.html                   # Interactive web frontend
│   ├── style.css                    # Glassmorphic responsive styling
│   └── app.js                       # Real-time client-side prediction logic
└── README.md                        # Documentation
```

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
