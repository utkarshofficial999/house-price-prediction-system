import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import json

# 1. Load Data
df = pd.read_csv('house_prices.csv')
features = [
    'Area_sqft', 'Bedrooms', 'Bathrooms', 'Floors', 'Age_years',
    'Parking_spaces', 'Distance_school_km', 'Distance_hospital_km',
    'Distance_metro_km', 'Crime_rate', 'Property_tax', 'Furnished',
    'Location_score'
]
target = 'Price'

X = df[features]
y = df[target]

# 2. Train/Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 3. Fit Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_y_pred_train = lr_model.predict(X_train)
lr_y_pred_test = lr_model.predict(X_test)

lr_r2_train = float(r2_score(y_train, lr_y_pred_train))
lr_r2_test = float(r2_score(y_test, lr_y_pred_test))
lr_rmse_test = float(np.sqrt(mean_squared_error(y_test, lr_y_pred_test)))
lr_mae_test = float(mean_absolute_error(y_test, lr_y_pred_test))
lr_mape_test = float(np.mean(np.abs((y_test - lr_y_pred_test) / y_test)) * 100)

# 4. Fit Decision Tree Regressor
dt_model = DecisionTreeRegressor(max_depth=7, min_samples_split=2, min_samples_leaf=1, random_state=42)
dt_model.fit(X_train, y_train)

dt_y_pred_train = dt_model.predict(X_train)
dt_y_pred_test = dt_model.predict(X_test)

dt_r2_train = float(r2_score(y_train, dt_y_pred_train))
dt_r2_test = float(r2_score(y_test, dt_y_pred_test))
dt_rmse_test = float(np.sqrt(mean_squared_error(y_test, dt_y_pred_test)))
dt_mae_test = float(mean_absolute_error(y_test, dt_y_pred_test))
dt_mape_test = float(np.mean(np.abs((y_test - dt_y_pred_test) / y_test)) * 100)

print("="*60)
print("     MODEL PERFORMANCE COMPARISON ON TEST SET")
print("="*60)
print(f"Linear Regression       -> R2: {lr_r2_test:.5f} | RMSE: INR {lr_rmse_test:,.2f} | MAE: INR {lr_mae_test:,.2f} | MAPE: {lr_mape_test:.2f}%")
print(f"Decision Tree Regressor -> R2: {dt_r2_test:.5f} | RMSE: INR {dt_rmse_test:,.2f} | MAE: INR {dt_mae_test:,.2f} | MAPE: {dt_mape_test:.2f}%")
print("="*60)

# 5. Save Models
joblib.dump(lr_model, 'lr_model.joblib')
joblib.dump(dt_model, 'dt_model.joblib')
# Also keep model.joblib as default linear regression
joblib.dump(lr_model, 'model.joblib')

# 6. Feature Statistics
feature_stats = {}
for col in features:
    feature_stats[col] = {
        'min': float(df[col].min()),
        'max': float(df[col].max()),
        'mean': float(df[col].mean()),
        'median': float(df[col].median()),
        'std': float(df[col].std())
    }

metadata = {
    'models': {
        'linear_regression': {
            'name': 'Linear Regression (OLS)',
            'algorithm': 'LinearRegression',
            'r2_train': round(lr_r2_train, 5),
            'r2_test': round(lr_r2_test, 5),
            'rmse_test': round(lr_rmse_test, 2),
            'mae_test': round(lr_mae_test, 2),
            'mape_test': round(lr_mape_test, 2),
            'intercept': round(float(lr_model.intercept_), 2),
            'coefficients': {col: round(float(coef), 2) for col, coef in zip(features, lr_model.coef_)}
        },
        'decision_tree': {
            'name': 'Decision Tree Regressor',
            'algorithm': 'DecisionTreeRegressor',
            'max_depth': 7,
            'r2_train': round(dt_r2_train, 5),
            'r2_test': round(dt_r2_test, 5),
            'rmse_test': round(dt_rmse_test, 2),
            'mae_test': round(dt_mae_test, 2),
            'mape_test': round(dt_mape_test, 2),
            'feature_importances': {col: round(float(imp), 4) for col, imp in zip(features, dt_model.feature_importances_)}
        }
    },
    'features': features,
    'feature_stats': feature_stats,
    'target': target,
    'total_samples': len(df),
    'training_samples': len(X_train),
    'test_samples': len(X_test)
}

with open('model_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)

print("Saved 'lr_model.joblib', 'dt_model.joblib', and 'model_metadata.json'.")
