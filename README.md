# AutoML-Visualizer-
An interactive Streamlit dashboard for automated data exploration, feature analysis, model training, and prediction — styled with a Power BI-inspired UI.

---

## 🚀 Features

- Upload and explore CSV datasets interactively
- Auto-detect problem type: classification or regression
- Preprocessing: missing value imputation, encoding, and feature scaling
- Train models: Random Forest, XGBoost, LightGBM, CatBoost, Linear/Ridge
- Visualize:
  - Data distributions
  - Correlation matrix
  - PCA-based cluster analysis
- Evaluate model metrics:
  - Classification: Accuracy, F1 Score, Confusion Matrix
  - Regression: RMSE, R², MAE
- What-If Analysis: interactive predictions based on feature changes

---

## 🛠 Tech Stack

- **Frontend**: [Streamlit]
- **Backend/ML**: `scikit-learn`, `xgboost`, `lightgbm`, `catboost`
- **Visualization**: `Plotly`, `Altair`

---

## 📦 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/advanced-analytics-dashboard.git
cd advanced-analytics-dashboard
```

## Install dependencies
```bash
pip install -r requirements.txt
```

## Project Structure
```bash
.
├── app.py                # Streamlit dashboard UI
├── ml_models.py          # Model training, evaluation, importance
├── preprocessing.py      # Data preprocessing pipeline
├── visualization.py      # Plotting and analysis tools
├── requirements.txt      # Python dependencies
```
