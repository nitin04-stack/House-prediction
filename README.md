# House Price Prediction

A machine learning project that predicts residential property sale prices using Linear Regression. 
The model is trained on structured housing data with feature engineering and preprocessing pipelines built in Python.

# Overview
This project builds a supervised regression model to estimate house sale prices based on property attributes such as zoning classification, 
lot configuration, building type, exterior material, and basement area. It covers the full ML workflow — from raw data ingestion to model evaluation.

# Dataset
- File: "HousePricePrediction.csv"
- Target Variable: "SalePrice"
- Key Features Used:
  - "MSZoning" — General zoning classification
  - "LotConfig" — Lot configuration
  - "BldgType"— Type of dwelling
  - "Exterior1st" — Exterior covering on house
  - "BsmtFinSF2" — Type 2 finished square feet (basement)
  - "TotalBsmtSF" — Total square feet of basement area
    
# Tech Stack
| Tool         | Purpose |
| Python       | Core programming language |
| Pandas       | Data manipulation & preprocessing |
| Scikit-learn | Model training & evaluation |

# Project Workflow
1. Data Loading — Load CSV data using Pandas
2. Handling Missing Values
   - Fill missing "SalePrice" values with column mean
   - Fill missing "BsmtFinSF2" and "TotalBsmtSF" with "0"
   - Drop remaining rows with null values
3. Feature Engineering
   - One-hot encode categorical columns: "MSZoning", "LotConfig", "BldgType", "Exterior1st"
   - Drop the "Id" column (non-predictive identifier)
4. Train/Test Split — 80/20 split with "random_state=42"
5. Model Training — Fit a "LinearRegression" model on training data
6. Evaluation — Measure model performance using (R² Score)
   
# Getting Started

##Prerequisites
```bash
pip install pandas scikit-learn
```

## Run the Notebook
```bash
jupyter notebook house_prediction.ipynb
```

# Model Performance
| Metric   | Description |
| R² Score | Proportion of variance in SalePrice explained by the model |
> The R² score is computed on the held-out test set (20% of the data).

# Repository Structure
-> house_prediction.ipynb   # Main Jupyter Notebook
-> HousePricePrediction.csv # Dataset
-> README.md                # Project documentation

# Future Improvements
- Try advanced models (Random Forest, XGBoost, Gradient Boosting)
- Add hyperparameter tuning with GridSearchCV
- Include more EDA with visualizations (correlation heatmap, distribution plots)
- Deploy model via a Flask or Streamlit web app

# License
This project is open-source and available under the [MIT License](LICENSE).
