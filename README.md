# Bangladesh Electricity Demand Forecasting using Machine Learning

A machine learning project for forecasting electricity demand in Bangladesh using historical power demand data. The project compares multiple machine learning and deep learning models and provides an interactive Streamlit dashboard for exploring the dataset, model performance, and forecasting results.

## 🚀 Live Dashboard

**Streamlit Web Dashboard:**  
[Open the Live Dashboard](https://bangladesh-electricity-demand-forecasting-whpo4go2fj2esg9mqlxt.streamlit.app/)

## 📌 Project Overview

This project focuses on electricity demand forecasting using historical electricity demand and generation data. The workflow includes data cleaning, exploratory data analysis, feature engineering, machine learning model development, model evaluation, and interactive dashboard development.

The following models were developed and evaluated:

- Linear Regression
- Random Forest
- XGBoost
- Long Short-Term Memory (LSTM)

The models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- R² Score

## 📊 Model Performance

| Model | MAE | RMSE | MAPE (%) | R² |
|---|---:|---:|---:|---:|
| Linear Regression | 363.01 | 1545.74 | 5.16 | 0.6715 |
| Random Forest | 318.73 | 1599.18 | 3.41 | 0.6483 |
| XGBoost | 562.41 | 1704.87 | 5.89 | 0.6003 |
| LSTM | 484.95 | 1514.40 | 5.58 | 0.6847 |

Based on the evaluation results, the LSTM achieved the highest R² score and the lowest RMSE, while Random Forest achieved the lowest MAE and MAPE.

## 🔍 Project Workflow

1. Dataset loading and inspection
2. Data cleaning and preprocessing
3. Date conversion and chronological ordering
4. Missing-value and duplicate investigation
5. Outlier analysis and removal
6. Exploratory data analysis
7. Time-based feature engineering
8. Lag feature creation
9. Dataset preparation and train/test splitting
10. Machine learning model development
11. Model evaluation and comparison
12. Forecast visualisation
13. Interactive Streamlit dashboard development

## 📈 Dashboard

The interactive dashboard provides:

- Dataset overview
- Model performance comparison
- Actual vs predicted demand visualisation
- Forecast exploration for individual models
- Summary evaluation metrics
- Model comparison using different evaluation metrics

The dashboard is built using Streamlit.

## 📁 Project Files

| File | Description |
|---|---|
| `Sk_Zayed_Mahmood_Project.ipynb` | Main machine learning notebook |
| `PGCB_date_power_demand.csv` | Dataset used for the project |
| `results.csv` | Model evaluation results |
| `lr_predictions.csv` | Linear Regression predictions |
| `rf_predictions.csv` | Random Forest predictions |
| `xgb_predictions.csv` | XGBoost predictions |
| `lstm_predictions.csv` | LSTM predictions |
| `webdashboard.py` | Streamlit dashboard application |
| `requirements.txt` | Python dependencies |
| `Sk. Zayed Mahmood_Project Report.pdf` | Project report |
| `Sk. Zayed Mahmood_Project Demo.mp4` | Project demonstration video |

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Streamlit
- Jupyter Notebook

## 🎯 Key Outcome

The project demonstrates the application of both traditional machine learning and deep learning techniques to electricity demand forecasting, together with an interactive dashboard for presenting model results and forecasts.

## 👤 Author

**Sk. Zayed Mahmood**

GitHub: [@skzayedm](https://github.com/skzayedm)