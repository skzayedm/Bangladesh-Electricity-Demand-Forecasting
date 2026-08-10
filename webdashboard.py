import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Load dataset
dataset = pd.read_csv("PGCB_date_power_demand.csv")

# Load model evaluation results
results = pd.read_csv("results.csv")

# Prediction files
lr_pred = pd.read_csv("lr_predictions.csv")
rf_pred = pd.read_csv("rf_predictions.csv")
xgb_pred = pd.read_csv("xgb_predictions.csv")
lstm_pred = pd.read_csv("lstm_predictions.csv")

# Convert datetime
dataset["datetime"] = pd.to_datetime(dataset["datetime"])

# Set datetime as index
dataset.set_index("datetime", inplace=True)

# Page configuration
st.set_page_config(
    page_title="Bangladesh Electricity Demand Forecast Dashboard",
    page_icon="⚡",
    layout="wide",
)

# Title
st.title("⚡ Bangladesh Electricity Demand Forecast Dashboard")

st.markdown("""
This dashboard demonstrates electricity demand forecasting using multiple machine learning models.

**Models Included**
- Linear Regression
- Random Forest
- XGBoost
- LSTM
""")

st.divider()

# Sidebar
st.sidebar.title("Dashboard Menu")

page = st.sidebar.radio("Go to", ["Dataset Overview", "Model Performance", "Forecast"])

if page == "Dataset Overview":
    st.header("📊 Dataset Overview")

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", f"{dataset.shape[0]:,}")
    col2.metric("Columns", dataset.shape[1])
    col3.metric(
        "Date Range", f"{dataset.index.min().date()} → {dataset.index.max().date()}"
    )

    st.divider()

    st.subheader("Dataset Preview")
    st.dataframe(dataset.head())

    st.divider()

    st.subheader("Historical Electricity Demand")

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(dataset.index, dataset["demand_mw"], color="royalblue", linewidth=0.8)

    ax.set_xlabel("Year")
    ax.set_ylabel("Demand (MW)")
    ax.set_title("Electricity Demand Over Time")

    st.pyplot(fig)

    st.divider()

    st.subheader("Demand Statistics")

    st.dataframe(dataset["demand_mw"].describe())

elif page == "Model Performance":
    st.header("📈 Model Performance")

    # ----------------------------
    # Evaluation Table
    # ----------------------------
    st.subheader("Evaluation Metrics")

    st.dataframe(
        results.style.format(
            {"MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE": "{:.2f}", "R2": "{:.4f}"}
        ),
        use_container_width=True,
    )

    st.divider()

    # ----------------------------
    # Best Model
    # ----------------------------
    best_model = results.loc[results["RMSE"].idxmin(), "Model"]

    st.success(f"🏆 Best Performing Model: {best_model}")

    col1, col2 = st.columns(2)

    col1.metric("Best RMSE", f"{results['RMSE'].min():.2f}")

    col2.metric("Best MAE", f"{results['MAE'].min():.2f}")

    st.divider()

    # ----------------------------
    # Interactive Comparison Chart
    # ----------------------------
    st.subheader("📊 Model Comparison")

    metric = st.selectbox(
        "Select Evaluation Metric", ["MAE", "RMSE", "MAPE", "R2"], key="metric_selector"
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(results["Model"], results[metric])

    if metric == "R2":
        best_idx = results[metric].idxmax()
    else:
        best_idx = results[metric].idxmin()

    bars[best_idx].set_color("green")

    ax.set_ylabel(metric)
    ax.set_title(f"{metric} Comparison Across Models")

    plt.xticks(rotation=15)

    for bar in bars:
        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    st.pyplot(fig)

elif page == "Forecast":
    st.header("🔮 Electricity Demand Forecast")

    # -----------------------------
    # Select Forecast Model
    # -----------------------------
    model_choice = st.selectbox(
        "Select Forecast Model",
        ["Linear Regression", "Random Forest", "XGBoost", "LSTM"],
        key="forecast_model",
    )

    # Load selected model predictions
    if model_choice == "Linear Regression":
        pred_df = lr_pred
        rmse = results.loc[results["Model"] == "Linear Regression", "RMSE"].values[0]

    elif model_choice == "Random Forest":
        pred_df = rf_pred
        rmse = results.loc[results["Model"] == "Random Forest", "RMSE"].values[0]

    elif model_choice == "XGBoost":
        pred_df = xgb_pred
        rmse = results.loc[results["Model"] == "XGBoost", "RMSE"].values[0]

    else:
        pred_df = lstm_pred
        rmse = results.loc[results["Model"] == "LSTM", "RMSE"].values[0]

    # -----------------------------
    # Summary Cards
    # -----------------------------
    st.subheader(f"Forecast using {model_choice}")

    col1, col2, col3 = st.columns(3)

    col1.metric("Latest Actual", f"{pred_df['Actual'].iloc[-1]:,.0f} MW")

    col2.metric("Latest Prediction", f"{pred_df['Predicted'].iloc[-1]:,.0f} MW")

    col3.metric("Model RMSE", f"{rmse:.2f}")

    st.divider()

    # -----------------------------
    # Interactive Sample Selector
    # -----------------------------
    num_samples = st.slider(
        "Number of samples to display",
        min_value=100,
        max_value=len(pred_df),
        value=1000,
        step=100,
    )

    plot_df = pred_df.head(num_samples)

    # -----------------------------
    # Actual vs Predicted Plot
    # -----------------------------
    st.subheader("Actual vs Predicted")

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(plot_df["Actual"], label="Actual", linewidth=2)

    ax.plot(plot_df["Predicted"], label="Predicted", linewidth=2)

    ax.set_title(f"{model_choice} Forecast")
    ax.set_xlabel("Test Samples")
    ax.set_ylabel("Electricity Demand (MW)")

    ax.legend()

    ax.grid(alpha=0.3)

    st.pyplot(fig)

    # -----------------------------
    # Prediction Table
    # -----------------------------
    with st.expander("View Prediction Data"):
        st.dataframe(plot_df, use_container_width=True)
