import os
import json
import pickle
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'DayAheadLoadForecaster':
            from src.models import DayAheadLoadForecaster
            return DayAheadLoadForecaster
        return super().find_class(module, name)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Streamlit Page Config
st.set_page_config(
    page_title="Day-Ahead Electricity Load Forecaster",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants & Paths
API_URL = "https://work-on-grid.onrender.com"
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join(OUTPUT_DIR, "models", "forecaster_lgb.pkl")
METRICS_PATH = os.path.join(OUTPUT_DIR, "metrics.json")
CONFIG_PATH = os.path.join(OUTPUT_DIR, "config.json")
ROBUSTNESS_PATH = os.path.join(OUTPUT_DIR, "robustness_checks.json")
DATA_PATH = os.path.join("individual+household+electric+power+consumption", "household_power_consumption.txt")
HOURLY_DATA_CSV = os.path.join(OUTPUT_DIR, "hourly_data.csv")

TARGET_COL = "Global_active_power"
EXOG_COLS = [
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

# Custom CSS for Premium Design Style
st.markdown("""
<style>
    /* Metric Card Styling */
    div[data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 700;
        color: #ff7f0e;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        font-weight: 600;
        color: #555;
    }
    /* Main Background adjustments */
    .stApp {
        background-color: #fafbfc;
    }
    /* Sidebar adjustments */
    section[data-testid="stSidebar"] {
        background-color: #1a1c23;
        color: #ffffff;
    }
    section[data-testid="stSidebar"] hr {
        border-top: 1px solid #2d3139;
    }
    .sidebar-title {
        color: #ff7f0e;
        font-size: 20px;
        font-weight: 800;
        margin-bottom: 5px;
    }
    .badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
        color: white;
    }
    .badge-online { background-color: #2ecc71; }
    .badge-offline { background-color: #e74c3c; }
    .badge-fallback { background-color: #f39c12; }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# DATA LOADING & CACHING
# ==============================================================================
@st.cache_data(show_spinner="Loading and preprocessing dataset (takes ~20s on first load, then instant)...")
def load_dataset():
    """
    Loads preprocessed hourly data. If not preprocessed, runs preprocessing once
    and saves to CSV for rapid future loads.
    """
    if os.path.exists(HOURLY_DATA_CSV):
        logger.info("Loading cached hourly dataset...")
        df = pd.read_csv(HOURLY_DATA_CSV, parse_dates=["Datetime"])
        df.set_index("Datetime", inplace=True)
        return df
    else:
        logger.info("Cached CSV not found. Preprocessing raw data...")
        from src.preprocess import preprocess_data
        df = preprocess_data(DATA_PATH)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        df.to_csv(HOURLY_DATA_CSV)
        return df


# ==============================================================================
# API HEALTH & INTEGRATION STATUS
# ==============================================================================
def check_api_status() -> dict:
    """Checks if the FastAPI backend is running and loaded."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=1.5)
        if response.status_code == 200:
            return {"status": "Online", "model_loaded": response.json().get("model_loaded", False)}
    except requests.exceptions.RequestException:
        pass
    return {"status": "Offline", "model_loaded": False}


# ==============================================================================
# FEATURE ENGINEERING & FORECAST FOR FALLBACK MODE
# ==============================================================================
def build_features_for_inference(df_history: pd.DataFrame, target_time: pd.Timestamp) -> pd.DataFrame:
    """
    Builds the feature row at the selected forecast origin timestamp t,
    without leakage and without dropping the row if future target values are NaN.
    """
    # Slice the history up to the target forecast origin t
    df_slice = df_history.loc[:target_time].copy()
    
    # We append 24 dummy rows at the end so shift(-h) does not produce NaNs at time t
    dummy_index = pd.date_range(start=target_time + pd.Timedelta(hours=1), periods=24, freq='h')
    dummy_df = pd.DataFrame(0.0, index=dummy_index, columns=df_slice.columns)
    df_extended = pd.concat([df_slice, dummy_df])
    
    # Run the feature builder
    from src.features import create_features_and_targets
    X, _ = create_features_and_targets(df_extended, outlier_clipping=False)
    
    # Retrieve the row corresponding to the target time
    X_origin = X.loc[[target_time]]
    return X_origin


def generate_local_predictions(df_history: pd.DataFrame, target_time: pd.Timestamp) -> tuple:
    """Generates forecasts using the serialized model locally (fallback mode)."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Pickled model file not found at {MODEL_PATH}. Run python main.py first.")
        return None, None, None
        
    with open(MODEL_PATH, "rb") as f:
        local_forecaster = CustomUnpickler(f).load()
        
    X_origin = build_features_for_inference(df_history, target_time)
    point_preds, lower_preds, upper_preds = local_forecaster.predict(X_origin)
    return point_preds[0], lower_preds[0], upper_preds[0]


# ==============================================================================
# SIDEBAR
# ==============================================================================
st.sidebar.markdown('<div class="sidebar-title">⚡ Load Forecaster</div>', unsafe_allow_html=True)
st.sidebar.caption("Day-Ahead Smart Meter Forecaster")
st.sidebar.markdown("---")

# Navigation Menu
nav_selection = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 Forecast Viewer",
        "📈 Model Performance",
        "🔍 Feature & Error Analysis",
        "⚙️ API & Retraining Control"
    ]
)

st.sidebar.markdown("---")

# Check API status
api_info = check_api_status()
st.sidebar.markdown("### Integration Status")

if api_info["status"] == "Online":
    st.sidebar.markdown(
        'Backend API Status: <span class="badge badge-online">ONLINE</span>',
        unsafe_allow_html=True
    )
    if api_info["model_loaded"]:
        st.sidebar.markdown(
            'ML Model Engine: <span class="badge badge-online">LOADED</span>',
            unsafe_allow_html=True
        )
    else:
        st.sidebar.markdown(
            'ML Model Engine: <span class="badge badge-fallback">NOT LOADED</span>',
            unsafe_allow_html=True
        )
    mode = "api"
else:
    st.sidebar.markdown(
        'Backend API Status: <span class="badge badge-offline">OFFLINE</span>',
        unsafe_allow_html=True
    )
    st.sidebar.markdown(
        'ML Model Engine: <span class="badge badge-fallback">LOCAL FALLBACK</span>',
        unsafe_allow_html=True
    )
    mode = "fallback"

st.sidebar.markdown("---")
st.sidebar.caption("Created with Antigravity AI Coding Assistant")


# ==============================================================================
# MAIN PAGE ROUTING
# ==============================================================================

# Load hourly dataset
df_hourly = load_dataset()
test_set_df = df_hourly.loc[df_hourly.index >= "2010-01-01 00:00:00"]


if nav_selection == "📊 Forecast Viewer":
    st.title("📊 Day-Ahead Electricity Load Forecast Viewer")
    st.write(
        "Generate 24-hour point forecasts and 90% confidence bands for any selected "
        "time in the test set. If the backend API is online, predictions are generated via the REST API; "
        "otherwise, the dashboard falls back to loading the model locally."
    )
    
    # Selector for Datetime
    col_sel1, col_sel2 = st.columns([3, 1])
    
    with col_sel1:
        # Prepopulate datetime picker with test set indices
        min_date = test_set_df.index.min().to_pydatetime()
        max_date = (test_set_df.index.max() - pd.Timedelta(hours=24)).to_pydatetime()
        
        # Streamlit Date and Time Pickers
        selected_date = st.date_input("Choose Forecast Origin Date", value=min_date, min_value=min_date, max_value=max_date)
        selected_hour = st.slider("Choose Forecast Origin Hour (t)", min_value=0, max_value=23, value=12)
        
        target_timestamp = pd.Timestamp(selected_date).replace(hour=selected_hour)
        
    with col_sel2:
        st.write("##### Quick Options")
        if st.button("🎲 Select Random Test Time", use_container_width=True):
            random_idx = np.random.choice(len(test_set_df) - 48)
            random_time = test_set_df.index[random_idx]
            # Update values
            st.session_state["sel_date"] = random_time.date()
            st.session_state["sel_hour"] = int(random_time.hour)
            st.rerun()

    # Apply quick options values if present
    if "sel_date" in st.session_state:
        selected_date = st.session_state.pop("sel_date")
        selected_hour = st.session_state.pop("sel_hour")
        target_timestamp = pd.Timestamp(selected_date).replace(hour=selected_hour)
        st.rerun()
        
    st.markdown("---")
    
    # Verify we have enough history
    history_needed = 168  # 7 days
    history_slice = df_hourly.loc[:target_timestamp]
    
    if len(history_slice) < history_needed:
        st.error(f"Insufficient history up to {target_timestamp} (requires 168 hours of historical readings).")
    else:
        # Load actuals for next 24 hours to overlay
        actuals_slice = df_hourly.loc[target_timestamp + pd.Timedelta(hours=1) : target_timestamp + pd.Timedelta(hours=24)]
        
        # Generate prediction
        with st.spinner("Generating forecasts..."):
            if mode == "api" and api_info["model_loaded"]:
                # Convert the preceding 168 hours of readings to API Request schema
                past_168 = df_hourly.loc[target_timestamp - pd.Timedelta(hours=167) : target_timestamp]
                history_payload = []
                for t_val, row in past_168.iterrows():
                    history_payload.append({
                        "timestamp": t_val.isoformat(),
                        "Global_active_power": float(row["Global_active_power"]),
                        "Global_reactive_power": float(row["Global_reactive_power"]),
                        "Voltage": float(row["Voltage"]),
                        "Global_intensity": float(row["Global_intensity"]),
                        "Sub_metering_1": float(row["Sub_metering_1"]),
                        "Sub_metering_2": float(row["Sub_metering_2"]),
                        "Sub_metering_3": float(row["Sub_metering_3"])
                    })
                
                try:
                    response = requests.post(f"{API_URL}/predict", json={"history": history_payload}, timeout=5)
                    if response.status_code == 200:
                        predictions_list = response.json()["predictions"]
                        point_preds = np.array([p["point_pred_kw"] for p in predictions_list])
                        lower_preds = np.array([p["lower_bound_90pct_kw"] for p in predictions_list])
                        upper_preds = np.array([p["upper_bound_90pct_kw"] for p in predictions_list])
                    else:
                        st.warning("API Prediction failed. Falling back to local model execution.")
                        point_preds, lower_preds, upper_preds = generate_local_predictions(df_hourly, target_timestamp)
                except Exception as e:
                    st.warning(f"Failed to connect to API ({str(e)}). Falling back to local model execution.")
                    point_preds, lower_preds, upper_preds = generate_local_predictions(df_hourly, target_timestamp)
            else:
                point_preds, lower_preds, upper_preds = generate_local_predictions(df_hourly, target_timestamp)
                
        if point_preds is not None:
            # Build forecast series
            forecast_index = pd.date_range(start=target_timestamp + pd.Timedelta(hours=1), periods=24, freq='h')
            
            # KPI stats for forecast window
            if len(actuals_slice) == 24:
                window_actuals = actuals_slice[TARGET_COL].values
                mae = np.mean(np.abs(window_actuals - point_preds))
                rmse = np.sqrt(np.mean((window_actuals - point_preds)**2))
                
                col_metric1, col_metric2, col_metric3 = st.columns(3)
                with col_metric1:
                    st.metric("Forecast Origin Time (t)", target_timestamp.strftime("%Y-%m-%d %H:%M"))
                with col_metric2:
                    st.metric("Forecast Window MAE", f"{mae:.4f} kW")
                with col_metric3:
                    st.metric("Forecast Window RMSE", f"{rmse:.4f} kW")
            else:
                st.warning("Actual future values are unavailable for this timeframe. Plotting predictions only.")
                
            # Create Plotly figure
            fig = go.Figure()
            
            # Historical load (past 24 hours) for context
            past_context = df_hourly.loc[target_timestamp - pd.Timedelta(hours=23) : target_timestamp]
            fig.add_trace(go.Scatter(
                x=past_context.index,
                y=past_context[TARGET_COL],
                name="Historical Load (Past 24h)",
                line=dict(color="#7f8c8d", width=2),
                mode="lines"
            ))
            
            # Actual values
            if len(actuals_slice) == 24:
                fig.add_trace(go.Scatter(
                    x=actuals_slice.index,
                    y=actuals_slice[TARGET_COL],
                    name="Actual Load",
                    line=dict(color="#2c3e50", width=3),
                    mode="lines"
                ))
                
            # Prediction line
            fig.add_trace(go.Scatter(
                x=forecast_index,
                y=point_preds,
                name="LightGBM Point Forecast",
                line=dict(color="#ff7f0e", width=2.5, dash="dash"),
                mode="lines"
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=forecast_index,
                y=lower_preds,
                name="Lower Bound (5th Pct)",
                line=dict(color="rgba(255, 127, 14, 0.2)", width=0),
                mode="lines",
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=forecast_index,
                y=upper_preds,
                name="90% Confidence Interval",
                fill="tonexty",
                fillcolor="rgba(255, 127, 14, 0.15)",
                line=dict(color="rgba(255, 127, 14, 0.2)", width=0),
                mode="lines"
            ))
            
            # Add vertical reference line for forecast origin
            fig.add_vline(x=target_timestamp, line_width=1.5, line_dash="dot", line_color="#e74c3c")
            fig.add_annotation(
                x=target_timestamp,
                y=max(past_context[TARGET_COL].max(), point_preds.max()) * 0.9,
                text="Forecast Origin",
                showarrow=True,
                arrowhead=1,
                arrowcolor="#e74c3c",
                ax=50,
                ay=-20
            )
            
            fig.update_layout(
                title=f"Day-Ahead Forecast Starting from {target_timestamp.strftime('%Y-%m-%d %H:%M')}",
                xaxis_title="Datetime",
                yaxis_title="Active Power (kW)",
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(l=40, r=40, t=80, b=40),
                height=550,
                plot_bgcolor="white"
            )
            
            fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor="#f1f2f6")
            fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor="#f1f2f6")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show predictions data table
            with st.expander("📄 View Forecast Details (Table)"):
                forecast_table = pd.DataFrame({
                    "Horizon": [f"t+{h}" for h in range(1, 25)],
                    "Timestamp": [t.strftime("%Y-%m-%d %H:%M") for t in forecast_index],
                    "Lower Bound 90% (kW)": lower_preds,
                    "Point Forecast (kW)": point_preds,
                    "Upper Bound 90% (kW)": upper_preds
                })
                if len(actuals_slice) == 24:
                    forecast_table["Actual (kW)"] = actuals_slice[TARGET_COL].values
                    forecast_table["Error (kW)"] = forecast_table["Actual (kW)"] - forecast_table["Point Forecast (kW)"]
                    
                st.dataframe(forecast_table.style.format(precision=4), use_container_width=True)


elif nav_selection == "📈 Model Performance":
    st.title("📈 Model Performance & Baselines Comparison")
    st.write("Compare the LightGBM Direct Forecaster's performance against persistence baselines on the test set.")
    
    # Load metrics
    if not os.path.exists(METRICS_PATH):
        st.error(f"Evaluation metrics not found at {METRICS_PATH}. Please run python main.py to train models.")
    else:
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
            
        lgbm = metrics["model_metrics"]
        daily = metrics["daily_naive"]
        weekly = metrics["weekly_naive"]
        rolling = metrics["rolling_avg_7d"]
        quantiles = metrics["quantiles"]
        
        # Metric Columns
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric(label="LightGBM Test MAE", value=f"{lgbm['MAE']:.4f} kW", delta=f"{((lgbm['MAE'] - daily['MAE'])/daily['MAE'])*100:+.2f}% vs Daily Naive", delta_color="inverse")
        with col_m2:
            st.metric(label="LightGBM Test RMSE", value=f"{lgbm['RMSE']:.4f} kW", delta=f"{((lgbm['RMSE'] - daily['RMSE'])/daily['RMSE'])*100:+.2f}% vs Daily Naive", delta_color="inverse")
        with col_m3:
            st.metric(label="Nominal 90% Interval Coverage", value=f"{quantiles['coverage_rate_90pct']:.2f}%", delta="Target: 90.00%")
        with col_m4:
            st.metric(label="LightGBM Test R² Score", value=f"{lgbm['R2']:.4f}", delta=f"{lgbm['R2'] - daily['R2']:+.4f} vs Daily Naive")

        st.markdown("---")
        
        # Benchmark Table
        st.subheader("Point Forecast Evaluation Summary")
        bench_df = pd.DataFrame({
            "MAE (kW)": [weekly["MAE"], daily["MAE"], rolling["MAE"], lgbm["MAE"]],
            "RMSE (kW)": [weekly["RMSE"], daily["RMSE"], rolling["RMSE"], lgbm["RMSE"]],
            "R² Score": [weekly["R2"], daily["R2"], rolling["R2"], lgbm["R2"]],
            "Improvement vs Daily Naive (MAE)": [
                f"{((weekly['MAE']-daily['MAE'])/daily['MAE'])*100:+.2f}%",
                "Baseline",
                f"{((rolling['MAE']-daily['MAE'])/daily['MAE'])*100:+.2f}%",
                f"{((daily['MAE']-lgbm['MAE'])/daily['MAE'])*100:+.2f}%"
            ]
        }, index=[
            "Weekly Naive (Same hour last week)",
            "Daily Naive (Persistence)",
            "7-Day Rolling Average Persistence",
            "LightGBM Direct Forecaster (Ours)"
        ])
        
        st.table(bench_df.style.format(precision=4))
        
        st.markdown("---")
        
        # Quantile Metrics
        st.subheader("Prediction Interval Evaluation (90% Nominal Confidence)")
        col_q1, col_q2 = st.columns(2)
        with col_q1:
            st.markdown(f"""
            - **Pinball Loss (q=0.05)**: `{quantiles['pinball_0.05']:.4f}`
            - **Pinball Loss (q=0.95)**: `{quantiles['pinball_0.95']:.4f}`
            """)
        with col_q2:
            st.write("##### Coverage Reliability Indicator")
            coverage = quantiles['coverage_rate_90pct']
            st.progress(coverage / 100.0)
            st.caption(f"Nominal 90% Prediction Interval covers {coverage:.2f}% of out-of-sample data points.")


elif nav_selection == "🔍 Feature & Error Analysis":
    st.title("🔍 Feature Importances & Robustness Error Breakdown")
    st.write("Understand which features drive model decisions and check performance stability across various segments.")
    
    # 1. Feature Importance (Check if pickled model is available to generate importances)
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            local_forecaster = CustomUnpickler(f).load()
            
        # Compile feature importances
        feat_imp_df = pd.DataFrame()
        for h, model in local_forecaster.point_models.items():
            df_temp = pd.DataFrame({'feature': model.feature_name_, f'importance_h{h}': model.feature_importances_})
            feat_imp_df = df_temp if feat_imp_df.empty else feat_imp_df.merge(df_temp, on='feature', how='outer')
            
        imp_cols = [c for c in feat_imp_df.columns if c != 'feature']
        feat_imp_df['mean_importance'] = feat_imp_df[imp_cols].mean(axis=1)
        feat_imp_df.sort_values('mean_importance', ascending=False, inplace=True)
        top_20 = feat_imp_df.head(20)
        
        st.subheader("💡 Top 20 Most Important Features")
        fig_imp = px.bar(
            top_20,
            x="mean_importance",
            y="feature",
            orientation="h",
            labels={"mean_importance": "Average Split Count Importance", "feature": "Feature Name"},
            color="mean_importance",
            color_continuous_scale="Oranges"
        )
        fig_imp.update_layout(yaxis={'categoryorder':'total ascending'}, height=500, plot_bgcolor="white")
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.warning("Model file not found. Feature importance is unavailable.")
        
    st.markdown("---")
    
    # 2. Robustness Check Breakdowns
    if os.path.exists(ROBUSTNESS_PATH):
        with open(ROBUSTNESS_PATH, "r") as f:
            robustness = json.load(f)
            
        st.subheader("📊 Robustness Checks (Error breakdowns)")
        
        # Selection of Breakdown Type
        breakdown_type = st.selectbox(
            "Select Breakdown Dimension",
            ["Hour of Day", "Day Type", "Season", "Load Level"]
        )
        
        if breakdown_type == "Hour of Day":
            hour_data = robustness["hour_of_day"]
            hours = sorted(list(map(int, hour_data.keys())))
            mae_vals = [hour_data[str(h)]["MAE"] for h in hours]
            rmse_vals = [hour_data[str(h)]["RMSE"] for h in hours]
            
            fig_err = go.Figure()
            fig_err.add_trace(go.Scatter(x=hours, y=mae_vals, name="MAE (kW)", line=dict(color="#ff7f0e", width=2.5), mode="lines+markers"))
            fig_err.add_trace(go.Scatter(x=hours, y=rmse_vals, name="RMSE (kW)", line=dict(color="#1f77b4", width=2), mode="lines+markers"))
            fig_err.update_layout(
                title="Forecast Error Metrics by Hour of Day",
                xaxis=dict(title="Hour of Day", tickmode="linear", tick0=0, dtick=1),
                yaxis_title="Error (kW)",
                hovermode="x unified",
                plot_bgcolor="white"
            )
            st.plotly_chart(fig_err, use_container_width=True)
            
        elif breakdown_type == "Day Type":
            day_data = robustness["day_type"]
            categories = list(day_data.keys())
            mae_vals = [day_data[c]["MAE"] for c in categories]
            rmse_vals = [day_data[c]["RMSE"] for c in categories]
            
            fig_err = go.Figure(data=[
                go.Bar(name='MAE', x=categories, y=mae_vals, marker_color='#ff7f0e'),
                go.Bar(name='RMSE', x=categories, y=rmse_vals, marker_color='#1f77b4')
            ])
            fig_err.update_layout(title="Forecast Error Metrics by Day Type (Weekday vs Weekend)", yaxis_title="Error (kW)", barmode='group', plot_bgcolor="white")
            st.plotly_chart(fig_err, use_container_width=True)
            
        elif breakdown_type == "Season":
            season_data = robustness["season"]
            seasons = list(season_data.keys())
            mae_vals = [season_data[s]["MAE"] for s in seasons]
            rmse_vals = [season_data[s]["RMSE"] for s in seasons]
            
            fig_err = go.Figure(data=[
                go.Bar(name='MAE', x=seasons, y=mae_vals, marker_color='#ff7f0e'),
                go.Bar(name='RMSE', x=seasons, y=rmse_vals, marker_color='#1f77b4')
            ])
            fig_err.update_layout(title="Forecast Error Metrics by Season", yaxis_title="Error (kW)", barmode='group', plot_bgcolor="white")
            st.plotly_chart(fig_err, use_container_width=True)
            
        elif breakdown_type == "Load Level":
            load_data = robustness["load_level"]
            levels = list(load_data.keys())
            mae_vals = [load_data[l]["MAE"] for l in levels]
            rmse_vals = [load_data[l]["RMSE"] for l in levels]
            
            fig_err = go.Figure(data=[
                go.Bar(name='MAE', x=levels, y=mae_vals, marker_color='#ff7f0e'),
                go.Bar(name='RMSE', x=levels, y=rmse_vals, marker_color='#1f77b4')
            ])
            fig_err.update_layout(title="Forecast Error Metrics by Load Level (Low-Load <= Median vs High-Load > Median)", yaxis_title="Error (kW)", barmode='group', plot_bgcolor="white")
            st.plotly_chart(fig_err, use_container_width=True)
    else:
        st.warning("Robustness checks JSON not found. Run evaluation script to populate error breakdowns.")


elif nav_selection == "⚙️ API & Retraining Control":
    st.title("⚙️ System Control & API Retraining Controller")
    st.write(
        "Monitor configuration metadata, inspect active hyperparameters, "
        "and asynchronously trigger model retraining on the server."
    )
    
    # 1. Active Hyperparameters
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
            
        st.subheader("🔧 Active Model Config & Hyperparameters")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.metric("Total Active Features", config.get("num_features", "N/A"))
        with col_c2:
            st.metric("Training Duration", f"{config.get('training_time_seconds', 0.0):.2f} seconds")
        with col_c3:
            st.metric("Last Training Run", config.get("timestamp", "N/A")[:19])
            
        st.json(config.get("hyperparameters", {}))
    else:
        st.warning("Configuration metadata not found. Run main pipeline to generate config.")
        
    st.markdown("---")
    
    # 2. Trigger Retraining
    st.subheader("🔄 Trigger Background Retraining")
    st.write(
        "Initiate background retraining of the 72 point and quantile LightGBM models. "
        "Optuna will search for optimal hyperparameters across a 3-split TimeSeriesSplit. "
        "If the FastAPI server is Online, retraining is handled asynchronously in the background. "
        "If the server is Offline, retraining will run synchronously on the local environment."
    )
    
    if st.button("🚀 Start Retraining", type="primary", use_container_width=True):
        if mode == "api":
            try:
                response = requests.post(f"{API_URL}/train")
                if response.status_code == 200:
                    st.success("Asynchronous model retraining triggered successfully on the FastAPI server!")
                    st.info("The server is training models in the background. You can check the terminal console logs.")
                else:
                    st.error(f"Server returned error code: {response.status_code}")
            except Exception as e:
                st.error(f"Failed to connect to API: {str(e)}")
        else:
            st.warning("FastAPI API is Offline. Running retraining locally in synchronous mode (this will block the UI for ~70s).")
            with st.spinner("Retraining 72 models (Preprocessing + Optuna HPO + Model Fit)..."):
                from src.train import main as run_training
                try:
                    run_training()
                    st.success("Model retrained successfully and serialized to disk!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Local retraining failed: {str(e)}")
