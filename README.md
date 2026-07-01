# Day-Ahead Electricity Load Forecaster ⚡

A production-grade, day-ahead electricity load forecasting system built for individual smart meters. It utilizes a **Direct Multi-Step LightGBM Regressor** optimized using **Optuna** and **TimeSeriesSplit**, compared against naive baselines and structured with a modular REST API and an interactive Streamlit UI dashboard.

---

## 🔗 Live Cloud Deployments
* **Interactive UI Dashboard (Streamlit Cloud)**: [https://work-on-grid.streamlit.app](https://work-on-grid.streamlit.app)
* **ML Backend API (Render)**: [https://work-on-grid.onrender.com](https://work-on-grid.onrender.com)
* **Interactive API Swagger Docs**: [https://work-on-grid.onrender.com/docs](https://work-on-grid.onrender.com/docs)

---

## 1. Business Understanding 📈
Electricity grid operators must forecast demand at least one day in advance to manage supply effectively. 
* **Underestimation** forces utilities to procure high-cost energy from the spot market or deploy expensive peaker plants.
* **Overestimation** leads to excess generation capacity and unused power, incurring waste.
* **Accurate forecasting** allows utilities to optimize power dispatch schedules, integrate renewable energy sources (like solar/wind), reduce grid imbalance penalties, and lower costs for consumers.

This project implements a day-ahead forecasting pipeline predicting the next **24 hourly load values** for a single smart meter using historical consumption data and exogenous parameters.

---

## 2. System & MLOps Architecture 🏗️
The training and inference pipelines are organized to match production ML systems:

```text
Raw Dataset (CSV)
       │
       ▼
Data Cleaning & Imputation (Linear interpolation + Seasonal Median Profiles)
       │
       ▼
Hourly Aggregation (SUM for consumption, MEAN for electrical measurements)
       │
       ▼
Time-Series Split (Train / Validation / Test)
       │
       ▼
Feature Engineering (Lags, Cyclical Time, Shifted Lags & Rolling Statistics)
       │
       ▼
Hyperparameter Tuning (Optuna + Chronological TimeSeriesSplit)
       │
       ▼
Point & Quantile Model Fitting (Direct Forecasting: 72 total LightGBM Models)
       │
       ▼
Serialization & Logging (Models pickled, experiments logged as JSON)
       ├──────────────────────────────────────────┐
       ▼                                          ▼
FastAPI Server (api/app.py)                Streamlit UI Dashboard (dashboard.py)
 (Prediction & Retraining endpoints)        (Interactive plots & Local Fallback Mode)
```

### Hybrid Deployment Architecture
* **FastAPI Server (ML Engine)**: Deployed on **Render** to expose standard REST API endpoints (`/predict`, `/train`, `/health`, `/metrics`).
* **Streamlit Dashboard (Frontend)**: Deployed on **Streamlit Community Cloud**. It communicates with the live FastAPI backend over HTTP. If the API is offline, it automatically falls back to **Local Fallback Mode**, loading the pickled models and local feature engineering libraries to run predictions serverlessly.

---

## 3. Data Pipeline & Cleaning 🧹
The pipeline processes minute-level data over ~4 years from the **UCI Individual Household Electric Power Consumption dataset** (2.07 million rows):
1. **Date & Time Parsing:** Columns are parsed, converted, sorted chronologically, and indexed as datetime.
2. **Missing Value Imputation:** 
   * **Short Gaps (≤ 30 minutes):** Filled using minute-level linear interpolation.
   * **Long Gaps (> 30 minutes):** Aggregated to hourly, then seasonally imputed using the training set median grouped by `[dayofweek, hour]`.
   * **Indicator Flags:** Imputation indicator columns (`is_short_imputed`, `is_profile_imputed`) are added so the model can track missingness.
3. **Split Aggregation Resampling (Hourly):**
   * **Energy Consumption columns** (`Global_active_power`, `Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3`): Aggregated using **SUM**.
   * **Electrical Measurement columns** (`Voltage`, `Global_reactive_power`, `Global_intensity`): Aggregated using **MEAN**.

---

## 4. Time-based Split 📅
To prevent data leakage, training, validation, and test data are split chronologically:
* **Train:** Dec 2006 to Jun 2009 (Ingested for training parameters)
* **Validation:** Jul 2009 to Dec 2009 (Tuning grid and early stopping evaluation)
* **Test:** Jan 2010 to Nov 2010 (Out-of-sample final evaluation)

---

## 5. Feature Engineering 🛠️
Features are engineered strictly on past historical values to prevent future data leakage (using `.shift(1)` for rolling aggregates):
* **Historical Lags:** target lags [1h, 2h, 3h, 24h, 25h, 48h, 168h, 169h] and exogenous lags [1h, 2h, 24h].
* **Shifted Rolling Aggregates:** 24h & 168h rolling mean and standard deviation.
* **Exponentially Weighted Moving Averages (EWMA):** 12h & 24h spans.
* **Cyclical Time Encodings:** Sine and Cosine transformations of `hour`, `dayofweek`, `month`, and `dayofyear`.
* **Public Holidays:** Binary indicator matching official France calendar holidays.
* **LightGBM Native Categorical Variables:** Category types passed for `hour`, `dayofweek`, and `month` to allow native categorical splits.
* **Datatype Downcasting:** Cast from `float64` to `float32` to reduce training memory footprint.

---

## 6. Model & Optimization 🤖
Because recursive forecasting compounds errors over time, we use a **Direct Multi-step Forecasting** strategy:
* We train **24 separate point regressors** (LightGBM) to forecast each hour $h \in [1, 24]$ ahead.
* We train **48 quantile regressors** (quantiles 0.05 and 0.95) to predict nominal 90% confidence bands.
* **Hyperparameter Tuning:** Conducted using **Optuna** over 10 trials with a 3-split chronological **TimeSeriesSplit** cross-validation on the validation set.
* **Model Explainability:** LightGBM feature importance is generated across all horizons.

---

## 7. Results & Benchmarks 📊
The forecaster is evaluated against three baselines:
1. **Weekly Naive:** Forecast $t+h$ is identical to same hour last week.
2. **Daily Naive:** Forecast $t+h$ is identical to same hour yesterday (standard persistence).
3. **7-Day Rolling Average:** Forecast is the average load over the last 168 hours.

### Point Forecast Performance (Test Set: Jan 2010 - Nov 2010)

| Model/Baseline | MAE (kW) | RMSE (kW) | MAPE (%) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| Weekly Naive | 36.3698 | 52.3787 | 18,253,645.78% | -0.2098 |
| Daily Naive (Tomorrow = Yesterday) | 33.3737 | 49.6135 | 7,785,666.12% | -0.0854 |
| 7-Day Rolling Average Persistence | 36.4740 | 45.6682 | 16,750,595.40% | 0.0803 |
| **LightGBM Direct Forecaster (Ours)** | **26.7776** | **36.3764** | **12,405,594.26%** | **0.4165** |

* **MAE Point Improvement:** **+19.76%** over Daily Naive.
* **RMSE Point Improvement:** **+26.68%** over Daily Naive.
* **Prediction Interval Nominal 90% Coverage Rate:** **86.49%** (highly calibrated to target 90%).

### Run Times
* **Model Training Time (72 models):** 76.62 seconds
* **Model Prediction Time (7,894 samples):** 2.61 seconds (0.33 ms/sample)

---

## 8. Diagnostic Visualizations 📈
Plots are saved in the `outputs/` folder:
* `forecast_comparison_intervals.png` - Out-of-sample forecast comparison showing predictions, actuals, and 90% confidence bands.
* `feature_importance.png` - Top 20 features averaged across all horizons.
* `residuals_vs_time.png` - Forecast errors plotted over time.
* `error_distribution.png` - Error histogram showing residuals distribution against a fitted normal distribution.
* `robustness_analysis.png` - Breakdowns of MAE by hour of day, day type, season, and load level.

---

## 9. FastAPI REST API 🚀
A containerized FastAPI server exposes prediction and monitoring endpoints.

### API Endpoints
* `GET /health` - Checks server health and model loaded status.
* `GET /model-info` - Returns training metadata, active features list, and hyperparameters.
* `GET /metrics` - Returns test set evaluation metrics.
* `POST /train` - Asynchronously triggers background retraining of the 72 LightGBM models.
* `POST /predict` - Generates 24-hour point and interval predictions given the last 168 hours of historical readings.
* `POST /batch_predict` - Generates 24-hour predictions for multiple smart meters.

---

## 10. How to Run 💻

### Running Locally (Virtual Environment)
1. **Initialize and Activate Virtual Environment:**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Execute Pipeline (Preprocessing, Tuning, Training, Evaluation, Plots):**
   ```bash
   python main.py
   ```
4. **Launch FastAPI Server (Backend):**
   ```bash
   uvicorn api.app:app --host 0.0.0.0 --port 8000
   ```
5. **Launch Streamlit Dashboard (Frontend):**
   ```bash
   streamlit run dashboard.py
   ```
6. **Run Tests:**
   ```bash
   pytest tests/
   ```

---

## 11. Written Q&A Responses 📝

### Q1: What would you change if you had to forecast for hundreds of thousands of meters at once instead of one?
1. **Global shared forecasting models:** Training 2.4 million individual models (24 horizons * 100k+ meters) is impossible to maintain. We transition to a single global model (e.g. global LightGBM or sequence-to-sequence deep learning) trained across all meters, passing static features (location, sector, contract type) or embeddings to allow the model to customize predictions.
2. **Horizon as an Input Feature:** Instead of 24 separate models, we pass the horizon h as a numerical feature, reducing model count from 24 to 1.
3. **Sequence-to-Sequence (MIMO) Deep Learning:** Use sequence-to-sequence neural networks (e.g. Temporal Fusion Transformers or DeepAR) that process the historical window and output the 24-hour forecast array at once.
4. **Distributed Processing:** Use Apache Spark or Ray to distribute feature engineering, training, and batched predictions.
5. **Centralized Feature Store:** Implement a feature store (like Feast) to precompute and serve rolling historical metrics at low latency.

### Q2: Do you think a model like this is used in practice by utilities, or would something simpler win?
* **Individual Household Level:** A complex model is **rarely** used. Single-household load is highly stochastic and erratic (e.g. driven by binary choices like turning on an oven). The signal-to-noise ratio is extremely low. Simple baselines (moving averages or historical persistence) perform similarly, costing virtually nothing to compute.
* **Aggregated Grid Level (Substations, Feeder lines, regional grids):** Yes, advanced models like LightGBM, XGBoost, and deep learning are **the industry standard**. Individual fluctuations cancel out, producing a smooth load curve. At this scale, even a 0.5% forecast accuracy improvement saves utilities millions of dollars in generation and dispatch costs, easily justifying model complexity.
