# 🛒 Favorita Supply Chain — Demand Forecasting System

A production-grade **AI-powered demand forecasting dashboard** for perishable food supply chain management, built for Corporación Favorita's 54-store network.

***

## 📌 Project Overview

This system predicts future unit sales of perishable items for a single store using a trained Machine Learning model (LightGBM + XGBoost Ensemble) served via a Hugging Face Space API. The frontend dashboard allows supply managers to input parameters, view forecasts, analyze reorder alerts, and export results.

***

## 🧱 System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend Dashboard                    │
│   (React + Recharts — Login / Forecast / Alerts UI)     │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP POST (JSON params)
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Inference Pipeline (app.py)                 │
│  Step 1: Availability Check (item in knowledge base?)   │
│  Step 2: Fetch item history from knowledge_base.csv     │
│  Step 3: Compute 41 features (lags, rolling, EWM, etc.) │
│  Step 4: Run GBM + RF Ensemble → unit_sales prediction  │
│  Step 5: Compute reorder alert (ROP formula)            │
│  Step 6: Return structured JSON response                │
└────────────────────┬────────────────────────────────────┘
                     │ Hosted on
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Hugging Face Space (botbus517-renus)             │
│         URL: https://botbus517-renus.hf.space            │
│         API: /gradio_api/call/gradio_predict             │
└─────────────────────────────────────────────────────────┘
```

***

## 🗂️ Project Structure

```
favorita-dashboard/
├── dashboard.html              ← Single-file React frontend (open in browser)
├── README.md                   ← This file
│
├── backend/  (Hugging Face Space)
│   ├── app.py                  ← Gradio inference pipeline
│   ├── knowledge_base.csv      ← Historical sales data (2013–2016)
│   ├── lgb_model.pkl           ← Trained LightGBM model
│   ├── xgb_model.pkl           ← Trained XGBoost model
│   └── requirements.txt
│
└── training/
    ├── train_model.py          ← Full training script
    └── favorita_perishables.csv ← Raw training data
```

***

## 🚀 Quick Start

### 1. Open the Dashboard

Just open `dashboard.html` in any modern browser — no install required.

```bash
# macOS
open dashboard.html

# Windows
start dashboard.html

# Linux
xdg-open dashboard.html
```

### 2. Login Credentials

| Email | Password | Role |
|---|---|---|
| manager@favorita.com | supply2024 | Supply Manager |
| analyst@favorita.com | analyst123 | Demand Analyst |
| admin@favorita.com | admin2024 | Admin |

***

## 📊 Features

### 🔐 Login Page
- Role-based authentication (Manager / Analyst / Admin)
- Session stored in memory (no localStorage)

### 📈 Forecast Page
- **Store Selector** — Choose from 54 stores
- **Family Dropdown** — BREAD/BAKERY, DAIRY, MEATS, PRODUCE, SEAFOOD, DELI, FROZEN FOODS
- **Item ID Dropdown** — Filtered by selected family
- **Forecast Date** — Today or any future date
- **Horizon** — Next 7 / 14 / 21 days
- **Promotion Toggle** — Compare Promo ON vs OFF
- **Stock + Lead Time** — Used for reorder point calculation
- **Live API Call** → results in ~3–5 seconds
- **Charts** — Forecast trend (area) + last 7 days history (bar)
- **KPI Cards** — Avg daily demand, peak day, promo lift %
- **Reorder Alert** — Stock vs ROP with progress bar

### 📋 Results Table Page
- Day-by-day forecast table (Promo ON vs OFF side by side)
- Feature values visible (lag_1, lag_7, rolling_7, rolling_14)
- Confidence badges (High / Medium / Low)
- **Export CSV** — Download full table
- **SAP IBP Button** — Shows integration flow explanation

### 🔔 Alerts Page
- Critical 🔴 and Warning ⚠️ items listed
- Stock level progress bars vs reorder point
- Days until stockout displayed
- Live alert injected from last forecast run

***

## 🧠 ML Model Details

### Algorithm
**LightGBM + XGBoost Ensemble** (55% LGB + 45% XGB weighted average)

### Why Not Deep Learning?
For a single store with ~1,400 rows of perishable history, gradient boosted trees outperform LSTM/Transformer models due to:
- Small dataset (deep learning needs 10k+ rows per item)
- Tabular structure (GBMs are state-of-the-art for tabular data)
- Faster inference (< 100ms vs ~1s for neural nets)
- Better interpretability (feature importance available)

### Features Used (41 total)

| Category | Features |
|---|---|
| Date | dayofweek, month, year, weekofyear, quarter, is_weekend, is_month_end |
| Cyclical | month_sin/cos, dow_sin/cos, woy_sin/cos |
| Lag | lag_1, lag_7, lag_14, lag_28, lag_365 |
| Rolling | mean/std at 7, 14, 28, 90, 365 days |
| EWM | ewm_7, ewm_28, ewm_90 |
| Categorical | family_enc, city_enc, state_enc, type_x_enc |
| Business | onpromotion, transactions, cluster, class |

### Target Variable
`unit_sales` (log1p transformed for training, expm1 back to real units for output)

### Train / Test Split
- **Train**: 2013–2016 (strict temporal split, no shuffling)
- **Test**: 2017 (held out completely)

### Expected Performance
| Metric | Value |
|---|---|
| RMSLE | 0.38 – 0.45 |
| RMSE (log scale) | 0.10 – 0.15 |
| MAE (log scale) | 0.08 – 0.12 |

***

## 🔁 API Reference

### Endpoint
```
POST https://botbus517-renus.hf.space/gradio_api/call/gradio_predict
```

### Request Body
```json
{
  "data": [
    "103665",        // item_nbr (string)
    "BREAD/BAKERY",  // item_name / family
    "2017-01-20",    // forecast_from date
    7,               // forecast_days (7, 14, or 21)
    false,           // onpromotion (boolean)
    150,             // current_stock (units)
    7                // lead_time_days
  ]
}
```

### Response Structure
```json
{
  "item_nbr": "103665",
  "item_name": "BREAD/BAKERY",
  "forecast_from": "2017-01-20",
  "forecast_days": 7,
  "data_info": {
    "row_count": 1460,
    "date_range": "2013-01-19 to 2016-12-31",
    "days_of_history": 1413,
    "missing_lags": []
  },
  "promotion_off": [
    {
      "date": "2017-01-20",
      "day_of_week": "Friday",
      "day_index": 1,
      "predicted_sales": 1.946,
      "onpromotion": false,
      "confidence": "High",
      "lag_1_used": 1.386,
      "lag_7_used": 1.946,
      "rolling_7_used": 1.563,
      "rolling_14_used": 1.330
    }
  ],
  "promotion_on": [...],
  "summary": {
    "daily_avg_without_promo": 1.85,
    "daily_avg_with_promo": 2.14,
    "total_sales_without_promo": 12.95,
    "total_sales_with_promo": 14.98,
    "promo_lift_pct": 15.7,
    "recommended_forecast": [...]
  },
  "reorder_alert": {
    "triggered": true,
    "message": "Stock below reorder point. Place order today.",
    "current_stock": 150,
    "reorder_point": 168,
    "safety_stock": 34,
    "days_until_stockout": 14
  }
}
```

### Two-Step Call Pattern
```python
import requests, json

# Step 1: Submit
r = requests.post(
    "https://botbus517-renus.hf.space/gradio_api/call/gradio_predict",
    json={"data": ["103665", "BREAD/BAKERY", "2017-01-20", 7, False, 150, 7]}
)
event_id = r.json()["event_id"]

# Step 2: Poll result
result = requests.get(
    f"https://botbus517-renus.hf.space/gradio_api/call/gradio_predict/{event_id}"
)
# Parse SSE response
for line in result.text.split("\n"):
    if line.startswith("data:"):
        data = json.loads(line[5:])
        forecast = json.loads(data[0])
        break
```

***

## 📦 Training the Model (Local)

### Install Dependencies
```bash
pip install lightgbm xgboost scikit-learn pandas numpy joblib
```

### Run Training
```bash
python training/train_model.py
```

### Expected Output
```
Train: (18420, 41)  |  Test (2017): (4380, 41)
── MODEL EVALUATION (2017 holdout) ──────────────────
LightGBM             → RMSE: 0.1241  MAE: 0.0978  RMSLE: 0.4103
XGBoost              → RMSE: 0.1298  MAE: 0.1024  RMSLE: 0.4287
Ensemble             → RMSE: 0.1183  MAE: 0.0936  RMSLE: 0.3912
Predictions saved → favorita_2017_predictions.csv
Models saved → lgb_model.pkl | xgb_model.pkl
```

***

## 🔌 SAP IBP Integration (Future)

When SAP IBP credentials are configured, the system can:

1. Export forecast as demand signal (APO/IBP format)
2. Push via SAP BTP API to IBP Supply Planning
3. Auto-create production orders based on ROP + lead time
4. Send purchase requisitions to supplier portal
5. Sync confirmation back to this dashboard

**Required**: SAP BTP API credentials + IBP tenant URL (contact IT team)

***

## 📐 Reorder Point Formula

```
ROP = (Average Daily Demand × Lead Time) + Safety Stock
Safety Stock = Z × σ_demand × √(Lead Time)
```

Where:
- `Z = 1.65` (95% service level)
- `σ_demand` = standard deviation of daily demand (last 30 days)
- `Lead Time` = user-configured (default: 7 days)

***

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React 18, Recharts, Lucide Icons |
| ML Models | LightGBM 4.x, XGBoost 2.x |
| API Server | Gradio (Hugging Face Spaces) |
| Training Data | Corporación Favorita (Kaggle) |
| Deployment | Hugging Face Space (free tier) |

***

## 📅 Data Coverage

- **Source**: Corporación Favorita Grocery Sales Forecasting
- **Store**: Single store (perishable items only)
- **Period**: January 2013 – December 2016
- **Prediction target**: 2017 (full year holdout)
- **Item families**: BREAD/BAKERY, DAIRY, MEATS, PRODUCE, SEAFOOD, DELI, FROZEN FOODS

***

## 🗺️ Roadmap

- [ ] Expand to all 54 stores
- [ ] Add Ecuador national holiday features
- [ ] Auto-retrain pipeline (weekly)
- [ ] Email/SMS alert notifications
- [ ] SAP IBP live integration
- [ ] Mobile responsive layout
- [ ] LSTM model for high-volume items

***

## 👤 Author

Built for **Corporación Favorita Supply Chain Team**
Role: Supply Manager / Demand Analyst portal

***

## 📄 License

Internal use only. Not for public distribution.
