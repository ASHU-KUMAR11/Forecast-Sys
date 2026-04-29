import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os
import json
import traceback
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

# ══════════════════════════════════════════════════════════════
#  GLOBALS
# ══════════════════════════════════════════════════════════════
DATA_PATH    = "knowledge_base.csv"
MODEL_PATH   = "models/"
DEFAULT_LEAD = 7           # default supplier lead time (days)
Z_SCORE      = 1.65        # 95 % service level for safety stock

df_kb        = None
model_gbm    = None
model_rf     = None
ENCODERS     = {}
FEATURE_COLS = []

REQUIRED_FEATURES = [
    "item_nbr", "class", "cluster", "onpromotion", "transactions",
    "dayofweek", "month", "year", "weekofyear", "day", "quarter", "dayofyear",
    "is_weekend", "is_month_start", "is_month_end", "is_quarter_end",
    "month_sin", "month_cos", "dow_sin", "dow_cos", "woy_sin", "woy_cos",
    "lag_1", "lag_7", "lag_14", "lag_28", "lag_365",
    "rolling_mean_7",  "rolling_std_7",
    "rolling_mean_14", "rolling_std_14",
    "rolling_mean_28", "rolling_std_28",
    "rolling_mean_90",
    "ewm_7", "ewm_28", "ewm_90",
    "family_enc", "city_enc", "state_enc", "type_x_enc",
]


# ══════════════════════════════════════════════════════════════
#  STARTUP — load CSV + models (or train on first run)
# ══════════════════════════════════════════════════════════════
def _engineer_for_training(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["item_nbr", "date"]).reset_index(drop=True)

    df["day"]       = df["date"].dt.day
    df["month"]     = df["date"].dt.month
    df["year"]      = df["date"].dt.year
    df["dayofweek"] = df["date"].dt.dayofweek
    df["dayofyear"] = df["date"].dt.day_of_year
    df["weekofyear"]= df["date"].dt.isocalendar().week.astype(int)
    df["quarter"]   = df["date"].dt.quarter

    df["is_weekend"]     = (df["dayofweek"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"]   = df["date"].dt.is_month_end.astype(int)
    df["is_quarter_end"] = df["date"].dt.is_quarter_end.astype(int)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["woy_sin"]   = np.sin(2 * np.pi * df["weekofyear"] / 52)
    df["woy_cos"]   = np.cos(2 * np.pi * df["weekofyear"] / 52)

    for lag in [1, 7, 14, 28, 365]:
        df[f"lag_{lag}"] = df.groupby("item_nbr")["unit_sales"].shift(lag)

    for win in [7, 14, 28, 90]:
        df[f"rolling_mean_{win}"] = df.groupby("item_nbr")["unit_sales"].transform(
            lambda x: x.shift(1).rolling(win, min_periods=1).mean())
        df[f"rolling_std_{win}"]  = df.groupby("item_nbr")["unit_sales"].transform(
            lambda x: x.shift(1).rolling(win, min_periods=1).std())

    for span in [7, 28, 90]:
        df[f"ewm_{span}"] = df.groupby("item_nbr")["unit_sales"].transform(
            lambda x: x.shift(1).ewm(span=span).mean())

    for col in ["family", "city", "state", "type_x"]:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        ENCODERS[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    return df


def load_everything():
    global df_kb, model_gbm, model_rf, FEATURE_COLS

    # ── 1. Load knowledge base ────────────────────────────────
    print("📂 Loading knowledge_base.csv …")
    df_kb = pd.read_csv(DATA_PATH, parse_dates=["date"])

    # Apply log1p only if values look like raw unit sales (> 20)
    if df_kb["unit_sales"].max() > 20:
        df_kb["unit_sales"] = np.log1p(df_kb["unit_sales"].clip(0))
        print("   Applied log1p transform to unit_sales")

    df_kb["onpromotion"] = (
        df_kb["onpromotion"]
        .map({True: 1, False: 0, "TRUE": 1, "FALSE": 0, 1: 1, 0: 0})
        .fillna(0).astype(int)
    )
    print(f"   ✅ {df_kb.shape[0]:,} rows  |  {df_kb['item_nbr'].nunique()} items  "
          f"|  {df_kb['date'].min().date()} → {df_kb['date'].max().date()}")

    # ── 2. Load cached models or train ───────────────────────
    cache_files = ["gbm.pkl", "rf.pkl", "features.pkl", "encoders.pkl"]
    if all(os.path.exists(MODEL_PATH + f) for f in cache_files):
        model_gbm    = joblib.load(MODEL_PATH + "gbm.pkl")
        model_rf     = joblib.load(MODEL_PATH + "rf.pkl")
        FEATURE_COLS = joblib.load(MODEL_PATH + "features.pkl")
        ENCODERS.update(joblib.load(MODEL_PATH + "encoders.pkl"))
        print("✅ Cached models loaded.")
    else:
        print("⚙️  No cached models — training now (first-run only) …")
        FEATURE_COLS[:] = REQUIRED_FEATURES      # in-place update
        df_fe = _engineer_for_training(df_kb.copy())
        train = df_fe.dropna(subset=REQUIRED_FEATURES)
        X, y  = train[REQUIRED_FEATURES], train["unit_sales"]
        print(f"   Training on {len(X):,} rows × {len(REQUIRED_FEATURES)} features")

        gbm = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.04, max_depth=8,
            min_samples_leaf=15, l2_regularization=0.1, random_state=42)
        gbm.fit(X, y)

        rf = RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=8,
            random_state=42, n_jobs=-1)
        rf.fit(X, y)

        os.makedirs(MODEL_PATH, exist_ok=True)
        joblib.dump(gbm,              MODEL_PATH + "gbm.pkl")
        joblib.dump(rf,               MODEL_PATH + "rf.pkl")
        joblib.dump(REQUIRED_FEATURES, MODEL_PATH + "features.pkl")
        joblib.dump(ENCODERS,          MODEL_PATH + "encoders.pkl")

        model_gbm    = gbm
        model_rf     = rf
        FEATURE_COLS[:] = REQUIRED_FEATURES
        print("✅ Models trained and saved.")


# ══════════════════════════════════════════════════════════════
#  PIPELINE — STEP 1 : data availability check
# ══════════════════════════════════════════════════════════════
def _check_availability(item_nbr: int, log: list) -> dict:
    log.append("📂 [Step 1] Checking knowledge base for item history …")

    hist = df_kb[df_kb["item_nbr"] == item_nbr].sort_values("date")

    if len(hist) == 0:
        log.append(f"   ❌ item_nbr={item_nbr} NOT found.")
        return {"available": False, "reason": f"item_nbr {item_nbr} not found in knowledge base"}

    n     = len(hist)
    dmin  = hist["date"].min().strftime("%Y-%m-%d")
    dmax  = hist["date"].max().strftime("%Y-%m-%d")
    span  = (hist["date"].max() - hist["date"].min()).days

    missing = []
    for lag_days, name in [(1,"lag_1"),(7,"lag_7"),(14,"lag_14"),(28,"lag_28"),(365,"lag_365")]:
        if span < lag_days:
            missing.append(name)

    log.append(f"   ✅ {n} rows  |  {dmin} → {dmax}  ({span} days history)")
    if missing:
        log.append(f"   ⚠️  Insufficient history for: {', '.join(missing)} (NaN → model uses imputation)")

    return {
        "available":        True,
        "row_count":        n,
        "date_range":       f"{dmin} → {dmax}",
        "days_of_history":  span,
        "missing_lags":     missing,
    }


# ══════════════════════════════════════════════════════════════
#  PIPELINE — STEP 2 : build one feature vector
# ══════════════════════════════════════════════════════════════
def _build_feature_vector(target: pd.Timestamp,
                           item_nbr: int,
                           onpromotion: int,
                           hist: pd.DataFrame) -> pd.DataFrame:
    last = hist.iloc[-1] if len(hist) > 0 else pd.Series(dtype=float)

    # ── lag helpers ───────────────────────────────────────────
    def lag(d: int) -> float:
        dt  = target - timedelta(days=d)
        row = hist[hist["date"] == dt]
        return float(row["unit_sales"].values[0]) if len(row) > 0 else np.nan

    # ── rolling helpers ───────────────────────────────────────
    def roll_mean(d: int) -> float:
        mask = (hist["date"] >= target - timedelta(days=d + 1)) & (hist["date"] < target)
        v    = hist.loc[mask, "unit_sales"]
        return float(v.mean()) if len(v) > 0 else np.nan

    def roll_std(d: int) -> float:
        mask = (hist["date"] >= target - timedelta(days=d + 1)) & (hist["date"] < target)
        v    = hist.loc[mask, "unit_sales"]
        return float(v.std()) if len(v) > 1 else 0.0

    # ── EWM helper ────────────────────────────────────────────
    def ewm(span: int) -> float:
        v = hist.loc[hist["date"] < target, "unit_sales"]
        return float(v.ewm(span=span).mean().iloc[-1]) if len(v) > 0 else np.nan

    woy = int(target.isocalendar()[1])

    feat = {
        # identity
        "item_nbr":          int(item_nbr),
        "class":             int(last.get("class", 2712)),
        "cluster":           int(last.get("cluster", 13)),
        "onpromotion":       int(onpromotion),
        "transactions":      float(last.get("transactions", 1400)),
        # date components
        "dayofweek":         int(target.dayofweek),
        "month":             int(target.month),
        "year":              int(target.year),
        "weekofyear":        woy,
        "day":               int(target.day),
        "quarter":           int(target.quarter),
        "dayofyear":         int(target.day_of_year),
        "is_weekend":        int(target.dayofweek >= 5),
        "is_month_start":    int(target.is_month_start),
        "is_month_end":      int(target.is_month_end),
        "is_quarter_end":    int(target.is_quarter_end),
        # cyclical
        "month_sin":  np.sin(2 * np.pi * target.month    / 12),
        "month_cos":  np.cos(2 * np.pi * target.month    / 12),
        "dow_sin":    np.sin(2 * np.pi * target.dayofweek /  7),
        "dow_cos":    np.cos(2 * np.pi * target.dayofweek /  7),
        "woy_sin":    np.sin(2 * np.pi * woy              / 52),
        "woy_cos":    np.cos(2 * np.pi * woy              / 52),
        # lag features
        "lag_1":   lag(1),
        "lag_7":   lag(7),
        "lag_14":  lag(14),
        "lag_28":  lag(28),
        "lag_365": lag(365),
        # rolling statistics
        "rolling_mean_7":   roll_mean(7),
        "rolling_std_7":    roll_std(7),
        "rolling_mean_14":  roll_mean(14),
        "rolling_std_14":   roll_std(14),
        "rolling_mean_28":  roll_mean(28),
        "rolling_std_28":   roll_std(28),
        "rolling_mean_90":  roll_mean(90),
        # EWM
        "ewm_7":   ewm(7),
        "ewm_28":  ewm(28),
        "ewm_90":  ewm(90),
        # encoded categoricals
        "family_enc": int(ENCODERS.get("family", {}).get(str(last.get("family", "BREAD/BAKERY")), 0)),
        "city_enc":   int(ENCODERS.get("city",   {}).get(str(last.get("city",   "Quito")),        0)),
        "state_enc":  int(ENCODERS.get("state",  {}).get(str(last.get("state",  "Pichincha")),    0)),
        "type_x_enc": int(ENCODERS.get("type_x", {}).get(str(last.get("type_x", "D")),            0)),
    }

    # fill any column the model expects but is missing
    for col in FEATURE_COLS:
        if col not in feat:
            feat[col] = 0

    return pd.DataFrame([feat])[FEATURE_COLS]


# ══════════════════════════════════════════════════════════════
#  PIPELINE — STEP 3 : reorder alert
# ══════════════════════════════════════════════════════════════
def _reorder_alert(no_promo_rows: list,
                   current_stock: float,
                   lead_time: int,
                   log: list) -> dict:
    """
    ROP  = avg_daily_sales × lead_time + safety_stock
    SS   = Z × σ_daily × √(lead_time)        [Z=1.65 → 95 % service level]
    """
    log.append("📦 [Step 4] Computing reorder alert …")

    vals       = [r["predicted_sales"] for r in no_promo_rows]
    avg_daily  = float(np.mean(vals))
    std_daily  = float(np.std(vals))
    ss         = round(Z_SCORE * std_daily * (lead_time ** 0.5), 2)
    rop        = round(avg_daily * lead_time + ss, 2)
    stockout_d = round(current_stock / avg_daily, 1) if avg_daily > 0 else 9999
    triggered  = current_stock <= rop

    if triggered:
        tag = "🔴 CRITICAL" if stockout_d <= lead_time else "⚠️  WARNING"
        msg = (f"{tag} — Place order NOW.  "
               f"Stock ({current_stock:.0f}) ≤ ROP ({rop:.0f}).  "
               f"Estimated stockout in ~{stockout_d:.0f} days.")
    else:
        days_buffer = round((current_stock - rop) / avg_daily, 1) if avg_daily > 0 else 9999
        msg = (f"✅ STOCK OK — {current_stock:.0f} units on hand.  "
               f"ROP = {rop:.0f}.  "
               f"Reorder in ~{days_buffer:.0f} days.")

    log.append(f"   avg={avg_daily:.3f}  std={std_daily:.3f}  SS={ss}  "
               f"ROP={rop}  stockout_in={stockout_d}d  triggered={triggered}")

    return {
        "triggered":           triggered,
        "current_stock":       current_stock,
        "reorder_point":       rop,
        "safety_stock":        ss,
        "avg_daily_sales":     round(avg_daily, 4),
        "std_daily_sales":     round(std_daily, 4),
        "days_until_stockout": stockout_d,
        "lead_time_days":      lead_time,
        "service_level_pct":   95,
        "message":             msg,
    }


# ══════════════════════════════════════════════════════════════
#  MASTER PIPELINE
# ══════════════════════════════════════════════════════════════
def run_pipeline(item_nbr_in, item_name_in, date_str,
                 days, onpromotion_in,
                 current_stock, lead_time_days) -> dict:
    log    = []
    errors = []

    try:
        # ── Step 0: resolve item ─────────────────────────────
        log.append("🔍 [Step 0] Resolving item identifier …")
        item_nbr = None

        raw_nbr = str(item_nbr_in).strip() if item_nbr_in else ""
        if raw_nbr.lstrip("-").isdigit():
            item_nbr = int(raw_nbr)
            log.append(f"   item_nbr={item_nbr} (numeric input)")

        if item_nbr is None and item_name_in and str(item_name_in).strip():
            name   = str(item_name_in).strip().upper()
            exact  = df_kb[df_kb["family"].str.upper() == name]["item_nbr"].unique()
            if len(exact) > 0:
                item_nbr = int(exact[0])
                log.append(f"   Exact family match '{item_name_in}' → item_nbr={item_nbr}")
            else:
                partial = df_kb[df_kb["family"].str.upper().str.contains(name, na=False)][
                    "item_nbr"].unique()
                if len(partial) > 0:
                    item_nbr = int(partial[0])
                    log.append(f"   Partial family match '{item_name_in}' → item_nbr={item_nbr}")

        if item_nbr is None:
            sample = sorted(df_kb["item_nbr"].unique().tolist())[:20]
            return {
                "status":  "error",
                "message": "Cannot resolve item. Provide a valid item_nbr or item_name (family).",
                "available_items_sample": sample,
                "pipeline_log": log, "errors": ["item_not_found"],
            }

        # parse inputs
        try:
            start_date = pd.to_datetime(date_str.strip())
        except Exception:
            return {"status": "error", "message": "Invalid date. Use YYYY-MM-DD.",
                    "pipeline_log": log, "errors": ["invalid_date"]}

        days       = max(1, min(int(days), 90))
        promo_flag = bool(onpromotion_in)
        stock      = float(current_stock)  if current_stock  else 0.0
        lead       = int(lead_time_days)   if lead_time_days else DEFAULT_LEAD

        log.append(f"   date={start_date.date()}  days={days}  "
                   f"onpromotion={promo_flag}  stock={stock}  lead={lead}d")

        # ── Step 1: data availability ────────────────────────
        avail = _check_availability(item_nbr, log)
        if not avail["available"]:
            return {
                "status": "error", "data_available": False,
                "message": avail["reason"],
                "pipeline_log": log, "errors": ["no_history"],
            }

        # ── Step 2: fetch history ────────────────────────────
        log.append("📋 [Step 2] Fetching item history …")
        hist      = df_kb[df_kb["item_nbr"] == item_nbr].sort_values("date").copy()
        item_name = str(hist["family"].iloc[-1]) if "family" in hist.columns else str(item_name_in or "")
        log.append(f"   {len(hist)} rows retrieved for item {item_nbr} ({item_name})")

        # ── Step 3: feature computation + prediction ─────────
        log.append(f"🧮 [Step 3] Computing {len(FEATURE_COLS)} features "
                   f"× {days} days × 2 promo scenarios …")

        rows_on  = []   # promo = 1
        rows_off = []   # promo = 0

        for i in range(days):
            d = start_date + timedelta(days=i)
            for p_val, bucket in [(1, rows_on), (0, rows_off)]:
                fv     = _build_feature_vector(d, item_nbr, p_val, hist)
                p_gbm  = model_gbm.predict(fv)[0]
                p_rf   = model_rf.predict(fv)[0]
                p_ens  = 0.60 * p_gbm + 0.40 * p_rf
                sales  = round(max(0.0, float(np.expm1(p_ens))), 4)
                conf   = "High" if i <= 7 else ("Medium" if i <= 30 else "Low")

                bucket.append({
                    "date":            d.strftime("%Y-%m-%d"),
                    "day_of_week":     d.strftime("%A"),
                    "day_index":       i + 1,
                    "onpromotion":     bool(p_val),
                    "predicted_sales": sales,
                    "confidence":      conf,
                    # expose key features used
                    "lag_1_used":       _safe(fv, "lag_1"),
                    "lag_7_used":       _safe(fv, "lag_7"),
                    "rolling_7_used":   _safe(fv, "rolling_mean_7"),
                    "rolling_14_used":  _safe(fv, "rolling_mean_14"),
                    "rolling_28_used":  _safe(fv, "rolling_mean_28"),
                })

        log.append(f"   ✅ {len(rows_on)} predictions per scenario")

        # ── Summary ──────────────────────────────────────────
        total_on  = round(sum(r["predicted_sales"] for r in rows_on),  3)
        total_off = round(sum(r["predicted_sales"] for r in rows_off), 3)
        avg_on    = round(total_on  / days, 4)
        avg_off   = round(total_off / days, 4)
        lift_pct  = round((total_on - total_off) / total_off * 100, 2) if total_off > 0 else 0.0

        log.append(f"   promo_on_total={total_on}  promo_off_total={total_off}  lift={lift_pct}%")

        # ── Step 4: reorder alert ────────────────────────────
        reorder = None
        if stock > 0:
            reorder = _reorder_alert(rows_off, stock, lead, log)
        else:
            log.append("ℹ️  [Step 4] current_stock=0 — reorder alert skipped")

        # ── Build response ───────────────────────────────────
        log.append("📤 [Step 5] Assembling response …")

        return {
            "status":         "success",
            "item_nbr":       item_nbr,
            "item_name":      item_name,
            "forecast_from":  start_date.strftime("%Y-%m-%d"),
            "forecast_days":  days,
            "data_available": True,
            "data_info":      avail,
            "summary": {
                "total_sales_with_promo":    total_on,
                "total_sales_without_promo": total_off,
                "daily_avg_with_promo":      avg_on,
                "daily_avg_without_promo":   avg_off,
                "promo_lift_pct":            lift_pct,
                "requested_onpromotion":     promo_flag,
                "recommended_forecast":      rows_on if promo_flag else rows_off,
            },
            "promotion_on":  rows_on,
            "promotion_off": rows_off,
            "reorder_alert": reorder,
            "pipeline_log":  log,
            "errors":        errors,
        }

    except Exception as exc:
        log.append(f"❌ PIPELINE EXCEPTION: {exc}")
        return {
            "status":       "error",
            "message":      str(exc),
            "traceback":    traceback.format_exc(),
            "pipeline_log": log,
            "errors":       [str(exc)],
        }


def _safe(df_row: pd.DataFrame, col: str):
    """Return rounded float or None from a single-row dataframe."""
    try:
        val = df_row[col].iloc[0]
        return None if pd.isna(val) else round(float(val), 4)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════
#  GRADIO WRAPPER  (called by UI and /run/predict API)
# ══════════════════════════════════════════════════════════════
def gradio_predict(item_nbr_str, item_name_str, date_str,
                   forecast_days, onpromotion,
                   current_stock, lead_time_days):
    """
    Returns:
      [0] json_str   — full structured response (for backend / API consumers)
      [1] summary_md — human-readable markdown (for Gradio UI)
      [2] table_df   — day-by-day dataframe    (for Gradio UI)
    """
    result   = run_pipeline(item_nbr_str, item_name_str, date_str,
                            forecast_days, onpromotion,
                            current_stock, lead_time_days)
    json_str = json.dumps(result, indent=2, default=str)

    # ── markdown summary ──────────────────────────────────────
    if result["status"] == "error":
        md = f"### ❌ Error\n\n{result.get('message', 'Unknown error')}"
        return json_str, md, None

    s  = result["summary"]
    ra = result.get("reorder_alert") or {}

    md = f"""### ✅ Forecast — Item `{result['item_nbr']}` ({result['item_name']})

**Period:** {result['forecast_from']}  +  {result['forecast_days']} days
**History available:** {result['data_info']['date_range']}  ({result['data_info']['row_count']} rows)

| | With Promotion | Without Promotion |
|---|---|---|
| **Total Sales** | {s['total_sales_with_promo']} units | {s['total_sales_without_promo']} units |
| **Daily Average** | {s['daily_avg_with_promo']} units | {s['daily_avg_without_promo']} units |
| **Promo Lift** | **+{s['promo_lift_pct']}%** | — |

---
**📦 Reorder Alert:** {ra.get('message', 'N/A (no stock level provided)')}
"""

    # ── day-by-day table ──────────────────────────────────────
    rows = s["recommended_forecast"]
    df_out = pd.DataFrame([{
        "Date":           r["date"],
        "Day":            r["day_of_week"],
        "Forecast (units)": r["predicted_sales"],
        "Promo":          "✓" if r["onpromotion"] else "✗",
        "Confidence":     r["confidence"],
        "lag_1":          r.get("lag_1_used"),
        "lag_7":          r.get("lag_7_used"),
        "rolling_7":      r.get("rolling_7_used"),
        "rolling_14":     r.get("rolling_14_used"),
    } for r in rows])

    return json_str, md, df_out


# ══════════════════════════════════════════════════════════════
#  GRADIO UI
# ══════════════════════════════════════════════════════════════
def build_ui():
    with gr.Blocks(
        title="Favorita Sales Forecaster API",
        theme=gr.themes.Soft(primary_hue="teal"),
        css="""
        #json-box textarea { font-family: monospace !important; font-size: 12px !important; }
        .header-block { text-align: center; padding: 20px 0 8px; }
        """,
    ) as demo:

        # ── header ────────────────────────────────────────────
        gr.HTML("""
        <div class="header-block">
          <h1 style="color:#0d9488;font-size:1.85rem;margin:0">
            🛒 Favorita Sales Forecaster
          </h1>
          <p style="color:#6b7280;margin:6px 0 4px">
            ML pipeline API &nbsp;·&nbsp; item + date + promotion → sales forecast + reorder alert
          </p>
          <code style="background:#f3f4f6;padding:3px 10px;border-radius:6px;font-size:0.78rem">
            POST &nbsp;/run/predict &nbsp;|&nbsp; gradio_client.predict() &nbsp;|&nbsp; requests.post()
          </code>
        </div>
        """)

        with gr.Tabs():

            # ────────────────────────────────────────────────
            # TAB 1 — Interactive UI
            # ────────────────────────────────────────────────
            with gr.Tab("🖥️  Forecast UI"):
                with gr.Row():

                    # inputs
                    with gr.Column(scale=1, min_width=280):
                        gr.Markdown("#### 📥 Request")
                        inp_nbr   = gr.Textbox(label="item_nbr",
                                               placeholder="e.g. 103665")
                        inp_name  = gr.Textbox(label="item_name / family",
                                               placeholder="e.g. BREAD/BAKERY")
                        inp_date  = gr.Textbox(label="date (YYYY-MM-DD)",
                                               value="2017-01-15")
                        inp_days  = gr.Slider(label="days", minimum=1,
                                              maximum=90, step=1, value=30)
                        inp_promo = gr.Checkbox(label="onpromotion", value=False)
                        inp_stock = gr.Number(label="current_stock (optional)", value=150)
                        inp_lead  = gr.Number(label="lead_time_days", value=7)
                        btn_run   = gr.Button("🚀 Run Forecast",
                                              variant="primary", size="lg")

                    # outputs
                    with gr.Column(scale=2):
                        gr.Markdown("#### 📤 Response")
                        out_md    = gr.Markdown()
                        out_table = gr.Dataframe(
                            label="Day-by-Day Forecast",
                            headers=["Date","Day","Forecast (units)","Promo",
                                     "Confidence","lag_1","lag_7",
                                     "rolling_7","rolling_14"],
                            interactive=False,
                            wrap=True,
                        )

                # hidden json output (still returned for API)
                _hidden_json = gr.Textbox(visible=False)

                btn_run.click(
                    fn=gradio_predict,
                    inputs=[inp_nbr, inp_name, inp_date, inp_days,
                            inp_promo, inp_stock, inp_lead],
                    outputs=[_hidden_json, out_md, out_table],
                )

            # ────────────────────────────────────────────────
            # TAB 2 — Raw JSON API
            # ────────────────────────────────────────────────
            with gr.Tab("🔌 JSON API"):
                gr.Markdown(
                    "Send a request and see the **full JSON response** "
                    "exactly as your backend receives it."
                )
                with gr.Row():
                    with gr.Column(scale=1, min_width=280):
                        gr.Markdown("#### Parameters")
                        j_nbr   = gr.Textbox(label="item_nbr",  placeholder="103665")
                        j_name  = gr.Textbox(label="item_name", placeholder="BREAD/BAKERY")
                        j_date  = gr.Textbox(label="date",       value="2017-01-15")
                        j_days  = gr.Slider(label="days", minimum=1, maximum=30,
                                            step=1, value=7)
                        j_promo = gr.Checkbox(label="onpromotion", value=False)
                        j_stock = gr.Number(label="current_stock", value=150)
                        j_lead  = gr.Number(label="lead_time_days", value=7)
                        j_btn   = gr.Button("📡 Get JSON", variant="primary")

                    with gr.Column(scale=2):
                        j_out = gr.Code(
                            label="Response JSON",
                            language="json",
                            elem_id="json-box",
                            lines=45,
                        )

                j_btn.click(
                    fn=lambda a, b, c, d, e, f, g: gradio_predict(a, b, c, d, e, f, g)[0],
                    inputs=[j_nbr, j_name, j_date, j_days, j_promo, j_stock, j_lead],
                    outputs=j_out,
                )

            # ────────────────────────────────────────────────
            # TAB 3 — API Docs
            # ────────────────────────────────────────────────
            with gr.Tab("📖 API Docs"):
                gr.Markdown("""
## REST API Reference

### Endpoint
```
POST https://YOUR_USERNAME-favorita-forecaster.hf.space/run/predict
Content-Type: application/json
```

### Request body
```json
{
  "data": [
    "103665",        // [0] item_nbr  — string (or empty string "")
    "BREAD/BAKERY",  // [1] item_name — family name (or empty string)
    "2017-03-15",    // [2] date      — YYYY-MM-DD
    30,              // [3] days      — integer 1–90
    true,            // [4] onpromotion — bool
    150,             // [5] current_stock — float (0 = skip reorder alert)
    7                // [6] lead_time_days — integer
  ]
}
```

> At least one of `item_nbr` or `item_name` must be provided.

---

### Response fields

| Field | Type | Description |
|---|---|---|
| `status` | str | `"success"` or `"error"` |
| `item_nbr` | int | Resolved item number |
| `item_name` | str | Product family |
| `data_available` | bool | Was history found? |
| `data_info.row_count` | int | Rows in knowledge base for this item |
| `data_info.date_range` | str | e.g. `"2013-01-01 → 2016-12-31"` |
| `data_info.missing_lags` | list | Which lags had insufficient history |
| `summary.total_sales_with_promo` | float | Sum over forecast window, promo=1 |
| `summary.total_sales_without_promo` | float | Sum over forecast window, promo=0 |
| `summary.promo_lift_pct` | float | `%` uplift from promotion |
| `summary.recommended_forecast` | list | Rows for the requested promo scenario |
| `promotion_on` | list | All days with `onpromotion=true` |
| `promotion_off` | list | All days with `onpromotion=false` |
| `reorder_alert.triggered` | bool | `true` = place order now |
| `reorder_alert.reorder_point` | float | ROP = avg_daily × lead + safety_stock |
| `reorder_alert.safety_stock` | float | Z × σ × √lead_time |
| `reorder_alert.days_until_stockout` | float | Estimated days remaining |
| `reorder_alert.message` | str | Human-readable alert |
| `pipeline_log` | list[str] | Step-by-step execution trace |

---

### Python — requests
```python
import requests, json

resp = requests.post(
    "https://YOUR_USERNAME-favorita-forecaster.hf.space/run/predict",
    json={"data": ["103665", "BREAD/BAKERY", "2017-03-15", 30, True, 150, 7]},
    timeout=120,
)
result = json.loads(resp.json()["data"][0])

# reorder alert
print(result["reorder_alert"]["message"])

# day-by-day forecast (requested promo scenario)
for row in result["summary"]["recommended_forecast"][:7]:
    print(row["date"], row["predicted_sales"], row["confidence"])
```

### Python — gradio_client
```python
from gradio_client import Client
import json

client = Client("YOUR_USERNAME/favorita-forecaster")
json_str, _, _ = client.predict(
    item_nbr_str   = "103665",
    item_name_str  = "BREAD/BAKERY",
    date_str       = "2017-03-15",
    forecast_days  = 30,
    onpromotion    = True,
    current_stock  = 150,
    lead_time_days = 7,
    api_name       = "/gradio_predict",
)
result = json.loads(json_str)
print(result["summary"]["promo_lift_pct"], "%")
```
                """)

    return demo


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print(" Favorita Sales Forecaster — starting up")
print("=" * 60)
load_everything()
print("🎯 Building Gradio interface …")
demo = build_ui()

if __name__ == "__main__":
    demo.launch()