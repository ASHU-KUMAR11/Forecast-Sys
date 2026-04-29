#!/usr/bin/env python3
"""
Run this LOCALLY before pushing to Hugging Face.
Trains models on your knowledge_base.csv and saves .pkl files.

Usage:
    python train_offline.py --data knowledge_base.csv
"""
import argparse, os, joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

ENCODERS     = {}
FEATURE_COLS = [
    "item_nbr","class","cluster","onpromotion","transactions",
    "dayofweek","month","year","weekofyear","day","quarter","dayofyear",
    "is_weekend","is_month_start","is_month_end","is_quarter_end",
    "month_sin","month_cos","dow_sin","dow_cos","woy_sin","woy_cos",
    "lag_1","lag_7","lag_14","lag_28","lag_365",
    "rolling_mean_7","rolling_std_7","rolling_mean_14","rolling_std_14",
    "rolling_mean_28","rolling_std_28","rolling_mean_90",
    "ewm_7","ewm_28","ewm_90",
    "family_enc","city_enc","state_enc","type_x_enc"
]

def engineer(df):
    df = df.sort_values(["item_nbr","date"]).reset_index(drop=True)
    df["day"]        = df["date"].dt.day
    df["month"]      = df["date"].dt.month
    df["year"]       = df["date"].dt.year
    df["dayofweek"]  = df["date"].dt.dayofweek
    df["dayofyear"]  = df["date"].dt.day_of_year
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"]    = df["date"].dt.quarter
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_start"]  = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"]    = df["date"].dt.is_month_end.astype(int)
    df["is_quarter_end"]  = df["date"].dt.is_quarter_end.astype(int)

    df["month_sin"] = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"] = np.cos(2*np.pi*df["month"]/12)
    df["dow_sin"]   = np.sin(2*np.pi*df["dayofweek"]/7)
    df["dow_cos"]   = np.cos(2*np.pi*df["dayofweek"]/7)
    df["woy_sin"]   = np.sin(2*np.pi*df["weekofyear"]/52)
    df["woy_cos"]   = np.cos(2*np.pi*df["weekofyear"]/52)

    for lag in [1,7,14,28,365]:
        df[f"lag_{lag}"] = df.groupby("item_nbr")["unit_sales"].shift(lag)

    for win in [7,14,28,90]:
        df[f"rolling_mean_{win}"] = df.groupby("item_nbr")["unit_sales"].transform(
            lambda x: x.shift(1).rolling(win, min_periods=1).mean())
        df[f"rolling_std_{win}"]  = df.groupby("item_nbr")["unit_sales"].transform(
            lambda x: x.shift(1).rolling(win, min_periods=1).std())

    for sp in [7,28,90]:
        df[f"ewm_{sp}"] = df.groupby("item_nbr")["unit_sales"].transform(
            lambda x: x.shift(1).ewm(span=sp).mean())

    for col in ["family","city","state","type_x"]:
        le = LabelEncoder()
        df[col+"_enc"] = le.fit_transform(df[col].astype(str))
        ENCODERS[col]  = dict(zip(le.classes_, le.transform(le.classes_)))

    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="knowledge_base.csv")
    args = parser.parse_args()

    print(f"📂 Loading {args.data}...")
    df = pd.read_csv(args.data, parse_dates=["date"])

    # Apply log1p if unit_sales is in raw scale (skip if already log-transformed)
    if df["unit_sales"].max() > 20:
        df["unit_sales"] = np.log1p(df["unit_sales"].clip(0))
        print("Applied log1p transform to unit_sales")

    df["onpromotion"] = df["onpromotion"].map(
        {True:1,False:0,"TRUE":1,"FALSE":0,1:1,0:0}).fillna(0).astype(int)

    print("🔧 Engineering features...")
    df_fe = engineer(df)
    train = df_fe.dropna(subset=FEATURE_COLS)
    X, y  = train[FEATURE_COLS], train["unit_sales"]
    print(f"✅ Training on {len(X):,} rows × {len(FEATURE_COLS)} features")

    print("🚀 Training HistGradientBoosting (LightGBM-equivalent)...")
    gbm = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.04, max_depth=8,
        min_samples_leaf=15, l2_regularization=0.1, random_state=42)
    gbm.fit(X, y)

    print("🌲 Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=200, max_depth=12, min_samples_leaf=8,
        random_state=42, n_jobs=-1)
    rf.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(gbm,          "models/gbm.pkl")
    joblib.dump(rf,           "models/rf.pkl")
    joblib.dump(FEATURE_COLS, "models/features.pkl")
    joblib.dump(ENCODERS,     "models/encoders.pkl")

    print("\n✅ All models saved!")
    print("   models/gbm.pkl · models/rf.pkl · models/features.pkl · models/encoders.pkl")
    print("\nNext: git add . && git commit -m '🚀 deploy' && git push")

if __name__ == "__main__":
    main()