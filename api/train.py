import os
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from newsapi import NewsApiClient
from transformers import pipeline
from prophet import Prophet
from datetime import datetime, timedelta

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "ВАШ_КЛЮЧ_ЗДЕСЬ")

# ИСТОЧНИК 1: Цены золота (yfinance)

def load_gold_prices():
    print("Загружаем цены золота...")
    gold = yf.download('GC=F', start='2019-01-01', end='2026-06-24', auto_adjust=True)
    df = gold[['Close']].copy()
    df.columns = ['Price']
    df.index = pd.to_datetime(df.index)
    return df

# ИСТОЧНИК 2: Новости (NewsAPI + FinBERT)

def load_sentiment():
    print("Используем нейтральный sentiment (FinBERT отключён для скорости)...")
    return pd.DataFrame(columns=['Sentiment'])


# FEATURE ENGINEERING

def build_features(df, sentiment_df):
    for lag in [1, 3, 5, 10]:
        df[f'Lag_{lag}'] = df['Price'].shift(lag)
    for w in [7, 21, 50]:
        df[f'MA_{w}']  = df['Price'].shift(1).rolling(w).mean()
        df[f'STD_{w}'] = df['Price'].shift(1).rolling(w).std()

    df['Return_1d'] = df['Price'].pct_change().shift(1)
    df['Return_5d'] = df['Price'].pct_change(5).shift(1)

    price_sh = df['Price'].shift(1)
    delta = price_sh.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI']  = 100 - (100 / (1 + gain / loss.replace(0, 1e-9)))
    df['MACD'] = price_sh.ewm(span=12).mean() - price_sh.ewm(span=26).mean()

    if not sentiment_df.empty:
        df = df.join(sentiment_df, how='left')
        df['Sentiment'] = df['Sentiment'].fillna(0)
    else:
        df['Sentiment'] = 0.0

    df.dropna(inplace=True)
    return df


# ОБУЧЕНИЕ PROPHET

def train_model(df):
    print("Обучаем Prophet...")
    feature_cols = ['Lag_1', 'MA_7', 'MA_21', 'RSI', 'MACD', 'Sentiment', 'Return_1d']

    prophet_df = df.reset_index()[['Date', 'Price']].rename(
        columns={'Date': 'ds', 'Price': 'y'}
    )
    for col in feature_cols:
        prophet_df[col] = df[col].values

    model = Prophet(
        changepoint_prior_scale=0.05,
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False
    )
    for col in feature_cols:
        model.add_regressor(col)

    model.fit(prophet_df)
    return model, df, feature_cols


# СОХРАНЕНИЕ

if __name__ == '__main__':
    os.makedirs('ml/artifacts', exist_ok=True)

    df = load_gold_prices()
    sentiment_df = load_sentiment()
    df = build_features(df, sentiment_df)
    model, df, feature_cols = train_model(df)

    joblib.dump(model, 'ml/artifacts/prophet_model.pkl')
    joblib.dump(df, 'ml/artifacts/gold_df.pkl')
    joblib.dump(feature_cols, 'ml/artifacts/feature_cols.pkl')

    print("✅ Модель сохранена в ml/artifacts/")