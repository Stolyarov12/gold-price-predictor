import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime

app = FastAPI(title="Gold Price API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загружаем модель при старте
MODEL_PATH = os.getenv("MODEL_PATH", "/app/ml/artifacts/prophet_model.pkl")
DF_PATH    = os.getenv("DF_PATH",    "/app/ml/artifacts/gold_df.pkl")
FEAT_PATH  = os.getenv("FEAT_PATH",  "/app/ml/artifacts/feature_cols.pkl")

try:
    model       = joblib.load(MODEL_PATH)
    df          = joblib.load(DF_PATH)
    feature_cols = joblib.load(FEAT_PATH)
    print("✅ Модель загружена!")
except Exception as e:
    model = None
    df    = None
    feature_cols = None
    print(f"⚠️ Модель не найдена: {e}")
    import traceback
    traceback.print_exc()


# ЭНДПОИНТЫ


@app.get("/")
def root():
    return {"status": "ok", "service": "Gold Price API"}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/price/history")
def get_history(days: int = 90):
    """Последние N дней исторических цен"""
    if df is None:
        return {"error": "Модель не загружена"}
    history = df['Price'].tail(days).reset_index()
    history.columns = ['date', 'price']
    history['date'] = history['date'].astype(str)
    return {"data": history.to_dict(orient='records')}


@app.get("/price/predict")
def predict_tomorrow():
    if model is None or df is None:
        return {"error": "Модель не загружена. Запустите train.py"}

    try:
        price_buffer = list(df['Price'].values)

        # Берём последнюю дату из обучающих данных + 1 день
        last_train_date = df.index[-1]
        next_date = last_train_date + pd.Timedelta(days=1)

        row = {
            'ds':        next_date,
            'Lag_1':     price_buffer[-1],
            'MA_7':      np.mean(price_buffer[-7:]),
            'MA_21':     np.mean(price_buffer[-21:]),
            'RSI':       _calc_rsi(price_buffer),
            'MACD':      _calc_macd(price_buffer),
            'Sentiment': 0.0,
            'Return_1d': (price_buffer[-1] - price_buffer[-2]) / price_buffer[-2],
        }

        X = pd.DataFrame([row])
        forecast = model.predict(X)

        last_price = float(price_buffer[-1])
        pred_price = float(forecast['yhat'].values[0])
        pred_lower = float(forecast['yhat_lower'].values[0])
        pred_upper = float(forecast['yhat_upper'].values[0])
        change_pct = (pred_price - last_price) / last_price * 100

        return {
            "last_price":   round(last_price, 2),
            "predicted":    round(pred_price, 2),
            "lower":        round(pred_lower, 2),
            "upper":        round(pred_upper, 2),
            "change_pct":   round(change_pct, 2),
            "direction":    "📈 Рост" if change_pct > 0 else "📉 Падение",
            "last_date":    str(last_train_date.date()),
            "predict_date": str(next_date.date())
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/price/forecast")
def forecast_future(days: int = 30):
    """Прогноз на N дней вперёд"""
    if model is None or df is None:
        return {"error": "Модель не загружена"}

    try:
        price_buffer = list(df['Price'].values)
        future_dates = pd.bdate_range(start=df.index[-1], periods=days + 1)[1:]
        records = []

        for date in future_dates:
            row = {
                'ds':        date,
                'Lag_1':     price_buffer[-1],
                'Lag_3':     price_buffer[-3],
                'Lag_5':     price_buffer[-5],
                'Lag_10':    price_buffer[-10],
                'MA_7':      np.mean(price_buffer[-7:]),
                'MA_21':     np.mean(price_buffer[-21:]),
                'RSI':       _calc_rsi(price_buffer),
                'MACD':      _calc_macd(price_buffer),
                'Sentiment': 0.0,
                'Return_1d': (price_buffer[-1] - price_buffer[-2]) / price_buffer[-2],
            }
            X = pd.DataFrame([row])
            f = model.predict(X)
            p = float(f['yhat'].values[0])
            records.append({'date': str(date.date()), 'price': round(p, 2)})
            price_buffer.append(p)

        return {"data": records}
    except Exception as e:
        return {"error": str(e)}

from fastapi.responses import FileResponse

@app.get("/plots/{plot_name}")
def get_plot(plot_name: str):
    """Возвращает график как изображение"""
    path = f"/app/ml/artifacts/plots/{plot_name}"
    if os.path.exists(path):
        return FileResponse(path, media_type="image/png")
    return {"error": "График не найден"}

# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ

def _calc_rsi(prices, period=14):
    s = pd.Series(prices[-period*2:])
    delta = s.diff()
    gain  = delta.clip(lower=0).rolling(period).mean().iloc[-1]
    loss  = (-delta.clip(upper=0)).rolling(period).mean().iloc[-1]
    return float(100 - (100 / (1 + gain / (loss + 1e-9))))

def _calc_macd(prices):
    s    = pd.Series(prices[-60:])
    macd = s.ewm(span=12).mean().iloc[-1] - s.ewm(span=26).mean().iloc[-1]
    return float(macd)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)