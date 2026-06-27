import os
import joblib
import pandas as pd
import numpy as np
import yfinance as yf
from newsapi import NewsApiClient
from transformers import pipeline
from prophet import Prophet
from datetime import datetime, timedelta

NEWS_API_KEY = os.getenv("NEWS_API_KEY", "3894c33e01854edf8061d194bc5ff623")

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
    print("Загружаем sentiment из новостей через FinBERT...")
    
    # Если уже есть сохранённый файл — используем его
    sentiment_cache = 'ml/artifacts/sentiment_cache.csv'
    
    try:
        import os
        os.environ['TRANSFORMERS_NO_PYTORCH'] = '0'
        from newsapi import NewsApiClient
        from transformers import pipeline

        NEWS_API_KEY = os.getenv("NEWS_API_KEY", "ВАШ_КЛЮЧ")
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)

        print("Загружаем FinBERT...")
        sentiment_pipe = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert"
        )

        records = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=28)
        current = start_date

        while current < end_date:
            next_day = current + timedelta(days=1)
            try:
                articles = newsapi.get_everything(
                    q='gold price OR XAU OR gold market',
                    from_param=current.strftime('%Y-%m-%d'),
                    to=next_day.strftime('%Y-%m-%d'),
                    language='en',
                    page_size=5
                )
                headlines = [a['title'] for a in articles['articles'] if a['title']]
                if headlines:
                    results = sentiment_pipe(headlines[:5], truncation=True, max_length=512)
                    score = sum(
                        r['score'] if r['label'] == 'positive'
                        else -r['score'] if r['label'] == 'negative'
                        else 0
                        for r in results
                    ) / len(results)
                else:
                    score = 0.0
            except:
                score = 0.0

            records.append({
                'Date': current.strftime('%Y-%m-%d'),
                'Sentiment': score
            })
            print(f"  {current.strftime('%Y-%m-%d')}: {score:.3f}")
            current = next_day

        df = pd.DataFrame(records)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # Сохраняем кэш
        df.to_csv(sentiment_cache)
        print(f"✅ Sentiment сохранён в {sentiment_cache}")
        return df

    except Exception as e:
        print(f"⚠️ FinBERT недоступен: {e}")
        # Пробуем загрузить кэш
        if os.path.exists(sentiment_cache):
            print("Используем кэшированный sentiment...")
            df = pd.read_csv(sentiment_cache, index_col='Date', parse_dates=True)
            return df
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

    # ── Train/Test split для графиков ──
    split_idx = int(len(prophet_df) * 0.8)
    train_df  = prophet_df.iloc[:split_idx]
    test_df   = prophet_df.iloc[split_idx:]
    forecast  = model.predict(test_df)

    os.makedirs('ml/artifacts/plots', exist_ok=True)

    # ── График Prophet ──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(prophet_df['ds'], prophet_df['y'],
            color='black', linewidth=1.2, label='Реальная цена')
    ax.axvspan(train_df['ds'].iloc[0], train_df['ds'].iloc[-1],
               alpha=0.07, color='steelblue', label='Train')
    ax.axvspan(test_df['ds'].iloc[0], test_df['ds'].iloc[-1],
               alpha=0.12, color='orange', label='Test')
    ax.plot(test_df['ds'], forecast['yhat'],
            color='#E63946', linewidth=1.2, label='Prophet предсказание')
    ax.fill_between(test_df['ds'],
                    forecast['yhat_lower'], forecast['yhat_upper'],
                    alpha=0.15, color='#E63946', label='Доверительный интервал')
    ax.set_title('Prophet — предсказание на тестовом периоде', fontsize=13, fontweight='bold')
    ax.set_ylabel('USD / oz')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('ml/artifacts/plots/prophet_test.png', dpi=150)
    plt.close()

    # ── График Linear Regression ──
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error

    X = df[feature_cols]
    y = df['Price']
    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_sc, y_train)
    pred_lr = lr.predict(X_test_sc)

    mae_lr = mean_absolute_error(y_test, pred_lr)
    r2_lr  = r2_score(y_test, pred_lr)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_test.index, y_test.values,
            color='black', linewidth=1.2, label='Реальная цена')
    ax.plot(y_test.index, pred_lr,
            color='royalblue', linewidth=1.2, label='Linear Regression')
    ax.set_title(f'Linear Regression — тестовый период | MAE: ${mae_lr:.1f} | R²: {r2_lr:.4f}',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('USD / oz')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('ml/artifacts/plots/lr_test.png', dpi=150)
    plt.close()

    print("📊 Графики сохранены в ml/artifacts/plots/")
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