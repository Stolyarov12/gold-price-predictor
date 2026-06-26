import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

API_URL = "http://api:8000"
#API_URL = "http://localhost:8000"  # для локального запуска

st.set_page_config(
    page_title="Gold Price Predictor",
    page_icon="🥇",
    layout="wide"
)


# СТИЛИ

st.markdown("""
<style>
    .main { background-color: #0f0f1a; }
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .predict-btn { width: 100%; }
</style>
""", unsafe_allow_html=True)


# ХЕДЕР

st.title("🥇 Gold Price Predictor")
st.markdown("**Prophet + FinBERT Sentiment** — предсказание цены золота (XAU/USD)")
st.divider()


# ПРОВЕРКА API

def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json().get("model_loaded", False)
    except:
        return False

api_ok = check_api()
if not api_ok:
    st.error("⚠️ API недоступен или модель не загружена. Запустите train.py и перезапустите API.")
    st.stop()


# САЙДБАР

with st.sidebar:
    st.header("⚙️ Настройки")
    history_days = st.slider("История (дней)", 30, 365, 90)
    forecast_days = st.slider("Прогноз (дней)", 7, 90, 30)
    st.divider()
    st.markdown("**Источники данных:**")
    st.markdown("- 📈 Yahoo Finance (yfinance)")
    st.markdown("- 📰 NewsAPI + FinBERT")
    st.divider()
    st.markdown("**Модель:** Prophet (Meta)")
    st.markdown("**Метрики:** R²=0.9885 | MAPE=0.87%")


# БЛОК ПРЕДСКАЗАНИЯ

st.subheader("🔮 Предсказание цены")

col1, col2, col3 = st.columns([1, 1, 1])

if st.button("🚀 Рассчитать завтрашнюю цену", use_container_width=True, type="primary"):
    with st.spinner("Считаем..."):
        try:
            r = requests.get(f"{API_URL}/price/predict", timeout=10)
            data = r.json()

            if "error" in data:
                st.error(f"Ошибка: {data['error']}")
            else:
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        label=f"📅 Последняя цена ({data['last_date']})",
                        value=f"${data['last_price']:,.2f}"
                    )
                with col2:
                    st.metric(
                        label=f"🔮 Прогноз на {data['predict_date']}",
                        value=f"${data['predicted']:,.2f}",
                        delta=f"{data['change_pct']:+.2f}%"
                    )
                with col3:
                    st.metric(
                        label="📊 Коридор прогноза",
                        value=f"${data['lower']:,.0f} — ${data['upper']:,.0f}"
                    )

                direction_color = "🟢" if data['change_pct'] > 0 else "🔴"
                st.success(f"{direction_color} Направление: **{data['direction']}** | Изменение: **{data['change_pct']:+.2f}%**")

        except Exception as e:
            st.error(f"Ошибка подключения к API: {e}")

st.divider()


# ГРАФИК ИСТОРИИ

st.subheader("📈 Историческая цена золота")

try:
    r = requests.get(f"{API_URL}/price/history?days={history_days}", timeout=10)
    history = r.json()['data']
    hist_df = pd.DataFrame(history)
    hist_df['date'] = pd.to_datetime(hist_df['date'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_df['date'], y=hist_df['price'],
        mode='lines', name='Цена золота',
        line=dict(color='gold', width=2)
    ))
    fig.update_layout(
        title=f'Цена золота за последние {history_days} дней',
        xaxis_title='Дата',
        yaxis_title='USD / oz',
        template='plotly_dark',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Ошибка загрузки истории: {e}")


# ГРАФИК ПРОГНОЗА

st.subheader("🔭 Прогноз на будущее")

try:
    r = requests.get(f"{API_URL}/price/forecast?days={forecast_days}", timeout=30)
    forecast = r.json()['data']
    fore_df = pd.DataFrame(forecast)
    fore_df['date'] = pd.to_datetime(fore_df['date'])

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=hist_df['date'], y=hist_df['price'],
        mode='lines', name='История',
        line=dict(color='gold', width=2)
    ))
    fig2.add_trace(go.Scatter(
        x=fore_df['date'], y=fore_df['price'],
        mode='lines', name='Прогноз',
        line=dict(color='#E63946', width=2, dash='dash')
    ))
    fig2.update_layout(
        title=f'Прогноз на {forecast_days} дней вперёд',
        xaxis_title='Дата',
        yaxis_title='USD / oz',
        template='plotly_dark',
        height=400
    )
    st.plotly_chart(fig2, use_container_width=True)
except Exception as e:
    st.error(f"Ошибка загрузки прогноза: {e}")



# ГРАФИКИ ОБУЧЕНИЯ МОДЕЛЕЙ

st.divider()
st.subheader("📚 Графики обучения моделей")

tab1, tab2 = st.tabs(["🔴 Prophet", "🔵 Linear Regression"])

with tab1:
    try:
        r = requests.get(f"{API_URL}/plots/prophet_test.png", timeout=10)
        if r.status_code == 200:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(r.content))
            st.image(img, caption="Prophet — предсказание на тестовом периоде", use_column_width=True)
        else:
            st.info("График не найден")
    except Exception as e:
        st.error(f"Ошибка: {e}")

with tab2:
    try:
        r = requests.get(f"{API_URL}/plots/lr_test.png", timeout=10)
        if r.status_code == 200:
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(r.content))
            st.image(img, caption="Linear Regression — тестовый период", use_column_width=True)
        else:
            st.info("График не найден")
    except Exception as e:
        st.error(f"Ошибка: {e}")

# ФУТЕР

st.divider()
st.markdown("""
<div style='text-align: center; color: #64748b; font-size: 14px;'>
    Gold Price Predictor | Prophet + FinBERT Sentiment | НИТУ МИСИС 2026
</div>
""", unsafe_allow_html=True)