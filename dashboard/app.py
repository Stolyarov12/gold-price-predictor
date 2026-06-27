import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

API_URL = "http://api:8000"

st.set_page_config(
    page_title="Gold Price Predictor",
    page_icon="🥇",
    layout="wide"
)

# ══════════════════════════════════════════════════════════════
# СТИЛИ
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .main { background-color: #0f0f1a; }
    .info-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #FFD700;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .info-card h4 { color: #FFD700; margin: 0 0 6px 0; font-size: 13px; }
    .info-card p { color: #e0e0e0; margin: 0; font-size: 15px; font-weight: 600; }
    .section-header {
        color: #FFD700;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin: 16px 0 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# ПРОВЕРКА API
# ══════════════════════════════════════════════════════════════
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json().get("model_loaded", False)
    except:
        return False

api_ok = check_api()

# ══════════════════════════════════════════════════════════════
# САЙДБАР
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🥇 Gold Predictor")
    st.markdown("---")

    st.markdown('<p class="section-header">⚙️ Настройки графиков</p>', unsafe_allow_html=True)
    history_days = st.slider("История (дней)", 30, 365, 90)
    forecast_days = st.slider("Прогноз (дней)", 7, 90, 30)

    st.markdown("---")
    st.markdown('<p class="section-header">📊 О модели</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h4>МОДЕЛЬ</h4>
        <p>Prophet (Meta)</p>
    </div>
    <div class="info-card">
        <h4>R² SCORE</h4>
        <p>0.9885</p>
    </div>
    <div class="info-card">
        <h4>MAPE</h4>
        <p>0.87%</p>
    </div>
    <div class="info-card">
        <h4>MAE</h4>
        <p>~$20-47</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<p class="section-header">🗄️ Источники данных</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-card">
        <h4>КОТИРОВКИ</h4>
        <p>Yahoo Finance (yfinance)</p>
    </div>
    <div class="info-card">
        <h4>НОВОСТИ</h4>
        <p>NewsAPI + FinBERT</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    status = "🟢 Online" if api_ok else "🔴 Offline"
    st.markdown(f'<div class="info-card"><h4>СТАТУС API</h4><p>{status}</p></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# ХЕДЕР
# ══════════════════════════════════════════════════════════════
st.markdown("# 🥇 Gold Price Predictor")
st.markdown("**Prophet + FinBERT Sentiment** — интеллектуальная система прогнозирования цены золота (XAU/USD)")

st.markdown("""
<div style='background: linear-gradient(135deg, #1a1a2e, #16213e); border-left: 4px solid #FFD700;
     border-radius: 8px; padding: 16px 20px; margin: 16px 0;'>
    <p style='color: #e0e0e0; margin: 0; font-size: 14px; line-height: 1.6;'>
    Система анализирует исторические котировки золота с 2019 года через <b>Yahoo Finance</b>,
    обогащает данные сентимент-анализом финансовых новостей через <b>NewsAPI + FinBERT</b>,
    и строит прогноз с помощью модели временных рядов <b>Prophet от Meta</b>.
    Все компоненты упакованы в <b>Docker</b> и взаимодействуют через <b>REST API</b>.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

if not api_ok:
    st.error("⚠️ API недоступен или модель не загружена.")
    st.stop()

# ══════════════════════════════════════════════════════════════
# KPI МЕТРИКИ СВЕРХУ
# ══════════════════════════════════════════════════════════════
try:
    r = requests.get(f"{API_URL}/price/history?days=2", timeout=10)
    hist_data = r.json()['data']
    last_price = hist_data[-1]['price']
    prev_price = hist_data[-2]['price'] if len(hist_data) > 1 else last_price
    daily_change = (last_price - prev_price) / prev_price * 100
    last_date = hist_data[-1]['date']

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Текущая цена", f"${last_price:,.2f}", f"{daily_change:+.2f}% за день")
    with col2:
        st.metric("📅 Последнее обновление", last_date)
    with col3:
        st.metric("🎯 Точность модели (R²)", "0.9885")
    with col4:
        st.metric("📉 Средняя ошибка (MAPE)", "0.87%")
except:
    pass

st.divider()

# ══════════════════════════════════════════════════════════════
# БЛОК ПРЕДСКАЗАНИЯ
# ══════════════════════════════════════════════════════════════
st.subheader("🔮 Предсказание цены")

col_btn1, col_btn2, col_btn3 = st.columns(3)
with col_btn1:
    btn_tomorrow = st.button("📅 Завтра", use_container_width=True, type="primary")
with col_btn2:
    btn_week = st.button("📅 Через 7 дней", use_container_width=True, type="primary")
with col_btn3:
    btn_month = st.button("📅 Через 30 дней", use_container_width=True, type="primary")

if btn_tomorrow or btn_week or btn_month:
    days_ahead = 1 if btn_tomorrow else 7 if btn_week else 30
    label = "завтра" if btn_tomorrow else "через 7 дней" if btn_week else "через 30 дней"

    with st.spinner(f"Считаем прогноз на {label}..."):
        try:
            r = requests.get(f"{API_URL}/price/forecast?days={days_ahead}", timeout=30)
            data = r.json()['data']
            target = data[-1]

            r2 = requests.get(f"{API_URL}/price/history?days=1", timeout=10)
            last = r2.json()['data'][-1]
            last_price = last['price']
            pred_price = target['price']
            change_pct = (pred_price - last_price) / last_price * 100

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"📅 Последняя цена ({last['date']})", f"${last_price:,.2f}")
            with col2:
                st.metric(f"🔮 Прогноз на {target['date']}", f"${pred_price:,.2f}", f"{change_pct:+.2f}%")
            with col3:
                st.metric("📊 Горизонт прогноза", label.capitalize())

            color = "🟢" if change_pct > 0 else "🔴"
            direction = "📈 Рост" if change_pct > 0 else "📉 Падение"
            st.success(f"{color} Направление: **{direction}** | Изменение за период: **{change_pct:+.2f}%**")

        except Exception as e:
            st.error(f"Ошибка: {e}")

st.divider()

# ══════════════════════════════════════════════════════════════
# ГРАФИКИ
# ══════════════════════════════════════════════════════════════
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
        line=dict(color='#FFD700', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 215, 0, 0.05)'
    ))
    fig.update_layout(
        title=f'Цена золота за последние {history_days} дней',
        xaxis_title='Дата', yaxis_title='USD / oz',
        template='plotly_dark', height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Ошибка: {e}")

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
        line=dict(color='#FFD700', width=2)
    ))
    fig2.add_trace(go.Scatter(
        x=fore_df['date'], y=fore_df['price'],
        mode='lines', name='Прогноз',
        line=dict(color='#E63946', width=2, dash='dash')
    ))
    fig2.add_vline(
        x=hist_df['date'].iloc[-1],
        line_dash="dot", line_color="gray",
        annotation_text="Начало прогноза"
    )
    fig2.update_layout(
        title=f'Прогноз на {forecast_days} дней вперёд',
        xaxis_title='Дата', yaxis_title='USD / oz',
        template='plotly_dark', height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    st.plotly_chart(fig2, use_container_width=True)
except Exception as e:
    st.error(f"Ошибка: {e}")

st.divider()

# ══════════════════════════════════════════════════════════════
# ГРАФИКИ ОБУЧЕНИЯ
# ══════════════════════════════════════════════════════════════
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
    except Exception as e:
        st.error(f"Ошибка: {e}")

st.divider()

# ══════════════════════════════════════════════════════════════
# ФУТЕР
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div style='text-align: center; color: #64748b; font-size: 13px; padding: 20px 0;'>
    Gold Price Predictor | Prophet + FinBERT Sentiment | НИТУ МИСИС 2026<br>
    <span style='color: #FFD700;'>Stolyarov12</span> · 
    <a href='https://github.com/Stolyarov12/gold-price-predictor' style='color: #FFD700;'>GitHub</a>
</div>
""", unsafe_allow_html=True)