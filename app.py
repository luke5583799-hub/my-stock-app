import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

st.set_page_config(page_title="AI 股價預言家 (完整版)", layout="wide", page_icon="🔮")

# ==========================================
# 📋 監控清單
# ==========================================
DEFAULT_STOCKS = [
    "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2303.TW", "2382.TW", "3711.TW", "3034.TW", "3035.TW", 
    "3231.TW", "2356.TW", "6669.TW", "2376.TW", "3017.TW", "2421.TW", "2412.TW",
    "2603.TW", "2609.TW", "2615.TW", "2618.TW", "2002.TW", "1605.TW", "1513.TW", "1519.TW",
    "2881.TW", "2882.TW", "2891.TW", "2886.TW", "5880.TW",
    "0050.TW", "0056.TW", "00878.TW", "00929.TW", "00919.TW",
    "NVDA", "TSLA", "AAPL", "MSFT", "GOOG", "AMZN", "META", "AMD", "INTC", "PLTR", "SMCI"
]

@st.cache_data(ttl=300)
def fetch_all_data(tickers):
    tickers_str = " ".join(tickers)
    try:
        data = yf.download(tickers_str, period="6mo", group_by='ticker', progress=False)
        return data
    except: return None

def calculate_metrics(ticker, df):
    try:
        if isinstance(df.columns, pd.MultiIndex): df = df.xs(ticker, axis=1, level=0)
        df = df.dropna(how='all')
        if len(df) < 50: return None

        close = df['Close']
        high = df['High']
        low = df['Low']

        # 指標
        ema_20 = EMAIndicator(close=close, window=20).ema_indicator()
        ema_60 = EMAIndicator(close=close, window=60).ema_indicator()
        macd = MACD(close=close)
        rsi = RSIIndicator(close=close).rsi()
        bb = BollingerBands(close=close, window=20, window_dev=2)
        atr = AverageTrueRange(high=high, low=low, close=close).average_true_range()
        
        ma_5 = close.rolling(window=5).mean()

        curr_price = close.iloc[-1]
        curr_ma5 = ma_5.iloc[-1]
        curr_atr = atr.iloc[-1]
        curr_rsi = rsi.iloc[-1]

        # 評分
        trend_score = 0
        rebound_score = 0
        
        if curr_price > ema_20.iloc[-1] > ema_60.iloc[-1]: trend_score += 40
        elif curr_price > ema_60.iloc[-1]: trend_score += 20
        if macd.macd().iloc[-1] > macd.macd_signal().iloc[-1]: trend_score += 20
        if 50 <= curr_rsi <= 75: trend_score += 20
        
        bias = ((curr_price - ema_20.iloc[-1]) / ema_20.iloc[-1]) * 100
        if curr_rsi < 30: rebound_score += 40
        elif curr_rsi < 40: rebound_score += 15
        if curr_price <= bb.bollinger_lband().iloc[-1]: rebound_score += 30
        if bias < -7: rebound_score += 30

        total_score = trend_score + rebound_score

        # --- 預測邏輯 ---
        recent_data = close.tail(20)
        x = np.arange(len(recent_data))
        y = recent_data.values
        slope, intercept = np.polyfit(x, y, 1)
        
        if slope > 0:
            pred_5 = f"{curr_price + (slope * 5):.2f}"
            pred_10 = f"{curr_price + (slope * 10):.2f}"
            pred_30 = f"{curr_price + (slope * 30):.2f}"
        else:
            pred_5 = pred_10 = pred_30 = "-"

        # --- 訊號判斷 ---
        action = "👀 觀望"
        buy_price = 0.0
        
        if trend_score >= 80:
            action = "🔥 強力買進"
            buy_price = curr_price 
        elif trend_score >= 60:
            action = "🔴 偏多操作"
            buy_price = curr_ma5
            if curr_price < buy_price: buy_price = curr_price
        elif rebound_score >= 60:
            action = "💎 嘗試抄底"
            buy_price = curr_price

        stop_loss = curr_price - (2 * curr_atr)

        return {
            "代號": ticker,
            "現價": round(curr_price, 2),
            "總分": total_score,   # 新增分數
            "RSI": round(curr_rsi, 1), # 新增 RSI
            "🎯 建議入手": round(buy_price, 2) if buy_price > 0 else "-",
            "訊號": action,
            "5日目標": pred_5,
            "10日目標": pred_10,
            "30日目標": pred_30,
            "建議停損": round(stop_loss, 2),
            "_sort": total_score
        }
    except: return None

# ==========================================
# 🖥️ 介面
# ==========================================
st.title("🔮 AI 股價預言家 (完整版)")
st.caption("全覽監控：分數、RSI、預測價位一次看清。")

if st.button("🔄 更新全市場數據", type="primary"):
    with st.spinner('正在進行深度分析...'):
        raw_data = fetch_all_data(DEFAULT_STOCKS)
        if raw_data is not None and not raw_data.empty:
            results = []
            for t in DEFAULT_STOCKS:
                try:
                    res = calculate_metrics(t, raw_data[t])
                    if res: results.append(res)
                except: continue
            
            df = pd.DataFrame(results)
            if not df.empty:
                df = df.sort_values(by='_sort', ascending=False).drop(columns=['_sort'])
                
                def highlight(val):
                    if "強力" in val: return 'background-color: #ffcccc; color: #8b0000; font-weight: bold'
                    if "偏多" in val: return 'background-color: #fff5e6; color: #d68910'
                    if "抄底" in val: return 'background-color: #e6fffa; color: #006666'
                    return 'color: #999999'

                st.dataframe(
                    df.style.applymap(highlight, subset=['訊號']),
                    use_container_width=True,
                    column_config={
                        "代號": st.column_config.TextColumn(width="small"),
                        "現價": st.column_config.NumberColumn(format="%.2f", width="small"),
                        "總分": st.column_config.ProgressColumn(format="%d", min_value=0, max_value=100, width="small"), # 用進度條顯示分數
                        "RSI": st.column_config.NumberColumn(format="%.1f", width="small"),
                        "🎯 建議入手": st.column_config.TextColumn(help="觀望股顯示 -", width="medium"),
                        "訊號": st.column_config.TextColumn(width="medium"),
                        # 把目標價縮小
                        "5日目標": st.column_config.TextColumn(width="small"),
                        "10日目標": st.column_config.TextColumn(width="small"),
                        "30日目標": st.column_config.TextColumn(width="small"),
                        "建議停損": st.column_config.NumberColumn(format="%.2f", help="防守底線", width="small")
                    }
                )
                
                st.success("✅ 更新完成！分數越高 (紅色進度條越長) 代表勝率越高。")
            else: st.info("無數據。")
        else: st.error("連線失敗")
