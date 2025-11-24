import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

st.set_page_config(page_title="AI 股市預言家 (ETF優化版)", layout="wide", page_icon="🔮")

# ==========================================
# 📋 監控清單
# ==========================================
DEFAULT_STOCKS = [
    # --- 主動式 ETF ---
    "00980A.TW", "00981A.TW", "00982A.TW", "00983A.TW",
    # --- 熱門權值 ---
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
        if len(df) < 5: return None

        close = df['Close']
        high = df['High']
        low = df['Low']
        
        curr_price = close.iloc[-1]
        
        # 判斷是否為 ETF (代號 00 開頭)
        is_etf = ticker.startswith("00")

        # --- 安全計算指標 ---
        def safe_indicator(func, default=0):
            try: return func()
            except: return pd.Series([default]*len(close))

        ema_20_s = safe_indicator(lambda: EMAIndicator(close=close, window=20).ema_indicator(), curr_price)
        ema_60_s = safe_indicator(lambda: EMAIndicator(close=close, window=60).ema_indicator(), curr_price)
        macd_obj = MACD(close=close)
        macd_s = safe_indicator(lambda: macd_obj.macd(), 0)
        signal_s = safe_indicator(lambda: macd_obj.macd_signal(), 0)
        rsi_s = safe_indicator(lambda: RSIIndicator(close=close).rsi(), 50)
        atr_s = safe_indicator(lambda: AverageTrueRange(high=high, low=low, close=close).average_true_range(), curr_price*0.02)
        
        try:
            bb = BollingerBands(close=close, window=20, window_dev=2)
            bb_lower = bb.bollinger_lband().iloc[-1]
        except: bb_lower = curr_price * 0.9

        ma_5 = close.rolling(window=5).mean()

        def get_last(series): return series.iloc[-1] if not pd.isna(series.iloc[-1]) else 0

        curr_ma5 = get_last(ma_5) if get_last(ma_5) > 0 else curr_price
        curr_atr = get_last(atr_s)
        curr_rsi = get_last(rsi_s)
        val_ema20 = get_last(ema_20_s)
        val_ema60 = get_last(ema_60_s)
        val_macd = get_last(macd_s)
        val_signal = get_last(signal_s)

        # --- 評分邏輯 (針對 ETF 優化) ---
        trend_score = 0
        rebound_score = 0
        
        # 1. 趨勢
        if curr_price > val_ema20 > val_ema60: trend_score += 40
        elif curr_price > val_ema60: trend_score += 20
        if val_macd > val_signal: trend_score += 20
        if 50 <= curr_rsi <= 75: trend_score += 20
        
        # 2. 抄底 (ETF 只要 RSI < 45 就算便宜)
        bias = ((curr_price - val_ema20) / val_ema20) * 100 if val_ema20 > 0 else 0
        
        rsi_threshold_1 = 45 if is_etf else 40
        rsi_threshold_2 = 35 if is_etf else 30

        if curr_rsi < rsi_threshold_2 and curr_rsi > 0: rebound_score += 40
        elif curr_rsi < rsi_threshold_1 and curr_rsi > 0: rebound_score += 20 # 放寬標準
        
        if curr_price <= bb_lower: rebound_score += 30
        if bias < -5: rebound_score += 30 # 放寬乖離

        total_score = trend_score + rebound_score

        # --- 預測邏輯 ---
        pred_5_str = "-"
        pred_10_str = "-"
        pred_30_str = "-"
        target_note = ""

        if len(close) > 10:
            recent_data = close.tail(20)
            x = np.arange(len(recent_data))
            y = recent_data.values
            try:
                slope, intercept = np.polyfit(x, y, 1)
                if slope > 0:
                    pred_5_str = f"{curr_price + (slope * 5):.2f}"
                    pred_10_str = f"{curr_price + (slope * 10):.2f}"
                    pred_30_str = f"{curr_price + (slope * 30):.2f}"
                    target_note = "趨勢推算"
                # 只要分數 > 40 就顯示反彈目標 (對 ETF 寬容)
                elif rebound_score >= 30: 
                    target = val_ema20 if val_ema20 > curr_price else (curr_price * 1.03)
                    pred_5_str = f"{target:.2f}"
                    pred_10_str = f"{target:.2f}"
                    target_note = "反彈目標"
            except: pass

        # --- 訊號判斷 ---
        action = "👀 觀望"
        buy_price = 0.0
        
        # 分數門檻調整：ETF 只要 50 分就可以考慮買
        buy_threshold = 50 if is_etf else 60

        if trend_score >= 80:
            action = "🔥 強力買進"
            buy_price = curr_price 
        elif total_score >= buy_threshold: # 綜合分數達標
            if trend_score > rebound_score:
                action = "🔴 偏多操作"
                buy_price = curr_ma5
            else:
                action = "💎 甜蜜買點" # ETF 專用詞
                buy_price = curr_price
            
            if curr_price < buy_price: buy_price = curr_price

        stop_loss = curr_price - (2 * curr_atr)

        return {
            "代號": ticker,
            "現價": round(curr_price, 2),
            "總分": total_score,
            "RSI": round(curr_rsi, 1),
            "🎯 建議入手": round(buy_price, 2) if buy_price > 0 else "-",
            "訊號": action,
            "5日目標": pred_5_str,
            "10日目標": pred_10_str,
            "30日目標": pred_30_str,
            "建議停損": round(stop_loss, 2),
            "_sort": total_score
        }
    except: return None

# ==========================================
# 🖥️ 介面
# ==========================================
st.title("🔮 AI 股市預言家 (ETF 優化版)")
st.caption("針對 ETF 波動較小的特性，優化買進門檻 (RSI < 45 即觸發甜蜜點)。")

if st.button("🔄 更新數據", type="primary"):
    with st.spinner('正在為 ETF 進行特殊調校...'):
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
                    if "甜蜜" in val or "抄底" in val: return 'background-color: #e6fffa; color: #006666; font-weight: bold'
                    return 'color: #999999'

                st.dataframe(
                    df.style.applymap(highlight, subset=['訊號']),
                    use_container_width=True,
                    column_config={
                        "代號": st.column_config.TextColumn(width="small"),
                        "現價": st.column_config.NumberColumn(format="%.2f", width="small"),
                        "總分": st.column_config.ProgressColumn(format="%d", min_value=0, max_value=100, width="small"),
                        "RSI": st.column_config.NumberColumn(format="%.1f", width="small"),
                        "🎯 建議入手": st.column_config.TextColumn(width="medium"),
                        "訊號": st.column_config.TextColumn(width="medium"),
                        "5日目標": st.column_config.TextColumn(width="small"),
                        "10日目標": st.column_config.TextColumn(width="small"),
                        "30日目標": st.column_config.TextColumn(width="small"),
                        "建議停損": st.column_config.NumberColumn(format="%.2f", width="small")
                    }
                )
            else: st.info("無數據。")
        else: st.error("連線失敗")
