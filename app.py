import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

st.set_page_config(page_title="AI 股市戰情室 (精簡版)", layout="wide", page_icon="⚡")

# ==========================================
# 📋 股票分類清單
# ==========================================
SECTORS = {
    "🚀 電子/AI": [
        "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2303.TW", "2382.TW", "3711.TW", "3034.TW", "3035.TW", 
        "3231.TW", "2356.TW", "6669.TW", "2376.TW", "3017.TW", "2421.TW", "2412.TW"
    ],
    "🚢 傳產/金融": [
        "2603.TW", "2609.TW", "2615.TW", "2618.TW", "2002.TW", "1605.TW", "1513.TW", "1519.TW",
        "2881.TW", "2882.TW", "2891.TW", "2886.TW", "5880.TW"
    ],
    "📊 ETF": [
        "00980A.TW", "00981A.TW", "00982A.TW", "00983A.TW",
        "0050.TW", "0056.TW", "00878.TW", "00929.TW", "00919.TW"
    ],
    "🇺🇸 美股": [
        "NVDA", "TSLA", "AAPL", "MSFT", "GOOG", "AMZN", "META", "AMD", "INTC", "PLTR", "SMCI"
    ]
}

ALL_STOCKS = [item for sublist in SECTORS.values() for item in sublist]

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
        is_etf = ticker.startswith("00") or ticker.endswith("A.TW")

        # 安全指標
        def safe_ind(func, default=0):
            try: return func()
            except: return pd.Series([default]*len(close))

        ema_20 = safe_ind(lambda: EMAIndicator(close=close, window=20).ema_indicator(), curr_price)
        ema_60 = safe_ind(lambda: EMAIndicator(close=close, window=60).ema_indicator(), curr_price)
        macd_val = safe_ind(lambda: MACD(close=close).macd(), 0)
        sig_val = safe_ind(lambda: MACD(close=close).macd_signal(), 0)
        rsi_val = safe_ind(lambda: RSIIndicator(close=close).rsi(), 50)
        atr_val = safe_ind(lambda: AverageTrueRange(high=high, low=low, close=close).average_true_range(), curr_price*0.02)
        
        try: bb_lower = BollingerBands(close=close, window=20, window_dev=2).bollinger_lband().iloc[-1]
        except: bb_lower = curr_price * 0.9

        ma_5 = close.rolling(window=5).mean()
        def get_last(s): return s.iloc[-1] if not pd.isna(s.iloc[-1]) else 0

        curr_ma5 = get_last(ma_5) if get_last(ma_5) > 0 else curr_price
        curr_atr = get_last(atr_val)
        curr_rsi = get_last(rsi_val)
        val_e20 = get_last(ema_20)
        val_e60 = get_last(ema_60)
        val_macd = get_last(macd_val)
        val_sig = get_last(sig_val)

        # 評分
        trend_score = 0
        rebound_score = 0
        
        if curr_price > val_e20 > val_e60: trend_score += 40
        elif curr_price > val_e60: trend_score += 20
        if val_macd > val_sig: trend_score += 20
        if 50 <= curr_rsi <= 75: trend_score += 20
        
        bias = ((curr_price - val_e20) / val_e20) * 100 if val_e20 > 0 else 0
        
        rsi_limit = 45 if is_etf else 40
        if curr_rsi < 30 and curr_rsi > 0: rebound_score += 40
        elif curr_rsi < rsi_limit and curr_rsi > 0: rebound_score += 20
        if curr_price <= bb_lower: rebound_score += 30
        if bias < -5: rebound_score += 30

        total_score = trend_score + rebound_score

        # 預測 (5/10/30日)
        pred_5_str = "-"
        pred_10_str = "-"
        pred_30_str = "-"
        
        if len(close) > 10:
            x = np.arange(len(close.tail(20)))
            y = close.tail(20).values
            try:
                slope, _ = np.polyfit(x, y, 1)
                if slope > 0:
                    pred_5_str = f"{curr_price + (slope * 5):.1f}" # 小數點一位省空間
                    pred_10_str = f"{curr_price + (slope * 10):.1f}"
                    pred_30_str = f"{curr_price + (slope * 30):.1f}"
                elif rebound_score >= 30:
                    target = val_e20 if val_e20 > curr_price else (curr_price * 1.03)
                    pred_5_str = f"{target:.1f}"
                    pred_10_str = f"{target:.1f}"
            except: pass

        # 訊號
        action = "👀" # 用符號省空間
        buy_price = 0.0
        buy_threshold = 50 if is_etf else 60

        if trend_score >= 80:
            action = "🔥 強力"
            buy_price = curr_price
        elif total_score >= buy_threshold:
            if trend_score > rebound_score:
                action = "🔴 偏多"
                buy_price = curr_ma5
            else:
                action = "💎 甜蜜"
                buy_price = curr_price
            if curr_price < buy_price: buy_price = curr_price

        stop_loss = curr_price - (2 * curr_atr)

        return {
            "代號": ticker.replace(".TW", ""), # 去掉 .TW 省空間
            "現價": round(curr_price, 1),
            "總分": total_score,
            "RSI": int(curr_rsi), # 取整數省空間
            "🎯買點": round(buy_price, 1) if buy_price > 0 else "-",
            "💡訊號": action,
            "5日": pred_5_str,
            "10日": pred_10_str,
            "30日": pred_30_str,
            "🛑停損": round(stop_loss, 1),
            "_sort": total_score
        }
    except: return None

# ==========================================
# 🖥️ 介面
# ==========================================
st.title("⚡ AI 股市戰情室 (精簡版)")

if st.button("🔄 更新數據", type="primary"):
    with st.spinner('掃描中...'):
        raw_data = fetch_all_data(ALL_STOCKS)
        
        if raw_data is not None and not raw_data.empty:
            tab1, tab2, tab3, tab4 = st.tabs(["🚀 電子", "🚢 金融傳產", "📊 ETF", "🇺🇸 美股"])
            
            def show_sector(stocks_list):
                results = []
                for t in stocks_list:
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
                        if "甜蜜" in val: return 'background-color: #e6fffa; color: #006666'
                        return 'color: #cccccc'

                    st.dataframe(
                        df.style.applymap(highlight, subset=['💡訊號']),
                        use_container_width=True,
                        column_config={
                            "代號": st.column_config.TextColumn(width="small"),
                            "現價": st.column_config.NumberColumn(format="%.1f", width="small"),
                            "總分": st.column_config.ProgressColumn(format="%d", min_value=0, max_value=100, width="small"),
                            "RSI": st.column_config.NumberColumn(format="%d", width="small"),
                            "🎯買點": st.column_config.TextColumn(width="small"),
                            "💡訊號": st.column_config.TextColumn(width="small"),
                            # 關鍵：這裡全部設為 small
                            "5日": st.column_config.TextColumn(width="small"),
                            "10日": st.column_config.TextColumn(width="small"),
                            "30日": st.column_config.TextColumn(width="small"),
                            "🛑停損": st.column_config.NumberColumn(format="%.1f", width="small")
                        }
                    )
                else: st.info("無數據")

            with tab1: show_sector(SECTORS["🚀 電子/AI"])
            with tab2: show_sector(SECTORS["🚢 傳產/金融"])
            with tab3: show_sector(SECTORS["📊 ETF"])
            with tab4: show_sector(SECTORS["🇺🇸 美股"])
            
        else: st.error("連線失敗")
