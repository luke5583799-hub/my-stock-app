import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

st.set_page_config(page_title="AI 股價預言家 (全覽版)", layout="wide", page_icon="🔮")

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

        # --- 預測邏輯 ---
        recent_data = close.tail(20)
        x = np.arange(len(recent_data))
        y = recent_data.values
        slope, intercept = np.polyfit(x, y, 1)
        
        # 只有趨勢向上才給目標價，不然給 "-"
        if slope > 0:
            pred_5 = curr_price + (slope * 5)
            pred_10 = curr_price + (slope * 10)
            pred_30 = curr_price + (slope * 30)
            target_5_str = f"{pred_5:.2f}"
            target_10_str = f"{pred_10:.2f}"
            target_30_str = f"{pred_30:.2f}"
        else:
            target_5_str = "-"
            target_10_str = "-"
            target_30_str = "-"

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
            "🎯 建議入手價": round(buy_price, 2) if buy_price > 0 else "-",
            "訊號": action,
            "5日目標": target_5_str,
            "10日目標": target_10_str,
            "30日目標": target_30_str,
            "建議停損": round(stop_loss, 2),
            "_sort": trend_score + rebound_score
        }
    except: return None

# ==========================================
# 🖥️ 介面
# ==========================================
st.title("🔮 AI 股價預言家 (全覽版)")
st.caption("顯示所有監控中的股票，包含觀望股。")

if st.button("🔄 更新全市場數據", type="primary"):
    with st.spinner('正在分析所有股票...'):
        raw_data = fetch_all_data(DEFAULT_STOCKS)
        if raw_data is not None and not raw_data.empty:
            results = []
            for t in DEFAULT_STOCKS:
                try:
                    res = calculate_metrics(t, raw_data[t])
                    if res: results.append(res) # 不再過濾，全部顯示
                except: continue
            
            df = pd.DataFrame(results)
            if not df.empty:
                # 排序：分數高的在上面
                df = df.sort_values(by='_sort', ascending=False).drop(columns=['_sort'])
                
                def highlight(val):
                    if "強力" in val: return 'background-color: #ffcccc; color: #8b0000; font-weight: bold'
                    if "偏多" in val: return 'background-color: #fff5e6; color: #d68910'
                    if "抄底" in val: return 'background-color: #e6fffa; color: #006666'
                    return 'color: #999999' # 觀望變成灰色

                st.dataframe(
                    df.style.applymap(highlight, subset=['訊號']),
                    use_container_width=True,
                    column_config={
                        "🎯 建議入手價": st.column_config.TextColumn(help="觀望股不建議入手，故顯示 -"),
                        "5日目標": st.column_config.TextColumn(help="趨勢向下時顯示 -"),
                        "建議停損": st.column_config.NumberColumn(format="%.2f", help="防守底線")
                    }
                )
                
                # 統計資訊
                buy_num = len(df[df['訊號'].str.contains('買|多|抄')])
                wait_num = len(df) - buy_num
                st.info(f"📊 市場掃描完畢：可操作 {buy_num} 檔，建議觀望 {wait_num} 檔。")
                
            else: st.info("無數據。")
        else: st.error("連線失敗")
