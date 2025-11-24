import streamlit as st
import yfinance as yf
import pandas as pd
from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

st.set_page_config(page_title="直接掛單助手", layout="wide", page_icon="💰")

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
        
        # 計算 5日均線 (作為拉回買點)
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

        # --- 決定掛單價格 ---
        action = "👀 觀望"
        buy_price = 0.0
        note = ""
        
        if trend_score >= 80:
            action = "🔥 強力買進"
            buy_price = curr_price # 強勢股直接市價買
            note = "現價直接買"
        elif trend_score >= 60:
            action = "🔴 偏多操作"
            buy_price = curr_ma5   # 偏多股掛 5日線等它
            note = f"掛 5日線 ({round(buy_price, 2)}) 等接"
            # 如果現價已經低於 5日線，就直接用現價
            if curr_price < buy_price:
                buy_price = curr_price
                note = "現價已低於5日線，可買"
        elif rebound_score >= 60:
            action = "💎 嘗試抄底"
            buy_price = curr_price # 抄底直接現價
            note = "跌深反彈，現價買"

        # 停損建議
        stop_loss = curr_price - (2 * curr_atr)

        return {
            "代號": ticker,
            "現價": round(curr_price, 2),
            "🎯 建議入手價": round(buy_price, 2), # 重點欄位
            "備註": note,
            "建議停損": round(stop_loss, 2),
            "訊號": action,
            "_sort": trend_score + rebound_score
        }
    except: return None

# ==========================================
# 🖥️ 介面
# ==========================================
st.title("💰 直接掛單助手")
st.caption("直接給你價格，不用猜。")

if st.button("🔄 更新掛單價格", type="primary"):
    with st.spinner('正在計算最佳掛單點...'):
        raw_data = fetch_all_data(DEFAULT_STOCKS)
        if raw_data is not None and not raw_data.empty:
            results = []
            for t in DEFAULT_STOCKS:
                try:
                    res = calculate_metrics(t, raw_data[t])
                    if res and "觀望" not in res['訊號']: # 只顯示要買的
                        results.append(res)
                except: continue
            
            df = pd.DataFrame(results)
            if not df.empty:
                df = df.sort_values(by='_sort', ascending=False).drop(columns=['_sort'])
                
                # 樣式
                def highlight(val):
                    if "強力" in val: return 'background-color: #ffcccc; color: #8b0000; font-weight: bold'
                    if "偏多" in val: return 'background-color: #fff5e6; color: #d68910'
                    if "抄底" in val: return 'background-color: #e6fffa; color: #006666'
                    return ''

                st.dataframe(
                    df.style.applymap(highlight, subset=['訊號']),
                    use_container_width=True,
                    column_config={
                        "🎯 建議入手價": st.column_config.NumberColumn(format="%.2f", help="請直接掛這個價格"),
                        "備註": st.column_config.TextColumn(width="medium"),
                        "建議停損": st.column_config.NumberColumn(format="%.2f", help="跌破請務必出場")
                    }
                )
            else: st.info("目前沒有好買點，現金為王。")
        else: st.error("連線失敗")
