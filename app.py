import streamlit as st
import yfinance as yf
import pandas as pd
from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# ⚙️ 頁面設定
# ==========================================
st.set_page_config(page_title="即時股市戰情室", layout="wide", page_icon="📈")

# ==========================================
# 📋 監控清單 (包含台股與美股熱門)
# ==========================================
DEFAULT_STOCKS = [
    # 台股權值
    "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2303.TW", "2382.TW", "3711.TW", "3034.TW", "3035.TW", 
    "3231.TW", "2356.TW", "6669.TW", "2376.TW", "3017.TW", "2421.TW",
    # 傳產/航運/重電
    "2603.TW", "2609.TW", "2615.TW", "2618.TW", "2002.TW", "1605.TW", "1513.TW", "1519.TW",
    # 金融
    "2881.TW", "2882.TW", "2891.TW", "2886.TW", "5880.TW",
    # ETF
    "0050.TW", "0056.TW", "00878.TW", "00929.TW", "00919.TW",
    # 美股
    "NVDA", "TSLA", "AAPL", "MSFT", "GOOG", "AMZN", "META", "AMD", "INTC", "PLTR", "MSTR", "COIN", "SMCI"
]

# ==========================================
# 🧠 核心邏輯
# ==========================================
def analyze_stock(ticker):
    try:
        # 下載數據 (只抓半年)
        df = yf.download(ticker, period="6mo", progress=False)
        if df.empty or len(df) < 60: return None
        if isinstance(df.columns, pd.MultiIndex): df = df.xs(ticker, axis=1, level=1)
        
        # 指標運算
        close = df['Close']
        ema_20 = EMAIndicator(close=close, window=20).ema_indicator()
        ema_60 = EMAIndicator(close=close, window=60).ema_indicator()
        macd = MACD(close=close)
        rsi = RSIIndicator(close=close).rsi()
        bb = BollingerBands(close=close, window=20, window_dev=2)
        atr = AverageTrueRange(high=df['High'], low=df['Low'], close=close).average_true_range()
        
        curr_price = close.iloc[-1]
        curr_rsi = rsi.iloc[-1]
        
        # 評分
        trend_score = 0
        rebound_score = 0
        
        # 順勢邏輯
        if curr_price > ema_20.iloc[-1] > ema_60.iloc[-1]: trend_score += 40
        elif curr_price > ema_60.iloc[-1]: trend_score += 20
        if macd.macd().iloc[-1] > macd.macd_signal().iloc[-1]: trend_score += 20
        if 50 <= curr_rsi <= 75: trend_score += 20
        
        # 抄底邏輯
        bias = ((curr_price - ema_20.iloc[-1]) / ema_20.iloc[-1]) * 100
        if curr_rsi < 30: rebound_score += 40
        elif curr_rsi < 40: rebound_score += 15
        if curr_price <= bb.bollinger_lband().iloc[-1]: rebound_score += 30
        if bias < -7: rebound_score += 30

        return {
            "代號": ticker,
            "現價": round(curr_price, 2),
            "順勢分": trend_score,
            "抄底分": rebound_score,
            "RSI": round(curr_rsi, 1),
            "建議停損": round(curr_price - 2 * atr.iloc[-1], 2)
        }
    except:
        return None

def fetch_data_parallel(stock_list):
    results = []
    # 使用 8 條執行緒平行抓取，加快速度
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(analyze_stock, t): t for t in stock_list}
        for future in futures:
            res = future.result()
            if res: results.append(res)
    return results

# ==========================================
# 🖥️ 網頁介面
# ==========================================
st.title("🛡️ 全方位股市戰情室 (Live)")
st.markdown("伺服器將在您點擊按鈕時，**即時**向 Yahoo Finance 抓取最新股價。")

if st.button("🔄 立即更新所有股價", type="primary"):
    with st.spinner('正在掃描 50+ 檔股票... (約需 5-10 秒)'):
        data = fetch_data_parallel(DEFAULT_STOCKS)
        df = pd.DataFrame(data)
        
        if not df.empty:
            # 樣式處理函式
            def highlight_scores(val):
                if val >= 80: return 'background-color: #ffcccc; color: black' # 紅底 (強)
                if val >= 60: return 'background-color: #ffe6e6; color: black' 
                return ''
            
            def highlight_rebound(val):
                if val >= 60: return 'background-color: #ccffcc; color: black' # 綠底 (超跌)
                return ''

            col1, col2 = st.columns(2)
            
            # 強勢股區塊
            with col1:
                st.subheader("🔥 強勢多頭 (順勢)")
                trend_df = df[df['順勢分'] >= 60].sort_values(by='順勢分', ascending=False)
                if not trend_df.empty:
                    st.dataframe(trend_df.style.applymap(highlight_scores, subset=['順勢分']), use_container_width=True)
                else:
                    st.info("無強勢股")

            # 超跌股區塊
            with col2:
                st.subheader("💎 超跌機會 (抄底)")
                rebound_df = df[df['抄底分'] >= 60].sort_values(by='抄底分', ascending=False)
                if not rebound_df.empty:
                    st.dataframe(rebound_df.style.applymap(highlight_rebound, subset=['抄底分']), use_container_width=True)
                else:
                    st.info("無超跌股")

            st.markdown("---")
            st.subheader("📋 完整監控清單")
            st.dataframe(df, use_container_width=True)
        else:
            st.error("數據抓取失敗，請稍後再試。")