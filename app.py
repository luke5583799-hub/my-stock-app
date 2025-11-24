import streamlit as st
import yfinance as yf
import pandas as pd
from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import datetime

# ==========================================
# ⚙️ 頁面設定
# ==========================================
st.set_page_config(page_title="穩定版股市戰情室", layout="wide", page_icon="🛡️")

# ==========================================
# 📋 監控清單 (50+ 檔)
# ==========================================
DEFAULT_STOCKS = [
    # 台股
    "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2303.TW", "2382.TW", "3711.TW", "3034.TW", "3035.TW", 
    "3231.TW", "2356.TW", "6669.TW", "2376.TW", "3017.TW", "2421.TW", "2412.TW",
    "2603.TW", "2609.TW", "2615.TW", "2618.TW", "2002.TW", "1605.TW", "1513.TW", "1519.TW",
    "2881.TW", "2882.TW", "2891.TW", "2886.TW", "5880.TW",
    "0050.TW", "0056.TW", "00878.TW", "00929.TW", "00919.TW",
    # 美股
    "NVDA", "TSLA", "AAPL", "MSFT", "GOOG", "AMZN", "META", "AMD", "INTC", "PLTR", "MSTR", "COIN", "SMCI"
]

# ==========================================
# 🛠️ 核心邏輯 (批次處理版)
# ==========================================

# 1. 設置快取：資料會保存 300 秒 (5分鐘)，避免重複抓取被封鎖
@st.cache_data(ttl=300)
def fetch_all_data(tickers):
    # 將列表轉為字串，用空格隔開，一次請求所有數據
    tickers_str = " ".join(tickers)
    try:
        # 下載數據 (group_by='ticker' 讓結構更好處理)
        data = yf.download(tickers_str, period="6mo", group_by='ticker', progress=False)
        return data
    except Exception as e:
        st.error(f"數據下載失敗: {e}")
        return None

def calculate_metrics(ticker, df):
    # 處理單一股票的 DataFrame
    try:
        # 移除多層索引 (如果有)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker, axis=1, level=0)
        
        # 確保數據足夠
        # yfinance 有時會回傳空列，需過濾
        df = df.dropna(how='all') 
        if len(df) < 50: return None

        # 指標運算
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        # 避免全是 NaN 的情況
        if close.isnull().all(): return None

        ema_20 = EMAIndicator(close=close, window=20).ema_indicator()
        ema_60 = EMAIndicator(close=close, window=60).ema_indicator()
        macd = MACD(close=close)
        rsi = RSIIndicator(close=close).rsi()
        bb = BollingerBands(close=close, window=20, window_dev=2)
        atr = AverageTrueRange(high=high, low=low, close=close).average_true_range()
        
        # 取得最新一筆有效數據
        curr_price = close.iloc[-1]
        curr_rsi = rsi.iloc[-1]
        
        # 處理 NaN (例如剛上市或數據缺漏)
        if pd.isna(curr_price) or pd.isna(curr_rsi): return None

        # 評分系統
        trend_score = 0
        rebound_score = 0
        
        # 順勢
        if curr_price > ema_20.iloc[-1] > ema_60.iloc[-1]: trend_score += 40
        elif curr_price > ema_60.iloc[-1]: trend_score += 20
        if macd.macd().iloc[-1] > macd.macd_signal().iloc[-1]: trend_score += 20
        if 50 <= curr_rsi <= 75: trend_score += 20
        
        # 抄底
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
    except Exception:
        return None

# ==========================================
# 🖥️ 網頁介面
# ==========================================
st.title("🛡️ 股市戰情室 (穩定版)")
st.caption(f"監控中: {len(DEFAULT_STOCKS)} 檔股票 | 自動快取: 5 分鐘")

if st.button("🔄 更新數據", type="primary"):
    with st.spinner('正在批次下載數據，請稍候...'):
        # 1. 一次下載所有數據 (Batch Download)
        raw_data = fetch_all_data(DEFAULT_STOCKS)
        
        if raw_data is not None and not raw_data.empty:
            final_results = []
            
            # 2. 逐一計算指標 (純數學運算，不聯網，速度極快)
            progress_bar = st.progress(0)
            total_stocks = len(DEFAULT_STOCKS)
            
            for i, ticker in enumerate(DEFAULT_STOCKS):
                # 提取該股票的數據
                try:
                    # yfinance 的批次結構: raw_data[ticker]
                    stock_df = raw_data[ticker]
                    res = calculate_metrics(ticker, stock_df)
                    if res:
                        final_results.append(res)
                except KeyError:
                    # 某些股票可能下載失敗，直接跳過
                    continue
                
                # 更新進度條
                progress_bar.progress((i + 1) / total_stocks)
            
            progress_bar.empty() # 跑完隱藏進度條

            # 3. 顯示結果
            df_res = pd.DataFrame(final_results)
            
            if not df_res.empty:
                # 樣式設定
                def highlight_trend(val):
                    return 'background-color: #ffcccc; color: black' if val >= 60 else ''
                def highlight_rebound(val):
                    return 'background-color: #ccffcc; color: black' if val >= 60 else ''

                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🔥 強勢多頭")
                    trend_df = df_res[df_res['順勢分'] >= 60].sort_values(by='順勢分', ascending=False)
                    if not trend_df.empty:
                        st.dataframe(trend_df.style.applymap(highlight_trend, subset=['順勢分']), use_container_width=True)
                    else:
                        st.info("無符合條件股票")
                
                with col2:
                    st.subheader("💎 超跌機會")
                    rebound_df = df_res[df_res['抄底分'] >= 60].sort_values(by='抄底分', ascending=False)
                    if not rebound_df.empty:
                        st.dataframe(rebound_df.style.applymap(highlight_rebound, subset=['抄底分']), use_container_width=True)
                    else:
                        st.info("無符合條件股票")

                st.markdown("---")
                st.subheader("📋 所有監控列表")
                st.dataframe(df_res, use_container_width=True)
            else:
                st.error("分析結果為空，可能是所有數據下載失敗或市場休市中。")
        else:
            st.error("無法連接 Yahoo Finance，請稍後再試。")
