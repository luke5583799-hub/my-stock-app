import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
import urllib.parse
from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(page_title="AI 雙核心戰情室", layout="wide", page_icon="⚡")

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

# ==========================================
# 📰 新聞情感分析
# ==========================================
def analyze_news_sentiment(ticker):
    stock_name = ticker.replace(".TW", "")
    # 簡單映射常見名稱優化搜尋
    name_map = {"2330": "台積電", "2317": "鴻海", "2603": "長榮", "2454": "聯發科", "3017": "奇鋐"}
    for k, v in name_map.items():
        if k in stock_name: stock_name = v
        
    encoded_name = urllib.parse.quote(stock_name)
    rss_url = f"https://news.google.com/rss/search?q={encoded_name}+when:5d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    
    try:
        feed = feedparser.parse(rss_url)
        if not feed.entries: return 0, "無新聞"
        
        pos_words = ["營收", "獲利", "新高", "大單", "買超", "調升", "漲停", "強勢", "填息", "完銷"]
        neg_words = ["虧損", "衰退", "賣超", "調降", "重挫", "跌停", "利空", "斬倉", "貼息"]
        
        score = 0
        title = feed.entries[0].title[:15] + "..." # 標題只取前15字，避免太長
        
        for entry in feed.entries[:3]: # 只看前3則
            t = entry.title
            for w in pos_words: 
                if w in t: score += 1
            for w in neg_words: 
                if w in t: score -= 1.5
        
        return score, title
    except: return 0, "分析失敗"

# ==========================================
# 🛠️ 核心運算
# ==========================================
@st.cache_data(ttl=300)
def fetch_all_data(tickers):
    tickers_str = " ".join(tickers)
    try:
        return yf.download(tickers_str, period="6mo", group_by='ticker', progress=False)
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

        # 指標運算
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
        curr_ma5 = ma_5.iloc[-1] if not pd.isna(ma_5.iloc[-1]) else curr_price
        curr_atr = atr_val.iloc[-1] if not pd.isna(atr_val.iloc[-1]) else 0
        curr_rsi = rsi_val.iloc[-1] if not pd.isna(rsi_val.iloc[-1]) else 50
        
        val_e20 = ema_20.iloc[-1]
        val_e60 = ema_60.iloc[-1]

        # 評分
        tech_score = 0
        rebound_score = 0
        
        # 1. 趨勢
        if curr_price > val_e20 > val_e60: tech_score += 40
        elif curr_price > val_e60: tech_score += 20
        if macd_val.iloc[-1] > sig_val.iloc[-1]: tech_score += 20
        if 50 <= curr_rsi <= 75: tech_score += 20
        
        # 2. 抄底 (ETF寬容)
        rsi_limit = 45 if is_etf else 40
        if curr_rsi < 30 and curr_rsi > 0: rebound_score += 40
        elif curr_rsi < rsi_limit and curr_rsi > 0: rebound_score += 20
        if curr_price <= bb_lower: rebound_score += 30

        total_tech_score = tech_score + rebound_score

        # 預測 (5日/10日/20日)
        p5, p10, p20 = "-", "-", "-"
        if len(close) > 10:
            x = np.arange(len(close.tail(20)))
            y = close.tail(20).values
            try:
                slope, _ = np.polyfit(x, y, 1)
                if slope > 0:
                    p5 = f"{curr_price + (slope * 5):.1f}"
                    p10 = f"{curr_price + (slope * 10):.1f}"
                    p20 = f"{curr_price + (slope * 20):.1f}"
                elif rebound_score >= 30:
                    target = val_e20 if val_e20 > curr_price else (curr_price * 1.03)
                    p5 = f"{target:.1f}"
            except: pass

        # 訊號
        action = "👀"
        buy_price = 0.0
        buy_threshold = 50 if is_etf else 60

        if tech_score >= 80:
            action = "🔥 強力"
            buy_price = curr_price
        elif total_tech_score >= buy_threshold:
            if tech_score > rebound_score:
                action = "🔴 偏多"
                buy_price = curr_ma5
            else:
                action = "💎 甜蜜"
                buy_price = curr_price
            if curr_price < buy_price: buy_price = curr_price

        stop_loss = curr_price - (2 * curr_atr)

        return {
            "id": ticker,
            "代號": ticker.replace(".TW", ""),
            "現價": round(curr_price, 1),
            "技術分": total_tech_score,
            "新聞": 0, # 先佔位
            "頭條": "-",
            "🎯買點": round(buy_price, 1) if buy_price > 0 else "-",
            "💡訊號": action,
            "5日": p5,
            "10日": p10,
            "20日": p20,
            "🛑停損": round(stop_loss, 1),
            "_sort": total_tech_score
        }
    except: return None

# ==========================================
# 🖥️ 介面
# ==========================================
st.title("⚡ AI 雙核心戰情室 (精簡版)")

if st.button("🔄 掃描全市場", type="primary"):
    with st.spinner('技術分析 + 新聞掃描中...'):
        raw_data = fetch_all_data(ALL_STOCKS)
        
        if raw_data is not None and not raw_data.empty:
            
            # 第一階段：技術運算
            all_res = []
            for t in ALL_STOCKS:
                r = calculate_metrics(t, raw_data[t])
                if r: all_res.append(r)
            
            df_all = pd.DataFrame(all_res)
            
            # 第二階段：只針對有訊號的股票抓新聞
            if not df_all.empty:
                targets = df_all[df_all['💡訊號'] != "👀"]
                
                news_map = {}
                title_map = {}
                
                with ThreadPoolExecutor(max_workers=5) as executor:
