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

st.set_page_config(page_title="AI 智能決策", layout="wide", page_icon="🤖")

# ==========================================
# 📋 股票清單
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
# 📰 後台新聞分析 (不顯示，只影響結果)
# ==========================================
def get_news_score(ticker):
    stock_name = ticker.replace(".TW", "")
    # 簡單映射優化搜尋
    name_map = {"2330": "台積電", "2317": "鴻海", "2603": "長榮", "2454": "聯發科", "3017": "奇鋐"}
    for k, v in name_map.items():
        if k in stock_name: stock_name = v
        
    encoded_name = urllib.parse.quote(stock_name)
    rss_url = f"https://news.google.com/rss/search?q={encoded_name}+when:5d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    
    try:
        feed = feedparser.parse(rss_url)
        if not feed.entries: return 0
        
        pos_words = ["營收", "獲利", "新高", "大單", "買超", "調升", "漲停", "強勢", "填息", "完銷"]
        neg_words = ["虧損", "衰退", "賣超", "調降", "重挫", "跌停", "利空", "斬倉", "貼息", "下修"]
        
        score = 0
        for entry in feed.entries[:5]:
            t = entry.title
            for w in pos_words: score += 1
            for w in neg_words: score -= 2 # 壞消息扣分更重，寧可錯殺
        return score
    except: return 0

# ==========================================
# 🛠️ 核心運算
# ==========================================
@st.cache_data(ttl=300)
def fetch_data(tickers):
    try: return yf.download(" ".join(tickers), period="6mo", group_by='ticker', progress=False)
    except: return None

def calculate(ticker, df):
    try:
        if isinstance(df.columns, pd.MultiIndex): df = df.xs(ticker, axis=1, level=0)
        df = df.dropna(how='all')
        if len(df) < 5: return None

        close = df['Close']
        curr = close.iloc[-1]
        is_etf = ticker.startswith("00")

        # 技術指標
        def safe(func): 
            try: return func()
            except: return pd.Series([0]*len(close))

        ema20 = safe(lambda: EMAIndicator(close=close, window=20).ema_indicator()).iloc[-1]
        ema60 = safe(lambda: EMAIndicator(close=close, window=60).ema_indicator()).iloc[-1]
        rsi = safe(lambda: RSIIndicator(close=close).rsi()).iloc[-1]
        atr = safe(lambda: AverageTrueRange(high=df['High'], low=df['Low'], close=close).average_true_range()).iloc[-1]
        ma5 = close.rolling(5).mean().iloc[-1]

        # 基礎評分
        t_score = 0
        r_score = 0
        if curr > ema20 > ema60: t_score += 40
        elif curr > ema60: t_score += 20
        if 50 <= rsi <= 75: t_score += 20
        
        rsi_limit = 45 if is_etf else 40
        if 0 < rsi < 30: r_score += 40
        elif 0 < rsi < rsi_limit: r_score += 20
        try:
            bb_low = BollingerBands(close, window=20, window_dev=2).bollinger_lband().iloc[-1]
            if curr <= bb_low: r_score += 30
        except: pass

        # 預測目標
        p5, p10, p20 = "-", "-", "-"
        if len(close) > 10:
            x = np.arange(len(close.tail(20)))
            y = close.tail(20).values
            try:
                s, _ = np.polyfit(x, y, 1)
                if s > 0:
                    p5 = f"{curr + s*5:.1f}"
                    p10 = f"{curr + s*10:.1f}"
                    p20 = f"{curr + s*20:.1f}"
                elif r_score >= 30:
                    target = ema20 if ema20 > curr else curr*1.03
                    p5 = f"{target:.1f}"
            except: pass

        stop_loss = curr - (2 * atr)

        return {
            "id": ticker,
            "代號": ticker.replace(".TW", ""),
            "現價": round(curr, 1),
            "技術分": t_score + r_score,
            "趨勢分": t_score,
            "抄底分": r_score,
            "MA5": ma5,
            "5日": p5, "10日": p10, "20日": p20,
            "停損": round(stop_loss, 1)
        }
    except: return None

# ==========================================
# 🖥️ 介面
# ==========================================
st.title("🤖 AI 智能決策 (結果導向版)")

if st.button("🔄 AI 決策分析", type="primary"):
    with st.spinner('AI 正在後台分析新聞與技術面...'):
        raw = fetch_data(ALL_STOCKS)
        
        if raw is not None:
            # 1. 技術面篩選
            tech_res = []
            for t in ALL_STOCKS:
                r = calculate(t, raw[t])
                if r: tech_res.append(r)
            
            # 2. 針對有機會的股票，後台查新聞
            candidates = [r for r in tech_res if r['技術分'] >= 40] # 分數太低連新聞都不用查
            
            news_map = {}
            with ThreadPoolExecutor(max_workers=5) as ex:
                future_map = {ex.submit(get_news_score, c['id']): c['id'] for c in candidates}
                for f in future_map:
                    try: news_map[future_map[f]] = f.result()
                    except: news_map[future_map[f]] = 0
            
            # 3. 整合最終判斷
            final_data = []
            for r in tech_res:
                n_score = news_map.get(r['id'], 0)
                
                # --- AI 決策大腦 ---
                signal = "👀"
                buy_at = 0.0
                is_etf = r['id'].startswith("00")
                pass_threshold = 50 if is_etf else 60

                # 基礎判斷
                if r['技術分'] >= pass_threshold:
                    if r['趨勢分'] > r['抄底分']:
                        signal = "🔴 偏多"
                        buy_at = r['MA5']
                    else:
                        signal = "💎 甜蜜"
                        buy_at = r['現價']
                    
                    if r['技術分'] >= 80: signal = "🔥 強力"
                    if r['現價'] < buy_at: buy_at = r['現價']

                # 加入新聞濾網 (最重要的修改)
                if signal != "👀":
                    if n_score <= -2:
                        signal = "⚠️ 有雷勿碰" # 技術面好，但新聞很差 -> 擋下
                        buy_at = 0
                    elif n_score >= 2:
                        signal += "(雙確認)" # 技術面好 + 新聞好 -> 加強信心

                r['💡AI判斷'] = signal
                r['🎯買點'] = round(buy_at, 1) if buy_at > 0 else "-"
                r['_sort'] = r['技術分'] + (n_score * 5)
                
                final_data.append(r)

            df = pd.DataFrame(final_data)
            df = df.sort_values(by='_sort', ascending=False)

            # 4. 顯示
            tab1, tab2, tab3, tab4 = st.tabs(["🚀 電子", "🚢 金融傳產", "📊 ETF", "🇺🇸 美股"])
            
            def show(s_list):
                sub = df[df['id'].isin(s_list)].copy()
                if not sub.empty:
                    # 樣式
                    def style(v):
                        if "強力" in v: return 'background-color: #ffcccc; color: #8b0000; font-weight: bold'
                        if "雙確認" in v: return 'background-color: #ffcccc; color: #8b0000; font-weight: bold'
                        if "偏多" in v: return 'background-color: #fff5e6; color: #d68910'
                        if "甜蜜" in v: return 'background-color: #e6fffa; color: #006666'
                        if "有雷" in v: return 'background-color: #ffe6e6; color: red; text-decoration: line-through'
                        return 'color: #cccccc'

                    st.dataframe(
                        sub.drop(columns=['id', '技術分', '趨勢分', '抄底分', 'MA5', '_sort']),
                        use_container_width=True,
                        column_config={
                            "代號": st.column_config.TextColumn(width="small"),
                            "現價": st.column_config.NumberColumn(format="%.1f", width="small"),
                            "🎯買點": st.column_config.TextColumn(width="small"),
                            "💡AI判斷": st.column_config.TextColumn(width="medium"),
                            "5日": st.column_config.TextColumn(width="small"),
                            "10日": st.column_config.TextColumn(width="small"),
                            "20日": st.column_config.TextColumn(width="small"),
                            "停損": st.column_config.NumberColumn(format="%.1f", width="small")
                        }
                    )
                else: st.info("無數據")

            with tab1: show(SECTORS["🚀 電子/AI"])
            with tab2: show(SECTORS["🚢 傳產/金融"])
            with tab3: show(SECTORS["📊 ETF"])
            with tab4: show(SECTORS["🇺🇸 美股"])

        else: st.error("連線失敗")
