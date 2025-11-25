import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser # 新增：用來抓新聞
import urllib.parse
from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(page_title="AI 雙核心操盤手", layout="wide", page_icon="🧠")

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
# 📰 新聞情感分析引擎 (NLP Engine)
# ==========================================
def analyze_news_sentiment(ticker):
    # 1. 處理股票名稱 (去掉 .TW 方便搜尋)
    stock_name = ticker.replace(".TW", "")
    if ticker in ["2330.TW"]: stock_name = "台積電"
    elif ticker in ["2317.TW"]: stock_name = "鴻海"
    elif ticker in ["2603.TW"]: stock_name = "長榮"
    # (可自行擴充更多對照表，或直接搜代號)
    
    # 2. 構建 Google News RSS URL
    encoded_name = urllib.parse.quote(stock_name)
    rss_url = f"https://news.google.com/rss/search?q={encoded_name}+when:7d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    
    try:
        feed = feedparser.parse(rss_url)
        if not feed.entries: return 0, "無近期新聞"
        
        # 3. 關鍵字定義
        pos_keywords = ["營收", "獲利", "創高", "成長", "大單", "買超", "調升", "漲停", "利多", "突破", "強勢", "配息", "填息"]
        neg_keywords = ["虧損", "衰退", "賣超", "調降", "重挫", "跌停", "利空", "疲弱", "下修", "斬倉", "貼息"]
        
        score = 0
        latest_title = feed.entries[0].title if feed.entries else ""
        
        # 4. 掃描前 5 則新聞標題
        for entry in feed.entries[:5]:
            title = entry.title
            for w in pos_keywords:
                if w in title: score += 1
            for w in neg_keywords:
                if w in title: score -= 1.5 # 壞消息權重通常比較大
        
        return score, latest_title
    except:
        return 0, "分析失敗"

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

        # --- 1. 技術面運算 ---
        close = df['Close']
        high = df['High']
        low = df['Low']
        curr_price = close.iloc[-1]
        is_etf = ticker.startswith("00")

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

        # 技術評分
        tech_score = 0
        rebound_score = 0
        if curr_price > val_e20 > val_e60: tech_score += 40
        elif curr_price > val_e60: tech_score += 20
        if val_macd > val_sig: tech_score += 20
        if 50 <= curr_rsi <= 75: tech_score += 20
        
        rsi_limit = 45 if is_etf else 40
        if curr_rsi < 30 and curr_rsi > 0: rebound_score += 40
        elif curr_rsi < rsi_limit and curr_rsi > 0: rebound_score += 20
        if curr_price <= bb_lower: rebound_score += 30

        # --- 2. 消息面運算 (News Sentiment) ---
        # 注意：為了速度，這裡我們只在 Streamlit 執行時即時抓取，不在此函式內做大量並行
        # 但為了展示，我們假設這裡呼叫 (實際執行在 UI 層做 Threading 優化)
        news_score = 0 
        news_summary = ""

        # --- 3. 訊號整合 ---
        total_tech_score = tech_score + rebound_score
        
        # 預測
        pred_5_str = "-"
        if len(close) > 10:
            x = np.arange(len(close.tail(20)))
            y = close.tail(20).values
            try:
                slope, _ = np.polyfit(x, y, 1)
                if slope > 0:
                    pred_5_str = f"{curr_price + (slope * 5):.1f}"
                elif rebound_score >= 30:
                    target = val_e20 if val_e20 > curr_price else (curr_price * 1.03)
                    pred_5_str = f"{target:.1f}"
            except: pass

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
            "RSI": int(curr_rsi),
            "🎯買點": round(buy_price, 1) if buy_price > 0 else "-",
            "💡訊號": action,
            "5日": pred_5_str,
            "🛑停損": round(stop_loss, 1),
            "_sort": total_tech_score
        }
    except: return None

# ==========================================
# 🖥️ 介面
# ==========================================
st.title("🧠 AI 雙核心操盤手 (技術+新聞)")
st.caption("同時分析「股價走勢」與「新聞風向」，避開地雷股。")

if st.button("🔄 啟動雙核心掃描 (需時較久)", type="primary"):
    with st.spinner('第一階段：技術面分析中...'):
        raw_data = fetch_all_data(ALL_STOCKS)
        
        if raw_data is not None and not raw_data.empty:
            
            # 第一階段：算出技術面好的股票
            tech_results = []
            for t in ALL_STOCKS:
                res = calculate_metrics(t, raw_data[t])
                if res: tech_results.append(res)
            
            # 建立 DataFrame
            df = pd.DataFrame(tech_results)
            
            if not df.empty:
                # 篩選：只對「有訊號」(非觀望) 的股票去抓新聞，節省時間
                target_stocks = df[df['💡訊號'] != "👀"]
                
                with st.spinner(f'第二階段：正在閱讀 {len(target_stocks)} 檔股票的新聞 (AI NLP)...'):
                    
                    news_scores = {}
                    news_titles = {}
                    
                    # 使用多執行緒抓新聞加速
                    with ThreadPoolExecutor(max_workers=5) as executor:
                        future_to_stock = {executor.submit(analyze_news_sentiment, row['id']): row['id'] for index, row in target_stocks.iterrows()}
                        for future in future_to_stock:
                            stock_id = future_to_stock[future]
                            try:
                                n_score, n_title = future.result()
                                news_scores[stock_id] = n_score
                                news_titles[stock_id] = n_title
                            except:
                                news_scores[stock_id] = 0
                                news_titles[stock_id] = "分析失敗"

                # 將新聞分數合併回 DataFrame
                df['新聞分'] = df['id'].map(news_scores).fillna(0)
                df['最新頭條'] = df['id'].map(news_titles).fillna("-")

                # 最終排序：(技術分 + 新聞分*10)
                df['_final_sort'] = df['技術分'] + (df['新聞分'] * 5)
                df = df.sort_values(by='_final_sort', ascending=False).drop(columns=['_final_sort', 'id'])

                # 樣式設定
                def highlight(val):
                    if "強力" in str(val): return 'background-color: #ffcccc; color: #8b0000; font-weight: bold'
                    if "偏多" in str(val): return 'background-color: #fff5e6; color: #d68910'
                    if "甜蜜" in str(val): return 'background-color: #e6fffa; color: #006666'
                    return 'color: #cccccc'
                
                def highlight_news(val):
                    if val > 0: return 'color: #d63031; font-weight: bold' # 紅字(利多)
                    if val < 0: return 'color: #00b894; font-weight: bold' # 綠字(利空)
                    return 'color: gray'

                # 顯示表格
                st.dataframe(
                    df.style.applymap(highlight, subset=['💡訊號'])
                            .applymap(highlight_news, subset=['新聞分']),
                    use_container_width=True,
                    column_config={
                        "代號": st.column_config.TextColumn(width="small"),
                        "現價": st.column_config.NumberColumn(format="%.1f", width="small"),
                        "技術分": st.column_config.ProgressColumn(format="%d", min_value=0, max_value=100, width="small"),
                        "新聞分": st.column_config.NumberColumn(format="%.1f", help="正分代表利多，負分代表利空", width="small"),
                        "最新頭條": st.column_config.TextColumn(width="large", help="最近一則新聞標題"),
                        "🎯買點": st.column_config.TextColumn(width="small"),
                        "💡訊號": st.column_config.TextColumn(width="small"),
                        "5日": st.column_config.TextColumn(width="small"),
                        "🛑停損": st.column_config.NumberColumn(format="%.1f", width="small")
                    }
                )
                
                st.markdown("""
                ### 📰 如何解讀「新聞分」？
                * **正分 (>0)：** 媒體都在報喜（營收創新高、獲利成長）。與技術面共振，**可安心買進**。
                * **負分 (<0)：** 雖然技術面有訊號，但媒體在報憂（虧損、賣壓）。**小心是「誘多」騙線，建議減少資金或觀望**。
                * **0 分：** 沒新聞或新聞中立，以技術面為主。
                """)
                
            else: st.info("無數據")
            
        else: st.error("連線失敗")
