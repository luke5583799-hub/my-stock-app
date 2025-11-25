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

st.set_page_config(page_title="AI 股市雷達 (全監控)", layout="wide", page_icon="📡")

# ==========================================
# 📋 股票與中文名稱對照表
# ==========================================
STOCK_MAP = {
    # 電子/AI
    "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "2308.TW": "台達電",
    "2303.TW": "聯電", "2382.TW": "廣達", "3711.TW": "日月光", "3034.TW": "聯詠",
    "3035.TW": "智原", "3231.TW": "緯創", "2356.TW": "英業達", "6669.TW": "緯穎",
    "2376.TW": "技嘉", "3017.TW": "奇鋐", "2421.TW": "建準", "2412.TW": "中華電",
    # 傳產/金融
    "2603.TW": "長榮", "2609.TW": "陽明", "2615.TW": "萬海", "2618.TW": "長榮航",
    "2002.TW": "中鋼", "1605.TW": "華新", "1513.TW": "中興電", "1519.TW": "華城",
    "2881.TW": "富邦金", "2882.TW": "國泰金", "2891.TW": "中信金", "2886.TW": "兆豐金",
    "5880.TW": "合庫金",
    # ETF
    "00980A.TW": "野村優選", "00981A.TW": "統一增長", 
    "00982A.TW": "群益強棒", "00983A.TW": "中信ARK",
    "0050.TW": "台灣50", "0056.TW": "高股息", "00878.TW": "國泰永續",
    "00929.TW": "復華科技", "00919.TW": "群益精選",
    # 美股
    "NVDA": "輝達", "TSLA": "特斯拉", "AAPL": "蘋果", "MSFT": "微軟",
    "GOOG": "谷歌", "AMZN": "亞馬遜", "META": "臉書", "AMD": "超微",
    "INTC": "英特爾", "PLTR": "帕蘭泰爾", "SMCI": "美超微"
}

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
# 📰 新聞分析
# ==========================================
def get_news_score(ticker):
    name = STOCK_MAP.get(ticker, ticker.replace(".TW",""))
    encoded_name = urllib.parse.quote(name)
    rss_url = f"https://news.google.com/rss/search?q={encoded_name}+when:5d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    try:
        feed = feedparser.parse(rss_url)
        if not feed.entries: return 0
        pos = ["營收", "獲利", "新高", "大單", "買超", "調升", "漲停", "強勢", "填息", "完銷"]
        neg = ["虧損", "衰退", "賣超", "調降", "重挫", "跌停", "利空", "斬倉", "貼息", "下修"]
        score = 0
        for entry in feed.entries[:5]:
            t = entry.title
            for w in pos: score += 1
            for w in neg: score -= 2
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
        is_etf = ticker.startswith("00") or ticker.endswith("A.TW")

        # 指標
        def safe(func): 
            try: return func()
            except: return pd.Series([0]*len(close))

        ema20 = safe(lambda: EMAIndicator(close=close, window=20).ema_indicator()).iloc[-1]
        ema60 = safe(lambda: EMAIndicator(close=close, window=60).ema_indicator()).iloc[-1]
        rsi = safe(lambda: RSIIndicator(close=close).rsi()).iloc[-1]
        atr = safe(lambda: AverageTrueRange(high=df['High'], low=df['Low'], close=close).average_true_range()).iloc[-1]
        ma5 = close.rolling(5).mean().iloc[-1]

        # 評分
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

        # 預測
        p5, p10, p20 = "-", "-", "-"
        if len(close) > 10:
            x = np.arange(len(close.tail(20)))
            y = close.tail(20).values
            try:
                s, _ = np.polyfit(x, y, 1)
                # 只要不是大跌趨勢，都給預測，方便觀察
                if s > -0.5:
                    p5 = f"{curr + s*5:.1f}"
                    p10 = f"{curr + s*10:.1f}"
                    p20 = f"{curr + s*20:.1f}"
                elif r_score >= 20: # 有一點點反彈跡象就給目標
                    target = ema20 if ema20 > curr else curr*1.03
                    p5 = f"{target:.1f}"
            except: pass

        stop_loss = curr - (2 * atr)
        
        clean_code = ticker.replace(".TW", "")
        stock_name = STOCK_MAP.get(ticker, "")
        display_name = f"{clean_code} {stock_name}"

        return {
            "id": ticker,
            "股票": display_name,
            "現價": round(curr, 1),
            "技術分": t_score + r_score,
            "趨勢分": t_score,
            "抄底分": r_score,
            "MA5": ma5,
            "5日": p5, "10日": p10, "20日": p20,
            "🛑停損": round(stop_loss, 1)
        }
    except: return None

# ==========================================
# 🖥️ 介面
# ==========================================
st.title("📡 AI 股市雷達 (全監控)")

if st.button("🔄 掃描全市場", type="primary"):
    with st.spinner('正在挖掘所有潛在機會...'):
        raw = fetch_data(ALL_STOCKS)
        
        if raw is not None:
            tech_res = []
            for t in ALL_STOCKS:
                r = calculate(t, raw[t])
                if r: tech_res.append(r)
            
            # 只對分數尚可的股票查新聞，省資源
            candidates = [r for r in tech_res if r['技術分'] >= 40]
            
            news_map = {}
            with ThreadPoolExecutor(max_workers=5) as ex:
                future_map = {ex.submit(get_news_score, c['id']): c['id'] for c in candidates}
                for f in future_map:
                    try: news_map[future_map[f]] = f.result()
                    except: news_map[future_map[f]] = 0
            
            final_data = []
            for r in tech_res:
                n_score = news_map.get(r['id'], 0)
                
                signal = "⚪ 弱勢" # 預設
                buy_at = 0.0
                is_etf = r['id'].startswith("00")
                pass_threshold = 50 if is_etf else 60
                watch_threshold = 40 # 觀察門檻

                # --- 訊號分級 ---
                # 1. 及格 (Buy)
                if r['技術分'] >= pass_threshold:
                    if r['趨勢分'] > r['抄底分']:
                        signal = "🔴 偏多"
                        buy_at = r['MA5']
                    else:
                        signal = "💎 甜蜜"
                        buy_at = r['現價']
                    
                    if r['技術分'] >= 80: signal = "🔥 強力"
                
                # 2. 蓄勢 (Watch) - 這是新加的！
                elif r['技術分'] >= watch_threshold:
                    signal = "🟡 蓄勢"
                    buy_at = 0 # 觀察中，暫不建議買

                if r['現價'] < buy_at: buy_at = r['現價']

                # --- 新聞濾網 ---
                if signal != "⚪ 弱勢":
                    if n_score <= -2:
                        if "甜蜜" in signal or "蓄勢" in signal:
                             signal = "🩸 恐懼貪婪" # 越跌越爛越要看
                             buy_at = r['現價']
                        else:
                             signal = "⚠️ 有雷"
                             buy_at = 0
                    elif n_score >= 2:
                         if "蓄勢" in signal: signal = "🔴 轉強(雙確認)" # 觀察股 + 好新聞 = 轉強
                         elif "強力" in signal or "偏多" in signal: signal += "(雙確認)"

                r['💡AI判斷'] = signal
                r['🎯買點'] = round(buy_at, 1) if buy_at > 0 else "-"
                r['_sort'] = r['技術分'] + abs(n_score * 5)
                
                final_data.append(r)

            df = pd.DataFrame(final_data)
            df = df.sort_values(by='_sort', ascending=False)

            tab1, tab2, tab3, tab4 = st.tabs(["🚀 電子", "🚢 金融傳產", "📊 ETF", "🇺🇸 美股"])
            
            def show(s_list):
                sub = df[df['id'].isin(s_list)].copy()
                if not sub.empty:
                    def style(v):
                        if "恐懼" in v: return 'background-color: #8b0000; color: white; font-weight: bold'
                        if "強力" in v: return 'background-color: #ffcccc; color: #8b0000; font-weight: bold'
                        if "雙確認" in v: return 'background-color: #ffcccc; color: #8b0000; font-weight: bold'
                        if "偏多" in v: return 'background-color: #fff5e6; color: #d68910'
                        if "轉強" in v: return 'background-color: #fff5e6; color: #d68910'
                        if "甜蜜" in v: return 'background-color: #e6fffa; color: #006666'
                        if "蓄勢" in v: return 'background-color: #ffffe0; color: #b7950b' # 黃色
                        if "有雷" in v: return 'background-color: #ffe6e6; color: red; text-decoration: line-through'
                        return 'color: #cccccc'

                    st.dataframe(
                        sub.drop(columns=['id', '技術分', '趨勢分', '抄底分', 'MA5', '_sort']),
                        use_container_width=True,
                        column_config={
                            "股票": st.column_config.TextColumn(width="medium"),
                            "現價": st.column_config.NumberColumn(format="%.1f", width="small"),
                            "🎯買點": st.column_config.TextColumn(width="small"),
                            "💡AI判斷": st.column_config.TextColumn(width="medium"),
                            "5日": st.column_config.TextColumn(width="small"),
                            "10日": st.column_config.TextColumn(width="small"),
                            "20日": st.column_config.TextColumn(width="small"),
                            "🛑停損": st.column_config.NumberColumn(format="%.1f", width="small")
                        }
                    )
                else: st.info("無數據")

            with tab1: show(SECTORS["🚀 電子/AI"])
            with tab2: show(SECTORS["🚢 傳產/金融"])
            with tab3: show(SECTORS["📊 ETF"])
            with tab4: show(SECTORS["🇺🇸 美股"])

        else: st.error("連線失敗")
