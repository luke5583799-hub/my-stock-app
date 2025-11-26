import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import feedparser
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

# 引入技術指標庫
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands

st.set_page_config(page_title="AI 量化避險基金", layout="wide", page_icon="🏦")

# ==========================================
# 📋 股票清單 (維持不變)
# ==========================================
STOCK_MAP = {
    "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "2308.TW": "台達電",
    "2303.TW": "聯電", "2382.TW": "廣達", "3711.TW": "日月光", "3034.TW": "聯詠",
    "3035.TW": "智原", "3231.TW": "緯創", "2356.TW": "英業達", "6669.TW": "緯穎",
    "2376.TW": "技嘉", "3017.TW": "奇鋐", "2421.TW": "建準", "2412.TW": "中華電",
    "3481.TW": "群創", "2409.TW": "友達",
    "2603.TW": "長榮", "2609.TW": "陽明", "2615.TW": "萬海", "2618.TW": "長榮航",
    "2002.TW": "中鋼", "1605.TW": "華新", "1513.TW": "中興電", "1519.TW": "華城",
    "2881.TW": "富邦金", "2882.TW": "國泰金", "2891.TW": "中信金", "2886.TW": "兆豐金",
    "5880.TW": "合庫金",
    "00980A.TW": "野村優選", "00981A.TW": "統一增長", 
    "00982A.TW": "群益強棒", "00983A.TW": "中信ARK",
    "0050.TW": "台灣50", "0056.TW": "高股息", "00878.TW": "國泰永續",
    "00929.TW": "復華科技", "00919.TW": "群益精選",
    "NVDA": "輝達", "TSLA": "特斯拉", "AAPL": "蘋果", "MSFT": "微軟",
    "GOOG": "谷歌", "AMZN": "亞馬遜", "META": "臉書", "AMD": "超微",
    "INTC": "英特爾", "PLTR": "帕蘭泰爾", "SMCI": "美超微"
}

SECTORS = {
    "🚀 電子/AI": ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2303.TW", "2382.TW", "3711.TW", "3034.TW", "3035.TW", "3231.TW", "2356.TW", "6669.TW", "2376.TW", "3017.TW", "2421.TW", "2412.TW", "3481.TW", "2409.TW"],
    "🚢 傳產/金融": ["2603.TW", "2609.TW", "2615.TW", "2618.TW", "2002.TW", "1605.TW", "1513.TW", "1519.TW", "2881.TW", "2882.TW", "2891.TW", "2886.TW", "5880.TW"],
    "📊 ETF": ["00980A.TW", "00981A.TW", "00982A.TW", "00983A.TW", "0050.TW", "0056.TW", "00878.TW", "00929.TW", "00919.TW"],
    "🇺🇸 美股": ["NVDA", "TSLA", "AAPL", "MSFT", "GOOG", "AMZN", "META", "AMD", "INTC", "PLTR", "SMCI"]
}
ALL_STOCKS = [item for sublist in SECTORS.values() for item in sublist]

# ==========================================
# 🧠 核心：歷史回測引擎 (Backtesting Engine)
# ==========================================
def run_backtest(close_prices):
    """
    簡單回測：模擬過去一年，如果用 '站上20日線買，跌破20日線賣' 的策略，
    這支股票的勝率和賠率是多少？用來判斷『股性』。
    """
    try:
        ma20 = close_prices.rolling(20).mean()
        # 訊號：1為持有，0為空手
        signals = (close_prices > ma20).astype(int)
        # 交易點：1為買入，-1為賣出
        trades = signals.diff()
        
        entry_price = 0
        profits = []
        wins = 0
        losses = 0
        
        buy_indices = trades[trades == 1].index
        sell_indices = trades[trades == -1].index
        
        # 確保買賣配對
        if len(sell_indices) > 0 and len(buy_indices) > 0:
            if sell_indices[0] < buy_indices[0]: sell_indices = sell_indices[1:]
            
        loop_len = min(len(buy_indices), len(sell_indices))
        
        for i in range(loop_len):
            buy_p = close_prices[buy_indices[i]]
            sell_p = close_prices[sell_indices[i]]
            profit = (sell_p - buy_p) / buy_p
            profits.append(profit)
            if profit > 0: wins += 1
            else: losses += 1
            
        if len(profits) == 0: return 0, 0, 0 # 無交易
        
        win_rate = wins / len(profits)
        avg_win = np.mean([p for p in profits if p > 0]) if wins > 0 else 0
        avg_loss = abs(np.mean([p for p in profits if p <= 0])) if losses > 0 else 0.01
        odds = avg_win / avg_loss # 賠率 (賺賠比)
        
        # 凱利公式 (Kelly Criterion) -> 建議倉位
        # f = (bp - q) / b
        kelly = 0
        if odds > 0:
            kelly = (odds * win_rate - (1 - win_rate)) / odds
        
        # 保守調整：凱利值通常太激進，我們取一半 (Half Kelly)
        kelly = max(0, kelly * 0.5)
        
        return win_rate, odds, kelly
    except:
        return 0, 0, 0

# ==========================================
# 📰 新聞分析
# ==========================================
def get_news_score(ticker):
    name = STOCK_MAP.get(ticker, ticker.replace(".TW",""))
    encoded_name = urllib.parse.quote(name)
    rss_url = f"https://news.google.com/rss/search?q={encoded_name}+when:2d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    try:
        feed = feedparser.parse(rss_url)
        if not feed.entries: return 0
        
        pos = ["營收", "獲利", "新高", "大單", "買超", "調升", "漲停", "強勢", "填息", "完銷", "反彈", "回升", "大漲", "復甦"]
        neg = ["虧損", "衰退", "調降", "重挫", "跌停", "利空", "斬倉", "貼息", "下修", "破底"]
        
        score = 0
        for entry in feed.entries[:5]:
            t = entry.title
            for w in pos: score += 1
            for w in neg: score -= 1
        return score
    except: return 0

# ==========================================
# 🛠️ 數據獲取
# ==========================================
@st.cache_data(ttl=300)
def fetch_data(tickers):
    try: 
        # 這次我們要抓 1 年 (1y) 的數據來做回測，而不只是 6mo
        return yf.download(" ".join(tickers), period="1y", group_by='ticker', progress=False)
    except: return None

def calculate(ticker, df):
    try:
        if isinstance(df.columns, pd.MultiIndex): df = df.xs(ticker, axis=1, level=0)
        df = df.dropna(how='all')
        if len(df) < 100: return None # 回測需要較長數據

        close = df['Close']
        curr = close.iloc[-1]
        is_etf = ticker.startswith("00") or ticker.endswith("A.TW")

        # 1. 技術指標
        def safe(func): 
            try: return func()
            except: return pd.Series([0]*len(close))

        ema20 = safe(lambda: EMAIndicator(close=close, window=20).ema_indicator()).iloc[-1]
        ema60 = safe(lambda: EMAIndicator(close=close, window=60).ema_indicator()).iloc[-1]
        rsi = safe(lambda: RSIIndicator(close=close).rsi()).iloc[-1]
        atr = safe(lambda: AverageTrueRange(high=df['High'], low=df['Low'], close=close).average_true_range()).iloc[-1]
        ma5 = close.rolling(5).mean().iloc[-1]

        # 2. 執行回測 (Backtest)
        win_rate, odds, kelly_pos = run_backtest(close)

        # 3. 評分系統 (加入勝率權重)
        t_score = 0
        r_score = 0
        
        # 趨勢分
        if curr > ema20 > ema60: t_score += 30
        elif curr > ema60: t_score += 15
        if 50 <= rsi <= 75: t_score += 15
        
        # 股性分 (新功能!)：如果這支股票過去很好賺，加分
        if win_rate > 0.5: t_score += 20 
        if odds > 1.5: t_score += 10

        # 抄底分
        rsi_limit = 45 if is_etf else 40
        if 0 < rsi < 30: r_score += 40
        elif 0 < rsi < rsi_limit: r_score += 20
        
        ma240 = close.rolling(240).mean().iloc[-1]
        if pd.isna(ma240): ma240 = curr
        margin = (ma240 - curr) / ma240 # 乖離率

        # 4. 預測目標
        p5_val, p10_val = 0, 0
        p5, p10, p20 = "-", "-", "-"
        if len(close) > 20:
            x = np.arange(len(close.tail(20)))
            y = close.tail(20).values
            try:
                s, _ = np.polyfit(x, y, 1)
                if s > -10: 
                    p5_val = curr + s*5
                    p10_val = curr + s*10
                    p5 = f"{p5_val:.1f}"
                    p10 = f"{p10_val:.1f}"
                    p20 = f"{curr + s*20:.1f}"
                elif r_score >= 20:
                    target = ema20 if ema20 > curr else curr*1.03
                    p5_val = target
                    p5 = f"{target:.1f}"
            except: pass

        # 停損：使用 ATR (波動率) 動態調整
        stop_loss = curr - (2.5 * atr)

        # --- 🔥 賣出訊號 ---
        sell_signal = "-"
        if curr < stop_loss: sell_signal = "🛑 破線"
        elif rsi > 75: sell_signal = "⚠️ 過熱"
        elif p5_val > 0 and curr >= p5_val: sell_signal = "💰 達標"

        # --- 🚀 買進訊號 ---
        signal = "⚪ 弱勢"
        buy_at = 0.0
        pass_threshold = 50 if is_etf else 60
        watch_threshold = 40

        total_score = t_score + r_score

        if total_score >= pass_threshold:
            if t_score > r_score:
                signal = "🔴 偏多"
                buy_at = ma5
            else:
                signal = "💎 甜蜜"
                buy_at = curr
            
            if total_score >= 80: signal = "🔥 強力"
        elif total_score >= watch_threshold:
            signal = "🟡 蓄勢"
            buy_at = ma5 * 0.98

        if curr < buy_at: buy_at = curr

        return {
            "id": ticker,
            "代號": STOCK_MAP.get(ticker, ticker),
            "現價": round(curr, 1),
            "技術分": total_score,
            "🎯買點": round(buy_at, 1) if buy_at > 0 else "-",
            "💡訊號": signal,
            "⚡賣點": sell_signal,
            "勝率%": f"{win_rate*100:.0f}%", # 新增
            "倉位%": f"{kelly_pos*100:.0f}%", # 新增：建議買多少
            "5日": p5, "10日": p10, "20日": p20,
            "停損": round(stop_loss, 1),
            "_sort": total_score
        }
    except: return None

# ==========================================
# 🖥️ 介面
# ==========================================
st.title("🏦 AI 量化避險基金 (回測+資金控管)")
st.caption("不只看線圖，AI 模擬過去一年交易，告訴你這支股票「股性」好不好，以及該買多少。")

if st.button("🚀 啟動量化運算", type="primary"):
    with st.spinner('正在進行歷史回測與蒙特卡羅模擬...'):
        raw = fetch_data(ALL_STOCKS)
        
        if raw is not None:
            tech_res = []
            for t in ALL_STOCKS:
                r = calculate(t, raw[t])
                if r: tech_res.append(r)
            
            # 新聞過濾
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
                signal = r['💡訊號']
                buy_at = r['🎯買點']

                if signal != "⚪ 弱勢":
                    if n_score <= -4:
                        if "甜蜜" in signal: signal = "🩸 恐懼" 
                        else: 
                            signal = "⚠️ 有雷"
                            buy_at = "-" 
                    elif n_score >= 2:
                         if "蓄勢" in signal: signal = "🔴 轉強"
                         elif "強力" in signal or "偏多" in signal: signal += "(雙確認)"

                r['💡訊號'] = signal
                r['🎯買點'] = buy_at
                r['_sort'] = r['技術分'] + abs(n_score * 5)
                final_data.append(r)

            df = pd.DataFrame(final_data)
            
            if not df.empty:
                df = df.sort_values(by='_sort', ascending=False)
                tab1, tab2, tab3, tab4 = st.tabs(["🚀 電子", "🚢 金融傳產", "📊 ETF", "🇺🇸 美股"])
                
                def show(s_list):
                    sub = df[df['id'].isin(s_list)].copy()
                    if not sub.empty:
                        def style_signal(v):
                            if "恐懼" in v: return 'background-color: #8b0000; color: white; font-weight: bold'
                            if "強力" in v: return 'background-color: #ffcccc; color: #8b0000; font-weight: bold'
                            if "雙確認" in v: return 'background-color: #ffcccc; color: #8b0000; font-weight: bold'
                            if "偏多" in v: return 'background-color: #fff5e6; color: #d68910'
                            if "轉強" in v: return 'background-color: #fff5e6; color: #d68910'
                            if "甜蜜" in v: return 'background-color: #e6fffa; color: #006666'
                            if "蓄勢" in v: return 'background-color: #ffffe0; color: #b7950b'
                            if "有雷" in v: return 'background-color: #ffe6e6; color: red; text-decoration: line-through'
                            return 'color: #cccccc'
                        
                        def style_sell(v):
                            if "破線" in v: return 'color: white; background-color: #ff0000'
                            if "達標" in v: return 'color: #009900; font-weight: bold'
                            if "過熱" in v: return 'color: #ff9900'
                            return 'color: #cccccc'

                        st.dataframe(
                            sub.drop(columns=['id', '技術分', '_sort']),
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "代號": st.column_config.TextColumn(width="small"),
                                "現價": st.column_config.NumberColumn(format="%.1f", width="small"),
                                "🎯買點": st.column_config.TextColumn(width="small"),
                                "💡訊號": st.column_config.TextColumn(width="medium"),
                                "勝率%": st.column_config.TextColumn(width="small", help="過去一年波段操作勝率"),
                                "倉位%": st.column_config.ProgressColumn(format="%s", min_value=0, max_value=100, width="small", help="凱利公式建議資金比例"),
                                "⚡賣點": st.column_config.TextColumn(width="small"),
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
            else:
                st.warning("暫無符合條件數據。")
        else: st.error("連線失敗")
