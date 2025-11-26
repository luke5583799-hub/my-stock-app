import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import feedparser
import urllib.parse
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# 引入技術指標運算
from ta.trend import MACD, EMAIndicator, SMAIndicator, IchimokuIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
from ta.volatility import BollingerBands, AverageTrueRange

# ==========================================
# ⚙️ 系統核心配置
# ==========================================
st.set_page_config(page_title="HedgeFund OS | 法人決策系統", layout="wide", page_icon="🏛️")

# 股票清單 (完整版)
SECTORS = {
    "🚀 電子權值": ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2303.TW", "3711.TW", "3008.TW", "3045.TW"],
    "🤖 AI 供應鏈": ["3231.TW", "2356.TW", "6669.TW", "2382.TW", "2376.TW", "3017.TW", "2421.TW", "3035.TW", "3443.TW"],
    "🚢 傳產金融": ["2603.TW", "2609.TW", "2615.TW", "2002.TW", "1605.TW", "1513.TW", "1519.TW", "2881.TW", "2882.TW", "2891.TW", "5880.TW"],
    "📺 面板雙虎": ["3481.TW", "2409.TW"],
    "📊 ETF": ["0050.TW", "0056.TW", "00878.TW", "00929.TW", "00919.TW", "00980A.TW", "00981A.TW", "00982A.TW"],
    "🇺🇸 美股七雄": ["NVDA", "TSLA", "AAPL", "MSFT", "GOOG", "AMZN", "META", "AMD", "INTC", "PLTR", "SMCI", "COIN"]
}

# 映射表 (Ticker -> Name)
NAME_MAP = {
    "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "2308.TW": "台達電", "2303.TW": "聯電",
    "3711.TW": "日月光", "3008.TW": "大立光", "3045.TW": "台灣大", "3231.TW": "緯創", "2356.TW": "英業達",
    "6669.TW": "緯穎", "2382.TW": "廣達", "2376.TW": "技嘉", "3017.TW": "奇鋐", "2421.TW": "建準",
    "3035.TW": "智原", "3443.TW": "創意", "2603.TW": "長榮", "2609.TW": "陽明", "2615.TW": "萬海",
    "2002.TW": "中鋼", "1605.TW": "華新", "1513.TW": "中興電", "1519.TW": "華城", "2881.TW": "富邦金",
    "2882.TW": "國泰金", "2891.TW": "中信金", "5880.TW": "合庫金", "3481.TW": "群創", "2409.TW": "友達",
    "0050.TW": "台灣50", "0056.TW": "高股息", "00878.TW": "國泰永續", "00929.TW": "復華科技", "00919.TW": "群益精選",
    "00980A.TW": "野村趨勢", "00981A.TW": "統一動力", "00982A.TW": "群益強棒",
    "NVDA": "輝達", "TSLA": "特斯拉", "AAPL": "蘋果", "MSFT": "微軟", "GOOG": "谷歌",
    "AMZN": "亞馬遜", "META": "臉書", "AMD": "超微", "INTC": "英特爾", "PLTR": "帕蘭泰爾",
    "SMCI": "美超微", "COIN": "Coinbase"
}

# ==========================================
# 🧱 模組一：數據工廠 (Data Factory)
# ==========================================
class DataEngine:
    @staticmethod
    @st.cache_data(ttl=300)
    def get_market_data(tickers):
        try:
            # 抓取 2 年數據以計算長期指標 (如 200MA, 斐波那契)
            data = yf.download(" ".join(tickers), period="2y", group_by='ticker', progress=False)
            return data
        except Exception as e:
            return None

    @staticmethod
    def get_news_sentiment(ticker):
        name = NAME_MAP.get(ticker, ticker.replace(".TW", ""))
        encoded = urllib.parse.quote(name)
        rss = f"https://news.google.com/rss/search?q={encoded}+when:2d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        try:
            feed = feedparser.parse(rss)
            if not feed.entries: return 0, []
            
            pos_keys = ["營收", "獲利", "新高", "大單", "買超", "漲停", "強勢", "填息", "完銷", "反彈"]
            neg_keys = ["虧損", "衰退", "重挫", "跌停", "利空", "斬倉", "貼息", "下修", "破底"]
            
            score = 0
            headlines = []
            for entry in feed.entries[:3]:
                t = entry.title
                headlines.append(t)
                for w in pos_keys: score += 1
                for w in neg_keys: score -= 1.5
            return score, headlines
        except: return 0, []

# ==========================================
# 🧠 模組二：分析核心 (Alpha Engine)
# ==========================================
class AlphaEngine:
    def __init__(self, ticker, df):
        self.ticker = ticker
        self.df = df.dropna(how='all')
        self.close = self.df['Close']
        self.high = self.df['High']
        self.low = self.df['Low']
        self.volume = self.df['Volume']
        self.name = NAME_MAP.get(ticker, ticker)
        
        # 自動計算指標
        self._calculate_indicators()

    def _calculate_indicators(self):
        # 趨勢
        self.ema20 = EMAIndicator(self.close, window=20).ema_indicator()
        self.ema60 = EMAIndicator(self.close, window=60).ema_indicator()
        self.sma200 = SMAIndicator(self.close, window=200).sma_indicator()
        
        # 動能
        self.rsi = RSIIndicator(self.close).rsi()
        self.macd = MACD(self.close).macd()
        self.signal = MACD(self.close).macd_signal()
        
        # 波動與量能
        self.atr = AverageTrueRange(self.high, self.low, self.close).average_true_range()
        self.obv = OnBalanceVolumeIndicator(self.close, self.volume).on_balance_volume()
        self.bb_low = BollingerBands(self.close, window=20).bollinger_lband()
        self.bb_high = BollingerBands(self.close, window=20).bollinger_hband()

    def get_technical_score(self):
        score = 0
        curr = self.close.iloc[-1]
        
        # 1. 趨勢濾網 (40分)
        if curr > self.ema20.iloc[-1] > self.ema60.iloc[-1]: score += 40
        elif curr > self.ema60.iloc[-1]: score += 20
        
        # 2. 動能濾網 (30分)
        if self.macd.iloc[-1] > self.signal.iloc[-1]: score += 15
        if 50 <= self.rsi.iloc[-1] <= 75: score += 15
        
        # 3. 籌碼/量能 (30分)
        # OBV 趨勢向上 (簡單判斷：現在 OBV > 20天前 OBV)
        if len(self.obv) > 20 and self.obv.iloc[-1] > self.obv.iloc[-20]: score += 30
        
        return score

    def get_rebound_score(self):
        # 專門計算「抄底」分數
        score = 0
        curr = self.close.iloc[-1]
        
        # RSI 超賣
        if self.rsi.iloc[-1] < 30: score += 50
        elif self.rsi.iloc[-1] < 40: score += 30
        
        # 觸碰布林下軌
        if curr <= self.bb_low.iloc[-1]: score += 30
        
        # 乖離過大 (負乖離 > 10%)
        bias = (curr - self.ema60.iloc[-1]) / self.ema60.iloc[-1]
        if bias < -0.1: score += 20
        
        return score

    def get_fibonacci_levels(self):
        # 計算最近半年的高低點
        recent_df = self.df.tail(120)
        max_p = recent_df['High'].max()
        min_p = recent_df['Low'].min()
        diff = max_p - min_p
        
        # 支撐位
        fib_0382 = max_p - (diff * 0.382)
        fib_0500 = max_p - (diff * 0.5)
        fib_0618 = max_p - (diff * 0.618) # 黃金支撐
        
        # 壓力位 (擴展)
        fib_ext_1382 = max_p + (diff * 0.382)
        
        return fib_0618, fib_0382, fib_ext_1382, max_p

# ==========================================
# ⚖️ 模組三：風險與資金管理 (Risk Engine)
# ==========================================
class RiskEngine:
    @staticmethod
    def calculate_kelly(df):
        # 計算過去一年的回測數據來決定凱利倉位
        try:
            daily_ret = df['Close'].pct_change().dropna()
            wins = daily_ret[daily_ret > 0]
            losses = daily_ret[daily_ret < 0]
            
            if len(losses) == 0: return 0.5 # 極端情況
            
            win_rate = len(wins) / len(daily_ret)
            avg_win = wins.mean()
            avg_loss = abs(losses.mean())
            
            odds = avg_win / avg_loss
            kelly = (odds * win_rate - (1 - win_rate)) / odds
            
            # 安全邊際：只用凱利值的 50%
            return max(0, min(kelly * 0.5, 0.5)) 
        except: return 0

# ==========================================
# 📝 模組四：交易執行 (Execution Engine)
# ==========================================
def generate_trade_plan(ticker, df, news_score):
    engine = AlphaEngine(ticker, df)
    
    curr_price = df['Close'].iloc[-1]
    curr_atr = engine.atr.iloc[-1]
    
    t_score = engine.get_technical_score()
    r_score = engine.get_rebound_score()
    fib_support, fib_res1, fib_target, recent_high = engine.get_fibonacci_levels()
    
    # --- 判斷多空方向 ---
    signal = "⚪ 觀望"
    buy_price = 0.0
    stop_loss = 0.0
    take_profit_1 = 0.0
    take_profit_2 = 0.0
    
    # 1. 順勢交易 (Trend Following)
    if t_score >= 60:
        signal = "🔥 強力買進" if t_score >= 80 else "🔴 偏多操作"
        # 順勢買點：回測 MA5 或 突破近期高點
        ma5 = df['Close'].rolling(5).mean().iloc[-1]
        buy_price = ma5
        if curr_price < ma5: buy_price = curr_price # 已經回檔，現價買
        
        # 停損：2倍 ATR
        stop_loss = buy_price - (2 * curr_atr)
        # 停利：前高 或 斐波那契擴展
        take_profit_1 = recent_high
        take_profit_2 = fib_target

    # 2. 逆勢交易 (Reversal)
    elif r_score >= 50:
        signal = "💎 甜蜜抄底"
        # 抄底買點：現價 或 黃金分割支撐
        buy_price = curr_price
        # 停損：跌破布林下軌再下去一點
        stop_loss = engine.bb_low.iloc[-1] - curr_atr
        # 停利：反彈到月線(EMA20)
        take_profit_1 = engine.ema20.iloc[-1]
        take_profit_2 = engine.ema60.iloc[-1]

    # --- 新聞濾網 (Circuit Breaker) ---
    # 如果新聞極差，強制中止買入建議，保留賣出與停損建議
    if news_score <= -2:
        if "抄底" in signal:
            signal = "🩸 恐懼接刀 (高險)" # 允許接刀但警告
        else:
            signal = "⚠️ 有雷 (暫緩)"
            buy_price = 0 # 撤銷買單建議

    # --- 賣出/減碼 邏輯 ---
    sell_note = ""
    if curr_price < (curr_price - 2*curr_atr): # 模擬持有
        sell_note = "🛑 破線快逃"
    elif engine.rsi.iloc[-1] > 75:
        sell_note = "⚠️ 過熱減碼"
    
    # --- 凱利倉位 ---
    kelly = RiskEngine.calculate_kelly(df)
    
    return {
        "ticker": ticker,
        "name": engine.name,
        "price": curr_price,
        "signal": signal,
        "buy_price": buy_price,
        "stop_loss": stop_loss,
        "tp1": take_profit_1,
        "tp2": take_profit_2,
        "kelly": kelly,
        "sell_note": sell_note,
        "score": max(t_score, r_score) + (news_score * 5),
        "rsi": engine.rsi.iloc[-1],
        "engine": engine # 保留物件供繪圖用
    }

# ==========================================
# 📊 模組五：視覺化 (Visualizer)
# ==========================================
def draw_chart(trade_plan):
    engine = trade_plan['engine']
    df = engine.df.tail(150)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # K線
    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                name='Price'), row=1, col=1)
    
    # 均線
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='orange', width=1), name='月線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA60'], line=dict(color='blue', width=1), name='季線'), row=1, col=1)
    
    # 買賣點標示 (如果有建議)
    if trade_plan['buy_price'] > 0:
        fig.add_hline(y=trade_plan['buy_price'], line_dash="dot", line_color="green", annotation_text="建議買點")
    if trade_plan['stop_loss'] > 0:
        fig.add_hline(y=trade_plan['stop_loss'], line_dash="dot", line_color="red", annotation_text="停損點")
    if trade_plan['tp1'] > 0:
        fig.add_hline(y=trade_plan['tp1'], line_dash="dot", line_color="gold", annotation_text="第一目標")

    # 成交量
    colors = ['red' if row['Open'] - row['Close'] >= 0 else 'green' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

    fig.update_layout(
        title=f"{trade_plan['name']} ({trade_plan['ticker']}) 戰略分析圖",
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        height=600,
        template="plotly_dark", # 使用深色主題看起來更專業
        margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig

# ==========================================
# 🚀 主程式
# ==========================================
def main():
    # 側邊欄控制
    with st.sidebar:
        st.header("🎛️ 戰情控制台")
        selected_sector = st.selectbox("選擇板塊", list(SECTORS.keys()))
        run_btn = st.button("🚀 啟動量化運算", type="primary")
        st.divider()
        st.info("本系統採用：\n1. 雙均線趨勢策略\n2. RSI/布林逆勢策略\n3. 凱利公式資金控管\n4. 新聞情緒濾網")

    st.title(f"🏛️ HedgeFund OS | {selected_sector}")

    if run_btn:
        target_tickers = SECTORS[selected_sector]
        
        with st.spinner('正在連線交易所數據庫...'):
            raw_data = DataEngine.get_market_data(target_tickers)
            
            if raw_data is None:
                st.error("數據源連線失敗")
                return

            # 並行運算加速
            results = []
            progress = st.progress(0)
            
            # 新聞分析需要時間，我們只對前幾名做
            # 先做技術分析排序
            pre_results = []
            for t in target_tickers:
                try:
                    # 處理數據結構
                    if isinstance(raw_data.columns, pd.MultiIndex):
                        df = raw_data[t].copy()
                    else:
                        df = raw_data.copy()
                    
                    # 預先計算分數
                    eng = AlphaEngine(t, df)
                    pre_results.append((t, df, max(eng.get_technical_score(), eng.get_rebound_score())))
                except: continue
            
            # 排序後，只取前 10 名或有訊號的去抓新聞 (優化效能)
            pre_results.sort(key=lambda x: x[2], reverse=True)
            
            for i, (ticker, df, raw_score) in enumerate(pre_results):
                # 只對分數高於 40 的抓新聞，節省資源
                n_score = 0
                if raw_score >= 40:
                    n_score, _ = DataEngine.get_news_sentiment(ticker)
                
                plan = generate_trade_plan(ticker, df, n_score)
                if plan: results.append(plan)
                progress.progress((i + 1) / len(pre_results))
            
            progress.empty()

            # --- 顯示層 ---
            if results:
                final_df = pd.DataFrame(results)
                final_df = final_df.sort_values(by='score', ascending=False)
                
                # 1. 總覽表格 (Dashboard)
                st.subheader("📋 戰略總表")
                
                def style_signal(v):
                    if "強力" in v: return 'background-color: #2e7d32; color: white; font-weight: bold'
                    if "偏多" in v: return 'background-color: #e8f5e9; color: #2e7d32'
                    if "甜蜜" in v: return 'background-color: #e3f2fd; color: #1565c0'
                    if "有雷" in v: return 'background-color: #ffebee; color: #c62828; text-decoration: line-through'
                    if "恐懼" in v: return 'background-color: #b71c1c; color: white; font-weight: bold'
                    return 'color: gray'

                st.dataframe(
                    final_df.drop(columns=['ticker', 'score', 'engine', 'sell_note', 'rsi']),
                    use_container_width=True,
                    column_config={
                        "name": st.column_config.TextColumn("股票名稱", width="small"),
                        "price": st.column_config.NumberColumn("現價", format="%.1f"),
                        "signal": st.column_config.TextColumn("AI 判斷", width="medium"),
                        "buy_price": st.column_config.NumberColumn("🎯 建議買點", format="%.1f", help="建議掛單價格"),
                        "tp1": st.column_config.NumberColumn("💰 第一停利", format="%.1f", help="短線目標"),
                        "tp2": st.column_config.NumberColumn("🚀 第二停利", format="%.1f", help="波段目標"),
                        "stop_loss": st.column_config.NumberColumn("🛑 停損價", format="%.1f", help="跌破必跑"),
                        "kelly": st.column_config.ProgressColumn("建議倉位", format="%.0f%%", min_value=0, max_value=1)
                    }
                )
                
                st.markdown("---")

                # 2. 詳細戰術板 (Tactical Board)
                st.subheader("🔍 戰術詳情 & K線圖")
                
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    selected_stock = st.radio("選擇股票查看詳情", final_df['name'] + " (" + final_df['ticker'] + ")")
                    sel_ticker = selected_stock.split("(")[1].replace(")", "")
                    sel_plan = next(p for p in results if p['ticker'] == sel_ticker)
                    
                    # 交易卡片
                    st.info(f"**{sel_plan['name']} 交易計畫**")
                    if sel_plan['buy_price'] > 0:
                        st.markdown(f"🟢 **買進：** {sel_plan['buy_price']:.1f}")
                        st.markdown(f"🔴 **停損：** {sel_plan['stop_loss']:.1f} (-{(sel_plan['price']-sel_plan['stop_loss'])/sel_plan['price']*100:.1f}%)")
                        st.markdown(f"💰 **獲利：** {sel_plan['tp1']:.1f} (+{(sel_plan['tp1']-sel_plan['price'])/sel_plan['price']*100:.1f}%)")
                        
                        risk_reward = (sel_plan['tp1'] - sel_plan['buy_price']) / (sel_plan['buy_price'] - sel_plan['stop_loss'])
                        st.markdown(f"⚖️ **盈虧比：** 1 : {risk_reward:.1f}")
                    else:
                        st.warning("目前不建議進場 (觀望或有雷)")
                    
                    if sel_plan['sell_note']:
                        st.error(f"⚠️ 持有警告：{sel_plan['sell_note']}")

                with col2:
                    # 畫圖
                    st.plotly_chart(draw_chart(sel_plan), use_container_width=True)

            else:
                st.info("目前無符合條件的股票，建議空手觀望。")

if __name__ == "__main__":
    main()
