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
from ta.trend import MACD, EMAIndicator, SMAIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volume import OnBalanceVolumeIndicator
from ta.volatility import BollingerBands, AverageTrueRange

# ==========================================
# ⚙️ 系統核心配置
# ==========================================
st.set_page_config(page_title="HedgeFund OS | 法人決策系統", layout="wide", page_icon="🏛️")

# 股票清單
SECTORS = {
    "🚀 電子權值": ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2303.TW", "3711.TW", "3008.TW", "3045.TW"],
    "🤖 AI 供應鏈": ["3231.TW", "2356.TW", "6669.TW", "2382.TW", "2376.TW", "3017.TW", "2421.TW", "3035.TW", "3443.TW"],
    "🚢 傳產金融": ["2603.TW", "2609.TW", "2615.TW", "2002.TW", "1605.TW", "1513.TW", "1519.TW", "2881.TW", "2882.TW", "2891.TW", "5880.TW"],
    "📺 面板雙虎": ["3481.TW", "2409.TW"],
    "📊 ETF": ["0050.TW", "0056.TW", "00878.TW", "00929.TW", "00919.TW", "00980A.TW", "00981A.TW", "00982A.TW"],
    "🇺🇸 美股七雄": ["NVDA", "TSLA", "AAPL", "MSFT", "GOOG", "AMZN", "META", "AMD", "INTC", "PLTR", "SMCI", "COIN"]
}

# 映射表
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
# 🧱 模組一：數據工廠
# ==========================================
class DataEngine:
    @staticmethod
    @st.cache_data(ttl=300)
    def get_market_data(tickers):
        try:
            data = yf.download(" ".join(tickers), period="2y", group_by='ticker', progress=False)
            return data
        except: return None

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
                for w in neg_keys: score -= 1
            return score, headlines
        except: return 0, []

# ==========================================
# 🧠 模組二：分析核心
# ==========================================
class AlphaEngine:
    def __init__(self, ticker, df):
        self.ticker = ticker
        self.df = df.dropna(how='all').copy() # 確保是副本
        self.close = self.df['Close']
        self.high = self.df['High']
        self.low = self.df['Low']
        self.volume = self.df['Volume']
        self.name = NAME_MAP.get(ticker, ticker)
        self._calculate_indicators()

    def _calculate_indicators(self):
        try:
            # 使用 fillna 避免畫圖時因為 NaN 報錯
            self.df['EMA20'] = EMAIndicator(self.close, window=20).ema_indicator().fillna(method='bfill')
            self.df['EMA60'] = EMAIndicator(self.close, window=60).ema_indicator().fillna(method='bfill')
            
            macd = MACD(self.close)
            self.df['MACD'] = macd.macd().fillna(0)
            self.df['Signal'] = macd.macd_signal().fillna(0)
            
            self.df['RSI'] = RSIIndicator(self.close).rsi().fillna(50)
            
            bb = BollingerBands(self.close, window=20, window_dev=2)
            self.df['BB_Low'] = bb.bollinger_lband().fillna(self.close)
            self.df['BB_High'] = bb.bollinger_hband().fillna(self.close)
            
            self.df['ATR'] = AverageTrueRange(self.high, self.low, self.close).average_true_range().fillna(0)
            self.df['OBV'] = OnBalanceVolumeIndicator(self.close, self.volume).on_balance_volume().fillna(0)
        except: pass

    def get_scores(self):
        t_score = 0
        r_score = 0
        
        try:
            curr = self.close.iloc[-1]
            ema20 = self.df['EMA20'].iloc[-1]
            ema60 = self.df['EMA60'].iloc[-1]
            
            # 趨勢分
            if curr > ema20 > ema60: t_score += 40
            elif curr > ema60: t_score += 20
            
            # 動能分
            if self.df['MACD'].iloc[-1] > self.df['Signal'].iloc[-1]: t_score += 15
            rsi = self.df['RSI'].iloc[-1]
            if 50 <= rsi <= 75: t_score += 15
            
            # 抄底分
            if rsi < 30: r_score += 50
            elif rsi < 40: r_score += 30
            if curr <= self.df['BB_Low'].iloc[-1]: r_score += 30
            
        except: pass
        return t_score, r_score

    def get_risk_metrics(self):
        try:
            ret = self.close.pct_change().dropna()
            vol = ret.std() * np.sqrt(252)
            sharpe = (ret.mean() * 252 - 0.02) / vol if vol > 0 else 0
            cum_ret = (1 + ret).cumprod()
            mdd = (cum_ret.cummax() - cum_ret).max()
            return vol, sharpe, mdd
        except: return 0, 0, 0

# ==========================================
# 📝 模組三：交易執行
# ==========================================
def generate_trade_plan(ticker, df, news_score):
    engine = AlphaEngine(ticker, df)
    
    curr_price = df['Close'].iloc[-1]
    t_score, r_score = engine.get_scores()
    
    # --- 修正買點邏輯：即使分數低，也給出參考支撐 ---
    buy_price = 0.0
    stop_loss = 0.0
    signal = "⚪ 觀望"
    
    ma5 = df['Close'].rolling(5).mean().iloc[-1]
    bb_low = engine.df['BB_Low'].iloc[-1]
    
    # 1. 順勢訊號
    if t_score >= 60:
        signal = "🔥 強力買進" if t_score >= 80 else "🔴 偏多操作"
        buy_price = ma5
        if curr_price < ma5: buy_price = curr_price
        
    # 2. 逆勢訊號
    elif r_score >= 40: # 門檻稍微降低
        signal = "💎 甜蜜抄底"
        buy_price = curr_price
        
    # 3. 弱勢股 (觀望中) -> 給出下方支撐作為參考
    else:
        # 即使觀望，也算出如果跌到哪裡可以接 (例如布林下軌)
        buy_price = bb_low 
        if news_score <= -2: 
            signal = "⚠️ 有雷 (暫緩)"
            buy_price = 0 # 有雷就真的別買了
    
    # 停損計算
    atr = engine.df['ATR'].iloc[-1]
    if buy_price > 0:
        stop_loss = buy_price - (2 * atr)
    else:
        stop_loss = 0

    # 賣出提示
    sell_note = ""
    if curr_price < (curr_price - 2*atr): sell_note = "🛑 破線快逃"
    elif engine.df['RSI'].iloc[-1] > 75: sell_note = "⚠️ 過熱減碼"

    vol, sharpe, mdd = engine.get_risk_metrics()
    
    # 凱利倉位簡化版
    kelly = 0
    if sharpe > 0: kelly = min(sharpe * 0.2, 0.5)

    return {
        "ticker": ticker,
        "name": engine.name,
        "price": curr_price,
        "signal": signal,
        "buy_price": buy_price,
        "stop_loss": stop_loss,
        "kelly": kelly,
        "sell_note": sell_note,
        "score": max(t_score, r_score) + (news_score * 2),
        "vol": vol, "sharpe": sharpe, "mdd": mdd,
        "engine": engine
    }

# ==========================================
# 📊 模組四：圖表 (修復 Bug 版)
# ==========================================
def draw_chart(plan):
    engine = plan['engine']
    df = engine.df.tail(120)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # K線
    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                name='K線'), row=1, col=1)
    
    # 安全加入指標 (檢查欄位是否存在)
    if 'EMA20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='orange', width=1), name='月線'), row=1, col=1)
    if 'EMA60' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['EMA60'], line=dict(color='blue', width=1), name='季線'), row=1, col=1)
    if 'BB_High' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], line=dict(color='gray', width=0.5, dash='dot'), name='布林上'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], line=dict(color='gray', width=0.5, dash='dot'), name='布林下'), row=1, col=1)

    # 成交量
    colors = ['red' if o - c >= 0 else 'green' for o, c in zip(df['Open'], df['Close'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='成交量', opacity=0.3), row=2, col=1)
    
    # 買賣點線
    if plan['buy_price'] > 0:
        fig.add_hline(y=plan['buy_price'], line_dash="dot", line_color="green", annotation_text="買點")
    if plan['stop_loss'] > 0:
        fig.add_hline(y=plan['stop_loss'], line_dash="dot", line_color="red", annotation_text="停損")

    fig.update_layout(height=600, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0), xaxis_rangeslider_visible=False)
    return fig

# ==========================================
# 🚀 主程式
# ==========================================
def main():
    # 側邊欄
    with st.sidebar:
        st.header("🎛️ 戰情控制台")
        selected_sector = st.selectbox("選擇板塊", list(SECTORS.keys()))
        run_btn = st.button("🚀 啟動量化運算", type="primary")

    st.title(f"🏛️ HedgeFund OS | {selected_sector}")

    if run_btn:
        target_tickers = SECTORS[selected_sector]
        
        with st.spinner('正在連線彭博級數據源...'):
            raw_data = DataEngine.get_market_data(target_tickers)
            
            if raw_data is None:
                st.error("數據源連線失敗")
                return

            results = []
            progress = st.progress(0)
            
            pre_results = []
            for t in target_tickers:
                try:
                    if isinstance(raw_data.columns, pd.MultiIndex): df = raw_data[t].copy()
                    else: df = raw_data.copy()
                    
                    eng = AlphaEngine(t, df)
                    # 這裡只做簡單排序，先不抓新聞
                    t_s, r_s = eng.get_scores()
                    pre_results.append((t, df, max(t_s, r_s)))
                except: continue
            
            pre_results.sort(key=lambda x: x[2], reverse=True)
            
            for i, (ticker, df, raw_score) in enumerate(pre_results):
                n_score = 0
                # 只對前段班或有潛力的抓新聞，加快速度
                if raw_score >= 30: 
                    n_score, _ = DataEngine.get_news_sentiment(ticker)
                
                plan = generate_trade_plan(ticker, df, n_score)
                if plan: results.append(plan)
                progress.progress((i + 1) / len(pre_results))
            
            progress.empty()

            if results:
                final_df = pd.DataFrame(results)
                final_df = final_df.sort_values(by='score', ascending=False)
                
                # 1. 總表
                st.subheader("📋 戰略總表")
                
                def style_sig(v):
                    if "強力" in v: return 'background-color: #2e7d32; color: white'
                    if "偏多" in v: return 'color: #2ecc71'
                    if "甜蜜" in v: return 'color: #29b6f6'
                    if "有雷" in v: return 'color: #ff5252; text-decoration: line-through'
                    return 'color: gray'

                st.dataframe(
                    final_df.drop(columns=['ticker', 'score', 'engine', 'sell_note', 'vol', 'sharpe', 'mdd']),
                    use_container_width=True,
                    column_config={
                        "name": st.column_config.TextColumn("名稱", width="small"),
                        "price": st.column_config.NumberColumn("現價", format="%.1f"),
                        "signal": st.column_config.TextColumn("AI 判斷", width="medium"),
                        "buy_price": st.column_config.NumberColumn("🎯 建議買點", format="%.1f"),
                        "stop_loss": st.column_config.NumberColumn("🛑 停損價", format="%.1f"),
                        "kelly": st.column_config.ProgressColumn("建議倉位", format="%.0f%%", min_value=0, max_value=1)
                    }
                )
                
                st.markdown("---")

                # 2. 詳細戰術板
                st.subheader("🔍 戰術詳情 & K線圖")
                
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    select_list = [f"{row['name']} ({row['ticker']})" for index, row in final_df.iterrows()]
                    selected_stock = st.radio("選擇股票查看詳情", select_list)
                    sel_ticker = selected_stock.split("(")[1].replace(")", "")
                    sel_plan = next(p for p in results if p['ticker'] == sel_ticker)
                    
                    st.info(f"**{sel_plan['name']} 診斷書**")
                    if sel_plan['buy_price'] > 0:
                        st.markdown(f"🟢 **買進：** {sel_plan['buy_price']:.1f}")
                        st.markdown(f"🔴 **停損：** {sel_plan['stop_loss']:.1f}")
                        st.markdown(f"📊 **波動率：** {sel_plan['vol']*100:.1f}%")
                        st.markdown(f"📈 **夏普值：** {sel_plan['sharpe']:.2f}")
                    else:
                        st.warning("⚠️ 風險過高，暫無建議買點")

                with col2:
                    try:
                        st.plotly_chart(draw_chart(sel_plan), use_container_width=True)
                    except Exception as e:
                        st.error(f"圖表繪製失敗: {e}")

            else:
                st.info("目前無數據。")

if __name__ == "__main__":
    main()
