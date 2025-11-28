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

# 📚 技術指標庫
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

st.set_page_config(page_title="AI 長線暴利獵人 (v6.0)", layout="wide", page_icon="🦄")

st.markdown("""
<style>
    .stPlotlyChart { width: 100%; }
    div[data-testid="stMetric"] { background-color: #1e2130; padding: 15px; border-radius: 5px; border: 1px solid #444; }
    .info-card {
        background-color: #f0f2f6; 
        padding: 20px; 
        border-radius: 10px; 
        border: 1px solid #e0e0e0;
        color: #333333;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 📋 成長股清單
# ==========================================
SECTORS = {
    "🦄 AI 核心與伺服器": ["2330.TW", "2317.TW", "2454.TW", "2382.TW", "3231.TW", "6669.TW", "2356.TW", "2376.TW", "3017.TW", "2421.TW", "3443.TW", "3661.TW", "6962.TW"],
    "⚡ 重電與能源飆股": ["1513.TW", "1519.TW", "1503.TW", "1504.TW", "1609.TW", "6806.TW", "9958.TW"],
    "👁️ 光學與矽智財": ["3008.TW", "3406.TW", "3529.TW", "3035.TW", "6531.TW", "3227.TW", "8069.TW"],
    "🇺🇸 美股破壞式創新": ["NVDA", "TSLA", "AAPL", "MSFT", "GOOG", "AMZN", "META", "AMD", "PLTR", "SMCI", "COIN", "ARM", "MSTR"]
}

NAME_MAP = {
    "2330.TW": "台積電", "2454.TW": "聯發科", "3661.TW": "世芯-KY", "3443.TW": "創意",
    "2317.TW": "鴻海", "2382.TW": "廣達", "3231.TW": "緯創", "6669.TW": "緯穎", "2356.TW": "英業達",
    "2376.TW": "技嘉", "3017.TW": "奇鋐", "2421.TW": "建準", "6962.TW": "AMAX",
    "1513.TW": "中興電", "1519.TW": "華城", "1503.TW": "士電", "1504.TW": "東元", "1609.TW": "大亞", "6806.TW": "森崴", "9958.TW": "世紀鋼",
    "3008.TW": "大立光", "3406.TW": "玉晶光", "3529.TW": "力旺", "3035.TW": "智原", "6531.TW": "愛普", "3227.TW": "原相", "8069.TW": "元太",
    "NVDA": "輝達", "TSLA": "特斯拉", "AAPL": "蘋果", "MSFT": "微軟", "GOOG": "谷歌",
    "AMZN": "亞馬遜", "META": "臉書", "AMD": "超微", "PLTR": "帕蘭泰爾", "SMCI": "美超微", "COIN": "Coinbase", "ARM": "安謀", "MSTR": "微策略"
}

# ==========================================
# 🧱 數據層
# ==========================================
class DataService:
    @staticmethod
    @st.cache_data(ttl=600)
    def get_batch_data(tickers):
        try:
            return yf.download(" ".join(tickers), period="2y", group_by='ticker', progress=False)
        except: return None

    @staticmethod
    def get_news_sentiment(ticker):
        name = NAME_MAP.get(ticker, ticker.replace(".TW", ""))
        encoded = urllib.parse.quote(name)
        rss = f"https://news.google.com/rss/search?q={encoded}+when:7d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        try:
            feed = feedparser.parse(rss)
            if not feed.entries: return 0, []
            pos = ["營收創新高", "獲利翻倍", "擴廠", "訂單爆滿", "調升目標", "殖利率", "成長", "轉虧為盈", "大漲", "強勢"]
            neg = ["衰退", "砍單", "下修", "利空", "違約", "假帳", "掏空", "調查", "重挫"]
            score = 0
            headlines = []
            for entry in feed.entries[:3]:
                t = entry.title
                headlines.append({"title": t, "link": entry.link})
                for w in pos: score += 2
                for w in neg: score -= 3
            return score, headlines
        except: return 0, []

# ==========================================
# 🧠 長線成長分析核心
# ==========================================
class GrowthAnalyzer:
    def __init__(self, ticker, df):
        self.ticker = ticker
        self.df = df.dropna(how='all').copy()
        self.close = self.df['Close']
        self.name = NAME_MAP.get(ticker, ticker)
        
        self.df.fillna(method='bfill', inplace=True)
        self.df.fillna(method='ffill', inplace=True)
        
        self.ema60 = EMAIndicator(self.close, window=60).ema_indicator()
        self.sma240 = SMAIndicator(self.close, window=240).sma_indicator()
        self.rsi = RSIIndicator(self.close, window=14).rsi()
        self.atr = AverageTrueRange(self.df['High'], self.df['Low'], self.close).average_true_range()

    def calculate_yearly_potential(self):
        try:
            recent_data = self.close.tail(120)
            if len(recent_data) < 60: return 0, 0
            
            x = np.arange(len(recent_data))
            y = recent_data.values
            slope, intercept = np.polyfit(x, y, 1)
            
            curr_price = self.close.iloc[-1]
            projected_price = curr_price + (slope * 252 * 0.8)
            
            potential_return = ((projected_price - curr_price) / curr_price) * 100
            return potential_return, projected_price
        except: return 0, 0

    def get_long_term_signal(self, news_score):
        curr = self.close.iloc[-1]
        ma60 = self.ema60.iloc[-1]
        ma240 = self.sma240.iloc[-1]
        rsi = self.rsi.iloc[-1]
        
        if curr < ma240: return "❄️ 空頭 (勿碰)", 0
            
        strength = 0
        if curr > ma60: strength += 1
        if ma60 > ma240: strength += 1
        
        is_dip = False
        if strength >= 2 and 40 <= rsi <= 60: is_dip = True
            
        if news_score <= -3: return "⚠️ 基本面有雷", 0
        
        if is_dip: return "💎 黃金回檔 (最佳)", 90
        if strength >= 2: return "🔥 強勢持有", 80
        if strength == 1: return "🟡 盤整/蓄勢", 60
        
        return "⚪ 觀察", 50

# ==========================================
# 🚀 主程式邏輯
# ==========================================
def main():
    with st.sidebar:
        st.header("🦄 長線暴利獵人")
        selected_sector = st.radio("選擇賽道", list(SECTORS.keys()))

    st.title(f"🚀 {selected_sector} - 長線潛力評估")

    with st.spinner('正在計算年化報酬率與成長潛力...'):
        tickers = SECTORS[selected_sector]
        raw_data = DataService.get_batch_data(tickers)
        
        if raw_data is None:
            st.error("數據連線失敗")
            return

        results = []
        progress = st.progress(0)
        
        for i, ticker in enumerate(tickers):
            try:
                if isinstance(raw_data.columns, pd.MultiIndex): df = raw_data[ticker].copy()
                else: df = raw_data.copy()
                
                analyzer = GrowthAnalyzer(ticker, df)
                potential_pct, target_price = analyzer.calculate_yearly_potential()
                
                news_score = 0
                if potential_pct > 10: 
                    news_score, _ = DataService.get_news_sentiment(ticker)
                
                signal, score = analyzer.get_long_term_signal(news_score)
                
                ma20 = df['Close'].rolling(20).mean().iloc[-1]
                ma60 = df['Close'].rolling(60).mean().iloc[-1]
                
                buy_zone = ma20
                if "回檔" in signal: buy_zone = ma60
                
                stop_loss = analyzer.close.iloc[-1] * 0.85

                results.append({
                    "ticker": ticker,
                    "name": analyzer.name,
                    "price": analyzer.close.iloc[-1],
                    "potential": potential_pct,
                    "target_1y": target_price,
                    "signal": signal,
                    "buy_at": buy_zone,
                    "stop": stop_loss,
                    "score": score + (potential_pct * 0.5),
                    "analyzer": analyzer
                })
            except: pass
            progress.progress((i + 1) / len(tickers))
        
        progress.empty()

        if results:
            df_res = pd.DataFrame(results)
            df_res = df_res.sort_values(by='potential', ascending=False)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("🏆 年度潛力排行榜")
                
                def style_potential(v):
                    if v > 50: return 'color: #00FF00; font-weight: bold; background-color: #1b5e20'
                    if v > 20: return 'color: #2ecc71; font-weight: bold'
                    if v < 0: return 'color: #ff5252'
                    return ''

                st.dataframe(
                    df_res.drop(columns=['ticker', 'score', 'analyzer']),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "name": st.column_config.TextColumn("股票", width="small"),
                        "price": st.column_config.NumberColumn("現價", format="%.1f", width="small"),
                        "potential": st.column_config.NumberColumn("🔥 年化潛力", format="+%.1f%%", help="預估一年後的潛在漲幅"),
                        "target_1y": st.column_config.NumberColumn("💰 1年目標價", format="%.1f"),
                        "signal": st.column_config.TextColumn("長線判斷", width="medium"),
                        "buy_at": st.column_config.NumberColumn("🎯 建議佈局價", format="%.1f", help="建議掛單位置"),
                        "stop": st.column_config.NumberColumn("🛡️ 寬停損(15%)", format="%.1f")
                    }
                )

            with col2:
                st.subheader("📈 趨勢透視")
                selected_name = st.selectbox("選擇股票", df_res['name'] + " (" + df_res['ticker'] + ")")
                sel_ticker = selected_name.split("(")[1].replace(")", "")
                sel_item = next(item for item in results if item['ticker'] == sel_ticker)
                
                st.markdown(f"""
                <div class="info-card">
                    <h3 style="color:#d63384">{sel_item['name']}</h3>
                    <p><b>🚀 潛在漲幅：</b> {sel_item['potential']:.1f}%</p>
                    <p><b>💰 1年後目標：</b> {sel_item['target_1y']:.1f}</p>
                    <hr>
                    <p><b>💡 策略：</b> {sel_item['signal']}</p>
                    <p><b>🛒 建議佈局：</b> {sel_item['buy_at']:.1f} (分批買)</p>
                </div>
                """, unsafe_allow_html=True)
                
                analyzer = sel_item['analyzer']
                df_chart = analyzer.df.tail(250)
                
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=df_chart.index,
                                open=df_chart['Open'], high=df_chart['High'],
                                low=df_chart['Low'], close=df_chart['Close'], name='日K'))
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Close'].rolling(60).mean(), 
                                line=dict(color='orange', width=2), name='季線'))
                fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['Close'].rolling(240).mean(), 
                                line=dict(color='blue', width=2), name='年線'))
                
                fig.update_layout(height=400, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0), xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("連線失敗")

if __name__ == "__main__":
    main()
