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
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

st.set_page_config(page_title="HedgeFund OS | 完美合體版", layout="wide", page_icon="💎")

st.markdown("""
<style>
    .stPlotlyChart { width: 100%; }
    div[data-testid="stMetric"] { background-color: #262730; padding: 15px; border-radius: 5px; border: 1px solid #444; }
    .info-card {
        background-color: #f8f9fa; 
        padding: 20px; 
        border-radius: 10px; 
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: #333333;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 📋 股票清單
# ==========================================
SECTORS = {
    "🚀 電子權值": ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2303.TW", "3711.TW", "3008.TW", "3045.TW"],
    "🤖 AI 供應鏈": ["3231.TW", "2356.TW", "6669.TW", "2382.TW", "2376.TW", "3017.TW", "2421.TW", "3035.TW", "3443.TW", "3317.TW", "6414.TW", "6962.TW"],
    "👁️ 光電與顯示": ["3481.TW", "2409.TW", "3034.TW", "4961.TW", "3545.TW", "8016.TW", "6668.TW", "3673.TW"],
    "⚡ 重電與綠能": ["1513.TW", "1519.TW", "1503.TW", "1504.TW", "1609.TW", "1605.TW", "6806.TW", "9958.TW"],
    "🚢 航運與傳產": ["2603.TW", "2609.TW", "2615.TW", "2618.TW", "2610.TW", "2002.TW", "1101.TW", "1301.TW", "1303.TW"],
    "🏦 金融護城河": ["2881.TW", "2882.TW", "2891.TW", "2886.TW", "2884.TW", "5880.TW", "2892.TW", "2880.TW", "2885.TW"],
    "📊 熱門 ETF": ["0050.TW", "0056.TW", "00878.TW", "00929.TW", "00919.TW", "00940.TW", "006208.TW", "00980A.TW", "00981A.TW", "00982A.TW"],
    "🇺🇸 美股七雄+": ["NVDA", "TSLA", "AAPL", "MSFT", "GOOG", "AMZN", "META", "AMD", "INTC", "PLTR", "SMCI", "COIN"]
}

NAME_MAP = {
    "2330.TW": "台積電", "2454.TW": "聯發科", "3711.TW": "日月光", "3661.TW": "世芯-KY", "3443.TW": "創意",
    "2317.TW": "鴻海", "2382.TW": "廣達", "3231.TW": "緯創", "6669.TW": "緯穎", "2356.TW": "英業達",
    "2376.TW": "技嘉", "3017.TW": "奇鋐", "2421.TW": "建準", "6962.TW": "AMAX",
    "1513.TW": "中興電", "1519.TW": "華城", "1503.TW": "士電", "1504.TW": "東元", "1609.TW": "大亞", "6806.TW": "森崴", "9958.TW": "世紀鋼",
    "3008.TW": "大立光", "3406.TW": "玉晶光", "3529.TW": "力旺", "3035.TW": "智原", "6531.TW": "愛普", "3227.TW": "原相", "8069.TW": "元太",
    "3481.TW": "群創", "2409.TW": "友達", "3034.TW": "聯詠", "4961.TW": "天鈺", "3545.TW": "敦泰", "8016.TW": "矽創", "6668.TW": "中揚光", "3673.TW": "宸鴻",
    "3317.TW": "尼克森", "6414.TW": "樺漢",
    "2603.TW": "長榮", "2609.TW": "陽明", "2615.TW": "萬海", "2618.TW": "長榮航", "2610.TW": "華航",
    "2002.TW": "中鋼", "1101.TW": "台泥", "1301.TW": "台塑", "1303.TW": "南亞", 
    "2881.TW": "富邦金", "2882.TW": "國泰金", "2891.TW": "中信金", "2886.TW": "兆豐金", "5880.TW": "合庫金",
    "2884.TW": "玉山金", "2892.TW": "第一金", "2880.TW": "華南金", "2885.TW": "元大金",
    "0050.TW": "台灣50", "0056.TW": "高股息", "00878.TW": "國泰永續", "00929.TW": "復華科技", "00919.TW": "群益精選",
    "00940.TW": "元大價值", "006208.TW": "富邦台50", "00980A.TW": "野村趨勢", "00981A.TW": "統一動力", "00982A.TW": "群益強棒",
    "NVDA": "輝達", "TSLA": "特斯拉", "AAPL": "蘋果", "MSFT": "微軟", "GOOG": "谷歌",
    "AMZN": "亞馬遜", "META": "臉書", "AMD": "超微", "PLTR": "帕蘭泰爾", "SMCI": "美超微", "COIN": "Coinbase", "ARM": "安謀", "MSTR": "微策略", "INTC": "英特爾"
}

ALL_TICKERS = [t for s in SECTORS.values() for t in s]

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
# 🧠 核心分析
# ==========================================
class Analyzer:
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
        self.bb = BollingerBands(self.close, window=20, window_dev=2)
        
        # MFI (資金流)
        self.mfi = MFIIndicator(self.df['High'], self.df['Low'], self.close, self.df['Volume'], window=14).money_flow_index()

    def calculate_potential(self):
        try:
            recent = self.close.tail(120)
            if len(recent) < 60: return 0
            x = np.arange(len(recent))
            y = recent.values
            s, _ = np.polyfit(x, y, 1)
            curr = self.close.iloc[-1]
            proj = curr + (s * 252 * 0.8)
            return ((proj - curr) / curr) * 100
        except: return 0

    def calculate_fair_value(self):
        # 合理價 = 年線
        val = self.sma240.iloc[-1]
        return val if not pd.isna(val) else self.close.iloc[-1]

    def calculate_kelly(self):
        # 凱利公式
        try:
            ret = self.close.pct_change().dropna().tail(120)
            wins = ret[ret > 0]
            losses = ret[ret < 0]
            if len(wins) == 0: return 0
            win_rate = len(wins) / len(ret)
            avg_win = wins.mean()
            avg_loss = abs(losses.mean())
            odds = avg_win / avg_loss if avg_loss > 0 else 1
            kelly = (odds * win_rate - (1 - win_rate)) / odds
            return max(0, min(kelly * 0.5, 0.5))
        except: return 0

    def get_signal(self, news_score):
        curr = self.close.iloc[-1]
        ma60 = self.ema60.iloc[-1]
        ma240 = self.sma240.iloc[-1]
        rsi = self.rsi.iloc[-1]
        
        if curr < ma240: return "❄️ 空頭"
        
        strength = 0
        if curr > ma60: strength += 1
        if ma60 > ma240: strength += 1
        
        if news_score <= -3: return "⚠️ 有雷"
        
        if strength >= 2 and 40 <= rsi <= 60: return "💎 黃金回檔"
        if strength >= 2: return "🔥 強勢持有"
        if strength == 1: return "🟡 盤整"
        return "⚪ 觀察"

# ==========================================
# 📝 策略生成
# ==========================================
def generate_strategy(ticker, df, news_score):
    az = Analyzer(ticker, df)
    curr = az.close.iloc[-1]
    
    pot = az.calculate_potential()
    fair = az.calculate_fair_value()
    signal = az.get_signal(news_score)
    kelly = az.calculate_kelly()
    
    # 買點：支撐位 (年線 或 季線 或 布林下軌)
    buy = az.sma240.iloc[-1]
    if "回檔" in signal or "強勢" in signal:
        buy = az.ema60.iloc[-1]
    
    # 布林下軌作為最後防線
    bb_low = az.bb.bollinger_lband().iloc[-1]
    if pd.isna(buy): buy = bb_low
    
    stop = curr * 0.85
    target = curr * 1.5 if pot > 50 else curr * 1.2
    
    upside = (fair - curr) / curr * 100

    return {
        "info": {
            "id": az.name,
            "ticker_code": ticker,
            "price": curr,
            "potential": pot,
            "fair_value": fair,
            "upside": upside,
            "signal": signal,
            "buy": buy,
            "stop": stop,
            "target": target,
            "kelly": kelly,
            "rsi": az.rsi.iloc[-1],
            "mfi": az.mfi.iloc[-1]
        },
        "analyzer": az
    }

# ==========================================
# 🎨 畫圖
# ==========================================
def draw_chart(az):
    df = az.df.tail(250)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA60'], line=dict(color='orange'), name='季線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA240'], line=dict(color='blue'), name='年線'), row=1, col=1)
    
    colors = ['red' if o - c >= 0 else 'green' for o, c in zip(df['Open'], df['Close'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='成交量'), row=2, col=1)
    
    fig.update_layout(height=500, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0), xaxis_rangeslider_visible=False)
    return fig

# ==========================================
# 🚀 主程式
# ==========================================
def main():
    with st.sidebar:
        st.header("💎 HedgeFund OS | 合體版")
        selected_sector = st.radio("選擇賽道", list(SECTORS.keys()))

    st.title(f"🚀 {selected_sector} - 戰情室")

    with st.spinner('AI 正在進行價值與動能運算...'):
        tickers = SECTORS[selected_sector]
        raw_data = DataService.get_batch_data(tickers)
        
        if raw_data is None:
            st.error("連線失敗")
            return

        results = []
        progress = st.progress(0)
        
        for i, ticker in enumerate(tickers):
            try:
                if isinstance(raw_data.columns, pd.MultiIndex): df = raw_data[ticker].copy()
                else: df = raw_data.copy()
                
                # 初步篩選：潛力 > 10% 才查新聞
                az_temp = Analyzer(ticker, df)
                pot = az_temp.calculate_potential()
                
                n_score = 0
                if pot > 10: 
                    n_score, _ = DataService.get_news_sentiment(ticker)
                
                res = generate_strategy(ticker, df, n_score)
                results.append(res)
            except: pass
            progress.progress((i + 1) / len(tickers))
        
        progress.empty()

        if results:
            df_res = pd.DataFrame([r['info'] for r in results])
            df_res = df_res.sort_values(by='potential', ascending=False)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("🏆 年度潛力排行榜")
                
                def style_pot(v):
                    if v > 50: return 'color: #00FF00; font-weight: bold'
                    if v < 0: return 'color: #FF5252'
                    return ''

                # 關鍵修改：errors='ignore' 防止 KeyError
                st.dataframe(
                    df_res.drop(columns=['ticker_code', 'mfi', 'sell_note', 'score'], errors='ignore'),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "id": st.column_config.TextColumn("股票", width="small"),
                        "price": st.column_config.NumberColumn("現價", format="%.1f", width="small"),
                        "potential": st.column_config.NumberColumn("🔥 年化潛力", format="%+.1f%%"),
                        "fair_value": st.column_config.NumberColumn("💰 合理價", format="%.1f"),
                        "upside": st.column_config.NumberColumn("空間%", format="%+.1f%%"),
                        "signal": st.column_config.TextColumn("訊號", width="medium"),
                        "buy": st.column_config.NumberColumn("🎯 買點", format="%.1f"),
                        "stop": st.column_config.NumberColumn("🛑 停損", format="%.1f"),
                        "kelly": st.column_config.ProgressColumn("倉位", format="%.0f%%", min_value=0, max_value=1),
                        "rsi": st.column_config.NumberColumn("RSI", format="%.1f"),
                    }
                )

            with col2:
                st.subheader("📈 趨勢透視")
                sel_name = st.selectbox("選擇股票", df_res['id'] + " (" + df_res['ticker_code'] + ")")
                sel_code = sel_name.split("(")[1].replace(")", "")
                sel_item = next(r for r in results if r['info']['ticker_code'] == sel_code)
                info = sel_item['info']
                
                st.markdown(f"""
                <div class="info-card">
                    <h3 style="color:#d63384">{info['id']}</h3>
                    <p><b>🚀 年化潛力：</b> {info['potential']:.1f}%</p>
                    <p><b>💰 合理估值：</b> {info['fair_value']:.1f}</p>
                    <p><b>🌊 RSI 指標：</b> {info['rsi']:.1f}</p>
                    <hr>
                    <p><b>💡 策略：</b> {info['signal']}</p>
                    <p><b>🛒 建議佈局：</b> {info['buy']:.1f}</p>
                    <p><b>🛑 停損防守：</b> {info['stop']:.1f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                fig = draw_chart(sel_item['analyzer'])
                st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("連線失敗")

if __name__ == "__main__":
    main()
