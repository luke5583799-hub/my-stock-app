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

st.set_page_config(page_title="HedgeFund OS | 終極合體版", layout="wide", page_icon="💎")

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
    "1513.TW": "中興電", "1519.TW": "華城", "1503.TW": "士電", "1504.TW": "東元", "1609.TW": "大亞", "1605.TW": "華新", "6806.TW": "森崴", "9958.TW": "世紀鋼",
    "3008.TW": "大立光", "3406.TW": "玉晶光", "3529.TW": "力旺", "3035.TW": "智原", "6531.TW": "愛普", "3227.TW": "原相", "8069.TW": "元太",
    "3481.TW": "群創", "2409.TW": "友達", "3034.TW": "聯詠", "4961.TW": "天鈺", "3545.TW": "敦泰", "8016.TW": "矽創", "6668.TW": "中揚光", "3673.TW": "宸鴻",
    "3317.TW": "尼克森", "6414.TW": "樺漢",
    "2603.TW": "長榮", "2609.TW": "陽明", "2615.TW": "萬海", "2618.TW": "長榮航", "2610.TW": "華航",
    "2002.TW": "中鋼", "1101.TW": "台泥", "1301.TW": "台塑", "1303.TW": "南亞", 
    "2881.TW": "富邦金", "2882.TW": "國泰金", "2891.TW": "中信金", "2886.TW": "兆豐金", "5880.TW": "合庫金",
    "2884.TW": "玉山金", "2892.TW": "第一金", "2880.TW": "華南金", "2885.TW": "元大金",
    "3008.TW": "大立光", "3045.TW": "台灣大", "3034.TW": "聯詠", "3481.TW": "群創", "2409.TW": "友達",
    "4961.TW": "天鈺", "3545.TW": "敦泰", "8016.TW": "矽創", "3673.TW": "宸鴻",
    "2303.TW": "聯電", "2308.TW": "台達電",
    "0050.TW": "台灣50", "0056.TW": "高股息", "00878.TW": "國泰永續", "00929.TW": "復華科技", "00919.TW": "群益精選",
    "00940.TW": "元大價值", "006208.TW": "富邦台50", "00980A.TW": "野村趨勢", "00981A.TW": "統一動力", "00982A.TW": "群益強棒",
    "NVDA": "輝達", "TSLA": "特斯拉", "AAPL": "蘋果", "MSFT": "微軟", "GOOG": "谷歌",
    "AMZN": "亞馬遜", "META": "臉書", "AMD": "超微", "INTC": "英特爾", "PLTR": "帕蘭泰爾",
    "SMCI": "美超微", "COIN": "Coinbase"
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
        rss = f"https://news.google.com/rss/search?q={encoded}+when:2d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        try:
            feed = feedparser.parse(rss)
            if not feed.entries: return 0, []
            pos = ["營收", "獲利", "新高", "大單", "買超", "漲停", "強勢", "填息", "完銷", "反彈", "大漲"]
            neg = ["虧損", "衰退", "重挫", "跌停", "利空", "斬倉", "貼息", "下修", "破底"]
            score = 0
            headlines = []
            for entry in feed.entries[:3]:
                t = entry.title
                headlines.append({"title": t, "link": entry.link})
                for w in pos: score += 1
                for w in neg: score -= 1
            return score, headlines
        except: return 0, []

# ==========================================
# 🧠 分析層
# ==========================================
class QuantAnalyzer:
    def __init__(self, ticker, df):
        self.ticker = ticker
        self.df = df.dropna(how='all').copy()
        self.close = self.df['Close']
        self.high = self.df['High']
        self.low = self.df['Low']
        self.volume = self.df['Volume']
        
        cn_name = NAME_MAP.get(ticker, "")
        clean_ticker = ticker.replace(".TW", "")
        self.display_name = f"{clean_ticker} {cn_name}"
        
        self._add_indicators()
        
    def _add_indicators(self):
        self.df.fillna(method='bfill', inplace=True)
        self.df.fillna(method='ffill', inplace=True)

        self.df['EMA20'] = EMAIndicator(self.close, window=20).ema_indicator()
        self.df['EMA60'] = EMAIndicator(self.close, window=60).ema_indicator()
        self.df['SMA240'] = SMAIndicator(self.close, window=240).sma_indicator()
        
        macd = MACD(self.close)
        self.df['MACD'] = macd.macd().fillna(0)
        self.df['Signal'] = macd.macd_signal().fillna(0)
        self.df['RSI'] = RSIIndicator(self.close).rsi().fillna(50)
        self.df['MFI'] = MFIIndicator(self.high, self.low, self.close, self.volume, window=14).money_flow_index().fillna(50)
        
        bb = BollingerBands(self.close, window=20, window_dev=2)
        self.df['BB_High'] = bb.bollinger_hband().fillna(self.close)
        self.df['BB_Low'] = bb.bollinger_lband().fillna(self.close)
        self.df['ATR'] = AverageTrueRange(self.high, self.low, self.close).average_true_range().fillna(0)

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

    def calculate_win_rate(self):
        try:
            window = self.df.tail(250)
            ma20 = window['EMA20']
            price = window['Close']
            buy_signals = (price > ma20) & (price.shift(1) < ma20.shift(1))
            wins = 0
            count = 0
            for date in buy_signals[buy_signals].index:
                try:
                    entry = window.loc[date]['Close']
                    idx = window.index.get_loc(date) + 5
                    if idx < len(window):
                        exit = window.iloc[idx]['Close']
                        if exit > entry: wins += 1
                        count += 1
                except: pass
            return (wins / count * 100) if count > 0 else 50.0
        except: return 50.0

    def get_scores(self):
        t_score = 0
        r_score = 0
        try:
            curr = self.close.iloc[-1]
            ema20 = self.df['EMA20'].iloc[-1]
            ema60 = self.df['EMA60'].iloc[-1]
            mfi = self.df['MFI'].iloc[-1]
            rsi = self.df['RSI'].iloc[-1]
            
            if curr > ema20 > ema60: t_score += 30
            elif curr > ema60: t_score += 15
            
            if self.df['MACD'].iloc[-1] > self.df['Signal'].iloc[-1]: t_score += 15
            if 50 <= rsi <= 75: t_score += 15
            if mfi > 60: t_score += 20
            
            if rsi < 30: r_score += 40
            elif rsi < 40: r_score += 20
            if curr <= self.df['BB_Low'].iloc[-1]: r_score += 30
            if mfi < 20: r_score += 10 
        except: pass
        return t_score, r_score

# ==========================================
# 📝 策略層
# ==========================================
def generate_strategy(ticker, df, news_score):
    az = QuantAnalyzer(ticker, df)
    curr = az.close.iloc[-1]
    
    t_score, r_score = az.get_scores()
    mfi_val = az.df['MFI'].iloc[-1]
    rsi_val = az.df['RSI'].iloc[-1]
    
    pot = az.calculate_potential()
    win_rate = az.calculate_win_rate()
    
    total_score = t_score + (news_score * 3)
    
    signal = "⚪ 觀望"
    buy_price = az.df['BB_Low'].iloc[-1] 
    ma5 = az.close.rolling(5).mean().iloc[-1]
    
    if total_score >= 80:
        signal = "🔥 強力買進"
        buy_price = curr
    elif total_score >= 60:
        signal = "🔴 偏多操作"
        buy_price = ma5 if curr > ma5 else curr
    elif r_score >= 40:
        signal = "💎 甜蜜抄底"
        buy_price = az.df['BB_Low'].iloc[-1]
    
    open_p = az.df['Open'].iloc[-1]
    is_red = curr > open_p
    if "多" in signal or "強力" in signal:
        if is_red and curr >= buy_price:
            signal = "✅ 確認轉強"
            buy_price = curr

    # 短衝邏輯
    p5, p10, p20 = "-", "-", "-"
    try:
        x = np.arange(len(df.tail(20)))
        y = df['Close'].tail(20).values
        s, _ = np.polyfit(x, y, 1)
        if s > -10: 
            p5_v = curr + s*5
            p10_v = curr + s*10
            p5 = f"{p5_v:.1f}"
            p10 = f"{p10_v:.1f}"
            p20 = f"{curr + s*20:.1f}"
            
            if p10_v > p5_v > curr and mfi_val > 60 and win_rate > 60:
                 if "觀望" not in signal and "有雷" not in signal:
                     signal += " 🚀短衝"
    except: pass
    
    if news_score <= -3:
        signal = "⚠️ 風險警示"
        buy_price = 0
        p5 = p10 = p20 = "-"
    
    atr = az.df['ATR'].iloc[-1]
    stop = curr - (2.5 * atr) if buy_price > 0 else 0
    
    sell_note = ""
    if stop > 0 and curr < stop: sell_note = "🛑 破線快逃"
    elif rsi_val > 75: sell_note = "⚠️ 過熱減碼"

    return {
        "info": {
            "id": az.display_name,
            "ticker_code": ticker,
            "price": curr,
            "potential": pot, # 1. 這裡看長線50%
            "win_rate": win_rate, # 2. 這裡看歷史勝率
            "signal": signal,
            "buy": buy_price,
            "stop": stop,
            "rsi": rsi_val,
            "score": max(total_score, r_score),
            "sell_note": sell_note,
            "5日": p5, "10日": p10, "20日": p20 # 3. 這裡看短線爆發力
        },
        "analyzer": az
    }

# ==========================================
# 🎨 視覺層
# ==========================================
def draw_chart(analyzer):
    df = analyzer.df.tail(80)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])

    fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 255, 255, 0.05)', name='布林通道'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], line=dict(color='#00FFFF', width=1.5, dash='dot'), name='地板'), row=1, col=1)
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='K線'), row=1, col=1)
    
    if 'EMA20' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='#FFD700', width=1), name='月線'), row=1, col=1)
    if 'EMA60' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['EMA60'], line=dict(color='#00BFFF', width=1), name='季線'), row=1, col=1)
    
    colors = ['#ef5350' if o - c >= 0 else '#26a69a' for o, c in zip(df['Open'], df['Close'])]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='成交量'), row=2, col=1)
    
    fig.update_xaxes(tickformat="%Y/%m")
    fig.update_layout(title=f"<b>{analyzer.display_name}</b> 技術分析", yaxis_title='價格', xaxis_rangeslider_visible=False, height=600, template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.02, x=0, xanchor="left"))
    return fig

# ==========================================
# 🚀 主程式
# ==========================================
def main():
    with st.sidebar:
        st.header("💎 HedgeFund OS")
        selected_sector = st.radio("選擇板塊", list(SECTORS.keys()))

    st.title(f"🚀 {selected_sector} - 戰情室")

    with st.spinner(f'正在下載 {selected_sector} 數據...'):
        tickers = SECTORS[selected_sector]
        raw_data = DataService.get_batch_data(tickers)
        
        if raw_data is None:
            st.error("數據連線失敗")
            return

        strategies = []
        progress_bar = st.progress(0)
        
        for i, ticker in enumerate(tickers):
            try:
                if isinstance(raw_data.columns, pd.MultiIndex): df_stock = raw_data[ticker].copy()
                else: df_stock = raw_data.copy()
                
                analyzer = QuantAnalyzer(ticker, df_stock)
                tech_score, _ = analyzer.get_scores()
                
                news_score = 0
                if tech_score >= 40:
                    news_score, _ = DataService.get_news_sentiment(ticker)
                
                result = generate_strategy(ticker, df_stock, news_score)
                strategies.append(result)
            except: pass
            progress_bar.progress((i + 1) / len(tickers))
        
        progress_bar.empty()

        if strategies:
            df_display = pd.DataFrame([s['info'] for s in strategies])
            df_display = df_display.sort_values(by='score', ascending=False)
            
            col_left, col_right = st.columns([2, 1])
            
            with col_left:
                st.subheader("📋 交易決策總表")
                
                def style_upside(v):
                    if v > 50: return 'color: #00FF00; font-weight: bold'
                    if v < 0: return 'color: #FF5252'
                    return 'color: white'

                st.dataframe(
                    df_display.drop(columns=['ticker_code', 'score', 'sell_note', 'mfi']),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "id": st.column_config.TextColumn("名稱", width="small"),
                        "price": st.column_config.NumberColumn("現價", format="%.1f", width="small"),
                        "potential": st.column_config.NumberColumn("🔥 年化潛力", format="%+.1f%%"),
                        "win_rate": st.column_config.ProgressColumn("🎲 勝率%", format="%.0f%%", min_value=0, max_value=100),
                        "signal": st.column_config.TextColumn("AI 判斷", width="medium"),
                        "buy": st.column_config.NumberColumn("🎯 買點", format="%.1f"),
                        "stop": st.column_config.NumberColumn("🛑 停損", format="%.1f"),
                        "rsi": st.column_config.NumberColumn("RSI", format="%.1f"),
                        "5日": st.column_config.TextColumn("5日預測", width="small"),
                        "10日": st.column_config.TextColumn("10日預測", width="small"),
                    }
                )

            with col_right:
                st.subheader("🔍 戰術分析")
                selected_id = st.selectbox("選擇股票", df_display['id'], key='stock_selector')
                sel_strategy = next(s for s in strategies if s['info']['id'] == selected_id)
                info = sel_strategy['info']
                
                st.markdown(f"""
                <div class="info-card">
                    <h3>{info['id']}</h3>
                    <p><b>🔥 訊號：</b> {info['signal']}</p>
                    <p><b>🚀 年化潛力：</b> <span style="color:{'green' if info['potential']>0 else 'red'}">{info['potential']:.1f}%</span></p>
                    <p><b>🌊 RSI 指標：</b> {info['rsi']:.1f}</p>
                    <p><b>💰 5日預測：</b> {info['5日']}</p>
                    <p><b>💰 10日預測：</b> {info['10日']}</p>
                    <hr>
                    <p><b>🎯 建議買點：</b> <span class="highlight">{info['buy']:.1f}</span></p>
                    <p><b>🛑 停損防守：</b> {info['stop']:.1f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if info['sell_note']:
                    st.error(f"⚠️ 持有警告：{info['sell_note']}")

                with st.expander("📰 最新新聞", expanded=False):
                    _, headlines = DataService.get_news_sentiment(info['ticker_code'])
                    if headlines:
                        for h in headlines:
                            st.markdown(f"- [{h['title']}]({h['link']})")
                    else: st.write("暫無新聞")

            st.markdown("---")
            if selected_id:
                fig = draw_chart(sel_strategy['analyzer'])
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{info['ticker_code']}")

        else:
            st.error("無法取得數據，請檢查網路連線。")

if __name__ == "__main__":
    main()
