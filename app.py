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

# 技術指標庫
from ta.trend import MACD, EMAIndicator, SMAIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange

# ==========================================
# ⚙️ 系統配置 & 全局變數
# ==========================================
st.set_page_config(page_title="QuantHedge Pro | 法人級量化終端", layout="wide", page_icon="🏛️")

# CSS 優化
st.markdown("""
<style>
    .metric-card {background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 5px solid #4CAF50;}
    .bearish-card {background-color: #1e1e1e; padding: 15px; border-radius: 10px; border-left: 5px solid #FF5252;}
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 4px 4px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #4CAF50; color: white; }
</style>
""", unsafe_allow_html=True)

# 股票池 (擴充版)
SECTORS = {
    "🚀 電子權值": ["2330.TW", "2317.TW", "2454.TW", "2308.TW", "2303.TW", "3711.TW", "3008.TW", "3045.TW"],
    "🤖 AI 供應鏈": ["3231.TW", "2356.TW", "6669.TW", "2382.TW", "2376.TW", "3017.TW", "2421.TW", "3035.TW", "3443.TW"],
    "🚢 傳產金融": ["2603.TW", "2609.TW", "2615.TW", "2002.TW", "1605.TW", "1513.TW", "1519.TW", "2881.TW", "2882.TW", "2891.TW", "5880.TW"],
    "📺 面板雙虎": ["3481.TW", "2409.TW"],
    "📊 熱門 ETF": ["0050.TW", "0056.TW", "00878.TW", "00929.TW", "00919.TW", "00980A.TW", "00981A.TW", "00982A.TW"],
    "🇺🇸 美股七雄": ["NVDA", "TSLA", "AAPL", "MSFT", "GOOG", "AMZN", "META", "AMD", "INTC", "PLTR", "SMCI", "COIN"]
}
ALL_TICKERS = [t for s in SECTORS.values() for t in s]

# 中文對照表 (簡化版)
NAME_MAP = {
    "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "2308.TW": "台達電",
    "2603.TW": "長榮", "2421.TW": "建準", "3017.TW": "奇鋐", "6669.TW": "緯穎",
    "3231.TW": "緯創", "2382.TW": "廣達", "0050.TW": "台灣50", "NVDA": "輝達"
}

# ==========================================
# 🏗️ Class: StockAnalyzer (核心分析引擎)
# ==========================================
class StockAnalyzer:
    def __init__(self, ticker, df):
        self.ticker = ticker
        self.df = df
        self.close = df['Close']
        self.high = df['High']
        self.low = df['Low']
        self.volume = df['Volume']
        self.name = NAME_MAP.get(ticker, ticker.replace(".TW", ""))
        
    def add_technical_indicators(self):
        # 1. 趨勢指標 (Trend)
        self.df['EMA20'] = EMAIndicator(self.close, window=20).ema_indicator()
        self.df['EMA60'] = EMAIndicator(self.close, window=60).ema_indicator()
        self.df['SMA200'] = SMAIndicator(self.close, window=200).sma_indicator()
        
        # MACD
        macd = MACD(self.close)
        self.df['MACD'] = macd.macd()
        self.df['Signal'] = macd.macd_signal()
        
        # Ichimoku Cloud (一目均衡表 - 機構愛用)
        ichimoku = IchimokuIndicator(self.high, self.low)
        self.df['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
        self.df['Ichimoku_Conv'] = ichimoku.ichimoku_conversion_line()
        self.df['Ichimoku_SpanA'] = ichimoku.ichimoku_a()
        self.df['Ichimoku_SpanB'] = ichimoku.ichimoku_b()

        # 2. 動能指標 (Momentum)
        self.df['RSI'] = RSIIndicator(self.close).rsi()
        stoch = StochasticOscillator(self.high, self.low, self.close)
        self.df['KD_K'] = stoch.stoch()
        self.df['KD_D'] = stoch.stoch_signal()

        # 3. 波動指標 (Volatility)
        bb = BollingerBands(self.close, window=20, window_dev=2)
        self.df['BB_High'] = bb.bollinger_hband()
        self.df['BB_Low'] = bb.bollinger_lband()
        self.df['ATR'] = AverageTrueRange(self.high, self.low, self.close).average_true_range()

    def calculate_risk_metrics(self):
        # 計算年化波動率與夏普值
        returns = self.close.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) # 年化波動率
        
        # 假設無風險利率 2%
        sharpe_ratio = (returns.mean() * 252 - 0.02) / volatility if volatility > 0 else 0
        
        # 最大回撤 (Max Drawdown)
        cum_returns = (1 + returns).cumprod()
        peak = cum_returns.cummax()
        drawdown = (cum_returns - peak) / peak
        max_drawdown = drawdown.min()
        
        return volatility, sharpe_ratio, max_drawdown

    def get_support_resistance(self):
        # 簡單計算近期支撐壓力 (Pivot Points 概念)
        recent_high = self.high.tail(60).max()
        recent_low = self.low.tail(60).min()
        
        # 斐波那契回撤 (Fibonacci Retracement)
        diff = recent_high - recent_low
        fib_0382 = recent_high - 0.382 * diff
        fib_0618 = recent_high - 0.618 * diff # 黃金分割支撐
        
        return recent_high, recent_low, fib_0618

    def generate_signal(self):
        curr = self.close.iloc[-1]
        prev = self.close.iloc[-2]
        ema20 = self.df['EMA20'].iloc[-1]
        ema60 = self.df['EMA60'].iloc[-1]
        rsi = self.df['RSI'].iloc[-1]
        macd = self.df['MACD'].iloc[-1]
        signal_line = self.df['Signal'].iloc[-1]
        bb_low = self.df['BB_Low'].iloc[-1]
        
        score = 0
        reasons = []
        
        # 趨勢評分
        if curr > ema20 > ema60: 
            score += 30
            reasons.append("✅ 均線多頭排列")
        elif curr < ema20 < ema60:
            score -= 30
            reasons.append("❌ 均線空頭排列")
            
        # 動能評分
        if macd > signal_line:
            score += 10
            if macd > 0: score += 5
        
        # RSI 濾網
        if 50 <= rsi <= 75: 
            score += 10
        elif rsi > 80:
            score -= 20
            reasons.append("⚠️ RSI 過熱警戒")
        elif rsi < 30:
            score += 20
            reasons.append("💎 RSI 超賣 (潛在反彈)")
            
        # 布林通道抄底
        if curr <= bb_low:
            score += 20
            reasons.append("📉 觸碰布林下軌 (超跌)")

        # 最終建議
        action = "👀 觀望"
        if score >= 60: action = "🔥 強力買進"
        elif score >= 40: action = "🔴 偏多操作"
        elif score <= -20: action = "🟢 建議放空/賣出"
        
        return score, action, reasons

# ==========================================
# 📊 視覺化模組 (Plotly Charts)
# ==========================================
def plot_advanced_chart(analyzer):
    df = analyzer.df.tail(120) # 只畫最近半年
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.7, 0.3])

    # 1. K線圖 + 均線 + 布林
    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                name='K線'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='orange', width=1), name='月線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA60'], line=dict(color='blue', width=1), name='季線'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], line=dict(color='gray', width=0.5, dash='dot'), name='布林上'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], line=dict(color='gray', width=0.5, dash='dot'), name='布林下'), row=1, col=1)

    # 2. 成交量 + MACD
    colors = ['red' if row['Open'] - row['Close'] >= 0 else 'green' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='成交量', opacity=0.3), row=2, col=1)
    
    # 布局設定
    fig.update_layout(
        title=f"{analyzer.name} ({analyzer.ticker}) 技術分析圖",
        yaxis_title='股價',
        xaxis_rangeslider_visible=False,
        height=600,
        template="plotly_dark",
        margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig

# ==========================================
# 🌐 數據獲取與新聞
# ==========================================
@st.cache_data(ttl=300)
def get_data(tickers):
    try: return yf.download(" ".join(tickers), period="2y", group_by='ticker', progress=False)
    except: return None

def get_news(ticker):
    name = NAME_MAP.get(ticker, ticker.replace(".TW",""))
    encoded = urllib.parse.quote(name)
    rss = f"https://news.google.com/rss/search?q={encoded}+when:2d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    try:
        feed = feedparser.parse(rss)
        if not feed.entries: return []
        return [{"title": e.title, "link": e.link} for e in feed.entries[:3]]
    except: return []

# ==========================================
# 🖥️ 主程式邏輯 (Main Loop)
# ==========================================
def main():
    st.title("🏛️ QuantHedge Pro | 法人級量化終端")
    
    # 側邊欄：控制台
    st.sidebar.header("🔧 控制台")
    sector_select = st.sidebar.selectbox("選擇板塊", list(SECTORS.keys()))
    selected_tickers = SECTORS[sector_select]
    
    if st.sidebar.button("🚀 啟動量化運算", type="primary"):
        with st.spinner('正在連線彭博級數據源...'):
            raw_data = get_data(selected_tickers)
            
            if raw_data is None:
                st.error("數據獲取失敗，請稍後再試。")
                return

            # 分析結果容器
            analysis_results = []
            
            progress = st.progress(0)
            for i, ticker in enumerate(selected_tickers):
                try:
                    # 處理數據
                    if isinstance(raw_data.columns, pd.MultiIndex):
                        df = raw_data[ticker].copy()
                    else:
                        df = raw_data.copy() # 單支股票情況
                        
                    df = df.dropna(how='all')
                    if len(df) < 100: continue
                    
                    # 初始化分析器
                    analyzer = StockAnalyzer(ticker, df)
                    analyzer.add_technical_indicators()
                    
                    # 計算指標
                    vol, sharpe, mdd = analyzer.calculate_risk_metrics()
                    high_p, low_p, fib = analyzer.get_support_resistance()
                    score, signal, reasons = analyzer.generate_signal()
                    
                    # 凱利公式建議倉位 (基於夏普值簡化)
                    # Sharpe > 1 建議 20%, Sharpe < 0 建議 0
                    kelly_pos = min(max(sharpe * 0.2, 0), 0.5) 

                    analysis_results.append({
                        "analyzer": analyzer, # 儲存物件以便畫圖
                        "代號": ticker,
                        "名稱": analyzer.name,
                        "現價": round(df['Close'].iloc[-1], 1),
                        "信號": signal,
                        "分數": score,
                        "波動率": f"{vol*100:.1f}%",
                        "夏普值": f"{sharpe:.2f}",
                        "MDD": f"{mdd*100:.1f}%",
                        "支撐(Fib)": round(fib, 1),
                        "建議倉位": f"{kelly_pos*100:.0f}%",
                        "_sort": score
                    })
                except Exception as e:
                    continue
                progress.progress((i + 1) / len(selected_tickers))
            
            progress.empty()
            
            # --- 顯示層 ---
            if analysis_results:
                df_res = pd.DataFrame(analysis_results)
                df_res = df_res.sort_values(by='_sort', ascending=False)
                
                # 1. 戰情總表 (Dashboard)
                st.subheader(f"📊 {sector_select} - 戰情總表")
                
                def style_signal(v):
                    if "強力" in v: return 'background-color: #2e7d32; color: white; font-weight: bold'
                    if "偏多" in v: return 'color: #2ecc71; font-weight: bold'
                    if "放空" in v: return 'color: #ff5252; font-weight: bold'
                    return 'color: gray'

                st.dataframe(
                    df_res.drop(columns=['analyzer', '_sort']),
                    use_container_width=True,
                    column_config={
                        "信號": st.column_config.TextColumn(width="medium"),
                        "分數": st.column_config.ProgressColumn(format="%d", min_value=-50, max_value=100),
                        "夏普值": st.column_config.NumberColumn(help="Sharpe Ratio: 越高代表風險調整後報酬越好 (>1 為佳)"),
                        "MDD": st.column_config.TextColumn(help="最大回撤: 歷史最慘跌幅"),
                        "建議倉位": st.column_config.ProgressColumn(format="%s", min_value=0, max_value=100)
                    }
                )
                
                st.markdown("---")
                
                # 2. 深度分析 (點擊查看詳情)
                st.subheader("🔍 個股深度診斷 (含 K線圖 & 新聞)")
                
                selected_stock = st.selectbox("請選擇要查看的股票", df_res['代號'] + " " + df_res['名稱'])
                target_code = selected_stock.split(" ")[0]
                
                # 找出對應的 analyzer 物件
                target_row = next(item for item in analysis_results if item["代號"] == target_code)
                analyzer = target_row['analyzer']
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # 繪製互動式圖表
                    fig = plot_advanced_chart(analyzer)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # 右側數據面板
                    st.markdown(f"### 📝 {analyzer.name} 診斷報告")
                    
                    curr_price = target_row['現價']
                    fib = target_row['支撐(Fib)']
                    dist_to_support = (curr_price - fib) / curr_price * 100
                    
                    # 風險指標卡片
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4>🎯 交易策略</h4>
                        <p><b>AI 判斷：</b> {target_row['信號']}</p>
                        <p><b>技術分數：</b> {target_row['分數']} 分</p>
                        <p><b>黃金支撐 (0.618)：</b> {fib}</p>
                        <p><b>離支撐距離：</b> {dist_to_support:.1f}%</p>
                    </div>
                    <br>
                    <div class="metric-card" style="border-left: 5px solid #2196F3;">
                        <h4>🛡️ 風險控管 (Risk)</h4>
                        <p><b>年化波動率：</b> {target_row['波動率']}</p>
                        <p><b>夏普比率：</b> {target_row['夏普值']} (越高越好)</p>
                        <p><b>最大回撤 (MDD)：</b> {target_row['MDD']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 即時新聞
                    st.markdown("#### 📰 最新情報")
                    news_list = get_news(target_code)
                    if news_list:
                        for n in news_list:
                            st.markdown(f"- [{n['title']}]({n['link']})")
                    else:
                        st.info("暫無相關重大新聞")

if __name__ == "__main__":
    main()
