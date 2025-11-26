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
from ta.trend import MACD, EMAIndicator, SMAIndicator, IchimokuIndicator, ADXIndicator, CCIIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice, ForceIndexIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel

# ==========================================
# ⚙️ 系統全域配置
# ==========================================
st.set_page_config(page_title="HedgeFund OS | 全自動軍用版", layout="wide", page_icon="🦅")

# 強制 CSS 修復圖表刷新與介面美化
st.markdown("""
<style>
    .stPlotlyChart { width: 100%; }
    div[data-testid="stMetric"] { background-color: #262730; padding: 15px; border-radius: 5px; border: 1px solid #444; }
    .css-1d391kg { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 📋 擴充級股票清單 (100+ 檔)
# ==========================================
SECTORS = {
    "🚀 台積電聯盟": ["2330.TW", "2454.TW", "3711.TW", "3035.TW", "3443.TW", "3661.TW", "3529.TW", "3131.TW", "6531.TW"],
    "🤖 AI 伺服器": ["2317.TW", "2382.TW", "3231.TW", "6669.TW", "2356.TW", "2376.TW", "2421.TW", "3017.TW", "3324.TW"],
    "⚡ 重電與綠能": ["1513.TW", "1519.TW", "1503.TW", "1504.TW", "1609.TW", "1605.TW", "6806.TW", "9958.TW"],
    "🚢 航運與傳產": ["2603.TW", "2609.TW", "2615.TW", "2618.TW", "2610.TW", "2002.TW", "1101.TW", "1301.TW", "1303.TW"],
    "🏦 金融護城河": ["2881.TW", "2882.TW", "2891.TW", "2886.TW", "2884.TW", "5880.TW", "2892.TW", "2880.TW", "2885.TW"],
    "📱 蘋概與光學": ["3008.TW", "2313.TW", "4938.TW", "4958.TW", "6269.TW", "3406.TW", "2474.TW"],
    "📺 面板與驅動": ["3481.TW", "2409.TW", "3034.TW", "4961.TW", "3545.TW"],
    "📊 熱門 ETF": ["0050.TW", "0056.TW", "00878.TW", "00929.TW", "00919.TW", "00940.TW", "00939.TW", "006208.TW", "00980A.TW", "00981A.TW", "00982A.TW"],
    "🇺🇸 美股七雄+": ["NVDA", "TSLA", "AAPL", "MSFT", "GOOG", "AMZN", "META", "AMD", "INTC", "PLTR", "SMCI", "COIN", "ARM", "AVGO", "QCOM"]
}

# 自動生成扁平清單
ALL_TICKERS = [t for s in SECTORS.values() for t in s]

# 中文對照表 (核心股)
NAME_MAP = {
    "2330.TW": "台積電", "2454.TW": "聯發科", "3711.TW": "日月光", "3661.TW": "世芯-KY", "3443.TW": "創意",
    "2317.TW": "鴻海", "2382.TW": "廣達", "3231.TW": "緯創", "6669.TW": "緯穎", "2356.TW": "英業達",
    "2376.TW": "技嘉", "3017.TW": "奇鋐", "2421.TW": "建準", "3324.TW": "雙鴻",
    "1513.TW": "中興電", "1519.TW": "華城", "1503.TW": "士電", "1609.TW": "大亞",
    "2603.TW": "長榮", "2609.TW": "陽明", "2615.TW": "萬海", "2618.TW": "長榮航",
    "2881.TW": "富邦金", "2882.TW": "國泰金", "2891.TW": "中信金", "2886.TW": "兆豐金", "5880.TW": "合庫金",
    "3008.TW": "大立光", "3406.TW": "玉晶光", "3034.TW": "聯詠", "3481.TW": "群創", "2409.TW": "友達",
    "0050.TW": "台灣50", "0056.TW": "高股息", "00878.TW": "國泰永續", "00929.TW": "復華科技",
    "00980A.TW": "野村趨勢", "00981A.TW": "統一動力", "00982A.TW": "群益強棒",
    "NVDA": "輝達", "TSLA": "特斯拉", "AAPL": "蘋果", "MSFT": "微軟", "GOOG": "谷歌"
}

# ==========================================
# 🧱 數據層 (Data Layer)
# ==========================================
class DataService:
    @staticmethod
    @st.cache_data(ttl=600) # 延長快取時間
    def get_batch_data(tickers):
        try:
            # 抓 2 年數據以計算長期均線與回測
            data = yf.download(" ".join(tickers), period="2y", group_by='ticker', progress=False)
            return data
        except Exception as e:
            st.error(f"數據下載失敗: {e}")
            return None

    @staticmethod
    def get_news(ticker):
        name = NAME_MAP.get(ticker, ticker.replace(".TW", ""))
        encoded = urllib.parse.quote(name)
        rss = f"https://news.google.com/rss/search?q={encoded}+when:3d&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        try:
            feed = feedparser.parse(rss)
            if not feed.entries: return 0, []
            
            # 擴充關鍵字庫
            pos_keys = ["營收", "獲利", "新高", "大單", "買超", "漲停", "強勢", "填息", "完銷", "反彈", "噴出", "看好", "目標價調升"]
            neg_keys = ["虧損", "衰退", "重挫", "跌停", "利空", "斬倉", "貼息", "下修", "破底", "不如預期", "裁員", "調查"]
            
            score = 0
            headlines = []
            for entry in feed.entries[:5]:
                t = entry.title
                headlines.append({"title": t, "link": entry.link, "published": entry.published})
                for w in pos_keys: score += 1
                for w in neg_keys: score -= 1
            return score, headlines
        except: return 0, []

# ==========================================
# 🧠 分析層 (Analytics Layer)
# ==========================================
class QuantAnalyzer:
    def __init__(self, ticker, df):
        self.ticker = ticker
        self.df = df.copy()
        self.close = self.df['Close']
        self.high = self.df['High']
        self.low = self.df['Low']
        self.volume = self.df['Volume']
        self.name = NAME_MAP.get(ticker, ticker)
        
        # 初始化計算
        self._add_indicators()
        
    def _add_indicators(self):
        # 填充缺失值避免報錯
        self.df.fillna(method='bfill', inplace=True)
        self.df.fillna(method='ffill', inplace=True)

        # 1. 趨勢 (Trend)
        self.df['EMA10'] = EMAIndicator(self.close, window=10).ema_indicator()
        self.df['EMA20'] = EMAIndicator(self.close, window=20).ema_indicator()
        self.df['EMA60'] = EMAIndicator(self.close, window=60).ema_indicator()
        self.df['SMA200'] = SMAIndicator(self.close, window=200).sma_indicator()
        
        # Ichimoku (一目均衡表)
        ichimoku = IchimokuIndicator(self.high, self.low)
        self.df['Cloud_A'] = ichimoku.ichimoku_a()
        self.df['Cloud_B'] = ichimoku.ichimoku_b()

        # 2. 動能 (Momentum)
        self.df['RSI'] = RSIIndicator(self.close).rsi()
        macd = MACD(self.close)
        self.df['MACD'] = macd.macd()
        self.df['Signal'] = macd.macd_signal()
        
        # 3. 波動 (Volatility)
        bb = BollingerBands(self.close, window=20, window_dev=2)
        self.df['BB_High'] = bb.bollinger_hband()
        self.df['BB_Low'] = bb.bollinger_lband()
        self.df['ATR'] = AverageTrueRange(self.high, self.low, self.close).average_true_range()
        
        # 4. 量能 (Volume)
        self.df['OBV'] = OnBalanceVolumeIndicator(self.close, self.volume).on_balance_volume()
        self.df['ForceIndex'] = ForceIndexIndicator(self.close, self.volume, window=13).force_index()

    def get_fundamental_score(self):
        # 模擬基本面評分 (因為免費 API 抓不到財報)
        # 我們用「技術面的長線強度」來模擬基本面好壞
        score = 0
        curr = self.close.iloc[-1]
        # 站上年線 +20分
        if curr > self.df['SMA200'].iloc[-1]: score += 20
        # OBV 創新高 (籌碼好) +20分
        if self.df['OBV'].iloc[-1] > self.df['OBV'].iloc[-20:].mean(): score += 20
        # 波動率穩定 +10分
        if self.df['ATR'].iloc[-1] / curr < 0.03: score += 10
        return score

    def get_signal_score(self):
        score = 0
        curr = self.close.iloc[-1]
        
        # 趨勢 (Trend)
        if curr > self.df['EMA20'].iloc[-1]: score += 20
        if self.df['EMA20'].iloc[-1] > self.df['EMA60'].iloc[-1]: score += 20
        
        # 動能 (Momentum)
        if self.df['MACD'].iloc[-1] > self.df['Signal'].iloc[-1]: score += 15
        rsi = self.df['RSI'].iloc[-1]
        if 50 <= rsi <= 75: score += 15
        elif rsi < 30: score += 30 # 超跌加分 (逆勢)
        
        # 通道 (Volatility)
        if curr <= self.df['BB_Low'].iloc[-1]: score += 20 # 觸底反彈機會
        
        return score, rsi

    def calculate_kelly_position(self):
        # 修正後的凱利公式：更寬容，避免都顯示 0%
        try:
            # 抓最近 120 天計算勝率
            window = self.df.tail(120)
            daily_ret = window['Close'].pct_change().dropna()
            
            wins = daily_ret[daily_ret > 0]
            losses = daily_ret[daily_ret < 0]
            
            if len(wins) == 0: return 0
            
            win_rate = len(wins) / len(daily_ret)
            avg_win = wins.mean()
            avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.01
            
            odds = avg_win / avg_loss
            
            # 凱利公式 f = (bp - q) / b
            kelly = (odds * win_rate - (1 - win_rate)) / odds
            
            # 調整：不讓它變成負數，且給予最低持倉建議
            if kelly <= 0:
                # 如果勝率還行 (>45%)，給 10% 觀察倉
                return 0.1 if win_rate > 0.45 else 0
            else:
                # 安全凱利 (Half Kelly) 最多 50%
                return min(kelly * 0.5, 0.5)
        except: return 0

# ==========================================
# 📝 策略層 (Strategy Layer)
# ==========================================
def generate_strategy(ticker, df, news_score):
    analyzer = QuantAnalyzer(ticker, df)
    
    curr_price = analyzer.close.iloc[-1]
    tech_score, rsi = analyzer.get_signal_score()
    fund_score = analyzer.get_fundamental_score()
    
    total_score = tech_score + fund_score + (news_score * 3)
    
    # 交易訊號生成
    signal = "⚪ 觀望"
    buy_price = 0.0
    
    ma5 = analyzer.close.rolling(5).mean().iloc[-1]
    ma20 = analyzer.df['EMA20'].iloc[-1]
    bb_low = analyzer.df['BB_Low'].iloc[-1]
    
    # 策略分支
    if total_score >= 80:
        signal = "🔥 強力買進"
        buy_price = curr_price # 追價
    elif total_score >= 60:
        signal = "🔴 偏多操作"
        buy_price = ma5 if curr_price > ma5 else curr_price # 拉回買
    elif rsi < 40 and total_score >= 40:
        signal = "💎 甜蜜抄底"
        buy_price = bb_low # 掛布林下軌
    
    # 新聞濾網 (如果新聞極差，強制降級)
    if news_score <= -3:
        signal = "⚠️ 風險警示"
        buy_price = 0 # 不建議買
    
    # 停損停利
    atr = analyzer.df['ATR'].iloc[-1]
    stop_loss = curr_price - (2.5 * atr)
    target_1 = curr_price + (3 * atr)
    
    # 凱利建議
    kelly = analyzer.calculate_kelly_position()
    
    return {
        "info": {
            "id": ticker,
            "name": analyzer.name,
            "price": curr_price,
            "signal": signal,
            "buy": buy_price,
            "stop": stop_loss,
            "target": target_1,
            "kelly": kelly,
            "score": total_score,
            "rsi": rsi
        },
        "analyzer": analyzer # 傳遞物件給繪圖用
    }

# ==========================================
# 🎨 視覺層 (View Layer)
# ==========================================
def draw_advanced_chart(analyzer):
    df = analyzer.df.tail(150)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, row_heights=[0.7, 0.3])

    # 1. 主圖：K線 + 均線 + 布林
    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
                name='K線'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA20'], line=dict(color='#FFD700', width=1.5), name='月線 (20MA)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA60'], line=dict(color='#00BFFF', width=1.5), name='季線 (60MA)'), row=1, col=1)
    
    # 布林通道 (淺色背景)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], line=dict(width=0), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], line=dict(width=0), fill='tonexty', 
                             fillcolor='rgba(255, 255, 255, 0.05)', showlegend=False), row=1, col=1)

    # 2. 副圖：成交量 + MACD 柱狀
    colors = ['#ef5350' if row['Open'] - row['Close'] >= 0 else '#26a69a' for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='成交量'), row=2, col=1)
    
    # 3. 標註買賣點
    curr = df['Close'].iloc[-1]
    
    fig.update_layout(
        title=f"<b>{analyzer.name} ({analyzer.ticker})</b> 專業技術分析",
        yaxis_title='價格',
        xaxis_rangeslider_visible=False,
        height=600,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1, x=0, xanchor="left", yanchor="bottom")
    )
    return fig

# ==========================================
# 🚀 應用程式主入口
# ==========================================
def main():
    # 側邊欄
    with st.sidebar:
        st.header("🎛️ HedgeFund OS")
        selected_sector = st.radio("選擇板塊", list(SECTORS.keys()))
        st.info("💡 **系統模式：** 全自動即時運算\n(切換板塊即刻更新)")

    st.title(f"🏛️ {selected_sector} - 戰情室")

    # 1. 自動加載數據 (無須按鈕)
    with st.spinner(f'正在連線交易所，下載 {selected_sector} 數據...'):
        tickers = SECTORS[selected_sector]
        raw_data = DataService.get_batch_data(tickers)
        
        if raw_data is None:
            st.error("無法連線至數據源，請稍後再試。")
            return

        # 2. 計算分析
        strategies = []
        
        # 使用進度條
        progress_bar = st.progress(0)
        
        for i, ticker in enumerate(tickers):
            try:
                if isinstance(raw_data.columns, pd.MultiIndex): df_stock = raw_data[ticker].copy()
                else: df_stock = raw_data.copy()
                
                # 簡單篩選：只對潛力股抓新聞 (加速)
                analyzer = QuantAnalyzer(ticker, df_stock)
                tech_score, _ = analyzer.get_signal_score()
                
                news_score = 0
                if tech_score >= 40: # 分數不錯才去查新聞
                    news_score, _ = DataService.get_news_sentiment(ticker)
                
                result = generate_strategy(ticker, df_stock, news_score)
                strategies.append(result)
                
            except Exception as e: pass
            progress_bar.progress((i + 1) / len(tickers))
        
        progress_bar.empty()

        # 3. 顯示結果
        if strategies:
            # 轉換為 DataFrame 顯示列表
            df_display = pd.DataFrame([s['info'] for s in strategies])
            df_display = df_display.sort_values(by='score', ascending=False)
            
            # --- 上半部：決策表格 ---
            col_left, col_right = st.columns([2, 1])
            
            with col_left:
                st.subheader("📋 交易決策總表")
                
                def style_rows(row):
                    v = row['signal']
                    if "強力" in v: return ['background-color: #1b5e20']*len(row)
                    if "偏多" in v: return ['background-color: #004d40']*len(row)
                    if "甜蜜" in v: return ['background-color: #0d47a1']*len(row)
                    if "警示" in v: return ['color: #ff5252']*len(row)
                    return ['']*len(row)

                st.dataframe(
                    df_display.drop(columns=['id', 'score', 'rsi']),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "name": st.column_config.TextColumn("股票", width="small"),
                        "price": st.column_config.NumberColumn("現價", format="%.1f"),
                        "signal": st.column_config.TextColumn("AI 判斷", width="medium"),
                        "buy": st.column_config.NumberColumn("🎯 建議買點", format="%.1f"),
                        "stop": st.column_config.NumberColumn("🛑 停損", format="%.1f"),
                        "target": st.column_config.NumberColumn("💰 目標價", format="%.1f"),
                        "kelly": st.column_config.ProgressColumn("建議倉位", format="%.0f%%", min_value=0, max_value=1),
                    }
                )

            # --- 下半部：互動詳情 ---
            with col_right:
                st.subheader("🔍 戰術分析")
                # 使用 unique key 避免 Streamlit 重繪錯誤
                selected_ticker_str = st.selectbox("選擇股票查看詳情", df_display['name'] + " (" + df_display['id'] + ")", key='stock_selector')
                
                if selected_ticker_str:
                    sel_id = selected_ticker_str.split("(")[1].replace(")", "")
                    # 找出對應的完整策略物件
                    sel_strategy = next(s for s in strategies if s['info']['id'] == sel_id)
                    info = sel_strategy['info']
                    
                    # 顯示數據卡片
                    st.markdown(f"""
                    <div style="background-color: #262730; padding: 15px; border-radius: 10px; border: 1px solid #555;">
                        <h3 style="color: #FFD700; margin:0;">{info['name']} ({info['id']})</h3>
                        <hr style="margin: 10px 0;">
                        <p><b>🔥 訊號：</b> {info['signal']}</p>
                        <p><b>📊 綜合分數：</b> {info['score']:.0f} 分</p>
                        <p><b>📈 RSI 強度：</b> {info['rsi']:.1f}</p>
                        <p><b>🏦 建議倉位：</b> {info['kelly']*100:.0f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # 顯示新聞
                    with st.expander("📰 查看最新新聞", expanded=False):
                        _, headlines = DataService.get_news_sentiment(sel_id)
                        if headlines:
                            for h in headlines:
                                st.markdown(f"- [{h['title']}]({h['link']})")
                        else:
                            st.write("暫無重大新聞")

            # --- 底部：K線圖 ---
            st.markdown("---")
            if selected_ticker_str:
                # 這裡強制重新繪圖，解決白屏問題
                fig = draw_advanced_chart(sel_strategy['analyzer'])
                st.plotly_chart(fig, use_container_width=True, key=f"chart_{sel_id}")

        else:
            st.error("無法取得數據，請檢查網路連線。")

if __name__ == "__main__":
    main()
