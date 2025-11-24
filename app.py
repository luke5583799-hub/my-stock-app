import streamlit as st
import yfinance as yf
import pandas as pd
from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange

# ==========================================
# ⚙️ 頁面設定
# ==========================================
st.set_page_config(page_title="AI 股市操盤手", layout="wide", page_icon="📈")

# ==========================================
# 📋 監控清單 (50+ 檔)
# ==========================================
DEFAULT_STOCKS = [
    # 台股權值/電子
    "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2303.TW", "2382.TW", "3711.TW", "3034.TW", "3035.TW", 
    "3231.TW", "2356.TW", "6669.TW", "2376.TW", "3017.TW", "2421.TW", "2412.TW",
    # 傳產/航運/重電
    "2603.TW", "2609.TW", "2615.TW", "2618.TW", "2002.TW", "1605.TW", "1513.TW", "1519.TW",
    # 金融
    "2881.TW", "2882.TW", "2891.TW", "2886.TW", "5880.TW",
    # ETF
    "0050.TW", "0056.TW", "00878.TW", "00929.TW", "00919.TW",
    # 美股
    "NVDA", "TSLA", "AAPL", "MSFT", "GOOG", "AMZN", "META", "AMD", "INTC", "PLTR", "MSTR", "COIN", "SMCI"
]

# ==========================================
# 🛠️ 核心運算邏輯
# ==========================================
@st.cache_data(ttl=300) # 快取 5 分鐘
def fetch_all_data(tickers):
    tickers_str = " ".join(tickers)
    try:
        # 批次下載
        data = yf.download(tickers_str, period="6mo", group_by='ticker', progress=False)
        return data
    except Exception:
        return None

def calculate_metrics(ticker, df):
    try:
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker, axis=1, level=0)
        
        df = df.dropna(how='all')
        if len(df) < 50: return None

        close = df['Close']
        high = df['High']
        low = df['Low']

        # 指標計算
        ema_20 = EMAIndicator(close=close, window=20).ema_indicator()
        ema_60 = EMAIndicator(close=close, window=60).ema_indicator()
        macd = MACD(close=close)
        rsi = RSIIndicator(close=close).rsi()
        bb = BollingerBands(close=close, window=20, window_dev=2)
        atr = AverageTrueRange(high=high, low=low, close=close).average_true_range()
        
        curr_price = close.iloc[-1]
        curr_rsi = rsi.iloc[-1]
        curr_atr = atr.iloc[-1]

        # 分數計算
        trend_score = 0
        rebound_score = 0
        
        # 順勢邏輯
        if curr_price > ema_20.iloc[-1] > ema_60.iloc[-1]: trend_score += 40
        elif curr_price > ema_60.iloc[-1]: trend_score += 20
        if macd.macd().iloc[-1] > macd.macd_signal().iloc[-1]: trend_score += 20
        if 50 <= curr_rsi <= 75: trend_score += 20
        
        # 抄底邏輯
        bias = ((curr_price - ema_20.iloc[-1]) / ema_20.iloc[-1]) * 100
        if curr_rsi < 30: rebound_score += 40
        elif curr_rsi < 40: rebound_score += 15
        if curr_price <= bb.bollinger_lband().iloc[-1]: rebound_score += 30
        if bias < -7: rebound_score += 30

        # --- 決策核心 (Decision Engine) ---
        action = "👀 觀望"
        action_type = "neutral" # 用於排序
        
        if trend_score >= 80:
            action = "🔥 強力買進"
            action_type = "strong_buy"
        elif trend_score >= 60:
            action = "🔴 偏多買進"
            action_type = "buy"
        elif rebound_score >= 60:
            action = "💎 嘗試抄底"
            action_type = "rebound"

        # 停損停利計算 (ATR 通道法)
        stop_loss = curr_price - (2 * curr_atr)   # 2倍波動停損
        take_profit = curr_price + (3 * curr_atr) # 3倍波動停利 (盈虧比 1.5)

        return {
            "代號": ticker,
            "現價": round(curr_price, 2),
            "建議操作": action,
            "建議停損": round(stop_loss, 2),
            "建議停利": round(take_profit, 2),
            "順勢分": trend_score,
            "抄底分": rebound_score,
            "RSI": round(curr_rsi, 1),
            "_sort_key": trend_score + rebound_score # 內部排序用
        }
    except Exception:
        return None

# ==========================================
# 🖥️ 網頁顯示
# ==========================================
st.title("🛡️ AI 股市操盤手 (決策版)")
st.caption("自動判斷買賣訊號 + 計算停損停利點 (Risk/Reward 1:1.5)")

if st.button("🔄 立即分析市場", type="primary"):
    with st.spinner('AI 正在計算最佳交易機會...'):
        raw_data = fetch_all_data(DEFAULT_STOCKS)
        
        if raw_data is not None and not raw_data.empty:
            final_results = []
            
            # 進度條
            progress_bar = st.progress(0)
            for i, ticker in enumerate(DEFAULT_STOCKS):
                try:
                    res = calculate_metrics(ticker, raw_data[ticker])
                    if res: final_results.append(res)
                except: continue
                progress_bar.progress((i + 1) / len(DEFAULT_STOCKS))
            progress_bar.empty()

            df = pd.DataFrame(final_results)
            
            if not df.empty:
                # 依照分數總和排序
                df = df.sort_values(by='_sort_key', ascending=False).drop(columns=['_sort_key'])
                
                # 樣式：將「建議操作」上色
                def highlight_action(val):
                    if "強力" in val: return 'background-color: #ffcccc; color: #8b0000; font-weight: bold' # 深紅
                    if "偏多" in val: return 'background-color: #ffe6e6; color: #cc0000' # 淺紅
                    if "抄底" in val: return 'background-color: #e6fffa; color: #006666; font-weight: bold' # 藍綠
                    return 'color: #888888' # 灰色
                
                # 顯示表格
                st.subheader("📊 AI 交易建議總表")
                st.dataframe(
                    df.style.applymap(highlight_action, subset=['建議操作']),
                    use_container_width=True,
                    column_config={
                        "建議停損": st.column_config.NumberColumn(format="%.2f", help="跌破此價位建議出場"),
                        "建議停利": st.column_config.NumberColumn(format="%.2f", help="觸碰此價位建議獲利了結"),
                    }
                )
                
                # 簡單統計
                buy_count = len(df[df['建議操作'].str.contains("買進")])
                rebound_count = len(df[df['建議操作'].str.contains("抄底")])
                st.info(f"今日掃描結果：發現 {buy_count} 檔適合順勢買進，{rebound_count} 檔適合抄底。")

            else:
                st.error("目前無數據。")
        else:
            st.error("連線失敗，請稍後再試。")
