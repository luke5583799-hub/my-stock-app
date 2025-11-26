import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(page_title="AI 價值投資大師 (李永樂版)", layout="wide", page_icon="🏛️")

# ==========================================
# 📋 優質長線觀察清單 (去除投機股)
# ==========================================
# 這裡只留基本面較好的權值股與ETF，適合長線估值
VALUE_STOCKS = [
    "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2303.TW", "2382.TW", "3711.TW", "3034.TW", 
    "3008.TW", "2412.TW", "2881.TW", "2882.TW", "2891.TW", "5880.TW", "1216.TW", "1101.TW",
    "0050.TW", "0056.TW", "00878.TW", "006208.TW", "00919.TW",
    "NVDA", "AAPL", "MSFT", "GOOG", "TSLA", "BRK-B"
]

# 股票中文對照
STOCK_MAP = {
    "2330.TW": "台積電", "2317.TW": "鴻海", "2454.TW": "聯發科", "2308.TW": "台達電",
    "2303.TW": "聯電", "2382.TW": "廣達", "3711.TW": "日月光", "3034.TW": "聯詠",
    "3008.TW": "大立光", "2412.TW": "中華電", "2881.TW": "富邦金", "2882.TW": "國泰金",
    "2891.TW": "中信金", "5880.TW": "合庫金", "1216.TW": "統一", "1101.TW": "台泥",
    "0050.TW": "台灣50", "0056.TW": "高股息", "00878.TW": "國泰永續", "006208.TW": "富邦台50",
    "00919.TW": "群益精選", "NVDA": "輝達", "AAPL": "蘋果", "MSFT": "微軟",
    "GOOG": "谷歌", "TSLA": "特斯拉", "BRK-B": "波克夏"
}

# ==========================================
# 🧮 核心數學模型 (李永樂老師影片理論)
# ==========================================

def calculate_value_investing_metrics(ticker):
    try:
        # 1. 獲取數據 (需要長一點的時間來計算勝率)
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        
        if len(hist) < 200: return None
        
        current_price = hist['Close'].iloc[-1]
        
        # --- A. 估值模型 (Video 1: 判斷價格是否合理) ---
        # 由於 DCF 需要複雜財報，我們使用「葛拉漢成長公式」或「本益比位階」做為簡化版估值
        # 估值 V = EPS * (8.5 + 2g)  (g為預期成長率)
        # 若無法取得 EPS (如 ETF)，則改用「統計學回歸」判斷便宜度
        
        fair_value = 0
        valuation_method = ""
        safety_margin = 0 # 安全邊際
        
        try:
            info = stock.info
            eps = info.get('trailingEps', 0)
            pe = info.get('trailingPE', 0)
            
            # 判斷是個股還是 ETF (ETF 通常沒 EPS)
            if eps and pe and not ticker.startswith("00"): 
                # 假設保守成長率 g = 5% (長線投資不假設暴漲)
                # 葛拉漢公式修正版: V = EPS * (8.5 + 2 * 成長率)
                growth_rate = 5 
                fair_value = eps * (8.5 + 2 * growth_rate)
                valuation_method = "葛拉漢估值法"
            else:
                # ETF 或無 EPS 個股：使用「年線乖離」作為價值錨點
                # 假設年線 (240MA) 是市場公認的價值中樞
                ma_240 = hist['Close'].rolling(240).mean().iloc[-1]
                fair_value = ma_240
                valuation_method = "年線價值法 (ETF)"
                
            # 計算安全邊際 (Margin of Safety)
            # 安全邊際 = (合理價 - 現價) / 合理價
            safety_margin = (fair_value - current_price) / fair_value
            
        except:
            # 萬一都失敗，用半年均線當基準
            ma_120 = hist['Close'].rolling(120).mean().iloc[-1]
            fair_value = ma_120
            valuation_method = "半年線基準"
            safety_margin = (fair_value - current_price) / fair_value

        # --- B. 凱利公式 (Video 2: 資金分配) ---
        # f = (bp - q) / b
        # p = 勝率 (Win Rate)
        # b = 賠率 (Odds) = 平均獲利 / 平均虧損
        
        # 計算過去一年的日漲跌
        daily_returns = hist['Close'].pct_change().dropna()
        
        # 勝率 p: 上漲天數 / 總天數
        winning_days = len(daily_returns[daily_returns > 0])
        total_days = len(daily_returns)
        p = winning_days / total_days
        q = 1 - p
        
        # 賠率 b: 平均漲幅 / 平均跌幅 (取絕對值)
        avg_win = daily_returns[daily_returns > 0].mean()
        avg_loss = abs(daily_returns[daily_returns < 0].mean())
        b = avg_win / avg_loss if avg_loss != 0 else 1
        
        # 凱利公式計算 (百分比)
        kelly_fraction = (b * p - q) / b
        
        # 李永樂老師提醒：凱利公式太激進，實務上建議「半凱利」甚至更低
        # 我們這裡設定更保守：如果估值太貴，倉位強制降低
        suggested_position = kelly_fraction * 0.5 # 半凱利
        
        # 如果算出來是負的，代表期望值為負，不該下注
        if suggested_position < 0: suggested_position = 0
        
        # 如果現價 > 合理價 (太貴)，強制減少倉位建議
        if safety_margin < 0: suggested_position *= 0.2 

        # --- C. 停損點 (Video 3: 避免賺小虧大) ---
        # 使用 ATR (真實波幅) 計算理性停損，而非情緒停損
        # 李永樂：跌 50% 要漲 100% 才能回本 -> 絕對不能讓虧損擴大
        high = hist['High']
        low = hist['Low']
        close = hist['Close']
        tr = np.maximum((high - low), np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        atr = tr.rolling(14).mean().iloc[-1]
        
        stop_loss_price = current_price - (2.5 * atr) # 2.5倍 ATR 為寬鬆停損，適合波段

        # --- 顯示名稱 ---
        clean_code = ticker.replace(".TW", "")
        name = STOCK_MAP.get(ticker, "")
        display_name = f"{clean_code} {name}"

        return {
            "代號": display_name,
            "現價": current_price,
            "合理估值": fair_value,
            "安全邊際": safety_margin * 100, # 轉百分比
            "估值法": valuation_method,
            "勝率": p * 100,
            "賠率": b,
            "建議倉位": suggested_position * 100, # 轉百分比
            "建議停損": stop_loss_price,
            "趨勢": "📈" if current_price > fair_value else "📉"
        }

    except Exception as e:
        return None

# ==========================================
# 🖥️ 介面
# ==========================================
st.title("🏛️ AI 價值投資大師 (李永樂數學版)")
st.markdown("""
此系統基於 **李永樂老師** 的四大投資理論設計：
1.  **貼現估值 (Value):** 算出股票的「真實價值」，只在便宜時買入。
2.  **凱利公式 (Kelly):** 根據勝率與賠率，科學計算「該買多少倉位」。
3.  **風險控制 (Stop Loss):** 避免「賺小虧大」，嚴格設定數學停損點。
""")

if st.button("🧮 啟動價值運算", type="primary"):
    with st.spinner('正在計算內在價值與凱利倉位...'):
        
        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_ticker = {executor.submit(calculate_value_investing_metrics, t): t for t in VALUE_STOCKS}
            for future in future_to_ticker:
                res = future.result()
                if res: results.append(res)
        
        df = pd.DataFrame(results)
        
        if not df.empty:
            # 依照「安全邊際」排序：越便宜的排越前面
            df = df.sort_values(by='安全邊際', ascending=False)
            
            # 樣式處理
            def style_margin(val):
                if val > 10: return 'background-color: #d4edda; color: #155724; font-weight: bold' # 綠色 (便宜)
                if val < -10: return 'background-color: #f8d7da; color: #721c24' # 紅色 (太貴)
                return ''
            
            def style_position(val):
                if val > 30: return 'color: #d63031; font-weight: bold' # 重倉
                if val == 0: return 'color: #b2bec3' # 空手
                return 'color: #0984e3' # 輕倉

            st.dataframe(
                df.style.applymap(style_margin, subset=['安全邊際'])
                      .applymap(style_position, subset=['建議倉位'])
                      .format({
                          "現價": "{:.1f}", 
                          "合理估值": "{:.1f}", 
                          "安全邊際": "{:.1f}%",
                          "勝率": "{:.1f}%", 
                          "賠率": "{:.2f}", 
                          "建議倉位": "{:.1f}%",
                          "建議停損": "{:.1f}"
                      }),
                use_container_width=True,
                column_config={
                    "代號": st.column_config.TextColumn(width="small"),
                    "合理估值": st.column_config.NumberColumn(help="根據葛拉漢公式或年線計算的理論價值"),
                    "安全邊際": st.column_config.NumberColumn(help="正數代表股價被低估(便宜)，負數代表高估(貴)"),
                    "建議倉位": st.column_config.NumberColumn(help="根據凱利公式計算，建議投入總資金的比例"),
                    "估值法": st.column_config.TextColumn(width="small")
                }
            )
            
            # 顯示分析結論
            top_pick = df.iloc[0]
            st.success(f"""
            ### 🏆 目前最具價值投資潛力：{top_pick['代號']}
            * **現價：** {top_pick['現價']:.1f} vs **合理價：** {top_pick['合理估值']:.1f}
            * **便宜程度：** {top_pick['安全邊際']:.1f}% (安全邊際)
            * **凱利建議：** 如果你有一筆資金，數學上建議投入 **{top_pick['建議倉位']:.1f}%** 的部位。
            """)
            
            st.warning("""
            **⚠️ 關於凱利公式的提醒 (李永樂老師)：**
            凱利公式計算的是「極限最佳解」，但現實中風險可能被低估。
            **建議實際下單時，將「建議倉位」再除以 2 (半凱利)，以策安全。**
            """)
            
        else:
            st.error("數據獲取失敗")
