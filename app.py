import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(page_title="AI 價值回歸預測系統", layout="wide", page_icon="⏳")

# ==========================================
# 📋 長線優質股名單
# ==========================================
VALUE_STOCKS = [
    "2330.TW", "2317.TW", "2454.TW", "2308.TW", "2303.TW", "2382.TW", "3711.TW", "3034.TW", 
    "3008.TW", "2412.TW", "2881.TW", "2882.TW", "2891.TW", "5880.TW", "1216.TW", "1101.TW",
    "0050.TW", "0056.TW", "00878.TW", "006208.TW", "00919.TW",
    "NVDA", "AAPL", "MSFT", "GOOG", "TSLA", "BRK-B"
]

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
# 🧮 核心數學模型
# ==========================================
def calculate_value_projection(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y") # 抓一年數據算機率
        
        if len(hist) < 200: return None
        
        current_price = hist['Close'].iloc[-1]
        
        # --- 1. 計算合理估值 (目標價) ---
        info = stock.info
        eps = info.get('trailingEps', 0)
        
        # 葛拉漢公式修正版 (個股) 或 年線回歸 (ETF)
        if eps and not ticker.startswith("00") and "TW" in ticker: 
            # 台股個股用葛拉漢 (給予保守成長率 3~5%)
            fair_value = eps * (8.5 + 2 * 4) 
        elif ticker.startswith("00") or not "TW" in ticker:
            # ETF 或 美股(資料源問題) 用年線 (240MA) 作為價值中樞
            fair_value = hist['Close'].rolling(240).mean().iloc[-1]
        else:
            fair_value = hist['Close'].rolling(120).mean().iloc[-1]

        # 確保 fair_value 有數值
        if pd.isna(fair_value): fair_value = current_price 

        # 安全邊際 (便宜多少?)
        margin = (fair_value - current_price) / fair_value

        # --- 2. 計算回歸時間 (Time to Recovery) ---
        # 計算每日漲跌幅
        pct_change = hist['Close'].pct_change().dropna()
        
        # 上漲日的平均漲幅 (Avg Gain on Up Days)
        avg_up_move = pct_change[pct_change > 0].mean()
        
        # 勝率 (Win Rate)
        win_rate = len(pct_change[pct_change > 0]) / len(pct_change)
        
        # 預估天數公式： 距離 / (股價 * 平均漲幅 * 勝率)
        # 這是模擬「在正常波動下，平均每天能推進多少距離」
        distance = fair_value - current_price
        daily_velocity = current_price * avg_up_move * win_rate
        
        if distance > 0 and daily_velocity > 0:
            days_to_target = int(distance / daily_velocity)
        else:
            days_to_target = 0 # 已達標或高估

        # --- 3. 凱利公式倉位建議 ---
        avg_loss = abs(pct_change[pct_change < 0].mean())
        odds = avg_up_move / avg_loss if avg_loss > 0 else 1
        kelly = (odds * win_rate - (1 - win_rate)) / odds
        position = max(0, kelly * 0.5) # 半凱利

        # 若太貴，倉位歸零
        if margin < 0: position = 0

        clean_code = ticker.replace(".TW", "")
        name = STOCK_MAP.get(ticker, "")
        display_name = f"{clean_code} {name}"

        return {
            "代號": display_name,
            "現價": current_price,
            "目標價": fair_value,
            "潛在漲幅": margin * 100,
            "預估天數": days_to_target,
            "建議倉位": position * 100,
            "勝率": win_rate * 100,
            "_sort": margin # 用便宜程度排序
        }

    except: return None

# ==========================================
# 🖥️ 介面
# ==========================================
st.title("⏳ AI 價值回歸預測系統")
st.markdown("""
**李永樂數學模型應用：**
* **目標價 (Target):** 根據公司獲利能力或長期均線算出的「應有價值」。
* **預估天數 (Time):** 基於該股票的歷史波動慣性，推算漲回目標價需要的「平均交易日」。
""")

if st.button("🧮 計算回歸時間與獲利", type="primary"):
    with st.spinner('AI 正在進行蒙特卡羅模擬與估值運算...'):
        
        results = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_ticker = {executor.submit(calculate_value_projection, t): t for t in VALUE_STOCKS}
            for future in future_to_ticker:
                res = future.result()
                if res: results.append(res)
        
        df = pd.DataFrame(results)
        
        if not df.empty:
            # 邏輯：只顯示「被低估」的股票 (潛在漲幅 > 0)
            df = df[df['潛在漲幅'] > 0]
            df = df.sort_values(by='_sort', ascending=False)
            
            # 樣式
            def style_days(val):
                if val > 250: return 'color: #b2bec3' # 超過一年，太久了
                if val < 30: return 'color: #d63031; font-weight: bold' # 快要漲到了
                return 'color: #0984e3'

            st.dataframe(
                df.style.applymap(style_days, subset=['預估天數'])
                      .format({
                          "現價": "{:.1f}", 
                          "目標價": "{:.1f}", 
                          "潛在漲幅": "+{:.1f}%",
                          "建議倉位": "{:.0f}%",
                          "預估天數": "約 {:.0f} 天"
                      }),
                use_container_width=True,
                column_config={
                    "代號": st.column_config.TextColumn(width="small"),
                    "目標價": st.column_config.NumberColumn(help="合理估值 (Fair Value)"),
                    "潛在漲幅": st.column_config.TextColumn(help="目前距離目標價還有多少空間"),
                    "預估天數": st.column_config.TextColumn(help="基於歷史動能推算的持有時間"),
                    "建議倉位": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100, help="凱利公式建議資金比例")
                }
            )
            
            if len(df) > 0:
                best = df.iloc[0]
                st.success(f"""
                ### 🎯 最佳價值機會：{best['代號']}
                * **現在買入：** {best['現價']:.1f}
                * **等待漲到：** **{best['目標價']:.1f}** (還有 +{best['潛在漲幅']})
                * **預計持有：** **{best['預估天數']}** (交易日)
                * **建議倉位：** 總資金的 {best['建議倉位']:.0f}%
                """)
                st.info("💡 註：預估天數僅供參考，代表依照該股票的「平均爬升速度」，理論上需要多久才能填補價值缺口。")
            else:
                st.warning("目前所有觀察名單的股價都高於合理估值（太貴了），建議空手觀望。")
            
        else:
            st.error("數據獲取失敗")
