import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import json, os

st.set_page_config(
    page_title="Trader Terminal v5.6",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# ====================== BASIC SECURITY (Password Protection) ======================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Trader Terminal v5.6")
    st.markdown("### Enter password to access")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if password == "yourstrongpassword123":   # ← CHANGE THIS TO YOUR OWN STRONG PASSWORD
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    st.stop()

# ====================== MAIN APP ======================
st.title("📈 Trader Terminal v5.6 • Clean Analysis")
st.markdown("**Any ticker • Unbiased 3-Bot System • No trading execution**")

# Sidebar - Watchlist (optional)
st.sidebar.header("Watchlist (Optional)")
WATCHLIST_FILE = "watchlist.json"

def load_watchlist():
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, "r") as f: return json.load(f)
    return []

if "watchlist" not in st.session_state:
    st.session_state.watchlist = load_watchlist()

new_ticker = st.sidebar.text_input("Add Ticker", "")
if st.sidebar.button("➕ Add"):
    t = new_ticker.upper().strip()
    if t and t not in [w["ticker"] for w in st.session_state.watchlist]:
        st.session_state.watchlist.append({"ticker": t, "added": datetime.now().strftime("%Y-%m-%d %H:%M")})
        with open(WATCHLIST_FILE, "w") as f: json.dump(st.session_state.watchlist, f)
        st.sidebar.success(f"{t} added!")

for item in st.session_state.watchlist[:]:
    c1, c2 = st.sidebar.columns([4,1])
    c1.write(item["ticker"])
    if c2.button("🗑", key=item["ticker"]):
        st.session_state.watchlist = [w for w in st.session_state.watchlist if w["ticker"] != item["ticker"]]
        with open(WATCHLIST_FILE, "w") as f: json.dump(st.session_state.watchlist, f)
        st.rerun()

# Main input - any ticker
ticker_input = st.text_input("Enter Ticker (stock or commodity)", value="PLTR").upper().strip()
if not ticker_input:
    st.error("Please enter a ticker (e.g. PLTR, GC=F for Gold, CL=F for Oil)")
    st.stop()

st.caption(f"**Analyzing: {ticker_input}**")

# Helper
@st.cache_data(ttl=15)
def get_data(ticker, interval="15m", period="5d"):
    try:
        return yf.download(ticker, interval=interval, period=period, progress=False)
    except:
        return pd.DataFrame()

def multi_timeframe_confirmation(ticker, direction):
    timeframes = ["3m", "5m", "15m", "30m", "1h", "4h", "1d"]
    align_count = 0
    for tf in timeframes:
        df = get_data(ticker, interval=tf, period="10d" if tf in ["4h","1d"] else "2d")
        if df.empty or len(df) < 20: continue
        close = df["Close"].iloc[-1]
        sma20 = ta.sma(df["Close"], 20).iloc[-1]
        rsi = ta.rsi(df["Close"], 14).iloc[-1]
        if (direction == "BULLISH" and close > sma20 and rsi > 50) or \
           (direction == "BEARISH" and close < sma20 and rsi < 50):
            align_count += 1
    return align_count >= 5

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Charts", "🤖 3-Bot Analysis"])

with tab1:
    st.subheader("Live Watchlist")
    if not st.session_state.watchlist:
        st.info("Add tickers in the sidebar to see them here.")
    else:
        cols = st.columns(4)
        for i, item in enumerate(st.session_state.watchlist):
            ticker = item["ticker"]
            df = get_data(ticker)
            if df.empty: continue
            price = df["Close"].iloc[-1]
            chg = (price - df["Close"].iloc[-2]) / df["Close"].iloc[-2] * 100
            rsi = ta.rsi(df["Close"], 14).iloc[-1]
            with cols[i % 4]:
                st.metric(ticker, f"${price:.2f}", f"{chg:+.2f}%")
                st.caption(f"RSI {rsi:.1f}")

with tab2:
    st.subheader(f"Chart — {ticker_input}")
    tf = st.selectbox("Timeframe", ["5m","15m","30m","1h","1d"], index=1)
    df = get_data(ticker_input, interval=tf, period="60d")
    if not df.empty:
        df["SMA20"] = ta.sma(df["Close"], 20)
        df["SMA50"] = ta.sma(df["Close"], 50)
        df["RSI"] = ta.rsi(df["Close"], 14)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.65, 0.2, 0.15])
        fig.add_trace(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df.SMA20, name="SMA20", line=dict(color="#00d4ff")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df.SMA50, name="SMA50", line=dict(color="#ffd700")), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df.Volume, name="Volume"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df.RSI, name="RSI", line=dict(color="#ff3355")), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        fig.update_layout(height=820, template="plotly_dark", title=f"{ticker_input} — {tf}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"Could not load data for {ticker_input}. Try symbols like GC=F for Gold.")

with tab3:
    st.subheader("🤖 Unbiased 3-Bot System")
    st.caption("Momentum • Reversal • Divergence → Clear BULLISH / BEARISH / NEUTRAL")

    df = get_data(ticker_input)
    if not df.empty and len(df) >= 50:
        closes = df["Close"]
        volumes = df["Volume"]
        price = closes.iloc[-1]
        rsi = ta.rsi(closes, 14).iloc[-1]
        sma20 = ta.sma(closes, 20).iloc[-1]

        volume_spike = volumes.iloc[-1] > volumes.rolling(10).mean().iloc[-1] * 2.2

        # Bot 1: Momentum
        mom = "BULLISH" if volume_spike and price > sma20 and rsi > 55 else \
              "BEARISH" if volume_spike and price < sma20 and rsi < 45 else "NONE"

        # Bot 2: Reversal
        rev = "BULLISH" if rsi < 32 else "BEARISH" if rsi > 68 else "NONE"

        # Bot 3: Divergence
        prev_rsi = ta.rsi(closes.iloc[:-5], 14).iloc[-1] if len(closes) > 5 else rsi
        div = "BEARISH" if price > closes.iloc[-5] and rsi < prev_rsi * 0.95 else \
              "BULLISH" if price < closes.iloc[-5] and rsi > prev_rsi * 1.05 else "NONE"

        # Overall signal
        if mom != "NONE":
            overall = mom
        elif rev != "NONE":
            overall = rev
        elif div != "NONE":
            overall = div
        else:
            overall = "NEUTRAL"

        color = "🟢" if overall == "BULLISH" else "🔴" if overall == "BEARISH" else "⚪"
        st.markdown(f"**{ticker_input} Overall Signal: {color} {overall}**")
        st.write(f"Price: ${price:.2f} | RSI: {rsi:.1f}")

        if overall != "NEUTRAL" and multi_timeframe_confirmation(ticker_input, overall):
            st.success(f"✅ Strong confirmation across timeframes for **{overall}**")
        elif overall != "NEUTRAL":
            st.info("Signal present but multi-timeframe confirmation is weak.")

        # Show individual bots
        st.write("**Bot Breakdown:**")
        st.write(f"• Momentum Bot: {mom}")
        st.write(f"• Reversal Bot: {rev}")
        st.write(f"• Divergence Bot: {div}")

    else:
        st.error("Not enough data for analysis.")

st.caption("v5.6 Clean Analysis • 3-Bot System • Password protected • Change password in code • Analysis only")