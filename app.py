import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time

st.set_page_config(page_title="TRADER TERMINAL v6.1", layout="wide", page_icon="📈")

st.markdown("""<style>
.stApp{background-color:#04080d!important}
section[data-testid="stSidebar"]{background-color:#070e17!important;border-right:1px solid #0d2035}
.stTabs [data-baseweb="tab-list"]{background-color:#070e17;gap:0}
.stTabs [data-baseweb="tab"]{color:#4a7a9a;font-family:'Courier New',monospace;font-size:.72rem;font-weight:700;letter-spacing:.1em;padding:10px 16px}
.stTabs [aria-selected="true"]{color:#00d4ff!important;border-bottom:2px solid #00d4ff!important}
[data-testid="stMetricLabel"]{color:#4a7a9a!important;font-size:.65rem!important;text-transform:uppercase;letter-spacing:.1em}
[data-testid="stMetricValue"]{color:white!important;font-family:'Courier New',monospace;font-weight:700}
.stTextInput>div>div>input{background-color:#0b1520!important;border:1px solid #0d2035!important;color:#00d4ff!important;font-family:'Courier New',monospace!important;font-weight:700!important;text-transform:uppercase}
.stButton>button{background:transparent!important;border:1px solid #00d4ff!important;color:#00d4ff!important;font-family:'Courier New',monospace!important;font-weight:700!important;letter-spacing:.1em!important;text-transform:uppercase!important;border-radius:0!important}
.stButton>button:hover{background:#00d4ff!important;color:#04080d!important}
footer{visibility:hidden}#MainMenu{visibility:hidden}
</style>""", unsafe_allow_html=True)


@st.cache_data(ttl=30)
def get_data(sym, interval="15m", period="5d"):
    try:
        df = yf.download(sym, interval=interval, period=period, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df if not df.empty and len(df) >= 20 else None
    except Exception:
        return None


def calc_rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p - 1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=p - 1, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


def calc_macd(s):
    m = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    sig = m.ewm(span=9, adjust=False).mean()
    return m, sig, m - sig


def calc_bb(s, p=20, k=2):
    m = s.rolling(p).mean()
    std = s.rolling(p).std()
    return m + k * std, m, m - k * std


def calc_atr(df, p=14):
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    return float(tr.rolling(p).mean().iloc[-1])


def calc_vwap(df):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()


def run_bots(df):
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    rsi = calc_rsi(c)
    ml, ms, mh = calc_macd(c)
    bu, bm, bl = calc_bb(c)
    vw = calc_vwap(df)
    atr = calc_atr(df)

    def safe(series, default=0):
        val = series.iloc[-1]
        return float(val) if not pd.isna(val) else default

    px = float(c.iloc[-1])
    rv = safe(rsi, 50)
    mv = safe(ml)
    msv = safe(ms)
    s20 = float(c.rolling(20).mean().iloc[-1]) if len(c) >= 20 else px
    s50 = float(c.rolling(50).mean().iloc[-1]) if len(c) >= 50 else px
    vv = float(vw.iloc[-1])
    bl_ = safe(bl, px * 0.95)
    bu_ = safe(bu, px * 1.05)
    pctB = (px - bl_) / (bu_ - bl_) if bu_ > bl_ else 0.5
    avg_v = float(v.iloc[-20:].mean())
    v_spike = float(v.iloc[-1]) > avg_v * 1.8
    rsi_sl = rsi.dropna().iloc[-14:]
    lo, hi = float(rsi_sl.min()), float(rsi_sl.max())
    stk = (rv - lo) / (hi - lo) * 100 if hi > lo else 50

    mom = "B" if (px > s20 and rv > 55 and mv > msv) else "R" if (px < s20 and rv < 45 and mv < msv) else "N"
    rev = "B" if rv < 30 else "R" if rv > 70 else "N"
    vs  = "B" if px > vv * 1.001 else "R" if px < vv * 0.999 else "N"
    bs  = "B" if pctB < 0.15 else "R" if pctB > 0.85 else "N"
    ss  = "B" if stk < 25 else "R" if stk > 75 else "N"

    arrow_up = "\u2191"
    arrow_dn = "\u2193"
    bots = [
        {"name": "Momentum",   "sig": mom, "detail": "RSI {:.1f} | MACD {} | Vol {}".format(rv, arrow_up if mv > msv else arrow_dn, "SPIKE" if v_spike else "Normal")},
        {"name": "Reversal",   "sig": rev, "detail": "RSI {:.1f} | {}".format(rv, "Oversold" if rv < 30 else "Overbought" if rv > 70 else "Neutral")},
        {"name": "VWAP",       "sig": vs,  "detail": "{:+.2f}% vs VWAP ${:.2f}".format((px / vv - 1) * 100, vv)},
        {"name": "Bollinger",  "sig": bs,  "detail": "%B {:.2f} | ATR ${:.2f}".format(pctB, atr)},
        {"name": "Stochastic", "sig": ss,  "detail": "%K {:.1f}".format(stk)},
    ]
    sc = sum(1 if b["sig"] == "B" else -1 if b["sig"] == "R" else 0 for b in bots)
    overall = "CALL" if sc >= 2 else "PUT" if sc <= -2 else "NEUTRAL"
    prev = float(c.iloc[-2]) if len(c) >= 2 else px
    return {
        "bots": bots, "overall": overall, "pct": abs(sc) / 5 * 100,
        "px": px, "chg": (px - prev) / prev * 100,
        "rv": rv, "atr": atr, "vv": vv, "s20": s20, "s50": s50,
        "mv": mv, "msv": msv, "pctB": pctB, "stk": stk, "sc": sc,
        "rsi": rsi, "ml": ml, "ms": ms, "mh": mh,
        "bu": bu, "bm": bm, "bl": bl, "vw": vw,
    }


def run_backtest(df, strat="momentum"):
    c, h, l = df["Close"], df["High"], df["Low"]
    rsi = calc_rsi(c)
    ml, ms, _ = calc_macd(c)
    s20 = c.rolling(20).mean()
    pos = None
    trades, equity, peak, wins, losses, maxDD = [], 10000.0, 10000.0, 0, 0, 0.0
    for i in range(30, len(c)):
        rv  = float(rsi.iloc[i]) if not pd.isna(rsi.iloc[i]) else None
        mv  = float(ml.iloc[i])  if not pd.isna(ml.iloc[i])  else None
        msv = float(ms.iloc[i])  if not pd.isna(ms.iloc[i])  else None
        s   = float(s20.iloc[i]) if not pd.isna(s20.iloc[i]) else None
        if None in (rv, mv, msv, s):
            continue
        ci = float(c.iloc[i])
        if strat == "momentum":
            sig = "B" if (ci > s and rv > 55 and mv > msv) else "R" if (ci < s and rv < 45 and mv < msv) else "N"
        else:
            sig = "B" if rv < 28 else "R" if rv > 72 else "N"
        if pos is None and sig != "N":
            pos = {"dir": sig, "entry": ci,
                   "sl": ci * 0.97 if sig == "B" else ci * 1.03,
                   "tp": ci * 1.05 if sig == "B" else ci * 0.95}
        elif pos:
            hi_ = float(h.iloc[i])
            lo_ = float(l.iloc[i])
            sl_hit = lo_ <= pos["sl"] if pos["dir"] == "B" else hi_ >= pos["sl"]
            tp_hit = hi_ >= pos["tp"] if pos["dir"] == "B" else lo_ <= pos["tp"]
            if sl_hit or tp_hit or i == len(c) - 1:
                ex = pos["tp"] if tp_hit else pos["sl"]
                pnl = (ex - pos["entry"]) / pos["entry"] if pos["dir"] == "B" else (pos["entry"] - ex) / pos["entry"]
                equity *= (1 + pnl * 0.1)
                peak = max(peak, equity)
                maxDD = max(maxDD, (peak - equity) / peak)
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                trades.append({
                    "dir": "LONG" if pos["dir"] == "B" else "SHORT",
                    "entry": round(pos["entry"], 2),
                    "exit": round(ex, 2),
                    "pnl%": round(pnl * 100, 2),
                    "result": "WIN" if pnl > 0 else "LOSS",
                })
                pos = None
    total = wins + losses
    return {
        "equity": round(equity, 2), "trades": trades[-8:],
        "wins": wins, "losses": losses,
        "wr": round(wins / total * 100, 1) if total else 0,
        "maxDD": round(maxDD * 100, 1), "total": total,
    }


def gen_ai(sym, sig):
    v = sig["overall"]
    cv = "HIGH" if sig["pct"] >= 80 else "MODERATE" if sig["pct"] >= 60 else "LOW"
    bots_txt = "\n".join(
        "* {:<12}{:<10}{}".format(
            b["name"],
            "BULLISH" if b["sig"] == "B" else "BEARISH" if b["sig"] == "R" else "NEUTRAL",
            b["detail"],
        )
        for b in sig["bots"]
    )
    if v == "CALL":
        rat = "Bullish confluence on {:.0f}% of indicators.\nCALL setup: Price above MAs. Target +5-8% | Stop below SMA20.".format(sig["pct"])
    elif v == "PUT":
        rat = "Bearish signals on {:.0f}% of indicators.\nPUT setup: Price below MAs. Target -5-8% | Stop above SMA20.".format(sig["pct"])
    else:
        rat = "Mixed signals - no clear edge. Wait for cleaner setup."
    above_below = "ABOVE" if sig["px"] > sig["s20"] else "BELOW"
    bull_bear = "Bullish" if sig["px"] > sig["s50"] else "Bearish"
    rsi_label = "OVERSOLD" if sig["rv"] < 30 else "OVERBOUGHT" if sig["rv"] > 70 else "NEUTRAL"
    macd_label = "Bullish" if sig["mv"] > sig["msv"] else "Bearish"
    lines = [
        "GROK AI ANALYSIS -- {}".format(sym),
        "-" * 38,
        "VERDICT:    {}".format(v),
        "CONVICTION: {} ({:.0f}%)".format(cv, sig["pct"]),
        "TIME:       {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "",
        "TECHNICAL:",
        "* ${:.2f} -- {} SMA20 (${:.2f})".format(sig["px"], above_below, sig["s20"]),
        "* Trend: {} vs SMA50 (${:.2f})".format(bull_bear, sig["s50"]),
        "* RSI(14): {:.1f} -- {}".format(sig["rv"], rsi_label),
        "* MACD: {}".format(macd_label),
        "",
        "BOT SIGNALS:",
        bots_txt,
        "",
        "RATIONALE:",
        rat,
        "",
        "! Educational use only. Not financial advice.",
    ]
    return "\n".join(lines)


# ── SIDEBAR ──
with st.sidebar:
    st.markdown("## TRADER TERMINAL v6.1")
    st.markdown("---")
    ticker = st.text_input("SYMBOL", value="PLTR").upper().strip() or "PLTR"
    tf_opts = {"5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "Daily": "1d"}
    tf = tf_opts[st.selectbox("TIMEFRAME", list(tf_opts.keys()), index=1)]
    st.markdown("---")
    st.markdown("**WATCHLIST**")
    if "wl" not in st.session_state:
        st.session_state.wl = ["SPY", "QQQ", "NVDA", "TSLA", "AAPL"]
    if "credits" not in st.session_state:
        st.session_state.credits = 100
    new_sym = st.text_input("Add symbol", placeholder="TICKER", label_visibility="collapsed", key="wl_inp")
    if st.button("+ ADD", use_container_width=True) and new_sym.strip():
        s = new_sym.strip().upper()
        if s not in st.session_state.wl:
            st.session_state.wl.append(s)
            st.rerun()
    for s in st.session_state.wl[:]:
        c1, c2 = st.columns([4, 1])
        c1.write(s)
        if c2.button("X", key="rm_" + s):
            st.session_state.wl.remove(s)
            st.rerun()
    st.markdown("---")
    st.write("Credits: {}".format(st.session_state.credits))


# ── LOAD DATA ──
st.markdown("## {}".format(ticker))
df = get_data(ticker, tf, "5d")
sig = None
if df is not None and len(df) >= 30:
    try:
        sig = run_bots(df)
    except Exception as e:
        st.error("Signal error: {}".format(e))

tab1, tab2, tab3, tab4, tab5 = st.tabs(["DASH", "CHART", "BOTS", "BACKTEST", "GROK AI"])


# ── DASH ──
with tab1:
    if sig:
        col1, col2 = st.columns([3, 1])
        with col1:
            chg_sign = "+" if sig["chg"] >= 0 else ""
            st.metric(ticker, "${:.2f}".format(sig["px"]), "{}{:.2f}%".format(chg_sign, sig["chg"]))
            st.caption("ATR: ${:.2f}  |  VWAP: ${:.2f}  |  SMA20: ${:.2f}".format(sig["atr"], sig["vv"], sig["s20"]))
        with col2:
            icon = "UP" if sig["overall"] == "CALL" else "DN" if sig["overall"] == "PUT" else "--"
            st.metric("VERDICT", "{} {}".format(icon, sig["overall"]), "{:.0f}% conviction".format(sig["pct"]))
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        rsi_lbl = "Oversold" if sig["rv"] < 30 else "Overbought" if sig["rv"] > 70 else "Neutral"
        m1.metric("RSI(14)", "{:.1f}".format(sig["rv"]), rsi_lbl)
        m2.metric("ATR", "${:.2f}".format(sig["atr"]))
        m3.metric("vs VWAP", "{:+.2f}%".format((sig["px"] / sig["vv"] - 1) * 100), "VWAP ${:.2f}".format(sig["vv"]))
        m4.metric("SMA50", "${:.2f}".format(sig["s50"]), "Above" if sig["px"] > sig["s50"] else "Below")
        st.markdown("---")
        cols = st.columns(3)
        for i, b in enumerate(sig["bots"]):
            lbl = "BULLISH" if b["sig"] == "B" else "BEARISH" if b["sig"] == "R" else "NEUTRAL"
            with cols[i % 3]:
                st.markdown("**{}** — {}".format(b["name"], lbl))
                st.caption(b["detail"])
    else:
        st.error("Could not load data for {}. Check the symbol.".format(ticker))


# ── CHART ──
with tab2:
    if df is not None and len(df) >= 20:
        overlay = st.radio("Overlay", ["SMA", "EMA", "VWAP", "BB"], horizontal=True)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            row_heights=[0.62, 0.18, 0.20], vertical_spacing=0.02)
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            increasing_line_color="#00ff88", decreasing_line_color="#ff3355",
            name=ticker, showlegend=False), row=1, col=1)
        if overlay == "SMA":
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(20).mean(), name="SMA20", line=dict(color="#00d4ff", width=1.2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(50).mean(), name="SMA50", line=dict(color="#ffd700", width=1.2)), row=1, col=1)
        elif overlay == "EMA":
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"].ewm(span=12).mean(), name="EMA12", line=dict(color="#00d4ff", width=1.2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"].ewm(span=26).mean(), name="EMA26", line=dict(color="#ffd700", width=1.2)), row=1, col=1)
        elif overlay == "VWAP" and sig:
            fig.add_trace(go.Scatter(x=df.index, y=sig["vw"], name="VWAP", line=dict(color="#ff9900", width=1.5, dash="dot")), row=1, col=1)
        elif overlay == "BB" and sig:
            fig.add_trace(go.Scatter(x=df.index, y=sig["bu"], name="BB Up", line=dict(color="rgba(0,212,255,.5)", width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=sig["bm"], name="BB Mid", line=dict(color="rgba(255,215,0,.4)", width=1, dash="dash")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=sig["bl"], name="BB Lo", line=dict(color="rgba(0,212,255,.5)", width=1), fill="tonexty", fillcolor="rgba(0,212,255,.03)"), row=1, col=1)
        if sig is not None:
            fig.add_trace(go.Scatter(x=df.index, y=sig["rsi"], name="RSI", line=dict(color="#ff3355", width=1.5), showlegend=False), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,51,85,.3)", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,255,136,.3)", row=2, col=1)
        colors = []
        for i in range(len(df)):
            if i == 0 or float(df["Close"].iloc[i]) >= float(df["Close"].iloc[i - 1]):
                colors.append("rgba(0,255,136,.3)")
            else:
                colors.append("rgba(255,51,85,.3)")
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=colors, showlegend=False), row=3, col=1)
        fig.update_layout(height=680, template="plotly_dark", paper_bgcolor="#04080d", plot_bgcolor="#070e17",
                          xaxis_rangeslider_visible=False,
                          font=dict(color="#a8c8e0", family="Courier New", size=10),
                          margin=dict(l=0, r=0, t=10, b=0))
        fig.update_yaxes(gridcolor="#0d2035")
        fig.update_xaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Cannot load chart for {}".format(ticker))


# ── BOTS ──
with tab3:
    if sig:
        st.info("5 bots vote BULLISH / BEARISH / NEUTRAL. Score >= +2 = CALL, <= -2 = PUT.")
        for b in sig["bots"]:
            lbl = "BULLISH" if b["sig"] == "B" else "BEARISH" if b["sig"] == "R" else "NEUTRAL"
            c1, c2, c3 = st.columns([2, 2, 4])
            c1.write("**{}**".format(b["name"]))
            c2.write(lbl)
            c3.caption(b["detail"])
            st.divider()
        st.write("Score: {}/5  =>  **{}**".format(sig["sc"], sig["overall"]))
    else:
        st.error("No data for {}".format(ticker))


# ── BACKTEST ──
with tab4:
    strat = st.radio("Strategy", ["momentum", "reversal"], horizontal=True)
    if st.button("RUN BACKTEST (60 days)", use_container_width=True):
        with st.spinner("Backtesting {}...".format(ticker)):
            df60 = get_data(ticker, "15m", "60d")
            if df60 is not None and len(df60) >= 30:
                bt = run_backtest(df60, strat)
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Total Trades", bt["total"])
                c2.metric("Win Rate", "{}%".format(bt["wr"]))
                c3.metric("Winners", bt["wins"])
                c4.metric("Losers", bt["losses"])
                c5.metric("Max Drawdown", "{}%".format(bt["maxDD"]))
                eq_delta = bt["equity"] - 10000
                st.metric("Final Equity", "${:,.2f}".format(bt["equity"]), "{:+,.2f} from $10,000".format(eq_delta))
                if bt["trades"]:
                    st.dataframe(pd.DataFrame(bt["trades"]), use_container_width=True, hide_index=True)
            else:
                st.error("Not enough data for backtest")


# ── GROK AI ──
with tab5:
    st.write("Credits remaining: {}  (10 per analysis)".format(st.session_state.credits))
    disabled = not sig or st.session_state.credits < 10
    if st.button("GET GROK AI ANALYSIS", use_container_width=True, disabled=disabled):
        with st.spinner("Analyzing {}...".format(ticker)):
            time.sleep(0.8)
            st.session_state["ai_result"] = gen_ai(ticker, sig)
            st.session_state["ai_sym"] = ticker
            st.session_state.credits -= 10
            st.rerun()
    if "ai_result" in st.session_state and st.session_state.get("ai_sym") == ticker:
        st.code(st.session_state["ai_result"], language=None)

st.caption("TRADER TERMINAL v6.1  |  Educational use only  |  Not financial advice")
