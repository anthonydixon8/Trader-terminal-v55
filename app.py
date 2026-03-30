import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
import json
import os
import anthropic
from dotenv import load_dotenv, set_key

load_dotenv()  # Load ANTHROPIC_API_KEY from .env if present
_ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")

st.set_page_config(page_title="TRADER TERMINAL v6.1", layout="wide", page_icon="📈")

st.markdown("""
<style>
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
</style>
""", unsafe_allow_html=True)


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

    px  = float(c.iloc[-1])
    rv  = safe(rsi, 50)
    mv  = safe(ml)
    msv = safe(ms)
    s20 = float(c.rolling(20).mean().iloc[-1]) if len(c) >= 20 else px
    s50 = float(c.rolling(50).mean().iloc[-1]) if len(c) >= 50 else px
    vv  = float(vw.iloc[-1])
    bl_ = safe(bl, px * 0.95)
    bu_ = safe(bu, px * 1.05)
    pctB = (px - bl_) / (bu_ - bl_) if bu_ > bl_ else 0.5
    avg_v   = float(v.iloc[-20:].mean())
    v_spike = float(v.iloc[-1]) > avg_v * 1.8
    rsi_sl  = rsi.dropna().iloc[-14:]
    lo, hi  = float(rsi_sl.min()), float(rsi_sl.max())
    stk     = (rv - lo) / (hi - lo) * 100 if hi > lo else 50

    mom = "B" if (px > s20 and rv > 55 and mv > msv) else "R" if (px < s20 and rv < 45 and mv < msv) else "N"
    rev = "B" if rv < 30 else "R" if rv > 70 else "N"
    vs  = "B" if px > vv * 1.001 else "R" if px < vv * 0.999 else "N"
    bs  = "B" if pctB < 0.15 else "R" if pctB > 0.85 else "N"
    ss  = "B" if stk < 25 else "R" if stk > 75 else "N"

    bots = [
        {"name": "Momentum",   "sig": mom, "detail": "RSI {:.1f} | MACD {} | Vol {}".format(rv, "UP" if mv > msv else "DN", "SPIKE" if v_spike else "Normal")},
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
        sig = ("B" if (ci > s and rv > 55 and mv > msv) else "R" if (ci < s and rv < 45 and mv < msv) else "N") if strat == "momentum" else ("B" if rv < 28 else "R" if rv > 72 else "N")
        if pos is None and sig != "N":
            pos = {"dir": sig, "entry": ci, "sl": ci * 0.97 if sig == "B" else ci * 1.03, "tp": ci * 1.05 if sig == "B" else ci * 0.95}
        elif pos:
            sl_hit = float(l.iloc[i]) <= pos["sl"] if pos["dir"] == "B" else float(h.iloc[i]) >= pos["sl"]
            tp_hit = float(h.iloc[i]) >= pos["tp"] if pos["dir"] == "B" else float(l.iloc[i]) <= pos["tp"]
            if sl_hit or tp_hit or i == len(c) - 1:
                ex  = pos["tp"] if tp_hit else pos["sl"]
                pnl = (ex - pos["entry"]) / pos["entry"] if pos["dir"] == "B" else (pos["entry"] - ex) / pos["entry"]
                equity *= (1 + pnl * 0.1)
                peak    = max(peak, equity)
                maxDD   = max(maxDD, (peak - equity) / peak)
                wins += 1 if pnl > 0 else 0
                losses += 1 if pnl <= 0 else 0
                trades.append({"dir": "LONG" if pos["dir"] == "B" else "SHORT", "entry": round(pos["entry"], 2), "exit": round(ex, 2), "pnl%": round(pnl * 100, 2), "result": "WIN" if pnl > 0 else "LOSS"})
                pos = None
    total = wins + losses
    return {"equity": round(equity, 2), "trades": trades[-8:], "wins": wins, "losses": losses, "wr": round(wins / total * 100, 1) if total else 0, "maxDD": round(maxDD * 100, 1), "total": total}


def gen_ai(sym, sig):
    v  = sig["overall"]
    cv = "HIGH" if sig["pct"] >= 80 else "MODERATE" if sig["pct"] >= 60 else "LOW"
    bots_txt = "\n".join("* {:<12}{:<10}{}".format(b["name"], "BULLISH" if b["sig"] == "B" else "BEARISH" if b["sig"] == "R" else "NEUTRAL", b["detail"]) for b in sig["bots"])
    rat = ("Bullish confluence on {:.0f}% of indicators.\nCALL setup: Price above MAs. Target +5-8% | Stop below SMA20.".format(sig["pct"]) if v == "CALL"
           else "Bearish signals on {:.0f}% of indicators.\nPUT setup: Price below MAs. Target -5-8% | Stop above SMA20.".format(sig["pct"]) if v == "PUT"
           else "Mixed signals - no clear edge. Wait for cleaner setup.")
    return "\n".join([
        "GROK AI ANALYSIS -- {}".format(sym), "-" * 38,
        "VERDICT:    {}".format(v), "CONVICTION: {} ({:.0f}%)".format(cv, sig["pct"]),
        "TIME:       {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), "",
        "TECHNICAL:",
        "* ${:.2f} -- {} SMA20 (${:.2f})".format(sig["px"], "ABOVE" if sig["px"] > sig["s20"] else "BELOW", sig["s20"]),
        "* Trend: {} vs SMA50 (${:.2f})".format("Bullish" if sig["px"] > sig["s50"] else "Bearish", sig["s50"]),
        "* RSI(14): {:.1f} -- {}".format(sig["rv"], "OVERSOLD" if sig["rv"] < 30 else "OVERBOUGHT" if sig["rv"] > 70 else "NEUTRAL"),
        "* MACD: {}".format("Bullish" if sig["mv"] > sig["msv"] else "Bearish"), "",
        "BOT SIGNALS:", bots_txt, "", "RATIONALE:", rat, "",
        "! Educational use only. Not financial advice."
    ])


def build_chart(df, sig, overlay, draw_mode):
    """Build TradingView-style Plotly chart with drawing tools."""
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.60, 0.20, 0.20],
        vertical_spacing=0.02,
        subplot_titles=("", "RSI(14)", "MACD(12,26,9)")
    )

    # ── Candlesticks ──
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        increasing_line_color="#00ff88", decreasing_line_color="#ff3355",
        increasing_fillcolor="rgba(0,255,136,0.85)", decreasing_fillcolor="rgba(255,51,85,0.85)",
        name="Price", showlegend=False, line=dict(width=1)
    ), row=1, col=1)

    # ── Overlays ──
    if overlay == "SMA":
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(20).mean(), name="SMA20", line=dict(color="#00d4ff", width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(50).mean(), name="SMA50", line=dict(color="#ffd700", width=1.2)), row=1, col=1)
    elif overlay == "EMA":
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"].ewm(span=12).mean(), name="EMA12", line=dict(color="#00d4ff", width=1.2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"].ewm(span=26).mean(), name="EMA26", line=dict(color="#ffd700", width=1.2)), row=1, col=1)
    elif overlay == "VWAP" and sig:
        fig.add_trace(go.Scatter(x=df.index, y=sig["vw"], name="VWAP", line=dict(color="#ff9900", width=1.5, dash="dot")), row=1, col=1)
    elif overlay == "BB" and sig:
        fig.add_trace(go.Scatter(x=df.index, y=sig["bu"], name="BB Up", line=dict(color="rgba(0,212,255,0.5)", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=sig["bm"], name="BB Mid", line=dict(color="rgba(255,215,0,0.4)", width=1, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=sig["bl"], name="BB Lo", line=dict(color="rgba(0,212,255,0.5)", width=1), fill="tonexty", fillcolor="rgba(0,212,255,0.03)"), row=1, col=1)

    # ── Volume (secondary y-axis on main pane) ──
    vol_colors = ["rgba(0,255,136,0.25)" if float(df["Close"].iloc[i]) >= float(df["Close"].iloc[i - 1]) else "rgba(255,51,85,0.25)" for i in range(len(df))]
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=vol_colors, showlegend=False, yaxis="y4"), row=1, col=1)

    # ── RSI ──
    if sig is not None:
        fig.add_trace(go.Scatter(x=df.index, y=sig["rsi"], name="RSI", line=dict(color="#ff3355", width=1.5), showlegend=False), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,51,85,0.35)", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,255,136,0.35)", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot",  line_color="rgba(74,122,154,0.2)",  row=2, col=1)

        # ── MACD ──
        macd_colors = ["rgba(0,255,136,0.6)" if v >= 0 else "rgba(255,51,85,0.6)" for v in sig["mh"].fillna(0)]
        fig.add_trace(go.Bar(x=df.index, y=sig["mh"], name="MACD Hist", marker_color=macd_colors, showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=sig["ml"], name="MACD",   line=dict(color="#00d4ff", width=1.2)), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=sig["ms"], name="Signal", line=dict(color="#ffd700", width=1.2)), row=3, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="rgba(74,122,154,0.3)", row=3, col=1)

    # ── Layout ──
    fig.update_layout(
        height=720,
        template="plotly_dark",
        paper_bgcolor="#04080d",
        plot_bgcolor="#070e17",
        xaxis_rangeslider_visible=False,
        font=dict(color="#a8c8e0", family="Courier New", size=10),
        margin=dict(l=0, r=0, t=28, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=9), orientation="h", y=1.02),
        dragmode=draw_mode,
        newshape=dict(
            line=dict(color="#00d4ff", width=2),
            fillcolor="rgba(0,212,255,0.07)",
            opacity=0.9,
        ),
        yaxis4=dict(
            overlaying="y",
            side="right",
            showgrid=False,
            showticklabels=False,
            range=[0, df["Volume"].max() * 5],
        ),
    )
    fig.update_yaxes(gridcolor="#0d2035", gridwidth=1, zerolinecolor="#0d2035")
    fig.update_xaxes(showgrid=False, rangeslider_visible=False)

    return fig


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
    # Seed API key from .env on first load
    if "ant_key" not in st.session_state:
        st.session_state["ant_key"] = os.getenv("ANTHROPIC_API_KEY", "")
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
    st.markdown("---")
    st.markdown("**ANTHROPIC KEY**")
    _key_display = ("sk-ant-..." + st.session_state["ant_key"][-6:]) if st.session_state["ant_key"] else ""
    _new_key = st.text_input(
        "API Key", type="password",
        placeholder="sk-ant-api03-...",
        value=_key_display,
        label_visibility="collapsed",
        key="ant_key_input",
    )
    _col1, _col2 = st.columns(2)
    if _col1.button("SAVE", use_container_width=True, key="save_key_btn"):
        _raw = _new_key.strip()
        if _raw.startswith("sk-ant-"):
            st.session_state["ant_key"] = _raw
            set_key(_ENV_PATH, "ANTHROPIC_API_KEY", _raw)
            st.success("Key saved to .env")
        else:
            st.error("Key must start with sk-ant-")
    if _col2.button("CLEAR", use_container_width=True, key="clear_key_btn"):
        st.session_state["ant_key"] = ""
        set_key(_ENV_PATH, "ANTHROPIC_API_KEY", "")
        st.rerun()
    if st.session_state["ant_key"]:
        st.caption("Key loaded ✓")
    else:
        st.caption("No key set")


# ── TRADESWARM ────────────────────────────────────────────────────────────────
SWARM_BOTS = [
    {"id": "ARIA",  "role": "MOMENTUM SCANNER",  "color": "#00ff88", "icon": "◈",
     "tags": ["RSI-14", "Volume", "Bollinger", "Momentum"]},
    {"id": "NEXUS", "role": "TREND ANALYST",      "color": "#00cfff", "icon": "⬡",
     "tags": ["50 SMA", "200 SMA", "EMA Cross", "VWAP"]},
    {"id": "SIGMA", "role": "SENTIMENT READER",   "color": "#ff6b35", "icon": "⬟",
     "tags": ["Put/Call", "IV Rank", "Dark Pool", "Options Flow"]},
    {"id": "DELTA", "role": "DIVERGENCE HUNTER",  "color": "#c084fc", "icon": "⬠",
     "tags": ["MACD Div", "RSI Div", "CVD Delta", "Exhaustion"]},
]


def _swarm_prompt(sym):
    return (
        'Analyze stock {}. Reply with ONLY this JSON (no markdown, no backticks, just the object):\n'
        '{{"ARIA":{{"verdict":"CALL","confidence":72,"analysis":"Two sentences about {} RSI and momentum.",'
        '"rsi":58,"vol":112,"mom":65}},'
        '"NEXUS":{{"verdict":"CALL","confidence":74,"analysis":"Two sentences about {} moving averages.",'
        '"sma":68,"ema":72,"trend":70}},'
        '"SIGMA":{{"verdict":"PUT","confidence":69,"analysis":"Two sentences about {} options flow.",'
        '"pcr":62,"ivr":45,"flow":55}},'
        '"DELTA":{{"verdict":"PUT","confidence":76,"rev":"NONE","analysis":"Two sentences about {} MACD/RSI divergence.",'
        '"macd":65,"rsid":58,"cvd":60}}}}\n'
        'Rules: Each bot different focus. Verdicts may differ. '
        'DELTA rev = BULLISH_REVERSAL, BEARISH_REVERSAL, or NONE. All numbers 0-100. Be specific to {}.'
    ).format(sym, sym, sym, sym, sym, sym)


def _extract_json(text):
    try:
        s = text.strip()
        a, b = s.find('{'), s.rfind('}')
        if a == -1 or b == -1:
            return None
        return json.loads(s[a:b + 1])
    except Exception:
        return None


def _parse_swarm_bot(bot_id, data):
    d = data.get(bot_id, {})
    verdict = "CALL" if "CALL" in str(d.get("verdict", "")).upper() else "PUT"
    confidence = min(94, max(55, int(d.get("confidence", 68))))
    analysis = str(d.get("analysis", "Analysis complete.")).strip()
    reversal = None
    if bot_id == "DELTA":
        rev = str(d.get("rev", "NONE"))
        reversal = rev if rev in ("BULLISH_REVERSAL", "BEARISH_REVERSAL") else "NONE"
    metrics_map = {
        "ARIA":  [("RSI-14", d.get("rsi", 55), 100), ("Volume %", d.get("vol", 55), 150), ("Momentum", d.get("mom", 55), 100)],
        "NEXUS": [("50 SMA Score", d.get("sma", 55), 100), ("EMA Strength", d.get("ema", 55), 100), ("Trend Align", d.get("trend", 55), 100)],
        "SIGMA": [("Put/Call Score", d.get("pcr", 55), 100), ("IV Rank", d.get("ivr", 55), 100), ("Bullish Flow", d.get("flow", 55), 100)],
        "DELTA": [("MACD Divergence", d.get("macd", 55), 100), ("RSI Divergence", d.get("rsid", 55), 100), ("CVD Imbalance", d.get("cvd", 55), 100)],
    }
    metrics = [{"label": lbl, "value": min(mx, max(0, float(v or 55))), "max": mx}
               for lbl, v, mx in metrics_map.get(bot_id, [])]
    return {"verdict": verdict, "confidence": confidence, "analysis": analysis,
            "metrics": metrics, "reversal": reversal}


def call_tradeswarm(sym, api_key):
    """Call Claude API and return list of 4 parsed bot results, or raise."""
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system="You are a stock analysis API. Output ONLY a raw JSON object. No markdown. No explanation. Just JSON.",
        messages=[{"role": "user", "content": _swarm_prompt(sym)}],
    )
    raw = "".join(c.text for c in msg.content if hasattr(c, "text")).strip()
    data = _extract_json(raw)
    if data is None:
        raise ValueError("Could not parse JSON from response: " + raw[:200])
    return [_parse_swarm_bot(b["id"], data) for b in SWARM_BOTS], raw


# ── LOAD DATA ──
st.markdown("## {}".format(ticker))
df = get_data(ticker, tf, "5d")
sig = None
if df is not None and len(df) >= 30:
    try:
        sig = run_bots(df)
    except Exception as e:
        st.error("Signal error: {}".format(e))

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["DASH", "CHART", "BOTS", "BACKTEST", "GROK AI", "TRADESWARM ◈"])


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
        m1.metric("RSI(14)", "{:.1f}".format(sig["rv"]), "Oversold" if sig["rv"] < 30 else "Overbought" if sig["rv"] > 70 else "Neutral")
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
        st.error("Could not load data for {}.".format(ticker))


# ── CHART ──
with tab2:
    if df is not None and len(df) >= 20:

        # Controls row
        ctrl1, ctrl2, ctrl3 = st.columns([3, 4, 3])
        with ctrl1:
            overlay = st.radio("Overlay", ["None", "SMA", "EMA", "VWAP", "BB"], horizontal=True, key="ov")
        with ctrl2:
            draw_tool = st.radio(
                "Drawing Tool",
                ["Pan", "Trend Line", "H-Line", "V-Line", "Rectangle", "Erase"],
                horizontal=True, key="dt"
            )
        with ctrl3:
            if st.button("Clear All Drawings", key="clear_shapes"):
                st.session_state["shapes"] = []
                st.rerun()

        # Map drawing tool to Plotly dragmode
        draw_mode_map = {
            "Pan":       "pan",
            "Trend Line": "drawline",
            "H-Line":    "drawline",
            "V-Line":    "drawline",
            "Rectangle": "drawrect",
            "Erase":     "eraseshape",
        }
        drag_mode = draw_mode_map[draw_tool]

        st.caption(
            "Tip: Select a drawing tool above, then click and drag on the chart to draw. "
            "Shapes are saved until you click **Clear All Drawings**."
        )

        fig = build_chart(df, sig, overlay if overlay != "None" else "", drag_mode)

        # Plotly config: enable full drawing toolbar
        config = {
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToAdd": [
                "drawline",
                "drawopenpath",
                "drawclosedpath",
                "drawcircle",
                "drawrect",
                "eraseshape",
            ],
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "displaylogo": False,
            "toImageButtonOptions": {"format": "png", "filename": "trader_terminal_chart"},
        }

        st.plotly_chart(fig, use_container_width=True, config=config, key="main_chart")

        # Drawing instructions
        st.markdown("""
        **Drawing Tools Guide**
        | Tool | How to use |
        |------|------------|
        | Trend Line | Click pencil icon in chart toolbar, drag to draw |
        | H-Line | Draw a horizontal line for support / resistance |
        | Rectangle | Click rect icon, drag to mark a price zone |
        | Erase | Click eraser icon, click a shape to delete it |
        | Pan / Zoom | Drag to pan, scroll to zoom, double-click to reset |
        """)
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
                st.metric("Final Equity", "${:,.2f}".format(bt["equity"]), "{:+,.2f} from $10,000".format(bt["equity"] - 10000))
                if bt["trades"]:
                    st.dataframe(pd.DataFrame(bt["trades"]), use_container_width=True, hide_index=True)
            else:
                st.error("Not enough data for backtest")


# ── GROK AI ──
with tab5:
    st.write("Credits remaining: {}  (10 per analysis)".format(st.session_state.credits))
    if st.button("GET GROK AI ANALYSIS", use_container_width=True, disabled=not sig or st.session_state.credits < 10):
        with st.spinner("Analyzing {}...".format(ticker)):
            time.sleep(0.8)
            st.session_state["ai_result"] = gen_ai(ticker, sig)
            st.session_state["ai_sym"] = ticker
            st.session_state.credits -= 10
            st.rerun()
    if "ai_result" in st.session_state and st.session_state.get("ai_sym") == ticker:
        st.code(st.session_state["ai_result"], language=None)

# ── TRADESWARM ──
with tab6:
    st.markdown("""
    <div style="text-align:center;padding:10px 0 18px">
      <div style="font-size:22px;font-weight:900;letter-spacing:7px;
        background:linear-gradient(90deg,#00ff88,#00cfff,#c084fc,#ff6b35);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent">TRADESWARM</div>
      <div style="color:#334;font-size:10px;letter-spacing:3px;margin-top:2px">
        4-BOT PUT/CALL ANALYZER · POWERED BY CLAUDE
      </div>
    </div>""", unsafe_allow_html=True)

    _api_key = st.session_state.get("ant_key", "")
    if not _api_key:
        st.warning("Add your Anthropic API key in the sidebar to use TRADESWARM.")
    else:
        st.caption("Using key: sk-ant-...{}  |  Ticker: **{}**".format(_api_key[-4:], ticker))

    _can_run = bool(_api_key) and sig is not None
    if st.button("▶ ANALYZE WITH TRADESWARM", use_container_width=True,
                 disabled=not _can_run, key="swarm_run_btn"):
        with st.spinner("Deploying swarm on {}...".format(ticker)):
            try:
                _results, _raw = call_tradeswarm(ticker, _api_key)
                st.session_state["swarm_result"] = _results
                st.session_state["swarm_sym"] = ticker
                st.session_state["swarm_raw"] = _raw
            except Exception as _e:
                st.session_state["swarm_error"] = str(_e)
                st.session_state["swarm_sym"] = ticker
                st.session_state.pop("swarm_result", None)

    # ── Results ──
    if st.session_state.get("swarm_sym") == ticker:
        _err = st.session_state.get("swarm_error")
        _res = st.session_state.get("swarm_result")

        if _err:
            st.error("Error: " + _err)

        if _res:
            # Bot cards — 2 columns
            _col_l, _col_r = st.columns(2)
            for _i, (_bot, _r) in enumerate(zip(SWARM_BOTS, _res)):
                _col = _col_l if _i % 2 == 0 else _col_r
                _vc  = "#00ff88" if _r["verdict"] == "CALL" else "#ff4466"
                _bc  = _bot["color"]
                _metrics_html = "".join(
                    '<div style="margin-bottom:6px">'
                    '<div style="display:flex;justify-content:space-between;font-size:10px;color:#556">'
                    '<span>{}</span><span style="color:{}">{:.0f}</span></div>'
                    '<div style="height:4px;background:#0f0f20;border-radius:3px;overflow:hidden;margin-top:2px">'
                    '<div style="width:{:.0f}%;height:100%;background:linear-gradient(90deg,{}55,{});border-radius:3px"></div>'
                    '</div></div>'.format(m["label"], _bc, m["value"],
                                         min(100, m["value"] / m["max"] * 100), _bc, _bc)
                    for m in _r["metrics"]
                )
                _rev_html = ""
                if _r.get("reversal") and _r["reversal"] != "NONE":
                    _rc = "#00ff88" if _r["reversal"] == "BULLISH_REVERSAL" else "#ff4466"
                    _rl = "⬆ BULLISH REVERSAL" if _r["reversal"] == "BULLISH_REVERSAL" else "⬇ BEARISH REVERSAL"
                    _rev_html = ('<div style="margin-top:8px;padding:4px 9px;background:{0}10;'
                                 'border:1px solid {0}40;border-radius:6px;color:{0};'
                                 'font-size:9px;font-weight:800">{1}</div>').format(_rc, _rl)
                _tags_html = "".join(
                    '<span style="padding:2px 5px;border-radius:3px;font-size:8px;'
                    'background:{0}0c;border:1px solid {0}18;color:{0}55;margin-right:3px">{1}</span>'.format(_bc, t)
                    for t in _bot["tags"]
                )
                _col.markdown("""
                <div style="background:linear-gradient(150deg,#0b0b1c,#0f0f26);
                  border:1px solid {bc}40;border-radius:10px;padding:14px;margin-bottom:10px">
                  <div style="display:flex;align-items:center;gap:9px;margin-bottom:8px">
                    <span style="font-size:18px;color:{bc}">{icon}</span>
                    <div>
                      <div style="color:{bc};font-weight:800;font-size:13px;letter-spacing:1px">{id}</div>
                      <div style="color:{bc}66;font-size:8px">{role}</div>
                    </div>
                    <div style="margin-left:auto;padding:2px 10px;border-radius:20px;
                      background:{vc}12;border:1px solid {vc}44;color:{vc};
                      font-weight:800;font-size:11px;letter-spacing:2px">{verdict}</div>
                  </div>
                  <div style="margin-bottom:8px">{tags}</div>
                  <div style="background:#05050f;border-radius:7px;padding:10px;
                    border:1px solid #141428;font-size:11px;line-height:1.8;
                    color:#8899b0;margin-bottom:8px">{analysis}</div>
                  {metrics}
                  {rev}
                  <div style="display:flex;justify-content:space-between;margin-top:8px">
                    <span style="color:#222;font-size:9px">CONFIDENCE</span>
                    <span style="color:{bc};font-size:11px;font-weight:800">{conf}%</span>
                  </div>
                </div>""".format(
                    bc=_bc, icon=_bot["icon"], id=_bot["id"], role=_bot["role"],
                    vc=_vc, verdict=_r["verdict"], tags=_tags_html,
                    analysis=_r["analysis"], metrics=_metrics_html,
                    rev=_rev_html, conf=_r["confidence"],
                ), unsafe_allow_html=True)

            # Verdict panel
            _calls = sum(1 for r in _res if r["verdict"] == "CALL")
            _puts  = len(_res) - _calls
            _final = "CALL" if _calls >= _puts else "PUT"
            _avg_c = round(sum(r["confidence"] for r in _res) / len(_res))
            _strength = "STRONG" if _avg_c >= 80 else "MODERATE" if _avg_c >= 66 else "WEAK"
            _fc = "#00ff88" if _final == "CALL" else "#ff4466"
            _delta_rev = _res[3].get("reversal") if len(_res) > 3 else None
            _rev_badge = ""
            if _delta_rev and _delta_rev != "NONE":
                _rc2 = "#00ff88" if _delta_rev == "BULLISH_REVERSAL" else "#ff4466"
                _rl2 = "⚡ BULLISH REVERSAL" if _delta_rev == "BULLISH_REVERSAL" else "⚡ BEARISH REVERSAL"
                _rev_badge = ('<div style="margin:0 auto 14px;padding:7px 16px;display:inline-block;'
                              'background:{0}10;border:1px solid {0}44;border-radius:9px;">'
                              '<span style="color:{0};font-size:13px;font-weight:800;letter-spacing:2px">'
                              '{1}</span></div>').format(_rc2, _rl2)
            _votes = "".join(
                '<div style="display:flex;flex-direction:column;align-items:center;gap:3px">'
                '<span style="color:{bc};font-size:9px">{id}</span>'
                '<span style="padding:2px 9px;border-radius:12px;background:{vc}14;'
                'border:1px solid {vc}44;color:{vc};font-size:11px;font-weight:800">{v}</span>'
                '</div>'.format(bc=b["color"], id=b["id"],
                                vc="#00ff88" if r["verdict"]=="CALL" else "#ff4466",
                                v=r["verdict"])
                for b, r in zip(SWARM_BOTS, _res)
            )
            st.markdown("""
            <div style="background:linear-gradient(145deg,#07071a,#0c0c22);
              border:2px solid {fc};border-radius:13px;padding:22px;text-align:center;
              box-shadow:0 0 60px {fc}18;margin-top:4px">
              <div style="color:#333;font-size:10px;letter-spacing:3px;margin-bottom:4px">▸ 4-BOT CONSENSUS ◂</div>
              <div style="color:#444;font-size:10px;margin-bottom:12px">
                {ticker} · {calls}/4 CALL · {puts}/4 PUT
              </div>
              <div style="display:flex;justify-content:center;gap:10px;margin-bottom:14px;flex-wrap:wrap">
                {votes}
              </div>
              <div style="font-size:52px;font-weight:900;color:{fc};letter-spacing:10px;
                line-height:1;text-shadow:0 0 40px {fc}66;margin-bottom:5px">{final}</div>
              <div style="color:{fc}77;font-size:12px;letter-spacing:4px;margin-bottom:12px">
                {strength} SIGNAL · {avgc}% CONFIDENCE
              </div>
              {rev_badge}
              <div style="display:inline-block;padding:5px 14px;background:#ffffff06;
                border:1px solid #181830;border-radius:7px;color:#334;font-size:10px;margin-top:6px">
                ⚠ AI analysis only — not financial advice
              </div>
            </div>""".format(
                fc=_fc, ticker=ticker, calls=_calls, puts=_puts,
                votes=_votes, final=_final, strength=_strength, avgc=_avg_c,
                rev_badge=_rev_badge,
            ), unsafe_allow_html=True)

            # Raw response toggle
            with st.expander("🔍 Raw API response"):
                st.code(st.session_state.get("swarm_raw", ""), language="json")

st.caption("TRADER TERMINAL v6.1  |  Educational use only  |  Not financial advice")
