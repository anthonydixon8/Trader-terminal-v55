import streamlit as st
import anthropic
import yfinance as yf
import requests
import pandas as pd
import numpy as np
import json
import os
from dotenv import load_dotenv, set_key

load_dotenv()
_ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")

st.set_page_config(page_title="TRADESWARM", layout="wide", page_icon="◈")

st.markdown("""
<style>
*{box-sizing:border-box}
.stApp{background-color:#050510!important}
section[data-testid="stSidebar"]{background-color:#07071a!important;border-right:1px solid #1c1c30}
[data-testid="stSidebarContent"]{padding-top:24px}
.stTextInput>div>div>input{
  background-color:#08081a!important;border:1px solid #202040!important;
  color:#00ff88!important;font-family:'Courier New',monospace!important;
  font-weight:700!important;text-transform:uppercase;letter-spacing:2px}
.stButton>button{
  background:transparent!important;border:1px solid #00ff8830!important;
  color:#00ff88!important;font-family:'Courier New',monospace!important;
  font-weight:700!important;letter-spacing:.1em!important;border-radius:6px!important}
.stButton>button:hover{background:#00ff8814!important}
.stButton>button:disabled{border-color:#1c1c30!important;color:#334!important}
footer{visibility:hidden}#MainMenu{visibility:hidden}
::-webkit-scrollbar{width:3px}
::-webkit-scrollbar-thumb{background:#181830;border-radius:2px}
</style>
""", unsafe_allow_html=True)

# ── Bot definitions ────────────────────────────────────────────────────────────
BOTS = [
    {"id": "ARIA",  "role": "MOMENTUM SCANNER",  "color": "#00ff88", "icon": "◈",
     "tags": ["RSI-14", "Volume", "Bollinger", "Momentum"]},
    {"id": "NEXUS", "role": "TREND ANALYST",      "color": "#00cfff", "icon": "⬡",
     "tags": ["SMA 7/20/50", "200 SMA", "EMA Cross", "VWAP"]},
    {"id": "SIGMA", "role": "SENTIMENT READER",   "color": "#ff6b35", "icon": "⬟",
     "tags": ["Put/Call", "IV Rank", "Dark Pool", "Options Flow"]},
    {"id": "DELTA", "role": "DIVERGENCE HUNTER",  "color": "#c084fc", "icon": "⬠",
     "tags": ["MACD Div", "RSI Div", "CVD Delta", "Exhaustion"]},
    {"id": "ATLAS", "role": "MULTI-TIMEFRAME",    "color": "#ffd700", "icon": "◎",
     "tags": ["3m", "5m", "15m", "30m", "1h", "4h", "1D", "1W", "1M"]},
]

# ── Data fetching ──────────────────────────────────────────────────────────────
_YF_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Referer": "https://finance.yahoo.com/",
}

def _fetch_df(sym, interval, period):
    """Try yf.Ticker().history() first, fall back to requests v8 API."""
    # Primary: yfinance Ticker history (avoids proxy issues better than download)
    try:
        tk = yf.Ticker(sym)
        df = tk.history(period=period, interval=interval, auto_adjust=True)
        if not df.empty and len(df) >= 5:
            df.index = df.index.tz_localize(None) if df.index.tzinfo is not None else df.index
            return df
    except Exception:
        pass
    # Fallback: direct Yahoo Finance v8 API via requests
    try:
        url = "https://query2.finance.yahoo.com/v8/finance/chart/" + sym
        r = requests.get(url, params={"interval": interval, "range": period},
                         headers=_YF_HEADERS, timeout=12)
        if r.ok:
            res = r.json()["chart"]["result"][0]
            ts  = res["timestamp"]
            q   = res["indicators"]["quote"][0]
            df  = pd.DataFrame({
                "Open": q["open"], "High": q["high"], "Low": q["low"],
                "Close": q["close"], "Volume": q["volume"],
            }, index=pd.to_datetime(ts, unit="s"))
            df = df.dropna(subset=["Close"])
            if len(df) >= 5:
                return df
    except Exception:
        pass
    return None


def _calc_rsi(c, p=14):
    d = c.diff()
    g = d.clip(lower=0).ewm(com=p - 1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=p - 1, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


@st.cache_data(ttl=60)
def fetch_market_data(sym):
    """Fetch daily OHLCV and compute all technical indicators. Returns dict or None."""
    try:
        df = _fetch_df(sym, "1d", "1y")
        if df is None or len(df) < 30:
            return None

        c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
        px   = float(c.iloc[-1])
        prev = float(c.iloc[-2])
        chg  = (px - prev) / prev * 100

        # RSI
        rsi_s    = _calc_rsi(c)
        rsi      = float(rsi_s.iloc[-1])
        rsi_prev = float(rsi_s.iloc[-2])

        # MACD
        ema12     = c.ewm(span=12, adjust=False).mean()
        ema26     = c.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        macd_sig  = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = macd_line - macd_sig

        # SMAs / EMAs
        sma7   = float(c.rolling(7).mean().iloc[-1])
        sma20  = float(c.rolling(20).mean().iloc[-1])
        sma50  = float(c.rolling(50).mean().iloc[-1]) if len(c) >= 50 else None
        sma200 = float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else None

        # Bollinger %B
        bm  = c.rolling(20).mean()
        std = c.rolling(20).std()
        bu_, bl_ = bm + 2 * std, bm - 2 * std
        pctB = float(((c - bl_) / (bu_ - bl_)).iloc[-1])

        # VWAP (20-period rolling approx)
        tp   = (h + l + c) / 3
        vwap = float((tp * v).rolling(20).sum().iloc[-1] / v.rolling(20).sum().iloc[-1])

        # Volume
        last_vol  = float(v.iloc[-1])
        avg_vol   = float(v.iloc[-20:].mean())
        vol_ratio = last_vol / avg_vol if avg_vol > 0 else 1.0

        mh_last = float(macd_hist.iloc[-1])
        mh_prev = float(macd_hist.iloc[-2])
        rsi_div = ("bullish" if (px < prev and rsi > rsi_prev)
                   else "bearish" if (px > prev and rsi < rsi_prev)
                   else "none")

        high_52 = float(h.rolling(min(252, len(h))).max().iloc[-1])
        low_52  = float(l.rolling(min(252, len(l))).min().iloc[-1])

        # Options PCR
        pcr = None
        try:
            tk   = yf.Ticker(sym)
            exps = tk.options
            if exps:
                chain = tk.option_chain(exps[0])
                put_oi  = chain.puts["openInterest"].sum()
                call_oi = chain.calls["openInterest"].sum()
                if call_oi > 0:
                    pcr = round(put_oi / call_oi, 2)
        except Exception:
            pass

        return {
            "px": round(px, 2), "chg": round(chg, 2),
            "high": round(float(h.iloc[-1]), 2), "low": round(float(l.iloc[-1]), 2),
            "volume": int(last_vol), "vol_ratio": round(vol_ratio, 2),
            "rsi": round(rsi, 1),
            "macd_line": round(float(macd_line.iloc[-1]), 4),
            "macd_sig":  round(float(macd_sig.iloc[-1]), 4),
            "macd_hist": round(mh_last, 4),
            "macd_hist_prev": round(mh_prev, 4),
            "ema12": round(float(ema12.iloc[-1]), 2),
            "ema26": round(float(ema26.iloc[-1]), 2),
            "sma7":  round(sma7, 2),
            "sma20": round(sma20, 2),
            "sma50": round(sma50, 2) if sma50 else None,
            "sma200": round(sma200, 2) if sma200 else None,
            "pctB": round(pctB, 3),
            "vwap": round(vwap, 2),
            "pcr": pcr,
            "rsi_div": rsi_div,
            "high_52": round(high_52, 2), "low_52": round(low_52, 2),
        }
    except Exception:
        return None


def _tf_bias(df):
    """3-signal majority vote → BULL / BEAR / NEUT."""
    if df is None or len(df) < 5:
        return "N/A"
    c = df["Close"]
    votes = 0
    try:
        if len(c) >= 20:
            votes += 1 if float(c.iloc[-1]) > float(c.ewm(span=20, adjust=False).mean().iloc[-1]) else -1
        if len(c) >= 15:
            rsi_val = float(_calc_rsi(c).iloc[-1])
            if not np.isnan(rsi_val):
                votes += 1 if rsi_val > 50 else -1
        if len(c) >= 26:
            ml  = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
            mhv = float((ml - ml.ewm(span=9, adjust=False).mean()).iloc[-1])
            if not np.isnan(mhv):
                votes += 1 if mhv > 0 else -1
    except Exception:
        pass
    return "BULL" if votes >= 2 else "BEAR" if votes <= -2 else "NEUT"


@st.cache_data(ttl=300)
def fetch_tf_data(sym):
    """Fetch 9 timeframes and return bias dict for ATLAS."""
    specs = [
        ("3min",  "2m",   "1d"),
        ("5min",  "5m",   "5d"),
        ("15min", "15m",  "5d"),
        ("30min", "30m",  "1mo"),
        ("1hr",   "60m",  "3mo"),
        ("4hr",   "60m",  "6mo"),   # will resample
        ("1day",  "1d",   "1y"),
        ("1wk",   "1wk",  "5y"),
        ("1mo",   "1mo",  "5y"),
    ]
    out = {}
    for label, interval, period in specs:
        try:
            df = _fetch_df(sym, interval, period)
            if label == "4hr" and df is not None:
                df = (df.resample("4h")
                        .agg({"Open": "first", "High": "max",
                              "Low": "min", "Close": "last", "Volume": "sum"})
                        .dropna())
            out[label] = _tf_bias(df)
        except Exception:
            out[label] = "N/A"
    return out


def _build_prompt(sym, d, tf):
    """Build Claude prompt. d=market data dict or None. tf=timeframe bias dict or None."""
    ab = lambda px, ref: "above" if ref and px > ref else "below" if ref else "N/A"

    if d:
        sma50_txt  = "${:.2f}".format(d["sma50"])  if d["sma50"]  else "N/A"
        sma200_txt = "${:.2f}".format(d["sma200"]) if d["sma200"] else "N/A"
        pcr_txt    = str(d["pcr"]) if d["pcr"] is not None else "unavailable"
        pos52 = ((d["px"] - d["low_52"]) / (d["high_52"] - d["low_52"]) * 100) if d["high_52"] != d["low_52"] else 50
        data_block = (
            "LIVE MARKET DATA FOR {sym}:\n"
            "  Price: ${px} ({sign}{chg}%)  52wH/L: ${h52}/${l52}\n"
            "  Volume: {vol:,} ({vr:.1f}x avg)\n\n"
            "ARIA  — RSI-14: {rsi} ({rsi_lbl})  %B: {pctB:.3f}  Vol: {vr:.1f}x\n"
            "NEXUS — SMA7: ${sma7} ({rel7})  SMA20: ${sma20} ({rel20})  SMA50: {sma50} ({rel50})\n"
            "        SMA200: {sma200} ({rel200})  EMA12/26: ${ema12}/${ema26} ({ecross})\n"
            "        VWAP: ${vwap} ({relvwap})\n"
            "SIGMA — Put/Call: {pcr}  52w pos: {pos:.0f}%\n"
            "DELTA — MACD line: {ml}  Signal: {ms}  Hist: {mh} ({mdir})  RSI div: {rdiv}\n"
        ).format(
            sym=sym, px=d["px"], sign="+" if d["chg"] >= 0 else "", chg=d["chg"],
            h52=d["high_52"], l52=d["low_52"], vol=d["volume"], vr=d["vol_ratio"],
            rsi=d["rsi"], rsi_lbl="oversold" if d["rsi"] < 30 else "overbought" if d["rsi"] > 70 else "neutral",
            pctB=d["pctB"],
            sma7=d["sma7"], rel7=ab(d["px"], d["sma7"]),
            sma20=d["sma20"], rel20=ab(d["px"], d["sma20"]),
            sma50=sma50_txt, rel50=ab(d["px"], d["sma50"]),
            sma200=sma200_txt, rel200=ab(d["px"], d["sma200"]),
            ema12=d["ema12"], ema26=d["ema26"],
            ecross="bullish" if d["ema12"] > d["ema26"] else "bearish",
            vwap=d["vwap"], relvwap=ab(d["px"], d["vwap"]),
            pcr=pcr_txt, pos=pos52,
            ml=d["macd_line"], ms=d["macd_sig"], mh=d["macd_hist"],
            mdir="expanding" if abs(d["macd_hist"]) > abs(d["macd_hist_prev"]) else "contracting",
            rdiv=d["rsi_div"],
        )
    else:
        data_block = "NOTE: Live data unavailable. Use your knowledge of {} to estimate all analysis.\n".format(sym)

    tf_block = "ATLAS — MULTI-TIMEFRAME BIAS:\n"
    if tf:
        tf_block += "  " + "  ".join("{}: {}".format(k, v) for k, v in tf.items()) + "\n"
    else:
        tf_block += "  Live TF data unavailable — estimate based on your knowledge.\n"

    return (
        "Analyze {sym}. Use the data below. "
        "Reply with ONLY this JSON (no markdown, no backticks):\n"
        '{{"ARIA":{{"verdict":"CALL","confidence":72,'
        '"analysis":"Two specific sentences about {sym} momentum/RSI/volume.",'
        '"rsi":58,"vol":112,"mom":65}},'
        '"NEXUS":{{"verdict":"CALL","confidence":74,'
        '"analysis":"Two specific sentences about {sym} SMA7/20/50/200 and trend.",'
        '"sma7":62,"sma":68,"ema":72,"trend":70}},'
        '"SIGMA":{{"verdict":"PUT","confidence":69,'
        '"analysis":"Two specific sentences about {sym} options/sentiment.",'
        '"pcr":62,"ivr":45,"flow":55}},'
        '"DELTA":{{"verdict":"PUT","confidence":76,"rev":"NONE",'
        '"analysis":"Two specific sentences about {sym} MACD/RSI divergence.",'
        '"macd":65,"rsid":58,"cvd":60}},'
        '"ATLAS":{{"verdict":"CALL","confidence":78,'
        '"analysis":"Two sentences summarizing {sym} multi-timeframe alignment.",'
        '"short_tf":65,"mid_tf":70,"long_tf":80,"bull_count":6}}}}\n'
        "Rules: verdicts may differ. DELTA rev = BULLISH_REVERSAL, BEARISH_REVERSAL, or NONE. "
        "ATLAS bull_count = number of timeframes showing BULL (0-9). "
        "Confidence 55-94. Metric values 0-100 (vol up to 150). Cite real numbers where available.\n\n"
        "{data}\n{tf}"
    ).format(sym=sym, data=data_block, tf=tf_block)


def _extract_json(text):
    try:
        s = text.strip()
        a, b = s.find('{'), s.rfind('}')
        if a == -1 or b == -1:
            return None
        return json.loads(s[a:b + 1])
    except Exception:
        return None


def _parse_bot(bot_id, data):
    d = data.get(bot_id, {})
    verdict    = "CALL" if "CALL" in str(d.get("verdict", "")).upper() else "PUT"
    confidence = min(94, max(55, int(d.get("confidence", 68))))
    analysis   = str(d.get("analysis", "Analysis complete.")).strip()
    reversal   = None
    if bot_id == "DELTA":
        rev = str(d.get("rev", "NONE"))
        reversal = rev if rev in ("BULLISH_REVERSAL", "BEARISH_REVERSAL") else "NONE"
    metrics_map = {
        "ARIA":  [("RSI-14",       d.get("rsi",  55), 100),
                  ("Volume %",     d.get("vol",  55), 150),
                  ("Momentum",     d.get("mom",  55), 100)],
        "NEXUS": [("SMA 7/20 Score",d.get("sma7", 55), 100),
                  ("EMA Strength", d.get("ema",  55), 100),
                  ("Trend Align",  d.get("trend",55), 100)],
        "SIGMA": [("Put/Call",     d.get("pcr",  55), 100),
                  ("IV Rank",      d.get("ivr",  55), 100),
                  ("Bullish Flow", d.get("flow", 55), 100)],
        "DELTA": [("MACD Div",     d.get("macd", 55), 100),
                  ("RSI Div",      d.get("rsid", 55), 100),
                  ("CVD Imbalance",d.get("cvd",  55), 100)],
        "ATLAS": [("Short TF",     d.get("short_tf", 55), 100),
                  ("Mid TF",       d.get("mid_tf",   55), 100),
                  ("Long TF",      d.get("long_tf",  55), 100)],
    }
    metrics = [{"label": lbl,
                "value": min(mx, max(0, float(v or 55))),
                "max": mx}
               for lbl, v, mx in metrics_map.get(bot_id, [])]
    return {"verdict": verdict, "confidence": confidence,
            "analysis": analysis, "metrics": metrics, "reversal": reversal}


def run_swarm(sym, api_key, market_data, tf_data):
    client = anthropic.Anthropic(api_key=api_key)
    prompt = _build_prompt(sym, market_data, tf_data)
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1400,
        system="You are a stock analysis API. Output ONLY a raw JSON object. No markdown. No explanation. Just JSON.",
        messages=[{"role": "user", "content": prompt}],
    )
    raw  = "".join(c.text for c in msg.content if hasattr(c, "text")).strip()
    data = _extract_json(raw)
    if data is None:
        raise ValueError("Could not parse JSON: " + raw[:200])
    return [_parse_bot(b["id"], data) for b in BOTS], raw


# ── Session state ──────────────────────────────────────────────────────────────
if "ant_key" not in st.session_state:
    st.session_state["ant_key"] = os.getenv("ANTHROPIC_API_KEY", "")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-size:18px;font-weight:900;letter-spacing:6px;
      background:linear-gradient(90deg,#00ff88,#00cfff,#c084fc,#ff6b35);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;
      margin-bottom:4px">TRADESWARM</div>
    <div style="color:#334;font-size:9px;letter-spacing:2px;margin-bottom:20px">
      5-BOT PUT/CALL ANALYZER
    </div>""", unsafe_allow_html=True)

    st.markdown('<div style="color:#334;font-size:9px;letter-spacing:1px;margin-bottom:4px">ANTHROPIC KEY</div>', unsafe_allow_html=True)
    new_key = st.text_input("API Key", type="password", placeholder="sk-ant-api03-...",
                            label_visibility="collapsed", key="key_input")
    c1, c2 = st.columns(2)
    if c1.button("SAVE", use_container_width=True, key="save_btn"):
        k = new_key.strip()
        if k.startswith("sk-ant-"):
            st.session_state["ant_key"] = k
            set_key(_ENV_PATH, "ANTHROPIC_API_KEY", k)
            st.success("Saved to .env ✓")
        else:
            st.error("Must start with sk-ant-")
    if c2.button("CLEAR", use_container_width=True, key="clear_btn"):
        st.session_state["ant_key"] = ""
        set_key(_ENV_PATH, "ANTHROPIC_API_KEY", "")
        st.rerun()

    if st.session_state["ant_key"]:
        st.markdown('<div style="color:#00ff8877;font-size:9px;margin-top:4px">● Key loaded</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#ff446677;font-size:9px;margin-top:4px">○ No key set</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="color:#334;font-size:9px;letter-spacing:1px;margin-bottom:4px">TICKER SYMBOL</div>', unsafe_allow_html=True)
    ticker = st.text_input("Ticker", placeholder="AAPL  TSLA  SPY...",
                           label_visibility="collapsed", key="ticker_inp").strip().upper()

    run_btn = st.button("▶ ANALYZE", use_container_width=True,
                        disabled=not (st.session_state["ant_key"] and ticker),
                        key="run_btn")
    st.markdown("---")
    st.markdown('<div style="color:#1e1e30;font-size:9px">⚠ AI only — not financial advice</div>', unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:22px 0 10px">
  <div style="color:#181828;font-size:9px;letter-spacing:4px;margin-bottom:4px">
    ▸ AUTONOMOUS TRADING INTELLIGENCE SYSTEM ◂
  </div>
  <div style="font-size:30px;font-weight:900;letter-spacing:10px;
    background:linear-gradient(90deg,#00ff88,#00cfff,#c084fc,#ff6b35);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:4px">
    TRADESWARM
  </div>
  <div style="color:#1e1e30;font-size:9px;letter-spacing:3px">
    5-BOT PUT/CALL · LIVE DATA · MULTI-TIMEFRAME · REVERSAL DETECTION
  </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state["ant_key"]:
    st.markdown("""
    <div style="text-align:center;padding:40px 20px;color:#334">
      <div style="font-size:32px;margin-bottom:12px">🔑</div>
      <div style="font-size:13px;letter-spacing:1px">Paste your Anthropic API key in the sidebar to get started.</div>
      <div style="font-size:10px;margin-top:8px;color:#222">Get a free key at console.anthropic.com</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

if not ticker:
    st.markdown("""
    <div style="text-align:center;padding:40px 20px;color:#334">
      <div style="font-size:32px;margin-bottom:12px">◈</div>
      <div style="font-size:13px;letter-spacing:1px">Enter a ticker symbol in the sidebar, then click ANALYZE.</div>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ── Live data ──────────────────────────────────────────────────────────────────
md = fetch_market_data(ticker)
tf = fetch_tf_data(ticker)
if md:
    up = md["chg"] >= 0
    cc = "#00ff88" if up else "#ff4466"
    pos52 = ((md["px"] - md["low_52"]) / (md["high_52"] - md["low_52"]) * 100) if md["high_52"] != md["low_52"] else 50
    st.markdown("""
    <div style="background:linear-gradient(135deg,#08081a,#0d0d22);
      border:1px solid {cc}30;border-radius:10px;padding:12px 18px;
      display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:14px">
      <div>
        <div style="color:#333;font-size:8px;letter-spacing:2px;margin-bottom:1px">LIVE PRICE</div>
        <div style="color:#e0e0e0;font-size:26px;font-weight:900;font-family:'Courier New',monospace">
          ${px}
        </div>
      </div>
      <div style="padding:4px 14px;background:{cc}14;border:1px solid {cc}44;border-radius:8px">
        <span style="color:{cc};font-weight:800;font-size:15px">{arrow} {pct}%</span>
      </div>
      <div style="display:flex;gap:18px">
        <div><div style="color:#333;font-size:8px">HIGH</div>
          <div style="color:#556;font-size:12px;font-family:'Courier New',monospace">${high}</div></div>
        <div><div style="color:#333;font-size:8px">LOW</div>
          <div style="color:#556;font-size:12px;font-family:'Courier New',monospace">${low}</div></div>
        <div><div style="color:#333;font-size:8px">VOLUME</div>
          <div style="color:#556;font-size:12px;font-family:'Courier New',monospace">{vol}M ({vr:.1f}x)</div></div>
        <div><div style="color:#333;font-size:8px">RSI-14</div>
          <div style="color:{rsi_c};font-size:12px;font-family:'Courier New',monospace">{rsi}</div></div>
        <div><div style="color:#333;font-size:8px">52W POS</div>
          <div style="color:#556;font-size:12px;font-family:'Courier New',monospace">{pos:.0f}%</div></div>
      </div>
      <div style="margin-left:auto;display:flex;align-items:center;gap:5px">
        <span style="width:6px;height:6px;border-radius:50%;background:#00ff88;display:inline-block"></span>
        <span style="color:#334;font-size:9px">LIVE</span>
      </div>
    </div>""".format(
        cc=cc, px=md["px"],
        arrow="▲" if up else "▼", pct=abs(md["chg"]),
        high=md["high"], low=md["low"],
        vol=round(md["volume"] / 1e6, 1), vr=md["vol_ratio"],
        rsi=md["rsi"],
        rsi_c="#ff4466" if md["rsi"] > 70 else "#00ff88" if md["rsi"] < 30 else "#667",
        pos=pos52,
    ), unsafe_allow_html=True)
else:
    st.info("Live price data unavailable for {} — analysis will run using AI estimation.".format(ticker))

# ── TradingView chart ──────────────────────────────────────────────────────────
tv_src = (
    "https://s.tradingview.com/widgetembed/?frameElementId=tv"
    "&symbol={sym}&interval=D&hidesidetoolbar=1&theme=dark&style=1"
    "&timezone=America%2FNew_York"
    "&studies=RSI%40tv-basicstudies%1FMACD%40tv-basicstudies&locale=en"
).format(sym=ticker)

st.markdown("""
<div style="background:#050510;border:1px solid #1c1c30;border-radius:12px;
  overflow:hidden;margin-bottom:14px">
  <div style="padding:8px 14px;border-bottom:1px solid #1c1c30;
    display:flex;align-items:center">
    <span style="color:#00cfff;font-size:11px;font-weight:800;letter-spacing:1px">
      📈 {sym} — TRADINGVIEW
    </span>
    <span style="color:#334;font-size:9px;margin-left:auto">DAILY · RSI · MACD</span>
  </div>
  <iframe src="{src}" width="100%" height="420" frameborder="0"
    allowfullscreen scrolling="no" title="chart-{sym}"></iframe>
</div>""".format(sym=ticker, src=tv_src), unsafe_allow_html=True)

# ── Run analysis ───────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Deploying 5-bot swarm on {}...".format(ticker)):
        try:
            results, raw = run_swarm(ticker, st.session_state["ant_key"], md, tf)
            st.session_state["swarm_results"] = results
            st.session_state["swarm_ticker"]  = ticker
            st.session_state["swarm_raw"]     = raw
            st.session_state["swarm_tf"]      = tf
            st.session_state.pop("swarm_error", None)
        except Exception as e:
            st.session_state["swarm_error"]   = str(e)
            st.session_state["swarm_ticker"]  = ticker
            st.session_state.pop("swarm_results", None)

# ── Display results ────────────────────────────────────────────────────────────
if st.session_state.get("swarm_ticker") != ticker:
    st.stop()

if st.session_state.get("swarm_error"):
    st.error("Error: " + st.session_state["swarm_error"])
    st.stop()

results = st.session_state.get("swarm_results")
if not results:
    st.stop()

def _bot_card_html(bot, r):
    bc = bot["color"]
    vc = "#00ff88" if r["verdict"] == "CALL" else "#ff4466"
    metrics_html = "".join(
        '<div style="margin-bottom:7px">'
        '<div style="display:flex;justify-content:space-between;font-size:10px;color:#556">'
        '<span>{lbl}</span><span style="color:{bc}">{val:.0f}</span></div>'
        '<div style="height:4px;background:#0f0f20;border-radius:3px;overflow:hidden;margin-top:2px">'
        '<div style="width:{pct:.0f}%;height:100%;background:linear-gradient(90deg,{bc}55,{bc});border-radius:3px"></div>'
        '</div></div>'.format(lbl=m["label"], val=m["value"], bc=bc, pct=min(100, m["value"] / m["max"] * 100))
        for m in r["metrics"]
    )
    rev_html = ""
    if r.get("reversal") and r["reversal"] != "NONE":
        rc = "#00ff88" if r["reversal"] == "BULLISH_REVERSAL" else "#ff4466"
        rl = "⬆ BULLISH REVERSAL" if r["reversal"] == "BULLISH_REVERSAL" else "⬇ BEARISH REVERSAL"
        rev_html = '<div style="margin-top:8px;padding:4px 9px;background:{c}10;border:1px solid {c}40;border-radius:6px;color:{c};font-size:9px;font-weight:800">{l}</div>'.format(c=rc, l=rl)
    tags_html = "".join(
        '<span style="padding:2px 5px;border-radius:3px;font-size:8px;margin-right:3px;background:{bc}0c;border:1px solid {bc}18;color:{bc}55">{t}</span>'.format(bc=bc, t=t)
        for t in bot["tags"]
    )
    return """
    <div style="background:linear-gradient(150deg,#0b0b1c,#0f0f26);border:1px solid {bc}40;border-radius:12px;padding:16px;margin-bottom:12px">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
        <div style="width:38px;height:38px;border-radius:9px;flex-shrink:0;background:{bc}18;border:1px solid {bc}30;display:flex;align-items:center;justify-content:center;font-size:18px;color:{bc}">{icon}</div>
        <div style="flex:1">
          <div style="color:{bc};font-weight:800;font-size:13px;letter-spacing:2px">{id}</div>
          <div style="color:{bc}66;font-size:8px">{role}</div>
        </div>
        <div style="padding:2px 10px;border-radius:20px;background:{vc}12;border:1px solid {vc}44;color:{vc};font-weight:800;font-size:11px;letter-spacing:2px">{verdict}</div>
      </div>
      <div style="margin-bottom:8px">{tags}</div>
      <div style="background:#05050f;border-radius:7px;padding:11px;min-height:54px;border:1px solid #141428;font-size:11px;line-height:1.8;color:#8899b0;margin-bottom:10px">{analysis}</div>
      {metrics}{rev}
      <div style="display:flex;justify-content:space-between;margin-top:8px">
        <span style="color:#222;font-size:9px">CONFIDENCE</span>
        <span style="color:{bc};font-size:11px;font-weight:800">{conf}%</span>
      </div>
    </div>""".format(bc=bc, icon=bot["icon"], id=bot["id"], role=bot["role"],
                     vc=vc, verdict=r["verdict"], tags=tags_html,
                     analysis=r["analysis"], metrics=metrics_html, rev=rev_html, conf=r["confidence"])

# Bot cards — first 4 in 2-column grid
col_l, col_r = st.columns(2)
for i in range(4):
    col = col_l if i % 2 == 0 else col_r
    col.markdown(_bot_card_html(BOTS[i], results[i]), unsafe_allow_html=True)

# ATLAS card — full width with timeframe grid
if len(results) >= 5:
    atlas_r  = results[4]
    atlas_bc = "#ffd700"
    atlas_vc = "#00ff88" if atlas_r["verdict"] == "CALL" else "#ff4466"
    saved_tf = st.session_state.get("swarm_tf") or {}
    tf_order = ["3min","5min","15min","30min","1hr","4hr","1day","1wk","1mo"]
    tf_labels = {"3min":"3MIN","5min":"5MIN","15min":"15M","30min":"30M","1hr":"1HR","4hr":"4HR","1day":"1DAY","1wk":"1WK","1mo":"1MO"}
    def tf_color(b):
        return "#00ff88" if b == "BULL" else "#ff4466" if b == "BEAR" else "#667" if b == "NEUT" else "#333"
    tf_cells = "".join(
        '<div style="padding:5px 4px;border-radius:5px;background:{bc}14;border:1px solid {bc}30;text-align:center">'
        '<div style="color:#556;font-size:8px">{lbl}</div>'
        '<div style="color:{bc};font-size:10px;font-weight:800">{v}</div>'
        '</div>'.format(lbl=tf_labels.get(k, k), v=saved_tf.get(k, "N/A"), bc=tf_color(saved_tf.get(k, "N/A")))
        for k in tf_order
    )
    atlas_metrics_html = "".join(
        '<div style="margin-bottom:7px">'
        '<div style="display:flex;justify-content:space-between;font-size:10px;color:#556">'
        '<span>{lbl}</span><span style="color:{bc}">{val:.0f}</span></div>'
        '<div style="height:4px;background:#0f0f20;border-radius:3px;overflow:hidden;margin-top:2px">'
        '<div style="width:{pct:.0f}%;height:100%;background:linear-gradient(90deg,{bc}55,{bc});border-radius:3px"></div>'
        '</div></div>'.format(lbl=m["label"], val=m["value"], bc=atlas_bc, pct=min(100, m["value"] / m["max"] * 100))
        for m in atlas_r["metrics"]
    )
    atlas_tags = "".join(
        '<span style="padding:2px 5px;border-radius:3px;font-size:8px;margin-right:3px;background:{bc}0c;border:1px solid {bc}18;color:{bc}55">{t}</span>'.format(bc=atlas_bc, t=t)
        for t in BOTS[4]["tags"]
    )
    st.markdown("""
    <div style="background:linear-gradient(150deg,#0b0b1c,#0f0f26);border:1px solid {bc}40;border-radius:12px;padding:16px;margin-bottom:12px">
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
        <div style="width:38px;height:38px;border-radius:9px;flex-shrink:0;background:{bc}18;border:1px solid {bc}30;display:flex;align-items:center;justify-content:center;font-size:18px;color:{bc}">◎</div>
        <div style="flex:1">
          <div style="color:{bc};font-weight:800;font-size:13px;letter-spacing:2px">ATLAS</div>
          <div style="color:{bc}66;font-size:8px">MULTI-TIMEFRAME</div>
        </div>
        <div style="padding:2px 10px;border-radius:20px;background:{vc}12;border:1px solid {vc}44;color:{vc};font-weight:800;font-size:11px;letter-spacing:2px">{verdict}</div>
      </div>
      <div style="margin-bottom:8px">{tags}</div>
      <div style="display:grid;grid-template-columns:repeat(9,1fr);gap:5px;margin-bottom:10px">{tf_cells}</div>
      <div style="background:#05050f;border-radius:7px;padding:11px;border:1px solid #141428;font-size:11px;line-height:1.8;color:#8899b0;margin-bottom:10px">{analysis}</div>
      {metrics}
      <div style="display:flex;justify-content:space-between;margin-top:8px">
        <span style="color:#222;font-size:9px">CONFIDENCE</span>
        <span style="color:{bc};font-size:11px;font-weight:800">{conf}%</span>
      </div>
    </div>""".format(
        bc=atlas_bc, vc=atlas_vc, verdict=atlas_r["verdict"],
        tags=atlas_tags, tf_cells=tf_cells,
        analysis=atlas_r["analysis"], metrics=atlas_metrics_html, conf=atlas_r["confidence"]
    ), unsafe_allow_html=True)

# Verdict panel
calls    = sum(1 for r in results if r["verdict"] == "CALL")
puts     = len(results) - calls
final    = "CALL" if calls >= puts else "PUT"
avg_c    = round(sum(r["confidence"] for r in results) / len(results))
strength = "STRONG" if avg_c >= 80 else "MODERATE" if avg_c >= 66 else "WEAK"
fc       = "#00ff88" if final == "CALL" else "#ff4466"

delta_rev = results[3].get("reversal") if len(results) > 3 else None
rev_badge = ""
if delta_rev and delta_rev != "NONE":
    rc2 = "#00ff88" if delta_rev == "BULLISH_REVERSAL" else "#ff4466"
    rl2 = "⚡ BULLISH REVERSAL" if delta_rev == "BULLISH_REVERSAL" else "⚡ BEARISH REVERSAL"
    rev_badge = ('<div style="margin:0 auto 14px;padding:7px 18px;display:inline-block;'
                 'background:{c}10;border:1px solid {c}44;border-radius:9px">'
                 '<span style="color:{c};font-size:13px;font-weight:800;letter-spacing:2px">'
                 '{l}</span></div>').format(c=rc2, l=rl2)

votes_html = "".join(
    '<div style="display:flex;flex-direction:column;align-items:center;gap:3px">'
    '<span style="color:{bc};font-size:9px">{id}</span>'
    '<span style="padding:2px 9px;border-radius:12px;background:{vc}14;'
    'border:1px solid {vc}44;color:{vc};font-size:11px;font-weight:800">{v}</span>'
    '</div>'.format(
        bc=b["color"], id=b["id"],
        vc="#00ff88" if r["verdict"] == "CALL" else "#ff4466",
        v=r["verdict"])
    for b, r in zip(BOTS, results)
)

st.markdown("""
<div style="background:linear-gradient(145deg,#07071a,#0c0c22);
  border:2px solid {fc};border-radius:14px;padding:26px;text-align:center;
  box-shadow:0 0 60px {fc}18;position:relative;overflow:hidden;margin-top:4px">
  <div style="position:absolute;inset:0;pointer-events:none;
    background:radial-gradient(ellipse at 50% 0%,{fc}08,transparent 65%)"></div>
  <div style="color:#333;font-size:10px;letter-spacing:3px;margin-bottom:5px">▸ 5-BOT CONSENSUS ◂</div>
  <div style="color:#444;font-size:10px;margin-bottom:14px">
    {ticker} &nbsp;·&nbsp; {calls}/5 CALL &nbsp;·&nbsp; {puts}/5 PUT
  </div>
  <div style="display:flex;justify-content:center;gap:12px;margin-bottom:16px;flex-wrap:wrap">
    {votes}
  </div>
  <div style="font-size:58px;font-weight:900;color:{fc};letter-spacing:12px;
    line-height:1;text-shadow:0 0 40px {fc}66;margin-bottom:6px">{final}</div>
  <div style="color:{fc}77;font-size:12px;letter-spacing:4px;margin-bottom:14px">
    {strength} SIGNAL &nbsp;·&nbsp; {avgc}% CONFIDENCE
  </div>
  {rev_badge}
  <div style="display:inline-block;padding:5px 16px;background:#ffffff06;
    border:1px solid #181830;border-radius:7px;color:#334;font-size:10px;margin-top:6px">
    ⚠ AI analysis only — not financial advice
  </div>
</div>""".format(
    fc=fc, ticker=ticker, calls=calls, puts=puts,
    votes=votes_html, final=final, strength=strength,
    avgc=avg_c, rev_badge=rev_badge,
), unsafe_allow_html=True)

with st.expander("🔍 Raw API response"):
    st.code(st.session_state.get("swarm_raw", ""), language="json")
