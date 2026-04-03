import streamlit as st
import anthropic
import requests
import pandas as pd
import numpy as np
import json
import os
import math
import datetime
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
    """Fetch OHLCV from Yahoo Finance v8 API via requests."""
    for host in ("query2.finance.yahoo.com", "query1.finance.yahoo.com"):
        try:
            url = "https://{}/v8/finance/chart/{}".format(host, sym)
            r = requests.get(url, params={"interval": interval, "range": period},
                             headers=_YF_HEADERS, timeout=12)
            if not r.ok:
                continue
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
            continue
    return None


def _calc_rsi(c, p=14):
    d = c.diff()
    g = d.clip(lower=0).ewm(com=p - 1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=p - 1, adjust=False).mean()
    return 100 - 100 / (1 + g / l.replace(0, np.nan))


# ── Black-Scholes Greeks ───────────────────────────────────────────────────────
def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _npdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def _bs_greeks(S, K, T, r, sigma, is_call):
    """Compute Black-Scholes Delta/Gamma/Theta/Vega for one contract."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None
    try:
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        delta = _ncdf(d1) if is_call else (_ncdf(d1) - 1.0)
        gamma = _npdf(d1) / (S * sigma * math.sqrt(T))
        vega  = S * _npdf(d1) * math.sqrt(T) / 100.0        # per 1% IV move
        if is_call:
            theta = (-(S * _npdf(d1) * sigma) / (2 * math.sqrt(T))
                     - r * K * math.exp(-r * T) * _ncdf(d2)) / 365.0
        else:
            theta = (-(S * _npdf(d1) * sigma) / (2 * math.sqrt(T))
                     + r * K * math.exp(-r * T) * _ncdf(-d2)) / 365.0
        return {"delta": round(delta, 3), "gamma": round(gamma, 5),
                "theta": round(theta, 4), "vega": round(vega, 4)}
    except Exception:
        return None


@st.cache_data(ttl=300)
def fetch_options_greeks(sym):
    """Fetch nearest-expiry ATM options, compute Greeks, return summary dict."""
    for host in ("query2.finance.yahoo.com", "query1.finance.yahoo.com"):
        try:
            url = "https://{}/v7/finance/options/{}".format(host, sym)
            r = requests.get(url, headers=_YF_HEADERS, timeout=12)
            if not r.ok:
                continue
            result = r.json().get("optionChain", {}).get("result", [])
            if not result:
                continue
            res   = result[0]
            quote = res.get("quote", {})
            spot  = float(quote.get("regularMarketPrice", 0) or 0)
            if spot <= 0:
                break

            options = res.get("options", [])
            if not options:
                break
            chain = options[0]

            exp_dates = res.get("expirationDates", [])
            if exp_dates:
                exp_dt = datetime.datetime.fromtimestamp(exp_dates[0])
                T = max(1 / 365, (exp_dt - datetime.datetime.now()).days / 365.0)
                days_out = (exp_dt - datetime.datetime.now()).days
            else:
                T, days_out = 30 / 365.0, 30

            calls = chain.get("calls", [])
            puts  = chain.get("puts",  [])
            if not calls or not puts:
                break

            atm_call = min(calls, key=lambda c: abs(float(c.get("strike", 0) or 0) - spot))
            atm_put  = min(puts,  key=lambda p: abs(float(p.get("strike", 0) or 0) - spot))
            K        = float(atm_call.get("strike", spot) or spot)
            call_iv  = float(atm_call.get("impliedVolatility", 0) or 0)
            put_iv   = float(atm_put.get("impliedVolatility",  0) or 0)
            avg_iv   = (call_iv + put_iv) / 2 if call_iv and put_iv else (call_iv or put_iv)

            r_rf = 0.05
            cg = _bs_greeks(spot, K, T, r_rf, call_iv, True)  if call_iv > 0 else None
            pg = _bs_greeks(spot, K, T, r_rf, put_iv,  False) if put_iv  > 0 else None

            call_oi = sum(int(c.get("openInterest", 0) or 0) for c in calls)
            put_oi  = sum(int(p.get("openInterest", 0) or 0) for p in puts)
            pcr = round(put_oi / call_oi, 2) if call_oi > 0 else None

            return {
                "spot":       round(spot, 2),
                "atm_strike": round(K, 2),
                "days_out":   days_out,
                "call_iv":    round(call_iv * 100, 1),
                "put_iv":     round(put_iv  * 100, 1),
                "avg_iv":     round(avg_iv  * 100, 1),
                "call_greeks": cg,
                "put_greeks":  pg,
                "call_bid":   round(float(atm_call.get("bid", 0) or 0), 2),
                "call_ask":   round(float(atm_call.get("ask", 0) or 0), 2),
                "put_bid":    round(float(atm_put.get("bid",  0) or 0), 2),
                "put_ask":    round(float(atm_put.get("ask",  0) or 0), 2),
                "call_oi":    call_oi,
                "put_oi":     put_oi,
                "pcr":        pcr,
            }
        except Exception:
            continue
    return None


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

        # Options PCR via Yahoo Finance v7 options API
        pcr = None
        try:
            r = requests.get(
                "https://query2.finance.yahoo.com/v7/finance/options/" + sym,
                headers=_YF_HEADERS, timeout=10)
            if r.ok:
                opts = r.json().get("optionChain", {}).get("result", [{}])[0]
                chain = opts.get("options", [{}])[0]
                call_oi = sum(c.get("openInterest", 0) for c in chain.get("calls", []))
                put_oi  = sum(p.get("openInterest", 0) for p in chain.get("puts",  []))
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


# ── Deterministic bot logic ───────────────────────────────────────────────────
def _tech_score(d, tf):
    """Return integer -5 (strongly bearish) to +5 (strongly bullish)."""
    if not d:
        return 0
    s = 0
    rsi = d["rsi"]
    if rsi > 70:   s -= 2
    elif rsi > 60: s -= 1
    elif rsi < 30: s += 2
    elif rsi < 40: s += 1

    px = d["px"]
    if px < d["sma7"] and px < d["sma20"]:  s -= 2
    elif px > d["sma7"] and px > d["sma20"]: s += 2
    if d.get("sma50")  and px < d["sma50"]:  s -= 1
    if d.get("sma200") and px < d["sma200"]: s -= 1
    if d.get("sma50")  and px > d["sma50"]:  s += 1
    if d.get("sma200") and px > d["sma200"]: s += 1

    mh, mhp = d["macd_hist"], d["macd_hist_prev"]
    if mh < 0 and abs(mh) > abs(mhp): s -= 2   # neg expanding
    elif mh > 0 and mh > mhp:         s += 2   # pos expanding
    elif mh < 0:                       s -= 1
    else:                              s += 1

    if d["ema12"] < d["ema26"]: s -= 1
    else:                        s += 1

    pos52 = ((px - d["low_52"]) / (d["high_52"] - d["low_52"]) * 100) if d["high_52"] != d["low_52"] else 50
    if pos52 > 80: s -= 1
    elif pos52 < 20: s += 1

    if tf:
        bull_tf = sum(1 for v in tf.values() if v == "BULL")
        bear_tf = sum(1 for v in tf.values() if v == "BEAR")
        if bear_tf >= 6:   s -= 2
        elif bull_tf >= 6: s += 2
        elif bear_tf > bull_tf: s -= 1
        elif bull_tf > bear_tf: s += 1

    return max(-5, min(5, s))


def _compute_bot_verdicts(d, tf):
    """Compute each bot's CALL/PUT strictly from indicator thresholds."""
    if not d:
        return {b["id"]: "CALL" for b in BOTS}

    px = d["px"]
    pos52 = ((px - d["low_52"]) / (d["high_52"] - d["low_52"]) * 100) if d["high_52"] != d["low_52"] else 50

    # ARIA — RSI + Bollinger + Volume
    if d["rsi"] < 35 or d["pctB"] < 0.15:
        aria = "CALL"
    elif d["rsi"] > 65 or d["pctB"] > 0.85:
        aria = "PUT"
    elif d["vol_ratio"] > 1.5:
        aria = "CALL" if d["chg"] > 0 else "PUT"
    else:
        aria = "CALL" if d["rsi"] > 52 else "PUT"

    # NEXUS — SMA/EMA/VWAP alignment
    sma50_ok  = d.get("sma50")  is not None
    sma200_ok = d.get("sma200") is not None
    bull_sig = sum([
        px > d["sma7"],
        px > d["sma20"],
        (not sma50_ok)  or px > d["sma50"],
        (not sma200_ok) or px > d["sma200"],
        d["ema12"] > d["ema26"],
        px > d["vwap"],
    ])
    nexus = "CALL" if bull_sig >= 4 else "PUT"

    # SIGMA — PCR + 52w position
    sig = 0
    pcr = d.get("pcr")
    if pcr is not None:
        if pcr > 1.2:   sig -= 2
        elif pcr > 0.9: sig -= 1
        elif pcr < 0.7: sig += 2
        else:           sig += 1
    if pos52 < 20:   sig += 1
    elif pos52 > 80: sig -= 1
    sigma = "PUT" if sig < 0 else "CALL"

    # DELTA — MACD histogram direction + RSI divergence
    mh, mhp = d["macd_hist"], d["macd_hist_prev"]
    neg_exp = mh < 0 and abs(mh) > abs(mhp)
    pos_exp = mh > 0 and mh > mhp
    if neg_exp:
        delta = "PUT"
    elif pos_exp:
        delta = "CALL"
    elif d["rsi_div"] == "bullish":
        delta = "CALL"
    elif d["rsi_div"] == "bearish":
        delta = "PUT"
    else:
        delta = "PUT" if mh < 0 else "CALL"

    # ATLAS — multi-timeframe majority
    if tf:
        bull_tf = sum(1 for v in tf.values() if v == "BULL")
        bear_tf = sum(1 for v in tf.values() if v == "BEAR")
        atlas = "CALL" if bull_tf > bear_tf else "PUT"
    else:
        atlas = "CALL" if d["chg"] > 0 else "PUT"

    return {"ARIA": aria, "NEXUS": nexus, "SIGMA": sigma, "DELTA": delta, "ATLAS": atlas}


def _compute_bot_confidences(d, tf):
    """Signal-strength-based confidence per bot. No LLM involved."""
    confs = {}
    if d:
        # ARIA — RSI distance from 50 + %B extremity + volume confirmation
        rsi_dist = abs(d["rsi"] - 50)           # 0 (neutral) → 50 (extreme)
        pctb_ext = abs(d["pctB"] - 0.5) * 2     # 0 (mid) → 1 (edge)
        aria_c = 55 + int(rsi_dist * 0.65 + pctb_ext * 14)
        if d["vol_ratio"] > 1.5:
            aria_c = min(93, aria_c + 6)         # volume confirms → +6
        confs["ARIA"] = min(93, max(55, aria_c))

        # NEXUS — how many of 6 signals agree (3=weakest, 6=strongest)
        px = d["px"]
        n_bull = sum([
            px > d["sma7"],
            px > d["sma20"],
            (not d.get("sma50"))  or px > d["sma50"],
            (not d.get("sma200")) or px > d["sma200"],
            d["ema12"] > d["ema26"],
            px > d["vwap"],
        ])
        alignment = max(n_bull, 6 - n_bull)      # 3 → 6
        nexus_c = 55 + int((alignment - 3) / 3 * 38)
        confs["NEXUS"] = min(93, max(55, nexus_c))

        # SIGMA — PCR deviation from 1.0 (neutral) + 52w-position extremes
        pcr = d.get("pcr")
        sigma_c = 57 if pcr is None else (55 + min(36, int(abs(pcr - 1.0) * 34)))
        pos52 = ((px - d["low_52"]) / (d["high_52"] - d["low_52"]) * 100) if d["high_52"] != d["low_52"] else 50
        if pos52 < 15 or pos52 > 85:
            sigma_c = min(93, sigma_c + 8)       # extreme 52w position → more confident
        confs["SIGMA"] = min(93, max(55, sigma_c))

        # DELTA — MACD expansion magnitude + divergence confirmation
        mh, mhp = d["macd_hist"], d["macd_hist_prev"]
        expanding = abs(mh) > abs(mhp)
        exp_ratio = abs(mh) / max(0.0001, abs(mhp)) if expanding else 0.5
        delta_c = 55 + min(36, int(exp_ratio * 14))
        if d["rsi_div"] in ("bullish", "bearish"):
            delta_c = min(93, delta_c + 8)       # divergence confirmed → stronger signal
        confs["DELTA"] = min(93, max(55, delta_c))
    else:
        for b in ["ARIA", "NEXUS", "SIGMA", "DELTA"]:
            confs[b] = 58

    # ATLAS — how many TFs agree with the majority
    if tf:
        bull_tf = sum(1 for v in tf.values() if v == "BULL")
        bear_tf = sum(1 for v in tf.values() if v == "BEAR")
        majority = max(bull_tf, bear_tf)
        atlas_c = 55 + int(majority / 9 * 38)
    else:
        atlas_c = 58
    confs["ATLAS"] = min(93, max(55, atlas_c))

    return confs


def _compute_agent_probs(tech_score, bot_calls, n_bots=5):
    """Compute bull/bear probability from objective signals only. Zero LLM influence."""
    tech_pct = (tech_score + 5) / 10      # maps -5→0.0, 0→0.5, +5→1.0
    bot_pct  = bot_calls / n_bots          # fraction of bots voting CALL
    # 55% weight to tech score, 45% to bot vote
    bull_raw = tech_pct * 0.55 + bot_pct * 0.45
    bull_prob = max(10, min(90, round(bull_raw * 100)))
    return bull_prob, 100 - bull_prob


def _compute_timeline(d, og, tech_score):
    """Determine hold duration from options data and signal strength — no LLM."""
    tl_map = {
        "scalp": ("SCALP", "0 – 1 DAY",     "#ff9500"),
        "short": ("SHORT", "1 – 5 DAYS",     "#ffd700"),
        "swing": ("SWING", "1 – 4 WEEKS",    "#00cfff"),
        "long":  ("LONG",  "1 – 3 MONTHS",   "#c084fc"),
    }
    key = "swing"
    if og:
        dte    = og["days_out"]
        avg_iv = og["avg_iv"]
        if dte <= 3:
            key = "scalp"
        elif dte <= 7:
            key = "short"
        elif avg_iv > 65:
            key = "short"   # expensive IV — exit quickly
        elif avg_iv < 22:
            key = "long"    # cheap IV — hold through the move
        elif abs(tech_score) >= 4:
            key = "short"   # very strong signal — momentum play
        else:
            key = "swing"
    else:
        if abs(tech_score) >= 4:
            key = "short"
        elif abs(tech_score) <= 1:
            key = "swing"
        else:
            key = "swing"
    return key, tl_map[key]


def _build_prompt(sym, d, tf, bot_verdicts):
    """Build LLM prompt — verdicts are PRE-COMPUTED; LLM writes analysis text only."""
    ab = lambda px, ref: "above" if ref and px > ref else "below" if ref else "N/A"

    if d:
        sma50_txt  = "${:.2f}".format(d["sma50"])  if d["sma50"]  else "N/A"
        sma200_txt = "${:.2f}".format(d["sma200"]) if d["sma200"] else "N/A"
        pcr_txt    = str(d["pcr"]) if d["pcr"] is not None else "unavailable"
        pos52 = ((d["px"] - d["low_52"]) / (d["high_52"] - d["low_52"]) * 100) if d["high_52"] != d["low_52"] else 50
        data_block = (
            "LIVE MARKET DATA FOR {sym}:\n"
            "  Price: ${px} ({sign}{chg}%)  52wH/L: ${h52}/${l52}  52w-pos: {pos:.0f}%\n"
            "  Volume: {vol:,} ({vr:.1f}x avg)\n\n"
            "ARIA  — RSI-14: {rsi}  %B: {pctB:.3f}  Vol: {vr:.1f}x\n"
            "NEXUS — SMA7: ${sma7} ({rel7})  SMA20: ${sma20} ({rel20})"
            "  SMA50: {sma50} ({rel50})  SMA200: {sma200} ({rel200})\n"
            "        EMA12: ${ema12}  EMA26: ${ema26} ({ecross})  VWAP: ${vwap} ({relvwap})\n"
            "SIGMA — Put/Call Ratio: {pcr}  52w pos: {pos:.0f}%\n"
            "DELTA — MACD hist: {mh} ({mdir})  RSI-div: {rdiv}\n"
        ).format(
            sym=sym, px=d["px"], sign="+" if d["chg"] >= 0 else "", chg=d["chg"],
            h52=d["high_52"], l52=d["low_52"], vol=d["volume"], vr=d["vol_ratio"],
            pos=pos52, rsi=d["rsi"], pctB=d["pctB"],
            sma7=d["sma7"],   rel7=ab(d["px"], d["sma7"]),
            sma20=d["sma20"], rel20=ab(d["px"], d["sma20"]),
            sma50=sma50_txt,  rel50=ab(d["px"], d["sma50"]),
            sma200=sma200_txt,rel200=ab(d["px"], d["sma200"]),
            ema12=d["ema12"], ema26=d["ema26"],
            ecross="bullish" if d["ema12"] > d["ema26"] else "bearish",
            vwap=d["vwap"],   relvwap=ab(d["px"], d["vwap"]),
            pcr=pcr_txt, mh=d["macd_hist"],
            mdir="expanding" if abs(d["macd_hist"]) > abs(d["macd_hist_prev"]) else "contracting",
            rdiv=d["rsi_div"],
        )
    else:
        data_block = "Live data unavailable. Use your knowledge of {} to estimate.\n".format(sym)

    tf_block = "ATLAS — MULTI-TIMEFRAME:\n  "
    if tf:
        bull_tf = sum(1 for v in tf.values() if v == "BULL")
        bear_tf = sum(1 for v in tf.values() if v == "BEAR")
        tf_block += "  ".join("{}: {}".format(k, v) for k, v in tf.items())
        tf_block += "  ({} BULL / {} BEAR)\n".format(bull_tf, bear_tf)
    else:
        tf_block += "Unavailable.\n"

    vd = bot_verdicts
    return (
        "Write analysis text for 5 trading bots analyzing {sym}.\n\n"
        "IMPORTANT: The verdicts below are FINAL and computed from hard indicator thresholds.\n"
        "DO NOT change any verdict. Your ONLY job is to write 2 sentences explaining WHY\n"
        "each bot reached its verdict, citing the specific numbers from the data below.\n\n"
        "FIXED VERDICTS:\n"
        "  ARIA={aria}  NEXUS={nexus}  SIGMA={sigma}  DELTA={delta}  ATLAS={atlas}\n\n"
        "{data}\n{tf}\n"
        "Reply with ONLY this JSON (no markdown):\n"
        '{{"ARIA":{{"verdict":"{aria}","confidence":72,'
        '"analysis":"2 sentences about {sym} RSI/Bollinger/volume explaining the {aria} verdict.",'
        '"rsi":55,"vol":80,"mom":60}},'
        '"NEXUS":{{"verdict":"{nexus}","confidence":70,'
        '"analysis":"2 sentences about {sym} SMA/EMA/VWAP alignment explaining the {nexus} verdict.",'
        '"sma7":55,"ema":60,"trend":58}},'
        '"SIGMA":{{"verdict":"{sigma}","confidence":68,'
        '"analysis":"2 sentences about {sym} options sentiment/PCR explaining the {sigma} verdict.",'
        '"pcr":55,"ivr":50,"flow":52}},'
        '"DELTA":{{"verdict":"{delta}","confidence":74,"rev":"NONE",'
        '"analysis":"2 sentences about {sym} MACD/divergence explaining the {delta} verdict.",'
        '"macd":60,"rsid":55,"cvd":58}},'
        '"ATLAS":{{"verdict":"{atlas}","confidence":71,'
        '"analysis":"2 sentences about {sym} multi-timeframe bias explaining the {atlas} verdict.",'
        '"short_tf":55,"mid_tf":60,"long_tf":58,"bull_count":{bull_count}}}}}'
    ).format(
        sym=sym, data=data_block, tf=tf_block,
        aria=vd["ARIA"], nexus=vd["NEXUS"], sigma=vd["SIGMA"],
        delta=vd["DELTA"], atlas=vd["ATLAS"],
        bull_count=(sum(1 for v in tf.values() if v == "BULL") if tf else 4),
    )


def _extract_json(text):
    try:
        s = text.strip()
        a, b = s.find('{'), s.rfind('}')
        if a == -1 or b == -1:
            return None
        return json.loads(s[a:b + 1])
    except Exception:
        return None


def _parse_bot(bot_id, data, forced_verdict=None, forced_confidence=None):
    d          = data.get(bot_id, {})
    verdict    = forced_verdict or ("CALL" if "CALL" in str(d.get("verdict", "")).upper() else "PUT")
    confidence = forced_confidence if forced_confidence is not None else min(93, max(55, int(d.get("confidence", 68))))
    analysis   = str(d.get("analysis", "Analysis complete.")).strip()
    reversal   = None
    if bot_id == "DELTA":
        rev = str(d.get("rev", "NONE"))
        reversal = rev if rev in ("BULLISH_REVERSAL", "BEARISH_REVERSAL") else "NONE"
    metrics_map = {
        "ARIA":  [("RSI-14",       d.get("rsi",  55), 100),
                  ("Volume %",     d.get("vol",  55), 150),
                  ("Momentum",     d.get("mom",  55), 100)],
        "NEXUS": [("SMA 7/20",     d.get("sma7", 55), 100),
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
    metrics = [{"label": lbl, "value": min(mx, max(0, float(v or 55))), "max": mx}
               for lbl, v, mx in metrics_map.get(bot_id, [])]
    return {"verdict": verdict, "confidence": confidence,
            "analysis": analysis, "metrics": metrics, "reversal": reversal}


def run_swarm(sym, api_key, provider, market_data, tf_data):
    # Step 1: compute verdicts from pure Python thresholds
    bot_verdicts = _compute_bot_verdicts(market_data, tf_data)
    prompt = _build_prompt(sym, market_data, tf_data, bot_verdicts)
    system = "You are a stock analysis API. Output ONLY a raw JSON object. No markdown. No explanation."

    if provider == "groq":
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "system", "content": system},
                             {"role": "user",   "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 1400,
                "response_format": {"type": "json_object"},
            },
            timeout=30,
        )
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()
    else:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1400,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(c.text for c in msg.content if hasattr(c, "text")).strip()

    data = _extract_json(raw)
    if data is None:
        data = {}
    # Step 2: enforce Python-computed verdicts AND confidence scores
    bot_confs = _compute_bot_confidences(market_data, tf_data)
    return [_parse_bot(b["id"], data,
                       forced_verdict=bot_verdicts.get(b["id"]),
                       forced_confidence=bot_confs.get(b["id"]))
            for b in BOTS], raw


# ── Agent definitions ─────────────────────────────────────────────────────────
_BULL_AGENT_ROLES = {
    "stock":     ["FUNDAMENTALS",    "MOMENTUM", "CATALYSTS",   "OPTIONS FLOW", "MACRO/SECTOR"],
    "commodity": ["SUPPLY/DEMAND",   "MOMENTUM", "MACRO DRIVER","OPTIONS FLOW", "MACRO TAILWIND"],
    "etf":       ["FUND COMPOSITION","MOMENTUM", "MACRO EVENT", "OPTIONS FLOW", "SECTOR ROTATION"],
    "index":     ["MARKET BREADTH",  "MOMENTUM", "MACRO CATALYST","INDEX FLOW",  "GLOBAL MACRO"],
}
_BEAR_AGENT_ROLES = {
    "stock":     ["VALUATION/RISK",  "TECH BREAKDOWN", "HEADWINDS",      "VOLATILITY",    "CONTRARIAN"],
    "commodity": ["OVERBOUGHT RISK", "TECH BREAKDOWN", "MACRO HEADWIND", "VOL/RISK",      "CONTRARIAN"],
    "etf":       ["OVEREXTENSION",   "TECH BREAKDOWN", "RISK-OFF",       "VOL/RISK",      "CONTRARIAN"],
    "index":     ["VALUATION RISK",  "TECH BREAKDOWN", "MACRO RISK",     "VOLATILITY",    "CONTRARIAN"],
}
BULL_AGENTS = [
    {"id": "ZEUS",     "color": "#ffd700", "icon": "⚡"},
    {"id": "HERMES",   "color": "#00ff88", "icon": "🜲"},
    {"id": "APOLLO",   "color": "#00cfff", "icon": "☀"},
    {"id": "ARES",     "color": "#ff9500", "icon": "⚔"},
    {"id": "POSEIDON", "color": "#40e0d0", "icon": "🜄"},
]
BEAR_AGENTS = [
    {"id": "KRONOS",  "color": "#ff4466", "icon": "⏳"},
    {"id": "HADES",   "color": "#c084fc", "icon": "☽"},
    {"id": "NEMESIS", "color": "#ff6b35", "icon": "⚖"},
    {"id": "TYPHON",  "color": "#ff2244", "icon": "☢"},
    {"id": "ERIS",    "color": "#dd44cc", "icon": "✦"},
]


# ── Asset type detection ───────────────────────────────────────────────────────
_COMMODITY_ETFS = {
    "GLD","IAU","SGOL","PHYS",                          # Gold
    "SLV","SIVR","PSLV",                                # Silver
    "USO","UCO","SCO","BNO",                            # Oil
    "UNG","BOIL","KOLD",                                # Nat gas
    "DBA","CORN","WEAT","SOYB",                         # Ag
    "PDBC","DJP","CPER",                                # Broad commodity
    "GDX","GDXJ","RING",                                # Miners
}
_BROAD_ETFS = {
    "SPY","VOO","IVV","VTI","QQQ","IWM","DIA","MDY",   # Broad market
    "TLT","IEF","SHY","BND","AGG","HYG","LQD","TIP",   # Fixed income
    "XLF","XLK","XLE","XLV","XLU","XLI","XLB","XLC","XLRE","XLP","XLY",  # Sectors
    "ARKK","ARKG","ARKF","ARKW","ARKO",                # Thematic
    "VXX","UVXY","SVXY",                                # Volatility
    "BITO","GBTC","IBIT","FBTC","ETHE","ETHU",         # Crypto ETFs
    "EEM","EFA","VEA","VWO","EWJ","FXI","MCHI",        # Int'l
}
_INDEX_SYMS = {"SPX","NDX","RUT","DJI","VIX","COMP","NYA"}

def _asset_type(sym):
    s = sym.upper().lstrip("^")
    if s in _INDEX_SYMS or sym.startswith("^"):
        return "index"
    if s in _COMMODITY_ETFS:
        return "commodity"
    if s in _BROAD_ETFS:
        return "etf"
    return "stock"


def _agent_roles(sym, atype):
    """Return per-agent focus instructions tailored to asset type."""
    _typhon = {
        "commodity": "volatility regime and risk events: VIX spikes, liquidity crunches, margin call cascades, or sudden ETF outflows that could crush this commodity",
        "etf":       "volatility risk: rising VIX, credit spread widening, ETF discount-to-NAV, and systemic risk factors that threaten this fund's holdings",
        "index":     "systemic volatility: VIX term structure, credit market stress, gamma-squeeze risks, and tail-risk events that could trigger a selloff",
        "stock":     "volatility and event risk: earnings miss probability, options skew, implied move vs realized, and binary event risk that could gap the stock down",
    }
    _eris = {
        "commodity": "contrarian signals: overcrowded long positioning in futures/ETF, retail FOMO, COT report extremes, or sentiment surveys showing excessive bullishness",
        "etf":       "contrarian signals: retail crowding, Rydex/sentiment surveys at extremes, fund flow exhaustion, and mean-reversion risk after extended move",
        "index":     "contrarian signals: AAII sentiment extremes, put/call ratio complacency, margin debt levels, and historical overextension vs prior peaks",
        "stock":     "contrarian signals: analyst price target consensus too bullish, short interest dropping (crowded longs), insider selling, or retail FOMO exhaustion",
    }
    if atype == "commodity":
        return {
            "ZEUS":     "physical supply/demand dynamics, global production data, ETF fund flows, and inventory/stockpile levels",
            "HERMES":   "price momentum using the RSI, MACD, and SMA data provided — cite exact numbers",
            "APOLLO":   "macro catalysts: Fed policy / dollar strength / inflation data / geopolitical events driving demand",
            "ARES":     "unusual options activity on {sym} or related ETFs, put/call ratio, and institutional positioning".format(sym=sym),
            "POSEIDON": "macro tailwinds: inflation hedge demand, currency debasement, central bank buying, or commodity supercycle",
            "KRONOS":   "overbought technicals, dollar strength risk, rising real yields, and demand destruction scenarios",
            "HADES":    "chart-based breakdown risk using the RSI, MACD hist, and SMA levels from the data provided — cite exact numbers",
            "NEMESIS":  "macro headwinds: Fed tightening, strong dollar, weak industrial demand, or ETF outflows",
            "TYPHON":   _typhon["commodity"],
            "ERIS":     _eris["commodity"],
        }
    elif atype == "etf":
        return {
            "ZEUS":     "index/fund composition strength, breadth of holdings, and sector allocation supporting a move up",
            "HERMES":   "price momentum using the RSI, MACD, and SMA data provided — cite exact numbers",
            "APOLLO":   "upcoming macro events, Fed policy, earnings season for top holdings, or fund flow catalysts",
            "ARES":     "unusual options activity, put/call ratio, and institutional ETF flow positioning",
            "POSEIDON": "macro tailwinds and sector rotation supporting the ETF's holdings and theme",
            "KRONOS":   "overextended valuation, overbought RSI/technicals, and concentration risk in top holdings",
            "HADES":    "chart-based breakdown risk using the RSI, MACD hist, and SMA levels from the data — cite exact numbers",
            "NEMESIS":  "macro headwinds: Fed policy, sector rotation away from holdings, or risk-off sentiment",
            "TYPHON":   _typhon["etf"],
            "ERIS":     _eris["etf"],
        }
    elif atype == "index":
        return {
            "ZEUS":     "market breadth, advance/decline, and index component strength supporting continued upside",
            "HERMES":   "index price momentum using the RSI, MACD, and SMA data provided — cite exact numbers",
            "APOLLO":   "upcoming macro catalysts: FOMC, CPI/PCE data, earnings season, and fiscal policy",
            "ARES":     "put/call ratio, VIX positioning, and institutional index options flow",
            "POSEIDON": "global macro tailwinds: liquidity, earnings growth expectations, and risk-on sentiment",
            "KRONOS":   "valuation stretch (P/E expansion), overbought breadth, and credit market stress signals",
            "HADES":    "technical breakdown risk using RSI, MACD hist, and key SMA support levels — cite exact numbers",
            "NEMESIS":  "macro headwinds: Fed tightening, geopolitical risk, recession signals, or earnings deterioration",
            "TYPHON":   _typhon["index"],
            "ERIS":     _eris["index"],
        }
    else:  # stock
        return {
            "ZEUS":     "recent earnings results, revenue growth, forward guidance, and balance sheet strength",
            "HERMES":   "price momentum using the RSI, MACD, and SMA data provided — cite exact numbers",
            "APOLLO":   "upcoming catalysts: earnings date, product launches, analyst upgrades, or news events",
            "ARES":     "unusual options activity, put/call ratio, dark pool prints, and institutional flow",
            "POSEIDON": "sector tailwinds, macro environment supporting the industry, and peer comparison",
            "KRONOS":   "valuation metrics (P/E vs peers), debt levels, and margin compression risk",
            "HADES":    "chart-based breakdown risk using the RSI, MACD hist, and SMA levels from the data — cite exact numbers",
            "NEMESIS":  "competitive threats, regulatory risk, macro headwinds, or insider selling signals",
            "TYPHON":   _typhon["stock"],
            "ERIS":     _eris["stock"],
        }


def _build_agent_prompt(sym, d, tf, bot_results, greeks, tech_bias_label="NEUTRAL (0/5)"):
    atype = _asset_type(sym)
    asset_label = {"commodity": "commodity/metal ETF", "etf": "broad ETF/fund",
                   "index": "market index", "stock": "individual stock"}.get(atype, "asset")
    roles = _agent_roles(sym, atype)

    bot_summary = " | ".join(
        "{}: {} {}%".format(BOTS[i]["id"], bot_results[i]["verdict"], bot_results[i]["confidence"])
        for i in range(len(bot_results))
    )
    px = d["px"] if d else "unknown"
    if d:
        pos52 = ((d["px"] - d["low_52"]) / (d["high_52"] - d["low_52"]) * 100) if d["high_52"] != d["low_52"] else 50
        data_s = (
            "Price ${px} ({sign}{chg}%) | RSI {rsi} | MACD hist {mh} ({mdir}) | "
            "EMA12/26 cross: {ec} | SMA7 ${sma7} | SMA20 ${sma20} | "
            "SMA200 {s200} | PCR {pcr} | 52w pos {pos:.0f}% | "
            "Vol {vr:.1f}x avg | 52w High ${h52} / Low ${l52}"
        ).format(
            px=d["px"], sign="+" if d["chg"] >= 0 else "", chg=d["chg"],
            rsi=d["rsi"], mh=d["macd_hist"],
            mdir="expanding" if abs(d["macd_hist"]) > abs(d["macd_hist_prev"]) else "contracting",
            ec="bullish" if d["ema12"] > d["ema26"] else "bearish",
            sma7=d["sma7"], sma20=d["sma20"],
            s200=("above ${:.2f}".format(d["sma200"]) if d.get("sma200") and d["px"] > d["sma200"]
                  else "below ${:.2f}".format(d["sma200"]) if d.get("sma200") else "N/A"),
            pcr=d["pcr"] if d["pcr"] is not None else "N/A",
            pos=pos52, vr=d["vol_ratio"],
            h52=d["high_52"], l52=d["low_52"],
        )
    else:
        data_s = "Live data unavailable — use your knowledge of {}.".format(sym)

    tf_s = ""
    if tf:
        bull_c = sum(1 for v in tf.values() if v == "BULL")
        bear_c = sum(1 for v in tf.values() if v == "BEAR")
        tf_s = "{} BULL / {} BEAR across 9 timeframes".format(bull_c, bear_c)
    else:
        tf_s = "TF data unavailable"

    # Greeks block for ARES and timeline guidance
    if greeks:
        cg = greeks.get("call_greeks") or {}
        pg = greeks.get("put_greeks")  or {}
        greeks_s = (
            "ATM STRIKE: ${atm} | Days to Expiry: {dte} | "
            "Call IV: {civ}% | Put IV: {piv}% | Avg IV: {aiv}%\n"
            "CALL Greeks — Delta: {cd} | Gamma: {cg} | Theta: ${ct}/day | Vega: ${cv}/1%IV | "
            "Bid/Ask: ${cb}/${ca}\n"
            "PUT  Greeks — Delta: {pd} | Gamma: {pgg} | Theta: ${pt}/day | Vega: ${pv}/1%IV | "
            "Bid/Ask: ${pb}/${pa}\n"
            "Open Interest — Calls: {coi:,} | Puts: {poi:,} | PCR: {pcr}"
        ).format(
            atm=greeks["atm_strike"], dte=greeks["days_out"],
            civ=greeks["call_iv"], piv=greeks["put_iv"], aiv=greeks["avg_iv"],
            cd=cg.get("delta","N/A"), cg=cg.get("gamma","N/A"),
            ct=cg.get("theta","N/A"), cv=cg.get("vega","N/A"),
            cb=greeks["call_bid"], ca=greeks["call_ask"],
            pd=pg.get("delta","N/A"), pgg=pg.get("gamma","N/A"),
            pt=pg.get("theta","N/A"), pv=pg.get("vega","N/A"),
            pb=greeks["put_bid"], pa=greeks["put_ask"],
            coi=greeks["call_oi"], poi=greeks["put_oi"],
            pcr=greeks["pcr"] if greeks["pcr"] is not None else "N/A",
        )
        # Timeline guidance based on IV and DTE
        dte = greeks["days_out"]
        avg_iv = greeks["avg_iv"]
        if dte <= 3:
            tl_hint = "scalp (0-1 day) — very short DTE, extreme theta decay"
        elif dte <= 10:
            tl_hint = "short (1-5 days) — low DTE, momentum play"
        elif avg_iv > 60:
            tl_hint = "short-to-swing — high IV means options are expensive, don't hold too long"
        elif avg_iv < 25:
            tl_hint = "swing-to-long (2-6 weeks) — cheap IV, good for holding through a move"
        else:
            tl_hint = "swing (1-4 weeks) — moderate IV and DTE"
    else:
        greeks_s = "Options Greeks unavailable — use your best estimate based on price action."
        tl_hint  = "swing (1-4 weeks) — default estimate"

    # Override ARES role to always focus on Greeks when available
    ares_role = (
        "options Greeks and structure: given the data above (Delta, Theta, Vega, IV, DTE), "
        "argue why the CALL side is the better options trade right now — "
        "address IV level, theta cost, and delta exposure"
    ) if greeks else roles["ARES"]
    kronos_role = (
        "options Greeks risk: given theta decay rate, IV level, and DTE, "
        "argue why the current options pricing makes buying calls risky — "
        "cite specific theta cost, IV crush risk, or unfavorable Greeks"
    ) if greeks else roles["KRONOS"]

    return (
        "{sym} is a {asset_label}. 10 AI agents debate CALL (price UP) vs PUT (price DOWN).\n\n"
        "OBJECTIVE TECHNICAL ASSESSMENT: {tech_bias}\n"
        "The debate outcome MUST be consistent with this technical reality.\n"
        "Bears should win if the score is negative. Bulls if positive. Do NOT override this with bias.\n\n"
        "MARKET DATA:\n{data}\n"
        "Timeframes: {tf}\n"
        "Technical bots (Python-computed, unbiased): {bots}\n\n"
        "OPTIONS & GREEKS DATA:\n{greeks}\n\n"
        "CRITICAL: {sym} is a {asset_label}. Arguments must be appropriate to this asset class.\n"
        "{no_earnings_note}\n\n"
        "BULL agents (argue for CALL / price UP):\n"
        "• ZEUS: {zeus}\n"
        "• HERMES: {hermes}\n"
        "• APOLLO: {apollo}\n"
        "• ARES: {ares}\n"
        "• POSEIDON: {poseidon}\n\n"
        "BEAR agents (argue for PUT / price DOWN — 5 agents for balance):\n"
        "• KRONOS: {kronos}\n"
        "• HADES: {hades}\n"
        "• NEMESIS: {nemesis}\n"
        "• TYPHON: {typhon}\n"
        "• ERIS: {eris}\n\n"
        "YOUR ONLY JOB: Write the argument text for each agent. Be specific, cite data.\n"
        "Probability, verdict, and timeline are computed separately from objective data — do NOT include them.\n"
        "Each agent: 2 sentences max, specific to their domain, citing actual numbers where available.\n"
        "price_target: realistic price level relative to ${px}. Do NOT default to current price.\n\n"
        "Reply with ONLY this JSON (no markdown, no backticks):\n"
        '{{"agents":{{'
        '"ZEUS":{{"argument":"...","price_target":0.0}},'
        '"HERMES":{{"argument":"...","price_target":0.0}},'
        '"APOLLO":{{"argument":"...","price_target":0.0}},'
        '"ARES":{{"argument":"...","price_target":0.0}},'
        '"POSEIDON":{{"argument":"...","price_target":0.0}},'
        '"KRONOS":{{"argument":"...","price_target":0.0}},'
        '"HADES":{{"argument":"...","price_target":0.0}},'
        '"NEMESIS":{{"argument":"...","price_target":0.0}},'
        '"TYPHON":{{"argument":"...","price_target":0.0}},'
        '"ERIS":{{"argument":"...","price_target":0.0}}'
        '}},'
        '"debate":"2 sentences on which domain had the strongest case and why."}}'
    ).format(
        sym=sym, asset_label=asset_label,
        data=data_s, tf=tf_s, bots=bot_summary, px=px,
        greeks=greeks_s, tech_bias=tech_bias_label,
        no_earnings_note=(
            "DO NOT mention company earnings, revenue, or P/E ratios — {sym} is NOT a company stock.".format(sym=sym)
            if atype in ("commodity", "etf", "index") else
            "Ground all fundamental claims in known or estimated data for {sym}.".format(sym=sym)
        ),
        zeus=roles["ZEUS"], hermes=roles["HERMES"], apollo=roles["APOLLO"],
        ares=ares_role, poseidon=roles["POSEIDON"],
        kronos=kronos_role, hades=roles["HADES"], nemesis=roles["NEMESIS"],
        typhon=roles["TYPHON"], eris=roles["ERIS"],
    )


def run_agents(sym, api_key, provider, market_data, tf_data, bot_results, greeks=None, tech_score=0):
    ts_label = (
        "STRONGLY BEARISH ({}/5) — bear agents have a significant edge".format(tech_score) if tech_score <= -3 else
        "MODERATELY BEARISH ({}/5) — bears have a slight edge".format(tech_score)           if tech_score <= -1 else
        "NEUTRAL ({}/5) — debate should be close".format(tech_score)                        if tech_score == 0  else
        "MODERATELY BULLISH ({}/5) — bulls have a slight edge".format(tech_score)           if tech_score <= 2  else
        "STRONGLY BULLISH ({}/5) — bull agents have a significant edge".format(tech_score)
    )
    prompt = _build_agent_prompt(sym, market_data, tf_data, bot_results, greeks,
                                 tech_bias_label=ts_label)
    system = "You are a multi-agent financial debate API. Output ONLY raw JSON. No markdown, no explanation, no backticks."

    if provider == "groq":
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "system", "content": system},
                             {"role": "user",   "content": prompt}],
                "temperature": 0.4,
                "max_tokens": 2200,
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
        r.raise_for_status()
        raw = r.json()["choices"][0]["message"]["content"].strip()
    else:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2200,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(c.text for c in msg.content if hasattr(c, "text")).strip()

    data = _extract_json(raw)
    if data is None:
        raise ValueError("Could not parse agent JSON: " + raw[:300])
    return data, raw


# ── Session state ──────────────────────────────────────────────────────────────
if "ant_key"  not in st.session_state:
    st.session_state["ant_key"]  = os.getenv("ANTHROPIC_API_KEY", "")
if "groq_key" not in st.session_state:
    st.session_state["groq_key"] = os.getenv("GROQ_API_KEY", "")
if "provider" not in st.session_state:
    st.session_state["provider"] = "groq" if os.getenv("GROQ_API_KEY") else "anthropic"
if "agent_results" not in st.session_state:
    st.session_state["agent_results"] = None
if "agent_raw" not in st.session_state:
    st.session_state["agent_raw"] = None
if "greeks_data" not in st.session_state:
    st.session_state["greeks_data"] = None
if "watchlist" not in st.session_state:
    _saved_wl = os.getenv("WATCHLIST", "")
    st.session_state["watchlist"] = [t.strip() for t in _saved_wl.split(",") if t.strip()]

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

    # ── Provider toggle ──
    st.markdown('<div style="color:#334;font-size:9px;letter-spacing:1px;margin-bottom:6px">AI PROVIDER</div>', unsafe_allow_html=True)
    provider_choice = st.radio("Provider", ["Groq (Free)", "Anthropic"],
                               index=0 if st.session_state["provider"] == "groq" else 1,
                               label_visibility="collapsed", horizontal=True, key="provider_radio")
    st.session_state["provider"] = "groq" if provider_choice == "Groq (Free)" else "anthropic"

    st.markdown("---")

    if st.session_state["provider"] == "groq":
        st.markdown('<div style="color:#334;font-size:9px;letter-spacing:1px;margin-bottom:4px">GROQ KEY <span style="color:#00ff8855">(free)</span></div>', unsafe_allow_html=True)
        new_groq = st.text_input("Groq Key", type="password", placeholder="gsk_...",
                                 label_visibility="collapsed", key="groq_input")
        g1, g2 = st.columns(2)
        if g1.button("SAVE", use_container_width=True, key="save_groq"):
            k = new_groq.strip()
            if k.startswith("gsk_"):
                st.session_state["groq_key"] = k
                set_key(_ENV_PATH, "GROQ_API_KEY", k)
                st.success("Saved ✓")
            else:
                st.error("Must start with gsk_")
        if g2.button("CLEAR", use_container_width=True, key="clear_groq"):
            st.session_state["groq_key"] = ""
            set_key(_ENV_PATH, "GROQ_API_KEY", "")
            st.rerun()
        if st.session_state["groq_key"]:
            st.markdown('<div style="color:#00ff8877;font-size:9px;margin-top:4px">● Groq key loaded</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#ff446677;font-size:9px;margin-top:4px">○ No key — get free key at console.groq.com</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:#334;font-size:9px;letter-spacing:1px;margin-bottom:4px">ANTHROPIC KEY</div>', unsafe_allow_html=True)
        new_ant = st.text_input("Anthropic Key", type="password", placeholder="sk-ant-api03-...",
                                label_visibility="collapsed", key="ant_input")
        a1, a2 = st.columns(2)
        if a1.button("SAVE", use_container_width=True, key="save_ant"):
            k = new_ant.strip()
            if k.startswith("sk-ant-"):
                st.session_state["ant_key"] = k
                set_key(_ENV_PATH, "ANTHROPIC_API_KEY", k)
                st.success("Saved ✓")
            else:
                st.error("Must start with sk-ant-")
        if a2.button("CLEAR", use_container_width=True, key="clear_ant"):
            st.session_state["ant_key"] = ""
            set_key(_ENV_PATH, "ANTHROPIC_API_KEY", "")
            st.rerun()
        if st.session_state["ant_key"]:
            st.markdown('<div style="color:#00ff8877;font-size:9px;margin-top:4px">● Anthropic key loaded</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#ff446677;font-size:9px;margin-top:4px">○ No key set</div>', unsafe_allow_html=True)

    st.markdown("---")
    # Apply watchlist selection before ticker widget renders
    if "pending_ticker" in st.session_state:
        st.session_state["ticker_inp"] = st.session_state.pop("pending_ticker")
    st.markdown('<div style="color:#334;font-size:9px;letter-spacing:1px;margin-bottom:4px">TICKER SYMBOL</div>', unsafe_allow_html=True)
    ticker = st.text_input("Ticker", placeholder="AAPL  TSLA  SPY...",
                           label_visibility="collapsed", key="ticker_inp").strip().upper()

    _active_key = st.session_state["groq_key"] if st.session_state["provider"] == "groq" else st.session_state["ant_key"]
    run_btn = st.button("▶ ANALYZE", use_container_width=True,
                        disabled=not (_active_key and ticker),
                        key="run_btn")
    st.markdown("---")

    # ── Watchlist ──────────────────────────────────────────────────────────────
    st.markdown('<div style="color:#334;font-size:9px;letter-spacing:1px;margin-bottom:6px">◈ WATCHLIST</div>', unsafe_allow_html=True)
    wl_inp = st.text_input("Add ticker to watchlist", placeholder="AAPL  TSLA  SPY...",
                           label_visibility="collapsed", key="wl_inp").strip().upper()
    wa, wb = st.columns(2)
    if wa.button("+ ADD", use_container_width=True, key="wl_add"):
        if wl_inp and wl_inp not in st.session_state["watchlist"]:
            st.session_state["watchlist"].append(wl_inp)
            set_key(_ENV_PATH, "WATCHLIST", ",".join(st.session_state["watchlist"]))
            st.rerun()
    if wb.button("CLEAR ALL", use_container_width=True, key="wl_clr"):
        st.session_state["watchlist"] = []
        set_key(_ENV_PATH, "WATCHLIST", "")
        st.rerun()

    if st.session_state["watchlist"]:
        for _wt in list(st.session_state["watchlist"]):
            _wca, _wcb = st.columns([3, 1])
            if _wca.button(_wt, key="wl_sel_" + _wt, use_container_width=True):
                st.session_state["pending_ticker"] = _wt
                st.rerun()
            if _wcb.button("✕", key="wl_rm_" + _wt, use_container_width=True):
                st.session_state["watchlist"].remove(_wt)
                set_key(_ENV_PATH, "WATCHLIST", ",".join(st.session_state["watchlist"]))
                st.rerun()
    else:
        st.markdown('<div style="color:#1e1e30;font-size:9px;margin-top:4px">No tickers saved yet.</div>', unsafe_allow_html=True)

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

_active_key = st.session_state["groq_key"] if st.session_state["provider"] == "groq" else st.session_state["ant_key"]
if not _active_key:
    _hint = "Get a free key at console.groq.com" if st.session_state["provider"] == "groq" else "Get a key at console.anthropic.com"
    st.markdown("""
    <div style="text-align:center;padding:40px 20px;color:#334">
      <div style="font-size:32px;margin-bottom:12px">🔑</div>
      <div style="font-size:13px;letter-spacing:1px">Paste your API key in the sidebar to get started.</div>
      <div style="font-size:10px;margin-top:8px;color:#222">{hint}</div>
    </div>""".format(hint=_hint), unsafe_allow_html=True)
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
og = fetch_options_greeks(ticker)
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

# ── Options Greeks panel ──────────────────────────────────────────────────────
if og:
    cg = og.get("call_greeks") or {}
    pg = og.get("put_greeks")  or {}
    def _gfmt(v):
        return "{:+.3f}".format(v) if isinstance(v, float) else str(v)
    iv_color = "#ff4466" if og["avg_iv"] > 60 else "#ffd700" if og["avg_iv"] > 35 else "#00ff88"
    st.markdown("""
<div style="background:linear-gradient(135deg,#08081a,#0d0d22);border:1px solid #1c1c3055;
  border-radius:10px;padding:10px 16px;margin-bottom:12px;overflow-x:auto">
  <div style="display:flex;align-items:center;gap:6px;margin-bottom:8px">
    <span style="color:#c084fc;font-size:10px;font-weight:800;letter-spacing:2px">⚙ OPTIONS GREEKS</span>
    <span style="color:#334;font-size:9px">· ATM ${atm} · {dte}d to expiry</span>
    <span style="margin-left:auto;padding:2px 8px;border-radius:6px;background:{ivc}18;
      border:1px solid {ivc}44;color:{ivc};font-size:9px;font-weight:800">IV {aiv}%</span>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
    <div style="background:#05050f;border-radius:7px;padding:9px;border:1px solid #00ff8818">
      <div style="color:#00ff88;font-size:9px;font-weight:800;margin-bottom:5px">▲ CALL · ${cb}–${ca}</div>
      <div style="display:flex;gap:12px;flex-wrap:wrap">
        <span style="color:#556;font-size:9px">Δ <span style="color:#00ff88">{cd}</span></span>
        <span style="color:#556;font-size:9px">Γ <span style="color:#00ff88">{cg}</span></span>
        <span style="color:#556;font-size:9px">Θ <span style="color:#ff4466">{ct}/day</span></span>
        <span style="color:#556;font-size:9px">ν <span style="color:#ffd700">{cv}/1%IV</span></span>
        <span style="color:#556;font-size:9px">IV <span style="color:#00cfff">{civ}%</span></span>
      </div>
    </div>
    <div style="background:#05050f;border-radius:7px;padding:9px;border:1px solid #ff446618">
      <div style="color:#ff4466;font-size:9px;font-weight:800;margin-bottom:5px">▼ PUT · ${pb}–${pa}</div>
      <div style="display:flex;gap:12px;flex-wrap:wrap">
        <span style="color:#556;font-size:9px">Δ <span style="color:#ff4466">{pd}</span></span>
        <span style="color:#556;font-size:9px">Γ <span style="color:#ff4466">{pgg}</span></span>
        <span style="color:#556;font-size:9px">Θ <span style="color:#ff4466">{pt}/day</span></span>
        <span style="color:#556;font-size:9px">ν <span style="color:#ffd700">{pv}/1%IV</span></span>
        <span style="color:#556;font-size:9px">IV <span style="color:#00cfff">{piv}%</span></span>
      </div>
    </div>
  </div>
  <div style="display:flex;gap:14px;margin-top:7px">
    <span style="color:#334;font-size:9px">OI Calls: <span style="color:#00ff8877">{coi:,}</span></span>
    <span style="color:#334;font-size:9px">OI Puts: <span style="color:#ff446677">{poi:,}</span></span>
    <span style="color:#334;font-size:9px">PCR: <span style="color:#ffd700">{pcr}</span></span>
  </div>
</div>""".format(
        atm=og["atm_strike"], dte=og["days_out"],
        aiv=og["avg_iv"], ivc=iv_color,
        cb=og["call_bid"], ca=og["call_ask"],
        cd=_gfmt(cg.get("delta")), cg=_gfmt(cg.get("gamma")),
        ct=_gfmt(cg.get("theta")), cv=_gfmt(cg.get("vega")), civ=og["call_iv"],
        pb=og["put_bid"], pa=og["put_ask"],
        pd=_gfmt(pg.get("delta")), pgg=_gfmt(pg.get("gamma")),
        pt=_gfmt(pg.get("theta")), pv=_gfmt(pg.get("vega")), piv=og["put_iv"],
        coi=og["call_oi"], poi=og["put_oi"],
        pcr=og["pcr"] if og["pcr"] is not None else "N/A",
    ), unsafe_allow_html=True)

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
            results, raw = run_swarm(ticker, _active_key, st.session_state["provider"], md, tf)
            st.session_state["swarm_results"] = results
            st.session_state["swarm_ticker"]  = ticker
            st.session_state["swarm_raw"]     = raw
            st.session_state["swarm_tf"]      = tf
            st.session_state.pop("swarm_error", None)
        except Exception as e:
            st.session_state["swarm_error"]   = str(e)
            st.session_state["swarm_ticker"]  = ticker
            st.session_state.pop("swarm_results", None)
    if st.session_state.get("swarm_results"):
        with st.spinner("⚡ Launching 8-agent debate on {}...".format(ticker)):
            try:
                _ts = _tech_score(md, tf)
                agent_data, agent_raw = run_agents(
                    ticker, _active_key, st.session_state["provider"],
                    md, tf, st.session_state["swarm_results"], og,
                    tech_score=_ts,
                )
                st.session_state["greeks_data"] = og
                st.session_state["agent_results"] = agent_data
                st.session_state["agent_raw"]     = agent_raw
                st.session_state.pop("agent_error", None)
            except Exception as e:
                st.session_state["agent_error"]   = str(e)
                st.session_state["agent_results"] = None

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

# ── Summary stats (computed once, used across tabs) ───────────────────────────
calls    = sum(1 for r in results if r["verdict"] == "CALL")
puts_n   = len(results) - calls
avg_c    = round(sum(r["confidence"] for r in results) / len(results))
final_bot = "CALL" if calls >= puts_n else "PUT"
fc        = "#00ff88" if final_bot == "CALL" else "#ff4466"
delta_rev = results[3].get("reversal") if len(results) > 3 else None

agent_data = st.session_state.get("agent_results")
agents_raw = agent_data.get("agents", {}) if agent_data else {}
debate_txt = str(agent_data.get("debate", "")) if agent_data else ""

# All numerical outputs computed from objective data — zero LLM influence
_ts_now = _tech_score(md, tf)
bull_prob, bear_prob = _compute_agent_probs(_ts_now, calls, len(results))

# Timeline: deterministic from data
_tl_key, (_tl_name, _tl_range, _tl_color) = _compute_timeline(md, og, _ts_now)
tl_name, tl_range, tl_color = _tl_name, _tl_range, _tl_color

cur_atype  = _asset_type(ticker)
bull_roles = _BULL_AGENT_ROLES.get(cur_atype, _BULL_AGENT_ROLES["stock"])
bear_roles = _BEAR_AGENT_ROLES.get(cur_atype, _BEAR_AGENT_ROLES["stock"])

if agent_data:
    bull_targets = [float(agents_raw.get(a["id"],{}).get("price_target",0) or 0) for a in BULL_AGENTS]
    bear_targets = [float(agents_raw.get(a["id"],{}).get("price_target",0) or 0) for a in BEAR_AGENTS]
    bull_avg = round(sum(t for t in bull_targets if t>0)/max(1,sum(1 for t in bull_targets if t>0)),2)
    bear_avg = round(sum(t for t in bear_targets if t>0)/max(1,sum(1 for t in bear_targets if t>0)),2)
    # Consensus target = prob-weighted average of all 10 agent price targets
    all_targets = [(t, True) for t in bull_targets] + [(t, False) for t in bear_targets]
    w_bull = bull_prob / 100
    w_bear = bear_prob / 100
    w_sum = sum((w_bull if is_b else w_bear) for t, is_b in all_targets if t > 0)
    consensus_t = round(
        sum(t * (w_bull if is_b else w_bear) for t, is_b in all_targets if t > 0) / max(0.001, w_sum), 2
    ) if w_sum > 0 else 0
else:
    bull_avg = bear_avg = 0
    consensus_t = 0

# Combined verdict = 55% tech + 45% bot consensus (fully objective, no LLM)
tech_call_pct = (_ts_now + 5) / 10
bot_call_pct  = calls / len(results)
combined_score = tech_call_pct * 55 + bot_call_pct * 45
combined_verdict = "CALL" if combined_score >= 50 else "PUT"
cvc = "#00ff88" if combined_verdict == "CALL" else "#ff4466"
# Confidence = how far from 50 the score is (stronger signal = higher conf)
combined_conf = min(93, max(55, 55 + int(abs(combined_score - 50) * 0.76)))

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_v, tab_b, tab_a = st.tabs(["⚡  VERDICT", "🤖  BOTS", "⚔  AGENTS"])

# ══════════════════ VERDICT TAB ══════════════════
with tab_v:
    # Compact bot vote row
    votes_html = "".join(
        '<div style="text-align:center;flex:1;min-width:52px">'
        '<div style="color:{bc};font-size:8px;letter-spacing:1px;margin-bottom:3px">{id}</div>'
        '<div style="padding:3px 6px;border-radius:8px;background:{vc}14;border:1px solid {vc}33;'
        'color:{vc};font-size:10px;font-weight:800">{v}</div>'
        '</div>'.format(bc=b["color"], id=b["id"],
                        vc="#00ff88" if r["verdict"]=="CALL" else "#ff4466", v=r["verdict"])
        for b, r in zip(BOTS, results)
    )
    st.markdown(
        '<div style="display:flex;gap:6px;justify-content:center;margin-bottom:12px;flex-wrap:wrap">'
        + votes_html + '</div>', unsafe_allow_html=True)

    # Main verdict card
    rev_html = ""
    if delta_rev and delta_rev != "NONE":
        rc2 = "#00ff88" if delta_rev=="BULLISH_REVERSAL" else "#ff4466"
        rl2 = "⚡ BULLISH REVERSAL" if delta_rev=="BULLISH_REVERSAL" else "⚡ BEARISH REVERSAL"
        rev_html = '<div style="margin-bottom:10px;padding:5px 14px;display:inline-block;background:{c}10;border:1px solid {c}44;border-radius:8px;color:{c};font-size:11px;font-weight:800">{l}</div>'.format(c=rc2,l=rl2)

    st.markdown("""
<div style="background:linear-gradient(145deg,#07071a,#0c0c22);border:2px solid {cvc};
  border-radius:14px;padding:22px;text-align:center;box-shadow:0 0 50px {cvc}14;
  position:relative;overflow:hidden">
  <div style="position:absolute;inset:0;pointer-events:none;
    background:radial-gradient(ellipse at 50% 0%,{cvc}08,transparent 65%)"></div>
  <div style="color:#333;font-size:9px;letter-spacing:3px;margin-bottom:4px">
    ▸ {label} ◂</div>
  <div style="color:#444;font-size:9px;margin-bottom:10px">
    {ticker} &nbsp;·&nbsp; Bots {bc}/5 CALL &nbsp;·&nbsp; {prob_line}
  </div>
  {rev}
  <div style="font-size:60px;font-weight:900;color:{cvc};letter-spacing:10px;
    line-height:1;text-shadow:0 0 40px {cvc}55;margin-bottom:6px">{cv}</div>
  <div style="color:{cvc}77;font-size:11px;letter-spacing:3px;margin-bottom:16px">
    {sig} &nbsp;·&nbsp; {conf}% CONFIDENCE
  </div>
  <div style="display:inline-flex;align-items:center;gap:10px;background:{tlc}10;
    border:1px solid {tlc}40;border-radius:10px;padding:8px 18px;margin-bottom:12px">
    <div style="text-align:left">
      <div style="color:{tlc};font-size:8px;letter-spacing:2px">⏱ HOLD RECOMMENDATION</div>
      <div style="color:{tlc};font-size:18px;font-weight:900;letter-spacing:4px">{tl_name}</div>
      <div style="color:{tlc}88;font-size:9px">{tl_range}</div>
    </div>
  </div>
  {tl_reason_html}
</div>""".format(
        cvc=cvc, ticker=ticker, bc=calls, cv=combined_verdict, conf=combined_conf,
        label="COMBINED VERDICT — BOTS + AGENTS" if agent_data else "5-BOT CONSENSUS",
        prob_line="Agents {}% BULL · 50/50 weight".format(bull_prob) if agent_data else "{}/5 CALL".format(calls),
        sig="COMBINED" if agent_data else ("STRONG" if avg_c>=80 else "MODERATE" if avg_c>=66 else "WEAK"),
        rev=rev_html,
        tlc=tl_color, tl_name=tl_name, tl_range=tl_range,
        tl_reason_html='<div style="color:#556;font-size:9px;max-width:460px;margin:0 auto 10px">Tech score: {}/5 — {}</div>'.format(
            _ts_now,
            "strongly bearish" if _ts_now<=-3 else "moderately bearish" if _ts_now<=-1 else
            "neutral" if _ts_now==0 else "moderately bullish" if _ts_now<=2 else "strongly bullish"
        ),
    ), unsafe_allow_html=True)

    # Price targets + prob bar (only if agents ran)
    if agent_data:
        st.markdown("""
<div style="display:flex;gap:10px;margin:12px 0;flex-wrap:wrap">
  <div style="flex:1;min-width:100px;text-align:center;padding:10px;
    background:#00ff8808;border:1px solid #00ff8820;border-radius:8px">
    <div style="color:#556;font-size:8px;margin-bottom:2px">BULL TARGET</div>
    <div style="color:#00ff88;font-size:16px;font-weight:800;font-family:'Courier New',monospace">${bull}</div>
  </div>
  <div style="flex:1;min-width:100px;text-align:center;padding:10px;
    background:#ffffff06;border:1px solid #ffffff10;border-radius:8px">
    <div style="color:#556;font-size:8px;margin-bottom:2px">CONSENSUS</div>
    <div style="color:#e0e0e0;font-size:16px;font-weight:800;font-family:'Courier New',monospace">${cons}</div>
  </div>
  <div style="flex:1;min-width:100px;text-align:center;padding:10px;
    background:#ff440808;border:1px solid #ff440820;border-radius:8px">
    <div style="color:#556;font-size:8px;margin-bottom:2px">BEAR TARGET</div>
    <div style="color:#ff4466;font-size:16px;font-weight:800;font-family:'Courier New',monospace">${bear}</div>
  </div>
</div>
<div style="margin-bottom:4px">
  <div style="color:#334;font-size:8px;margin-bottom:4px">PROBABILITY SPLIT — 5v5 DEBATE</div>
  <div style="height:16px;border-radius:8px;overflow:hidden;background:#0f0f20;display:flex">
    <div style="width:{bp}%;background:linear-gradient(90deg,#00ff8844,#00ff88);
      display:flex;align-items:center;justify-content:center;font-size:8px;font-weight:800;color:#050510">{bp}% BULL</div>
    <div style="width:{brp}%;background:linear-gradient(90deg,#ff446644,#ff4466);
      display:flex;align-items:center;justify-content:center;font-size:8px;font-weight:800;color:#050510">{brp}% BEAR</div>
  </div>
</div>
{debate_html}""".format(
            bull=bull_avg, cons=round(consensus_t,2), bear=bear_avg,
            bp=bull_prob, brp=bear_prob,
            debate_html='<div style="color:#667;font-size:10px;line-height:1.7;padding:10px;background:#05050f;border-radius:8px;border:1px solid #141428;margin-top:8px">{}</div>'.format(debate_txt) if debate_txt else "",
        ), unsafe_allow_html=True)

    st.markdown('<div style="text-align:center;color:#1e1e30;font-size:9px;margin-top:8px">⚠ AI analysis only — not financial advice</div>', unsafe_allow_html=True)

# ══════════════════ BOTS TAB ══════════════════
with tab_b:
    col_l, col_r = st.columns(2)
    for i in range(4):
        col = col_l if i % 2 == 0 else col_r
        col.markdown(_bot_card_html(BOTS[i], results[i]), unsafe_allow_html=True)

    # ATLAS full-width
    if len(results) >= 5:
        atlas_r  = results[4]
        atlas_bc = "#ffd700"
        atlas_vc = "#00ff88" if atlas_r["verdict"]=="CALL" else "#ff4466"
        saved_tf = st.session_state.get("swarm_tf") or {}
        tf_order = ["3min","5min","15min","30min","1hr","4hr","1day","1wk","1mo"]
        tf_lbls  = {"3min":"3M","5min":"5M","15min":"15M","30min":"30M","1hr":"1H","4hr":"4H","1day":"1D","1wk":"1W","1mo":"1MO"}
        def _tfc(b): return "#00ff88" if b=="BULL" else "#ff4466" if b=="BEAR" else "#667"
        tf_cells = "".join(
            '<div style="padding:5px 2px;border-radius:5px;background:{bc}14;border:1px solid {bc}30;text-align:center">'
            '<div style="color:#556;font-size:7px">{lbl}</div>'
            '<div style="color:{bc};font-size:9px;font-weight:800">{v}</div>'
            '</div>'.format(lbl=tf_lbls.get(k,k), v=saved_tf.get(k,"N/A"), bc=_tfc(saved_tf.get(k,"N/A")))
            for k in tf_order
        )
        atlas_metrics = "".join(
            '<div style="margin-bottom:6px">'
            '<div style="display:flex;justify-content:space-between;font-size:10px;color:#556">'
            '<span>{lbl}</span><span style="color:{bc}">{val:.0f}</span></div>'
            '<div style="height:3px;background:#0f0f20;border-radius:2px;overflow:hidden;margin-top:2px">'
            '<div style="width:{pct:.0f}%;height:100%;background:linear-gradient(90deg,{bc}55,{bc})"></div>'
            '</div></div>'.format(lbl=m["label"],val=m["value"],bc=atlas_bc,pct=min(100,m["value"]/m["max"]*100))
            for m in atlas_r["metrics"]
        )
        atlas_tags = "".join(
            '<span style="padding:2px 4px;border-radius:3px;font-size:7px;margin-right:2px;background:{bc}0c;border:1px solid {bc}18;color:{bc}55">{t}</span>'.format(bc=atlas_bc,t=t)
            for t in BOTS[4]["tags"]
        )
        st.markdown("""
<div style="background:linear-gradient(150deg,#0b0b1c,#0f0f26);border:1px solid {bc}40;
  border-radius:12px;padding:14px;margin-bottom:10px">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
    <div style="width:34px;height:34px;border-radius:8px;background:{bc}18;border:1px solid {bc}30;
      display:flex;align-items:center;justify-content:center;font-size:16px;color:{bc}">◎</div>
    <div style="flex:1">
      <div style="color:{bc};font-weight:800;font-size:12px;letter-spacing:2px">ATLAS</div>
      <div style="color:{bc}66;font-size:8px">MULTI-TIMEFRAME</div>
    </div>
    <div style="padding:2px 10px;border-radius:20px;background:{vc}12;border:1px solid {vc}44;
      color:{vc};font-weight:800;font-size:11px;letter-spacing:2px">{verdict}</div>
  </div>
  <div style="margin-bottom:6px">{tags}</div>
  <div style="display:grid;grid-template-columns:repeat(9,1fr);gap:4px;margin-bottom:8px">{cells}</div>
  <div style="background:#05050f;border-radius:7px;padding:10px;border:1px solid #141428;
    font-size:11px;line-height:1.7;color:#8899b0;margin-bottom:8px">{analysis}</div>
  {metrics}
  <div style="display:flex;justify-content:space-between;margin-top:6px">
    <span style="color:#222;font-size:9px">CONFIDENCE</span>
    <span style="color:{bc};font-size:11px;font-weight:800">{conf}%</span>
  </div>
</div>""".format(bc=atlas_bc,vc=atlas_vc,verdict=atlas_r["verdict"],tags=atlas_tags,
                 cells=tf_cells,analysis=atlas_r["analysis"],metrics=atlas_metrics,conf=atlas_r["confidence"]),
                    unsafe_allow_html=True)

# ══════════════════ AGENTS TAB ══════════════════
with tab_a:
    if not agent_data:
        st.markdown('<div style="text-align:center;padding:30px;color:#334;font-size:12px">Run ANALYZE to see the 10-agent debate.</div>', unsafe_allow_html=True)
    else:
        def _mini_agent_card(ag_def, side, role_lbl):
            a   = agents_raw.get(ag_def["id"], {})
            bc  = ag_def["color"]
            sc  = "#00ff88" if side=="BULL" else "#ff4466"
            arg = str(a.get("argument","")).strip()
            pt  = float(a.get("price_target", 0) or 0)
            conf= min(94, max(55, int(a.get("confidence", 68))))
            return """
<div style="background:linear-gradient(150deg,#0b0b1c,#0f0f26);border:1px solid {bc}40;
  border-radius:10px;padding:11px;margin-bottom:8px">
  <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px">
    <span style="font-size:14px;color:{bc}">{icon}</span>
    <div style="flex:1;min-width:0">
      <div style="color:{bc};font-weight:800;font-size:11px;letter-spacing:1px">{id}</div>
      <div style="color:{bc}55;font-size:7px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{role}</div>
    </div>
    <div style="padding:1px 6px;border-radius:8px;background:{sc}12;border:1px solid {sc}33;
      color:{sc};font-size:8px;font-weight:800;flex-shrink:0">{side}</div>
  </div>
  <div style="font-size:9px;line-height:1.6;color:#7788a0;margin-bottom:6px">{arg}</div>
  <div style="display:flex;justify-content:space-between;font-size:8px">
    <span style="color:{bc}">TARGET ${pt:.2f}</span>
    <span style="color:#444">{conf}% conf</span>
  </div>
</div>""".format(bc=bc, icon=ag_def["icon"], id=ag_def["id"], role=role_lbl,
                 sc=sc, side="▲ BULL" if side=="BULL" else "▼ BEAR",
                 arg=arg, pt=pt, conf=conf)

        st.markdown('<div style="color:#ffd70077;font-size:9px;font-weight:800;letter-spacing:3px;margin-bottom:6px">▲ BULL — 5 AGENTS</div>', unsafe_allow_html=True)
        ba_cols = st.columns(5)
        for i, ag in enumerate(BULL_AGENTS):
            ba_cols[i].markdown(_mini_agent_card(ag, "BULL", bull_roles[i]), unsafe_allow_html=True)

        st.markdown('<div style="color:#ff446677;font-size:9px;font-weight:800;letter-spacing:3px;margin:6px 0">▼ BEAR — 5 AGENTS</div>', unsafe_allow_html=True)
        be_cols = st.columns(5)
        for i, ag in enumerate(BEAR_AGENTS):
            be_cols[i].markdown(_mini_agent_card(ag, "BEAR", bear_roles[i]), unsafe_allow_html=True)

        if debate_txt:
            st.markdown("""
<div style="background:#07071a;border:1px solid #1c1c30;border-radius:10px;padding:14px;margin-top:8px">
  <div style="color:#334;font-size:9px;letter-spacing:2px;margin-bottom:6px">▸ DEBATE OUTCOME</div>
  <div style="color:#7788a0;font-size:11px;line-height:1.8">{}</div>
</div>""".format(debate_txt), unsafe_allow_html=True)

with st.expander("🔍 Raw API responses"):
    st.code(st.session_state.get("swarm_raw",""), language="json")
    st.code(st.session_state.get("agent_raw",""), language="json")

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
