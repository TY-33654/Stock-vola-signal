# -*- coding: utf-8 -*-
"""
標準偏差ボラティリティ トレードシグナル (単一ファイル版)
=========================================================
iPhoneだけでデプロイできるよう全コードを1ファイルに統合。
ロジック: BB(21,±1σ) + 標準偏差ボラティリティ(26) + ADX(14)

デプロイ: GitHubにこのファイルと requirements.txt を置き、
share.streamlit.io から公開 → iPhoneで「ホーム画面に追加」
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from datetime import datetime

# ---------------------------------------------------------------- パラメータ
BB_PERIOD = 21       # ボリンジャーバンド期間
BB_SIGMA = 1.0       # バンド幅(±1σ)
VOL_PERIOD = 26      # 標準偏差ボラティリティ期間
ADX_PERIOD = 14      # ADX期間
RISING_LOOKBACK = 1  # 「上昇中」判定の比較日数

DEFAULT_TICKERS = [
    "^N225",     # 日経平均
    "7203.T",    # トヨタ
    "6758.T",    # ソニーG
    "8306.T",    # 三菱UFJ
    "USDJPY=X",  # ドル円
    "AAPL",
    "NVDA",
]


# ---------------------------------------------------------------- 指標計算
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """OHLC DataFrameに BB / StdVol / ADX / シグナルを追加する"""
    df = df.copy()
    close, high, low = df["Close"], df["High"], df["Low"]

    # --- ボリンジャーバンド(21, ±1σ)
    mid = close.rolling(BB_PERIOD).mean()
    sd = close.rolling(BB_PERIOD).std(ddof=0)
    df["BB_mid"] = mid
    df["BB_up"] = mid + BB_SIGMA * sd
    df["BB_dn"] = mid - BB_SIGMA * sd

    # --- 標準偏差ボラティリティ(26期間の終値標準偏差)
    df["StdVol"] = close.rolling(VOL_PERIOD).std(ddof=0)

    # --- ADX (Wilder)
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    alpha = 1.0 / ADX_PERIOD
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["ADX"] = dx.ewm(alpha=alpha, adjust=False).mean()

    # --- 上昇中フラグ
    vol_rising = df["StdVol"] > df["StdVol"].shift(RISING_LOOKBACK)
    adx_rising = df["ADX"] > df["ADX"].shift(RISING_LOOKBACK)
    trend_on = vol_rising & adx_rising          # トレンド発生条件
    trend_off = (~vol_rising) & (~adx_rising)   # 両方低下=ピークアウト

    above = close > df["BB_up"]
    below = close < df["BB_dn"]

    # --- シグナル判定
    signal = pd.Series("NO TRADE", index=df.index)
    signal[above & trend_on] = "UP SIGNAL"
    signal[below & trend_on] = "DOWN SIGNAL"
    signal[(above | below) & trend_off] = "PEAK OUT"
    df["Signal"] = signal
    df["VolRising"] = vol_rising
    df["AdxRising"] = adx_rising
    return df


def summarize(ticker: str, df: pd.DataFrame) -> dict:
    """直近のシグナル状態を要約"""
    last = df.iloc[-1]
    # シグナルが現在の状態になってからの継続日数
    sig = df["Signal"]
    run = 1
    for i in range(len(sig) - 2, -1, -1):
        if sig.iloc[i] == last["Signal"]:
            run += 1
        else:
            break
    ts = df.index[-1]
    fmt = "%m-%d %H:%M" if (ts.hour or ts.minute) else "%Y-%m-%d"
    return {
        "ticker": ticker,
        "date": ts.strftime(fmt),
        "close": round(float(last["Close"]), 2),
        "signal": last["Signal"],
        "days": run,
        "stdvol_rising": "↑" if last["VolRising"] else "↓",
        "adx": round(float(last["ADX"]), 1),
        "adx_rising": "↑" if last["AdxRising"] else "↓",
        "bb_pos": "+1σ上" if last["Close"] > last["BB_up"]
                  else ("-1σ下" if last["Close"] < last["BB_dn"] else "バンド内"),
    }


def make_chart(ticker: str, df: pd.DataFrame):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    d = df.tail(180)
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.55, 0.225, 0.225],
        subplot_titles=("価格 + BB±1σ", "標準偏差ボラティリティ(26)", "ADX(14)"),
    )
    fig.add_trace(go.Candlestick(
        x=d.index, open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
        name="価格", increasing_line_color="#16a34a", decreasing_line_color="#dc2626",
    ), row=1, col=1)
    for col, name, color in [("BB_up", "+1σ", "#2563eb"),
                             ("BB_mid", "21SMA", "#9ca3af"),
                             ("BB_dn", "-1σ", "#2563eb")]:
        fig.add_trace(go.Scatter(x=d.index, y=d[col], name=name,
                                 line=dict(color=color, width=1)), row=1, col=1)

    # シグナルマーカー
    for sig, color, symbol, yk in [("UP SIGNAL", "#16a34a", "triangle-up", "Low"),
                                   ("DOWN SIGNAL", "#dc2626", "triangle-down", "High")]:
        m = d[d["Signal"] == sig]
        if len(m):
            fig.add_trace(go.Scatter(
                x=m.index, y=m[yk] * (0.995 if sig == "UP SIGNAL" else 1.005),
                mode="markers", name=sig,
                marker=dict(color=color, symbol=symbol, size=9)), row=1, col=1)

    fig.add_trace(go.Scatter(x=d.index, y=d["StdVol"], name="StdVol",
                             line=dict(color="#7c3aed", width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d["ADX"], name="ADX",
                             line=dict(color="#0891b2", width=1.5)), row=3, col=1)

    fig.update_layout(height=720, xaxis_rangeslider_visible=False,
                      margin=dict(l=40, r=20, t=50, b=30),
                      legend=dict(orientation="h", y=1.06))
    return fig




st.set_page_config(page_title="ボラティリティシグナル", page_icon="📊",
                   layout="centered")

SIGNAL_STYLE = {
    "UP SIGNAL":   ("🟢", "#16a34a", "上昇トレンド(買いゾーン)"),
    "DOWN SIGNAL": ("🔴", "#dc2626", "下降トレンド(売りゾーン)"),
    "PEAK OUT":    ("🟠", "#d97706", "ボラ低下(手仕舞い警戒)"),
    "NO TRADE":    ("⚪", "#6b7280", "様子見ゾーン"),
}
PERIOD_FOR = {"1d": "2y", "1h": "3mo", "15m": "1mo", "5m": "5d", "1m": "5d"}


@st.cache_data(ttl=60, show_spinner=False)
def fetch(ticker: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=PERIOD_FOR[interval], interval=interval,
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ---------------------------------------------------------------- サイドバー
with st.sidebar:
    st.header("⚙️ 設定")
    tickers_text = st.text_input(
        "銘柄(カンマ区切り)",
        value="NVDA, AAPL, SOXL, ^N225, 7203.T, USDJPY=X",
        help="日本株は 7203.T、指数は ^N225、FXは USDJPY=X の形式",
    )
    interval = st.selectbox("足種", ["1d", "1h", "15m", "5m", "1m"], index=0,
                            help="米国株はほぼリアルタイム / 日本株は約20分遅延")
    show_all_charts = st.toggle("全銘柄のチャートを表示", value=False)
    st.caption(
        f"BB({BB_PERIOD}, ±{BB_SIGMA:g}σ) / StdVol({VOL_PERIOD}) / "
        f"ADX({ADX_PERIOD})\n\nデータは60秒キャッシュ。"
        "再読込(プルリフレッシュ)で更新。"
    )

tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]

st.title("📊 ボラティリティシグナル")
st.caption("標準偏差ボラティリティ トレードモデル(BB±1σ + StdVol + ADX)")

if st.button("🔄 今すぐ更新", use_container_width=True):
    fetch.clear()
    st.rerun()

# ---------------------------------------------------------------- スキャン
results, frames = [], {}
errors = []
with st.spinner("スキャン中..."):
    for t in tickers:
        try:
            df = fetch(t, interval)
            if len(df) < max(BB_PERIOD, VOL_PERIOD, ADX_PERIOD) + 10:
                errors.append(f"{t}: データ不足")
                continue
            df = compute_indicators(df)
            results.append(summarize(t, df))
            frames[t] = df
        except Exception as e:
            errors.append(f"{t}: {e}")

if errors:
    st.warning(" / ".join(errors))
if not results:
    st.stop()

# ---------------------------------------------------------------- シグナル一覧
order = {"UP SIGNAL": 0, "DOWN SIGNAL": 1, "PEAK OUT": 2, "NO TRADE": 3}
results.sort(key=lambda r: order[r["signal"]])

active = [r for r in results if r["signal"] in ("UP SIGNAL", "DOWN SIGNAL")]
c1, c2, c3 = st.columns(3)
c1.metric("監視銘柄", len(results))
c2.metric("シグナル点灯", len(active))
c3.metric("足種", interval)

for r in results:
    icon, color, desc = SIGNAL_STYLE[r["signal"]]
    with st.container(border=True):
        top = st.columns([2, 3])
        top[0].markdown(f"### {icon} {r['ticker']}")
        top[1].markdown(
            f"<div style='text-align:right'>"
            f"<span style='color:{color};font-weight:700;font-size:18px'>"
            f"{r['signal']}</span><br>"
            f"<span style='color:#6b7280;font-size:12px'>{desc} / "
            f"{r['days']}本継続</span></div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(4)
        cols[0].metric("終値", r["close"])
        cols[1].metric("BB位置", r["bb_pos"])
        cols[2].metric("StdVol", r["stdvol_rising"])
        cols[3].metric("ADX", f"{r['adx']}{r['adx_rising']}")
        st.caption(f"最終データ: {r['date']}")
        if show_all_charts or st.toggle("チャート", key=f"chart_{r['ticker']}"):
            st.plotly_chart(make_chart(r["ticker"], frames[r["ticker"]]),
                            use_container_width=True)

st.caption("※情報提供目的のツールであり投資助言ではありません。"
           "watch足の直近1本は未確定のため判定が変わることがあります。")
