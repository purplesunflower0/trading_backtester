
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

from phase5_risk_metrics import buy_and_hold_equity_curve
from phase6_visualization import drawdown_series

HOLDOUT_START = "2024-01-01"
ASSET_LABELS = {"gold": "GOLD", "crude": "CRUDE OIL", "sp500": "S&P 500"}
ASSETS = list(ASSET_LABELS.keys())

st.set_page_config(page_title="MA Crossover + RSI Backtester", page_icon="📈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #10141C;
    --panel: #171B24;
    --panel-border: #262C38;
    --gold: #C9A227;
    --teal: #4FA88A;
    --brick: #C1554A;
    --text: #E8E6E1;
    --muted: #8B93A1;
}

.stApp { background-color: var(--bg); color: var(--text); }
[data-testid="stHeader"] { background-color: var(--bg); height: 3rem; }
.block-container { padding-top: 2rem !important; }
[data-testid="stSidebar"] { background-color: var(--panel); border-right: 1px solid var(--panel-border); }

/* Streamlit's own widget labels (e.g. "Asset", "Period for metrics below")
   were nearly invisible against our dark background - force them visible */
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
    color: var(--text) !important;
    opacity: 0.85;
}

/* Restyle Streamlit's default red radio-button accent to match the palette */
[data-testid="stSidebar"] [data-baseweb="radio"] div[aria-checked="true"] > div:first-child {
    border-color: var(--gold) !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] div[aria-checked="true"] > div:first-child > div {
    background-color: var(--gold) !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] div[role="radio"] { border-color: var(--muted) !important; }

h1, h2, h3 { font-family: 'Fraunces', serif !important; color: var(--text) !important; font-weight: 600 !important; }

.hero-sub { color: var(--muted); font-size: 1.05rem; margin-top: -0.6rem; margin-bottom: 1.6rem; }

.metric-card {
    background-color: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 6px;
    padding: 14px 16px;
    margin-bottom: 8px;
}
.metric-label { color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }
.metric-row { display: flex; justify-content: space-between; font-family: 'IBM Plex Mono', monospace; font-size: 1.05rem; }
.metric-strategy { color: var(--gold); }
.metric-bh { color: var(--muted); }

.stamp {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 10px 20px;
    border-radius: 4px;
    border: 2px solid;
    transform: rotate(-2deg);
}
.stamp-win { color: var(--teal); border-color: var(--teal); background-color: rgba(79, 168, 138, 0.08); }
.stamp-lose { color: var(--brick); border-color: var(--brick); background-color: rgba(193, 85, 74, 0.08); }

.caveat { color: var(--muted); font-size: 0.85rem; border-top: 1px solid var(--panel-border); padding-top: 12px; margin-top: 24px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_risk_metrics():
    return pd.read_csv("data_processed/risk_metrics.csv")


@st.cache_data
def load_asset_data(asset: str):
    strat_equity = pd.read_csv(
        f"data_processed/{asset}_equity_curve.csv", index_col="date", parse_dates=True
    )["equity"]
    price_df = pd.read_csv(f"data_raw/{asset}.csv", index_col="Date", parse_dates=True).sort_index()
    bh_equity = buy_and_hold_equity_curve(price_df, price_df.index.min())
    trades = pd.read_csv(f"data_processed/{asset}_trades.csv", parse_dates=["date"])
    return strat_equity, bh_equity, trades


risk_df = load_risk_metrics()

st.sidebar.markdown("### Controls")
asset = st.sidebar.radio("Asset", ASSETS, format_func=lambda a: ASSET_LABELS[a])
period_choice = st.sidebar.radio(
    "Period for metrics below",
    ["Full period (2016-present)", f"Holdout only ({HOLDOUT_START}-present)"],
)

strat_equity, bh_equity, trades = load_asset_data(asset)

is_holdout_view = period_choice.startswith("Holdout")
if is_holdout_view:
    strat_equity = strat_equity.loc[strat_equity.index >= HOLDOUT_START]
    bh_equity = bh_equity.loc[bh_equity.index >= HOLDOUT_START]
    trades = trades[trades["date"] >= HOLDOUT_START]

st.markdown("# MA Crossover + RSI Backtester")
st.markdown(
    "<div class='hero-sub'>A trend-following overlay on Gold, Crude, and the S&P 500 — "
    "walk-forward optimized, backtested with transaction costs, and tested honestly against "
    "doing nothing (buy-and-hold).</div>",
    unsafe_allow_html=True,
)

row = risk_df[(risk_df["asset"] == asset) & (risk_df["period"] == period_choice)]
strat_row = row[row["series"] == "strategy"].iloc[0]
bh_row = row[row["series"] == "buy_and_hold"].iloc[0]

card_col, stamp_col = st.columns([3, 1])

with card_col:
    metric_specs = [
        ("Annualized Return", "annualized_return", "{:.2f}%"),
        ("Sharpe Ratio", "sharpe", "{:.3f}"),
        ("Max Drawdown", "max_drawdown", "{:.2f}%"),
        ("Calmar Ratio", "calmar", "{:.3f}"),
    ]
    cols = st.columns(4)
    for col, (label, key, fmt) in zip(cols, metric_specs):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-row"><span class="metric-strategy">{fmt.format(strat_row[key])} <span style="font-size:0.65rem;color:var(--muted);">STRAT</span></span></div>
                <div class="metric-row"><span class="metric-bh">{fmt.format(bh_row[key])} <span style="font-size:0.65rem;color:var(--muted);">B&H</span></span></div>
            </div>
            """, unsafe_allow_html=True)

with stamp_col:
    strategy_wins = strat_row["calmar"] > bh_row["calmar"]
    stamp_class = "stamp-win" if strategy_wins else "stamp-lose"
    stamp_text = "Risk-Adjusted Win" if strategy_wins else "Buy & Hold Wins"
    st.markdown(f"""
    <div style="padding-top: 30px; text-align: center;">
        <div class="stamp {stamp_class}">{stamp_text}</div>
        <div style="color: var(--muted); font-size: 0.75rem; margin-top: 10px;">based on Calmar ratio<br>for this asset/period</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("### Equity Curve")

fig = go.Figure()
fig.add_trace(go.Scatter(x=strat_equity.index, y=strat_equity.values, name="Strategy",
                          line=dict(color="#C9A227", width=2)))
fig.add_trace(go.Scatter(x=bh_equity.index, y=bh_equity.values, name="Buy & Hold",
                          line=dict(color="#8B93A1", width=2)))
if not is_holdout_view:
    fig.add_vrect(x0=pd.Timestamp(HOLDOUT_START), x1=strat_equity.index.max(),
                  fillcolor="#E8E6E1", opacity=0.05, line_width=0,
                  annotation_text="Holdout", annotation_position="top left",
                  annotation_font_color="#8B93A1")
fig.update_layout(
    plot_bgcolor="#171B24", paper_bgcolor="#171B24", font_color="#E8E6E1",
    xaxis=dict(gridcolor="#262C38"), yaxis=dict(gridcolor="#262C38", title="Portfolio Value ($)"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    height=420, margin=dict(t=40, l=10, r=10, b=10),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("### Drawdown")

strat_dd = drawdown_series(strat_equity)
bh_dd = drawdown_series(bh_equity)

fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(x=strat_dd.index, y=strat_dd.values, name="Strategy",
                             fill="tozeroy", line=dict(color="#C9A227", width=1)))
fig_dd.add_trace(go.Scatter(x=bh_dd.index, y=bh_dd.values, name="Buy & Hold",
                             fill="tozeroy", line=dict(color="#8B93A1", width=1), opacity=0.6))
fig_dd.update_layout(
    plot_bgcolor="#171B24", paper_bgcolor="#171B24", font_color="#E8E6E1",
    xaxis=dict(gridcolor="#262C38"), yaxis=dict(gridcolor="#262C38", title="Drawdown (%)"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    height=320, margin=dict(t=40, l=10, r=10, b=10),
)
st.plotly_chart(fig_dd, use_container_width=True)

if asset == "crude":
    st.markdown(
        "<div class='caveat'>⚠ Crude's buy-and-hold drawdown includes April 20, 2020, when WTI futures "
        "briefly traded negative for the first time in history. Prices were floored at $0.01 to keep "
        "calculations stable, which distorts this one data point — treat any drawdown near -100% here "
        "as a known artifact, not a real result.</div>",
        unsafe_allow_html=True,
    )

with st.expander(f"Trade log — {ASSET_LABELS[asset]} ({len(trades)} trades)"):
    st.dataframe(trades, use_container_width=True)