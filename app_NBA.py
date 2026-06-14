
"""
Parquet Capital — Front Office Dashboard
Roster valuation, player forecasts, live cap optimization, and a trade simulator,
all driven by the trained models in models.py and the clean_roster dataset.
 
Run:  streamlit run app_NBA.py
(Build the dataset first:  python build_dataset.py)
"""
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
 
import models as M
 
st.set_page_config(page_title="Parquet Capital", layout="wide",
                   initial_sidebar_state="expanded")
 
# ----------------------------------------------------------------------
# Design tokens — "front office terminal": ink navy, parquet amber, court lines
# ----------------------------------------------------------------------
INK = "#11151C"; PANEL = "#1A2029"; LINE = "#2C3543"
AMBER = "#E8A14B"; CHALK = "#EAE6DD"; MUTE = "#8A93A2"
GREEN = "#4BAE8C"; RED = "#D2674F"; BLUE = "#5B8DB8"
 
st.markdown(f"""
<style>
  .stApp {{ background:{INK}; color:{CHALK}; }}
  section[data-testid="stSidebar"] {{ background:{PANEL}; border-right:1px solid {LINE}; }}
  h1,h2,h3,h4 {{ font-family:'DM Serif Display',Georgia,serif !important; color:{CHALK}; letter-spacing:.2px; }}
  .stApp, p, label, span, div {{ font-family:'Inter',system-ui,sans-serif; }}
  .eyebrow {{ color:{AMBER}; font-size:.72rem; letter-spacing:.22em; text-transform:uppercase; font-weight:600; }}
  .metric-card {{ background:{PANEL}; border:1px solid {LINE}; border-left:3px solid {AMBER};
                 padding:14px 16px; border-radius:4px; }}
  .metric-card .v {{ font-size:1.7rem; font-weight:700; font-family:'DM Serif Display',serif; }}
  .metric-card .l {{ color:{MUTE}; font-size:.74rem; letter-spacing:.04em; text-transform:uppercase; }}
  .flag-over {{ color:{RED}; font-weight:700; }}
  .flag-under {{ color:{GREEN}; font-weight:700; }}
  .flag-fair {{ color:{MUTE}; }}
  .verdict {{ background:{PANEL}; border:1px solid {LINE}; border-radius:6px; padding:18px 20px;
             font-size:1.02rem; line-height:1.5; }}
    .stDataFrame {{ border:1px solid {LINE}; }}
    section[data-testid="stSidebar"] label,
    .stApp label,
    .stRadio label,
    .stSelectbox label {{ color:#B8C0CC !important; }}
  div[data-testid="stRadio"] label,
  div[data-testid="stRadio"] label p {{ color:#B8C0CC !important; }}
  hr {{ border-color:{LINE}; }}
</style>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)
 
 
# ----------------------------------------------------------------------
# Data + model loading (cached)
# ----------------------------------------------------------------------
@st.cache_resource
def load_everything():
    df = M.load()
    train = M.build_training_table(df)
    models = M.train_quantile_models(train)
    backtest = M.evaluate(train, models)
    band_widen, _ = M.calibrate_band_widening(train)
    sens = M._estimate_stat_sensitivities(train)
    forecasts = M.forecast_three_seasons(df, models, band_widen=band_widen,
                                         sensitivities=sens)
    valued = M.value_players(forecasts)
 
    # current team/season = each player's most recent season on record
    latest = df.sort_values("season").groupby("name_key").tail(1)[["name_key", "Team", "season"]]
    latest = latest.rename(columns={"Team": "current_team", "season": "last_season"})
    valued = valued.merge(latest, on="name_key", how="left")
    valued["current_team"] = valued["current_team"].fillna(valued["Team"])
    # a "current roster" = players whose most recent season is the latest in the data
    max_season = int(df["season"].max())
    valued["is_current"] = valued["last_season"] == max_season
    return df, valued, backtest, models
 
 
# Friendly failure if the dataset hasn't been built yet, instead of a raw traceback.
try:
    df, VAL, BACKTEST, MODELS = load_everything()
except FileNotFoundError as e:
    st.error("Dataset not found — the app needs `clean_roster.csv` before it can load.")
    st.code(str(e), language="text")
    st.info("Build it once with `python build_dataset.py`, then refresh this page.")
    st.stop()
 
FEATURED = ["GSW", "LAL", "BOS", "DEN", "PHO", "NYK", "MIL", "DAL"]
cur = VAL[VAL["is_current"]]
teams_present = [t for t in FEATURED if t in cur["current_team"].unique()]
other = sorted([t for t in cur["current_team"].dropna().unique() if t not in teams_present])
ALL_TEAMS = teams_present + other
CAP = 154_000_000
 
 
def flag_html(f):
    if f == "Overvalued":  return f'<span class="flag-over">▲ Overvalued</span>'
    if f == "Undervalued": return f'<span class="flag-under">▼ Undervalued</span>'
    if f == "Fair Value":  return f'<span class="flag-fair">— Fair Value</span>'
    if f == "Elite (max-tier) - ceiling-capped":
        return (f'<span style="color:{BLUE};font-weight:700">★ Elite — ceiling-capped</span>')
    if f == "Below replacement - not priced":
        return f'<span class="flag-fair">· Below replacement</span>'
    return f'<span class="flag-fair">· n/a</span>'
 
 
# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown('<div class="eyebrow">Roster Valuation &amp; Risk Engine</div>', unsafe_allow_html=True)
st.markdown("# Parquet Capital")
st.markdown(f'<p style="color:{MUTE};margin-top:-8px">'
            f'Human capital as an asset — projecting performance, pricing contracts, '
            f'and optimizing the cap.</p>', unsafe_allow_html=True)
 
with st.sidebar:
    st.markdown('<div class="eyebrow">Front Office</div>', unsafe_allow_html=True)
    team = st.selectbox("Team", ALL_TEAMS, index=0)
    st.markdown("---")
    st.markdown(f'<p style="color:{MUTE};font-size:.8rem">Model backtest<br>'
                f'<span style="color:{CHALK}">{BACKTEST}</span></p>',
                unsafe_allow_html=True)
 
roster = VAL[(VAL["current_team"] == team) & (VAL["is_current"])].copy()
roster = roster.sort_values("salary_m", ascending=False)
 
# top metrics
c1, c2, c3, c4 = st.columns(4)
spend = roster["salary_m"].sum()
n_over = (roster["valuation_flag"] == "Overvalued").sum()
n_high = (roster["injury_risk_tier"] == "High").sum()
for col, val, lab in [
    (c1, f"{len(roster)}", "Players"),
    (c2, f"${spend:.0f}M", "Salary committed"),
    (c3, f"{n_over}", "Overvalued contracts"),
    (c4, f"{n_high}", "High injury risk")]:
    col.markdown(f'<div class="metric-card"><div class="v">{val}</div>'
                 f'<div class="l">{lab}</div></div>', unsafe_allow_html=True)
 
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["Roster Valuation", "Cap Optimizer", "Trade Simulator"])
 
# ----------------------------------------------------------------------
# Tab 1 — valuation table + player detail
# ----------------------------------------------------------------------
with tab1:
    show = roster[["Player", "pos_group", "Age", "salary_m", "current_bpm",
                   "bpm_t1_p50", "bpm_t3_p50", "valuation_flag",
                   "multiyear_flag", "years_remaining", "injury_risk_tier"]].copy()
    show.columns = ["Player", "Pos", "Age", "Salary $M", "BPM now",
                    "BPM +1", "BPM +3", "Valuation", "Multi-yr", "Yrs left", "Injury risk"]
    for c in ["Salary $M", "BPM now", "BPM +1", "BPM +3"]:
        show[c] = show[c].round(1)
 
    def color_flag(v):
        return (f"color:{RED};font-weight:700" if v == "Overvalued"
                else f"color:{GREEN};font-weight:700" if v == "Undervalued"
                else f"color:{BLUE};font-weight:700" if v == "Elite (max-tier) - ceiling-capped"
                else f"color:{BLUE}")
    def color_risk(v):
        return (f"color:{RED}" if v == "High"
                else f"color:{AMBER}" if v == "Medium" else f"color:{GREEN}")
    styled = (show.style
              .map(color_flag, subset=["Valuation"])
              .map(color_risk, subset=["Injury risk"])
              .format({"Salary $M": "{:.1f}", "BPM now": "{:+.1f}",
                       "BPM +1": "{:+.1f}", "BPM +3": "{:+.1f}"}))
    st.dataframe(styled, use_container_width=True, height=430, hide_index=True)
 
    st.markdown("#### Player detail")
    pick = st.selectbox("Inspect a player", roster["Player"].tolist())
    pr = roster[roster["Player"] == pick].iloc[0]
 
    dc1, dc2 = st.columns([3, 2])
    with dc1:
        traj = pd.DataFrame({
            "Season": ["Now", "+1", "+2", "+3"],
            "p50": [pr["current_bpm"], pr["bpm_t1_p50"], pr["bpm_t2_p50"], pr["bpm_t3_p50"]],
            "p10": [pr["current_bpm"], pr["bpm_t1_p10"], pr["bpm_t2_p10"], pr["bpm_t3_p10"]],
            "p90": [pr["current_bpm"], pr["bpm_t1_p90"], pr["bpm_t2_p90"], pr["bpm_t3_p90"]],
        })
        order = ["Now", "+1", "+2", "+3"]
        band = alt.Chart(traj).mark_area(opacity=0.18, color=AMBER).encode(
            x=alt.X("Season", sort=order, axis=alt.Axis(labelColor=MUTE, titleColor=MUTE)),
            y=alt.Y("p10", title="Projected BPM", axis=alt.Axis(labelColor=MUTE, titleColor=MUTE)),
            y2="p90")
        line = alt.Chart(traj).mark_line(color=AMBER, point=True, strokeWidth=2.5).encode(
            x=alt.X("Season", sort=order), y="p50")
        st.altair_chart((band + line).properties(height=260, background=PANEL,
                        title=alt.TitleParams(
                            f"{pick} — 3-season BPM projection (10th–90th pct)",
                            color=CHALK, fontSize=20, anchor="middle")),
                        use_container_width=True)
    with dc2:
        st.markdown(f'<div class="metric-card"><div class="v">${pr.salary_m:.1f}M</div>'
                    f'<div class="l">Current salary</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="verdict">{flag_html(pr.valuation_flag)}<br>'
                    f'<span style="color:{MUTE};font-size:.88rem">'
                    f'Age {int(pr.Age)} · {pr.pos_group} · injury risk '
                    f'<b>{pr.injury_risk_tier}</b><br>'
                    f'Comp rate ${pr.comp_dollar_per_bpm:.1f}M per value point'
                    if not pd.isna(pr.comp_dollar_per_bpm) else
                    f'<div class="verdict">{flag_html(pr.valuation_flag)}<br>'
                    f'<span style="color:{MUTE};font-size:.88rem">Age {int(pr.Age)} · {pr.pos_group}'
                    f'</span></div>', unsafe_allow_html=True)
 
# ----------------------------------------------------------------------
# Tab 2 — cap optimizer
# ----------------------------------------------------------------------
with tab2:
    st.markdown("Build the cap-efficient roster from this team's pool plus available "
                "talent, maximizing projected BPM under the salary cap.")
    mode = st.radio("Objective", ["Upside (median projection)", "Floor (10th pct)"],
                    horizontal=True)
    mkey = "upside" if mode.startswith("Upside") else "floor"
 
    if st.button("Build optimal roster", type="primary"):
        # pool: this team + a league free-agent pool of strong-value players
        pool = pd.concat([roster, VAL.nlargest(120, "bpm_t1_p50")]).drop_duplicates("name_key")
        with st.spinner("Solving knapsack under cap constraints…"):
            sel, proj, spend_opt = M.optimize_roster(pool, cap=CAP, mode=mkey)
        oc1, oc2, oc3 = st.columns(3)
        oc1.markdown(f'<div class="metric-card"><div class="v">{len(sel)}</div>'
                     f'<div class="l">Roster size</div></div>', unsafe_allow_html=True)
        oc2.markdown(f'<div class="metric-card"><div class="v">${spend_opt:.0f}M</div>'
                     f'<div class="l">Cap used of ${CAP/1e6:.0f}M</div></div>', unsafe_allow_html=True)
        oc3.markdown(f'<div class="metric-card"><div class="v">{proj:+.1f}</div>'
                     f'<div class="l">Projected BPM sum</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        out = sel[["Player", "pos_group", "Age", "salary_m", "bpm_t1_p50",
                   "valuation_flag"]].copy()
        out.columns = ["Player", "Pos", "Age", "Salary $M", "BPM +1", "Valuation"]
        out["kept"] = out["Player"].isin(roster["Player"]).map({True: "current", False: "ADD"})
        st.dataframe(out.round(1), use_container_width=True, hide_index=True, height=430)
 
# ----------------------------------------------------------------------
# Tab 3 — trade simulator
# ----------------------------------------------------------------------
with tab3:
    st.markdown("Evaluate a one-for-one trade: cap impact, projected next-season "
                "BPM delta, and how performance volatility changes.")
    tc1, tc2 = st.columns(2)
    with tc1:
        out_player = st.selectbox("Trade away", roster["Player"].tolist())
    with tc2:
        pool_in = VAL[VAL["current_team"] != team].sort_values("bpm_t1_p50", ascending=False)
        in_player = st.selectbox("Acquire", pool_in["Player"].tolist())
 
    if st.button("Evaluate trade", type="primary"):
        o = roster[roster.Player == out_player].iloc[0]
        i = pool_in[pool_in.Player == in_player].iloc[0]
        cap_delta = o["salary_m"] - i["salary_m"]
        bpm_delta = (i["bpm_t1_p50"] - o["bpm_t1_p50"])
        vol_o = o["bpm_t1_p90"] - o["bpm_t1_p10"]
        vol_i = i["bpm_t1_p90"] - i["bpm_t1_p10"]
        vol_word = ("increased" if vol_i > vol_o else "reduced")
 
        legal, reason = M.trade_is_legal(o["salary_m"], i["salary_m"], over_cap=True)
        legal_badge = (f'<span style="color:{GREEN};font-weight:700">&#10003; Salary-matching legal</span>'
                       if legal else
                       f'<span style="color:{RED};font-weight:700">&#10007; Fails salary matching</span>')
        cap_phrase = (f"frees ${cap_delta:.1f}M in cap space" if cap_delta > 0
                      else f"adds ${abs(cap_delta):.1f}M in salary")
        bpm_phrase = (f"improves projected next-season BPM by {bpm_delta:+.1f}"
                      if bpm_delta >= 0 else
                      f"lowers projected next-season BPM by {bpm_delta:.1f}")
        dc_out = o.get("dead_cap_m", np.nan)
        dc_note = (f'<br><span style="color:{AMBER};font-size:.85rem">Note: '
                   f'{out_player} carries ${dc_out:.1f}M in existing dead cap.</span>'
                   if pd.notna(dc_out) and dc_out > 0 else "")
        st.markdown(f'<div class="verdict">{legal_badge}<br>'
                    f'<span style="color:{MUTE};font-size:.85rem">{reason}</span><br><br>'
                    f'Trading <b>{out_player}</b> for '
                    f'<b>{in_player}</b> {cap_phrase} and {bpm_phrase}, '
                    f'at <b>{vol_word}</b> performance volatility.{dc_note}<br><br>'
                    f'<span style="color:{MUTE};font-size:.9rem">'
                    f'{out_player}: {flag_html(o.valuation_flag)} · '
                    f'{in_player}: {flag_html(i.valuation_flag)}</span></div>',
                    unsafe_allow_html=True)
 
st.markdown("---")
st.markdown(f'<p style="color:{MUTE};font-size:.76rem">Built on 4,860 player-seasons '
            f'(2017–2025) · BPM quantile forecast · comp-based valuation · '
            f'PuLP cap optimization. Salary = headline cap figure; dead-cap mechanics '
            f'are a documented v2 extension.</p>', unsafe_allow_html=True)