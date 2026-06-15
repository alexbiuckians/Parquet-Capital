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
 
  /* Make inactive tabs easier to read */
  button[data-baseweb="tab"] p {{
      color: #8FA3C4 !important;
      font-weight: 500 !important;
  }}
 
  button[data-baseweb="tab"][aria-selected="true"] p {{
      color: #FF4B4B !important;
      font-weight: 700 !important;
  }}
 
  button[data-baseweb="tab"]:hover p {{
      color: #EAE6DD !important;
  }}
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
    band_widen, _ = M.calibrate_band_widening(train, df=df)
    sens = M._estimate_stat_sensitivities(train)
    forecasts = M.forecast_three_seasons(df, models, band_widen=band_widen,
                                         sensitivities=sens)
    valued = M.value_players(forecasts)
 
    # Surface the uncertainty the validation layer already computes, per player:
    #   confidence_label  — from this player's own t+1 band width
    #   vorp_flag / agreement_label — does a 2nd independent target (VORP) agree?
    # Additive only; production value_players output is untouched.
    try:
        import ui_confidence as UC
        valued = UC.attach_all(valued, df)
    except Exception:
        valued["confidence_label"] = "Unknown"
        valued["vorp_flag"] = "n/a"
        valued["agreement_label"] = "n/a"
 
    # current team/season = each player's most recent season on record
    latest = df.sort_values("season").groupby("name_key").tail(1)[["name_key", "Team", "season"]]
    latest = latest.rename(columns={"Team": "current_team", "season": "last_season"})
    valued = valued.merge(latest, on="name_key", how="left")
    valued["current_team"] = valued["current_team"].fillna(valued["Team"])
    # a "current roster" = players whose most recent season is the latest in the data
    max_season = int(df["season"].max())
    valued["is_current"] = valued["last_season"] == max_season
    return df, valued, backtest, models
 
 
@st.cache_resource
def load_validation():
    """Run the (slower) validation suite once and cache it: out-of-sample
    quantile calibration and the held-out valuation-decision backtest. Imported
    lazily so the dashboard still loads if validation.py is absent."""
    try:
        import validation as V
    except Exception as e:  # pragma: no cover - defensive
        return {"error": f"validation module unavailable: {e}"}
    df_v = M.load()
    train_v = M.build_training_table(df_v)
    out = {}
    try:
        out["calibration"] = V.quantile_calibration(train_v, verbose=False)
    except Exception as e:
        out["calibration"] = None
        out["cal_error"] = str(e)
    try:
        summary, report = V.backtest_valuations(df_v, verbose=False)
        out["bt_summary"], out["bt_report"] = summary, report
    except Exception as e:
        out["bt_summary"], out["bt_report"] = None, str(e)
    return out
 
 
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
def concern_score(row):
    score = 0
 
    if row["valuation_flag"] == "Overvalued":
        score += 100
    elif row["valuation_flag"] == "Fair Value":
        score += 40
    elif row["valuation_flag"] == "Undervalued":
        score -= 25
    elif row["valuation_flag"] == "Elite (max-tier) - ceiling-capped":
        score -= 10
 
    if row["multiyear_flag"] == "Overvalued (multi-yr)":
        score += 60
    elif row["multiyear_flag"] == "Undervalued (multi-yr)":
        score -= 20
 
    if row["injury_risk_tier"] == "High":
        score += 25
    elif row["injury_risk_tier"] == "Medium":
        score += 10
 
    score += row["salary_m"] * 0.5
 
    return score
 
roster["concern_score"] = roster.apply(concern_score, axis=1)
roster = roster.sort_values("concern_score", ascending=False)
 
# top metrics
c1, c2, c3, c4 = st.columns(4)
spend = roster["salary_m"].sum()
cap_m = CAP / 1_000_000
cap_space = cap_m - spend
 
if cap_space >= 0:
    salary_label = f"${spend:.0f}M / ${cap_m:.0f}M"
    salary_sub = f"Salary committed | ${cap_space:.0f}M under cap"
else:
    salary_label = f"${spend:.0f}M / ${cap_m:.0f}M"
    salary_sub = f"Salary committed | ${abs(cap_space):.0f}M over cap"
 
n_over = (roster["valuation_flag"] == "Overvalued").sum()
n_high = (roster["injury_risk_tier"] == "High").sum()
 
for col, val, lab in [
    (c1, f"{len(roster)}", "Players"),
    (c2, salary_label, salary_sub),
    (c3, f"{n_over}", "Overvalued contracts"),
    (c4, f"{n_high}", "High injury risk")
]:
    col.markdown(f'<div class="metric-card"><div class="v">{val}</div>'
                 f'<div class="l">{lab}</div></div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
def explain_player_flag(p):
    reasons = []
 
    if p["valuation_flag"] == "Overvalued":
        reasons.append("salary is high relative to comparable projected value")
    elif p["valuation_flag"] == "Undervalued":
        reasons.append("projected value appears strong relative to salary")
    elif p["valuation_flag"] == "Fair Value":
        reasons.append("salary is broadly in line with comparable projected value")
 
    if p["bpm_t3_p50"] < p["current_bpm"]:
        reasons.append("the three-season BPM projection trends downward")
    elif p["bpm_t3_p50"] > p["current_bpm"]:
        reasons.append("the three-season BPM projection trends upward")
 
    if p["injury_risk_tier"] == "High":
        reasons.append("injury risk is elevated")
    elif p["injury_risk_tier"] == "Medium":
        reasons.append("injury risk is moderate")
 
    if p["Age"] >= 32:
        reasons.append("age increases decline risk")
    elif p["Age"] <= 24:
        reasons.append("age leaves room for development upside")
 
    if p["salary_m"] >= 20:
        reasons.append("current salary creates meaningful cap exposure")
    elif p["salary_m"] <= 5:
        reasons.append("low salary limits downside risk")
 
    if len(reasons) == 0:
        return "No major valuation, projection, or injury-risk concern is driving this profile."
 
    return "Flagged because " + ", ".join(reasons[:-1]) + (
        f", and {reasons[-1]}." if len(reasons) > 1 else f"{reasons[-1]}."
    )
 
st.markdown(
    f"""
    <div class="verdict">
    <b>Valuation note:</b> Elite max-tier players are ceiling-capped. 
    They are not automatically labeled Overvalued just because they have max-level salaries. 
    The model reserves Overvalued flags for contracts whose salary is high relative to comparable projected value.
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(
    ["Roster Valuation", "Cap Optimizer", "Trade Simulator", "Model Validation"])
 
# ----------------------------------------------------------------------
# Tab 1 — valuation table + player detail
# ----------------------------------------------------------------------
with tab1:
    show = roster[["Player", "pos_group", "Age", "salary_m", "current_bpm",
                   "bpm_t1_p50", "bpm_t3_p50", "valuation_flag",
                   "confidence_label", "agreement_label",
                   "multiyear_flag", "years_remaining", "injury_risk_tier"]].copy()
    show.columns = ["Player", "Pos", "Age", "Salary $M", "BPM now",
                    "BPM +1", "BPM +3", "Valuation", "Confidence", "2nd-metric",
                    "Multi-yr", "Yrs left", "Injury risk"]
    for c in ["Salary $M", "BPM now", "BPM +1", "BPM +3"]:
        show[c] = show[c].round(1)
    show["Yrs left"] = show["Yrs left"].fillna(0).astype(int)
    def color_flag(v):
        return (f"color:{RED};font-weight:700" if v == "Overvalued"
                else f"color:{GREEN};font-weight:700" if v == "Undervalued"
                else f"color:{BLUE};font-weight:700" if v == "Elite (max-tier) - ceiling-capped"
                else f"color:{BLUE}")
    def color_risk(v):
        return (f"color:{RED}" if v == "High"
                else f"color:{AMBER}" if v == "Medium" else f"color:{GREEN}")
    def color_conf(v):
        return (f"color:{GREEN}" if v == "High"
                else f"color:{AMBER}" if v == "Moderate"
                else f"color:{RED}" if v == "Low" else f"color:{MUTE}")
    def color_agree(v):
        return (f"color:{GREEN}" if v == "Confirmed"
                else f"color:{AMBER}" if v == "Mixed"
                else f"color:{RED};font-weight:700" if v == "Contradicts"
                else f"color:{MUTE}")
    styled = (show.style
              .map(color_flag, subset=["Valuation"])
              .map(color_risk, subset=["Injury risk"])
              .map(color_conf, subset=["Confidence"])
              .map(color_agree, subset=["2nd-metric"])
              .format({"Salary $M": "{:.1f}", "BPM now": "{:+.1f}",
                       "BPM +1": "{:+.1f}", "BPM +3": "{:+.1f}"}))
    st.dataframe(styled, use_container_width=True, height=430, hide_index=True)
    st.caption("Confidence = width of this player's own projection band (wide = "
               "low). 2nd-metric = whether an independent target (VORP) reaches "
               "the same verdict. A red 'Contradicts' means the call is "
               "target-dependent — do not act on it from this tool alone.")
    st.markdown("#### Front Office Takeaways")
    over = roster[roster["valuation_flag"] == "Overvalued"].head(3)
    under = roster[roster["valuation_flag"] == "Undervalued"].head(3)
    high_risk = roster[roster["injury_risk_tier"] == "High"].head(3)
 
    top_over = over.iloc[0]["Player"] if len(over) else "No clear overvalued contract"
    top_under = under.iloc[0]["Player"] if len(under) else "No clear undervalued player"
    top_risk = high_risk.iloc[0]["Player"] if len(high_risk) else "No high-risk player"
 
    st.markdown(
        f"""
        <div class="verdict">
        <b>Contract risk:</b> {top_over} grades as one of the biggest concerns based on salary, projected value, and valuation flag.<br><br>
 
        <b>Best value:</b> {top_under} appears underpriced relative to comparable projected value.<br><br>
 
        <b>Medical risk:</b> {top_risk} carries one of the highest injury-risk concerns on this roster.
        </div>
        """,
        unsafe_allow_html=True
    )
 
    st.markdown("#### Player detail")
    pick = st.selectbox("Inspect a player", roster["Player"].tolist())
    pr = roster[roster["Player"] == pick].iloc[0]
    why_flagged = explain_player_flag(pr)
    try:
        import ui_confidence as UC
        caveat = UC.confidence_sentence(pr)
    except Exception:
        caveat = ""
    # t+1 band string + VORP second-opinion line for the card
    band_str = (f'{pr["bpm_t1_p10"]:+.1f} to {pr["bpm_t1_p90"]:+.1f}'
                if pd.notna(pr.get("bpm_t1_p10")) else "n/a")
    vorp_line = (f'{pr.get("vorp_flag", "n/a")} '
                 f'({pr.get("agreement_label", "n/a")})')
 
    st.markdown(
        f"""
        <div class="verdict">
        <h3>{pr["Player"]}</h3>
 
        <b>Position:</b> {pr["pos_group"]}<br>
        <b>Age:</b> {pr["Age"]}<br>
        <b>Salary:</b> ${pr["salary_m"]:.1f}M<br><br>
 
        <b>Current BPM:</b> {pr["current_bpm"]:+.1f}<br>
        <b>Projected BPM +1:</b> {pr["bpm_t1_p50"]:+.1f}
            <span style="color:{MUTE}">(10–90 band: {band_str})</span><br>
        <b>Projected BPM +3:</b> {pr["bpm_t3_p50"]:+.1f}<br><br>
 
        <b>Valuation (BPM):</b> {pr["valuation_flag"]}
            &nbsp;·&nbsp; <b>Confidence:</b> {pr.get("confidence_label","Unknown")}<br>
        <b>2nd metric (VORP):</b> {vorp_line}<br>
        <b>Multi-year valuation:</b> {pr["multiyear_flag"]}<br>
        <b>Years remaining:</b> {pr["years_remaining"]}<br>
        <b>Injury risk:</b> {pr["injury_risk_tier"]}<br><br>
 
        <b>Why flagged:</b> {why_flagged}<br><br>
        <span style="color:{MUTE}">{caveat}</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)
    
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
                    f'Comp rate ${pr.comp_dollar_per_value:.1f}M per value point'
                    if not pd.isna(pr.comp_dollar_per_value) else
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
 
# ----------------------------------------------------------------------
# Tab 4 — model validation (calibration + held-out decision backtest)
# ----------------------------------------------------------------------
with tab4:
    st.markdown("Out-of-sample evidence that the engine does what it claims — "
                "shown on held-out seasons the models never trained on.")
    V = load_validation()
    if V.get("error"):
        st.warning(V["error"])
    else:
        vc1, vc2 = st.columns([1, 1])
 
        # --- quantile calibration reliability diagram ---
        with vc1:
            st.markdown("#### Quantile band calibration")
            cal = V.get("calibration")
            if cal is not None and len(cal):
                ideal = pd.DataFrame({"nominal": [0, 1], "empirical": [0, 1]})
                diag = alt.Chart(ideal).mark_line(
                    color=MUTE, strokeDash=[4, 4]).encode(
                    x=alt.X("nominal", title="Predicted quantile",
                            axis=alt.Axis(labelColor=MUTE, titleColor=MUTE)),
                    y=alt.Y("empirical", title="Empirical coverage",
                            axis=alt.Axis(labelColor=MUTE, titleColor=MUTE)))
                pts = alt.Chart(cal).mark_point(
                    color=AMBER, filled=True, size=90).encode(
                    x="nominal", y="empirical")
                ln = alt.Chart(cal).mark_line(color=AMBER).encode(
                    x="nominal", y="empirical")
                st.altair_chart((diag + ln + pts).properties(
                    height=260, background=PANEL,
                    title=alt.TitleParams("Reliability — on the dashed line = calibrated",
                                          color=CHALK, fontSize=15, anchor="middle")),
                    use_container_width=True)
                st.caption(f"Mean |empirical − nominal| = "
                           f"{cal['abs_error'].mean():.3f} (0 = perfect).")
            else:
                st.info("Calibration unavailable: " + V.get("cal_error", "n/a"))
 
        # --- valuation decision backtest ---
        with vc2:
            st.markdown("#### Do the flags hold up out-of-sample?")
            bt = V.get("bt_summary")
            if bt is not None and len(bt):
                disp = bt.rename(columns={
                    "flag": "Flag", "n": "N",
                    "median_salary_m": "Median $M",
                    "median_realized_value": "Realized value",
                    "median_realized_dollar_per_value": "Realized $/value"})
                st.dataframe(disp.round(2), use_container_width=True,
                             hide_index=True)
                st.caption("Realized **$/value** is actual next-season cost per "
                           "delivered value point. A working engine ranks "
                           "Overvalued > Fair > Undervalued — you pay most per "
                           "delivered unit on the contracts it flagged.")
                try:
                    o = bt.loc[bt.flag == "Overvalued",
                               "median_realized_dollar_per_value"].iloc[0]
                    fv = bt.loc[bt.flag == "Fair Value",
                                "median_realized_dollar_per_value"].iloc[0]
                    u = bt.loc[bt.flag == "Undervalued",
                               "median_realized_dollar_per_value"].iloc[0]
                    if o > fv > u:
                        st.markdown(
                            f'<div class="verdict"><span style="color:{GREEN};'
                            f'font-weight:700">✓ Signal confirmed</span><br>'
                            f'<span style="color:{MUTE};font-size:.88rem">'
                            f'Overvalued ${o:.2f} &gt; Fair ${fv:.2f} &gt; '
                            f'Undervalued ${u:.2f} per delivered value point.'
                            f'</span></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(
                            f'<div class="verdict"><span style="color:{AMBER};'
                            f'font-weight:700">~ Mixed signal</span><br>'
                            f'<span style="color:{MUTE};font-size:.88rem">'
                            f'Ordering not fully monotone on this sample — '
                            f'reported honestly rather than overclaimed.'
                            f'</span></div>', unsafe_allow_html=True)
                except (IndexError, KeyError):
                    pass
            else:
                st.info("Backtest unavailable: " + str(V.get("bt_report", "n/a")))
 
st.markdown("---")
st.markdown(f'<p style="color:{MUTE};font-size:.76rem">Built on 4,860 player-seasons '
            f'(2017–2025) · BPM quantile forecast · comp-based valuation · '
            f'PuLP cap optimization. Salary = headline cap figure; dead-cap mechanics '
            f'are a documented v2 extension.</p>', unsafe_allow_html=True)
