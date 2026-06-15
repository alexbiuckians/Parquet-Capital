
"""
Parquet Capital — Phases 2 & 3: Forecasting, Valuation, Optimization
Reads clean_roster.csv and produces:
  - forecasts.csv     3-season BPM projection (10th/50th/90th pct) per current player
  - valuations.csv    comp-based Overvalued/Fair/Undervalued flag per player
Plus an optimize_roster() function the dashboard calls live.
 
Modeling choices kept deliberately transparent and fast (no GPU/LSTM training
needed to demonstrate the framework). The forecast uses a gradient-boosted
trajectory model with quantile outputs — same inputs/outputs the LSTM spec
calls for (BPM, age, position, aging-curve delta, injury signal -> T+1..T+3),
and the uncertainty band plays the role of the Monte-Carlo-dropout spread.
"""
 
import numpy as np
import pandas as pd
import os
import sys
import argparse
from sklearn.ensemble import GradientBoostingRegressor
 
# Ensure accented player names (Jokić, Dončić, ...) print on consoles whose
# default encoding isn't UTF-8 (e.g. Windows cp1252), instead of raising
# UnicodeEncodeError. No-op where stdout is already UTF-8.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass
 
# Output folder configurable via PARQUET_OUT env var or --out flag.
# Default is a `parquet_out` folder next to THIS script (not the current working
# directory), so the app finds the data regardless of where it's launched from —
# e.g. Streamlit may run with a different cwd than the build step.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("PARQUET_OUT", os.path.join(_SCRIPT_DIR, "parquet_out"))
FEATURES = ["Age", "BPM", "PER", "WS_per_48", "VORP", "USG%",
            "injury_events_3yr", "aging_curve_delta"]
 
 
def load():
    path = os.path.join(OUT, "clean_roster.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"clean_roster.csv not found at:\n    {path}\n\n"
            f"Run the data-build step first so this file exists, e.g.:\n"
            f"    python build_dataset.py --out \"{OUT}\"\n\n"
            f"Or point both scripts at an existing output folder via the "
            f"PARQUET_OUT environment variable. The build step reads the three "
            f"raw CSVs (advanced stats, salaries, injuries) and writes "
            f"clean_roster.csv, which this app then loads.")
    df = pd.read_csv(path)
    for c in FEATURES + ["BPM"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
 
 
# ----------------------------------------------------------------------
# Phase 2 — BPM trajectory forecast with quantile uncertainty band
# ----------------------------------------------------------------------
def build_training_table(df):
    """Each row: features at season T -> BPM at T+1. Trained across all players
    (shared weights) exactly as the spec's single-shared-model approach. Also
    carries next-season rate stats (next_PER, ...) so the roll-forward can fit
    how those stats co-move with BPM."""
    d = df.sort_values(["name_key", "season"]).copy()
    d["target_bpm"] = d.groupby("name_key")["BPM"].shift(-1)
    d["next_season"] = d.groupby("name_key")["season"].shift(-1)
    for stat in ["PER", "WS_per_48", "VORP", "USG%"]:
        if stat in d.columns:
            d[f"next_{stat}"] = d.groupby("name_key")[stat].shift(-1)
    d = d[d["next_season"] == d["season"] + 1]
    d = d.dropna(subset=FEATURES + ["target_bpm"])
    return d
 
 
def train_quantile_models(train):
    """Three GBMs at the 10th/50th/90th quantile -> the projection band."""
    X, y = train[FEATURES], train["target_bpm"]
    models = {}
    for q, name in [(0.1, "p10"), (0.5, "p50"), (0.9, "p90")]:
        m = GradientBoostingRegressor(loss="quantile", alpha=q,
                                      n_estimators=200, max_depth=3,
                                      learning_rate=0.05, subsample=0.8,
                                      random_state=42)
        m.fit(X, y)
        models[name] = m
    return models
 
 
def build_aging_lookup(df, exclude_latest=False):
    """(pos_group, Age) -> mean aging_curve_delta, recovered from clean_roster.
    Lets the roll-forward refresh the aging signal as a player ages, instead of
    freezing year-T's value across all three projected seasons.
 
    Leakage note: the aging curve is a STRUCTURAL prior (how the average player
    at a position/age changes year over year), not a per-player label, so fitting
    it across all seasons is defensible — a player's own future BPM is never used
    as a training target here, only the population's historical year-over-year
    deltas. For strict purity, `exclude_latest=True` drops the most recent season
    (the one being forecast from) so no projected season contributes to the
    curve it is later scored against. The two differ negligibly because aging
    cells pool hundreds of player-seasons; the flag exists so the choice is
    explicit and auditable rather than implicit.
    """
    a = df.dropna(subset=["aging_curve_delta"])
    if exclude_latest and "season" in a.columns and len(a):
        a = a[a["season"] < int(df["season"].max())]
    lut = (a.groupby(["pos_group", "Age"])["aging_curve_delta"].mean())
    return lut.to_dict()
 
 
# How rate stats co-move with BPM in the roll-forward. A 1.0 BPM-point change is
# associated with roughly these per-step shifts, estimated from consecutive-season
# pairs in the training data (see _estimate_stat_sensitivities). Falls back to
# these literals if estimation is unavailable.
_DEFAULT_SENS = {"PER": 0.55, "WS_per_48": 0.006, "VORP": 0.18, "USG%": 0.10}
 
 
def _estimate_stat_sensitivities(train):
    """Per-stat slope d(stat)/d(BPM) from year-over-year changes, so the
    roll-forward can move the supporting rate stats with predicted BPM instead
    of holding them constant."""
    sens = {}
    d = train.copy()
    d["bpm_chg"] = d["target_bpm"] - d["BPM"]
    for stat in ["PER", "WS_per_48", "VORP", "USG%"]:
        nxt = f"next_{stat}"
        if nxt not in d.columns:
            sens[stat] = _DEFAULT_SENS[stat]
            continue
        chg = d[nxt] - d[stat]
        m = d["bpm_chg"].notna() & chg.notna() & (d["bpm_chg"].abs() > 1e-6)
        if m.sum() < 50:
            sens[stat] = _DEFAULT_SENS[stat]
        else:
            slope = np.polyfit(d.loc[m, "bpm_chg"], chg[m], 1)[0]
            sens[stat] = float(slope)
    return sens
 
 
def _advance_features(feat, p50, pos, aging_lut, sensitivities):
    """Advance a feature dict one season forward given the median BPM prediction.
    Shared by both the production roll-forward and the multi-step backtest so the
    two can never drift apart. Mutates and returns `feat`."""
    bpm_chg = p50 - feat["BPM"]
    for stat, slope in sensitivities.items():
        if stat in feat:
            feat[stat] = feat[stat] + slope * bpm_chg
    feat["BPM"] = p50
    feat["Age"] = feat["Age"] + 1
    feat["aging_curve_delta"] = aging_lut.get(
        (pos, feat["Age"]), feat["aging_curve_delta"])
    if "injury_events_3yr" in feat:
        feat["injury_events_3yr"] = feat["injury_events_3yr"] * 0.7
    return feat
 
 
def forecast_three_seasons(df, models, band_widen=None, sensitivities=None,
                           exclude_latest_in_aging=False):
    """Roll the median forecast forward 3 seasons, advancing the FULL feature
    vector at each step (not just BPM and Age):
      - Age      : +1 each season
      - BPM      : set to the model's median prediction
      - PER/WS_per_48/VORP/USG% : moved with predicted BPM via fitted slopes
      - aging_curve_delta       : refreshed from the (pos_group, new Age) curve
      - injury_events_3yr       : decayed toward 0 (the 3yr window rolls forward)
    band_widen scales the band each step; pass calibrated factors from
    calibrate_band_widening() to get empirically honest coverage.
 
    Leakage stance: in a LIVE forecast nothing is held out — we project forward
    FROM the latest season and never score against it — so the aging curve may
    safely include every season. `exclude_latest_in_aging=True` is offered only
    so a reviewer can confirm the live numbers barely move when the most recent
    season is dropped from the structural aging prior; it is not needed for
    correctness. Every BACKTEST path already fits aging/sensitivities on
    past-only data (see evaluate_multistep / _multistep_band_scale / the
    valuation backtest, all of which slice df[season <= base])."""
    if band_widen is None:
        band_widen = {1: 1.0, 2: 1.4, 3: 1.8}
    if sensitivities is None:
        sensitivities = _DEFAULT_SENS
    aging_lut = build_aging_lookup(df, exclude_latest=exclude_latest_in_aging)
 
    latest = (df.sort_values("season").groupby("name_key").tail(1).copy())
    latest = latest.dropna(subset=FEATURES)
    rows = []
    for _, r in latest.iterrows():
        feat = r[FEATURES].to_dict()
        cur_bpm = r["BPM"]
        pos = r["pos_group"]
        rec = {"name_key": r["name_key"], "Player": r["Player"],
               "Team": r["Team"], "pos_group": pos,
               "Age": r["Age"], "salary_m": r["salary_m"],
               "current_bpm": cur_bpm,
               "injury_risk_tier": r["injury_risk_tier"],
               # multi-year contract terms (NaN for players without a contract match)
               "contract_total_m": r.get("contract_total_m", np.nan),
               "contract_aav_m": r.get("contract_aav_m", np.nan),
               "years_remaining": r.get("years_remaining", np.nan),
               "dead_cap_m": r.get("dead_cap_m", np.nan)}
        for step in (1, 2, 3):
            X = pd.DataFrame([feat])[FEATURES]
            p50 = float(models["p50"].predict(X)[0])
            p10 = float(models["p10"].predict(X)[0])
            p90 = float(models["p90"].predict(X)[0])
            w = band_widen[step]
            mid = p50
            rec[f"bpm_t{step}_p10"] = mid - (mid - p10) * w
            rec[f"bpm_t{step}_p50"] = mid
            rec[f"bpm_t{step}_p90"] = mid + (p90 - mid) * w
 
            # --- advance the full feature vector for the next step ---
            _advance_features(feat, p50, pos, aging_lut, sensitivities)
        rows.append(rec)
    return pd.DataFrame(rows)
 
 
def evaluate(train, models):
    """Hold out the last 2 seasons; report MAE + directional accuracy in
    plain business language."""
    cut = train["season"].max() - 1
    tr, te = train[train.season < cut], train[train.season >= cut]
    if len(te) < 30:
        return "insufficient holdout"
    m = GradientBoostingRegressor(loss="quantile", alpha=0.5, n_estimators=200,
                                  max_depth=3, learning_rate=0.05,
                                  subsample=0.8, random_state=42)
    m.fit(tr[FEATURES], tr["target_bpm"])
    pred = m.predict(te[FEATURES])
    mae = np.mean(np.abs(pred - te["target_bpm"]))
    pred_dir = np.sign(pred - te["BPM"])
    true_dir = np.sign(te["target_bpm"] - te["BPM"])
    dir_acc = np.mean(pred_dir == true_dir)
    return f"MAE {mae:.2f} BPM | direction correct {dir_acc:.0%} of the time"
 
 
def evaluate_multistep(df, horizons=(1, 2, 3)):
    """Validate the ROLL-FORWARD itself, not just the one-step model.
 
    The production forecast feeds its own t+1 prediction back in to produce t+2
    and t+3, so single-step MAE understates real error at longer horizons. Here
    we actually roll the median model forward and compare each horizon against
    ground truth, so the degradation is measured rather than assumed.
 
    Method: train the median model only on seasons <= (max_season - max_horizon)
    so every evaluated horizon lands on a season the model never saw. For each
    player with an unbroken run of consecutive seasons starting at the cutoff
    base year, roll the feature vector forward h steps and compare to the true
    BPM h seasons later. Returns a per-horizon MAE / directional-accuracy report
    plus a naive 'persistence' baseline (assume BPM stays flat) so the numbers
    have something to beat."""
    max_season = int(df["season"].max())
    max_h = max(horizons)
    base = max_season - max_h            # last season the model may learn from
    if base < int(df["season"].min()) + 1:
        return "insufficient history for multi-step backtest"
 
    # train median model strictly before the evaluation window
    d = df.sort_values(["name_key", "season"]).copy()
    d["target_bpm"] = d.groupby("name_key")["BPM"].shift(-1)
    d["next_season"] = d.groupby("name_key")["season"].shift(-1)
    tr = d[(d["next_season"] == d["season"] + 1) & (d["season"] < base)]
    tr = tr.dropna(subset=FEATURES + ["target_bpm"])
    if len(tr) < 100:
        return "insufficient history for multi-step backtest"
    m = GradientBoostingRegressor(loss="quantile", alpha=0.5, n_estimators=200,
                                  max_depth=3, learning_rate=0.05,
                                  subsample=0.8, random_state=42)
    m.fit(tr[FEATURES], tr["target_bpm"])
 
    aging_lut = build_aging_lookup(df[df["season"] <= base])
    sens = _estimate_stat_sensitivities(tr.assign(
        **{f"next_{s}": d.groupby("name_key")[s].shift(-1)
           for s in ["PER", "WS_per_48", "VORP", "USG%"] if s in d.columns}))
 
    # ground-truth BPM indexed by (player, season)
    truth = df.set_index(["name_key", "season"])["BPM"].to_dict()
 
    # seed: each player's feature row in the base season
    seed = df[df["season"] == base].dropna(subset=FEATURES)
    err = {h: [] for h in horizons}          # rolled-forward abs errors
    base_err = {h: [] for h in horizons}     # persistence-baseline abs errors
    dir_hit = {h: [] for h in horizons}
 
    for _, r in seed.iterrows():
        feat = r[FEATURES].to_dict()
        pos = r["pos_group"]
        start_bpm = r["BPM"]
        for step in range(1, max_h + 1):
            p50 = float(m.predict(pd.DataFrame([feat])[FEATURES])[0])
            actual = truth.get((r["name_key"], base + step))
            if step in horizons and actual is not None and not np.isnan(actual):
                err[step].append(abs(p50 - actual))
                base_err[step].append(abs(start_bpm - actual))  # flat baseline
                dir_hit[step].append(
                    np.sign(p50 - start_bpm) == np.sign(actual - start_bpm))
            _advance_features(feat, p50, pos, aging_lut, sens)
 
    lines = []
    for h in horizons:
        if not err[h]:
            lines.append(f"t+{h}: no eval pairs")
            continue
        lines.append(
            f"t+{h}: MAE {np.mean(err[h]):.2f} BPM "
            f"(persistence {np.mean(base_err[h]):.2f}) | "
            f"direction {np.mean(dir_hit[h]):.0%} | n={len(err[h])}")
    return "multi-step roll-forward backtest:\n  " + "\n  ".join(lines)
 
 
def _scale_for_coverage(lo_gap, hi_gap, resid, nominal):
    """Find the symmetric band scale s such that the fraction of residuals
    falling inside [-(lo_gap*s), +(hi_gap*s)] about the median hits `nominal`.
    lo_gap = p50-p10, hi_gap = p90-p50, resid = actual - p50 (all per-row).
    Bisection on s; returns (s, achieved_coverage)."""
    lo_gap = np.asarray(lo_gap); hi_gap = np.asarray(hi_gap)
    resid = np.asarray(resid)
 
    def coverage(s):
        return np.mean((resid >= -lo_gap * s) & (resid <= hi_gap * s))
 
    lo_s, hi_s, s = 0.2, 6.0, 1.0
    for _ in range(40):
        s = 0.5 * (lo_s + hi_s)
        if coverage(s) < nominal:
            lo_s = s
        else:
            hi_s = s
    return s, coverage(s)
 
 
def _multistep_band_scale(df, horizon, nominal=0.80):
    """Measure the band scale needed at a GIVEN HORIZON from the ACTUAL
    roll-forward, rather than assuming sqrt(h). Trains quantile models strictly
    before the evaluation window, rolls each player's full feature vector forward
    `horizon` steps (the same _advance_features path production uses), and finds
    the scale that brings the rolled p10-p90 band to `nominal` coverage of the
    realized BPM `horizon` seasons later.
 
    Returns (scale, coverage, n_pairs) or (None, None, 0) if there is not enough
    held-out history to evaluate this horizon honestly."""
    max_season = int(df["season"].max())
    base = max_season - horizon            # last season the model may learn from
    if base < int(df["season"].min()) + 1:
        return None, None, 0
 
    d = df.sort_values(["name_key", "season"]).copy()
    d["target_bpm"] = d.groupby("name_key")["BPM"].shift(-1)
    d["next_season"] = d.groupby("name_key")["season"].shift(-1)
    tr = d[(d["next_season"] == d["season"] + 1) & (d["season"] < base)]
    tr = tr.dropna(subset=FEATURES + ["target_bpm"])
    if len(tr) < 100:
        return None, None, 0
 
    qm = {}
    for q, name in [(0.1, "p10"), (0.5, "p50"), (0.9, "p90")]:
        m = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=200,
                                      max_depth=3, learning_rate=0.05,
                                      subsample=0.8, random_state=42)
        m.fit(tr[FEATURES], tr["target_bpm"])
        qm[name] = m
 
    aging_lut = build_aging_lookup(df[df["season"] <= base])
    sens = _estimate_stat_sensitivities(tr.assign(
        **{f"next_{s}": d.groupby("name_key")[s].shift(-1)
           for s in ["PER", "WS_per_48", "VORP", "USG%"] if s in d.columns}))
    truth = df.set_index(["name_key", "season"])["BPM"].to_dict()
    seed = df[df["season"] == base].dropna(subset=FEATURES)
 
    lo_gaps, hi_gaps, resids = [], [], []
    for _, r in seed.iterrows():
        feat = r[FEATURES].to_dict()
        pos = r["pos_group"]
        p10 = p50 = p90 = None
        for step in range(1, horizon + 1):
            X = pd.DataFrame([feat])[FEATURES]
            p50 = float(qm["p50"].predict(X)[0])
            p10 = float(qm["p10"].predict(X)[0])
            p90 = float(qm["p90"].predict(X)[0])
            _advance_features(feat, p50, pos, aging_lut, sens)
        actual = truth.get((r["name_key"], base + horizon))
        if actual is None or np.isnan(actual):
            continue
        lo_gaps.append(p50 - p10); hi_gaps.append(p90 - p50)
        resids.append(actual - p50)
    if len(resids) < 20:
        return None, None, len(resids)
 
    s, cov = _scale_for_coverage(lo_gaps, hi_gaps, resids, nominal)
    return round(float(s), 3), float(cov), len(resids)
 
 
def calibrate_band_widening(train, nominal=0.80, df=None):
    """Calibrate the t+1/t+2/t+3 band-widening factors so the quantile bands
    actually cover ~`nominal` of held-out outcomes, instead of hand-picked
    multipliers.
 
    Method:
      1. t+1 is calibrated directly: train quantile models on all but the last
         two seasons, then find the scale that brings the one-step p10-p90 band
         to `nominal` coverage on the held-out seasons (bisection about median).
      2. t+2 / t+3 are calibrated against the ACTUAL multi-step roll-forward
         when enough held-out history exists (see _multistep_band_scale): we
         roll the full feature vector forward h steps exactly as production does
         and measure the scale each horizon truly needs. This replaces the
         sqrt(h) assumption with a measured factor.
      3. sqrt(h) random-walk widening is retained ONLY as an explicit fallback
         for horizons with too little held-out history to measure, and the
         report says which horizons were measured vs. assumed.
 
    `df` (the full player-season panel) is required for the measured t+2/t+3
    path; without it the function falls back to sqrt(h) and labels it as such.
    Returns (band_widen_dict, report_string)."""
    cut = train["season"].max() - 1
    tr, te = train[train.season < cut], train[train.season >= cut]
    if len(te) < 30:
        return {1: 1.0, 2: 1.4, 3: 1.8}, "bands uncalibrated (insufficient holdout)"
 
    qm = {}
    for q, name in [(0.1, "p10"), (0.5, "p50"), (0.9, "p90")]:
        m = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=200,
                                      max_depth=3, learning_rate=0.05,
                                      subsample=0.8, random_state=42)
        m.fit(tr[FEATURES], tr["target_bpm"])
        qm[name] = m
 
    p10 = qm["p10"].predict(te[FEATURES])
    p50 = qm["p50"].predict(te[FEATURES])
    p90 = qm["p90"].predict(te[FEATURES])
    y = te["target_bpm"].to_numpy()
 
    raw_cov = np.mean((y >= p10) & (y <= p90))
    s1, cal_cov = _scale_for_coverage(p50 - p10, p90 - p50, y - p50, nominal)
 
    # --- t+2 / t+3: measure from the real roll-forward where we can ---
    band = {1: round(float(s1), 3)}
    source = {1: "measured (1-step holdout)"}
    cov_by_h = {1: cal_cov}
    for h in (2, 3):
        s_h, cov_h, n_h = (None, None, 0)
        if df is not None:
            s_h, cov_h, n_h = _multistep_band_scale(df, h, nominal=nominal)
        if s_h is not None:
            band[h] = s_h
            source[h] = f"measured (roll-forward, n={n_h})"
            cov_by_h[h] = cov_h
        else:
            band[h] = round(float(s1) * np.sqrt(h), 3)
            source[h] = "assumed (sqrt(h) fallback - insufficient holdout)"
            cov_by_h[h] = None
 
    cov_str = "/".join(f"{cov_by_h[h]:.0%}" if cov_by_h[h] is not None else "n/a"
                       for h in (1, 2, 3))
    report = (f"band calibration: raw t+1 coverage {raw_cov:.0%} -> "
              f"calibrated {cal_cov:.0%} (target {nominal:.0%}); "
              f"widen factors t1/t2/t3 = {band[1]}/{band[2]}/{band[3]}; "
              f"achieved coverage t1/t2/t3 = {cov_str}; "
              f"t2 {source[2]}, t3 {source[3]}")
    return band, report
 
 
# ----------------------------------------------------------------------
# Phase 3 — comps valuation engine
# ----------------------------------------------------------------------
# A player's BPM is a per-100-possessions rate that runs negative below
# replacement level, so salary / BPM blows up (or flips sign) for low-impact
# players and the resulting "$/BPM" ratio is noise, not signal. We instead value
# on a positive, monotone "value score": BPM mapped onto a replacement-anchored
# scale where replacement level (BPM ≈ -2.0, the conventional anchor) maps to a
# small positive floor and impact scales linearly above it. Salary / value_score
# is then a stable $/win-style rate. Crucially, when a player's production sits
# at or below replacement we ABSTAIN (flag 'Below replacement - not priced')
# rather than emit a meaningless ratio: a near-zero, noise-dominated denominator
# should not be allowed to drive an Overvalued/Undervalued call.
REPLACEMENT_BPM = -2.0      # conventional replacement-level BPM anchor
VALUE_FLOOR = 0.5           # value-score floor at replacement level
# below this value score, $/value is too noise-dominated to price a contract
MIN_VALUE_FOR_PRICING = 1.0
 
# --- max-contract ceiling-cap handling ---------------------------------------
# The comp pool has no salary tier ABOVE the league max, so any elite producer on
# a max contract is necessarily compared against cheaper players and reads
# "Overvalued" — the flag ends up detecting "is on a max deal" as much as "is a
# bad deal." That conflates an MVP-level center earning the max (a fair,
# ceiling-capped contract the market literally cannot price higher) with a genuine
# dead-cap overpay (a max salary attached to replacement-level production).
#
# We separate the two by looking at PRODUCTION, not just price. A contract in the
# top salary tier whose production is also elite is flagged "Elite (max-tier) -
# ceiling-capped": an explicit abstention, because the comp set cannot fairly
# price it. A top-tier salary attached to non-elite production is left to the
# normal comp logic, where it correctly flags Overvalued — that is a real overpay,
# not a pricing artifact. Thresholds are data-driven (high salary and BPM
# percentiles within the priced pool), not hand-set dollar figures, so they track
# the league's actual cap environment rather than a fixed number.
MAX_TIER_SALARY_PCTL = 0.90   # top decile of priced salaries = "max-tier" pay
ELITE_BPM_PCTL = 0.90         # top decile of priced production = "elite"
 
 
def _value_score(bpm):
    """Map BPM onto a positive, replacement-anchored value scale.
    BPM == REPLACEMENT_BPM -> VALUE_FLOOR; each BPM point above adds 1.0."""
    return VALUE_FLOOR + np.maximum(bpm - REPLACEMENT_BPM, 0.0)
 
 
def value_players(forecasts, verbose=False):
    """Comp-based contract valuation. Each player's $/value-point is compared to
    the median $/value-point of comparable players (same position group, similar
    age and similar production).
 
    Denominator: we price on a replacement-anchored `value_score` (see
    `_value_score`) instead of raw BPM. This is strictly positive and monotone,
    so the $/value rate is stable rather than exploding as BPM approaches zero.
 
    Abstention: players whose value_score is below MIN_VALUE_FOR_PRICING are at
    or near replacement level, where salary / value is noise-dominated; these are
    flagged 'Below replacement - not priced' rather than forced into an
    Overvalued/Undervalued bucket on the strength of a meaningless ratio. They
    still serve as comps for others only if above the pricing floor.
 
    Coverage: a progressively looser set of (age, value) tiers is tried, with a
    final position-only tier as a backstop, so every priceable player receives a
    market rate. The only unrated players are those with no salary on record or
    those below the replacement pricing floor (an explicit, honest abstention,
    not a silent gap). The realized share is printed when verbose=True."""
    f = forecasts.copy()
    f["value_score"] = _value_score(f["current_bpm"])
    f["priceable"] = (f["salary_m"] > 0) & (f["value_score"] >= MIN_VALUE_FOR_PRICING)
    f["dollar_per_value"] = np.where(f["priceable"],
                                     f["salary_m"] / f["value_score"], np.nan)
    pool = f[f["dollar_per_value"].notna()]
 
    # Data-driven thresholds for the ceiling-cap rule, taken from the priced pool
    # so they track the league's real cap/talent distribution rather than fixed
    # dollar/BPM constants. A player who is BOTH in the top salary tier and the top
    # production tier is on a fair-but-unpriceable max deal, not an overpay.
    if len(pool):
        max_tier_salary = pool["salary_m"].quantile(MAX_TIER_SALARY_PCTL)
        elite_bpm = pool["current_bpm"].quantile(ELITE_BPM_PCTL)
    else:
        max_tier_salary, elite_bpm = np.inf, np.inf
 
    # progressively looser (age_band, value_band) tiers; None value_band = position-only
    TIERS = [(2, 2), (3, 3), (4, 4), (5, 6), (6, None)]
    MIN_COMPS = 3
 
    flags, comp_rates = [], []
    for _, r in f.iterrows():
        comp_rate, ratio = np.nan, np.nan
        if r["salary_m"] <= 0 or pd.isna(r["salary_m"]):
            comp_rates.append(np.nan)
            flags.append("Unrated (no salary)")
            continue
        if not r["priceable"]:
            comp_rates.append(np.nan)
            flags.append("Below replacement - not priced")
            continue
        # Ceiling-cap abstention: elite production on a max-tier salary. The comp
        # pool has no higher-paid tier to price this against, so a comp call would
        # spuriously read Overvalued. Abstain explicitly instead of conflating an
        # MVP-level max deal with a genuine dead-cap overpay. (A max-tier salary on
        # NON-elite production falls through to the comp logic below and is allowed
        # to flag Overvalued — that overpay is real, not an artifact.)
        if r["salary_m"] >= max_tier_salary and r["current_bpm"] >= elite_bpm:
            comp_rates.append(np.nan)
            flags.append("Elite (max-tier) - ceiling-capped")
            continue
        for age_band, value_band in TIERS:
            m = ((pool.pos_group == r.pos_group)
                 & (pool.Age.between(r.Age - age_band, r.Age + age_band))
                 & (pool.name_key != r.name_key))
            if value_band is not None:
                m = m & (pool.value_score.between(
                    r.value_score - value_band, r.value_score + value_band))
            comps = pool[m]
            if len(comps) >= MIN_COMPS:
                comp_rate = comps["dollar_per_value"].median()
                ratio = (r["dollar_per_value"] / comp_rate
                         if comp_rate and comp_rate > 0 else np.nan)
                break
        comp_rates.append(comp_rate)
        if pd.isna(ratio):
            flags.append("Insufficient comps")
        elif ratio > 1.30:
            flags.append("Overvalued")
        elif ratio < 0.70:
            flags.append("Undervalued")
        else:
            flags.append("Fair Value")
 
    f["comp_dollar_per_value"] = comp_rates
    f["valuation_flag"] = flags
 
    # ------------------------------------------------------------------
    # Multi-year contract valuation (additive — does not alter the single-season
    # flag above). Where a player has real multi-year terms, we price the TOTAL
    # guaranteed dollars over the remaining years against the projected VALUE the
    # player delivers across those same years, using the forecast median BPM at
    # t+1/t+2/t+3 mapped through the same replacement-anchored value_score. This is
    # the contract-level read the headline-cap single season cannot give: a flat
    # veteran with three guaranteed years left is a worse asset than the same
    # production on an expiring deal, and only this captures it.
    # ------------------------------------------------------------------
    def projected_value_years(row):
        # value delivered over the remaining contract years (cap at 3 = forecast
        # horizon; beyond that we hold the t+3 projection flat, stated as a known
        # simplification rather than extrapolating the model past its tested range)
        yrs = row.get("years_remaining", np.nan)
        if pd.isna(yrs) or yrs <= 0:
            return np.nan
        yrs = int(min(yrs, 3))
        bpms = [row.get(f"bpm_t{t}_p50", np.nan) for t in range(1, yrs + 1)]
        vals = [_value_score(b) for b in bpms if pd.notna(b)]
        return float(np.sum(vals)) if vals else np.nan
 
    f["proj_contract_value"] = f.apply(projected_value_years, axis=1)
    has_contract = f["contract_total_m"].notna() & (f["contract_total_m"] > 0)
    f["contract_dollar_per_value"] = np.where(
        has_contract & f["proj_contract_value"].notna() & (f["proj_contract_value"] > 0),
        f["contract_total_m"] / f["proj_contract_value"], np.nan)
 
    # comp the multi-year $/value against the same priced pool's single-year rate,
    # scaled by an average contract length, so the two live on a comparable scale.
    pool_rate = pool["dollar_per_value"].median() if len(pool) else np.nan
    my_flags = []
    for _, r in f.iterrows():
        cdv = r["contract_dollar_per_value"]
        # Inherit the single-season abstentions so the multi-year track does not
        # re-introduce the very bug the ceiling-cap rule fixed: the largest, longest
        # max deals mechanically score worst against a flat per-year benchmark, so an
        # MVP-tier max would spuriously read Overvalued. Abstain on the same honest
        # grounds (elite max-tier, below replacement, no salary) rather than emit a
        # verdict the comp set cannot actually support.
        base = r["valuation_flag"]
        if base in ("Elite (max-tier) - ceiling-capped", "Below replacement - not priced",
                    "Unrated (no salary)"):
            my_flags.append(base)
            continue
        if pd.isna(cdv) or pd.isna(pool_rate) or pool_rate <= 0:
            my_flags.append("No multi-year contract")
            continue
        yrs = int(min(r.get("years_remaining", 1) or 1, 3))
        bench = pool_rate * yrs            # expected total $/value over the same span
        ratio = cdv / bench if bench > 0 else np.nan
        if pd.isna(ratio):
            my_flags.append("No multi-year contract")
        elif ratio > 1.30:
            my_flags.append("Overvalued (multi-yr)")
        elif ratio < 0.70:
            my_flags.append("Undervalued (multi-yr)")
        else:
            my_flags.append("Fair Value (multi-yr)")
    f["multiyear_flag"] = my_flags
    # dead-cap exposure surfaced for the app (NaN -> no stranded money)
    f["dead_cap_m"] = f.get("dead_cap_m", np.nan)
    if verbose:
        n = len(f)
        not_rated = ["Insufficient comps", "Unrated (no salary)",
                     "Below replacement - not priced",
                     "Elite (max-tier) - ceiling-capped"]
        rated = (~f["valuation_flag"].isin(not_rated)).sum()
        priceable = int(f["priceable"].sum())
        ceiling = int((f["valuation_flag"] == "Elite (max-tier) - ceiling-capped").sum())
        print(f"valuation coverage: {rated}/{n} rated ({rated/n:.1%}); "
              f"{priceable} priceable (salary on record AND above replacement); "
              f"{ceiling} elite max-tier abstained (ceiling-capped); "
              f"flags: {f['valuation_flag'].value_counts().to_dict()}")
    return f
 
# ----------------------------------------------------------------------
# Phase 3 — PuLP roster optimizer (called live by the dashboard)
# ----------------------------------------------------------------------
def optimize_roster(players: pd.DataFrame, cap=154_000_000, mode="upside",
                    roster_size=13):
    """Knapsack: pick `roster_size` players maximizing projected BPM under the
    salary cap, with position minimums and a no-single-player>35%-of-cap rule.
    mode='upside' maximizes median projection; 'floor' maximizes the p10."""
    import pulp
    p = players.dropna(subset=["salary_m"]).copy()
    p = p[p["salary_m"] > 0].reset_index(drop=True)
    val_col = "bpm_t1_p50" if mode == "upside" else "bpm_t1_p10"
 
    prob = pulp.LpProblem("roster", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in p.index}
 
    prob += pulp.lpSum(p.loc[i, val_col] * x[i] for i in p.index)
    prob += pulp.lpSum(x[i] for i in p.index) == roster_size
    prob += pulp.lpSum(p.loc[i, "salary_m"] * 1e6 * x[i] for i in p.index) <= cap
    # position minimums
    for grp, lo in [("G", 3), ("F", 3), ("C", 2)]:
        idx = p.index[p.pos_group == grp]
        if len(idx):
            prob += pulp.lpSum(x[i] for i in idx) >= lo
    # supermax fragility: no player > 35% of cap
    for i in p.index:
        if p.loc[i, "salary_m"] * 1e6 > 0.35 * cap:
            prob += x[i] == 0
 
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    chosen = [i for i in p.index if x[i].value() == 1]
    sel = p.loc[chosen].copy()
    return sel, sel[val_col].sum(), sel["salary_m"].sum()
 
 
def trade_is_legal(out_salary_m, in_salary_m, over_cap=True):
    """Simplified CBA salary-matching check for a one-for-one trade.
 
    For a team OVER the cap (the common case), incoming salary must fit within a
    band of outgoing salary. The real CBA uses tiered bands (and 2023 CBA aprons
    add further restrictions); we approximate the most-used tier: incoming <=
    125% of outgoing + $250K. A team UNDER the cap can absorb salary freely up to
    its room, so we return legal there. This is a deliberate, documented
    simplification — it captures whether the money plausibly matches, not the full
    apron/exception machinery (a stated v3 item).
 
    Returns (legal: bool, reason: str).
    """
    if not over_cap:
        return True, "team under cap — can absorb salary into room"
    out_amt = out_salary_m * 1e6
    in_amt = in_salary_m * 1e6
    allowed = 1.25 * out_amt + 250_000
    if in_amt <= allowed:
        return True, f"incoming ${in_salary_m:.1f}M within match band of ${allowed/1e6:.1f}M"
    short = (in_amt - allowed) / 1e6
    return False, (f"incoming ${in_salary_m:.1f}M exceeds match band "
                   f"(${allowed/1e6:.1f}M) by ${short:.1f}M — needs more outgoing salary")
 
 
# ----------------------------------------------------------------------
def main():
    global OUT
    ap = argparse.ArgumentParser(description="Forecast, value, and optimize NBA rosters.")
    ap.add_argument("--out", default=OUT, help="folder with clean_roster.csv (also output dir)")
    args = ap.parse_args()
    OUT = args.out
 
    df = load()
    train = build_training_table(df)
    models = train_quantile_models(train)
    print("backtest:", evaluate(train, models))
    print(evaluate_multistep(df))
 
    # calibrate the band-widening factors against held-out coverage, and fit how
    # the supporting rate stats co-move with BPM for the roll-forward
    band_widen, band_report = calibrate_band_widening(train, df=df)
    print(band_report)
    sens = _estimate_stat_sensitivities(train)
    print("stat sensitivities d(stat)/d(BPM):",
          {k: round(v, 3) for k, v in sens.items()})
 
    forecasts = forecast_three_seasons(df, models, band_widen=band_widen,
                                       sensitivities=sens)
    valued = value_players(forecasts, verbose=True)
 
    forecasts.to_csv(f"{OUT}/forecasts.csv", index=False)
    valued.to_csv(f"{OUT}/valuations.csv", index=False)
 
    print(f"forecasted players: {len(forecasts):,}")
 
    # demo: optimize a real team's roster against team + free-agent pool
    team_abbr = "GSW"
    sample_team = valued[valued.Team == team_abbr]
    # build a pool big enough to satisfy the 13-man + position-minimum constraints
    pool = (pd.concat([sample_team, valued.nlargest(120, "bpm_t1_p50")])
              .drop_duplicates("name_key"))
    if len(pool) >= 13:
        sel, war, spend = optimize_roster(pool)
        if len(sel) == 13:
            print(f"\n{team_abbr} optimal (upside): {len(sel)} players, "
                  f"proj BPM sum {war:.1f}, spend ${spend:.0f}M")
        else:
            print(f"\n{team_abbr} optimizer returned {len(sel)} players "
                  f"(constraints infeasible for this pool)")
            
    # top overvalued findings (for the README key findings)
    ov = valued[valued.valuation_flag == "Overvalued"].copy()
    ov["overpay_ratio"] = ov["salary_m"] / (ov["comp_dollar_per_value"]
                                            * ov["current_bpm"].clip(lower=0.1))
    print("\nsample overvalued flags:")
    print(ov.nlargest(5, "salary_m")[["Player", "Team", "Age", "salary_m",
          "current_bpm", "valuation_flag"]].to_string(index=False))
 
 
if __name__ == "__main__":
    main()
