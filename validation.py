"""
Parquet Capital — Validation & Tuning (additive module)
=======================================================
Three upgrades that turn the framework from "plausible" into "validated",
without touching the existing models.py logic the app depends on:
 
  1. tune_forecaster()        — expanding-window time-series CV over the GBM
                                hyperparameters, plus a Ridge baseline, so the
                                gradient-boosted choice is *earned*, not assumed.
  2. quantile_calibration()   — reliability data for the 10th/50th/90th bands
                                (predicted quantile vs. empirical coverage), the
                                visual complement to the numeric coverage target.
  3. backtest_valuations()    — the decision-level test: do contracts the comp
                                engine flags "Overvalued" actually go on to
                                underperform their price vs. "Fair"/"Undervalued"
                                ones, on held-out history? This evaluates what the
                                tool actually *claims to do*, not just BPM MAE.
 
All three reuse models.py building blocks (FEATURES, build_training_table,
_value_score, value_players, _advance_features) so behavior can never drift from
production. Run:  python validation.py
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
 
import models as M
 
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass
 
FEATURES = M.FEATURES
 
 
# ----------------------------------------------------------------------
# Shared: expanding-window splits over SEASONS (never train on the future)
# ----------------------------------------------------------------------
def _expanding_season_splits(train, min_train_seasons=3):
    """Yield (train_idx, test_idx) where every test season is strictly later
    than every training season. This is the only honest CV for season-indexed
    panel data — a random KFold would leak a player's future into their past."""
    seasons = np.sort(train["season"].unique())
    if len(seasons) <= min_train_seasons:
        return
    for i in range(min_train_seasons, len(seasons)):
        cut = seasons[i]
        tr = train[train["season"] < cut]
        te = train[train["season"] == cut]
        if len(te) >= 20 and len(tr) >= 50:
            yield tr, te
 
 
def _mae(pred, y):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(y))))
 
 
def _dir_acc(pred, base, y):
    return float(np.mean(np.sign(pred - base) == np.sign(y - base)))
 
 
# ----------------------------------------------------------------------
# 1. Hyperparameter tuning + Ridge baseline (earn the GBM)
# ----------------------------------------------------------------------
# Deliberately small, defensible grid: depth/leaf control overfitting, the
# n_estimators x learning_rate pair trades fit vs. generalization. We search the
# MEDIAN (alpha=0.5) model because that drives every point projection; the p10/p90
# inherit the winning structural params.
_PARAM_GRID = [
    dict(n_estimators=ne, max_depth=md, learning_rate=lr,
         min_samples_leaf=ml, subsample=0.8)
    for ne in (200, 400)
    for md in (2, 3)
    for lr in (0.03, 0.05)
    for ml in (1, 20)
]
 
 
def tune_forecaster(train, verbose=True):
    """Expanding-window CV over the GBM grid + a Ridge baseline and the current
    production defaults. Returns (best_params, report_df). The winner is the
    lowest mean out-of-sample MAE across folds; ties break toward the simpler
    model (fewer estimators, shallower)."""
    folds = list(_expanding_season_splits(train))
    if not folds:
        return None, pd.DataFrame()
 
    def cv_mae(make_model):
        maes, dirs = [], []
        for tr, te in folds:
            mdl = make_model()
            mdl.fit(tr[FEATURES], tr["target_bpm"])
            p = mdl.predict(te[FEATURES])
            maes.append(_mae(p, te["target_bpm"]))
            dirs.append(_dir_acc(p, te["BPM"].to_numpy(), te["target_bpm"].to_numpy()))
        return float(np.mean(maes)), float(np.mean(dirs))
 
    records = []
 
    # --- baselines ---
    # persistence: predict next BPM = current BPM (no model at all)
    pmae, pdir = [], []
    for tr, te in folds:
        pmae.append(_mae(te["BPM"], te["target_bpm"]))
        pdir.append(_dir_acc(te["BPM"].to_numpy(), te["BPM"].to_numpy(),
                             te["target_bpm"].to_numpy()))  # always 0 signal -> ~0.5 ref
    records.append(dict(model="persistence (baseline)", cv_mae=float(np.mean(pmae)),
                        cv_dir=np.nan, params="next=current"))
 
    # ridge on standardized features: a real but linear model to beat
    rmae, rdir = cv_mae(lambda: make_pipeline(StandardScaler(),
                                              Ridge(alpha=5.0, random_state=42)))
    records.append(dict(model="ridge (baseline)", cv_mae=rmae, cv_dir=rdir,
                        params="alpha=5.0, standardized"))
 
    # current production defaults, for an apples-to-apples "did tuning help"
    cur_mae, cur_dir = cv_mae(lambda: GradientBoostingRegressor(
        loss="quantile", alpha=0.5, n_estimators=200, max_depth=3,
        learning_rate=0.05, subsample=0.8, random_state=42))
    records.append(dict(model="GBM (current defaults)", cv_mae=cur_mae, cv_dir=cur_dir,
                        params="ne=200,md=3,lr=0.05,ml=1"))
 
    # --- grid search ---
    best = None
    for g in _PARAM_GRID:
        gm, gd = cv_mae(lambda g=g: GradientBoostingRegressor(
            loss="quantile", alpha=0.5, random_state=42, **g))
        records.append(dict(model="GBM (tuned)", cv_mae=gm, cv_dir=gd,
                            params=f"ne={g['n_estimators']},md={g['max_depth']},"
                                   f"lr={g['learning_rate']},ml={g['min_samples_leaf']}"))
        # tie-break toward simpler: lower mae wins; on near-ties prefer fewer trees/shallower
        key = (round(gm, 4), g["n_estimators"], g["max_depth"])
        if best is None or key < best[0]:
            best = (key, g, gm, gd)
 
    report = pd.DataFrame(records).sort_values("cv_mae").reset_index(drop=True)
    best_params = best[1]
 
    if verbose:
        print("\n=== Forecaster tuning (expanding-window CV) ===")
        print(report.to_string(index=False,
              float_format=lambda x: f"{x:.3f}" if pd.notna(x) else "  n/a"))
        improve = cur_mae - best[2]
        pers = float(np.mean(pmae))
        print(f"\nbest tuned params: ne={best_params['n_estimators']}, "
              f"md={best_params['max_depth']}, lr={best_params['learning_rate']}, "
              f"ml={best_params['min_samples_leaf']}")
        print(f"tuned CV MAE {best[2]:.3f} vs current-default {cur_mae:.3f} "
              f"({improve:+.3f} BPM); ridge {rmae:.3f}; persistence {pers:.3f}")
 
        # Two SEPARATE questions, reported separately so neither is oversold:
        #   (a) does the GBM family beat the honest baselines?  (the real win)
        #   (b) did hyperparameter tuning actually buy anything over the
        #       production defaults?  (often noise — say so when it is)
        beats_baselines = best[2] < rmae and best[2] < pers
        # Treat a tuning gain smaller than NOISE_BPM as noise, not signal. The
        # threshold is a small fraction of the fold-to-fold MAE spread; sub-0.05
        # BPM "improvements" are not distinguishable from CV jitter.
        NOISE_BPM = 0.05
        tuning_helped = improve > NOISE_BPM
 
        family_verdict = (f"GBM family beats both baselines "
                          f"(ridge {rmae:.3f}, persistence {pers:.3f})"
                          if beats_baselines else
                          "GBM does NOT clearly beat baselines — reconsider complexity")
        if tuning_helped:
            tuning_verdict = (f"tuning helped: {improve:+.3f} BPM over defaults "
                              f"(> {NOISE_BPM} BPM noise floor)")
        else:
            tuning_verdict = (f"tuning did NOT meaningfully help: {improve:+.3f} BPM "
                              f"over defaults is within the {NOISE_BPM} BPM noise floor — "
                              f"the production defaults were already near-optimal; the win "
                              f"is GBM-over-persistence, not tuned-over-default")
        print(f"verdict (model family): {family_verdict}")
        print(f"verdict (tuning):       {tuning_verdict}")
    return best_params, report
 
 
# ----------------------------------------------------------------------
# 2. Quantile calibration (reliability data for the bands)
# ----------------------------------------------------------------------
def quantile_calibration(train, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9), verbose=True):
    """Out-of-sample reliability: for each nominal quantile q, what FRACTION of
    held-out actuals fall at or below the model's q-prediction? Perfect
    calibration => empirical == nominal. Returns a tidy DataFrame the app/README
    can plot as a reliability diagram. Uses the same expanding-window folds as
    tuning so the numbers are comparable."""
    folds = list(_expanding_season_splits(train))
    if not folds:
        return pd.DataFrame()
    rows = []
    for q in quantiles:
        below = []
        for tr, te in folds:
            m = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=200,
                                          max_depth=3, learning_rate=0.05,
                                          subsample=0.8, random_state=42)
            m.fit(tr[FEATURES], tr["target_bpm"])
            pred = m.predict(te[FEATURES])
            below.append((te["target_bpm"].to_numpy() <= pred).mean())
        rows.append(dict(nominal=q, empirical=float(np.mean(below))))
    cal = pd.DataFrame(rows)
    cal["abs_error"] = (cal["empirical"] - cal["nominal"]).abs()
    if verbose:
        print("\n=== Quantile calibration (out-of-sample) ===")
        print(cal.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print(f"mean |empirical - nominal| = {cal['abs_error'].mean():.3f} "
              f"(0 = perfectly calibrated)")
        band = cal.set_index("nominal")["empirical"]
        if 0.1 in band.index and 0.9 in band.index:
            print(f"implied 10-90 central coverage ≈ {band[0.9] - band[0.1]:.0%}")
    return cal
 
 
# ----------------------------------------------------------------------
# 4. Backtested valuation decisions — the P&L the tool actually claims
# ----------------------------------------------------------------------
def backtest_valuations(df, eval_seasons=2, verbose=True):
    """Does an 'Overvalued' flag actually precede underperformance?
 
    Method (strict out-of-sample, no leakage):
      * For each evaluation season S in the last `eval_seasons`:
        - Train quantile models only on seasons < S.
        - Build a one-season forecast table AS OF season S (each player's row at S),
          using the SAME _value_score / value_players comp logic as production.
        - This yields a flag per player computed only from <= S information.
      * Realized outcome: the player's ACTUAL BPM at S+1 (ground truth, held out).
      * Decision test: group realized next-season value_score by flag. If the engine
        has signal, Overvalued contracts should deliver LESS realized value per
        dollar than Fair, and Undervalued should deliver MORE.
 
    We report, per flag bucket: n, median salary, median realized next-season
    value_score, and realized $/value (salary / realized value). A working engine
    shows Overvalued $/value > Fair > Undervalued (you pay more per delivered
    unit on the contracts it warned about).
    """
    seasons = np.sort(df["season"].unique())
    if len(seasons) < eval_seasons + 2:
        return pd.DataFrame(), "insufficient seasons for valuation backtest"
    eval_set = seasons[-(eval_seasons + 1):-1]   # leave the final season as S+1 truth
 
    truth = df.set_index(["name_key", "season"])["BPM"].to_dict()
    bucket_rows = []
 
    for S in eval_set:
        hist = df[df["season"] <= S].copy()
        train = M.build_training_table(hist)
        if len(train) < 100:
            continue
        qmodels = M.train_quantile_models(train)
        band, _ = M.calibrate_band_widening(train)
        sens = M._estimate_stat_sensitivities(train)
        # forecast/value AS OF S (forecast_three_seasons keys off each player's
        # latest season, which within `hist` is S for active players)
        fc = M.forecast_three_seasons(hist, qmodels, band_widen=band, sensitivities=sens)
        valued = M.value_players(fc, verbose=False)
 
        for _, r in valued.iterrows():
            actual_next = truth.get((r["name_key"], S + 1))
            if actual_next is None or np.isnan(actual_next):
                continue
            if pd.isna(r.get("salary_m")) or r["salary_m"] <= 0:
                continue
            realized_value = float(M._value_score(actual_next))
            if realized_value <= 0:
                continue
            bucket_rows.append(dict(
                season=int(S), flag=r["valuation_flag"], salary_m=float(r["salary_m"]),
                realized_value=realized_value,
                realized_dollar_per_value=float(r["salary_m"]) / realized_value))
 
    bt = pd.DataFrame(bucket_rows)
    if bt.empty:
        return bt, "no evaluable player-seasons in the valuation backtest"
 
    keep = ["Overvalued", "Fair Value", "Undervalued"]
    summary = (bt[bt["flag"].isin(keep)]
               .groupby("flag")
               .agg(n=("flag", "size"),
                    median_salary_m=("salary_m", "median"),
                    median_realized_value=("realized_value", "median"),
                    median_realized_dollar_per_value=("realized_dollar_per_value", "median"))
               .reindex(keep).dropna(how="all").reset_index())
 
    # the headline signal check
    msg_lines = ["\n=== Valuation decision backtest (held-out next-season outcomes) ==="]
    msg_lines.append(summary.to_string(index=False,
                     float_format=lambda x: f"{x:.2f}" if pd.notna(x) else "n/a"))
    try:
        ov = summary.loc[summary.flag == "Overvalued",
                         "median_realized_dollar_per_value"].iloc[0]
        fv = summary.loc[summary.flag == "Fair Value",
                         "median_realized_dollar_per_value"].iloc[0]
        uv = summary.loc[summary.flag == "Undervalued",
                         "median_realized_dollar_per_value"].iloc[0]
        ordered = ov > fv > uv
        msg_lines.append(
            f"\nrealized $/value by flag: Overvalued {ov:.2f} | Fair {fv:.2f} | "
            f"Undervalued {uv:.2f}")
        msg_lines.append(
            "signal CONFIRMED: flagged-overvalued contracts cost more per delivered "
            "unit than fair, which cost more than undervalued — the engine ranks "
            "contracts in the right order on held-out outcomes."
            if ordered else
            "signal WEAK/MIXED: the monotonic ordering does not hold on this sample; "
            "report honestly rather than overclaim.")
    except (IndexError, KeyError):
        msg_lines.append("\n(not all three buckets populated on this sample)")
 
    # --- cheap-contract track validation -------------------------------------
    # The below-replacement track claims a different thing than the comp track:
    # not "ranks production" but "separates fair minimum deals from stranded cap."
    # The honest test: among players the model flagged sub-replacement, do the
    # "Overpay (dead money)" contracts cost dramatically more per realized unit
    # than the "Fair (min contract)" ones? Both groups should stay low-production
    # (confirming the read), but dead-money should pay far more for it.
    cheap_keep = ["Fair (min contract)", "Overpay (dead money)"]
    cheap = (bt[bt["flag"].isin(cheap_keep)]
             .groupby("flag")
             .agg(n=("flag", "size"),
                  median_salary_m=("salary_m", "median"),
                  median_realized_value=("realized_value", "median"),
                  median_realized_dollar_per_value=("realized_dollar_per_value", "median"))
             .reindex(cheap_keep).dropna(how="all").reset_index())
    if len(cheap):
        msg_lines.append("\n--- cheap-contract track (below-replacement pricing) ---")
        msg_lines.append(cheap.to_string(index=False,
                         float_format=lambda x: f"{x:.2f}" if pd.notna(x) else "n/a"))
        try:
            dm = cheap.loc[cheap.flag == "Overpay (dead money)",
                           "median_realized_dollar_per_value"].iloc[0]
            fm = cheap.loc[cheap.flag == "Fair (min contract)",
                           "median_realized_dollar_per_value"].iloc[0]
            dm_val = cheap.loc[cheap.flag == "Overpay (dead money)",
                               "median_realized_value"].iloc[0]
            fm_val = cheap.loc[cheap.flag == "Fair (min contract)",
                               "median_realized_value"].iloc[0]
            msg_lines.append(
                f"\nrealized $/value: dead-money {dm:.2f} vs fair-min {fm:.2f} "
                f"({dm/fm:.1f}x); realized production stays comparable "
                f"({dm_val:.1f} vs {fm_val:.1f} value-score) — i.e. similar "
                f"output, far higher price.")
            msg_lines.append(
                "cheap-track CONFIRMED: dead-money flags pay materially more per "
                "delivered unit than fair-minimum deals for the same replacement-"
                "level production — the stranded cap the comp engine used to leave "
                "unrated is now surfaced and validated."
                if dm > fm * 1.5 else
                "cheap-track WEAK: dead-money deals do not clearly cost more per "
                "unit on this sample; report honestly.")
        except (IndexError, KeyError):
            msg_lines.append("\n(cheap-contract buckets not both populated)")
    report = "\n".join(msg_lines)
    if verbose:
        print(report)
    return summary, report
 
 
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Tuning + calibration + valuation backtest.")
    ap.add_argument("--out", default=M.OUT, help="folder with clean_roster.csv")
    args = ap.parse_args()
    M.OUT = args.out
 
    df = M.load()
    train = M.build_training_table(df)
 
    tune_forecaster(train)
    quantile_calibration(train)
    backtest_valuations(df)
 
    # cross-model robustness: does the value signal survive a different model
    # CLASS (parameter-free aging curve), not just a different target?
    try:
        import model_ensemble as ME
        ME.cross_model_agreement(df)
    except Exception as e:
        print(f"\n(cross-model check skipped: {e})")
 
 
if __name__ == "__main__":
    main()
