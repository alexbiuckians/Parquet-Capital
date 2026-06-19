"""
Parquet Capital - Multi-target robustness check (additive module)
=================================================================
The production valuation chain rides entirely on a single forecast target, BPM.
A single noisy advanced stat (t+1 MAE is ~2 BPM, large relative to the spread of
most rotation players) carrying every Overvalued/Undervalued call is the deepest
limitation of the framework. This module tests ROBUSTNESS to that choice without
touching production: it re-runs the exact same forecast -> value_score -> comp
pipeline on a SECOND independent target (VORP), then reports how often the two
targets AGREE on each contract's verdict.

Design:
  * Reuses models.py building blocks (build_training_table mechanics,
    GradientBoostingRegressor quantile setup, the comp logic in value_players)
    so the VORP track is the same machinery pointed at a different column - it is
    a genuine replication, not a parallel reimplementation that could flatter the
    result.
  * VORP has its own replacement anchor: a replacement-level player is ~0.0 VORP
    by construction (that is what "replacement" means in VORP), versus BPM's
    conventional -2.0. _vorp_value_score mirrors models._value_score with that
    anchor.
  * The headline output is the AGREEMENT RATE: of players both targets price,
    how often do they land the same Overvalued/Fair/Undervalued bucket, and how
    often do they directly contradict (one Overvalued, the other Undervalued)?
    High agreement => the flags reflect a real value signal, not a BPM artifact.
    Frequent contradiction => the verdict is target-dependent and should be
    reported with that caveat.

Run:  python multi_target.py
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

import models as M

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

FEATURES = M.FEATURES

# VORP replacement anchor: a replacement-level player is ~0 VORP by definition.
VORP_REPLACEMENT = 0.0
VORP_VALUE_FLOOR = 0.5            # mirror models.VALUE_FLOOR on the VORP scale
VORP_MIN_VALUE_FOR_PRICING = 1.0


def _vorp_value_score(vorp):
    """VORP analogue of models._value_score: replacement-anchored, strictly
    positive, monotone. VORP == VORP_REPLACEMENT -> floor; each VORP point above
    adds 1.0."""
    return VORP_VALUE_FLOOR + np.maximum(np.asarray(vorp) - VORP_REPLACEMENT, 0.0)


def _build_vorp_training_table(df):
    """Same shift-by-one construction as models.build_training_table, but the
    target is next-season VORP instead of next-season BPM."""
    d = df.sort_values(["name_key", "season"]).copy()
    d["target_vorp"] = d.groupby("name_key")["VORP"].shift(-1)
    d["next_season"] = d.groupby("name_key")["season"].shift(-1)
    d = d[d["next_season"] == d["season"] + 1]
    d = d.dropna(subset=FEATURES + ["target_vorp"])
    return d


def _train_vorp_median(train):
    m = GradientBoostingRegressor(loss="quantile", alpha=0.5, n_estimators=200,
                                  max_depth=3, learning_rate=0.05,
                                  subsample=0.8, random_state=42)
    m.fit(train[FEATURES], train["target_vorp"])
    return m


def _forecast_vorp_t1(df, model):
    """One-season-ahead VORP forecast for each player's latest season. We only
    need t+1 here: the agreement check is on the current contract verdict, which
    the single-season comp flag already keys off current production."""
    latest = df.sort_values("season").groupby("name_key").tail(1).copy()
    latest = latest.dropna(subset=FEATURES)
    X = latest[FEATURES]
    latest = latest.assign(vorp_t1_p50=model.predict(X))
    return latest[["name_key", "vorp_t1_p50", "VORP"]]


def _vorp_value_flags(df, fc_vorp):
    """Run the SAME comp structure models.value_players uses, but on the VORP
    value score. Returns a name_key -> flag mapping over priceable players.

    We replicate the comp loop (rather than call value_players, which is hard-
    wired to current_bpm) but keep every threshold identical: same age/value
    tiers, same MIN_COMPS, same 1.30/0.70 ratio cutoffs, same salary join. The
    only substitution is the value score's underlying stat."""
    latest = df.sort_values("season").groupby("name_key").tail(1).copy()
    base = latest.merge(fc_vorp[["name_key", "vorp_t1_p50"]], on="name_key", how="inner")
    base["value_score"] = _vorp_value_score(base["VORP"])
    base["priceable"] = ((base["salary_m"] > 0)
                         & (base["value_score"] >= VORP_MIN_VALUE_FOR_PRICING))
    base["dollar_per_value"] = np.where(
        base["priceable"], base["salary_m"] / base["value_score"], np.nan)
    pool = base[base["dollar_per_value"].notna()]

    if len(pool):
        max_tier_salary = pool["salary_m"].quantile(M.MAX_TIER_SALARY_PCTL)
        elite_v = pool["value_score"].quantile(M.ELITE_BPM_PCTL)
    else:
        max_tier_salary, elite_v = np.inf, np.inf

    TIERS = [(2, 2), (3, 3), (4, 4), (5, 6), (6, None)]
    MIN_COMPS = 3
    flags = {}
    for _, r in base.iterrows():
        if r["salary_m"] <= 0 or pd.isna(r["salary_m"]):
            flags[r["name_key"]] = "Unrated (no salary)"; continue
        if not r["priceable"]:
            flags[r["name_key"]] = "Below replacement - not priced"; continue
        if r["salary_m"] >= max_tier_salary and r["value_score"] >= elite_v:
            flags[r["name_key"]] = "Elite (max-tier) - ceiling-capped"; continue
        ratio = np.nan
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
        if pd.isna(ratio):
            flags[r["name_key"]] = "Insufficient comps"
        elif ratio > 1.30:
            flags[r["name_key"]] = "Overvalued"
        elif ratio < 0.70:
            flags[r["name_key"]] = "Undervalued"
        else:
            flags[r["name_key"]] = "Fair Value"
    return flags


_DECISIVE = ("Overvalued", "Fair Value", "Undervalued")


def cross_target_agreement(df, verbose=True):
    """Forecast VORP in parallel, flag valuations off it with the same comp
    machinery, and measure agreement with the production BPM flags.

    Reports, over players BOTH targets give a decisive verdict (Overvalued /
    Fair / Undervalued):
      * exact agreement rate (same bucket),
      * adjacent disagreement (off by one bucket, e.g. Fair vs Overvalued),
      * direct contradiction (Overvalued vs Undervalued) - the worst case,
      * Cohen's-kappa-style chance-corrected agreement.
    Returns (summary_dict, crosstab_df)."""
    # --- production BPM flags ---
    train_bpm = M.build_training_table(df)
    qm = M.train_quantile_models(train_bpm)
    band, _ = M.calibrate_band_widening(train_bpm, df=df)
    sens = M._estimate_stat_sensitivities(train_bpm)
    fc_bpm = M.forecast_three_seasons(df, qm, band_widen=band, sensitivities=sens)
    valued_bpm = M.value_players(fc_bpm)
    bpm_flag = dict(zip(valued_bpm["name_key"], valued_bpm["valuation_flag"]))

    # --- parallel VORP flags ---
    train_v = _build_vorp_training_table(df)
    vmodel = _train_vorp_median(train_v)
    fc_v = _forecast_vorp_t1(df, vmodel)
    vorp_flag = _vorp_value_flags(df, fc_v)

    # --- align on players both targets priced decisively ---
    keys = [k for k in bpm_flag
            if bpm_flag.get(k) in _DECISIVE and vorp_flag.get(k) in _DECISIVE]
    if not keys:
        msg = "no players received a decisive verdict from BOTH targets"
        if verbose:
            print(msg)
        return {"n": 0, "agreement": np.nan}, pd.DataFrame()

    b = [bpm_flag[k] for k in keys]
    v = [vorp_flag[k] for k in keys]
    cross = pd.crosstab(pd.Series(b, name="BPM_flag"),
                        pd.Series(v, name="VORP_flag")).reindex(
        index=_DECISIVE, columns=_DECISIVE).fillna(0).astype(int)

    n = len(keys)
    exact = sum(1 for x, y in zip(b, v) if x == y)
    order = {"Undervalued": 0, "Fair Value": 1, "Overvalued": 2}
    contradiction = sum(1 for x, y in zip(b, v)
                        if abs(order[x] - order[y]) == 2)
    adjacent = n - exact - contradiction

    # chance-corrected (Cohen's kappa)
    po = exact / n
    from collections import Counter
    cb, cv = Counter(b), Counter(v)
    pe = sum((cb[c] / n) * (cv[c] / n) for c in _DECISIVE)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 1e-9 else np.nan

    summary = {"n": n, "agreement": po, "adjacent": adjacent / n,
               "contradiction": contradiction / n, "kappa": kappa}

    if verbose:
        print("\n=== Cross-target valuation agreement (BPM vs VORP) ===")
        print(f"players priced decisively by BOTH targets: {n}")
        print(cross.to_string())
        print(f"\nexact agreement      : {po:.1%}")
        print(f"adjacent (off by one): {adjacent / n:.1%}")
        print(f"direct contradiction : {contradiction / n:.1%}  "
              f"(Overvalued vs Undervalued)")
        print(f"chance-corrected kappa: {kappa:.3f}")
        if po >= 0.6 and contradiction / n <= 0.05:
            verdict = ("ROBUST: the two independent targets largely agree and "
                       "rarely contradict - the valuation signal is not a BPM "
                       "artifact.")
        elif contradiction / n > 0.15:
            verdict = ("FRAGILE: the targets contradict often - verdicts are "
                       "target-dependent and should be reported with that caveat.")
        else:
            verdict = ("MIXED: broad agreement with some target sensitivity - "
                       "treat single-bucket calls near a threshold as soft.")
        print(f"verdict: {verdict}")
    return summary, cross


def main():
    ap = argparse.ArgumentParser(description="BPM-vs-VORP valuation robustness.")
    ap.add_argument("--out", default=M.OUT, help="folder with clean_roster.csv")
    args = ap.parse_args()
    M.OUT = args.out
    df = M.load()
    cross_target_agreement(df)


if __name__ == "__main__":
    main()
