"""
Parquet Capital — Cross-MODEL robustness (additive module)
==========================================================
The VORP check in multi_target.py tests robustness to the forecast TARGET, but
both tracks share the same GradientBoosting machinery — so a systematic bias in
that model family would be inherited by both, and the agreement would look like
signal when it is really shared error. This module closes that gap by adding a
forecaster from a genuinely DIFFERENT model class and asking whether the value
verdicts survive the change of model, not just the change of target.
 
The second forecaster is deliberately the OPPOSITE kind of model from the GBM:
 
  * `AgingCurveForecaster` — no machine learning at all. It projects next-season
    BPM as `current_BPM + position/age aging delta`, using exactly the structural
    aging lookup models.build_aging_lookup already computes. It has no fitted
    feature weights, cannot overfit the feature set, and carries none of the GBM's
    inductive bias. If a contract reads Overvalued under BOTH a 200-tree gradient-
    boosted model AND a parameter-free aging-curve roll-forward, the verdict is
    not an artifact of either model's assumptions.
 
We then report, over players both models price decisively, the same agreement /
contradiction / kappa triple multi_target.py reports for targets — but now the
axis of disagreement is MODEL CLASS. High agreement here is the stronger claim:
the value signal is robust to the modeling approach itself.
 
Nothing here mutates production. It reuses models.value_players unchanged, only
swapping in the aging-curve point forecast where the GBM median normally feeds.
 
Run:  python model_ensemble.py
"""
import sys
import argparse
import numpy as np
import pandas as pd
 
import models as M
 
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass
 
_ORDER = {"Undervalued": 0, "Fair Value": 1, "Overvalued": 2}
_DECISIVE = ("Overvalued (multi-yr)", "Fair Value (multi-yr)", "Undervalued (multi-yr)")
_BUCKET = {"Overvalued (multi-yr)": "Overvalued",
           "Fair Value (multi-yr)": "Fair Value",
           "Undervalued (multi-yr)": "Undervalued"}
 
 
class AgingCurveForecaster:
    """Parameter-free BPM forecaster: next BPM = current BPM + aging delta for
    the player's (position, age) cell. The aging lookup is the same structural
    prior the production roll-forward uses to refresh its aging signal, so this
    is a real, defensible model — just one with zero fitted feature weights and a
    completely different inductive bias from the gradient-boosted trees.
 
    Exposes the SAME forecast columns models.value_players consumes (current_bpm
    plus bpm_t1/t2/t3_p50 medians) so it is a drop-in second opinion. Bands are
    intentionally omitted — this model exists to second-guess the POINT verdict,
    not to re-derive uncertainty."""
 
    def __init__(self, df, exclude_latest_in_aging=False):
        self.lut = M.build_aging_lookup(df, exclude_latest=exclude_latest_in_aging)
 
    def _delta(self, pos, age):
        # fall back to the nearest aging cell, then to zero (flat) if the curve
        # has no entry for this position/age — an honest "no information" prior.
        d = self.lut.get((pos, age))
        if d is not None:
            return d
        for da in (1, -1, 2, -2):
            d = self.lut.get((pos, age + da))
            if d is not None:
                return d
        return 0.0
 
    def forecast(self, df):
        """One row per current player with current_bpm and 3-season median BPM
        projections, mirroring models.forecast_three_seasons' output schema (the
        subset value_players needs). Rolls the aging delta forward each season."""
        latest = df.sort_values("season").groupby("name_key").tail(1).copy()
        latest = latest.dropna(subset=["BPM", "Age", "pos_group"])
        rows = []
        for _, r in latest.iterrows():
            pos, age, bpm = r["pos_group"], float(r["Age"]), float(r["BPM"])
            rec = {"name_key": r["name_key"], "Player": r["Player"],
                   "Team": r["Team"], "pos_group": pos, "Age": age,
                   "salary_m": r.get("salary_m", np.nan), "current_bpm": bpm,
                   "injury_risk_tier": r.get("injury_risk_tier", "Low"),
                   "contract_total_m": r.get("contract_total_m", np.nan),
                   "contract_aav_m": r.get("contract_aav_m", np.nan),
                   "years_remaining": r.get("years_remaining", np.nan),
                   "dead_cap_m": r.get("dead_cap_m", np.nan)}
            b = bpm
            for step in (1, 2, 3):
                b = b + self._delta(pos, age + step - 1)
                b = float(np.clip(b, -10, 15))
                # no real band; set a nominal symmetric placeholder so downstream
                # code that reads p10/p90 does not break (value_players uses p50)
                rec[f"bpm_t{step}_p10"] = b - 2.0
                rec[f"bpm_t{step}_p50"] = b
                rec[f"bpm_t{step}_p90"] = b + 2.0
            rows.append(rec)
        return pd.DataFrame(rows)
 
 
def _aging_value_flags(df):
    """Run the production comp engine (models.value_players) UNCHANGED on the
    aging-curve forecast, returning name_key -> MULTI-YEAR valuation_flag.
 
    We compare the MULTI-YEAR flag, not the single-season one, on purpose: the
    single-season `valuation_flag` keys off current BPM (which both models share
    by construction), so it cannot distinguish model classes — it would trivially
    agree 100%. The multi-year flag prices the 3-season BPM PROJECTION, which is
    exactly the output the two model classes compute differently. That makes it
    the only honest axis on which a GBM-vs-aging-curve disagreement can show up."""
    fc = AgingCurveForecaster(df).forecast(df)
    valued = M.value_players(fc, verbose=False)
    return dict(zip(valued["name_key"], valued["multiyear_flag"]))
 
 
def aging_bucket_flags(df):
    """Per-player aging-curve verdict in the same bucket space the UI uses
    ('Overvalued' / 'Fair Value' / 'Undervalued'), over players the aging model
    prices decisively. name_key -> bucket. Used by ui_confidence to show a
    cross-MODEL agreement label on the player card, alongside the cross-TARGET
    (VORP) one. Non-decisive players are simply omitted (caller treats missing
    as 'n/a')."""
    raw = _aging_value_flags(df)           # name_key -> multi-yr flag
    return {k: _BUCKET[v] for k, v in raw.items() if v in _BUCKET}
 
 
def cross_model_agreement(df, verbose=True):
    """Compare production GBM valuation flags against the aging-curve model's
    flags from the SAME comp engine. Reports exact / adjacent / contradiction
    rates and chance-corrected kappa over players both models price decisively.
 
    Returns (summary_dict, crosstab_df)."""
    # production GBM flags (multi-year track — the forecast-dependent verdict)
    train = M.build_training_table(df)
    qm = M.train_quantile_models(train)
    band, _ = M.calibrate_band_widening(train, df=df)
    sens = M._estimate_stat_sensitivities(train)
    fc_gbm = M.forecast_three_seasons(df, qm, band_widen=band, sensitivities=sens)
    valued_gbm = M.value_players(fc_gbm)
    gbm_flag = dict(zip(valued_gbm["name_key"], valued_gbm["multiyear_flag"]))
 
    # aging-curve flags (different model class, same comp engine)
    aging_flag = _aging_value_flags(df)
 
    keys = [k for k in gbm_flag
            if gbm_flag.get(k) in _DECISIVE and aging_flag.get(k) in _DECISIVE]
    if not keys:
        if verbose:
            print("no players priced decisively by BOTH models")
        return {"n": 0, "agreement": np.nan}, pd.DataFrame()
 
    g = [_BUCKET[gbm_flag[k]] for k in keys]
    a = [_BUCKET[aging_flag[k]] for k in keys]
    buckets = ("Overvalued", "Fair Value", "Undervalued")
    cross = pd.crosstab(pd.Series(g, name="GBM_flag"),
                        pd.Series(a, name="Aging_flag")).reindex(
        index=buckets, columns=buckets).fillna(0).astype(int)
 
    n = len(keys)
    exact = sum(1 for x, y in zip(g, a) if x == y)
    contradiction = sum(1 for x, y in zip(g, a)
                        if abs(_ORDER[x] - _ORDER[y]) == 2)
    adjacent = n - exact - contradiction
 
    from collections import Counter
    cg, ca = Counter(g), Counter(a)
    po = exact / n
    pe = sum((cg[c] / n) * (ca[c] / n) for c in buckets)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 1e-9 else np.nan
 
    summary = {"n": n, "agreement": po, "adjacent": adjacent / n,
               "contradiction": contradiction / n, "kappa": kappa}
 
    if verbose:
        print("\n=== Cross-MODEL valuation agreement (GBM vs aging-curve) ===")
        print(f"players priced decisively by BOTH model classes: {n}")
        print(cross.to_string())
        print(f"\nexact agreement      : {po:.1%}")
        print(f"adjacent (off by one): {adjacent / n:.1%}")
        print(f"direct contradiction : {contradiction / n:.1%}  "
              f"(Overvalued vs Undervalued)")
        print(f"chance-corrected kappa: {kappa:.3f}")
        if po >= 0.6 and contradiction / n <= 0.05:
            verdict = ("ROBUST TO MODEL CLASS: a 200-tree gradient-boosted model "
                       "and a parameter-free aging curve largely agree — the value "
                       "signal is not an artifact of the GBM's inductive bias.")
        elif contradiction / n > 0.15:
            verdict = ("MODEL-SENSITIVE: the two model classes contradict often — "
                       "verdicts depend on the modeling approach and should be "
                       "reported with that caveat.")
        else:
            verdict = ("MIXED: broad cross-model agreement with some sensitivity — "
                       "treat single-bucket calls near a threshold as soft.")
        print(f"verdict: {verdict}")
    return summary, cross
 
 
def main():
    ap = argparse.ArgumentParser(description="GBM-vs-aging-curve valuation robustness.")
    ap.add_argument("--out", default=M.OUT, help="folder with clean_roster.csv")
    args = ap.parse_args()
    M.OUT = args.out
    df = M.load()
    cross_model_agreement(df)
 
 
if __name__ == "__main__":
    main()