
"""
Parquet Capital — Dollar-impact of the valuation signal (additive module)
=========================================================================
The valuation backtest in validation.py proves the ORDERING holds on held-out
seasons (Overvalued contracts cost more per delivered unit than Fair, which cost
more than Undervalued). This module restates that ordering on a dollar scale, so
the price-efficiency gap is legible in the units a front office thinks in.
 
It re-runs the exact same out-of-sample valuation backtest (training only on
seasons < S, scoring realized BPM at S+1), takes the Fair bucket's realized
dollars-per-value as a reference market rate, and for each Overvalued contract
measures the PREMIUM it paid over that rate for the production it actually
delivered.
 
==============================  READ THIS  ==================================
What this figure IS and IS NOT — stated up front because the aggregate is easy
to misread:
 
  * It is an ILLUSTRATIVE restatement of the held-out price-efficiency ordering
    in dollars. The honest, defensible headline is the PER-CONTRACT median
    premium — a typical Overvalued contract paid ~$X.XM over the Fair rate for
    what it delivered.
 
  * It is NOT a forecast of recoverable dollars, and the aggregate total is NOT
    "money a front office would have saved." Two reasons it overstates a real
    saving:
      (1) SELECTION / MEAN-REVERSION. The Overvalued bucket is selected partly
          for players who were likely to regress. Some of the measured premium
          is that regression playing out — which the flag SORTED for but did not
          uniquely PREDICT. Crediting the engine with the full gap conflates
          "ranked correctly" with "caused the saving."
      (2) NO COUNTERFACTUAL MARKET. The Fair-bucket median is a reference rate,
          not a price the team could actually have re-signed every player at.
          Real rosters can't swap each overpaid star for a median-priced
          equivalent at will.
 
  So: report the per-contract median as the result; treat the aggregate as a
  gross upper-bound illustration with the two caveats above attached, never as a
  bottom-line saving. The code prints it that way on purpose.
============================================================================
 
Nothing here mutates production. It reuses validation.backtest_valuations' exact
out-of-sample construction so the dollar figure can never drift from the ordering
result the rest of the framework reports.
 
Run:  python dollar_impact.py
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
 
 
def _backtest_rows(df, eval_seasons=2):
    """Strict out-of-sample flag + realized-outcome rows, mirroring
    validation.backtest_valuations exactly (train on seasons < S, realize at
    S+1). Returns a tidy frame: one row per evaluable player-season with the
    flag computed as-of S and the realized value/$ at S+1."""
    seasons = np.sort(df["season"].unique())
    if len(seasons) < eval_seasons + 2:
        return pd.DataFrame()
    eval_set = seasons[-(eval_seasons + 1):-1]
    truth = df.set_index(["name_key", "season"])["BPM"].to_dict()
 
    rows = []
    for S in eval_set:
        hist = df[df["season"] <= S].copy()
        train = M.build_training_table(hist)
        if len(train) < 100:
            continue
        qm = M.train_quantile_models(train)
        band, _ = M.calibrate_band_widening(train)
        sens = M._estimate_stat_sensitivities(train)
        fc = M.forecast_three_seasons(hist, qm, band_widen=band, sensitivities=sens)
        valued = M.value_players(fc, verbose=False)
        for _, r in valued.iterrows():
            nxt = truth.get((r["name_key"], S + 1))
            if nxt is None or np.isnan(nxt):
                continue
            if pd.isna(r.get("salary_m")) or r["salary_m"] <= 0:
                continue
            realized = float(M._value_score(nxt))
            if realized <= 0:
                continue
            rows.append(dict(season=int(S), name_key=r["name_key"],
                             Player=r.get("Player", r["name_key"]),
                             flag=r["valuation_flag"], salary_m=float(r["salary_m"]),
                             realized_value=realized,
                             realized_dpv=float(r["salary_m"]) / realized))
    return pd.DataFrame(rows)
 
 
def dollar_impact(df, eval_seasons=2, verbose=True):
    """Restate the held-out valuation ordering on a dollar scale.
 
    Logic: the Fair-bucket median realized $/value is a REFERENCE market rate for
    delivered production. Every Overvalued contract paid its realized output at a
    higher rate; the EXCESS over that reference (its salary, minus what the same
    realized production would have cost at the Fair rate) is the per-contract
    PREMIUM. We report it per-contract (the defensible headline) and as a gross
    aggregate (an illustrative upper bound only — see the module docstring's
    selection / no-counterfactual caveats; the aggregate is NOT a recoverable
    saving).
 
    This is deliberately conservative in one direction — it credits each
    Overvalued contract with its FULL realized production at the fair price, so
    the engine is charged nothing for the player's actual output, only the
    premium over market is measured. It is NOT conservative about selection: the
    Overvalued bucket is chosen partly for players who were going to regress, so
    part of the premium is mean-reversion the flag sorted for rather than
    uniquely predicted. Returns (summary_dict, per_contract_df)."""
    bt = _backtest_rows(df, eval_seasons=eval_seasons)
    if bt.empty:
        if verbose:
            print("dollar-impact: no evaluable player-seasons "
                  "(need >= eval_seasons+2 seasons of history)")
        return {"n_overvalued": 0, "avoidable_m": np.nan}, pd.DataFrame()
 
    fair = bt[bt["flag"] == "Fair Value"]
    ov = bt[bt["flag"] == "Overvalued"].copy()
    uv = bt[bt["flag"] == "Undervalued"]
    if fair.empty or ov.empty:
        if verbose:
            print("dollar-impact: need both Fair and Overvalued buckets populated "
                  f"(fair n={len(fair)}, overvalued n={len(ov)})")
        return {"n_overvalued": len(ov), "avoidable_m": np.nan}, pd.DataFrame()
 
    fair_rate = float(fair["realized_dpv"].median())   # $M per realized value point
 
    # fair-priced cost of each Overvalued contract's ACTUAL realized output
    ov["fair_cost_m"] = ov["realized_value"] * fair_rate
    # avoidable premium = what they were paid minus the fair price of what they gave
    ov["avoidable_m"] = (ov["salary_m"] - ov["fair_cost_m"]).clip(lower=0)
 
    total_avoidable = float(ov["avoidable_m"].sum())
    total_salary = float(ov["salary_m"].sum())
    n_ov = len(ov)
    per_contract = float(ov["avoidable_m"].median())
 
    ov_rate = float(ov["realized_dpv"].median())
    uv_rate = float(uv["realized_dpv"].median()) if len(uv) else np.nan
 
    summary = {
        "n_overvalued": n_ov,
        "fair_rate_m_per_value": fair_rate,
        "overvalued_rate_m_per_value": ov_rate,
        "undervalued_rate_m_per_value": uv_rate,
        "total_overvalued_salary_m": total_salary,
        "avoidable_m": total_avoidable,
        "avoidable_pct_of_overvalued_salary": (total_avoidable / total_salary
                                               if total_salary else np.nan),
        "median_avoidable_per_contract_m": per_contract,
    }
 
    if verbose:
        print("\n=== Dollar-impact of the Overvalued signal (held-out seasons) ===")
        print(f"market (Fair-bucket) realized rate : ${fair_rate:.2f}M per value point")
        print(f"Overvalued realized rate           : ${ov_rate:.2f}M per value point")
        if not np.isnan(uv_rate):
            print(f"Undervalued realized rate          : ${uv_rate:.2f}M per value point")
        print(f"\nflagged-Overvalued contracts       : {n_ov}")
        print(f"total salary on those contracts    : ${total_salary:.1f}M")
        print(f"\nHEADLINE (per-contract, defensible):")
        print(f"  median premium over market rate  : ${per_contract:.2f}M per contract")
        print(f"  i.e. a typical Overvalued deal paid ~${per_contract:.1f}M more than the "
              f"Fair rate\n  for the production it actually delivered.")
        print(f"\nillustrative gross aggregate       : ${total_avoidable:.1f}M "
              f"({summary['avoidable_pct_of_overvalued_salary']:.0%} of their salary)")
        print("  CAVEAT: this aggregate is an UPPER-BOUND illustration, NOT a recoverable")
        print("  saving. The Overvalued bucket is selected partly for players who were")
        print("  going to regress, so some of this premium is mean-reversion the flag")
        print("  SORTED for rather than uniquely predicted; and the Fair-bucket rate is a")
        print("  reference, not a price every player could have been re-signed at. Read")
        print("  the per-contract median above as the result; treat this total as scale,")
        print("  not as dollars a front office would have banked.")
        top = ov.nlargest(min(8, n_ov), "avoidable_m")[
            ["season", "Player", "salary_m", "realized_value",
             "fair_cost_m", "avoidable_m"]]
        print("\nlargest premiums over market flagged (illustrative):")
        print(top.to_string(index=False,
              float_format=lambda x: f"{x:.2f}"))
    return summary, ov.sort_values("avoidable_m", ascending=False)
 
 
def main():
    ap = argparse.ArgumentParser(description="Dollar-impact of the valuation signal.")
    ap.add_argument("--out", default=M.OUT, help="folder with clean_roster.csv")
    ap.add_argument("--eval-seasons", type=int, default=2)
    args = ap.parse_args()
    M.OUT = args.out
    df = M.load()
    dollar_impact(df, eval_seasons=args.eval_seasons)
 
 
if __name__ == "__main__":
    main()
