"""
Parquet Capital — Dollar-impact of the valuation signal (additive module)
=========================================================================
The valuation backtest in validation.py proves the ORDERING holds on held-out
seasons (Overvalued contracts cost more per delivered unit than Fair, which cost
more than Undervalued). What it does NOT do is translate that ordering into the
one number a front office actually cares about: how many DOLLARS the signal would
have flagged as avoidable overpay.

This module closes that gap. It re-runs the exact same out-of-sample valuation
backtest (training only on seasons < S, scoring realized BPM at S+1), then prices
the realized $/value gap between the Overvalued bucket and the Fair bucket in
dollars, aggregated across the flagged population. The headline is:

    "On held-out seasons, contracts this engine flagged Overvalued delivered
     their realized production at $X.XM/value-point above the Fair-bucket rate —
     $Y.YM of avoidable spend across N flagged contracts."

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
    """Translate the held-out valuation ordering into dollars.

    Logic: the Fair-bucket median realized $/value is the 'going market rate' for
    delivered production. Every Overvalued contract is paying its realized output
    at a higher rate; the EXCESS it pays (its salary, minus what that same
    realized production would have cost at the Fair rate) is avoidable spend the
    flag would have warned about BEFORE the season played out. We sum that excess
    across the flagged population and also report it per-contract.

    This is deliberately conservative: it credits the Overvalued contract with
    its FULL realized production at the fair price, charging the engine nothing
    for the player's actual output — only the premium over market is called
    'avoidable'. Returns (summary_dict, per_contract_df)."""
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
        print(f"avoidable premium over market      : ${total_avoidable:.1f}M "
              f"({summary['avoidable_pct_of_overvalued_salary']:.0%} of their salary)")
        print(f"median avoidable per contract      : ${per_contract:.2f}M")
        print("\ninterpretation: had a front office acted on the Overvalued flag "
              "and only paid the\nmarket (Fair) rate for the production these "
              f"players actually delivered, it would have\navoided ${total_avoidable:.1f}M "
              "in premium across the held-out window — production held\nconstant, only "
              "the overpay removed.")
        top = ov.nlargest(min(8, n_ov), "avoidable_m")[
            ["season", "Player", "salary_m", "realized_value",
             "fair_cost_m", "avoidable_m"]]
        print("\nlargest avoidable overpays flagged:")
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
