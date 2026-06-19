"""
Parquet Capital — Trajectory feature upgrade + honest A/B (additive module)
===========================================================================
The deepest limitation of the forecaster is that it sees each player-season as a
LEVEL: current BPM, current age, current rate stats. It has no idea whether a
player is on the way UP or DOWN — a 24-year-old who jumped +3 BPM last year and a
30-year-old who fell -3 BPM look identical to the model if their current values
match. Momentum and durability are exactly the signal a level-only model throws
away, and they are cheap to recover from the panel we already have.

This module:
  1. Adds trajectory / durability / injury-recency features per player-season,
     computed with STRICTLY PAST information (no leakage — every feature at
     season S uses only seasons <= S):
       * bpm_slope_2y      — least-squares slope of BPM over the last up-to-3 seasons
       * bpm_delta_1y      — last single-season BPM change (raw momentum)
       * vorp_slope_2y     — same momentum signal on the second target
       * usg_delta_1y      — role/usage change (rising vs shrinking role)
       * seasons_played    — experience (proxy for established vs volatile)
       * injury_recency    — severity weighted toward the MOST RECENT season,
                             not a flat 3-yr count (a fresh major injury matters
                             more than one three years ago)
  2. Re-runs the SAME expanding-window CV used in validation.tune_forecaster to
     measure whether the enriched feature set actually lowers out-of-sample MAE
     and raises directional accuracy vs. the production FEATURES — reported as an
     honest A/B with the noise floor stated, so a non-improvement is called a
     non-improvement.

It writes clean_roster_plus.csv (original + new columns) so the rest of the
pipeline can adopt the features by pointing FEATURES at the richer set, but it
does NOT mutate production or the on-disk roster unless you ask it to.

Run:  python feature_upgrade.py            # build features + A/B report
      python feature_upgrade.py --write    # also write clean_roster_plus.csv
"""
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

NEW_FEATURES = ["bpm_slope_2y", "bpm_delta_1y", "vorp_slope_2y",
                "usg_delta_1y", "seasons_played", "injury_recency"]


def _slope(vals):
    """Least-squares slope of a short series vs. 0,1,2,...; 0.0 if <2 points.
    Used for momentum features — sign and magnitude of recent trend."""
    y = np.asarray([v for v in vals if pd.notna(v)], dtype=float)
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    x -= x.mean()
    denom = (x * x).sum()
    return float((x * (y - y.mean())).sum() / denom) if denom > 0 else 0.0


def add_trajectory_features(df):
    """Return df with NEW_FEATURES added. Every feature at season S is computed
    from that player's seasons <= S only (expanding history), so it is safe to
    use as a predictor of S+1 with no leakage.

    Implementation: sort by (player, season), then for each player walk their
    seasons in order accumulating history. This guarantees the window for row S
    never includes any season after S."""
    d = df.sort_values(["name_key", "season"]).copy()
    out = {f: np.zeros(len(d)) for f in NEW_FEATURES}
    # map row position for assignment
    d = d.reset_index(drop=True)

    for _, idx in d.groupby("name_key").groups.items():
        idx = list(idx)
        bpm_hist, vorp_hist, usg_hist = [], [], []
        sev_hist = []
        for n, i in enumerate(idx):
            row = d.loc[i]
            # features use history UP TO AND INCLUDING the current season's
            # level, but only PAST seasons for deltas/slopes (no future)
            bpm_hist.append(row.get("BPM", np.nan))
            vorp_hist.append(row.get("VORP", np.nan))
            usg_hist.append(row.get("USG%", np.nan))
            sev_hist.append(row.get("severity_weighted_events", 0) or 0)

            last3_bpm = bpm_hist[-3:]
            last3_vorp = vorp_hist[-3:]
            out["bpm_slope_2y"][i] = _slope(last3_bpm)
            out["vorp_slope_2y"][i] = _slope(last3_vorp)
            out["bpm_delta_1y"][i] = (
                float(bpm_hist[-1] - bpm_hist[-2])
                if len(bpm_hist) >= 2 and pd.notna(bpm_hist[-1])
                and pd.notna(bpm_hist[-2]) else 0.0)
            out["usg_delta_1y"][i] = (
                float(usg_hist[-1] - usg_hist[-2])
                if len(usg_hist) >= 2 and pd.notna(usg_hist[-1])
                and pd.notna(usg_hist[-2]) else 0.0)
            out["seasons_played"][i] = n + 1
            # recency-weighted injury severity: most recent season weight 1.0,
            # prior 0.5, prior 0.25 — a fresh major injury dominates an old one
            w = [0.25, 0.5, 1.0][-len(sev_hist[-3:]):]
            sv = sev_hist[-3:]
            out["injury_recency"][i] = float(np.dot(w, sv) / sum(w)) if sv else 0.0

    for f in NEW_FEATURES:
        d[f] = out[f]
    return d


def _build_training(df, feature_list):
    d = df.sort_values(["name_key", "season"]).copy()
    d["target_bpm"] = d.groupby("name_key")["BPM"].shift(-1)
    d["next_season"] = d.groupby("name_key")["season"].shift(-1)
    d = d[d["next_season"] == d["season"] + 1]
    d = d.dropna(subset=feature_list + ["target_bpm"])
    return d


def _expanding_splits(train, min_train_seasons=3):
    seasons = np.sort(train["season"].unique())
    for i in range(min_train_seasons, len(seasons)):
        cut = seasons[i]
        tr = train[train["season"] < cut]
        te = train[train["season"] == cut]
        if len(te) >= 20 and len(tr) >= 50:
            yield tr, te


def _cv(train, feature_list):
    """Expanding-window CV MAE + directional accuracy for a given feature set,
    using the production GBM median config so the only thing that varies is the
    feature columns."""
    maes, dirs = [], []
    for tr, te in _expanding_splits(train):
        m = GradientBoostingRegressor(loss="quantile", alpha=0.5, n_estimators=200,
                                      max_depth=3, learning_rate=0.05,
                                      subsample=0.8, random_state=42)
        m.fit(tr[feature_list], tr["target_bpm"])
        p = m.predict(te[feature_list])
        maes.append(np.mean(np.abs(p - te["target_bpm"])))
        dirs.append(np.mean(np.sign(p - te["BPM"]) == np.sign(te["target_bpm"] - te["BPM"])))
    return float(np.mean(maes)), float(np.mean(dirs)), len(maes)


def ab_test(df, verbose=True):
    """Honest A/B: production FEATURES vs FEATURES + NEW_FEATURES, same model,
    same expanding-window CV. Reports whether the enrichment beats the noise
    floor (0.05 BPM, the same threshold validation.py uses for tuning)."""
    enriched = add_trajectory_features(df)
    base_feats = M.FEATURES
    plus_feats = M.FEATURES + NEW_FEATURES

    # build a single training table carrying BOTH feature sets, so the CV rows
    # are identical and the comparison is apples-to-apples
    train = _build_training(enriched, plus_feats)
    if len(train) < 100:
        if verbose:
            print("feature A/B: insufficient training rows")
        return {}

    base_mae, base_dir, nfold = _cv(train, base_feats)
    plus_mae, plus_dir, _ = _cv(train, plus_feats)
    improve = base_mae - plus_mae
    NOISE = 0.05

    summary = {"base_mae": base_mae, "plus_mae": plus_mae,
               "improvement_bpm": improve, "base_dir": base_dir,
               "plus_dir": plus_dir, "folds": nfold}

    if verbose:
        print("\n=== Feature upgrade A/B (expanding-window CV) ===")
        print(f"folds: {nfold}")
        print(f"production features ({len(base_feats)}): "
              f"MAE {base_mae:.3f} | direction {base_dir:.1%}")
        print(f"+ trajectory features ({len(plus_feats)}): "
              f"MAE {plus_mae:.3f} | direction {plus_dir:.1%}")
        print(f"\nMAE change: {improve:+.3f} BPM  "
              f"(direction {plus_dir - base_dir:+.1%})")
        if improve > NOISE:
            print(f"verdict: the trajectory features HELP — {improve:+.3f} BPM beats "
                  f"the {NOISE} BPM noise floor. Adopt by setting "
                  f"models.FEATURES += NEW_FEATURES and rebuilding.")
        elif improve < -NOISE:
            print(f"verdict: the trajectory features HURT on this sample "
                  f"({improve:+.3f} BPM). Do not adopt as-is.")
        else:
            print(f"verdict: NO meaningful change ({improve:+.3f} BPM, within the "
                  f"{NOISE} BPM noise floor). The level-only model already captures "
                  f"most of the signal on THIS data; re-test on the real roster, "
                  f"where momentum/durability vary more than in synthetic data.")
        # feature importances on the full enriched fit, for insight
        full = GradientBoostingRegressor(loss="quantile", alpha=0.5, n_estimators=200,
                                         max_depth=3, learning_rate=0.05,
                                         subsample=0.8, random_state=42)
        full.fit(train[plus_feats], train["target_bpm"])
        imp = sorted(zip(plus_feats, full.feature_importances_),
                     key=lambda t: t[1], reverse=True)
        print("\nfeature importances (enriched model):")
        for f, v in imp:
            tag = "  <- NEW" if f in NEW_FEATURES else ""
            print(f"  {f:18s} {v:.3f}{tag}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Trajectory feature upgrade + A/B.")
    ap.add_argument("--out", default=M.OUT, help="folder with clean_roster.csv")
    ap.add_argument("--write", action="store_true",
                    help="write clean_roster_plus.csv with the new columns")
    args = ap.parse_args()
    M.OUT = args.out
    df = M.load()
    ab_test(df)
    if args.write:
        import os
        enriched = add_trajectory_features(df)
        path = os.path.join(args.out, "clean_roster_plus.csv")
        enriched.to_csv(path, index=False)
        print(f"\nwrote enriched roster -> {path} "
              f"(+{len(NEW_FEATURES)} columns)")


if __name__ == "__main__":
    main()
