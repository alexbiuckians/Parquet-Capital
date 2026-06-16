
"""
Parquet Capital — UI confidence layer (additive, read-only over production)
===========================================================================
The validation scripts already KNOW how uncertain each call is — the projection
band, the t+1 MAE, and the BPM-vs-VORP agreement all exist. They just live in
console strings the dashboard never shows. This module surfaces them per player
so a flag never appears on screen without the humility that's already computed.
 
It produces columns the app can merge onto `valued`:
 
  * confidence_label  — "High" / "Moderate" / "Low", from the WIDTH of the
    player's own t+1 projection band relative to the roster. A wide band means
    the model itself is unsure; the flag should be read as soft regardless of
    its color.
 
  * vorp_flag / agreement_label — how the SECOND independent target (VORP)
    verdicts this same contract: "Confirmed" (same bucket), "Mixed" (off by
    one), or "Contradicts" (one Overvalued, the other Undervalued). This is the
    single most honest thing the framework can say about a call, and right now
    it's buried in multi_target.py. We reuse that module's exact comp logic so
    the UI and the robustness report can never disagree.
 
Nothing here mutates production. value_players / forecast_three_seasons are
untouched; this only reads their output.
"""
import numpy as np
import pandas as pd
 
import models as M
import multi_target as MT
import model_ensemble as ME
 
 
_ORDER = {"Undervalued": 0, "Fair Value": 1, "Overvalued": 2}
 
 
def attach_confidence(valued: pd.DataFrame) -> pd.DataFrame:
    """Add `confidence_label` from each player's own t+1 band width.
 
    Band width (p90 - p10) is the model's self-reported uncertainty for that
    player. We rank it within the priced population and bucket into thirds, so
    the label means 'how unsure is this call vs. the others on screen.'
    """
    v = valued.copy()
    if not {"bpm_t1_p10", "bpm_t1_p90"}.issubset(v.columns):
        v["confidence_label"] = "Unknown"
        v["band_width"] = np.nan
        return v
    width = (v["bpm_t1_p90"] - v["bpm_t1_p10"]).astype(float)
    v["band_width"] = width
    valid = width.notna()
    q = width[valid].rank(pct=True)
    label = pd.Series("Unknown", index=v.index)
    label.loc[q.index[q <= 1 / 3]] = "High"
    label.loc[q.index[(q > 1 / 3) & (q <= 2 / 3)]] = "Moderate"
    label.loc[q.index[q > 2 / 3]] = "Low"
    v["confidence_label"] = label
    return v
 
 
def attach_cross_target(valued: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Add `vorp_flag` and `agreement_label` by running the SAME VORP comp track
    multi_target.py uses, then comparing per player.
 
    agreement_label values:
      * "Confirmed"   — VORP lands the same bucket as BPM
      * "Mixed"       — off by one bucket (e.g. Fair vs Overvalued)
      * "Contradicts" — direct opposite (Overvalued vs Undervalued)
      * "n/a"         — VORP did not give this player a decisive verdict
    """
    v = valued.copy()
    try:
        train_v = MT._build_vorp_training_table(df)
        vmodel = MT._train_vorp_median(train_v)
        fc_v = MT._forecast_vorp_t1(df, vmodel)
        vorp_flag = MT._vorp_value_flags(df, fc_v)   # name_key -> flag
    except Exception:
        v["vorp_flag"] = "n/a"
        v["agreement_label"] = "n/a"
        return v
 
    v["vorp_flag"] = v["name_key"].map(vorp_flag).fillna("n/a")
 
    def agree(row):
        b, vf = row["valuation_flag"], row["vorp_flag"]
        if b not in _ORDER or vf not in _ORDER:
            return "n/a"
        d = abs(_ORDER[b] - _ORDER[vf])
        return "Confirmed" if d == 0 else "Mixed" if d == 1 else "Contradicts"
 
    v["agreement_label"] = v.apply(agree, axis=1)
    return v
 
 
def attach_cross_model(valued: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Add `aging_flag` and `model_agreement_label` by running the parameter-free
    aging-curve model (model_ensemble) and comparing its verdict to production's.
 
    This is the cross-MODEL companion to attach_cross_target's cross-TARGET (VORP)
    check: does the call survive a different model CLASS, not just a different
    stat? Crucially we compare against production's MULTI-YEAR flag, not the
    single-season one — the single-season flag keys off current BPM (shared by
    both models by construction, so it would agree trivially), while the
    multi-year flag prices the 3-season projection, the only place the two model
    classes actually differ.
 
    model_agreement_label values:
      * "Confirmed"   — aging-curve model lands the same bucket as the GBM
      * "Mixed"       — off by one bucket
      * "Contradicts" — direct opposite (Overvalued vs Undervalued)
      * "n/a"         — one of the two models gave no decisive multi-year verdict
    """
    v = valued.copy()
    # bucket production's multi-year flag into the shared Over/Fair/Under space
    _MY = {"Overvalued (multi-yr)": "Overvalued",
           "Fair Value (multi-yr)": "Fair Value",
           "Undervalued (multi-yr)": "Undervalued"}
    try:
        aging_flag = ME.aging_bucket_flags(df)       # name_key -> bucket
    except Exception:
        v["aging_flag"] = "n/a"
        v["model_agreement_label"] = "n/a"
        return v
 
    v["aging_flag"] = v["name_key"].map(aging_flag).fillna("n/a")
 
    def agree(row):
        prod = _MY.get(row.get("multiyear_flag"), "n/a")
        af = row["aging_flag"]
        if prod not in _ORDER or af not in _ORDER:
            return "n/a"
        d = abs(_ORDER[prod] - _ORDER[af])
        return "Confirmed" if d == 0 else "Mixed" if d == 1 else "Contradicts"
 
    v["model_agreement_label"] = v.apply(agree, axis=1)
    return v
 
 
def attach_all(valued: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    return attach_cross_model(
        attach_cross_target(attach_confidence(valued), df), df)
 
 
def confidence_sentence(row) -> str:
    """One plain-language caveat string for the player card."""
    conf = row.get("confidence_label", "Unknown")
    agree = row.get("agreement_label", "n/a")
    vf = row.get("vorp_flag", "n/a")
    bits = []
    if conf == "Low":
        bits.append("the model's own projection band for this player is wide, "
                    "so this call is low-confidence")
    elif conf == "High":
        bits.append("the projection band is tight, so the model is relatively "
                    "confident in this trajectory")
    elif conf == "Moderate":
        bits.append("the projection band is moderate")
 
    if agree == "Confirmed":
        bits.append("a second independent metric (VORP) reaches the same verdict")
    elif agree == "Mixed":
        bits.append(f"a second metric (VORP) reads this contract as '{vf}', "
                    "one step off — treat the call as soft")
    elif agree == "Contradicts":
        bits.append(f"a second metric (VORP) DISAGREES, reading it as '{vf}' — "
                    "this verdict is target-dependent and should not be acted on "
                    "from this tool alone")
 
    m_agree = row.get("model_agreement_label", "n/a")
    af = row.get("aging_flag", "n/a")
    if m_agree == "Confirmed":
        bits.append("a different model class (a parameter-free aging curve) "
                    "reaches the same multi-year verdict")
    elif m_agree == "Mixed":
        bits.append(f"a different model class (aging curve) reads the multi-year "
                    f"deal as '{af}', one step off — treat the call as soft")
    elif m_agree == "Contradicts":
        bits.append(f"a different model class (aging curve) DISAGREES on the "
                    f"multi-year deal, reading it as '{af}' — this verdict is "
                    "model-dependent and should not be acted on from this tool "
                    "alone")
 
    if not bits:
        return ""
    return "Confidence: " + "; ".join(bits) + "."