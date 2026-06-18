"""
Parquet Capital — test suite
============================
Targeted tests on the correctness-critical paths, where a silent bug would
quietly corrupt everything downstream. Run:  pytest -q
 
Covers:
  * normalize_name      — the join key every source depends on
  * severity_score      — the ordinal that drives injury tiers
  * trade_is_legal      — the CBA salary-match boundary
  * _value_score        — replacement anchor + monotonicity
  * _advance_features   — the roll-forward feature advance
  * value_players       — the honest abstentions (no-salary / below-replacement)
  * optimize_roster     — hard cap + roster-size + 35%-rule constraints
"""
import numpy as np
import pandas as pd
import pytest
 
import models as M
import build_dataset as B
 
 
# ----------------------------------------------------------------------
# normalize_name — accents, suffixes, slash aliases, junk input
# ----------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("Nikola Jokić", "nikola jokic"),          # accent stripped
    ("Luka Dončić", "luka doncic"),            # accent stripped
    ("Gary Payton II", "gary payton"),         # suffix dropped
    ("Tim Hardaway Jr.", "tim hardaway"),      # suffix + period dropped
    ("Kay / Kahlil Felder", "kay"),            # first slash-alias only
    ("  Jaren  Jackson   Jr. ", "jaren jackson"),  # collapse whitespace + suffix
    ("Karl-Anthony Towns", "karlanthony towns"),   # hyphen punctuation removed
])
def test_normalize_name_known_cases(raw, expected):
    assert B.normalize_name(raw) == expected
 
 
@pytest.mark.parametrize("bad", [None, np.nan, 123, ""])
def test_normalize_name_non_string_is_empty(bad):
    assert B.normalize_name(bad) == ""
 
 
def test_normalize_name_is_idempotent():
    once = B.normalize_name("Nikola Jokić Jr.")
    assert B.normalize_name(once) == once
 
 
# ----------------------------------------------------------------------
# severity_score — ordinal tiers + rest handled by caller
# ----------------------------------------------------------------------
def test_severity_season_ending_is_highest():
    assert B.severity_score("Placed on IL, out for season (torn ACL)") == 6
 
 
def test_severity_major_surgery_tear_fracture():
    for note in ["underwent surgery", "torn meniscus", "fractured wrist", "Achilles rupture"]:
        assert B.severity_score(note) == 4
 
 
def test_severity_notable_sprain_strain():
    for note in ["ankle sprain", "strained hamstring", "thigh contusion"]:
        assert B.severity_score(note) == 2
 
 
def test_severity_routine_default():
    for note in ["sore ankle", "illness", "placed on IL", "rest"]:
        assert B.severity_score(note) == 1
 
 
def test_severity_ordering_is_monotone():
    assert (B.severity_score("out for season") > B.severity_score("torn")
            > B.severity_score("sprain") > B.severity_score("soreness"))
 
 
def test_severity_non_string_is_routine():
    assert B.severity_score(None) == 1
    assert B.severity_score(np.nan) == 1
 
 
# ----------------------------------------------------------------------
# trade_is_legal — the 125% + $250K boundary
# ----------------------------------------------------------------------
def test_trade_under_cap_always_legal():
    legal, _ = M.trade_is_legal(out_salary_m=5.0, in_salary_m=40.0, over_cap=False)
    assert legal is True
 
 
def test_trade_exactly_on_band_is_legal():
    # outgoing $10M -> band = 1.25*10M + 250k = $12.75M; incoming exactly there is legal
    legal, _ = M.trade_is_legal(out_salary_m=10.0, in_salary_m=12.75, over_cap=True)
    assert legal is True
 
 
def test_trade_just_over_band_fails():
    legal, _ = M.trade_is_legal(out_salary_m=10.0, in_salary_m=12.76, over_cap=True)
    assert legal is False
 
 
def test_trade_well_within_band_is_legal():
    legal, _ = M.trade_is_legal(out_salary_m=20.0, in_salary_m=15.0, over_cap=True)
    assert legal is True
 
 
# ----------------------------------------------------------------------
# _value_score — replacement anchor + monotonicity
# ----------------------------------------------------------------------
def test_value_score_at_replacement_is_floor():
    assert M._value_score(M.REPLACEMENT_BPM) == pytest.approx(M.VALUE_FLOOR)
 
 
def test_value_score_one_point_above_adds_one():
    assert (M._value_score(M.REPLACEMENT_BPM + 1.0)
            == pytest.approx(M.VALUE_FLOOR + 1.0))
 
 
def test_value_score_below_replacement_clamped_to_floor():
    # the max(.,0) means sub-replacement does not go below the floor
    assert M._value_score(M.REPLACEMENT_BPM - 5.0) == pytest.approx(M.VALUE_FLOOR)
 
 
def test_value_score_is_monotone_nondecreasing():
    xs = np.linspace(-8, 13, 50)
    vs = np.array([M._value_score(x) for x in xs])
    assert np.all(np.diff(vs) >= -1e-9)
 
 
# ----------------------------------------------------------------------
# _advance_features — the roll-forward step
# ----------------------------------------------------------------------
def _seed_feat():
    return {"Age": 25.0, "BPM": 2.0, "PER": 16.0, "WS_per_48": 0.12,
            "VORP": 1.5, "USG%": 22.0, "injury_events_3yr": 2.0,
            "aging_curve_delta": 0.3}
 
 
def test_advance_sets_bpm_to_prediction_and_ages_up():
    feat = _seed_feat()
    out = M._advance_features(feat, p50=4.0, pos="G", aging_lut={},
                              sensitivities=M._DEFAULT_SENS)
    assert out["BPM"] == 4.0
    assert out["Age"] == 26.0
 
 
def test_advance_decays_injury_signal():
    feat = _seed_feat()
    before = feat["injury_events_3yr"]
    out = M._advance_features(feat, p50=2.0, pos="G", aging_lut={},
                              sensitivities=M._DEFAULT_SENS)
    assert out["injury_events_3yr"] < before
    assert out["injury_events_3yr"] == pytest.approx(before * 0.7)
 
 
def test_advance_moves_rate_stats_with_bpm():
    feat = _seed_feat()
    base_per = feat["PER"]
    # positive BPM jump should push PER up given a positive sensitivity slope
    out = M._advance_features(feat, p50=feat["BPM"] + 2.0, pos="G", aging_lut={},
                              sensitivities={"PER": 0.55})
    assert out["PER"] > base_per
 
 
def test_advance_refreshes_aging_from_lookup():
    feat = _seed_feat()
    lut = {("G", 26.0): -0.9}
    out = M._advance_features(feat, p50=2.0, pos="G", aging_lut=lut,
                              sensitivities=M._DEFAULT_SENS)
    assert out["aging_curve_delta"] == -0.9
 
 
# ----------------------------------------------------------------------
# value_players — honest abstentions
# ----------------------------------------------------------------------
def _forecast_row(**kw):
    base = dict(name_key="x", Player="X", Team="GSW", pos_group="G", Age=27,
                salary_m=10.0, current_bpm=3.0,
                bpm_t1_p50=3.0, bpm_t2_p50=3.0, bpm_t3_p50=3.0,
                injury_risk_tier="Low", contract_total_m=np.nan,
                years_remaining=np.nan, dead_cap_m=np.nan)
    base.update(kw)
    return base
 
 
def test_no_salary_player_is_abstained():
    fc = pd.DataFrame([_forecast_row(name_key="a", salary_m=np.nan),
                       _forecast_row(name_key="b", salary_m=0.0)])
    out = M.value_players(fc)
    assert set(out["valuation_flag"]) == {"Unrated (no salary)"}
 
 
def test_below_replacement_min_salary_is_fair_not_abstained():
    # cheap-contract track: a sub-replacement player at the salary FLOOR is a fair
    # minimum deal, not an abstention. Build a pool so the median wage is high
    # enough that an $0.9M player sits below the dead-money line.
    pool = [_forecast_row(name_key=f"hi{i}", current_bpm=6.0, salary_m=20.0)
            for i in range(5)]
    pool.append(_forecast_row(name_key="minguy", current_bpm=-3.0, salary_m=0.9))
    out = M.value_players(pd.DataFrame(pool)).set_index("name_key")
    assert out.loc["minguy", "valuation_flag"] == "Fair (min contract)"
 
 
def test_below_replacement_big_salary_is_dead_money():
    # a sub-replacement player on real money (above the median wage) is stranded
    # cap and must read 'Overpay (dead money)'.
    pool = [_forecast_row(name_key=f"lo{i}", current_bpm=-3.0, salary_m=0.8)
            for i in range(5)]
    pool.append(_forecast_row(name_key="deadweight", current_bpm=-4.0, salary_m=25.0))
    out = M.value_players(pd.DataFrame(pool)).set_index("name_key")
    assert out.loc["deadweight", "valuation_flag"] == "Overpay (dead money)"
 
 
def test_cheap_track_lifts_coverage_above_replacement_gap():
    # no priced player should be left with the OLD blanket abstention label; every
    # salaried below-replacement player now gets a fair/dead-money verdict.
    pool = [_forecast_row(name_key=f"hi{i}", current_bpm=6.0, salary_m=20.0)
            for i in range(5)]
    pool += [_forecast_row(name_key=f"lo{i}", current_bpm=-3.0, salary_m=s)
             for i, s in enumerate([0.5, 1.0, 6.0, 18.0])]
    out = M.value_players(pd.DataFrame(pool))
    assert "Below replacement - not priced" not in set(out["valuation_flag"])
    sub = out[out["current_bpm"] < M.REPLACEMENT_BPM]["valuation_flag"]
    assert set(sub).issubset({"Fair (min contract)", "Overpay (dead money)"})
 
 
def test_priceable_flag_requires_salary_and_above_replacement():
    fc = pd.DataFrame([
        _forecast_row(name_key="good", current_bpm=5.0, salary_m=10.0),   # priceable
        _forecast_row(name_key="poor", current_bpm=-3.0, salary_m=10.0),  # below repl
        _forecast_row(name_key="free", current_bpm=5.0, salary_m=0.0),    # no salary
    ])
    out = M.value_players(fc).set_index("name_key")
    assert bool(out.loc["good", "priceable"]) is True
    assert bool(out.loc["poor", "priceable"]) is False
    assert bool(out.loc["free", "priceable"]) is False
 
 
# ----------------------------------------------------------------------
# optimize_roster — hard constraints must hold
# ----------------------------------------------------------------------
def _player_pool(n=30, seed=0):
    rng = np.random.default_rng(seed)
    pos = (["G"] * 12) + (["F"] * 12) + (["C"] * 6)
    return pd.DataFrame(dict(
        name_key=[f"p{i}" for i in range(n)],
        Player=[f"P{i}" for i in range(n)],
        pos_group=pos[:n],
        salary_m=rng.uniform(1, 30, n),
        bpm_t1_p50=rng.uniform(-2, 9, n),
        bpm_t1_p10=rng.uniform(-5, 5, n),
    ))
 
 
def test_optimizer_respects_roster_size_and_cap():
    pool = _player_pool()
    cap = 154_000_000
    sel, proj, spend = M.optimize_roster(pool, cap=cap, roster_size=13)
    assert len(sel) == 13
    assert spend * 1e6 <= cap + 1.0          # cap not exceeded (float tolerance)
 
 
def test_optimizer_enforces_position_minimums():
    pool = _player_pool()
    sel, _, _ = M.optimize_roster(pool, roster_size=13)
    counts = sel["pos_group"].value_counts()
    assert counts.get("G", 0) >= 3
    assert counts.get("F", 0) >= 3
    assert counts.get("C", 0) >= 2
 
 
def test_optimizer_excludes_over_35pct_player():
    pool = _player_pool()
    cap = 154_000_000
    # force one player above 35% of cap -> must be excluded
    pool.loc[0, "salary_m"] = 0.40 * cap / 1e6
    pool.loc[0, "bpm_t1_p50"] = 99.0   # huge value, still must be excluded
    sel, _, _ = M.optimize_roster(pool, cap=cap, roster_size=13)
    assert pool.loc[0, "name_key"] not in set(sel["name_key"])
 
 
# ----------------------------------------------------------------------
# _scale_for_coverage — the band calibrator core (monotone, hits target)
# ----------------------------------------------------------------------
def test_scale_for_coverage_hits_nominal_on_symmetric_resid():
    rng = np.random.default_rng(0)
    n = 4000
    resid = rng.normal(0, 1.0, n)
    lo_gap = np.full(n, 1.0)
    hi_gap = np.full(n, 1.0)
    s, cov = M._scale_for_coverage(lo_gap, hi_gap, resid, nominal=0.80)
    assert abs(cov - 0.80) < 0.03           # achieves the requested coverage
    # for unit gaps and N(0,1) resid, 80% central mass needs ~1.28 sigma
    assert 1.1 < s < 1.45
 
 
def test_scale_for_coverage_wider_resid_needs_bigger_scale():
    rng = np.random.default_rng(1)
    n = 4000
    lo_gap = hi_gap = np.full(n, 1.0)
    s_narrow, _ = M._scale_for_coverage(lo_gap, hi_gap,
                                        rng.normal(0, 1.0, n), 0.80)
    s_wide, _ = M._scale_for_coverage(lo_gap, hi_gap,
                                      rng.normal(0, 2.0, n), 0.80)
    assert s_wide > s_narrow                 # more dispersion -> wider band
 
 
# ----------------------------------------------------------------------
# multi_target — VORP value score mirrors the BPM one at its own anchor
# ----------------------------------------------------------------------
def test_vorp_value_score_at_replacement_is_floor():
    import multi_target as MT
    assert MT._vorp_value_score(MT.VORP_REPLACEMENT) == pytest.approx(MT.VORP_VALUE_FLOOR)
 
 
def test_vorp_value_score_is_monotone_and_clamped():
    import multi_target as MT
    xs = np.linspace(-4, 8, 40)
    vs = MT._vorp_value_score(xs)
    assert np.all(np.diff(vs) >= -1e-9)                       # monotone
    assert MT._vorp_value_score(-3.0) == pytest.approx(MT.VORP_VALUE_FLOOR)  # clamp
 
 
# ----------------------------------------------------------------------
# model_ensemble — the parameter-free aging-curve forecaster (2nd model class)
# ----------------------------------------------------------------------
def _mini_panel():
    # two players, two consecutive seasons each, so build_aging_lookup has cells
    rows = []
    for nk, ages, bpms in [("a", (24, 25), (2.0, 3.0)),
                           ("b", (29, 30), (4.0, 2.0))]:
        for season, age, bpm in zip((2024, 2025), ages, bpms):
            rows.append(dict(name_key=nk, Player=nk.upper(), season=season,
                             Age=age, Team="GSW", pos_group="G", BPM=bpm,
                             PER=15.0, WS_per_48=0.1, VORP=1.0,
                             salary_m=10.0, injury_risk_tier="Low",
                             aging_curve_delta=0.0))
    return pd.DataFrame(rows)
 
 
def test_aging_forecaster_advances_bpm_by_aging_delta():
    import model_ensemble as ME
    df = _mini_panel()
    f = ME.AgingCurveForecaster(df)
    # craft a lookup with a known delta and confirm the roll-forward uses it
    f.lut = {("G", 30.0): -1.0, ("G", 31.0): -1.0, ("G", 32.0): -1.0}
    fc = f.forecast(df)
    row = fc[fc.name_key == "b"].iloc[0]
    # player b is 30 in latest season, BPM 2.0; one step applies the age-30 delta
    assert row["bpm_t1_p50"] == pytest.approx(1.0)       # 2.0 + (-1.0)
    assert row["current_bpm"] == pytest.approx(2.0)
 
 
def test_aging_forecaster_missing_cell_falls_back_to_flat():
    import model_ensemble as ME
    df = _mini_panel()
    f = ME.AgingCurveForecaster(df)
    f.lut = {}                                  # no aging information at all
    fc = f.forecast(df)
    row = fc[fc.name_key == "a"].iloc[0]
    # with an empty lookup the projection holds BPM flat (zero delta)
    assert row["bpm_t1_p50"] == pytest.approx(row["current_bpm"])
 
 
def test_aging_forecaster_output_schema_feeds_value_players():
    import model_ensemble as ME
    df = _mini_panel()
    fc = ME.AgingCurveForecaster(df).forecast(df)
    # the comp engine must run unchanged on this forecast (drop-in second opinion)
    valued = M.value_players(fc, verbose=False)
    assert "valuation_flag" in valued.columns
    assert "multiyear_flag" in valued.columns
 
if __name__ == "__main__":
    import sys
