"""Generate a synthetic clean_roster.csv with the exact schema models.py expects,
so the new tuning / backtest code can be exercised without the private raw CSVs.
Players follow plausible aging-curve BPM trajectories with noise + injury signal."""
import numpy as np, pandas as pd, os
 
rng = np.random.default_rng(7)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parquet_out")
os.makedirs(OUT, exist_ok=True)
 
POS = ["G", "F", "C"]
SEASONS = list(range(2017, 2026))          # 2017..2025, matches the real span
N_PLAYERS = 480
 
def aging_delta(age):
    # rough real-shape curve: improve into mid-20s, decline after ~30
    return 0.9 - 0.06 * (age - 24) - 0.012 * max(0, age - 30) ** 2 / 4
 
rows = []
for pid in range(N_PLAYERS):
    pos = rng.choice(POS, p=[0.45, 0.4, 0.15])
    start_age = int(rng.integers(19, 31))
    career = int(rng.integers(2, 9))
    base = rng.normal(0.5, 3.0)            # latent talent in BPM units
    bpm = base
    debut = int(rng.choice(SEASONS[: max(1, len(SEASONS) - 2)]))
    salary = float(np.clip(rng.normal(12, 11), 0.9, 52)) if rng.random() > 0.12 else np.nan
    knee = int(rng.random() < 0.15)
 
    # Multi-year contract terms, mirroring the schema build_dataset.py writes
    # (contract_total_m / contract_aav_m / years_remaining / dead_cap_m). The
    # real pipeline matches a subset of players to a multi-year deal; we mirror
    # that here so the cross-MODEL robustness check (model_ensemble.py) — which
    # prices the 3-season projection against multi-year contracts and is
    # otherwise un-exercisable on synthetic data — has decisive players to
    # compare. Players with no salary, or randomly left unmatched, get NaN terms
    # exactly as the real unmatched remainder does.
    if np.isnan(salary) or rng.random() < 0.30:
        years_remaining = contract_aav_m = contract_total_m = dead_cap_m = np.nan
    else:
        years_remaining = int(rng.integers(1, 5))           # 1..4 guaranteed yrs
        contract_aav_m = float(np.clip(salary * rng.uniform(0.9, 1.15), 0.9, 55))
        contract_total_m = float(contract_aav_m * years_remaining)
        dead_cap_m = (float(np.clip(rng.uniform(1, 8), 0, contract_total_m))
                      if (salary > 6 and rng.random() < 0.15) else np.nan)
    for k in range(career):
        season = debut + k
        if season > 2025:
            break
        age = start_age + k
        bpm = bpm + aging_delta(age) * 0.4 + rng.normal(0, 1.6)
        bpm = float(np.clip(bpm, -8, 13))
        sev = rng.choice([0, 1, 2, 4, 6], p=[0.55, 0.2, 0.15, 0.07, 0.03])
        rows.append(dict(
            name_key=f"player_{pid}", Player=f"Player {pid}", season=season,
            Age=age, Team=rng.choice(["GSW","LAL","BOS","DEN","PHO","NYK","MIL","DAL"]),
            pos_group=pos,
            BPM=bpm,
            PER=float(np.clip(15 + bpm * 0.9 + rng.normal(0, 2), 3, 35)),
            WS_per_48=float(np.clip(0.10 + bpm * 0.011 + rng.normal(0, 0.02), -0.05, 0.32)),
            VORP=float(np.clip(bpm * 0.32 + rng.normal(0, 0.4), -2, 8)),
            **{"USG%": float(np.clip(20 + bpm * 0.6 + rng.normal(0, 3), 8, 38))},
            salary_m=salary,
            injury_events_3yr=float(rng.poisson(0.6)),
            severity_weighted_events=int(sev),
            max_event_severity=int(sev),
            had_knee_injury=knee,
            aging_curve_delta=aging_delta(age),
            injury_risk_tier=("High" if sev >= 6 else "Medium" if sev >= 2 else "Low"),
            contract_total_m=contract_total_m,
            contract_aav_m=contract_aav_m,
            years_remaining=years_remaining,
            dead_cap_m=dead_cap_m,
        ))
 
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "clean_roster.csv"), index=False)
print(f"wrote {len(df)} player-seasons, {df.name_key.nunique()} players -> {OUT}/clean_roster.csv")
print("\n" + "=" * 72)
print("NOTE: this is SYNTHETIC data for exercising the pipeline, validation, and")
print("tests without the private raw CSVs. Numbers produced from it are NOT the")
print("project's results — momentum/durability and contract structure are muted")
print("by construction, so they understate what the modules find on the real")
print("roster. The figures that count are the real-roster ones in the README.")
print("=" * 72)
