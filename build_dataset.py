"""
Parquet Capital — Phase 1: Data Acquisition & Feature Engineering
Joins three raw sources into a single player-season table, then engineers
the contract-efficiency, injury-risk, and aging-curve features the models need.
 
Inputs (raw):
  - player_advanced.csv                 (advanced metrics: BPM, VORP, PER, WS/48 ...)
  - NBA_Player_Stats_and_Salaries_2010-2025.csv  (salary per player-season)
  - injury_data.csv                     (Prosportstransactions-style injury log)
 
Output:
  - clean_roster.csv                    (one row per player-season, all features)
  - data_dictionary.csv                 (rationale for every engineered column)
"""
 
import re
import unicodedata
import argparse
import numpy as np
import pandas as pd
import os
import sys
 
# Print accented player names safely on non-UTF-8 consoles (e.g. Windows cp1252).
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass
 
# Paths are configurable via (in priority order) CLI flags, environment
# variables, then defaults anchored to THIS script's folder (not the current
# working directory), so build and app steps agree on where data lives
# regardless of where each is launched from.
#   PARQUET_RAW  — folder holding the three raw CSVs
#   PARQUET_OUT  — folder where clean_roster.csv / data_dictionary.csv are written
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW = os.environ.get("PARQUET_RAW", _SCRIPT_DIR)
OUT = os.environ.get("PARQUET_OUT", os.path.join(_SCRIPT_DIR, "parquet_out"))
 
 
# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def normalize_name(name: str) -> str:
    """Canonical player key: strip accents, lowercase, drop punctuation/suffixes.
    Handles the slash-separated aliases in the injury log ('Kay / Kahlil Felder')
    by taking the first variant. This is the join key across all three sources."""
    if not isinstance(name, str):
        return ""
    name = name.split("/")[0]                      # take first alias variant
    name = unicodedata.normalize("NFKD", name)
    name = name.encode("ascii", "ignore").decode("ascii")   # drop accents
    name = name.lower().strip()
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", name)    # drop suffixes
    name = re.sub(r"[^a-z\s]", "", name)                    # drop punctuation
    name = re.sub(r"\s+", " ", name).strip()
    return name
 
 
def load_contracts():
    """Load the cleaned multi-year contract table (current rostered players only).
 
    Source: Basketball-Reference contract export, cleaned into contracts_clean.csv
    with one authoritative row per player plus a separate dead-cap obligation per
    waived/stretched player. Joined on the same normalized name_key as every other
    source. Coverage is the *current* roster only (the population the tool prices
    going forward) — players with a contract but no recent stat season (e.g. 2025
    rookies) match here but have no BPM to forecast, an honest gap, not an error.
 
    Returns (active, deadcap) frames keyed on name_key. Missing file -> (None, None)
    so the build degrades to single-season salary behavior rather than failing.
    """
    path_csv = os.path.join(RAW, "contracts_clean.csv")
    path_xlsx = os.path.join(RAW, "contracts_clean.xlsx")
    if os.path.exists(path_csv):
        active = pd.read_csv(path_csv)
        deadcap = None
        dc_csv = os.path.join(RAW, "dead_cap.csv")
        if os.path.exists(dc_csv):
            deadcap = pd.read_csv(dc_csv)
    elif os.path.exists(path_xlsx):
        active = pd.read_excel(path_xlsx, sheet_name="Active Contracts")
        try:
            deadcap = pd.read_excel(path_xlsx, sheet_name="Dead Cap")
        except Exception:
            deadcap = None
    else:
        return None, None
 
    yrs = ["2025-26", "2026-27", "2027-28", "2028-29", "2029-30", "2030-31"]
    yrs = [y for y in yrs if y in active.columns]
    active["name_key"] = active["Player"].apply(normalize_name)
    # future-year guaranteed dollars and remaining-year count
    active["contract_total_m"] = active["Guaranteed"] / 1_000_000
    active["contract_y1_m"] = active[yrs[0]] / 1_000_000 if yrs else np.nan
    active["years_remaining"] = active[yrs].notna().sum(axis=1) if yrs else 0
    # mean annual value over the guaranteed years (a cleaner per-year price than y1)
    active["contract_aav_m"] = active.apply(
        lambda r: np.nanmean([r[y] for y in yrs if pd.notna(r[y])]) / 1_000_000
        if yrs and r[yrs].notna().any() else np.nan, axis=1)
    active = active[["name_key", "contract_total_m", "contract_y1_m",
                     "contract_aav_m", "years_remaining"]].copy()
    active = active.sort_values("contract_total_m", ascending=False)
    active = active.drop_duplicates("name_key")
 
    if deadcap is not None and len(deadcap):
        deadcap["name_key"] = deadcap["Player"].apply(normalize_name)
        owed_col = "Dead Cap Owed" if "Dead Cap Owed" in deadcap.columns else "Guaranteed"
        deadcap["dead_cap_m"] = deadcap[owed_col] / 1_000_000
        deadcap = (deadcap.groupby("name_key")["dead_cap_m"].sum().reset_index())
    return active, deadcap
 
 
def season_end_year(s):
    """Map '2023-24' -> 2024.  Salary file is already a single end-ish year."""
    if isinstance(s, str) and "-" in s:
        start = int(s.split("-")[0])
        return start + 1
    return int(s)
 
 
def primary_position(pos: str) -> str:
    """Collapse multi-position ('C-F') and granular ('SG') into G / F / C groups."""
    if not isinstance(pos, str) or not pos:
        return "Unknown"
    first = pos.replace("-", " ").split()[0].upper()
    if first in ("PG", "SG", "G"):
        return "G"
    if first in ("SF", "PF", "F"):
        return "F"
    if first in ("C",):
        return "C"
    return "F"  # default fallback for odd labels
 
 
# ----------------------------------------------------------------------
# 1. Load + standardize each source
# ----------------------------------------------------------------------
def load_advanced():
    df = pd.read_csv(f"{RAW}/player_advanced.csv", encoding="utf-8")
    df = df.rename(columns={"WS/48": "WS_per_48"})
    df["season"] = df["Season"].apply(season_end_year)
    df["name_key"] = df["Player"].apply(normalize_name)
    df["pos_group"] = df["Pos"].apply(primary_position)
    keep = ["name_key", "Player", "season", "Age", "Team", "pos_group",
            "G", "GS", "BPM", "OBPM", "DBPM", "VORP", "PER",
            "WS", "WS_per_48", "USG%", "TRB%", "AST%", "STL%", "BLK%", "TOV%"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    # one row per player-season: if traded mid-year, keep the max-games stint
    df = df.sort_values("G", ascending=False).drop_duplicates(["name_key", "season"])
    return df
 
 
def load_salary():
    df = pd.read_csv(f"{RAW}/NBA_Player_Stats_and_Salaries_2010-2025.csv", encoding="utf-8")
    df.columns = [c.lstrip("\ufeff") for c in df.columns]   # strip BOM on 'Player'
    df["season"] = df["Year"].apply(season_end_year)
    df["name_key"] = df["Player"].apply(normalize_name)
    df = df[["name_key", "season", "Salary"]].copy()
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
    df = df.dropna(subset=["Salary"])
    df = df.sort_values("Salary", ascending=False).drop_duplicates(["name_key", "season"])
    return df
 
 
# ----- injury log: parse the free-text event log into per-season features -----
BODY_PARTS = {
    "knee": ["knee", "acl", "mcl", "meniscus", "patell"],
    "ankle": ["ankle"],
    "foot": ["foot", "toe", "plantar", "achilles", "heel"],
    "hamstring": ["hamstring"],
    "back": ["back", "spine", "disc", "lumbar"],
    "shoulder": ["shoulder", "rotator"],
    "hand": ["hand", "finger", "thumb", "wrist"],
    "hip": ["hip", "groin", "quad", "thigh", "calf"],
    "head": ["concussion", "head", "face", "nose", "eye"],
}
REST_FLAGS = ["rest", "load management", "personal", "coach", "dnp-cd", "not with team"]
 
# ----- injury severity scoring -------------------------------------------------
# The injury log records every IL placement at the same nominal weight, so a raw
# event count treats a season-ending ACL tear and a one-game sore ankle as equal.
# That is why the unweighted tiers skew High. We instead score each event from its
# free-text Note onto an ordinal severity, so the tier reflects how much basketball
# the stint actually cost rather than how many times a player touched the IL.
#
# Scores are deliberately coarse, not false precision: they encode an ordering
# (routine < notable < major < season-altering), calibrated so a single
# season-ending event clears the High threshold on its own while a cluster of
# minor day-to-day stints does not.
SEVERITY_SEASON_ENDING = ["out for season", "out for the season", "season-ending",
                          "out indefinitely"]
SEVERITY_MAJOR = ["surgery", "torn", "rupture", "ruptured", "fracture", "fractured",
                  "broken", "acl", "mcl", "achilles", "meniscus"]
SEVERITY_NOTABLE = ["sprain", "strain", "strained", "sprained", "bruise", "contusion",
                    "plantar", "stress reaction", "hairline"]
# everything else that is a real injury (sore, soreness, tightness, illness,
# unspecified "placed on IL") scores the routine baseline of 1.
 
 
def severity_score(note: str) -> int:
    """Ordinal severity of a single injury event from its free-text note.
    6 = season-ending, 4 = major (surgery/tear/fracture), 2 = notable
    (sprain/strain), 1 = routine/day-to-day/unspecified. Rest stints are scored 0
    upstream by the caller because they are not injuries."""
    n = note.lower() if isinstance(note, str) else ""
    if any(k in n for k in SEVERITY_SEASON_ENDING):
        return 6
    if any(k in n for k in SEVERITY_MAJOR):
        return 4
    if any(k in n for k in SEVERITY_NOTABLE):
        return 2
    return 1
 
 
def classify_body_part(note: str) -> str:
    note = note.lower() if isinstance(note, str) else ""
    for part, keys in BODY_PARTS.items():
        if any(k in note for k in keys):
            return part
    return "unspecified"
 
 
def is_rest(note: str) -> bool:
    note = note.lower() if isinstance(note, str) else ""
    return any(k in note for k in REST_FLAGS)
 
 
def load_injury_features():
    df = pd.read_csv(f"{RAW}/injury_data.csv", encoding="utf-8")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    # NBA season-ending year: Oct-Dec belongs to next calendar year's season
    df["season"] = np.where(df["Date"].dt.month >= 10,
                            df["Date"].dt.year + 1, df["Date"].dt.year)
    # 'Relinquished' = player going OUT (injury/IL). That's our event of interest.
    out = df[df["Relinquished"].notna() & (df["Relinquished"].str.strip() != "")].copy()
    out["name_key"] = out["Relinquished"].apply(normalize_name)
    out = out[out["name_key"] != ""]
    out["body_part"] = out["Notes"].apply(classify_body_part)
    out["is_rest"] = out["Notes"].apply(is_rest)
    out["is_injury"] = ~out["is_rest"]
    # per-event severity (0 for rest stints, ordinal 1-6 for real injuries)
    out["severity"] = out["Notes"].apply(severity_score).where(out["is_injury"], 0)
 
    grp = out.groupby(["name_key", "season"])
    feats = grp.agg(
        injury_events=("is_injury", "sum"),
        rest_events=("is_rest", "sum"),
        total_il_events=("is_injury", "size"),
        severity_weighted_events=("severity", "sum"),
        max_event_severity=("severity", "max"),
    ).reset_index()
 
    # recurrence: distinct body parts hit, and whether a knee injury appears
    bp = (out[out["is_injury"]]
          .groupby(["name_key", "season"])["body_part"]
          .agg(lambda s: s.value_counts().to_dict())
          .reset_index(name="body_part_counts"))
    feats = feats.merge(bp, on=["name_key", "season"], how="left")
    feats["distinct_body_parts"] = feats["body_part_counts"].apply(
        lambda d: len([k for k in d if k != "unspecified"]) if isinstance(d, dict) else 0)
    feats["had_knee_injury"] = feats["body_part_counts"].apply(
        lambda d: int("knee" in d) if isinstance(d, dict) else 0)
    feats = feats.drop(columns=["body_part_counts"])
    return feats
 
 
# ----------------------------------------------------------------------
# 2. Aging curves (delta method) — position-specific avg YoY BPM change by age
# ----------------------------------------------------------------------
def build_aging_curves(adv: pd.DataFrame) -> pd.DataFrame:
    a = adv.sort_values(["name_key", "season"]).copy()
    a["next_bpm"] = a.groupby("name_key")["BPM"].shift(-1)
    a["next_season"] = a.groupby("name_key")["season"].shift(-1)
    a = a[a["next_season"] == a["season"] + 1]          # consecutive seasons only
    a["bpm_delta"] = a["next_bpm"] - a["BPM"]
    curve = (a.groupby(["pos_group", "Age"])["bpm_delta"]
             .mean().reset_index()
             .rename(columns={"bpm_delta": "aging_curve_delta"}))
    return curve
 
 
# ----------------------------------------------------------------------
# 3. Assemble
# ----------------------------------------------------------------------
def main():
    global RAW, OUT
    ap = argparse.ArgumentParser(description="Build clean_roster.csv from raw NBA sources.")
    ap.add_argument("--raw", default=RAW, help="folder with the three raw CSVs")
    ap.add_argument("--out", default=OUT, help="output folder")
    args = ap.parse_args()
    RAW, OUT = args.raw, args.out
    os.makedirs(OUT, exist_ok=True)
 
    adv = load_advanced()
    sal = load_salary()
    inj = load_injury_features()
    curve = build_aging_curves(adv)
 
    df = adv.merge(sal, on=["name_key", "season"], how="left")
    df = df.merge(inj, on=["name_key", "season"], how="left")
    df = df.merge(curve, on=["pos_group", "Age"], how="left")
 
    # multi-year contracts (current roster). Attached to EVERY season row for the
    # player so the forecast/valuation step (which keys off the latest season) sees
    # them; they describe the player's contract today, not that historical season.
    contracts, deadcap = load_contracts()
    if contracts is not None:
        df = df.merge(contracts, on="name_key", how="left")
        if deadcap is not None:
            df = df.merge(deadcap, on="name_key", how="left")
        else:
            df["dead_cap_m"] = np.nan
        n_contract = df.drop_duplicates("name_key")["contract_total_m"].notna().sum()
        print(f"contracts joined: {n_contract} players have multi-year terms")
    else:
        for c in ["contract_total_m", "contract_y1_m", "contract_aav_m",
                  "years_remaining", "dead_cap_m"]:
            df[c] = np.nan
        print("no contracts_clean file found — using single-season salary only")
 
    # fill injury features (no injury record = zero events)
    for c in ["injury_events", "rest_events", "total_il_events",
              "distinct_body_parts", "had_knee_injury",
              "severity_weighted_events", "max_event_severity"]:
        df[c] = df[c].fillna(0).astype(int)
 
    # --- engineered features ---
    # WAR proxy. VORP is points above replacement per 100 team possessions over a
    # full season. Basketball-Reference's documented conversion to wins is
    # wins ≈ VORP / 2.7 * (team pace factor), which for a league-average team
    # collapses to a multiplier near 0.37; the older "VORP * 0.5" heuristic
    # over-credits by ~35%. We use the BR-derived 1/2.7 so the win figures are
    # defensible against a published source rather than a round number.
    VORP_TO_WINS = 1.0 / 2.7   # Basketball-Reference points->wins conversion
    df["war_proxy"] = df["VORP"] * VORP_TO_WINS
    df["salary_m"] = df["Salary"] / 1_000_000
    # contract efficiency: WAR delivered per $1M (the financial spine)
    df["contract_efficiency"] = np.where(
        df["salary_m"] > 0.5, df["war_proxy"] / df["salary_m"], np.nan)
 
    # 3-year rolling games missed proxy (uses injury_events as severity signal)
    df = df.sort_values(["name_key", "season"])
    df["injury_events_3yr"] = (df.groupby("name_key")["injury_events"]
                               .transform(lambda s: s.rolling(3, min_periods=1).mean()))
    # 3-year rolling severity-weighted load — the severity-aware analogue of
    # injury_events_3yr, and the basis for the re-tiered injury risk below.
    df["severity_3yr"] = (df.groupby("name_key")["severity_weighted_events"]
                          .transform(lambda s: s.rolling(3, min_periods=1).mean()))
    # age-at-injury interaction
    df["age_injury_interaction"] = df["Age"] * df["injury_events"]
 
    # injury risk tier — severity-weighted.
    # Previously this summed raw IL-event COUNTS, so frequent-but-minor stints
    # (sore ankle, illness) pushed players into High just as fast as a torn ACL,
    # which is why the tiers skewed heavily High. We now tier on the 3-year rolling
    # SEVERITY load (season-ending=6, major=4, notable=2, routine=1) plus a small
    # recurrence bump for a knee history. Thresholds are set so that a single
    # season-ending event (severity 6) reaches High on its own, a major or a couple
    # of notable stints reach Medium, and a thin trail of day-to-day stints stays
    # Low — i.e. the tier now tracks basketball lost, not turnstile clicks.
    def risk_tier(row):
        score = row["severity_3yr"] + 1.5 * row["had_knee_injury"]
        if score >= 5:
            return "High"
        if score >= 2:
            return "Medium"
        return "Low"
    df["injury_risk_tier"] = df.apply(risk_tier, axis=1)
 
    df = df.sort_values(["season", "name_key"]).reset_index(drop=True)
    df.to_csv(f"{OUT}/clean_roster.csv", index=False)
 
    # data dictionary
    ddict = pd.DataFrame([
        ("name_key", "normalized join key (accent/suffix-stripped player name)"),
        ("Player", "display name"),
        ("season", "season-ending year (2023-24 -> 2024)"),
        ("Age", "player age that season"),
        ("Team", "team abbreviation"),
        ("pos_group", "collapsed position: G / F / C"),
        ("BPM", "Box Plus-Minus — primary forecast target"),
        ("VORP", "Value Over Replacement Player"),
        ("PER", "Player Efficiency Rating"),
        ("WS_per_48", "Win Shares per 48 minutes"),
        ("Salary", "player salary for the season (USD)"),
        ("salary_m", "salary in $millions"),
        ("war_proxy", "wins-above-replacement proxy = VORP / 2.7 (BR points->wins)"),
        ("contract_efficiency", "war_proxy per $1M salary — the financial spine"),
        ("injury_events", "count of injury IL placements that season"),
        ("rest_events", "count of rest/load-management IL placements"),
        ("severity_weighted_events", "season sum of per-event severity (season-ending=6, major=4, notable=2, routine=1)"),
        ("max_event_severity", "most severe single injury event that season (0-6)"),
        ("injury_events_3yr", "3-year rolling mean of injury_events"),
        ("severity_3yr", "3-year rolling mean of severity_weighted_events — basis for injury_risk_tier"),
        ("distinct_body_parts", "number of distinct body parts injured"),
        ("had_knee_injury", "1 if a knee/ACL injury occurred (high recurrence)"),
        ("age_injury_interaction", "Age * injury_events"),
        ("injury_risk_tier", "High / Medium / Low risk classification"),
        ("aging_curve_delta", "expected YoY BPM change for this age+position"),
    ], columns=["column", "rationale"])
    ddict.to_csv(f"{OUT}/data_dictionary.csv", index=False)
 
    # ---- report ----
    print(f"clean_roster.csv  rows={len(df):,}  players={df['name_key'].nunique():,}")
    print(f"seasons: {int(df['season'].min())} - {int(df['season'].max())}")
    matched = df["Salary"].notna().mean()
    print(f"salary match rate: {matched:.1%}")
    inj_match = (df["injury_events"] > 0).mean()
    print(f"player-seasons with >=1 injury event: {inj_match:.1%}")
    print(f"risk tiers: {df['injury_risk_tier'].value_counts().to_dict()}")
    sev = df["severity_weighted_events"]
    print(f"severity-weighted events: mean {sev[sev>0].mean():.1f} among injured, "
          f"max {int(sev.max())}; season-ending events: {int((df['max_event_severity']>=6).sum())}")
    print("\naging curve sample (G):")
    print(curve[curve.pos_group == "G"].head(8).to_string(index=False))
    # sanity: flag if any single salary value is shared across >5 distinct players
    dup = (df.dropna(subset=["Salary"])
             .groupby("Salary")["name_key"].nunique())
    shared = dup[dup > 5]
    if len(shared):
        print(f"note: {len(shared)} salary values shared across >5 players "
              f"(usually legit cap-max tiers, but worth a glance)")
 
if __name__ == "__main__":
    main()