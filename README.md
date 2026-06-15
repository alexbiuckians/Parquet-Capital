# Parquet Capital — Roster Valuation & Risk Engine

A front-office decision tool that treats NBA players as financial assets: **it forecasts performance, prices contracts against the market, and optimizes roster construction under the salary cap.**


NBA teams routinely lose tens of millions to "dead cap" by paying for past performance while ignoring aging curves and injury volatility. Parquet Capital flags which contracts are becoming toxic assets before the damage shows up in the standings.

## What it does

- **Roster valuation table** — every priceable player flagged Overvalued / Fair / Undervalued vs. comparable contracts, with a 3-season BPM projection and an injury-risk tier. Players at or below replacement level are explicitly held back from a verdict rather than mislabeled (see Valuation coverage below).
- **Performance forecast** —    a quantile model projects each player's Box Plus-Minus for the next three seasons at the 10th / 50th / 90th percentile, so a +2.0 median with a -3.0 floor reads differently than the same median with a +1.0 floor. The band-widening factors are **calibrated against held-out coverage** (not hand-picked): the raw one-step band covered ~74% of actual outcomes, so it is scaled to hit the nominal 80%, then widened over the horizon on a random-walk (√h) progression. The roll-forward ages the **full feature vector** each season — rate stats move with predicted BPM via fitted slopes, the aging-curve delta is refreshed at the new age, and the injury signal decays — rather than freezing year-T values.
- **Cap optimizer** —  a PuLP linear program solves the roster knapsack: maximize projected performance under the salary cap, position minimums, and a no-single-player-over-35%-of-cap rule. Toggle between an upside roster (median projection) and a floor roster (10th percentile).
- **Trade simulator**  —    evaluate a one-for-one swap and get a plain-English verdict: a CBA salary-matching legality check (the over-the-cap ~125% + $250K band), cap impact, projected next-season BPM delta, the change in performance volatility, and a flag for any existing dead cap the outgoing player carries.
- **Multi-year contract valuation** — alongside the single-season comp flag, priceable players on real multi-year deals are valued on total guaranteed dollars over their remaining years against projected value across those same years (the forecast's t+1/t+2/t+3 medians, mapped through the same replacement-anchored value score). The same ceiling-cap and below-replacement abstentions carry over, so an MVP-tier max deal does not spuriously read Overvalued just because it is the largest, longest contract on the board.

## Architecture

```
raw CSVs ──► build_dataset.py ──► clean_roster.csv ──► models.py ──► app.py
(stats,        join on            (one row per         (BPM forecast,   (Streamlit
 salary,       player-season,      player-season,       comps engine,    dashboard)
 injuries)     engineer features)  all features)        PuLP optimizer)
```

## Tech stack

| Layer | Tool |
|---|---|
| Data engineering | pandas, numpy |
| Forecasting | scikit-learn gradient-boosted quantile regression |
| Optimization | PuLP (CBC solver) |
| Dashboard | Streamlit + Altair |

## Data

4,860 player-seasons, 2017–2025, joined from three public sources (advanced metrics, salary, and a Prosportstransactions-style injury log). The join is on a normalized player key that strips accents and suffixes and resolves the injury log's slash-separated name aliases.

Multi-year contracts: A fourth source — a cleaned Basketball-Reference contract export — adds forward-looking multi-year terms for the current roster: guaranteed dollars per future season through 2030-31, years remaining, and per-player dead-cap obligations for waived/stretched contracts. The raw export required real cleaning (it arrived as an HTML table mislabeled .xls, with trade/waiver duplicates and dead-cap rows mixed into the player list); the cleaner dedupes exact repeats, splits each multi-team player into one authoritative active contract plus separate flagged dead-cap rows, and keys everything on the same normalized name. 410 of the current roster's players match to multi-year terms on the name key; the unmatched remainder is dominated by 2025 rookies who have a contract but no prior stat season to forecast from — an honest gap, not a silent one, consistent with the abstention philosophy throughout. The single-season headline-cap figure remains the fallback for players the contract file does not cover. Full apron/exception mechanics (the 2023 CBA's two aprons, Bird rights, sign-and-trade rules) are the remaining v3 frontier.

The injury-risk tiers are **severity-weighted**, not event counts. Each IL placement is scored from its free-text note onto an ordinal severity (season-ending = 6, major surgery/tear/fracture = 4, notable sprain/strain = 2, routine day-to-day = 1), and the tier is set from a 3-year rolling severity load plus a small recurrence bump for a knee history. This was a deliberate refinement: an earlier version summed raw IL-event counts, so frequent-but-minor stints (a sore ankle, an illness) pushed a player into High just as fast as a torn ACL, and the tiers skewed heavily High (≈1,466 High / 1,821 Medium / 1,573 Low). Tiering on severity instead of frequency rebalances them to ≈1,219 High / 1,668 Medium / 1,973 Low — High is now the smallest band, as it should be, and a single season-ending injury clears the High threshold on its own while a thin trail of day-to-day stints stays Low. Validation confirms the signal tracks basketball lost rather than turnstile clicks: every player-season containing a season-ending event lands High or Medium (none slip to Low).


## Model performance

**One-step backtest.** Held out on the last two seasons: MAE ≈ 2.15 BPM, with the model calling the direction of a player's change (improve vs. decline) correctly about 64% of the time. The uncertainty band is coverage-calibrated: after scaling, the 10th–90th band covers ≈80% of held-out outcomes (its nominal target), versus ≈74% before calibration.

**Multi-step validation.** Because the t+2 / t+3 projections feed the model's own t+1 prediction back in, single-step error understates longer-horizon error. A dedicated roll-forward backtest (evaluate_multistep) trains only on seasons before the evaluation window and rolls each player's full feature vector forward, comparing every horizon against ground truth: **MAE ≈ 1.8 / 2.2 / 2.2 BPM at t+1 / t+2 / t+3**, beating a flat-persistence baseline (≈2.2 / 2.7 / 2.6) at every horizon. The roll-forward degrades gracefully rather than diverging — the aging-curve and feature-advance machinery adds real signal over "assume no change."

**Valuation coverage.** Contracts are priced on a replacement-anchored value score — BPM mapped onto a strictly-positive, monotone scale where replacement level (BPM ≈ −2.0) maps to a small positive floor — rather than raw BPM. This keeps the value rate stable instead of exploding as production approaches zero. Players at or below replacement level are **explicitly abstained from** ("Below replacement - not priced") rather than forced into an Overvalued/Undervalued call on a noise-dominated denominator; that is an honest abstention, not a silent gap. The priced population is therefore the players for whom a $/win comparison is actually meaningful: those with a salary on record and above-replacement production. This is a deliberate accuracy-over-coverage trade: the earlier "floor BPM at 0.1" approach reported higher nominal coverage but did so by pricing sub-replacement players on a meaningless ratio.



Concretely, of the **1,349 forecasted current players, ≈31%** receive a verdict (Overvalued / Undervalued / Fair Value). The remaining unrated players are honest abstentions, not silent gaps: ≈650 below-replacement (production too low to price), **262 with no salary on record** (a genuine source-data gap), and **12 elite max-tier players** held back as ceiling-capped (the comp pool cannot price them; see Elite max-contract handling below). The 262 (≈19% of forecasted players) is consistent with the 86% salary match rate quoted in Data — the two figures use different denominators: the 86% is over all 4,860 player-seasons, while the 262 counts only each player's single most-recent season, where missing-salary rows concentrate.


## Key findings (illustrative)

- Several max-contract veterans are flagged **Overvalued** — paying top-of-market $/value rates while their projected BPM trajectory is flat or declining, the classic dead-cap risk profile.
- The comps engine surfaces **Undervalued** young players whose $/value rate sits well below positional peers — the "buy low" targets.
The optimizer's upside vs. floor rosters expose the **risk premium**: the projected performance a team sacrifices to build a safer, more predictable roster.
**Elite max-contract handling:** because the comp pool has no salary tier more expensive than the league max, an elite producer on a max contract is necessarily compared against cheaper players and reads "Overvalued" — the flag would otherwise detect "is on a max deal" as much as "is a bad deal." The engine now separates the two on production, not just price: a top-salary-tier contract whose production is also top-tier is flagged **"Elite (max-tier) — ceiling-capped"**, an explicit abstention, because the comp set literally cannot price it higher. A top-tier salary on non-elite production is left to the normal comp logic and still flags Overvalued — that overpay is real, not an artifact. Both thresholds are data-driven (top-decile salary and BPM within the priced pool) rather than fixed dollar figures, so they track the league's actual cap environment. In practice this cleanly splits the two populations the original tool conflated: MVP-tier max deals (Jokić, Giannis, SGA, Curry, Dončić, Tatum) now abstain as ceiling-capped, while genuine dead-cap max deals (Bradley Beal and John Wall at sub-replacement BPM, Ben Simmons, an aged-out Kemba Walker) remain correctly flagged Overvalued.

## Run locally
Paths are configurable but optional. By default, build_dataset.py and models.py read the three raw CSVs from — and write clean_roster.csv into a parquet_out/ folder next to — the scripts themselves, so the app finds the data no matter what directory you launch it from. No code editing or environment setup is required if the raw CSVs sit alongside the scripts.

pip install -r requirements_parquetcapital.txt

# 1. build the dataset (writes parquet_out/clean_roster.csv next to the script)
python build_dataset.py

# 2. build forecasts + valuations; prints backtests + calibration report
python models.py

# 3. launch the dashboard
streamlit run app_NBA.py

To point at custom folders, use flags or environment variables (flags take priority):

# option A: flags
python build_dataset.py --raw ./data --out ./parquet_out
python models.py        --out ./parquet_out

# option B: env vars
export PARQUET_RAW=./data PARQUET_OUT=./parquet_out
python build_dataset.py
python models.py
streamlit run app_NBA.py

If you launch the app before building the dataset, it fails gracefully with an in-browser message telling you to run build_dataset.py first, rather than a raw traceback.


Data acquired from these three sources: 
https://www.kaggle.com/datasets/jacquesoberweis/2016-2025-nba-injury-data; 
https://www.kaggle.com/datasets/ratin21/nba-player-stats-and-salaries-2010-2025
https://www.kaggle.com/datasets/jacquesoberweis/2016-2025-nba-player-advanced-season-stats
https://www.basketball-reference.com/contracts/
https://www.basketball-reference.com/contracts/players.html
