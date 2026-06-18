# Parquet Capital — Roster Valuation & Risk Engine
This is a portfolio project that demonstrates front-office quantitative judgment end to end: joining messy public data, forecasting a noisy target with calibrated uncertainty, pricing contracts with explicit abstentions where the data can't support a verdict, and validating the decisions on held-out seasons. Who it's for: hiring managers and analysts evaluating how I reason about modeling under uncertainty — not an NBA front office for live cap decisions. What decision it informs: which contracts look rich or cheap relative to comparable players, as a first-pass screen to direct human attention. What it replaces: a spreadsheet of gut-feel "good/bad contract" labels with a reproducible, backtested, uncertainty-aware ranking. What it explicitly is NOT: (1) it is not a substitute for a real front office's proprietary tracking data, medical staff, or scouts; (2) the forecast target (BPM) carries ~2.0 BPM of t+1 error — real but modest — so single-player flags are a screen, not a verdict, and the UI now shows a per-player confidence label plus two independent agreement checks — a second metric (VORP) and a second model class (a parameter-free aging curve) — so the user sees when a call is soft; (3) it does not model the full 2023 CBA apron machinery — the trade check is the common salary-match tier only. The honest read of the model's edge: the GBM beats a do-nothing persistence baseline by ~0.4 BPM (1.80 vs 2.21 t+1 MAE) — a genuine but small edge, and the project is built to surface that modesty rather than hide it.

A front-office decision tool that treats NBA players as financial assets: **it forecasts performance, prices contracts against the market, and optimizes roster construction under the salary cap.**

NBA teams routinely lose tens of millions to "dead cap" by paying for past performance while ignoring aging curves and injury volatility. Parquet Capital flags which contracts are becoming toxic assets before the damage shows up in the standings.

## What it does

- **Roster valuation table** — every salaried player flagged Overvalued / Fair / Undervalued vs. comparable contracts, with a 3-season BPM projection and an injury-risk tier. Players above replacement are comp-priced; players below replacement are priced on a separate cheap-contract track against the league-minimum salary floor (a min-salary deal reads **Fair (min contract)**, a non-minimum salary on sub-replacement production reads **Overpay (dead money)** — the stranded cap a comp ratio cannot price). Together these lift verdict coverage from ~32% of the league to ~80%; the only genuinely unrated players are no-salary records and the few the comp set cannot serve (insufficient comps, elite max-tier ceiling-capped).
**Performance forecast** —  a quantile model projects each player's Box Plus-Minus for the next three seasons at the 10th / 50th / 90th percentile, so a +2.0 median with a -3.0 floor reads differently than the same median with a +1.0 floor. The band-widening factors are calibrated against held-out coverage at every horizon (not hand-picked, and no longer assumed past t+1): the raw one-step band covered ~74% of actual outcomes, so it is scaled to hit the nominal 80%; t+2 and t+3 are then calibrated the same way against the actual multi-step roll-forward — each horizon's widening factor is the scale that brings that horizon's rolled-forward band to 80% coverage of realized BPM, replacing the earlier √h assumption with a measured factor (the √h progression is retained only as an explicit fallback when a horizon has too little held-out history to measure). The roll-forward ages the full feature vector each season — rate stats move with predicted BPM via fitted slopes, the aging-curve delta is refreshed at the new age, and the injury signal decays — rather than freezing year-T values.
- **Cap optimizer** —a PuLP linear program solves the roster knapsack: maximize projected performance under the salary cap, position minimums, and a no-single-player-over-35%-of-cap rule. Toggle between an upside roster (median projection) and a floor roster (10th percentile).
- **Trade simulator**  — evaluate a one-for-one swap and get a plain-English verdict: a CBA salary-matching legality check (the over-the-cap ~125% + $250K band), cap impact, projected next-season BPM delta, the change in performance volatility, and a flag for any existing dead cap the outgoing player carries.
- **Multi-year contract valuation** — alongside the single-season comp flag, priceable players on real multi-year deals are valued on total guaranteed dollars over their remaining years against projected value across those same years (the forecast's t+1/t+2/t+3 medians, mapped through the same replacement-anchored value score). The ceiling-cap and no-salary abstentions carry over, and cheap-contract verdicts (Fair-min / dead-money) carry across years too — with flat sub-replacement production the multi-year value is dominated by guaranteed dollars, exactly what the salary-floor model already judged — so an MVP-tier max deal does not spuriously read Overvalued just because it is the largest, longest contract on the board.

## Architecture

```
raw CSVs ──► build_dataset.py ──► clean_roster.csv ──► models.py ──► app.py
(stats,        join on            (one row per         (BPM forecast,   (Streamlit
 salary,       player-season,      player-season,       comps engine,    dashboard)
 injuries)     engineer features)  all features)        PuLP optimizer)
```

## Tech stack

| Layer | Tool |
|:---|:---|
| Data engineering | pandas, numpy |
| Forecasting | scikit-learn gradient-boosted quantile regression |
| Optimization | PuLP (CBC solver) |
| Dashboard | Streamlit + Altair |

## Data

4,860 player-seasons, 2017–2025, joined from three public sources (advanced metrics, salary, and a Prosportstransactions-style injury log). The join is on a normalized player key that strips accents and suffixes and resolves the injury log's slash-separated name aliases.

Multi-year contracts: A fourth source — a cleaned Basketball-Reference contract export — adds forward-looking multi-year terms for the current roster: guaranteed dollars per future season through 2030-31, years remaining, and per-player dead-cap obligations for waived/stretched contracts. The raw export required real cleaning (it arrived as an HTML table mislabeled .xls, with trade/waiver duplicates and dead-cap rows mixed into the player list); the cleaner dedupes exact repeats, splits each multi-team player into one authoritative active contract plus separate flagged dead-cap rows, and keys everything on the same normalized name. 410 of the current roster's players match to multi-year terms on the name key; the unmatched remainder is dominated by 2025 rookies who have a contract but no prior stat season to forecast from — an honest gap, not a silent one, consistent with the abstention philosophy throughout. The single-season headline-cap figure remains the fallback for players the contract file does not cover. Full apron/exception mechanics (the 2023 CBA's two aprons, Bird rights, sign-and-trade rules) are the remaining v3 frontier.

The injury-risk tiers are **severity-weighted**, not event counts. Each IL placement is scored from its free-text note onto an ordinal severity (season-ending = 6, major surgery/tear/fracture = 4, notable sprain/strain = 2, routine day-to-day = 1), and the tier is set from a 3-year rolling severity load plus a small recurrence bump for a knee history. This was a deliberate refinement: an earlier version summed raw IL-event counts, so frequent-but-minor stints (a sore ankle, an illness) pushed a player into High just as fast as a torn ACL, and the tiers skewed heavily High (≈1,466 High / 1,821 Medium / 1,573 Low). Tiering on severity instead of frequency rebalances them to ≈1,219 High / 1,668 Medium / 1,973 Low — High is now the smallest band, as it should be, and a single season-ending injury clears the High threshold on its own while a thin trail of day-to-day stints stays Low. Validation confirms the signal tracks basketball lost rather than turnstile clicks: every player-season containing a season-ending event lands High or Medium (none slip to Low).


## Modeling decision: GBM instead of LSTM

The original project concept considered an LSTM trajectory model for player performance decay. In the final implementation, Parquet Capital uses gradient-boosted quantile regression instead.

That choice is deliberate. The dataset is tabular, relatively small, and contains short player histories, which makes a tree-based model more stable and easier to validate than a deep sequence model. The model still captures trajectory behavior by rolling the full feature vector forward across three seasons: age increases, predicted BPM updates, supporting rate stats move with BPM through fitted sensitivities, aging-curve delta refreshes by age and position, and the injury signal decays over time.

This keeps the model transparent, fast, and defensible while still producing 10th / 50th / 90th percentile forecasts for front-office risk analysis.

## Model performance

**One-step backtest.** Held out on the last two seasons: MAE ≈ 2.15 BPM, with the model calling the direction of a player's change (improve vs. decline) correctly about 64% of the time. The uncertainty band is coverage-calibrated: after scaling, the 10th–90th band covers ≈80% of held-out outcomes (its nominal target), versus ≈74% before calibration.

**Multi-step validation.** 
Because the t+2 / t+3 projections feed the model's own t+1 prediction back in, single-step error understates longer-horizon error. A dedicated roll-forward backtest (evaluate_multistep) trains only on seasons before the evaluation window and rolls each player's full feature vector forward, comparing every horizon against ground truth: **MAE ≈ 1.8 / 2.2 / 2.2 BPM at t+1 / t+2 / t+3**, beating a flat-persistence baseline (≈2.2 / 2.7 / 2.6) at every horizon. The roll-forward degrades gracefully rather than diverging — the aging-curve and feature-advance machinery adds real signal over "assume no change." The **band-widening factors at t+2 / t+3 are calibrated against this same roll-forward** (not assumed √h): each factor is measured as the scale that brings that horizon's rolled band to its 80% coverage target, so all three horizons hit ≈80% empirical coverage rather than only t+1.

**Valuation coverage.**  Contracts are priced on a replacement-anchored value score — BPM mapped onto a strictly-positive, monotone scale where replacement level (BPM ≈ −2.0) maps to a small positive floor — rather than raw BPM. This keeps the value rate stable instead of exploding as production approaches zero. Above-replacement players are **comp-priced:** their $/value rate is compared to similar players (position, age, production). Below-replacement players, where a $/value ratio is noise-dominated, are instead priced on a separate **cheap-contract track** against the league-minimum salary floor — a salary at/near the minimum band reads "Fair (min contract)", a salary above the median NBA wage on sub-replacement production reads "Overpay (dead money)" (the genuine stranded cap). This two-track design lifts verdict coverage from ~32% to ~80% of the league without pricing anyone on a meaningless ratio: when production is flat the salary side carries the verdict, which is the honest thing to key on. Both the dead-money line (median salaried wage) and the min-salary floor (a low salary percentile) are data-driven, so they track the cap environment rather than fixed dollar figures. The only remaining unrated players are no-salary records and the few the comp set cannot serve. The cheap-contract track is itself **backtested** (see below): dead-money flags cost ~5–6× more per delivered unit than fair-minimum deals for the same realized production.

Concretely, of the **1,349 forecasted current players, ≈31%** receive a strict comp-bucket **verdict** (Overvalued / Undervalued / Fair Value) — and the two-track design lifts total coverage to **≈80%** once the cheap-contract verdicts (Fair-min / dead-money, which price flat sub-replacement production on the salary side) are included. The ~31% and ~80% figures are not in tension: the former counts only the comp-ratio buckets, the latter every player who gets any honest verdict. The remaining unrated players are abstentions, not silent gaps: **262 with no salary on record** (a genuine source-data gap) and **12 elite max-tier players** held back as ceiling-capped (the comp pool cannot price them; see Elite max-contract handling below). The 262 (≈19% of forecasted players) is consistent with the 86% salary match rate quoted in Data — the two figures use different denominators: the 86% is over all 4,860 player-seasons, while the 262 counts only each player's single most-recent season, where missing-salary rows concentrate.

## Validation & Testing (v3 additions)

Three additions move the project from plausible to validated. None of them alter the production logic in models.py — they wrap it, so behavior can never drift from what the dashboard serves.

1. The forecaster is now earned, not assumed — validation.py: tune_forecaster()

The gradient-boosted choice is validated against honest baselines under expanding-window cross-validation (every test season is strictly later than every training season — a random K-fold would leak a player's future into their past). Three references are reported side by side:

persistence — predict next BPM = current BPM (no model at all),
ridge — a standardized linear model, a real but simple competitor,
GBM (current defaults) vs. GBM (tuned) over a small, defensible grid
(n_estimators, max_depth, learning_rate, min_samples_leaf).

The winner is the lowest mean out-of-sample MAE, with ties broken toward the simpler model. The run prints two **separate** verdicts so neither is oversold: one on the model family ("GBM beats both baselines" — the real win, GBM ≈ 2.05 vs. persistence ≈ 2.40 and ridge ≈ 2.13), and one on **tuning specifically**. The tuning verdict is honest about magnitude: the best tuned grid point beats the production defaults by only ≈ 0.01–0.02 BPM, which is **within the fold-to-fold** noise floor, so the run reports that tuning did not meaningfully help and that the production defaults were already near-optimal. The win to claim is GBM-over-persistence, not tuned-over-default — and the code says so rather than letting a noise-sized grid improvement read as if tuning earned something.

On the LSTM question: the earlier note about an "LSTM spec" has been resolved in favor of evidence. The quantile-GBM is retained because, under the CV above, it matches or beats the simpler baselines while remaining fast, interpretable, and free of sequence-model training overhead — and a deep sequence model is not justified by the per-player sequence lengths in this dataset (most careers are a handful of seasons). The framework is model-agnostic: tune_forecaster is where a sequence model would be dropped in and made to prove itself on the same folds.

2. Quantile calibration — validation.py: quantile_calibration()


The numeric 80%-coverage target is now backed by a reliability diagram: for each nominal quantile (10/25/50/75/90), the fraction of held-out actuals at or below the model's prediction. On the dashed 45° line = perfectly calibrated. The dashboard renders this in the new Model Validation tab, and the function reports mean |empirical − nominal| as a single calibration score.


3. Backtested valuation decisions — validation.py: backtest_valuations()
The headline test: do the flags actually hold up? For each of the last evaluation seasons, models are trained only on prior seasons, the comp engine flags every contract using the same production value_players logic, and each flag is checked against the player's actual next-season production (held out).

Bucketing realized $/value (real next-season cost per delivered value point) by flag, a working engine should rank Overvalued > Fair > Undervalued — i.e. you pay the most per delivered unit on exactly the contracts it warned about. The function prints this ordering and a CONFIRMED / MIXED verdict, and the dashboard surfaces the same table. This validates what the tool claims to do (catch dead-cap contracts before they hurt), not just BPM MAE.

On held-out next-season outcomes the ranking comes out in exactly the right order — you pay roughly **8.6× more per delivered value point on Overvalued-flagged contracts than on Undervalued ones:**

| Flag | n | Median salary $(in millions) | Median realized value | Median realized $/value ||
|-------|:---:|:---:|:---:|:---:|
| Overvalued | 250 | 14.60 | 2.80 | **6.54** |
| Fair | 80 | 4.60 | 2.60 | 2.58 |
| Undervalued | 99 | 1.99 | 2.80 | **.76** |

Realized production is essentially flat across the three buckets (≈2.6–2.8 value points) while price per delivered unit falls monotonically — the engine is sorting contracts by price efficiency on outcomes it did not see at flag time, which is the whole claim.

The **cheap-contract track is backtested the same way**: among players flagged sub-replacement, "Overpay (dead money)" contracts should cost far more per realized unit than "Fair (min contract)" ones while delivering comparable (low) production. On held-out seasons they do:

| Flag | n | Median salary $(in millions) | Median realized value | Median realized $/value |
|-------|:---:|:---:|:---:|:---:|
| Fair (min contract) | 130 | .92 | .50 | 1.12 |
| Overpay (dead money) | 205 | 4.43 | .50 | **6.33** | 

Same replacement-level output (0.50 value-score on both sides), **5.7× the realized $/value** on the dead-money flags — so the verdicts that lifted coverage from ~32% to ~80% are earned on held-out data, not asserted.


4. Multi-target robustness — multi_target.py: cross_target_agreement()

The whole valuation chain rides on a single forecast target, BPM — one noisy advanced stat (t+1 MAE ≈ 2 BPM is large next to the spread of most rotation players) carrying every Overvalued/Undervalued call. This addition tests robustness to that choice by re-running the same forecast → value-score → comp pipeline on a **second, independent target, VORP** (with its own replacement anchor — a replacement player is ≈ 0 VORP by construction, vs. BPM's −2.0), then measuring how often the two targets land the same verdict. It reports exact agreement, adjacent disagreement (off by one bucket), **direct** contradiction (one target says Overvalued, the other Undervalued — the worst case), and a chance-corrected Cohen's κ. High agreement with rare contradiction means the flags reflect a real value signal rather than a BPM artifact; frequent contradiction would mean verdicts are target-dependent and the run says so. This directly addresses the single-stat dependency that is otherwise the framework's deepest limitation. The VORP track is a genuine replication of the production machinery pointed at a different column, not a parallel reimplementation that could flatter the result.

On the current data this check lands **MIXED, and that is reported as such rather than smoothed over**: BPM and VORP agree on the exact bucket **58%** of the time, are adjacent (off by one) **30%**, and **directly contradict 12%** (κ ≈ 0.35). The honest read is that the target matters more than the model — the value signal is far more robust to a change of model class (next section, κ ≈ 0.52, 3.4% contradiction) than to a change of stat. This is exactly why the per-player VORP agreement label is surfaced in the dashboard: a call where the second target disagrees is shown as soft at the point of use, not hidden behind a clean aggregate. A framework that quietly claimed both checks "confirmed" would be overselling; the asymmetry between them is the actual finding.

5. Cross-MODEL robustness — model_ensemble.py: cross_model_agreement()
The VORP check above swaps the forecast target but keeps the same gradient-boosted machinery, so a systematic bias in that model family would be inherited by both tracks and read as agreement. This addition swaps the **model class** instead: it adds an AgingCurveForecaster — a parameter-free projector that sets next-season BPM to current BPM plus the structural position/age aging delta, with no fitted feature weights and none of the GBM's inductive bias — and runs the unchanged production comp engine on its forecast. The comparison is made on the multi-year flag on purpose: the single-season flag keys off current BPM (shared by both models by construction, so it would trivially agree 100%), whereas the multi-year flag prices the 3-season projection, which is exactly where two model classes genuinely differ. On the current data the two classes reach the same bucket 72% of the time with only **3.4%** direct contradiction and a chance-corrected **κ ≈ 0.52** — higher agreement than the VORP target check **(κ ≈ 0.35)**, and a stronger claim: the value signal survives not just a different stat but a fundamentally different modeling approach. The aging-curve model is also where a future sequence model would slot in as a third independent class


Tests — test_parquet_capital.py

45 targeted tests on the correctness-critical paths where a silent bug would corrupt everything downstream: the normalize_name join key (accents, suffixes, slash-aliases, junk input), severity_score ordinal tiers, the trade_is_legal CBA salary-match boundary, _value_score replacement-anchoring and monotonicity, the _advance_features roll-forward step, the value_players abstentions **and the cheap-contract track (fair-minimum vs dead-money splits, and that no salaried below-replacement player is left unrated)**, the optimize_roster hard constraints (cap, roster size, position minimums, the 35%-of-cap rule), the _scale_for_coverage band calibrator (hits the nominal coverage target; wider residuals demand a wider band), the VORP value-score anchor used by the multi-target check, and **the parameter-free AgingCurveForecaster (aging-delta roll-forward, flat fallback on missing cells, and that its output feeds the comp engine unchanged).**


## Key findings (illustrative)

- Several max-contract veterans are flagged **Overvalued** — paying top-of-market $/value rates while their projected BPM trajectory is flat or declining, the classic dead-cap risk profile.
- The cheap-contract track surfaces **dead money** the comp engine used to leave unrated: sub-replacement players on real salaries (e.g. an injured-out max like Kawhi Leonard at ~$49M on −5.8 BPM, plus mid-tier flat veterans on $20–34M) are flagged **Overpay (dead money)**, while the hundreds of genuine minimum deals correctly read **Fair (min contract)** rather than being lumped together.
- The comps engine surfaces **Undervalued** young players whose $/value rate sits well below positional peers — the "buy low" targets. The optimizer's upside vs. floor rosters expose the risk premium: the projected performance a team sacrifices to build a safer, more predictable roster.
- **Elite max-contract handling**: because the comp pool has no salary tier more expensive than the league max, an elite producer on a max contract is necessarily compared against cheaper players and reads "Overvalued" — the flag would otherwise detect "is on a max deal" as much as "is a bad deal." The engine now separates the two on production, not just price: a top-salary-tier contract whose production is also top-tier is flagged **"Elite (max-tier) — ceiling-capped"**, an explicit abstention, because the comp set literally cannot price it higher. A top-tier salary on non-elite production is left to the normal comp logic and still flags Overvalued — that overpay is real, not an artifact. Both thresholds are data-driven (top-decile salary and BPM within the priced pool) rather than fixed dollar figures, so they track the league's actual cap environment. In practice this cleanly splits the two populations the original tool conflated: MVP-tier max deals (Jokić, Giannis, SGA, Curry, Dončić, Tatum) now abstain as ceiling-capped, while genuine dead-cap max deals (Bradley Beal and John Wall at sub-replacement BPM, Ben Simmons, an aged-out Kemba Walker) remain correctly flagged Overvalued.




## Run locally
Paths are configurable but optional. By default, build_dataset.py and models.py read the three raw CSVs from — and write clean_roster.csv into a parquet_out/ folder next to — the scripts themselves, so the app finds the data no matter what directory you launch it from. No code editing or environment setup is required if the raw CSVs sit alongside the scripts.

pip install -r requirements.txt

# 1. build the dataset (writes parquet_out/clean_roster.csv next to the script)
python build_dataset.py

# 2. build forecasts + valuations; prints backtests + calibration report
python models.py


# 3.validate: tuning vs. baselines, quantile calibration, decision backtest
python validation.py

## 4.  robustness: re-price on a second target (VORP) and check flag agreement
python multi_target.py

## 4b.cross-model robustness: re-price with a parameter-free aging-curve model and check flag agreement across model classes
python model_ensemble.py

# 5. run the test suite (45 tests on the correctness-critical paths)
pytest -q

# 6. launch the dashboard
streamlit run app_NBA.py

To point at custom folders, use flags or environment variables (flags take priority):

# option A: flags
python build_dataset.py --raw ./data --out ./parquet_out
python models.py        --out ./parquet_out
python validation.py    --out ./parquet_out


# option B: env vars
export PARQUET_RAW=./data PARQUET_OUT=./parquet_out
python build_dataset.py
python models.py
streamlit run app_NBA.py

If you launch the app before building the dataset, it fails gracefully with an in-browser message telling you to run build_dataset.py first, rather than a raw traceback.

No raw data? Generate a schema-matched synthetic clean_roster.csv to exercise the
pipeline, validation, and tests:

python make_synthetic.py

Data acquired from these sources:
https://www.kaggle.com/datasets/jacquesoberweis/2016-2025-nba-injury-data;
https://www.kaggle.com/datasets/ratin21/nba-player-stats-and-salaries-2010-2025
https://www.kaggle.com/datasets/jacquesoberweis/2016-2025-nba-player-advanced-season-stats
https://www.basketball-reference.com/contracts/
https://www.basketball-reference.com/contracts/players.html
