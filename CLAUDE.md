# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A KBO (Korean Baseball Organization) pitcher-substitution decision pipeline. Three stages, each a standalone script, run in sequence:

1. **`catboost_train.py`** — trains a 9-class CatBoost classifier predicting plate-appearance (PA) outcome (`1B/2B/3B/HR/BB/SO/OUT/GDP/SF`) from game-state features.
2. **`monte_carlo_simulate.py`** — uses the trained classifier to Monte Carlo-simulate games forward from any state and estimate WPA (Win Probability Added) distributions (`E[WPA]`, `Var[WPA]`, 95% CI) for a given pitcher.
3. **`dqn_agent.py`** / **`dqn_recommend.py`** — trains/uses a DQN agent that chooses among "keep current pitcher" vs. 3 bullpen candidates, optimizing a Markowitz-style utility `U = -E[WPA] - λ·Var[WPA]` (defense's perspective) across several risk-aversion levels λ (manager "personality").

Run order matters: `catboost_train.py` → `monte_carlo_simulate.py` (also runnable as a standalone demo) → `dqn_agent.py` (trains one agent per λ) → `dqn_recommend.py` (loads trained agents for recommendations).

## Commands

No `requirements.txt` exists; dependencies are declared per-script in each file's docstring:

```bash
pip install catboost scikit-learn pandas pyarrow   # catboost_train.py, monte_carlo_simulate.py
pip install torch                                   # dqn_agent.py, dqn_recommend.py
```

Run each stage directly (no CLI args, no test runner/build step in this repo):

```bash
python catboost_train.py           # → catboost_output/ (model + meta json + feature importance)
python monte_carlo_simulate.py     # standalone demo of the simulator; requires catboost_output/ to exist
python dqn_agent.py                # trains agents for λ ∈ {0.0, 0.5, 1.0, 2.0} → dqn_output/
python dqn_recommend.py            # loads dqn_output/*.pth, prints recommendation tables
```

`catboost_output/` and `dqn_output/` are gitignored training artifacts — they must be regenerated locally by running the pipeline in order; they are not checked in.

There is no formal test suite. `data_analysis/methods/we_re_lookup.py` has embedded sanity-check assertions runnable via `python data_analysis/methods/we_re_lookup.py`; the other scripts' `if __name__ == "__main__"` blocks double as smoke-test demos.

## Data — read `HANDOFF_FULL.md` before touching any parquet file

`HANDOFF_FULL.md` is the authoritative spec for the dataset: column semantics, sign conventions, and policy decisions (extra-innings handling, terminal-PA score correction, etc.). Per that document, conventions were re-derived from the data by inference multiple times in the past and got reversed each time — **do not re-infer conventions from the data; look them up in `HANDOFF_FULL.md` (chapters 3–4) or ask.**

Load-bearing conventions from that doc (do not "fix" or re-derive these):
- `half`: `"top"` = away team batting, `"bot"` = home team batting.
- `score_diff_attacker`: **already converted to the batting team's perspective** — do not sign-flip it again.
- `score_diff` (raw): home − away.
- `reward_wpa_computed`: ΔWE from the **attacking team's** perspective — the learning target. Pitcher/defense perspective requires negating it.
- `reward_wpa` (raw, from source site): 79.97% missing — never use it; use `reward_wpa_computed`.
- Splits must be **time-ordered by `game_id`'s leading 8-digit date**, never random — this repo's convention is a fixed 70/15/15 split (see `catboost_train.py`'s "4. 시간순 분할" section for the exact slicing pattern; `HANDOFF_FULL.md` §6 documents the same approach for the newer dataset).
- Extra innings (10th+) are already excluded from the dataset; 9th-inning walk-off / tie-at-9 boundary cases are handled specially (see `HANDOFF_FULL.md` §4-2 and `get_we_with_boundary` in `we_re_lookup.py`).

**Important gotcha**: `HANDOFF_FULL.md` describes a newer, much larger main dataset (`data_analysis/results/all_pa_enriched.parquet`, 558,064 rows across all 10 KBO teams, 2016–2025) and explicitly marks the older one (`hsk_pa_with_wpa.parquet`, ~153 games, one team matchup) as "rollback point only." However, the current code (`catboost_train.py`, `dqn_agent.py`) still points at `data_analysis/results/hsk_pa_with_wpa.parquet`. Confirm with the user before assuming which dataset a given task should target — repointing scripts at `all_pa_enriched.parquet` is a deliberate migration, not a bug fix to make silently.

## Architecture

**`data_analysis/methods/we_re_lookup.py`** — Pure lookup module implementing Run Expectancy (RE) and Win Expectancy (WE) tables transcribed from a published KBO sabermetrics paper (문형우 외 2016). `get_re(out_count, base_state)` looks up a static 24-cell table. `get_we(...)` linearly interpolates between 3rd- and 7th-inning WE tables (the paper only tabulates those two anchor innings) and clips score differential to ±4. `get_we_with_boundary(...)` layers walk-off/extra-innings edge cases on top and is what the rest of the pipeline actually calls. No dependency on the other modules — this is the shared "ground truth" for win probability that both the training-label generation and the Monte Carlo simulator rely on.

**`monte_carlo_simulate.py`** — Defines `GameState` (one plate appearance's starting state: inning/half/outs/base state/score diff/pitcher & batter features) and `apply_pa_result()`, a pure state-transition function that maps a predicted PA outcome class onto runs scored + next base/out state using bitmask base-state arithmetic (1B=bit0, 2B=bit1, 3B=bit2). `MonteCarloSimulator` loads the CatBoost model + its `model_meta.json`, then for a given starting `GameState` runs N parallel "universes" forward PA-by-PA: at each step it **batches all still-live universes into one `predict_proba` call** (not N separate calls) for performance, samples an outcome per universe from the predicted class distribution, applies the transition, and terminates a universe when the game ends (9th-inning walk-off, or `max_innings` cutoff as a safety valve against unbounded extra innings). Final WPA per universe = `WE(end) - WE(start)`, aggregated into `e_wpa`/`var_wpa`/`ci_95`. `compare_pitchers()` wraps this to rank several pitcher candidates by Markowitz utility.

**`catboost_train.py`** — Standalone training script (not imported elsewhere) for the PA-outcome classifier: loads parquet → drops flagged/UNK rows → merges `IBB`/`HBP` into `BB` (keeps `3B`/`GDP`/`SF` separate because they have distinct WPA state transitions) → time-ordered 70/15/15 split by `game_id` date prefix → sqrt-inverse-frequency class weighting → trains `CatBoostClassifier` (`MultiClass` loss) → writes `catboost_output/catboost_pa_classifier.cbm` + `model_meta.json` (also embeds a `wpa_class_mapping` used conceptually by the simulator's transition logic) + feature importances.

**`dqn_agent.py`** — Depends on `monte_carlo_simulate.py` (imports `MonteCarloSimulator`, `GameState`) for its reward signal. Action space per decision point is always exactly 4: index 0 = keep current pitcher, indices 1–3 = three bullpen candidates chosen by `select_bullpen()` using three fixed heuristics — best matchup (`pitcher_vs_batter_avg` min), freshest (`total_pitch_count` min), and a "balanced" middle-of-combined-score pick — after a hard fatigue cutoff (`total_pitch_count >= 120` excluded). `encode_state()` turns a `GameState` + λ into a fixed 10-dim vector fed to `QNetwork` (a plain MLP regressing scalar Q-value per action, evaluated once per candidate rather than outputting a 4-way head). Reward per step = `-E[WPA] - λ·Var[WPA]` from a `MonteCarloSimulator.simulate()` call (defense-perspective Markowitz utility). Trains one independent agent per λ in `LAMBDA_LIST` (0.0=aggressive, 0.5=balanced, 1.0=conservative, 2.0=very conservative), saving each to `dqn_output/dqn_agent_lambda{λ}.pth`.

**`dqn_recommend.py`** — Inference-only: loads one trained agent per λ, reconstructs the same 4-way action set via `build_action_set()`, and prints both the DQN Q-value ranking and a higher-fidelity Monte Carlo re-estimate side by side for a given live game state. `compare_by_lambda()` shows how the recommendation shifts across manager risk profiles.

## Conventions to preserve when editing

- Console output and comments throughout are Korean; match the existing style rather than switching to English mid-file.
- Each script is a self-contained `if __name__ == "__main__"` pipeline stage with heavy `print()` progress/report output — this doubles as the only verification path in the absence of a test suite. Preserve this pattern when adding new stages rather than introducing a separate test framework.
- `GameState` is a plain dataclass threaded by value through transition functions (`apply_pa_result`, `next_half_inning`) — these are pure functions that return new `GameState` instances, not mutators. Keep new transition logic pure for the same reason the Monte Carlo batching works (many parallel `GameState` copies advanced independently).
