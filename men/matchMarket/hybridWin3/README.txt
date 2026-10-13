HYBRIDWIN3 — T20 SECOND-INNINGS WIN PROBABILITY PROJECT
=====================================================

1. PURPOSE AND OVERVIEW
-----------------------
This project predicts whether the batting team will successfully complete a T20
chase, using the current match state across all 120 balls of the second innings.

It combines two approaches:
  A. A ball-outcome model predicts what can happen on the next legal delivery.
     A simulator uses those probabilities to estimate the chance of winning.
  B. A direct win model predicts the chance of winning straight from match state.

The simulation probability is calibrated against historical outcomes. A learned,
state-dependent blend combines it with the direct probability. The final output
column is matchWinProbability, expressed from 0 to 1.

The standard state contains:
  runsRequired          Runs still needed to win.
  inningBallsRemaining  Legal balls remaining.
  totalInningWickets    Wickets already lost, from 0 to 9.
  ord                   Striker's batting order, from 1 to 11.
  nonStrikerOrd         Non-striker's batting order, from 1 to 11.

These are match-state models, without player identity inputs. The ball model
uses the striker's order; the simulator tracks both batters and strike changes.
The direct model and blend use both batting orders.

2. INSTALLATION AND REQUIRED FILES
----------------------------------
Place the hybridWin3 folder inside your existing armadaCricket project at:
  men/matchMarket/hybridWin3

Keep the existing project-root paths.py, which must expose PROJECT_ROOT. All
project reads and writes use PROJECT_ROOT rather than the working directory.
Use your Python 3.10 cricket environment and install requirements.txt there.
The requirements include NumPy, pandas, SciPy, scikit-learn 1.7.1, CatBoost,
Numba, matplotlib and threadpoolctl.

Required inputs supplied in the minimal package:
  data/dataClean.csv
    Historical delivery data, including match IDs, match states, second-innings
    win outcomes, batsman runs and wicket outcomes. The ball model expects the
    supplied cleaned legal-delivery data; this pipeline is not a raw-data cleaner.
  auxiliaries/1_chaseLookup.csv
    Provides the old neural state coordinates above 24 balls remaining and old
    neural probabilities for comparison. Neural probabilities are not predictors.
  auxiliaries/6_masterLookup.csv
    Provides the existing extras inputs and remaining-runs resource mean/standard
    deviation used by the simulator and direct model.

The cv/ and outputs/ folders are supplied empty. Keep both folders present:
these scripts write into them and do not create missing directories. No saved
models or predictions are required for the first complete run.

3. RUN ORDER
-------------
Run the five scripts in numerical order, waiting for each to finish successfully:

  1_deathBallProbs.py
    Train the full-innings ball-outcome models and generate probability lookups.
  2_simModel.py
    Simulate chases and calculate exact final-over win probabilities.
  3_directWinModel.py
    Fit the direct win models and generate historical/production predictions.
  4_blendWinProbs.py
    Fit simulation calibration and blending, then generate final probabilities.
  5_simModelAnalysis.py
    Evaluate historical predictions, export tables and display charts.

The deathBall filenames are retained, but the model covers all 20 overs.
For a fresh package, script 5 cannot run until scripts 1–4 have generated outputs.
Once a full run is complete, run script 5 alone to inspect existing results.
Plots use plt.show(), suitable for display in PyCharm.

4. BALL-OUTCOME MODEL — SCRIPT 1
--------------------------------
Model type: multiclass CatBoost, a gradient-boosted decision-tree classifier.
Its target is the joint next-ball outcome: batsman runs AND whether a wicket fell.

There are 13 possible categories: runs 0–6 paired with wicket/no wicket, excluding
6 runs with a wicket. Examples are 0_noWicket, 0_wicket and 1_noWicket. Modelling
them jointly preserves the relationship between runs and dismissals. Probabilities
sum to one; runs and wicket probabilities can be obtained by summing categories.
Extras are handled separately in script 2 rather than added to this target.

Inputs include runs required, balls remaining, wickets lost, striker order,
required runs per ball, log scoring pressure, position within the over, whether
the striker is the newest batter, tail-order indicators and derived interactions.
Explicit powerplay, death-phase and log-balls features cover innings phase.

Selected parameters:
  iterations: 1200; tree depth: 5; learning rate: 0.035; L2 leaf penalty: 100.
Trees are fitted sequentially to improve multiclass loss. The penalty limits
complexity and helps reduce overfitting.

Training uses eligible historical second-innings deliveries, not simulated labels.
The script filters missing/invalid states and outcomes and excludes certain
inconsistent matches. Each eligible historical delivery receives an out-of-fold
probability vector. Separate full-data models support production predictions.

5. SIMULATION AND EXACT ENDGAME — SCRIPT 2
------------------------------------------
Model type: a probabilistic chase simulator driven by the fitted ball model.
It is not another classifier fitted directly to match-win labels.

For each starting situation, the simulator tracks runs required, legal balls,
wickets and both batters. It samples outcomes, applies the existing extras and
scoring rules, updates the state and strike, and repeats until the chase succeeds
or the innings ends. Successful chases divided by simulated paths gives raw
simulation win probability. Ties are not counted as successful completed chases;
the project does not model a Super Over.

The supplied pipeline uses 1,000 Monte Carlo paths per starting state with
7–120 balls remaining. Research experiments used 100 paths. Monte Carlo introduces
sampling variation, especially when differences in win probability are small.
Numba compiles the simulation loops for speed.

With 1–6 balls remaining, the script integrates possible outcomes rather than
sampling paths. This is exact under the retained transition/extras rules, apart
from the existing truncation at 12 invalid deliveries and the state-lookup
approximations. The omitted-path probability bound is reported. It does not
invent a new discrete extras distribution: existing fractional extras inputs
are retained. simSample=0 on these rows denotes integration, not zero wins.

6. DIRECT WIN MODEL — SCRIPT 3
------------------------------
Model type: regularised logistic regression using quadratic B-spline bases and
tensor interactions. It predicts the binary match outcome directly.

Splines allow curved relationships instead of a single straight-line effect.
Interactions allow the effect of runs required to depend on balls and wickets.
Curvature penalties encourage smooth surfaces. ALPHA is 0.0001.

Two component models are fitted:
  Resource component:
    Uses (runs required minus expected remaining runs) / remaining-runs standard
    deviation as its main difficulty coordinate, plus balls, wickets and orders.
    The mean/standard deviation come from the supplied resource lookup.
  State component:
    Uses log runs required as its main difficulty coordinate, plus balls,
    wickets and orders, without depending on resource lookup availability.

The model includes richer runs/difficulty–balls–wickets tensor interactions and
balls/order and wickets/order interactions. Where both components are available,
their probabilities are averaged equally. Where resource values are unavailable,
the state-only prediction remains available; missing resource values are not
invented. The blend subsequently restricts coverage to valid resource states.

Endgame anchor:
Within the final six balls, the direct output is progressively anchored to the
same-fold exact endgame probability from script 2. Spline weight is
  clip((balls remaining - 1) / 5, 0, 1).
At six balls the spline retains full weight; at one ball the exact endgame
probability takes full weight. This is separate from script 4's blend taper.

An optional final prediction cell accepts statesToPredict.csv in the project
folder and writes customDirectWinProbabilities.csv using saved coefficients.
It predicts the direct component, not the final hybrid blend. Use it only after
training and generating the required coefficient, resource and endgame files.

7. CALIBRATION AND BLEND — SCRIPT 4
-----------------------------------
Simulation calibration type: regularised logistic calibration.
It transforms raw simulation log-odds with learned intercept/slope corrections,
including full-innings horizon and late-innings effects. This corrects systematic
probability bias using historical match results. Penalty: 0.003.

Blend type: a learned logistic gating model producing a convex weighted average.
  final probability = directWeight * direct probability
                    + simulationWeight * calibrated simulation probability
  simulationWeight = 1 - directWeight.

The gate uses scoring pressure, balls remaining, wickets lost and both batting
orders, with horizon/final-over features. Wicket and non-striker effects are
reduced near the finish. Coefficients minimise historical log loss with a penalty
of 0.0003. A best constant blend weight is learned first, providing the shrinkage
centre; it is not a fixed 25% direct preference.

The direct weight also has a fixed endgame taper:
  directWeight = learned gate * clip((balls remaining - 1) / 12, 0, 1).
Thus the maximum direct weight is 100% at 13+ balls, 91.7% at 12 balls, 41.7% at
6 balls, 8.3% at 2 balls and zero at 1 ball. Actual weights can be lower. On the
last ball the final blend uses 100% calibrated simulation, which can differ from
the raw exact probability because calibration is still applied.

Calibration/blending slightly reduce the influence of matches contributing many
rows, using inverse match-row-count weights raised to power 0.1. This is mild
match adjustment, not full equal-match training weighting.

8. TRAINING, CROSS-VALIDATION AND PRODUCTION
-------------------------------------------
The pipeline uses five folds split by WHOLE MATCH, stratified by chase outcome.
All deliveries from one match stay together. This avoids putting some balls of
a test match into the fitting data for its predictions.

Base-model fitting consists of:
  Five outer fits, each excluding one fold.
  Ten paired inner fits, each excluding a different pair of folds.
  One full-data production fit.
The ball stage therefore fits 16 models. The direct stage fits resource and state
components for the corresponding jobs.

For an outer test fold, calibration/blending train on predictions generated by
paired inner fits. Those fits exclude both the outer test fold and the inner
validation fold. Therefore the outer match is excluded from base-model fitting,
calibration and blending. Using ordinary global OOF predictions for that inner
step would not provide the same protection.

OOF means out of fold: historical predictions made without fitting on that match.
Use these for evaluation. Production base models fit all eligible historical data;
production calibration/blending learn from the outer OOF prediction set, then
apply to the full-data base-model predictions. Use production probabilities for
the state lookup, not to score historical training accuracy.

Recipe selection compared ball, direct and calibration/blending alternatives.
Experiments used folds 0–3, with fold 4 excluded from this remodel's selection.
Running scripts 1–5 fits the selected recipe; it does not repeat that search.
All-fold OOF scores include folds used in recipe selection. Fold 4 is the cleaner
selection check, but previous project development used this historical dataset.
The supplied fixed extras/resource lookups have unknown upstream training
provenance. These results are not a new external or temporal holdout test.
The old neural probabilities are an in-sample benchmark, not an untouched rival.

9. GRID COVERAGE
----------------
The number of starting states is determined by the supplied lookup grid and
eligible batting-order combinations; consult the generated run metadata.
Above 24 balls, runs/balls/wicket coordinates are copied from 1_chaseLookup.csv.
At 1–24 balls the previous band-based grid is retained. Eligible batting-order
pairs are represented in both striker orientations.

Additional internal ball-probability states support transitions such as rapid
collapses. productionGrid=False marks transition support, not extra starting
states exported as production situations. The nearest-run transition lookup,
extras rules, strike rotation and final-over calculation are retained.
Historical states outside the grid can have direct predictions but are not
silently added to the simulation/blend production grid. Missing resource coverage
can further restrict blend rows. Compare models on identical eligible rows.

10. OUTPUTS AND ANALYSIS — SCRIPT 5
-----------------------------------
Important generated files in outputs/:
  deathBallModel.cbm                 Full-data fitted CatBoost model.
  deathBallProbs.csv                 Production ball-probability lookup.
  deathBallProbs_oof.csv             Historical unseen ball probabilities.
  endgameWinProbabilities.csv        Integrated final-over probabilities.
  chaseSimSituations.csv             Production raw simulation probabilities.
  chaseSimSituations_oof.csv         Historical unseen simulation predictions.
  directWinCoefficients.csv          Portable fitted spline coefficients.
  directWinProbabilities.csv         Production direct predictions.
  directWinProbabilities_oof.csv     Historical unseen direct predictions.
  blendCoefficients.csv              Calibration and blend coefficients.
  chaseWinProbabilities.csv          Final production hybrid probabilities.
  chaseWinProbabilities_oof.csv      Final historical unseen hybrid predictions.
  productionComparison.csv          Production component comparison.
  runInfo.csv                       Run settings and readiness metadata.

Use matchWinProbability as the final prediction. The final tables also expose
component probabilities and directWeight/simulationWeight.

Script 5 reports log loss, Brier score, equal-match log loss, mean probability and
actual win rate, including comparisons by fold/state and innings segment. Lower
log loss and Brier are better. Equal-match log loss first averages within each
match, then across matches. Ball analysis reports joint-event/category losses and
predicted versus observed mean runs. Charts include calibration and blended
5-by-2 PiYG heatmaps, one panel for each wicket count. Heatmaps depend on the
batting-order selection/aggregation in the analysis script.

11. CHECKPOINTS, RESTARTS AND INPUT CHANGES
------------------------------------------
cv/ starts empty but fills with fold models, paired-inner predictions, coefficient
checkpoints and run metadata. These files are needed by later training stages;
do not empty cv/ between scripts 1 and 4.

Scripts 1 and 3 reuse matching fitted-model/coefficient checkpoints. Script 1
regenerates probability CSVs even when it reuses fitted models. Not every stage
is an automatic resume-from-last-row job; rerunning simulation can repeat work.
Data, fold, grid and resource hashes plus readiness flags guard against mixing
incompatible stage outputs. If an input or relevant configuration changes,
rerun the affected stage and downstream stages. For changed training data or grid,
run 1–5 again. Do not combine files from different project versions.

Script 1 defaults to two concurrent fits with four CatBoost threads per fit.
PARALLEL_FITS=1 reduces simultaneous fitting load. Script 2 defaults to four
Numba threads unless NUMBA_NUM_THREADS is supplied. Runtime depends strongly on
hardware and the size of generated probability grids; intermediate CSVs can be
much larger than the minimal download.
