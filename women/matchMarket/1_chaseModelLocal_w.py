import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold, GroupKFold
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from paths import PROJECT_ROOT
from architectureTesting import (evaluate_architecture_cv, sweep_architecture, final_test_check,
                                  choose_test_groups, apply_test_split, report_bias_tables)


# ============================================================================================
# MODEL/ARCHITECTURE NOTES (women's local experimental file - production 1_chaseModel_w.py is untouched)
#
# This file mirrors the structure landed on in 1_chaseModelLocal.py (men's), but every architecture below
# was independently re-decided from evidence on the women's dataset (~175k training rows / ~1900 matches),
# rather than copied from men's (~487k rows / ~7500 matches) - a model with real spare capacity is more
# dangerous on a dataset with a quarter as many matches, so nothing was assumed to transfer.
#
# Models in this file, and why each exists:
#
# - Main: the only model that feeds the final chaseLookup['m_chaseWin%'] output. Trained across the WHOLE
#   innings on Death's old feature set - ['runsRequired', 'ratioRequired', 'daysGroup', 'inningBallsRemaining']
#   - same reasoning as men's: this represents what "Death extended to the full innings" used to be, and
#   ratioRequired's wickets-dependent denominator still carries wicket information even without an explicit
#   totalInningWickets feature.
#
# - Death (old family) and LastOver: REMOVED, for the same reason as men's - Main absorbed what Death-extended
#   used to provide, and LastOver was dropped in the same simplification so Main alone drives the output from
#   ball 120 down to ball 1. trainDataDeath (<36-ball slice) is kept only as a data source for the new-family
#   Death model below.
#
# - Death (new family, modelDeathNew, feeds chaseLookupNew_w.csv only): same purpose as men's - tests
#   runsRequiredStd + RA_Sum as two separate inputs vs the pre-combined runsRequiredAdj. Kept out of
#   production for the same reason as men's (RA_Sum not yet trustworthy enough, no real per-state RA_Sum in
#   chaseLookup - see the RA_Sum scenario grid below).
#
# Why (64,) for Main, NOT men's (64,32):
#   A 7-architecture CV sweep (GroupKFold by match, seeds 42/7/123) showed (64,32) is clearly worse here:
#     (64,)     held-out log-loss 0.3962, overfit gap 0.0009
#     (128,)    held-out log-loss 0.3972, overfit gap 0.0031
#     (32,16)   held-out log-loss 0.3995, overfit gap 0.0029
#     (64,32)   held-out log-loss 0.4055, overfit gap 0.0027   <- worst of the four, ~3x (64,)'s overfit gap
#   (64,32) did look best on an in-sample bias check (smallest actual-vs-predicted gap by year/ball-bucket),
#   but that's the overfitting showing through, not real calibration - a higher-capacity net will always fit
#   its own training data more closely; the CV numbers (scored on held-out matches) are the trustworthy
#   signal, and they say (64,) generalises best and most consistently. A side-by-side check at 120 balls
#   remaining, binned by runsRequired, also showed no consistent edge for (64,32) (won some bins, lost others
#   by more) and confirmed all three candidates beat the CURRENT PRODUCTION Main (4,2) almost everywhere.
#
# Why (32,16) for the new-family Death model, NOT men's (64,32):
#   The first pass compared old vs new family at a fixed (8,4) (men's methodology) and new family looked
#   like a clear loss (0.421 vs old family's 0.338). That was misleading - (8,4) simply isn't enough capacity
#   to extract signal from the extra RA_Sum input. Sweeping architectures properly for BOTH feature sets
#   showed new family's best ((32,16), 0.286 held-out log-loss) clearly beats old family's best
#   ((64,32), 0.312) - a reversal of the fixed-architecture conclusion, and a reminder that a feature-family
#   comparison at one arbitrary architecture can hide a real effect if that architecture starves the richer
#   feature set of capacity.
#
# Architecture sweep methodology is now shared with men's (architectureTesting.py) - see
# 1_chaseModelLocal.py's header note for the full protocol description. Both files run the identical
# test: candidates are always current +/- one step (each layer -50%/+50%, same depth), scored
# with seeded GroupKFold CV on a train+val pool, with a final test set held out and touched only once
# per architecture decision.
# ============================================================================================


# quick CV harness to sanity-check hidden_layer_sizes choices before committing to them below - only
# sweeps Main and DeathNew, the two models that actually feed a real output; old-family Death was
# removed from production so there's nothing live to sweep an architecture for there
RUN_ARCHITECTURE_DIAGNOSTIC = True
# set to False to skip fitting the new-family (runsRequiredStd+RA_Sum) Death model and building chaseLookupNew_w -
# only 1_chaseLookup_w.csv (the old family) gets produced when this is off
RUN_NEW_FAMILY_LOOKUP = True
DIAGNOSTIC_SEEDS = [42, 7, 123, 2024, 99]
# matches held out ONCE per run and never touched until final_test_check() at the very end - you never
# pick an architecture based on how it does on these matches, only on the sweep's CV validation score
TEST_FRACTION = 0.2
# states the model is already near-certain about aren't interesting - log loss, bias, and every other
# number reported below (train AND held-out/test) is scored ONLY on rows whose OWN predicted
# probability falls in this range, not just filtered for display. Set to None to score everything.
PROB_RANGE = (0.10, 0.90)
# THE two production architectures - used both as the "current" baseline when RUN_ARCHITECTURE_DIAGNOSTIC
# sweeps current/-50%/+50%, AND as what actually gets fit when it's off. After a diagnostic run picks a
# different winner, update these to match and the next (non-diagnostic) run deploys it - one place to change.
MAIN_ARCHITECTURE = (32,)
DEATHNEW_ARCHITECTURE = (16, 8)
# caps the TRAINING side of each CV fold for speed (see evaluate_architecture_cv). Checked 2026-08-24:
# Main's trainval domain is ~156k rows (6.2x the old 25000 default), DeathNew's is ~36k (1.4x) - both
# were being subsampled more than necessary given how much smaller women's dataset is than men's.
# Raised to fully cover DeathNew (no more subsampling loss there) and give Main meaningfully more too,
# without the runaway-runtime risk a similar increase would carry for men's much bigger Main domain.
MAX_SAMPLES = 40000


# feature experiment: does giving the model runsRequiredStd + RA_Sum as two separate inputs (rather than the
# single pre-combined runsRequiredAdj = runsRequiredStd - RA_Sum) help - i.e. can the network learn a better
# combination than a straight subtraction? Same architecture/scoring as the production diagnostic above, so
# this isolates the feature-representation question from the node-count question.
def report_feature_comparison(name, variants, y, scoreMask, groups, hidden_layer_sizes=(8, 4), prob_range=PROB_RANGE):
    rows = []
    for label, X in variants.items():
        seedRuns = evaluate_architecture_cv(X, y, scoreMask, groups, hidden_layer_sizes, seeds=DIAGNOSTIC_SEEDS,
                                             prob_range=prob_range)
        trainLosses = [r['trainLoss'] for r in seedRuns]
        heldoutLosses = [r['heldoutLoss'] for r in seedRuns]
        rows.append({
            'features': label,
            'heldout_mean': np.mean(heldoutLosses),
            'heldout_std': np.std(heldoutLosses),
            'heldout_worst': np.max(heldoutLosses),
            'train_mean': np.mean(trainLosses),
        })
    results = pd.DataFrame(rows).sort_values('heldout_mean').reset_index(drop=True)
    print(f"\n--- feature comparison: {name} (hidden_layer_sizes={hidden_layer_sizes}, n={len(next(iter(variants.values())))}, seeds={DIAGNOSTIC_SEEDS}) ---")
    print(results.to_string(index=False))


# import
trainData = pd.read_csv(PROJECT_ROOT / 'women/expBall&runsToCome/data/dataClean_w.csv', parse_dates=['date'])
trainData = trainData[trainData['competition'] != "The Hundred (Women's Comp)"]
masterLookup = pd.read_csv(PROJECT_ROOT / 'women/expBall&runsToCome/outputs/5_masterLookup_w.csv')
chaseSituations = pd.read_csv(PROJECT_ROOT / 'women/matchMarket/auxiliaries/chaseSituationBuilder_w.csv')
# chaseLookupLive = pd.read_csv(PROJECT_ROOT / 'women/matchMarket/outputs/1_chaseLookupLive_w.csv')


# drop nans from adj
trainData = trainData.dropna(axis=0, subset=['runsRequiredAdj'])
# now when running for adj simply change std runs required to adj
trainData['runsRequiredStd'] = trainData['runsRequired']
# take out the below 2 lines when running standard runs model
trainData['runsRequired'] = trainData['runsRequiredAdj']
# round to nearest int
trainData['runsRequired'] = trainData['runsRequired'].round()


# Create a new dataframe with expanded rows from the max runs required defined in chase situation builder
chaseSituationsRows = []
for _, row in chaseSituations.iterrows():
    for runs in range(1, row['maxRunsRequired'] + 1):
        chaseSituationsRows.append({
            'inningBallNumber': row['inningBallNumber'],
            'inningBallsRemaining': row['inningBallsRemaining'],
            'totalInningWickets': row['totalInningWickets'],
            'runsRequired': runs
        })
# Create the new dataframe with the expanded out rows
chaseSituations = pd.DataFrame(chaseSituationsRows)
chaseSituations = chaseSituations.sort_values(by=['inningBallsRemaining', 'runsRequired', 'totalInningWickets']).reset_index(drop=True)


# we only want innings 2 for the chase predictions, and shuffle the data
trainData = trainData[trainData['inningNumber'] == 2]
trainData = trainData.sample(frac=1, random_state=42).reset_index(drop=True)
# we need to remove duplicates in runs to come so just select batting order 1
masterLookup = masterLookup[(masterLookup['ord'] == 1) & (masterLookup['daysGroup'] == 10)]
# merge in runs to come
trainData = trainData.merge(masterLookup.loc[:, ['totalInningRunsToComeSimBiasSplineYear', 'totalInningWickets', 'inningBallNumber', 'totalInningValidBallsFacedToCome', 'bowledOut', 'sample']].rename(columns={'sample': 'ballWicketSample'}), how='left', on=['totalInningWickets', 'inningBallNumber'])
# ballWicketSample = sample size behind the (wickets, ball) state, kept on every row so it can be used to mask out unreliable states when scoring/checking the model (not when training it)
# create a ratio of runs to come to be used as a predictor, drop any nans
trainData['ratioRequired'] = trainData['runsRequired'] / trainData['totalInningRunsToComeSimBiasSplineYear'] # changing to include year
trainData['ratioRequiredStd'] = trainData['runsRequiredStd'] / trainData['totalInningRunsToComeSimBiasSplineYear']
trainData = trainData.dropna(axis=0, subset=['ratioRequired'])
trainData = trainData.dropna(axis=0, subset=['ratioRequiredStd'])


# create an empty dataframe
chaseLookup = pd.pivot_table(trainData, values=['sample', 'chaseWin', 'totalInningRunsToCome', 'totalInningWicketsToCome', 'runsRequiredStd'],
                            index=['totalInningWickets', 'inningBallNumber', 'runsRequired'],
                            aggfunc={'sample': 'sum', 'chaseWin': 'sum', 'totalInningRunsToCome': 'mean', 'totalInningWicketsToCome': 'mean', 'runsRequiredStd': 'mean'}).reset_index()
chaseLookup['chaseWin%'] = chaseLookup['chaseWin'] / chaseLookup['sample']
chaseLookup = chaseSituations.merge(chaseLookup, how='left', on=['totalInningWickets', 'inningBallNumber', 'runsRequired'])
chaseLookup = chaseLookup.rename(columns={'sample': 'chaseSample'})
chaseLookup = chaseLookup.merge(masterLookup.loc[:, ['totalInningWickets', 'inningBallNumber', 'sample', 'totalInningRunsToComeSimBiasSplineYear', 'totalInningValidBallsFacedToCome', 'bowledOut']], how='left', on=['totalInningWickets', 'inningBallNumber'])
chaseLookup = chaseLookup.rename(columns={'sample': 'ballWicketSample'})
chaseLookup['ratioRequired'] = chaseLookup['runsRequired'] / chaseLookup['totalInningRunsToComeSimBiasSplineYear'] # changing to include year
chaseLookup['daysGroup'] = 12.7

chaseLookup = chaseLookup.dropna(axis=0, subset=['totalInningRunsToComeSimBiasSplineYear']).reset_index(drop=True)
chaseLookup['in'] = 1

# remove chases which are effectively lost
trainData = trainData.merge(chaseLookup.loc[:, ['in', 'totalInningWickets', 'runsRequired', 'inningBallNumber']], how='left', on=['totalInningWickets', 'runsRequired', 'inningBallNumber'])
trainData = trainData[trainData['in'] == 1]


# start of innings model - trained on Death's inputs across the whole innings (no separate Death/LastOver models)
trainDataMain = trainData.copy()

# prepare the data
y = trainDataMain['chaseWin']
X_std = trainDataMain[['runsRequired', 'ratioRequired', 'daysGroup', 'inningBallsRemaining']]
if RUN_ARCHITECTURE_DIAGNOSTIC:
    # test matches held out ONCE, shared with the DeathNew sweep further down so no model's
    # "test" data leaks into another model's training/validation via a different split
    testMatchIds = choose_test_groups(trainData['matchID'].unique(), test_fraction=TEST_FRACTION, random_state=42)
    trainvalMain, testMain = apply_test_split(trainDataMain, 'matchID', testMatchIds)
    summaryMain, _, _, _ = sweep_architecture(
        'Main', trainvalMain[['runsRequired', 'ratioRequired', 'daysGroup', 'inningBallsRemaining']],
        trainvalMain['chaseWin'], trainvalMain['ballWicketSample'] >= 100, trainvalMain['matchID'],
        current=MAIN_ARCHITECTURE, seeds=DIAGNOSTIC_SEEDS, prob_range=PROB_RANGE, max_samples=MAX_SAMPLES,
        wickets=trainvalMain['totalInningWickets'], ballsRemaining=trainvalMain['inningBallsRemaining'],
        output_dir=PROJECT_ROOT / 'women/matchMarket/outputs',
        # women's Main is currently 1-layer, unlike men's 2-layer (128,64) - checking whether depth
        # helps here too. (64,32) follows the same "second layer = half the first" convention already
        # used throughout this file's other architectures.
        extra_candidates={'2-layer': (64, 32)})
    winnerMain = summaryMain.iloc[0]['architecture']
    testMainMasked = testMain[testMain['ballWicketSample'] >= 100]
    final_test_check(
        'Main', trainvalMain[['runsRequired', 'ratioRequired', 'daysGroup', 'inningBallsRemaining']], trainvalMain['chaseWin'],
        testMainMasked[['runsRequired', 'ratioRequired', 'daysGroup', 'inningBallsRemaining']], testMainMasked['chaseWin'],
        testMainMasked['totalInningWickets'], testMainMasked['inningBallsRemaining'], chosen_architecture=winnerMain,
        prob_range=PROB_RANGE, output_dir=PROJECT_ROOT / 'women/matchMarket/outputs')
if not RUN_ARCHITECTURE_DIAGNOSTIC:
    scaler = StandardScaler()
    scaler.fit(X_std)
    X_std = scaler.transform(X_std)

    # (64,) rather than men's (64,32) - a 7-architecture CV sweep found (64,32) clearly overfits on women's smaller
    # sample (~1900 matches vs men's ~7500): worse held-out log-loss (0.4055 vs 0.3962) and a ~3x larger train/held-out
    # gap. (64,)'s apparent in-sample bias disadvantage vs (64,32) is the overfitting showing through, not a real edge.
    model = MLPClassifier(hidden_layer_sizes=MAIN_ARCHITECTURE, random_state=42, activation='logistic', batch_size='auto', learning_rate='constant', max_iter=5000, early_stopping=False, learning_rate_init=0.001)
    model.fit(X_std, y)
    trainDataMain['m_chaseWin%Main'] = model.predict_proba(X_std)[:, 1]

    # real (full-sample) bias check - this fit used 100% of the data (no held-out split), so its own
    # predictions already give the full-sample bias tables the diagnostic sweep can only approximate
    # with a limited sample; no extra fit needed
    report_bias_tables(
        'Main', trainDataMain['m_chaseWin%Main'].values, trainDataMain['chaseWin'].values,
        trainDataMain['totalInningWickets'].values, trainDataMain['inningBallsRemaining'].values,
        start_balls_remaining=120, start_wickets=0, prob_range=PROB_RANGE,
        output_dir=PROJECT_ROOT / 'women/matchMarket/outputs', output_prefix='fullSampleBiasCheck',
        log_loss_label=f'full-sample (in-sample) log loss (architecture={MAIN_ARCHITECTURE})')

    # now predict the chase situations outside of training
    chaseLookupMain = chaseLookup.copy()
    X = chaseLookupMain[['runsRequired', 'ratioRequired', 'daysGroup', 'inningBallsRemaining']]
    X = scaler.transform(X)
    chaseLookupMain['m_chaseWin%Main'] = model.predict_proba(X)[:, 1]

# kept around only so the new-family Death model below (and its RA_Sum grid) has a data source to train on -
# no old-family Death model is fit here any more, and nothing from it feeds the blend
trainDataDeath = trainData.copy()
trainDataDeath = trainDataDeath[(trainDataDeath['inningBallsRemaining'] < 36)]

if RUN_ARCHITECTURE_DIAGNOSTIC:
    y_death = trainDataDeath['chaseWin']
    X_stdDeath = trainDataDeath[['runsRequired', 'ratioRequired', 'daysGroup', 'inningBallsRemaining']]
    # no architecture sweep for old-family Death any more - it was removed from production (see the
    # header note), so there's no live "current" architecture left to sweep +/- one step from. The
    # feature-representation question below (runsRequiredAdj vs runsRequiredStd+RA_Sum) still matters
    # for DeathNew, so that stays.
    report_feature_comparison(
        'Death',
        {
            'runsRequiredAdj (current)': X_stdDeath,
            'runsRequiredStd + RA_Sum (new)': trainDataDeath[['runsRequiredStd', 'RA_Sum', 'ratioRequiredStd', 'daysGroup', 'inningBallsRemaining']],
        },
        y_death, (trainDataDeath['ballWicketSample'] >= 100) & (trainDataDeath['inningBallsRemaining'] <= 30), trainDataDeath['matchID']
    )

if RUN_NEW_FAMILY_LOOKUP:
    # ---- new-family models (runsRequiredStd + RA_Sum + ratioRequiredStd) and their own prediction table ----
    # chaseLookup has no real RA_Sum (its states are generic, not tied to specific batters/bowlers/pitch), so rather
    # than predicting once at a fake "neutral" RA_Sum=0, build a grid of RA_Sum scenarios and cross it with chaseLookup
    # (balls 1-30 only, the new family's actual domain) to produce a lookup that varies by RA_Sum scenario as well.
    # Kept entirely separate from chaseLookup, which continues to serve the old family (Main only) exactly as before.
    deathFeaturesNew = ['runsRequiredStd', 'RA_Sum', 'ratioRequiredStd', 'daysGroup', 'inningBallsRemaining']

    if RUN_ARCHITECTURE_DIAGNOSTIC:
        # same protocol as Main (shared architectureTesting.py) - 120/0 isn't in Death's domain (<36
        # balls), so use 12 balls remaining instead, with the modal wickets count at that ball count
        # auto-detected from the real data (start_wickets=None) rather than a hand-picked guess
        trainvalDeath, testDeath = apply_test_split(trainDataDeath, 'matchID', testMatchIds)
        summaryDeathNew, _, _, _ = sweep_architecture(
            'DeathNew', trainvalDeath[deathFeaturesNew], trainvalDeath['chaseWin'],
            trainvalDeath['ballWicketSample'] >= 100, trainvalDeath['matchID'],
            current=DEATHNEW_ARCHITECTURE, seeds=DIAGNOSTIC_SEEDS, prob_range=PROB_RANGE, max_samples=MAX_SAMPLES,
            wickets=trainvalDeath['totalInningWickets'],
            ballsRemaining=trainvalDeath['inningBallsRemaining'], start_balls_remaining=12, start_wickets=None,
            output_dir=PROJECT_ROOT / 'women/matchMarket/outputs')
        winnerDeathNew = summaryDeathNew.iloc[0]['architecture']
        testDeathMasked = testDeath[testDeath['ballWicketSample'] >= 100]
        final_test_check(
            'DeathNew', trainvalDeath[deathFeaturesNew], trainvalDeath['chaseWin'],
            testDeathMasked[deathFeaturesNew], testDeathMasked['chaseWin'],
            testDeathMasked['totalInningWickets'], testDeathMasked['inningBallsRemaining'],
            chosen_architecture=winnerDeathNew, start_balls_remaining=12, start_wickets=None, prob_range=PROB_RANGE,
            output_dir=PROJECT_ROOT / 'women/matchMarket/outputs')

    if not RUN_ARCHITECTURE_DIAGNOSTIC:
        # (32,16) rather than men's (64,32) - a 7-architecture sweep found (32,16) clearly best for women's smaller
        # sample (0.286 held-out log-loss vs the old family's best of 0.312 at (64,32)); (64,32) does worse here (0.319).
        scalerDeathNew = StandardScaler()
        X_deathNew = scalerDeathNew.fit_transform(trainDataDeath[deathFeaturesNew])
        modelDeathNew = MLPClassifier(hidden_layer_sizes=DEATHNEW_ARCHITECTURE, random_state=42, activation='logistic', batch_size='auto', learning_rate='constant', max_iter=5000, early_stopping=False, learning_rate_init=0.001)
        modelDeathNew.fit(X_deathNew, trainDataDeath['chaseWin'])

        # real (full-sample) bias check - same reasoning as Main's: this fit already used 100% of
        # trainDataDeath, so its own predictions give the full-sample bias tables for free
        report_bias_tables(
            'DeathNew', modelDeathNew.predict_proba(X_deathNew)[:, 1], trainDataDeath['chaseWin'].values,
            trainDataDeath['totalInningWickets'].values, trainDataDeath['inningBallsRemaining'].values,
            start_balls_remaining=12, start_wickets=None, prob_range=PROB_RANGE,
            output_dir=PROJECT_ROOT / 'women/matchMarket/outputs', output_prefix='fullSampleBiasCheck',
            log_loss_label=f'full-sample (in-sample) log loss (architecture={DEATHNEW_ARCHITECTURE})')

        # RA_Sum scenario grid: per-ballsRemaining, spanning that ball's own 1st-99th percentile of real RA_Sum (+/- a
        # buffer), rather than one flat range for every ball count. True min/max is outlier-driven and would waste rows
        # on values that essentially never occur; percentile bounds scale sensibly with ball count (narrow near the end
        # of the innings, wider early on). <=12 balls remaining uses a fine (1) step; above that, coarse (2), even-aligned.
        RA_SUM_BUFFER = 2
        RA_SUM_STEP_COARSE = 2
        RA_SUM_STEP_FINE = 1
        RA_SUM_FINE_BALLS_THRESHOLD = 12
        raSumBounds = trainData[(trainData['inningBallsRemaining'] >= 1) & (trainData['inningBallsRemaining'] <= 30)].groupby('inningBallsRemaining')['RA_Sum'].quantile([0.01, 0.99]).unstack()
        raSumBounds.columns = ['p01', 'p99']
        raSumRows = []
        for ballsRemaining, bounds in raSumBounds.iterrows():
            lo = np.floor(bounds['p01'] - RA_SUM_BUFFER)
            hi = np.ceil(bounds['p99'] + RA_SUM_BUFFER)
            if ballsRemaining <= RA_SUM_FINE_BALLS_THRESHOLD:
                step = RA_SUM_STEP_FINE
            else:
                step = RA_SUM_STEP_COARSE
                lo = 2 * np.floor(lo / 2)
                hi = 2 * np.ceil(hi / 2)
            for v in np.arange(lo, hi + step / 2, step):
                raSumRows.append({'inningBallsRemaining': ballsRemaining, 'RA_Sum': v})
        raSumGrid = pd.DataFrame(raSumRows)
        print(f"RA_Sum grid: {len(raSumGrid)} (ballsRemaining, RA_Sum) scenario rows, {raSumGrid.groupby('inningBallsRemaining').size().min()}-{raSumGrid.groupby('inningBallsRemaining').size().max()} scenarios per ball count")

        chaseLookup2 = chaseLookup[(chaseLookup['inningBallsRemaining'] >= 1) & (chaseLookup['inningBallsRemaining'] <= 30)].copy()
        chaseLookup2 = chaseLookup2.merge(raSumGrid, on='inningBallsRemaining', how='left')
        # chaseLookup's hypothetical runsRequired/ratioRequired has no adjustment split to begin with, so the "std"
        # versions are just the existing values - RA_Sum is the new, separate scenario axis instead
        chaseLookup2['runsRequiredStd'] = chaseLookup2['runsRequired']
        chaseLookup2['ratioRequiredStd'] = chaseLookup2['ratioRequired']

        XPredDeathNew = scalerDeathNew.transform(chaseLookup2[deathFeaturesNew])
        chaseLookup2['m_chaseWin%_DeathNew'] = modelDeathNew.predict_proba(XPredDeathNew)[:, 1]

        chaseLookupNew = chaseLookup2
        lastCols = ['totalInningWickets', 'inningBallsRemaining', 'runsRequired', 'RA_Sum', 'm_chaseWin%_DeathNew']
        chaseLookupNew = chaseLookupNew[[c for c in chaseLookupNew.columns if c not in lastCols] + lastCols]
        print(f"chaseLookupNew rows: {len(chaseLookupNew)}")
        chaseLookupNew.to_csv(PROJECT_ROOT / 'women/matchMarket/outputs/1_chaseLookupNew_w.csv', index=False)

if RUN_ARCHITECTURE_DIAGNOSTIC:
    print("\n=== RUN_ARCHITECTURE_DIAGNOSTIC is on - stopping here, production model fit/export skipped ===")
    sys.exit(0)


# Main alone drives the output now - it's trained on Death's old inputs/architecture (64,32) across the whole
# innings, and neither Death nor LastOver feed the blend any more.
chaseLookup = chaseLookupMain.copy()
chaseLookup['m_chaseWin%'] = chaseLookup['m_chaseWin%Main']

trainDataMain['m_chaseWin%'] = trainDataMain['m_chaseWin%Main']
trainData = trainDataMain


# order correctly for illogical situations
cols = chaseLookup.loc[:, ['totalInningWickets', 'runsRequired', 'inningBallsRemaining', 'm_chaseWin%']]
colsWrong = cols.sort_values(by=['totalInningWickets', 'runsRequired', 'inningBallsRemaining'], axis=0).reset_index(drop=True)
colsRight = cols.sort_values(by=['totalInningWickets', 'runsRequired', 'm_chaseWin%'], axis=0).reset_index(drop=True)
colsWrong['m_chaseWin%'] = colsRight['m_chaseWin%']
colsWrong = colsWrong.sort_values(by=['inningBallsRemaining', 'runsRequired', 'totalInningWickets'], axis=0).reset_index(drop=True)
chaseLookup['m_chaseWin%'] = colsWrong['m_chaseWin%']

# add in an identifier/lookup column
chaseLookup['state_id'] = (
    chaseLookup['totalInningWickets']
    + (chaseLookup['inningBallsRemaining'] / 1000)
    + (chaseLookup['runsRequired'] / 1_000_000)
).round(6)


# some checks, the below doesn't affect the model
# bias check
bias = pd.pivot_table(trainData[trainData['ballWicketSample'] >= 100], values=['m_chaseWin%', 'chaseWin', 'sample'], aggfunc='sum', index=['totalInningWickets']).reset_index()
bias['bias'] = bias['m_chaseWin%'] / bias['chaseWin']
bias['win%'] = bias['chaseWin'] / bias['sample']
print(f"\n=== bias by totalInningWickets (ballWicketSample>=100) ===")
print(bias.to_string(index=False))

# ============================================================================================
# MODEL EVALUATION: log loss, calibration SD, bias by year, bias by year x predicted-probability
# bin. Standing checks so every future run reports them without re-deriving this analysis from
# scratch. See men/matchMarket/1_chaseModelLocal.py for the full reasoning - identical here.
# ============================================================================================
evalData = trainData[trainData['ballWicketSample'] >= 100].copy()
evalData['m_chaseWin%'] = np.clip(evalData['m_chaseWin%'], 1e-6, 1 - 1e-6)

overallLogLoss = log_loss(evalData['chaseWin'], evalData['m_chaseWin%'])
overallResidualSD = (evalData['m_chaseWin%'] - evalData['chaseWin']).std()
print(f"\n=== overall model evaluation (ballWicketSample>=100, n={len(evalData)}) ===")
print(f"log loss:                   {overallLogLoss:.5f}")
print(f"residual SD (pred - actual): {overallResidualSD:.5f}")

# bias by year
biasYear = pd.pivot_table(evalData, values=['m_chaseWin%', 'chaseWin', 'sample'], aggfunc='sum', index=['year']).reset_index()
biasYear['bias'] = biasYear['m_chaseWin%'] / biasYear['chaseWin']
biasYear['win%'] = biasYear['chaseWin'] / biasYear['sample']
print(f"\n=== bias by year (ballWicketSample>=100) ===")
print(biasYear.to_string(index=False))
# bias is undefined (inf) for any cell with zero actual chaseWin (e.g. a sparse early-year/low-
# probability cell that happened to have no real wins) - excluded from the SD, not from the table
finiteYearBias = biasYear['bias'][np.isfinite(biasYear['bias'])]
print(f"SD of bias across years: {finiteYearBias.std():.5f} ({len(biasYear) - len(finiteYearBias)} cell(s) excluded, chaseWin=0)")

# bias by year x predicted-probability bin (10pt buckets, standard calibration check) - does the
# model's own confidence level track its real accuracy consistently across years?
evalData['probBin'] = (np.floor(evalData['m_chaseWin%'] * 10) * 10).clip(0, 90).astype(int)
biasYearBin = pd.pivot_table(evalData, values=['m_chaseWin%', 'chaseWin', 'sample'], aggfunc='sum', index=['year', 'probBin']).reset_index()
biasYearBin['bias'] = biasYearBin['m_chaseWin%'] / biasYearBin['chaseWin']
biasYearBin['win%'] = biasYearBin['chaseWin'] / biasYearBin['sample']
print(f"\n=== bias by year x predicted-probability bin (10pt buckets, ballWicketSample>=100) ===")
print(biasYearBin.to_string(index=False))
finiteYearBinBias = biasYearBin['bias'][np.isfinite(biasYearBin['bias'])]
print(f"SD of bias across year x %bin cells: {finiteYearBinBias.std():.5f} ({len(biasYearBin) - len(finiteYearBinBias)} cell(s) excluded, chaseWin=0)")

# # compare
# chaseLookup = chaseLookup.merge(chaseLookupLive.loc[:, ['m_chaseWin%', 'totalInningWickets', 'runsRequired', 'inningBallsRemaining']],
#                                 how='left', on=['totalInningWickets', 'runsRequired', 'inningBallsRemaining'], suffixes=('', 'Live'))
# chaseLookup['m_diff'] = chaseLookup['m_chaseWin%'] - chaseLookup['m_chaseWin%Live']


# chase win % year
years = pd.pivot_table(trainData, index=['totalInningWickets', 'runsRequired', 'inningBallsRemaining'], values=['m_chaseWin%'], aggfunc='mean').reset_index()
chaseLookup = chaseLookup.merge(years, how='left', on=['totalInningWickets', 'runsRequired', 'inningBallsRemaining'], suffixes=('', 'Year'))

# insert lookup column for inserting into RAS
col_position = chaseLookup.columns.get_loc('m_chaseWin%')
chaseLookup.insert(col_position, 'lookup', (chaseLookup['totalInningWickets'] + (chaseLookup['inningBallsRemaining'] / 1000) + (chaseLookup['runsRequired'] / 1000000)).round(6))


# exports
chaseLookup.to_csv(PROJECT_ROOT / 'women/matchMarket/outputs/1_chaseLookup_w.csv', index=False)

chaseLookupComparison = chaseLookup.loc[:, ['inningBallNumber', 'inningBallsRemaining', 'totalInningWickets', 'runsRequired', 'chaseSample', 'chaseWin%', 'm_chaseWin%Main', 'm_chaseWin%', 'm_chaseWin%Year']]
