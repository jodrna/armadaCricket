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
# MODEL/ARCHITECTURE NOTES (men's local experimental file - production 1_chaseModel.py is untouched)
#
# Models in this file, and why each exists:
#
# - Main: the only model that feeds the final chaseLookup['m_chaseWin%'] output. Trained across the WHOLE
#   innings (no ball-count cutoff) on Death's old feature set - ['runsRequired', 'ratioRequired', 'daysGroup',
#   'inningBallsRemaining'] - rather than Main's original feature set (which explicitly included
#   'totalInningWickets'). This switch happened because Death's inputs, extended across the full innings,
#   tested as a better representation than Main's own original inputs. Dropping the explicit wickets feature
#   does NOT remove wicket-sensitivity from predictions: 'ratioRequired' = runsRequired / a
#   totalInningRunsToComeSimBiasSpline(Year) denominator that is itself computed per wickets-state, so wicket
#   information still leaks in through normalisation.
#
# - Death (old family) and LastOver: REMOVED. Death used to be blended in for balls <31 (fading out from
#   ball 30 to ball 12), and LastOver used to be averaged 50/50 with Main for the last 6 balls. Once Main was
#   retrained on Death's own inputs/architecture across the whole innings, Main effectively became what
#   "Death extended to the full innings" already was, so blending the original Death model back in added
#   nothing - it was redundant with Main, not complementary. LastOver was dropped in the same simplification
#   pass so Main alone drives the output from ball 120 down to ball 1, with no separate end-of-innings model.
#   trainDataDeath (the <36-ball training slice) is still built purely because the new-family Death model
#   below needs it as a data source - no old-family Death model is fit from it any more.
#
# - Death (new family, modelDeathNew, feeds chaseLookupNew.csv only): kept entirely separate from the
#   production-style chaseLookup output above. Tests giving the model runsRequiredStd + RA_Sum as two
#   separate inputs (rather than the single pre-combined runsRequiredAdj = runsRequiredStd - RA_Sum) - i.e.
#   can the network learn a better combination than a straight subtraction? This tested as a genuine win for
#   Death within its normal <=30-ball domain. It is NOT blended into production because RA_Sum itself is not
#   yet trustworthy enough (see the RA_Sum improvement plan - batting order, balls-faced, performance-vs-
#   expected, outlier checking, in that order, before this gets blended in) and because chaseLookup's generic
#   states have no real per-state RA_Sum (not tied to actual batters/bowlers/pitch) - hence the RA_Sum
#   scenario grid built below rather than predicting at one fake "neutral" value.
#
# Why (128, 64) for both Main and the new-family Death model (updated after the 2026-08-11 RA_Sum
# rebuild - order adjustment, balls-faced adjustment, dismissed-batter fix, chronological-ordering
# fix all changed the RA_Sum/runsRequiredAdj inputs these models train on, so the architecture was
# re-checked against the new data rather than assumed to still hold):
#   (64,32) had been the settled choice pre-rebuild, but a log-loss CV sweep on the new data showed
#   (128,64) beating it clearly for both models - and log loss alone isn't the full story here (see
#   sweep_architecture()/final_test_check() in architectureTesting.py): checked both in aggregate and
#   at specific states (120 balls/0 wickets for Main, 12 balls/5 wickets for DeathNew), confirmed
#   (128,64) is better calibrated too, not just a log-loss artifact. DeathNew's gain was the bigger of
#   the two - log loss 0.307 -> 0.186, a ~40% relative drop.
#   See RUN_ARCHITECTURE_DIAGNOSTIC below to rerun the comparison - it always tests current vs current
#   with every layer -50%/+50% (same depth), never a hand-picked list, using one shared protocol
#   (architectureTesting.py) also used by 1_chaseModelLocal_w.py, so men's and women's run the
#   identical test. This was decided on the much larger men's dataset (~487k training rows / ~7500
#   matches) - see 1_chaseModelLocal_w.py for the women's version, which has its own ~4x-smaller
#   dataset and was not re-checked as part of this pass.
# ============================================================================================


# quick CV harness to sanity-check hidden_layer_sizes choices before committing to them below - only
# sweeps Main and DeathNew, the two models that actually feed a real output; old-family Death was
# removed from production (see note above) so there's nothing live to sweep an architecture for there
RUN_ARCHITECTURE_DIAGNOSTIC = False
# set to False to skip fitting the new-family (runsRequiredStd+RA_Sum) Death model and building chaseLookupNew -
# only 1_chaseLookup.csv (the old family) gets produced when this is off
RUN_NEW_FAMILY_LOOKUP = True
DIAGNOSTIC_SEEDS = [42, 7, 123, 2024, 99]
# matches held out ONCE per run and never touched until final_test_check() at the very end - you never
# pick an architecture based on how it does on these matches, only on the sweep's CV validation score
TEST_FRACTION = 0.2
# states the model is already near-certain about aren't interesting - log loss, bias, and every other
# number reported below (train AND held-out/test) is scored ONLY on rows whose OWN predicted
# probability falls in this range, not just filtered for display. Set to None to score everything.
PROB_RANGE = (0.10, 0.90)
# early in the innings (ballsRemaining > this) PROB_RANGE is bypassed entirely - full 0-100% is
# scored there instead. Only Main's sweep ever sees ballsRemaining this high (DeathNew's domain is
# <36 balls), so this only actually changes Main's scoring.
FULL_RANGE_BALLS_THRESHOLD = 100
# THE two production architectures - used both as the "current" baseline when RUN_ARCHITECTURE_DIAGNOSTIC
# sweeps current/-50%/+50%, AND as what actually gets fit when it's off. After a diagnostic run picks a
# different winner, update these to match and the next (non-diagnostic) run deploys it - one place to change.
MAIN_ARCHITECTURE = (64, 32)
DEATHNEW_ARCHITECTURE = (64, 32)
# caps the TRAINING side of each CV fold for speed (see evaluate_architecture_cv). Checked 2026-08-24:
# Main's trainval domain is ~439k rows (17.6x this cap), DeathNew's is ~105k (4.2x) - both already
# heavily subsampled. Verified this is a deliberate speed tradeoff, not an oversight: an earlier
# attempt to remove the cap for men's Main produced zero output after 30+ minutes (killed). Leave as
# is - see women's file for a case where raising it was actually worth it.
MAX_SAMPLES = 25000


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
trainData = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/data/dataClean.csv', parse_dates=['date'])
masterLookup = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/outputs/5_masterLookup.csv')
chaseSituations = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/auxiliaries/chaseSituationBuilder.csv')
chaseLookupLive = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/outputs/1_chaseLookupLive.csv')


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
masterLookup = masterLookup[(masterLookup['ord'] == 1) & (masterLookup['daysGroup'] == 11)]
# merge in runs to come
trainData = trainData.merge(masterLookup.loc[:, ['totalInningRunsToComeSimBiasSplineYear', 'totalInningWickets', 'inningBallNumber', 'totalInningValidBallsFacedToCome', 'bowledOut', 'sample']].rename(columns={'sample': 'ballWicketSample'}), how='left', on=['totalInningWickets', 'inningBallNumber'])
# ballWicketSample = sample size behind the (wickets, ball) state, kept on every row so it can be used to mask out unreliable states when scoring/checking the model (not when training it)
# create a ratio of runs to come to be used as a predictor, drop any nans
trainData['ratioRequired'] = trainData['runsRequired'] / trainData['totalInningRunsToComeSimBiasSplineYear']
trainData['ratioRequiredStd'] = trainData['runsRequiredStd'] / trainData['totalInningRunsToComeSimBiasSplineYear']
trainData = trainData.dropna(axis=0, subset=['ratioRequired'])


# test = trainData.copy()
# test['wickets_group'] = np.round(test['totalInningWickets'] / 3, 0) * 3
# test['year_group'] = np.round(test['year'] / 3, 0) * 3
# test['runsRequired_round'] = np.round(test['runsRequiredAdj'], 0)
# test = pd.pivot_table(test, values=['sample', 'chaseWin'],
#                             index=['wickets_group', 'inningBallNumber', 'runsRequired_round', 'year_group'],
#                             aggfunc={'sample': 'sum', 'chaseWin': 'sum'}).reset_index()
# test['chase_win%'] = test['chaseWin'] / test['sample']
# test = test[(test['wickets_group'] < 7) & (test['inningBallNumber'] == 115)] #



# create an empty dataframe
chaseLookup = pd.pivot_table(trainData, values=['sample', 'chaseWin', 'totalInningRunsToCome', 'totalInningWicketsToCome', 'runsRequiredStd'],
                            index=['totalInningWickets', 'inningBallNumber', 'runsRequired'],
                            aggfunc={'sample': 'sum', 'chaseWin': 'sum', 'totalInningRunsToCome': 'mean', 'totalInningWicketsToCome': 'mean', 'runsRequiredStd': 'mean'}).reset_index()
chaseLookup['chaseWin%'] = chaseLookup['chaseWin'] / chaseLookup['sample']
chaseLookup = chaseSituations.merge(chaseLookup, how='left', on=['totalInningWickets', 'inningBallNumber', 'runsRequired'])
chaseLookup = chaseLookup.rename(columns={'sample': 'chaseSample'})
chaseLookup = chaseLookup.merge(masterLookup.loc[:, ['totalInningWickets', 'inningBallNumber', 'sample', 'totalInningRunsToComeSimBiasSpline', 'totalInningValidBallsFacedToCome', 'bowledOut']], how='left', on=['totalInningWickets', 'inningBallNumber'])
chaseLookup = chaseLookup.rename(columns={'sample': 'ballWicketSample'})

chaseLookup['ratioRequired'] = chaseLookup['runsRequired'] / chaseLookup['totalInningRunsToComeSimBiasSpline']
chaseLookup['daysGroup'] = 11.8
chaseLookup = chaseLookup.dropna(axis=0, subset=['totalInningRunsToComeSimBiasSpline']).reset_index(drop=True)
chaseLookup['in'] = 1

# remove chases which are effectively lost
trainData = trainData.merge(chaseLookup.loc[:, ['in', 'totalInningWickets', 'runsRequired', 'inningBallNumber']], how='left', on=['totalInningWickets', 'runsRequired', 'inningBallNumber'])
trainData = trainData[trainData['in'] == 1]

# start of innings model
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
        full_range_balls_threshold=FULL_RANGE_BALLS_THRESHOLD,
        wickets=trainvalMain['totalInningWickets'], ballsRemaining=trainvalMain['inningBallsRemaining'],
        output_dir=PROJECT_ROOT / 'men/matchMarket/outputs')
    winnerMain = summaryMain.iloc[0]['architecture']
    testMainMasked = testMain[testMain['ballWicketSample'] >= 100]
    final_test_check(
        'Main', trainvalMain[['runsRequired', 'ratioRequired', 'daysGroup', 'inningBallsRemaining']], trainvalMain['chaseWin'],
        testMainMasked[['runsRequired', 'ratioRequired', 'daysGroup', 'inningBallsRemaining']], testMainMasked['chaseWin'],
        testMainMasked['totalInningWickets'], testMainMasked['inningBallsRemaining'], chosen_architecture=winnerMain,
        prob_range=PROB_RANGE, full_range_balls_threshold=FULL_RANGE_BALLS_THRESHOLD,
        output_dir=PROJECT_ROOT / 'men/matchMarket/outputs')
if not RUN_ARCHITECTURE_DIAGNOSTIC:
    scaler = StandardScaler()
    scaler.fit(X_std)
    X_std = scaler.transform(X_std)

    # build the model
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
        full_range_balls_threshold=FULL_RANGE_BALLS_THRESHOLD, output_dir=PROJECT_ROOT / 'men/matchMarket/outputs',
        output_prefix='fullSampleBiasCheck', log_loss_label=f'full-sample (in-sample) log loss (architecture={MAIN_ARCHITECTURE})')

    # now predict the chase situations outside of training
    chaseLookupMain = chaseLookup.copy()
    X = chaseLookupMain[['runsRequired', 'ratioRequired', 'daysGroup', 'inningBallsRemaining']]
    X = scaler.transform(X)
    chaseLookupMain['m_chaseWin%Main'] = model.predict_proba(X)[:, 1]

    # ========================================================================================
    # chaseLookupAllYears: the SAME situational grid as chaseLookup (runsRequired/ratioRequired/
    # inningBallsRemaining/totalInningWickets - unchanged, still built off the fixed-daysGroup=11
    # spline baseline chaseLookup itself uses), but predicted at the MIDPOINT of every real year
    # (2015 through the current year, 2026) instead of chaseLookup's single fixed "today"
    # daysGroup=11.8. Reuses the same fitted Main model/scaler - only the daysGroup feature fed
    # into the model varies; the situation description itself does not.
    #
    # daysGroup=N is the START of year 2015+N (confirmed against real dates: daysGroup=7.0 lands
    # on 2021-12-30, i.e. the Jan-1-2022 boundary), NOT the middle - so the midpoint of year
    # 2015+N is daysGroup=N+0.5 (confirmed: daysGroup=7.5 lands on 2022-07-01).
    # ========================================================================================
    allYearsBase = chaseLookupMain[['totalInningWickets', 'inningBallNumber', 'inningBallsRemaining',
                                     'runsRequired', 'ratioRequired']].copy()
    allYearsDaysGroups = [dg + 0.5 for dg in range(0, 12)]  # midpoint of 2015 through 2026
    chaseLookupAllYears = pd.concat(
        [allYearsBase.assign(daysGroup=dg) for dg in allYearsDaysGroups], ignore_index=True)
    X_allYears = scaler.transform(
        chaseLookupAllYears[['runsRequired', 'ratioRequired', 'daysGroup', 'inningBallsRemaining']])
    chaseLookupAllYears['m_chaseWin%Main'] = model.predict_proba(X_allYears)[:, 1]
    chaseLookupAllYears['year'] = (chaseLookupAllYears['daysGroup'] - 0.5 + 2015).astype(int)
    chaseLookupAllYears.to_csv(PROJECT_ROOT / 'men/matchMarket/outputs/1_chaseLookupAllYears.csv', index=False)
    print(f"\nchaseLookupAllYears written to outputs/1_chaseLookupAllYears.csv "
          f"({len(chaseLookupAllYears)} rows = {len(allYearsBase)} situations x {len(allYearsDaysGroups)} years)")

# DEATH model
trainDataDeath = trainData.copy()
trainDataDeath = trainDataDeath[(trainDataDeath['inningBallsRemaining'] < 36)]

# prepare the data
y = trainDataDeath['chaseWin']
X_stdDeath = trainDataDeath[['runsRequired', 'ratioRequired', 'daysGroup', 'inningBallsRemaining']]
# , 'totalInningWickets'
if RUN_ARCHITECTURE_DIAGNOSTIC:
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
        y, (trainDataDeath['ballWicketSample'] >= 100) & (trainDataDeath['inningBallsRemaining'] <= 30), trainDataDeath['matchID']
    )

    # ---- how far back does the runsRequiredStd + RA_Sum advantage hold, where it actually matters for the blend?
    # cumulative "balls < cutoff" windows mix rows where Death is 100% of the output with rows where its blend
    # weight has already faded to near zero (mainWeight = (ballsRemaining-12)/18 reaches 1.0 AT ball 30, not just
    # above it), so an aggregate over such a window can't tell you whether an advantage lives where it counts.
    # Testing disjoint bands defined by Death's actual blend weight instead, so each band means something in
    # terms of production impact. Uses (64,32): (8,4) is known unstable and would swamp the signal with noise.
    mainWeightAll = np.select([trainData['inningBallsRemaining'] > 30, trainData['inningBallsRemaining'] < 12],
                               [1.0, 0.0], default=(trainData['inningBallsRemaining'] - 12) / 18)
    deathWeightAll = 1 - mainWeightAll
    WEIGHT_BANDS = [
        ('Death=100% (balls<12)', (deathWeightAll >= 0.999)),
        ('Death 75-99% (~balls 12-16)', (deathWeightAll >= 0.75) & (deathWeightAll < 0.999)),
        ('Death 50-75% (~balls 16-21)', (deathWeightAll >= 0.50) & (deathWeightAll < 0.75)),
        ('Death 25-50% (~balls 21-25)', (deathWeightAll >= 0.25) & (deathWeightAll < 0.50)),
        ('Death 1-25% (~balls 25-30)', (deathWeightAll > 0) & (deathWeightAll < 0.25)),
    ]
    bandRows = []
    for bandLabel, bandFilter in WEIGHT_BANDS:
        bandData = trainData[bandFilter]
        yBand = bandData['chaseWin']
        maskBand = bandData['ballWicketSample'] >= 100
        groupsBand = bandData['matchID']
        variantsBand = {
            'current (runsRequiredAdj)': bandData[['runsRequired', 'ratioRequired', 'daysGroup', 'inningBallsRemaining']],
            'new (runsRequiredStd + RA_Sum)': bandData[['runsRequiredStd', 'RA_Sum', 'ratioRequiredStd', 'daysGroup', 'inningBallsRemaining']],
        }
        for label, X in variantsBand.items():
            seedRuns = evaluate_architecture_cv(X, yBand, maskBand, groupsBand, (64, 32), seeds=DIAGNOSTIC_SEEDS,
                                                 prob_range=PROB_RANGE)
            heldoutLosses = [r['heldoutLoss'] for r in seedRuns]
            bandRows.append({'death_weight_band': bandLabel, 'features': label, 'n': len(bandData),
                              'heldout_mean': np.mean(heldoutLosses), 'heldout_std': np.std(heldoutLosses)})
    bandResults = pd.DataFrame(bandRows)
    bandPivot = bandResults.pivot(index='death_weight_band', columns='features', values='heldout_mean')
    bandPivot = bandPivot.reindex([b[0] for b in WEIGHT_BANDS])
    bandPivot['new_advantage'] = bandPivot['current (runsRequiredAdj)'] - bandPivot['new (runsRequiredStd + RA_Sum)']
    print("\n=== runsRequiredStd+RA_Sum vs runsRequiredAdj, banded by Death's actual blend weight ===")
    print(bandResults.to_string(index=False))
    print("\n--- advantage by band (positive = new representation wins) ---")
    print(bandPivot.to_string())

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
            output_dir=PROJECT_ROOT / 'men/matchMarket/outputs')
        winnerDeathNew = summaryDeathNew.iloc[0]['architecture']
        testDeathMasked = testDeath[testDeath['ballWicketSample'] >= 100]
        final_test_check(
            'DeathNew', trainvalDeath[deathFeaturesNew], trainvalDeath['chaseWin'],
            testDeathMasked[deathFeaturesNew], testDeathMasked['chaseWin'],
            testDeathMasked['totalInningWickets'], testDeathMasked['inningBallsRemaining'],
            chosen_architecture=winnerDeathNew, start_balls_remaining=12, start_wickets=None, prob_range=PROB_RANGE,
            output_dir=PROJECT_ROOT / 'men/matchMarket/outputs')

    if not RUN_ARCHITECTURE_DIAGNOSTIC:
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
            output_dir=PROJECT_ROOT / 'men/matchMarket/outputs', output_prefix='fullSampleBiasCheck',
            log_loss_label=f'full-sample (in-sample) log loss (architecture={DEATHNEW_ARCHITECTURE})')

        # LastOver's new-family model is dropped for now - the rigorous 7-seed test found no reliable benefit there
        # (t-stat -0.83, new won only 1/7 seeds), unlike Death's clear win, so it's not worth building out yet

        # RA_Sum scenario grid: per-ballsRemaining, spanning that ball's own 1st-99th percentile of real RA_Sum (+/- a
        # buffer), rather than one flat range for every ball count. True min/max is outlier-driven (e.g. ball=1's true
        # min is -18.2 but only 1% of real values sit below -3.2) and would waste rows on values that essentially never
        # occur; percentile bounds scale sensibly with ball count (narrow near the end of the innings, wider early on).
        RA_SUM_BUFFER = 2
        RA_SUM_STEP_COARSE = 2
        RA_SUM_STEP_FINE = 1
        RA_SUM_FINE_BALLS_THRESHOLD = 12  # <=12 balls remaining uses the fine (1) step; above that, coarse (2), even-aligned
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
        chaseLookupNew.to_csv(PROJECT_ROOT / 'men/matchMarket/outputs/1_chaseLookupNew.csv', index=False)

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
# scratch. All computed on trainData (real historical rows with actual chaseWin outcomes), masked
# to ballWicketSample>=100 (trustworthy states only) - same masking convention as the by-wickets
# bias check above. 'bias' throughout = ratio of predicted-sum / actual-sum (>1 = model
# over-predicts win%, <1 = under-predicts), matching the by-wickets bias check's own definition.
# ============================================================================================
evalData = trainData[trainData['ballWicketSample'] >= 100].copy()
evalData['m_chaseWin%'] = np.clip(evalData['m_chaseWin%'], 1e-6, 1 - 1e-6)

overallLogLoss = log_loss(evalData['chaseWin'], evalData['m_chaseWin%'])
overallResidualSD = (evalData['m_chaseWin%'] - evalData['chaseWin']).std()
print(f"\n=== overall model evaluation (ballWicketSample>=100, n={len(evalData)}) ===")
print(f"log loss:                   {overallLogLoss:.5f}")
print(f"residual SD (pred - actual): {overallResidualSD:.5f}")

# predicted vs real chaseWin%, EQUAL-SAMPLE-SIZE bins of predicted probability - complements the
# fixed-width 10pt bins further down, which can carry very uneven sample counts per bin (chase
# win probability is bounded and heavily clustered near 0%/100%, so a fixed-width bin near the
# middle can have far fewer rows backing it than one at the extremes)
def bias_by_equal_sample_bin(df, n_bins=10):
    d = df.copy()
    d['probQuantileBin'] = pd.qcut(d['m_chaseWin%'], n_bins, labels=False, duplicates='drop')
    out = d.groupby('probQuantileBin').agg(
        n=('chaseWin', 'size'),
        predRangeLo=('m_chaseWin%', 'min'),
        predRangeHi=('m_chaseWin%', 'max'),
        pred_pct=('m_chaseWin%', 'mean'),
        real_pct=('chaseWin', 'mean'),
    ).reset_index()
    for col in ['predRangeLo', 'predRangeHi', 'pred_pct', 'real_pct']:
        out[col] *= 100
    out['diff'] = out['pred_pct'] - out['real_pct']
    return out


N_QUANTILE_BINS = 10
biasQuantile = bias_by_equal_sample_bin(evalData, N_QUANTILE_BINS)
print(f"\n=== overall predicted vs real chaseWin%, EQUAL-SAMPLE-SIZE bins "
      f"({N_QUANTILE_BINS} bins, ~{len(evalData) // N_QUANTILE_BINS} rows each, ballWicketSample>=100) ===")
print(biasQuantile.to_string(index=False))

# same thing, restricted to the start-of-chase state only (120 ballsRemaining, 0 wickets - no
# ballWicketSample filter needed here, this state has plenty of its own sample)
startStateData = trainData[(trainData['inningBallsRemaining'] == 120) & (trainData['totalInningWickets'] == 0)].copy()
startStateData['m_chaseWin%'] = np.clip(startStateData['m_chaseWin%'], 1e-6, 1 - 1e-6)
biasQuantileStart = bias_by_equal_sample_bin(startStateData, N_QUANTILE_BINS)
print(f"\n=== 120 ballsRemaining / 0 wickets ONLY: predicted vs real chaseWin%, EQUAL-SAMPLE-SIZE bins "
      f"({N_QUANTILE_BINS} bins, ~{len(startStateData) // N_QUANTILE_BINS} rows each, n={len(startStateData)}) ===")
print(biasQuantileStart.to_string(index=False))

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

# compare
chaseLookup = chaseLookup.merge(chaseLookupLive.loc[:, ['m_chaseWin%', 'totalInningWickets', 'runsRequired', 'inningBallsRemaining']],
                                how='left', on=['totalInningWickets', 'runsRequired', 'inningBallsRemaining'], suffixes=('', 'Live'))
chaseLookup['m_diff'] = chaseLookup['m_chaseWin%'] - chaseLookup['m_chaseWin%Live']


# chase win % year
years = pd.pivot_table(trainData, index=['totalInningWickets', 'runsRequired', 'inningBallsRemaining'], values=['m_chaseWin%'], aggfunc='mean').reset_index()
chaseLookup = chaseLookup.merge(years, how='left', on=['totalInningWickets', 'runsRequired', 'inningBallsRemaining'], suffixes=('', 'Year'))

# insert lookup column for inserting into RAS
col_position = chaseLookup.columns.get_loc('m_chaseWin%')  # gets index of 'B'
chaseLookup.insert(col_position, 'lookup', (chaseLookup['totalInningWickets'] + (chaseLookup['inningBallsRemaining'] / 1000) + (chaseLookup['runsRequired'] / 1000000)).round(6))

# # graph of predictions
# fig, axes = plt.subplots(10, 4, figsize=(20, 40))           # create a figure of dimension 10 (Wickets) by 5 (number of graphs for each wicket)
# for x in np.arange(0, 10, 1):                               # loop 0-10 for wickets
#     graph_data = chaseLookup.copy()
#     graph_data = graph_data[graph_data['totalInningWickets'] == x]       # filter the dataframe for the wicket in question
#     # graph_data['chase_adj%'] = graph_data['blendr_win%'] - graph_data['X_win%']
#     # create tables of the numbers to be plotted
#     actual = pd.pivot_table(graph_data, index='runsRequired', columns='inningBallsRemaining', values='chaseWin%', aggfunc='mean')
#     old = pd.pivot_table(graph_data, index='runsRequired', columns='inningBallsRemaining', values='m_chaseWin%', aggfunc='mean')
#     new = pd.pivot_table(graph_data, index='runsRequired', columns='inningBallsRemaining', values='m_chaseWin%', aggfunc='mean')
#     diff = pd.pivot_table(graph_data, index='runsRequired', columns='inningBallsRemaining', values='chaseSample', aggfunc='mean')
#     # plot in a heatmap
#     sns.heatmap(ax=axes[x, 0], data=actual, cmap=plt.cm.get_cmap('PiYG', 1000), vmin=0, vmax=1, center=0.5, xticklabels=10, yticklabels=10)
#     sns.heatmap(ax=axes[x, 1], data=old, cmap=plt.cm.get_cmap('PiYG', 1000), vmin=0, vmax=1, center=0.5, xticklabels=10, yticklabels=10)
#     sns.heatmap(ax=axes[x, 2], data=new, cmap=plt.cm.get_cmap('PiYG', 1000), vmin=0, vmax=1, center=0.5, xticklabels=10, yticklabels=10)
#     sns.heatmap(ax=axes[x, 3], data=diff, cmap=plt.cm.get_cmap('PiYG', 1000), vmin=0, vmax=500, center=62, xticklabels=10, yticklabels=10)
#
#     # set titles for each graph
#     title1 = f"actual_win% - {x} wickets lost"
#     axes[x, 0].set_title(title1)
#     title2 = f"old - {x} wickets lost"
#     axes[x, 1].set_title(title2)
#     title3 = f"new {x} wickets lost"
#     axes[x, 2].set_title(title3)
#     title4 = f"diff - {x} wickets lost"
#     axes[x, 3].set_title(title4)
#     # title5 = f"blendr_win%_ - {x} wickets lost"
#     # axes[x, 4].set_title(title4)
# plt.tight_layout()
# plt.show()






# # over/under performance heatmap
# # x = daysGroup, split into 0.04-year steps
# # y = balls remaining
# # colour = actual chase win% - expected chase win%
#
# heat_data = trainData.copy()
# heat_data = heat_data.drop(columns=['m_chaseWin%'])
# heat_data = heat_data.merge(chaseLookup.loc[:, ['inningBallsRemaining', 'totalInningWickets', 'runsRequired', 'm_chaseWin%']], how='left', on=['inningBallsRemaining', 'totalInningWickets', 'runsRequired'])
# heat_data = heat_data.dropna(
#     subset=[
#         'daysGroup',
#         'inningBallsRemaining',
#         'chaseWin',
#         'm_chaseWin%'
#     ]
# )
#
# year_centres = np.arange(
#     np.floor(heat_data['daysGroup'].min()),
#     np.ceil(heat_data['daysGroup'].max()) + 0.04,
#     0.04
# )
#
# ball_centres = np.arange(1, 121, 1)
#
# rows = []
#
# for yc in year_centres:
#     for bc in ball_centres:
#         mask = (
#             (heat_data['daysGroup'].sub(yc).abs() <= 0.5) &
#             (heat_data['inningBallsRemaining'].sub(bc).abs() <= 5)
#         )
#
#         cell = heat_data.loc[mask]
#
#         sample = len(cell)
#
#         if sample < 100:
#             actual = np.nan
#             expected = np.nan
#             over_under = np.nan
#         else:
#             actual = cell['chaseWin'].mean()
#             expected = cell['m_chaseWin%'].mean()
#             over_under = actual - expected
#
#         rows.append({
#             'daysGroupCentre': yc,
#             'yearLabel': 2015 + yc,
#             'inningBallsRemaining': bc,
#             'sample': sample,
#             'actual_chaseWin%': actual,
#             'expected_chaseWin%': expected,
#             'over_under': over_under
#         })
#
# heatmap_df = pd.DataFrame(rows)
#
# heatmap_pivot = heatmap_df.pivot(
#     index='inningBallsRemaining',
#     columns='daysGroupCentre',
#     values='over_under'
# )
#
# plt.figure(figsize=(16, 10))
#
# sns.heatmap(
#     heatmap_pivot,
#     cmap='RdYlGn',
#     center=0,
#     vmin=-0.08,
#     vmax=0.08,
#     linewidths=0,
#     cbar_kws={'label': 'Actual chase win% - expected chase win%'}
# )
#
# plt.title('Women chasing over/under performance by year and balls remaining')
# plt.xlabel('Year')
# plt.ylabel('Balls remaining')
#
# xtick_positions = np.arange(0, len(year_centres), 25)
# xtick_labels = [
#     str(int(2015 + year_centres[pos]))
#     for pos in xtick_positions
# ]
#
# plt.xticks(
#     xtick_positions,
#     xtick_labels,
#     rotation=0
# )
#
# plt.gca().invert_yaxis()
#
# plt.tight_layout()
# plt.show()




# exports
chaseLookup.to_csv(PROJECT_ROOT / 'men/matchMarket/outputs/1_chaseLookup.csv', index=False)

chaseLookupComparison = chaseLookup.loc[:, ['inningBallNumber', 'inningBallsRemaining', 'totalInningWickets', 'runsRequired', 'chaseSample', 'chaseWin%', 'm_chaseWin%Main', 'm_chaseWin%', 'm_chaseWin%Year']]
