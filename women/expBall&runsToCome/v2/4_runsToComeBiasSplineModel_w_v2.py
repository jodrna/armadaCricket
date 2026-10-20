import pandas as pd
import numpy as np
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from scipy.interpolate import UnivariateSpline
from paths import PROJECT_ROOT

# women's port of men's v2/4_runsToComeBiasSplineModel_v2.py (same method, women's files) - v2 of 4_runsToComeBiasSplineModel_w.py - calibrates the sim's runs to come to real first innings, by situation.
# changes from v1:
#   - reads the situation summary from 3_runsToComeSimClassOrd_w_v2.py (discrete sim, step 1 v2 inputs) instead of the raw sim balls
#   - the sim vs real bias is fit as one smooth surface over ball x wickets (log ratio, tensor product splines, ridge),
#     weighted by how noisy each situation's ratio is, instead of an unweighted depth 3 random forest per wicket
#   - the correction is applied at every ball (v1 skipped balls 110+ where the sim bias is largest)
#   - situations with thin real data keep their sim value, the correction there is borrowed from neighbouring
#     situations and shrinks toward no correction (v1 dropped them, leaving a third of situations empty)
#   - the smoothing spline is weighted by the sim's standard error, so it only smooths out sim noise
#   - the "more wickets can't mean more runs" fix only touches the runs to come value (v1 moved whole rows)
#   - no hand overrides at ball 120 / balls 1-5, and the surface's flexibility is picked by cross validation on held out
#     matches, judged on state level calibration
#   - sim STD, min and max get scaled by the same correction as the mean, skew and kurtosis are scale free so unchanged
#   - a residual stage after the surface: what the surface still misses (real minus prediction per situation) is fitted
#     locally (local linear) over nearby balls at the same wickets and added on in full, tapered where real data is thin


# import necessary data
trainData = pd.read_csv(PROJECT_ROOT / 'women/expBall&runsToCome/data/dataClean_w.csv',
                        usecols=['matchID', 'inningNumber', 'inningBallNumber', 'totalInningWickets', 'totalInningRunsToCome', 'totalInningWicketsToCome', 'competition', 'date'])
trainData = trainData[trainData['inningNumber'] == 1].reset_index(drop=True)
# v2: The Hundred is left out - its innings end at ball 100, so in this 120 ball frame its runs to come, per ball rates by
# ball number and crease states don't match a T20 (it pulled real runs to come down ~2-3 runs before ball 100 and made a
# step at ball 100 that step 4 couldn't remove). Note the joins are on inningBallNumber, which 3_hundredAdjusts_w.py doesn't
# rescale (only inningBallsRemaining), so the Hundred can't simply be mapped onto this 120 ball frame either
trainData = trainData[trainData['competition'] != "The Hundred (Women's Comp)"].reset_index(drop=True)
# surge-era WBBL (from WBBL|08, 2022 - user, 15 Oct 2026) out: it must not shape ball-by-ball values. It stays in dataClean_w for 5_runsToComeYearModel only
trainData = trainData[~((trainData['competition'] == "Women's Big Bash League") & (pd.to_datetime(trainData['date']) > pd.Timestamp('2022-06-01')))].reset_index(drop=True)
simSituationRunsToCome = pd.read_csv(PROJECT_ROOT / 'women/expBall&runsToCome/outputs/ballSimsClassOrdSummary_w_v2.csv')
masterLookup = pd.read_csv(PROJECT_ROOT / 'women/expBall&runsToCome/outputs/2_masterLookup_w_v2.csv')

stateKeys = ['inningBallNumber', 'totalInningWickets']
# same naming as v1 after its real vs sim merge
simSituationRunsToCome = simSituationRunsToCome.rename(columns={'totalInningRunsToCome': 'totalInningRunsToComeSim', 'sample': 'sampleSim'})

# settings
MIN_REAL_SAMPLE = 5                     # situations need at least this many real balls to be used when fitting the bias
BALL_KNOTS_GRID = [8, 16, 24]           # knots across inningBallNumber for the bias surface
WICKET_KNOTS_GRID = [6, 10]             # knots across totalInningWickets for the bias surface
ALPHA_GRID = [0.01, 0.1, 1, 10]         # ridge strength - higher shrinks the correction toward none where data is thin
CV_FOLDS = 5
# residual stage - picked 2026-10-09 on held out matches: the 5 ball bias grid (100+ real ball situations, bootstrap by match
# standard errors) and held out bias pooled by how many wickets worse than average the situation is. The surface leaves a
# collapse shape (men 6 down mid innings +3 to +4 runs). Shrinking the correction toward none (tried 25-2500 real balls) or
# borrowing from the wickets either side left part of that shape in, so the correction is applied in full along each wicket
# row and only tapered where there's little real data nearby (500 rather than more so the first situations of each wickets
# row get their full correction too). Costs some state level calibration (held out RMS ~1.80 -> 1.88 men) for removing
# the shape; the curves across balls stay smooth. Local linear rather than a plain kernel average: the plain average left the
# first situations of the 1-3 wickets rows (very early wickets) +1 to +1.6 runs high
RESIDUAL_BALL_BANDWIDTH = 4             # gaussian kernel width across inningBallNumber, within the same wickets
RESIDUAL_FULL_SAMPLE = 500              # kernel weighted real balls nearby needed for the full correction, tapered linearly below


def real_situation_runs_to_come(data):
    return data.groupby(stateKeys).agg(
        totalInningRunsToCome=('totalInningRunsToCome', 'mean'),
        totalInningWicketsToCome=('totalInningWicketsToCome', 'mean'),
        sample=('totalInningRunsToCome', 'size')
    ).reset_index()


def bias_basis(states, ballKnots, wicketKnots):
    # tensor product of cubic splines over ball and quadratic splines over wickets, knots fixed on the full grid
    ballSpline = SplineTransformer(n_knots=ballKnots, degree=3, extrapolation='linear').fit(np.arange(1, 121).reshape(-1, 1))
    wicketSpline = SplineTransformer(n_knots=wicketKnots, degree=2, extrapolation='linear').fit(np.arange(0, 10).reshape(-1, 1))
    B = ballSpline.transform(states[['inningBallNumber']].to_numpy())
    W = wicketSpline.transform(states[['totalInningWickets']].to_numpy())
    return np.einsum('ij,ik->ijk', B, W).reshape(len(states), -1)


def fit_log_bias(situations, ballKnots, wicketKnots, alpha):
    # log(real / sim) per situation, weighted by 1 / its variance (delta method, sim STD standing in for real STD)
    fit = situations[(situations['sample'] >= MIN_REAL_SAMPLE) & (situations['totalInningRunsToCome'] > 0) &
                     (situations['totalInningRunsToComeSimCount'] >= 2)]
    y = np.log(fit['totalInningRunsToCome'] / fit['totalInningRunsToComeSim'])
    variance = fit['totalInningRunsToComeSimSTD'] ** 2 * (1 / (fit['sample'] * fit['totalInningRunsToCome'] ** 2) +
                                                          1 / (fit['totalInningRunsToComeSimCount'] * fit['totalInningRunsToComeSim'] ** 2))
    weights = 1 / variance
    weights = weights / weights.mean()
    model = Ridge(alpha=alpha).fit(bias_basis(fit, ballKnots, wicketKnots), y, sample_weight=weights)
    return model.predict(bias_basis(situations, ballKnots, wicketKnots))


def build_runs_to_come(situations, logBias):
    situations = situations.copy()
    situations['logBias'] = logBias
    situations['totalInningRunsToComeSimBias'] = situations['totalInningRunsToComeSim'] * np.exp(logBias)

    # smooth each wicket's curve over balls, weighted by the sim's standard error so it only irons out sim noise
    # (s = number of points is the standard choice when weights are 1 / standard error)
    standardError = situations['totalInningRunsToComeSimSTD'] * np.exp(logBias) / np.sqrt(situations['totalInningRunsToComeSimCount'])
    situations['splineWeight'] = np.where(standardError > 0, 1 / standardError, 1e-3)
    situations['totalInningRunsToComeSimBiasSpline'] = situations['totalInningRunsToComeSimBias']
    for w in range(10):
        idx = situations.index[situations['totalInningWickets'] == w]
        s = situations.loc[idx].sort_values('inningBallNumber')
        if len(s) > 3:
            spline = UnivariateSpline(s['inningBallNumber'], s['totalInningRunsToComeSimBias'], w=s['splineWeight'], s=len(s))
            situations.loc[s.index, 'totalInningRunsToComeSimBiasSpline'] = spline(s['inningBallNumber'])

    # within each ball, more wickets lost can't mean more runs to come - weighted isotonic fit on the value only
    for b, s in situations.groupby('inningBallNumber'):
        s = s.sort_values('totalInningWickets')
        iso = IsotonicRegression(increasing=False).fit(s['totalInningWickets'], s['totalInningRunsToComeSimBiasSpline'],
                                                       sample_weight=s['totalInningRunsToComeSimCount'])
        situations.loc[s.index, 'totalInningRunsToComeSimBiasSpline'] = iso.predict(s['totalInningWickets'])

    return situations


def residual_correct(situations):
    # real minus prediction per situation with real data, fitted locally along each wickets row (kernel weighted by real
    # sample, local linear so the first and last situations of a row aren't diluted by neighbours on one side only), applied
    # in full where there's enough real data nearby and tapered toward no correction where there isn't
    fit = situations.dropna(subset=['totalInningRunsToCome', 'totalInningRunsToComeSimBiasSpline'])
    fit = fit[fit['sample'] >= 1]
    residual = (fit['totalInningRunsToCome'] - fit['totalInningRunsToComeSimBiasSpline']).to_numpy()
    ballGap = fit['inningBallNumber'].to_numpy()[None, :] - situations['inningBallNumber'].to_numpy()[:, None]
    sameWickets = situations['totalInningWickets'].to_numpy()[:, None] == fit['totalInningWickets'].to_numpy()[None, :]
    kernel = np.exp(-0.5 * (ballGap / RESIDUAL_BALL_BANDWIDTH) ** 2) * sameWickets * fit['sample'].to_numpy()[None, :]
    # weighted least squares of residual on ball gap at each situation, the intercept is the correction there
    s0, s1, s2 = kernel.sum(axis=1), (kernel * ballGap).sum(axis=1), (kernel * ballGap ** 2).sum(axis=1)
    t0, t1 = kernel @ residual, (kernel * ballGap) @ residual
    det = s0 * s2 - s1 ** 2
    localConstant = np.divide(t0, s0, out=np.zeros(len(situations)), where=s0 > 0)
    localLinear = np.divide(s2 * t0 - s1 * t1, det, out=localConstant.copy(), where=det > 1e-9 * np.maximum(s0 * s2, 1e-12))
    situations = situations.copy()
    situations['residualCorrection'] = localLinear * np.minimum(1, s0 / RESIDUAL_FULL_SAMPLE)
    situations['totalInningRunsToComeSimBiasSpline'] = situations['totalInningRunsToComeSimBiasSpline'] + situations['residualCorrection']

    # the correction can break the ordering, so again: within each ball more wickets lost can't mean more runs to come
    for b, s in situations.groupby('inningBallNumber'):
        s = s.sort_values('totalInningWickets')
        iso = IsotonicRegression(increasing=False).fit(s['totalInningWickets'], s['totalInningRunsToComeSimBiasSpline'],
                                                       sample_weight=s['totalInningRunsToComeSimCount'])
        situations.loc[s.index, 'totalInningRunsToComeSimBiasSpline'] = iso.predict(s['totalInningWickets'])
    return situations


def situations_from(realData):
    return simSituationRunsToCome.merge(real_situation_runs_to_come(realData), how='left', on=stateKeys)


# pick the knots and ridge strength by cross validation on held out matches - the real means are rebuilt from the
# training matches in each fold, then scored against the held out matches. Chosen on STATE LEVEL calibration (sample
# weighted RMS of held out situation mean minus prediction) - per ball RMSE is dominated by the natural spread of runs
# to come, barely moves across settings and always picked the stiffest surface, which under-fits the sim's bias
cvResults = []
for fold, (trainIdx, testIdx) in enumerate(GroupKFold(n_splits=CV_FOLDS).split(trainData, groups=trainData['matchID'])):
    foldSituations = situations_from(trainData.iloc[trainIdx])
    testStates = real_situation_runs_to_come(trainData.iloc[testIdx])
    for wicketKnots in WICKET_KNOTS_GRID:
        for ballKnots in BALL_KNOTS_GRID:
            for alpha in ALPHA_GRID:
                pred = build_runs_to_come(foldSituations, fit_log_bias(foldSituations, ballKnots, wicketKnots, alpha))
                scored = testStates.merge(pred[stateKeys + ['totalInningRunsToComeSimBiasSpline']], how='inner', on=stateKeys)
                err = scored['totalInningRunsToCome'] - scored['totalInningRunsToComeSimBiasSpline']
                cvResults.append((wicketKnots, ballKnots, alpha, fold, (scored['sample'] * err ** 2).sum(), scored['sample'].sum()))

cvResults = pd.DataFrame(cvResults, columns=['wicketKnots', 'ballKnots', 'alpha', 'fold', 'wsse', 'n'])
cvSummary = cvResults.groupby(['wicketKnots', 'ballKnots', 'alpha'])[['wsse', 'n']].sum()
cvSummary['stateCalibration'] = np.sqrt(cvSummary['wsse'] / cvSummary['n'])
print('\n=== cross validated state level calibration error on held out matches (runs) ===')
print(cvSummary[['stateCalibration']].round(4).to_string())
bestWicketKnots, bestKnots, bestAlpha = cvSummary['stateCalibration'].idxmin()
print(f'\nusing {bestWicketKnots} wicket knots, {bestKnots} ball knots, alpha {bestAlpha}')


# final fit on all first innings
situationRunsToCome = situations_from(trainData)
situationRunsToCome = build_runs_to_come(situationRunsToCome, fit_log_bias(situationRunsToCome, bestKnots, bestWicketKnots, bestAlpha))
situationRunsToCome = residual_correct(situationRunsToCome)
print(f"\nresidual stage correction (runs): mean {situationRunsToCome['residualCorrection'].mean():+.3f}, "
      f"largest {situationRunsToCome['residualCorrection'].min():+.2f} / {situationRunsToCome['residualCorrection'].max():+.2f}")

# keep v1's column meanings: simBias and m_simBias are sim / real, raw and modelled
situationRunsToCome['simBias'] = situationRunsToCome['totalInningRunsToComeSim'] / situationRunsToCome['totalInningRunsToCome']
situationRunsToCome['m_simBias'] = np.exp(-situationRunsToCome['logBias'])
# the spread scales with the mean
for col in ['totalInningRunsToComeSimSTD', 'totalInningRunsToComeSimMin', 'totalInningRunsToComeSimMax']:
    situationRunsToCome[col] = situationRunsToCome[col] / situationRunsToCome['m_simBias']

print('\n=== modelled sim / real bias by wickets and phase ===')
phase = pd.cut(situationRunsToCome['inningBallNumber'], [0, 36, 90, 109, 120], labels=['1-36', '37-90', '91-109', '110-120'])
print(situationRunsToCome.pivot_table(index='totalInningWickets', columns=phase, values='m_simBias', aggfunc='mean', observed=False).round(3).to_string())


# now we can merge the situation runs to come model numbers into the master lookup table which includes the ball by ball values
masterLookup = masterLookup.merge(situationRunsToCome.loc[:, ['totalInningWickets', 'inningBallNumber', 'totalInningRunsToCome', 'totalInningRunsToComeSim', 'simBias',
                                                              'm_simBias', 'totalInningRunsToComeSimBias', 'totalInningRunsToComeSimBiasSpline', 'totalInningRunsToComeSimSTD',
                                                              'totalInningRunsToComeSimSkew', 'totalInningRunsToComeSimKurt', 'totalInningRunsToComeSimMin', 'totalInningRunsToComeSimMax',
                                                              'totalInningValidBallsFacedToCome', 'bowledOut']],
                                  how='left', on=['totalInningWickets', 'inningBallNumber'])

print(f"\nsituations with no runs to come value (never reached in the sim): "
      f"{masterLookup.drop_duplicates(subset=stateKeys)['totalInningRunsToComeSimBiasSpline'].isna().sum()} of {masterLookup.drop_duplicates(subset=stateKeys).shape[0]}")


# export - new file name so step 5 can be switched over to it when ready
masterLookup.to_csv(PROJECT_ROOT / 'women/expBall&runsToCome/outputs/4_masterLookup_w_v2.csv', index=False)
