import pandas as pd
import numpy as np
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
from scipy.interpolate import UnivariateSpline
from paths import PROJECT_ROOT

# v2 of 4_runsToComeBiasSplineModel.py - calibrates the sim's runs to come to real first innings, by situation.
# changes from v1:
#   - reads the situation summary from 3_runsToComeSimClassOrd_v2.py (discrete sim, step 1 v2 inputs) instead of the raw sim balls
#   - the sim vs real bias is fit as one smooth surface over ball x wickets (log ratio, tensor product splines, ridge),
#     weighted by how noisy each situation's ratio is, instead of an unweighted depth 3 random forest per wicket
#   - the correction is applied at every ball (v1 skipped balls 110+ where the sim bias is largest)
#   - situations with thin real data keep their sim value, the correction there is borrowed from neighbouring
#     situations and shrinks toward no correction (v1 dropped them, leaving a third of situations empty)
#   - the smoothing spline is weighted by the sim's standard error, so it only smooths out sim noise
#   - the "more wickets can't mean more runs" fix only touches the runs to come value (v1 moved whole rows)
#   - no hand overrides at ball 120 / balls 1-5, and the smoothing strength is picked by cross validation on held out matches
#   - sim STD, min and max get scaled by the same correction as the mean, skew and kurtosis are scale free so unchanged


# import necessary data
trainData = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/data/dataClean.csv',
                        usecols=['matchID', 'inningNumber', 'inningBallNumber', 'totalInningWickets', 'totalInningRunsToCome', 'totalInningWicketsToCome'])
trainData = trainData[trainData['inningNumber'] == 1].reset_index(drop=True)
simSituationRunsToCome = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/outputs/ballSimsClassOrdSummary_v2.csv')
masterLookup = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/outputs/2_masterLookup_v2.csv')

stateKeys = ['inningBallNumber', 'totalInningWickets']
# same naming as v1 after its real vs sim merge
simSituationRunsToCome = simSituationRunsToCome.rename(columns={'totalInningRunsToCome': 'totalInningRunsToComeSim', 'sample': 'sampleSim'})

# settings
MIN_REAL_SAMPLE = 5                     # situations need at least this many real balls to be used when fitting the bias
BALL_KNOTS_GRID = [8, 12, 16]           # knots across inningBallNumber for the bias surface
WICKET_KNOTS = 6                        # knots across totalInningWickets for the bias surface
ALPHA_GRID = [0.01, 0.1, 1, 10, 100]    # ridge strength - higher shrinks the correction toward none where data is thin
CV_FOLDS = 5


def real_situation_runs_to_come(data):
    return data.groupby(stateKeys).agg(
        totalInningRunsToCome=('totalInningRunsToCome', 'mean'),
        totalInningWicketsToCome=('totalInningWicketsToCome', 'mean'),
        sample=('totalInningRunsToCome', 'size')
    ).reset_index()


def bias_basis(states, ballKnots):
    # tensor product of cubic splines over ball and quadratic splines over wickets, knots fixed on the full grid
    ballSpline = SplineTransformer(n_knots=ballKnots, degree=3, extrapolation='linear').fit(np.arange(1, 121).reshape(-1, 1))
    wicketSpline = SplineTransformer(n_knots=WICKET_KNOTS, degree=2, extrapolation='linear').fit(np.arange(0, 10).reshape(-1, 1))
    B = ballSpline.transform(states[['inningBallNumber']].to_numpy())
    W = wicketSpline.transform(states[['totalInningWickets']].to_numpy())
    return np.einsum('ij,ik->ijk', B, W).reshape(len(states), -1)


def fit_log_bias(situations, ballKnots, alpha):
    # log(real / sim) per situation, weighted by 1 / its variance (delta method, sim STD standing in for real STD)
    fit = situations[(situations['sample'] >= MIN_REAL_SAMPLE) & (situations['totalInningRunsToCome'] > 0) &
                     (situations['totalInningRunsToComeSimCount'] >= 2)]
    y = np.log(fit['totalInningRunsToCome'] / fit['totalInningRunsToComeSim'])
    variance = fit['totalInningRunsToComeSimSTD'] ** 2 * (1 / (fit['sample'] * fit['totalInningRunsToCome'] ** 2) +
                                                          1 / (fit['totalInningRunsToComeSimCount'] * fit['totalInningRunsToComeSim'] ** 2))
    weights = 1 / variance
    weights = weights / weights.mean()
    model = Ridge(alpha=alpha).fit(bias_basis(fit, ballKnots), y, sample_weight=weights)
    return model.predict(bias_basis(situations, ballKnots))


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


def situations_from(realData):
    return simSituationRunsToCome.merge(real_situation_runs_to_come(realData), how='left', on=stateKeys)


# pick the knots and ridge strength by cross validation on held out matches - the real means are rebuilt from the
# training matches in each fold, then scored against every ball of the held out matches
cvResults = []
for fold, (trainIdx, testIdx) in enumerate(GroupKFold(n_splits=CV_FOLDS).split(trainData, groups=trainData['matchID'])):
    foldSituations = situations_from(trainData.iloc[trainIdx])
    testBalls = trainData.iloc[testIdx]
    for ballKnots in BALL_KNOTS_GRID:
        for alpha in ALPHA_GRID:
            pred = build_runs_to_come(foldSituations, fit_log_bias(foldSituations, ballKnots, alpha))
            scored = testBalls.merge(pred[stateKeys + ['totalInningRunsToComeSimBiasSpline']], how='inner', on=stateKeys)
            err = scored['totalInningRunsToCome'] - scored['totalInningRunsToComeSimBiasSpline']
            cvResults.append((ballKnots, alpha, fold, (err ** 2).sum(), err.abs().sum(), len(err)))

cvResults = pd.DataFrame(cvResults, columns=['ballKnots', 'alpha', 'fold', 'sse', 'sae', 'n'])
cvSummary = cvResults.groupby(['ballKnots', 'alpha'])[['sse', 'sae', 'n']].sum()
cvSummary['rmse'] = np.sqrt(cvSummary['sse'] / cvSummary['n'])
cvSummary['mae'] = cvSummary['sae'] / cvSummary['n']
print('\n=== cross validated error on held out matches ===')
print(cvSummary[['rmse', 'mae']].round(4).to_string())
bestKnots, bestAlpha = cvSummary['rmse'].idxmin()
print(f'\nusing {bestKnots} ball knots, alpha {bestAlpha}')


# final fit on all first innings
situationRunsToCome = situations_from(trainData)
situationRunsToCome = build_runs_to_come(situationRunsToCome, fit_log_bias(situationRunsToCome, bestKnots, bestAlpha))

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
masterLookup.to_csv(PROJECT_ROOT / 'men/expBall&runsToCome/outputs/4_masterLookup_v2.csv', index=False)
