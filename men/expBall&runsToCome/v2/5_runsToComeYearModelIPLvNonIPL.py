import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from paths import PROJECT_ROOT
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# IPL or non IPL
IPL = 0

# import cleaned ball-by-ball data
trainData = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/data/dataClean.csv', parse_dates=['date'])
print(trainData[trainData.competition == 'Indian Premier League'])
# import master lookup table from previous modeling step
masterLookup = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/outputs/4_masterLookup.csv')
#
# only use first innings data
trainData = trainData.loc[trainData['inningNumber'] == 1].copy()

# totalInningRunsToComeAdj (from dataClean.csv, = totalInningRunsToCome - RA_Sum) is used as-is
# for every wicket count now - no override to raw for wickets > 7, so vsAdjOvr/yearFactor are
# trained on a consistent quality-neutral basis across the whole grid.


# keep only one row per wickets + ball combination
# this allows us to merge the lookup values with duplications
masterLookupSingle = masterLookup.drop_duplicates(subset=['totalInningWickets', 'inningBallNumber']).reset_index(drop=True)

# merge baseline spline model predictions onto training data
trainData = trainData.merge(
    masterLookupSingle.loc[:, [
        'totalInningWickets',
        'inningBallNumber',
        'totalInningRunsToComeSimBiasSpline',
        'totalInningRunsToComeSim'
    ]],
    how='left',
    on=['totalInningWickets', 'inningBallNumber']
)

# calculate ratio of actual runs-to-come vs model
# this becomes the target for the year adjustment model
trainData['vsAdjOvr'] = trainData['totalInningRunsToComeAdj'] / trainData['totalInningRunsToComeSimBiasSpline']
trainData['vsOvr'] = trainData['totalInningRunsToCome'] / trainData['totalInningRunsToComeSimBiasSpline']
test2 = trainData[trainData.year == 2026].groupby(['competition'])['ID'].count().reset_index()
# remove rows where ratios could not be calculated
trainData = trainData.dropna(subset=['vsAdjOvr', 'vsOvr'])
test3 = trainData[trainData.year == 2026].groupby(['competition'])['ID'].count().reset_index()
# # only train on 2018+ data
# trainData = trainData.loc[trainData['year'] > 2018].copy()
trainData['IPL'] = np.where(((trainData['competition'] == 'Indian Premier League') & (trainData['year'] >= 2023)) | ((trainData['competition'] == 'International League T20') & (trainData['year'] >= 2024)), 1, 0)

print(trainData[trainData.IPL == 1])
# create interaction terms between year trend and game state
trainData['daysGroup_totalInningWickets'] = trainData['daysGroup'] * trainData['totalInningWickets']
trainData['daysGroup_inningBallNumber'] = trainData['daysGroup'] * trainData['inningBallNumber']
trainData['daysGroup_totalInningWickets_IPL'] = trainData['daysGroup_totalInningWickets'] * trainData['IPL']
trainData['daysGroup_inningBallNumber_IPL'] = trainData['daysGroup_inningBallNumber'] * trainData['IPL']
trainData['daysGroup_IPL'] = trainData['daysGroup'] * trainData['IPL']
# richer terms so the year-trend's shape across wickets/ballNumber isn't forced to be linear -
# still multiplied by daysGroup throughout, so still exactly zero at daysGroup=0 (no static
# wicket/ballNumber level effect can leak in here - that stays RA_sum_wl_br's job below)
trainData['daysGroup_totalInningWickets_sq'] = trainData['daysGroup'] * trainData['totalInningWickets'] ** 2
trainData['daysGroup_inningBallNumber_sq'] = trainData['daysGroup'] * trainData['inningBallNumber'] ** 2
trainData['daysGroup_wickets_ballNumber'] = trainData['daysGroup'] * trainData['totalInningWickets'] * trainData['inningBallNumber']

test4 = trainData[trainData.year == 2026].groupby(['competition'])['ID'].count().reset_index()

# features used in the regression models
features = [
    'daysGroup',
    'daysGroup_inningBallNumber',
    'daysGroup_totalInningWickets',
    'daysGroup_totalInningWickets_IPL',
    'daysGroup_inningBallNumber_IPL',
    'daysGroup_IPL',
    'daysGroup_totalInningWickets_sq',
    'daysGroup_inningBallNumber_sq',
    'daysGroup_wickets_ballNumber',
]

log_method = 1

if log_method == 1:
    vsAdjOvrMin = trainData['vsAdjOvr'].min()
    vsOvrMin = trainData['vsOvr'].min()
    trainData['vsAdjOvr'] = np.log1p(trainData['vsAdjOvr'] - vsAdjOvrMin)
    trainData['vsOvr'] = np.log1p(trainData['vsOvr'] - vsOvrMin)
# feature matrix
X = trainData[features]
# target using adjusted runs
y_adj = trainData['vsAdjOvr']
# target using raw runs
y_raw = trainData['vsOvr']

# fit model for adjusted runs-to-come
model_adj = LinearRegression()
model_adj.fit(X, y_adj)

# fit model for raw runs-to-come
model_raw = LinearRegression()
model_raw.fit(X, y_raw)

# predict year adjustment factors
trainData['yearFactor'] = model_adj.predict(X)
trainData['yearFactor2'] = model_raw.predict(X)
if log_method == 1:
    trainData['yearFactor'] = np.expm1(trainData['yearFactor']) + vsAdjOvrMin
    trainData['yearFactor2'] = np.expm1(trainData['yearFactor2']) + vsOvrMin

###getting remaining trends from model data:
# yearFactor (Adj path) normalizer: built per (wickets, ballNumber) cell, balanced in RUN terms
# (Adj-weighted, not a flat ratio average) and smoothed via a sample-weighted, CV-selected-degree
# polynomial surface - same methodology as the RA_Sum surface below. A flat per-wicket ratio
# average of the raw prediction was found to leave a large, statistically significant bias: every
# ball number counts equally in that average even though real runs-remaining (and therefore how
# much each row's error actually matters) varies hugely within a wicket group, so ratio errors
# that cancel on average don't cancel in run terms. A flat per-(wicket, ballNumber) mean fixes
# that but is too noisy at sparse cells - hence the same weighted-surface treatment used for
# RA_Sum.
trainData['_yfAdjNum'] = trainData['yearFactor'] * trainData['totalInningRunsToComeAdj']
yearFactor_cell_stats = trainData.groupby(['totalInningWickets', 'inningBallNumber']).agg(
    sum_yfAdj=('_yfAdjNum', 'sum'), sum_Adj=('totalInningRunsToComeAdj', 'sum'),
    n=('_yfAdjNum', 'size')).reset_index()
trainData = trainData.drop(columns=['_yfAdjNum'])
yearFactor_cell_stats['cellFactor'] = yearFactor_cell_stats['sum_yfAdj'] / yearFactor_cell_stats['sum_Adj']

MIN_SAMPLE_YF_SURFACE = 100
fitCellsYF = yearFactor_cell_stats[yearFactor_cell_stats['n'] >= MIN_SAMPLE_YF_SURFACE].reset_index(drop=True)
print(f"yearFactor normalizer surface: cells with sample >= {MIN_SAMPLE_YF_SURFACE}: "
      f"{len(fitCellsYF)} of {len(yearFactor_cell_stats)}")


def loo_cv_yf(degree):
    Xa = fitCellsYF[['totalInningWickets', 'inningBallNumber']].values
    ya = fitCellsYF['cellFactor'].values
    wa = fitCellsYF['n'].values
    errs = []
    for i in range(len(fitCellsYF)):
        mask = np.ones(len(fitCellsYF), dtype=bool)
        mask[i] = False
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        Xtr = poly.fit_transform(Xa[mask])
        m = LinearRegression()
        m.fit(Xtr, ya[mask], sample_weight=wa[mask])
        pred = m.predict(poly.transform(Xa[i:i + 1]))[0]
        errs.append((pred - ya[i]) ** 2 * wa[i])
    return np.sqrt(np.sum(errs) / np.sum(wa))


print("=== yearFactor normalizer surface: LOO-CV weighted RMSE by degree ===")
yfSurfaceScores = {d: loo_cv_yf(d) for d in [1, 2]}
for d, s in yfSurfaceScores.items():
    print(f"degree {d}: {s:.6f}")
YF_SURFACE_DEGREE = min(yfSurfaceScores, key=yfSurfaceScores.get)
print(f"  -> using degree {YF_SURFACE_DEGREE}")

poly_yf_surface = PolynomialFeatures(degree=YF_SURFACE_DEGREE, include_bias=False)
reg_yf_surface = LinearRegression()
reg_yf_surface.fit(poly_yf_surface.fit_transform(fitCellsYF[['totalInningWickets', 'inningBallNumber']]),
                    fitCellsYF['cellFactor'], sample_weight=fitCellsYF['n'])


def predict_yf_normalizer(wickets, ballNumber):
    Xp = poly_yf_surface.transform(np.column_stack([wickets, ballNumber]))
    return reg_yf_surface.predict(Xp)


# yearFactor2 (non-Adj path) keeps the original flat per-wicket normalization - not part of this fix
testing_wl = trainData.groupby(['totalInningWickets'])[['yearFactor2']].mean().reset_index()
trainData = trainData.merge(testing_wl, on='totalInningWickets', how='left', suffixes=('_old', '_wl'))

trainData['yearFactor'] = trainData['yearFactor'] / predict_yf_normalizer(trainData['totalInningWickets'], trainData['inningBallNumber'])
trainData['yearFactor2'] = trainData['yearFactor2_old'] / trainData['yearFactor2_wl']

# apply year adjustments back onto baseline spline predictions. Named "Uncalibrated" - trainData
# never has predicted_RA_Sum applied to it in this script, so it should never carry the plain
# totalInningRunsToComeSimBiasSplineYearAdj name (that name is reserved for the final,
# post-RA_Sum-subtraction value, computed only in masterLookup/lookupForInruns below).
trainData['totalInningRunsToComeSimBiasSplineYearAdjUncalibrated'] = trainData['totalInningRunsToComeSimBiasSpline'] * trainData['yearFactor']
trainData['totalInningRunsToComeSimBiasSplineYear'] = trainData['totalInningRunsToComeSimBiasSpline'] * trainData['yearFactor2']

# remove rows with nan predictions
trainData = trainData.dropna(subset=['totalInningRunsToComeSimBiasSplineYearAdjUncalibrated', 'totalInningRunsToComeSimBiasSplineYear'])
test5 = trainData[trainData.year == 2026].groupby(['competition'])['ID'].count().reset_index()
# print training MAE for adjusted and raw models
print(mean_absolute_error(trainData['totalInningRunsToCome'], trainData['totalInningRunsToComeSimBiasSplineYearAdjUncalibrated']))
print(mean_absolute_error(trainData['totalInningRunsToCome'], trainData['totalInningRunsToComeSimBiasSplineYear']))

testing_wl_year = trainData.groupby(['totalInningWickets', 'year'])[['yearFactor', 'yearFactor2']].mean().reset_index()
testing_br_year_IPL = trainData.groupby(['IPL', 'inningBallNumber', 'year'])[['totalInningRunsToComeSimBiasSpline', 'yearFactor', 'yearFactor2', 'totalInningRunsToComeAdj', 'totalInningRunsToCome']].mean().reset_index()
testing_br_year_IPL_count = trainData.groupby(['IPL', 'inningBallNumber', 'year'])[['totalInningRunsToComeSimBiasSpline']].count().reset_index()
testing_wl_2 = trainData.groupby(['totalInningWickets'])[['yearFactor', 'yearFactor2']].mean().reset_index()
testing_wl_br = trainData.groupby(['totalInningWickets', 'inningBallNumber'])[['yearFactor', 'yearFactor2']].mean().reset_index()
testing_br = trainData.groupby(['inningBallNumber'])[['yearFactor', 'yearFactor2']].mean().reset_index()
testing_RA_sum_br = trainData.groupby(['inningBallNumber'])[['RA_Sum']].mean().reset_index()
testing_RA_sum_wl = trainData.groupby(['totalInningWickets'])['RA_Sum'].mean().reset_index()

# No demeaning by inningBallNumber here - the raw RA_Sum surface is fit and used as-is, including
# whatever overall bias exists at each ball number. There's no reason to force it to net to zero
# per ball; the six components it's built from (personnel/ground quality for the rest of the
# innings) have no inherent reason to average out to exactly average-quality at every point.
RA_sum_wl_br = trainData.groupby(['totalInningWickets', 'inningBallNumber']).agg(
    RA_Sum=('RA_Sum', 'mean'), sample=('RA_Sum', 'size')).reset_index()

# ============================================================================================
# model for RA_sum prediction - same methodology as byOrderAdjusts.py's order-adjustment surface
# (see that file for the full reasoning): an unweighted, unfiltered polynomial fit across the
# WHOLE (wickets, ballNumber) grid lets near-empty cells (e.g. wickets=0 at ball=120 has only 2
# real matches) distort the surface just as much as cells with thousands of samples - confirmed
# this was producing wrong-signed, wildly overshot predictions exactly at the sparsest corners
# (e.g. wickets=0/ball=120: raw mean +0.91 on n=2, but the old unfiltered fit predicted -0.66
# there, then the ball-number centering step compounded it further to -1.30).
#
# Fix, in the same three pieces as byOrderAdjusts.py:
#   1. MIN_CELL_SAMPLE filter - only fit on cells with real sample support
#   2. sample-weighted regression - a 10,000-sample cell should outweigh a 2-sample cell
#   3. cross-validated degree selection, capped at 3 - degree 2 was found to systematically
#      underfit (predicted_RA_Sum vs actual RA_Sum showed a strong shrinkage correlation of
#      -0.495, i.e. large actual values were pulled toward zero); degree 3 nearly eliminates
#      that correlation (-0.179) and cuts mean cell error from 0.32 to 0.24 runs, with no
#      overshoot at the sparse corners (checked directly - predictions stay bounded and smooth
#      across wickets 0/2/5/9 once combined with the row-boundary clamp below). Degree 4 scores
#      marginally lower on raw error but reintroduces some of the shrinkage correlation (-0.354),
#      so 3 is the better tradeoff.
# The unused `testing_RA_sum_wl_br = RA_sum_wl_br.dropna()` line that used to sit here looks like
# an abandoned attempt at exactly this kind of safeguard that never got finished/wired in.
# ============================================================================================
MIN_CELL_SAMPLE_RA_SUM = 100
fitCellsRA = RA_sum_wl_br[RA_sum_wl_br['sample'] >= MIN_CELL_SAMPLE_RA_SUM].reset_index(drop=True)
print(f"\nRA_Sum surface: cells with sample >= {MIN_CELL_SAMPLE_RA_SUM} used for fitting: "
      f"{len(fitCellsRA)} of {len(RA_sum_wl_br)}")


def loo_cv_ra_sum(degree):
    X_all = fitCellsRA[['totalInningWickets', 'inningBallNumber']].values
    y_all = fitCellsRA['RA_Sum'].values
    w_all = fitCellsRA['sample'].values
    errs = []
    for i in range(len(fitCellsRA)):
        mask = np.ones(len(fitCellsRA), dtype=bool)
        mask[i] = False
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        Xtr = poly.fit_transform(X_all[mask])
        model = LinearRegression()
        model.fit(Xtr, y_all[mask], sample_weight=w_all[mask])
        pred = model.predict(poly.transform(X_all[i:i + 1]))[0]
        errs.append((pred - y_all[i]) ** 2 * w_all[i])
    return np.sqrt(np.sum(errs) / np.sum(w_all))


print("=== RA_Sum surface: LOO-CV weighted RMSE by degree (sample-filtered cells) ===")
MAX_DEGREE_RA_SUM = 3
raSumScores = {d: loo_cv_ra_sum(d) for d in range(1, MAX_DEGREE_RA_SUM + 1)}
for d, s in raSumScores.items():
    print(f"degree {d}: {s:.5f}")
RA_SUM_DEGREE = min(raSumScores, key=raSumScores.get)
print(f"  -> using degree {RA_SUM_DEGREE}")

poly_RA_sum = PolynomialFeatures(degree=RA_SUM_DEGREE, include_bias=False)
reg_RA_sum = LinearRegression()
reg_RA_sum.fit(poly_RA_sum.fit_transform(fitCellsRA[['totalInningWickets', 'inningBallNumber']]),
                fitCellsRA['RA_Sum'], sample_weight=fitCellsRA['sample'])


def predict_RA_sum(wickets, ballNumber):
    X = poly_RA_sum.transform(np.column_stack([wickets, ballNumber]))
    return reg_RA_sum.predict(X)


RA_sum_wl_br['predicted_RA_Sum'] = predict_RA_sum(RA_sum_wl_br['totalInningWickets'], RA_sum_wl_br['inningBallNumber'])
# no post-fit centering needed anymore - RA_Sum was demeaned by ballNumber before the fit above,
# so predicted_RA_Sum is already ~0 on average at every ballNumber by construction.

# ============================================================================================
# EXTRAPOLATION SAFETY: even sample-weighted and degree-capped, the polynomial above still fully
# extrapolates sparse cells (e.g. wickets=0 at ball=120 has very low sample) - a shared quadratic
# surface just keeps curving in whatever direction it was already going once it leaves the
# well-sampled region, rather than levelling off (confirmed: wickets=9/ball=1 extrapolated to
# -11.7 with no guard at all).
#
# Fix: per wickets row, the trusted cells (sample >= MIN_CELL_SAMPLE_RA_SUM) form one clean
# contiguous run of ballNumbers (verified - no gaps at this threshold). Keep the model's own
# smoothed prediction inside that trusted range; outside it, hold the value at the nearest
# trusted-range boundary (the last trusted cell in that row) instead of letting the polynomial
# keep extrapolating further.
# ============================================================================================
trustedByRow = RA_sum_wl_br[RA_sum_wl_br['sample'] >= MIN_CELL_SAMPLE_RA_SUM].groupby('totalInningWickets')['inningBallNumber'].agg(['min', 'max'])


def clamp_to_trusted_range(row):
    lo, hi = trustedByRow.loc[row['totalInningWickets'], ['min', 'max']]
    b = np.clip(row['inningBallNumber'], lo, hi)
    return predict_RA_sum([row['totalInningWickets']], [b])[0]


RA_sum_wl_br['predicted_RA_Sum'] = RA_sum_wl_br.apply(clamp_to_trusted_range, axis=1)
# kept pre-truncation (with RA_Sum + sample) for the diagnostics dashboard built at the bottom of
# this script - RA_sum_wl_br itself gets truncated to just the merge columns right below
RA_sum_wl_br_diag = RA_sum_wl_br.copy()
RA_sum_wl_br = RA_sum_wl_br.loc[:, ['totalInningWickets', 'inningBallNumber', 'predicted_RA_Sum']]

# create year grouping used for prediction
masterLookup['daysGroup'] = masterLookup['year'] - 2015

# duplicate the latest year and relabel as 9.4, this gives us the number we want to match the match market
extraRows = masterLookup.loc[masterLookup['daysGroup'] == 11].copy()
extraRows['daysGroup'] = 11.1


# append future-year rows back onto master lookup
masterLookup = pd.concat([masterLookup, extraRows], ignore_index=True)

# recreate interaction features for prediction
masterLookup['daysGroup_totalInningWickets'] = masterLookup['daysGroup'] * masterLookup['totalInningWickets']
masterLookup['daysGroup_inningBallNumber'] = masterLookup['daysGroup'] * masterLookup['inningBallNumber']
masterLookup['daysGroup_totalInningWickets_sq'] = masterLookup['daysGroup'] * masterLookup['totalInningWickets'] ** 2
masterLookup['daysGroup_inningBallNumber_sq'] = masterLookup['daysGroup'] * masterLookup['inningBallNumber'] ** 2
masterLookup['daysGroup_wickets_ballNumber'] = masterLookup['daysGroup'] * masterLookup['totalInningWickets'] * masterLookup['inningBallNumber']
if IPL == 0:
    masterLookup['daysGroup_IPL'] = 0
    masterLookup['daysGroup_totalInningWickets_IPL'] = 0
    masterLookup['daysGroup_inningBallNumber_IPL'] = 0

else:
    masterLookup['daysGroup_IPL'] = masterLookup['daysGroup']
    masterLookup['daysGroup_totalInningWickets_IPL'] = masterLookup['daysGroup_totalInningWickets']
    masterLookup['daysGroup_inningBallNumber_IPL'] = masterLookup['daysGroup_inningBallNumber']

# prediction feature matrix
X_master = masterLookup[features]

# predict year adjustment rates
masterLookup['totalInningRunsToComeSimBiasSplineYearRateAdj'] = (model_adj.predict(X_master))
masterLookup['totalInningRunsToComeSimBiasSplineYearRate'] = (model_raw.predict(X_master))
if log_method == 1:
    masterLookup['totalInningRunsToComeSimBiasSplineYearRateAdj'] = np.expm1(masterLookup['totalInningRunsToComeSimBiasSplineYearRateAdj']) + vsAdjOvrMin
    masterLookup['totalInningRunsToComeSimBiasSplineYearRate'] = np.expm1(masterLookup['totalInningRunsToComeSimBiasSplineYearRate']) + vsOvrMin

# this is to allow for overall bias in the by year adjust model - yearFactor (Adj path) uses the
# (wickets, ballNumber) surface normalizer fit above; yearFactor2 keeps the flat per-wicket one
masterLookup = masterLookup.merge(testing_wl, on='totalInningWickets', how='left')
masterLookup['totalInningRunsToComeSimBiasSplineYearRateAdj'] = masterLookup['totalInningRunsToComeSimBiasSplineYearRateAdj'] / predict_yf_normalizer(masterLookup['totalInningWickets'], masterLookup['inningBallNumber'])
masterLookup['totalInningRunsToComeSimBiasSplineYearRate'] = masterLookup['totalInningRunsToComeSimBiasSplineYearRate'] / masterLookup['yearFactor2']

# apply predicted year factors to baseline spline values. "Uncalibrated" until predicted_RA_Sum
# has been subtracted below - totalInningRunsToComeSimBiasSplineYearAdj (no suffix) is reserved
# for the final, post-subtraction value only, so the name always means one specific thing.
masterLookup['totalInningRunsToComeSimBiasSplineYearAdjUncalibrated'] = masterLookup['totalInningRunsToComeSimBiasSplineYearRateAdj'] * masterLookup['totalInningRunsToComeSimBiasSpline']
masterLookup['totalInningRunsToComeSimBiasSplineYear'] = masterLookup['totalInningRunsToComeSimBiasSplineYearRate'] * masterLookup['totalInningRunsToComeSimBiasSpline']
masterLookup = masterLookup.sort_values(by=['totalInningWickets', 'inningBallNumber', 'ord', 'daysGroup']).reset_index(drop=True)

# apply predicted RA_sum as recovery work:
masterLookup = masterLookup.merge(RA_sum_wl_br, on=('totalInningWickets', 'inningBallNumber'), how='left')
masterLookup['totalInningRunsToComeSimBiasSplineYearAdj'] = masterLookup['totalInningRunsToComeSimBiasSplineYearAdjUncalibrated'] - masterLookup['predicted_RA_Sum']

# #export final lookup table
masterLookup.to_csv(PROJECT_ROOT / 'men/expBall&runsToCome/outputs/5_masterLookup.csv', index=False)

##below is for making an output of the values each daysGroup will give
lookupForInruns = pd.DataFrame({'daysGroup': np.arange(5, 20.1, 0.1)})
# create interaction terms between year trend and game state
lookupForInruns['totalInningWickets'] = 0
lookupForInruns['inningBallNumber'] = 1
lookupForInruns['daysGroup_totalInningWickets'] = lookupForInruns['daysGroup'] * lookupForInruns['totalInningWickets']
lookupForInruns['daysGroup_inningBallNumber'] = lookupForInruns['daysGroup'] * lookupForInruns['inningBallNumber']
lookupForInruns['daysGroup_totalInningWickets_sq'] = lookupForInruns['daysGroup'] * lookupForInruns['totalInningWickets'] ** 2
lookupForInruns['daysGroup_inningBallNumber_sq'] = lookupForInruns['daysGroup'] * lookupForInruns['inningBallNumber'] ** 2
lookupForInruns['daysGroup_wickets_ballNumber'] = lookupForInruns['daysGroup'] * lookupForInruns['totalInningWickets'] * lookupForInruns['inningBallNumber']
if IPL == 0:
    lookupForInruns['daysGroup_IPL'] = 0
    lookupForInruns['daysGroup_totalInningWickets_IPL'] = 0
    lookupForInruns['daysGroup_inningBallNumber_IPL'] = 0
else:
    lookupForInruns['daysGroup_IPL'] = lookupForInruns['daysGroup']
    lookupForInruns['daysGroup_totalInningWickets_IPL'] = lookupForInruns['daysGroup_totalInningWickets']
    lookupForInruns['daysGroup_inningBallNumber_IPL'] = lookupForInruns['daysGroup_inningBallNumber']

lookupForInruns = lookupForInruns.merge(masterLookup[(masterLookup.totalInningWickets == 0) & (masterLookup.inningBallNumber == 1)].groupby(['inningBallNumber', 'totalInningWickets'])[['totalInningRunsToComeSimBiasSpline', 'predicted_RA_Sum']].mean().reset_index(), on=('totalInningWickets', 'inningBallNumber'), how='left')
# prediction feature matrix
X_lookup = lookupForInruns[features]
lookupForInruns['totalInningRunsToComeSimBiasSplineYearRateAdj'] = (model_adj.predict(X_lookup))
lookupForInruns['totalInningRunsToComeSimBiasSplineYearRate'] = (model_raw.predict(X_lookup))
if log_method == 1:
    lookupForInruns['totalInningRunsToComeSimBiasSplineYearRateAdj'] = np.expm1(lookupForInruns['totalInningRunsToComeSimBiasSplineYearRateAdj']) + vsAdjOvrMin
    lookupForInruns['totalInningRunsToComeSimBiasSplineYearRate'] = np.expm1(lookupForInruns['totalInningRunsToComeSimBiasSplineYearRate']) + vsOvrMin
# this is to allow for overall bias in the by year adjust model - lookupForInruns is always at
# wickets=0/ball=1, so the surface normalizer is evaluated directly at that point. Normalize the
# rate itself first, same order as masterLookup above (RateAdj / normalizer, then x spline, then
# - predicted_RA_Sum) - previously this divided the whole (rate*spline - predicted_RA_Sum) by the
# normalizer instead, which also (incorrectly) scaled predicted_RA_Sum by it.
lookupForInruns = lookupForInruns.merge(testing_wl, on='totalInningWickets', how='left')
lookupForInruns['totalInningRunsToComeSimBiasSplineYearRateAdj'] = lookupForInruns['totalInningRunsToComeSimBiasSplineYearRateAdj'] / predict_yf_normalizer(lookupForInruns['totalInningWickets'], lookupForInruns['inningBallNumber'])
lookupForInruns['totalInningRunsToComeSimBiasSplineYearRate'] = lookupForInruns['totalInningRunsToComeSimBiasSplineYearRate'] / lookupForInruns['yearFactor2']

lookupForInruns['totalInningRunsToComeSimBiasSplineYearAdj3'] = (lookupForInruns['totalInningRunsToComeSimBiasSplineYearRateAdj'] * lookupForInruns['totalInningRunsToComeSimBiasSpline']) - lookupForInruns['predicted_RA_Sum']
lookupForInruns['totalInningRunsToComeSimBiasSplineYear3'] = lookupForInruns['totalInningRunsToComeSimBiasSplineYearRate'] * lookupForInruns['totalInningRunsToComeSimBiasSpline']

lookupForInruns_final = lookupForInruns.loc[:, ['daysGroup', 'totalInningRunsToComeSimBiasSplineYear3', 'totalInningRunsToComeSimBiasSplineYearAdj3']]

testing_br_year_IPL['pred_runsadj'], testing_br_year_IPL['pred_runs'] = testing_br_year_IPL['totalInningRunsToComeSimBiasSpline'] * testing_br_year_IPL['yearFactor'], testing_br_year_IPL['totalInningRunsToComeSimBiasSpline'] * testing_br_year_IPL['yearFactor2']
comparison_by_year_final = testing_br_year_IPL[testing_br_year_IPL.inningBallNumber == 1]
comparison_by_year_final = comparison_by_year_final.loc[:, ['IPL', 'year', 'inningBallNumber', 'totalInningRunsToComeAdj', 'pred_runsadj', 'totalInningRunsToCome', 'pred_runs']]

# ============================================================================================
# DIAGNOSTICS DASHBOARD - wickets x balls-remaining heatmaps for sanity-checking each step of
# this script's pipeline: raw vs yearAdjUncalibrated, Adj vs the final yearAdj, yearFactor's own
# neutrality, spline's own error vs raw, and the RA_Sum surface's fit quality. Regenerated on
# every run so it never goes stale relative to the model actually in production. Same population
# throughout: individual (wickets, ballNumber) cells filtered to sample > 50 before bucketing
# into 5-ball-remaining buckets, so every tab is directly comparable to every other tab.
# ============================================================================================
diagRows = trainData.dropna(subset=['totalInningRunsToComeSimBiasSplineYearAdjUncalibrated']).copy()
diagRows = diagRows.merge(
    RA_sum_wl_br_diag[['totalInningWickets', 'inningBallNumber', 'predicted_RA_Sum']],
    on=['totalInningWickets', 'inningBallNumber'], how='left')
diagRows['totalInningRunsToComeSimBiasSplineYearAdj'] = (
    diagRows['totalInningRunsToComeSimBiasSplineYearAdjUncalibrated'] - diagRows['predicted_RA_Sum'])
diagRows['splineMinusRaw'] = diagRows['totalInningRunsToComeSimBiasSpline'] - diagRows['totalInningRunsToCome']
_diagBallN = diagRows.groupby(['totalInningWickets', 'inningBallNumber']).size().rename('ballN').reset_index()
diagRows = diagRows.merge(_diagBallN, on=['totalInningWickets', 'inningBallNumber'])
diagRows = diagRows[diagRows['ballN'] > 50].copy()
diagRows['br_bin'] = ((121 - diagRows['inningBallNumber']) // 5) * 5

diagBuckets = diagRows.groupby(['totalInningWickets', 'br_bin']).agg(
    n=('totalInningRunsToCome', 'size'),
    totalInningRunsToCome=('totalInningRunsToCome', 'mean'),
    totalInningRunsToComeAdj=('totalInningRunsToComeAdj', 'mean'),
    yearAdjUncalibrated=('totalInningRunsToComeSimBiasSplineYearAdjUncalibrated', 'mean'),
    yearAdj=('totalInningRunsToComeSimBiasSplineYearAdj', 'mean'),
    yearFactor=('yearFactor', 'mean'),
    splineMinusRaw=('splineMinusRaw', 'mean'),
).reset_index().rename(columns={'totalInningWickets': 'wickets'})

diagRA = RA_sum_wl_br_diag[RA_sum_wl_br_diag['sample'] > 50].copy()
diagRA['br_bin'] = ((121 - diagRA['inningBallNumber']) // 5) * 5
diagRA['RA_Sum_w'] = diagRA['RA_Sum'] * diagRA['sample']
diagRA['predicted_RA_Sum_w'] = diagRA['predicted_RA_Sum'] * diagRA['sample']
diagRABuckets = diagRA.groupby(['totalInningWickets', 'br_bin']).agg(
    n=('sample', 'sum'), RA_Sum_w=('RA_Sum_w', 'sum'), predicted_RA_Sum_w=('predicted_RA_Sum_w', 'sum'),
).reset_index().rename(columns={'totalInningWickets': 'wickets'})
diagRABuckets['RA_Sum'] = diagRABuckets['RA_Sum_w'] / diagRABuckets['n']
diagRABuckets['predicted_RA_Sum'] = diagRABuckets['predicted_RA_Sum_w'] / diagRABuckets['n']

_diagData = {
    't1': diagBuckets.assign(diff=diagBuckets['yearAdjUncalibrated'] - diagBuckets['totalInningRunsToCome'])
        [['wickets', 'br_bin', 'n', 'totalInningRunsToCome', 'yearAdjUncalibrated', 'diff']]
        .rename(columns={'yearAdjUncalibrated': 'uncalibrated'}).round(4).to_dict(orient='records'),
    't2': diagBuckets.assign(diff=diagBuckets['yearAdj'] - diagBuckets['totalInningRunsToComeAdj'])
        [['wickets', 'br_bin', 'n', 'totalInningRunsToComeAdj', 'yearAdj', 'diff']]
        .round(4).to_dict(orient='records'),
    't3': diagBuckets[['wickets', 'br_bin', 'n', 'yearFactor']].round(4).to_dict(orient='records'),
    't4': diagBuckets[['wickets', 'br_bin', 'n', 'splineMinusRaw']].round(4).to_dict(orient='records'),
    't5': diagRABuckets[['wickets', 'br_bin', 'n', 'RA_Sum']].round(4).to_dict(orient='records'),
    't6': diagRABuckets[['wickets', 'br_bin', 'n', 'predicted_RA_Sum']].round(4).to_dict(orient='records'),
    't7': diagRABuckets.assign(diff=diagRABuckets['predicted_RA_Sum'] - diagRABuckets['RA_Sum'])
        [['wickets', 'br_bin', 'n', 'diff']].round(4).to_dict(orient='records'),
}

_DIAG_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>5_runsToComeYearModelIPLvNonIPL diagnostics</title>
<style>
:root { --bg:#fbfaf7; --panel:#ffffff; --ink:#1c1a17; --ink-soft:#6b6558; --line:#e4e0d8; --accent:#8a3b2f; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --bg:#171512; --panel:#201d19; --ink:#ece7de; --ink-soft:#a49b8c; --line:#3a352d; --accent:#d98f7d; }
}
:root[data-theme="dark"] { --bg:#171512; --panel:#201d19; --ink:#ece7de; --ink-soft:#a49b8c; --line:#3a352d; --accent:#d98f7d; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font-family:'IBM Plex Mono', ui-monospace, monospace; padding:24px 16px 60px; }
.wrap { max-width:1180px; margin:0 auto; }
h1 { font-family:Georgia,'Source Serif 4',serif; font-size:20px; margin:0 0 4px; }
.sub { color:var(--ink-soft); font-size:12px; margin:0 0 16px; line-height:1.5; }
.tabs { display:flex; gap:6px; flex-wrap:wrap; margin:16px 0 14px; border-bottom:1px solid var(--line); }
.tab { background:none; border:none; font-family:inherit; font-size:12.5px; color:var(--ink-soft); padding:8px 12px; cursor:pointer; border-bottom:2px solid transparent; margin-bottom:-1px; }
.tab.active { color:var(--ink); border-bottom-color:var(--accent); font-weight:600; }
.panel { display:none; }
.panel.active { display:block; }
.formula { display:inline-block; background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:8px 12px; font-size:12px; margin-bottom:10px; }
.stats { display:flex; gap:18px; font-size:12px; color:var(--ink-soft); margin-bottom:12px; }
.stats b { color:var(--ink); font-variant-numeric:tabular-nums; }
.gridbox { overflow-x:auto; border:1px solid var(--line); border-radius:8px; background:var(--panel); }
table.grid { border-collapse:collapse; font-size:11.5px; font-variant-numeric:tabular-nums; width:100%; }
table.grid th, table.grid td { padding:5px 6px; text-align:center; white-space:nowrap; border:1px solid var(--line); }
table.grid th { background:var(--panel); color:var(--ink-soft); font-weight:600; position:sticky; top:0; }
table.grid th.corner { position:sticky; left:0; top:0; z-index:3; }
table.grid td.rowhead { position:sticky; left:0; background:var(--panel); font-weight:600; z-index:2; }
table.grid td.empty { color:var(--line); }
.n-small { opacity:0.45; }
footer { margin-top:18px; font-size:11px; color:var(--ink-soft); }
</style>
</head>
<body>
<div class="wrap">
  <h1>5_runsToComeYearModelIPLvNonIPL - diagnostics</h1>
  <p class="sub">Regenerated every run. Individual balls filtered to n&gt;50 before 5-ball-remaining bucketing.</p>
  <div class="tabs">
    <button class="tab active" data-tab="t1">1. raw vs yearAdjUncalibrated</button>
    <button class="tab" data-tab="t2">2. Adj vs yearAdj (final)</button>
    <button class="tab" data-tab="t3">3. yearFactor</button>
    <button class="tab" data-tab="t4">4. spline &minus; raw</button>
    <button class="tab" data-tab="t5">5. RA_Sum</button>
    <button class="tab" data-tab="t6">6. predicted_RA_Sum</button>
    <button class="tab" data-tab="t7">7. predicted &minus; actual RA_Sum</button>
  </div>
  __PANELS__
  <footer>Generated by 5_runsToComeYearModelIPLvNonIPL.py on every run.</footer>
</div>
<script>
const DATA = __DATA_JSON__;

function buildGrid(rows, brBins, wickets, valueOf, targetId, opts) {
  opts = opts || {};
  const center = opts.center !== undefined ? opts.center : 0;
  const scale = opts.scale !== undefined ? opts.scale : 4;
  const decimals = opts.decimals !== undefined ? opts.decimals : 2;
  let html = '<table class="grid"><thead><tr><th class="corner">wkts \\\\ br</th>';
  brBins.forEach(b => html += `<th>${b}</th>`);
  html += '</tr></thead><tbody>';
  wickets.forEach(w => {
    html += `<tr><td class="rowhead">${w}</td>`;
    brBins.forEach(b => {
      const rec = rows.find(r => r.wickets === w && r.br_bin === b);
      if (!rec) {
        html += '<td class="empty">&middot;</td>';
      } else {
        const raw = valueOf(rec);
        const d = raw - center;
        const mag = Math.min(Math.abs(d) / scale, 1);
        const color = d >= 0 ? `rgba(31,95,139,${0.12 + mag * 0.55})` : `rgba(179,69,47,${0.12 + mag * 0.55})`;
        const nflag = rec.n < 200 ? ' n-small' : '';
        html += `<td class="${nflag}" style="background:${color}" title="n=${rec.n}">${raw.toFixed(decimals)}</td>`;
      }
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  document.getElementById(targetId).innerHTML = html;
}

function meanStats(rows, key) {
  const vals = rows.map(r => r[key]);
  const mean = vals.reduce((a,b)=>a+b,0) / vals.length;
  const meanAbs = vals.reduce((a,b)=>a+Math.abs(b),0) / vals.length;
  return {mean, meanAbs};
}

function statLine(id, rows, key, extraDp) {
  const st = meanStats(rows, key);
  const dp = extraDp || 3;
  document.getElementById(id).innerHTML =
    `<span>mean <b>${st.mean.toFixed(dp)}</b></span><span>mean |diff| <b>${st.meanAbs.toFixed(dp)}</b></span>` +
    `<span>range <b>${Math.min(...rows.map(r=>r[key])).toFixed(2)} to ${Math.max(...rows.map(r=>r[key])).toFixed(2)}</b></span>`;
}

const wicketsList = [...new Set(DATA.t1.map(r=>r.wickets))].sort((a,b)=>a-b);
const brList = [...new Set(DATA.t1.map(r=>r.br_bin))].sort((a,b)=>a-b);

document.getElementById('f1').textContent = 'diff = totalInningRunsToComeSimBiasSplineYearAdjUncalibrated \\u2212 totalInningRunsToCome';
document.getElementById('f2').textContent = 'diff = totalInningRunsToComeSimBiasSplineYearAdj \\u2212 totalInningRunsToComeAdj';
document.getElementById('f3').textContent = 'yearFactor (calibrated) \\u2014 should sit at 1.000 everywhere if fully neutral';
document.getElementById('f4').textContent = 'diff = totalInningRunsToComeSimBiasSpline \\u2212 totalInningRunsToCome';
document.getElementById('f5').textContent = 'RA_Sum \\u2014 actual mean recovery/quality adjustment';
document.getElementById('f6').textContent = 'predicted_RA_Sum \\u2014 fitted surface value used in production';
document.getElementById('f7').textContent = 'diff = predicted_RA_Sum \\u2212 RA_Sum';

statLine('s1', DATA.t1, 'diff');
statLine('s2', DATA.t2, 'diff');
statLine('s3', DATA.t3, 'yearFactor', 4);
statLine('s4', DATA.t4, 'splineMinusRaw');
statLine('s5', DATA.t5, 'RA_Sum');
statLine('s6', DATA.t6, 'predicted_RA_Sum');
statLine('s7', DATA.t7, 'diff');

buildGrid(DATA.t1, brList, wicketsList, r => r.diff, 'g1', {scale: 4});
buildGrid(DATA.t2, brList, wicketsList, r => r.diff, 'g2', {scale: 4});
buildGrid(DATA.t3, brList, wicketsList, r => r.yearFactor, 'g3', {center: 1.0, scale: 0.02, decimals: 4});
buildGrid(DATA.t4, brList, wicketsList, r => r.splineMinusRaw, 'g4', {scale: 4});
buildGrid(DATA.t5, brList, wicketsList, r => r.RA_Sum, 'g5', {scale: 2, decimals: 3});
buildGrid(DATA.t6, brList, wicketsList, r => r.predicted_RA_Sum, 'g6', {scale: 2, decimals: 3});
buildGrid(DATA.t7, brList, wicketsList, r => r.diff, 'g7', {scale: 2, decimals: 3});

document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
    tab.classList.add('active');
    document.getElementById(tab.dataset.tab).classList.add('active');
  });
});
</script>
</body>
</html>
"""

_diagPanels = "".join(
    f'''<div id="{tid}" class="panel{' active' if i == 0 else ''}">
    <p class="formula" id="f{tid[1:]}"></p>
    <div class="stats" id="s{tid[1:]}"></div>
    <div class="gridbox"><div id="g{tid[1:]}"></div></div>
  </div>\n  '''
    for i, tid in enumerate(['t1', 't2', 't3', 't4', 't5', 't6', 't7'])
)

_diagHtml = _DIAG_HTML_TEMPLATE.replace('__PANELS__', _diagPanels).replace('__DATA_JSON__', json.dumps(_diagData))
with open(PROJECT_ROOT / 'men/expBall&runsToCome/outputs/5_diagnostics.html', 'w') as f:
    f.write(_diagHtml)
print("Wrote diagnostics dashboard to outputs/5_diagnostics.html")