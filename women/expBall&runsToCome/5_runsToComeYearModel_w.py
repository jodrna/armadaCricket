import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from paths import PROJECT_ROOT
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from datetime import timedelta, date

# import cleaned ball-by-ball data
trainData = pd.read_csv(PROJECT_ROOT / 'women/expBall&runsToCome/data/dataClean_w100.csv', parse_dates=['date'])
# trainData = trainData[(trainData['competition'] != "Women's Big Bash League") | (trainData['date'] < pd.Timestamp(2020, 6, 6))]
# hundred_test = trainData[(trainData['competition'] == "The Hundred (Women's Comp)") & (trainData['inningBallsRemaining'] == 100) & (trainData['inningNumber'] == 1)]
# hundred_test = hundred_test.groupby(['year'])[['totalInningRunsToCome', 'totalInningRunsToComeAdj']].mean().reset_index()
# trainData = trainData[trainData['competition'] != "The Hundred (Women's Comp)"]


# import master lookup table from previous modelling step
masterLookup = pd.read_csv(PROJECT_ROOT / 'women/expBall&runsToCome/outputs/4_masterLookup_w.csv')

# only use first innings data
trainData = trainData.loc[trainData['inningNumber'] == 1].copy()

# totalInningRunsToComeAdj (from dataClean_w100.csv, = totalInningRunsToCome - RA_Sum) is used
# as-is for every wicket count now - no override to raw for wickets > 7, so vsAdjOvr/yearFactor
# are trained on a consistent quality-neutral basis across the whole grid (matches men's fix).


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

# remove rows where ratios could not be calculated
trainData = trainData.dropna(subset=['vsAdjOvr', 'vsOvr'])

# # only train on 2018+ data
# trainData = trainData.loc[trainData['year'] > 2018].copy()

# create interaction terms between year trend and game state
trainData['daysGroup_totalInningWickets'] = trainData['daysGroup'] * trainData['totalInningWickets']
trainData['daysGroup_inningBallNumber'] = trainData['daysGroup'] * trainData['inningBallNumber']
trainData['daysGroup_daysGroup'] = trainData['daysGroup'] * trainData['daysGroup']
trainData['daysGroup_daysGroup_daysGroup'] =  trainData['daysGroup'] * trainData['daysGroup'] * trainData['daysGroup']
trainData['daysGroup_daysGroup_daysGroup_daysGroup'] =  trainData['daysGroup'] * trainData['daysGroup'] * trainData['daysGroup'] * trainData['daysGroup']
trainData['daysGroup_daysGroup_daysGroup_daysGroup_daysGroup'] =  trainData['daysGroup'] * trainData['daysGroup'] * trainData['daysGroup'] * trainData['daysGroup'] * trainData['daysGroup']

# features used in the regression models
features = [
    'daysGroup',
    'daysGroup_inningBallNumber',
    'daysGroup_totalInningWickets'
    , 'daysGroup_daysGroup'
    , 'daysGroup_daysGroup_daysGroup'
    , 'daysGroup_daysGroup_daysGroup_daysGroup'
    # , 'daysGroup_daysGroup_daysGroup_daysGroup_daysGroup'
]

log_method = 1

if log_method == 1:
    vsAdjOvrMin = trainData['vsAdjOvr'].min()
    vsOvrMin = trainData['vsOvr'].min()
    trainData['vsAdjOvr'] = np.log1p(trainData['vsAdjOvr'] - vsAdjOvrMin)
    trainData['vsOvr'] = np.log1p(trainData['vsOvr'] - vsOvrMin)
# feature matrix
X = trainData[features]
##make a new raw train data set without big bash in as BBL needs to be adjusted for surge
trainData_raw = trainData[(trainData['competition'] != "Women's Big Bash League") | (trainData['date'] < pd.Timestamp(2020, 6, 6))]
X_raw = trainData_raw[features]
# target using adjusted runs
y_adj = trainData['vsAdjOvr']
# target using raw runs
y_raw = trainData_raw['vsOvr']
# target using just 120br runs
trainData120 = trainData_raw[(trainData_raw['inningBallNumber'] == 1) & (trainData_raw['year'] > 2018)]
X120 = trainData120[['daysGroup']]#, 'daysGroup_daysGroup']]
y_120 = trainData120['vsAdjOvr']


# fit model for adjusted runs-to-come
model_adj = LinearRegression()
model_adj.fit(X, y_adj)

# degree = 2 # change this to whatever degree you want
#
# model_adj = make_pipeline(PolynomialFeatures(degree), LinearRegression())
# model_adj.fit(X, y_adj)

# fit model for raw runs-to-come
model_raw = LinearRegression()
model_raw.fit(X_raw, y_raw)

# fit model for adjusted 120br runs-to-come
# from sklearn.isotonic import IsotonicRegression
# model_120 = IsotonicRegression(increasing=True)
model_120 = LinearRegression()
model_120.fit(X120, y_120)

# predict year adjustment factors
trainData['yearFactor'] = model_adj.predict(X)
trainData['yearFactor2'] = model_raw.predict(X)
X120_predict = trainData[['daysGroup']]#, 'daysGroup_daysGroup']]
trainData['yearFactor120'] = model_120.predict(X120_predict)
if log_method == 1:
    trainData['yearFactor'] = np.expm1(trainData['yearFactor']) + vsAdjOvrMin
    trainData['yearFactor2'] = np.expm1(trainData['yearFactor2']) + vsOvrMin
    trainData['yearFactor120'] = np.expm1(trainData['yearFactor120']) + vsAdjOvrMin

trainData['yearFactor120'] = np.where(trainData['inningBallNumber'] == 1, trainData['yearFactor120'], np.nan)

###getting remaining trends from model data:
# yearFactor (Adj path) normalizer: built per (wickets, ballNumber) cell, balanced in RUN terms
# (Adj-weighted, not a flat ratio average) and smoothed via a sample-weighted, CV-selected-degree
# polynomial surface - same fix as men's model. A flat per-wicket ratio average of the raw
# prediction treats every ball number as equally important even though real runs-remaining (and
# so how much each row's error actually matters) varies hugely within a wicket group, so ratio
# errors that cancel on average don't cancel in run terms. A flat per-(wicket, ballNumber) mean
# fixes that but is too noisy at sparse cells - hence the same weighted-surface treatment used
# for RA_Sum below.
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


# yearFactor2 (non-Adj path) and yearFactor120 (ball=1-only path) keep the original flat
# per-wicket normalization - not part of this fix
testing_wl = trainData.groupby(['totalInningWickets'])[['yearFactor2', 'yearFactor120']].mean().reset_index()
trainData = trainData.merge(testing_wl, on='totalInningWickets', how='left', suffixes=('_old', '_wl'))

trainData['yearFactor'] = trainData['yearFactor'] / predict_yf_normalizer(trainData['totalInningWickets'], trainData['inningBallNumber'])
trainData['yearFactor2'] = trainData['yearFactor2_old'] / trainData['yearFactor2_wl']
trainData['yearFactor120'] = trainData['yearFactor120_old'] / trainData['yearFactor120_wl']

# apply year adjustments back onto baseline spline predictions
trainData['totalInningRunsToComeSimBiasSplineYearAdj'] = trainData['totalInningRunsToComeSimBiasSpline'] * trainData['yearFactor']
trainData['totalInningRunsToComeSimBiasSplineYear'] = trainData['totalInningRunsToComeSimBiasSpline'] * trainData['yearFactor2']

# remove rows with nan predictions
trainData = trainData.dropna(subset=['totalInningRunsToComeSimBiasSplineYearAdj', 'totalInningRunsToComeSimBiasSplineYear'])

# print training MAE for adjusted and raw models
print(mean_absolute_error(trainData['totalInningRunsToCome'], trainData['totalInningRunsToComeSimBiasSplineYearAdj']))
print(mean_absolute_error(trainData['totalInningRunsToCome'], trainData['totalInningRunsToComeSimBiasSplineYear']))

testing_wl_year = trainData.groupby(['totalInningWickets', 'year'])[['yearFactor', 'yearFactor2']].mean().reset_index()
testing_br_year = trainData.groupby(['inningBallNumber', 'year'])[['totalInningRunsToComeSimBiasSpline', 'yearFactor', 'yearFactor2', 'totalInningRunsToComeAdj', 'totalInningRunsToCome']].mean().reset_index()
testing_wl_2 = trainData.groupby(['totalInningWickets'])[['yearFactor', 'yearFactor2']].mean().reset_index()
testing_wl_br = trainData.groupby(['totalInningWickets', 'inningBallNumber'])[['yearFactor', 'yearFactor2']].mean().reset_index()
testing_br = trainData.groupby(['inningBallNumber'])[['yearFactor', 'yearFactor2']].mean().reset_index()
testing_RA_sum_br = trainData.groupby(['inningBallNumber'])[['RA_Sum']].mean().reset_index()
testing_RA_sum_wl = trainData.groupby(['totalInningWickets'])['RA_Sum'].mean().reset_index()
# No demeaning by inningBallNumber - the raw RA_Sum surface is fit and used as-is, including
# whatever overall bias exists at each ball number (matches men's fix - no reason to force it to
# net to zero per ball).
RA_sum_wl_br = trainData.groupby(['totalInningWickets', 'inningBallNumber']).agg(
    RA_Sum=('RA_Sum', 'mean'), sample=('RA_Sum', 'size')).reset_index()

# ============================================================================================
# model for RA_sum prediction - same methodology as men's model: an unweighted, unfiltered
# polynomial fit across the WHOLE (wickets, ballNumber) grid lets near-empty cells distort the
# surface just as much as cells with thousands of samples - especially relevant here given
# women's data has fewer samples per cell than men's (hence step 6's own extra wickets-smoothing
# pass downstream). Fix, in the same three pieces:
#   1. MIN_CELL_SAMPLE filter - only fit on cells with real sample support
#   2. sample-weighted regression - a large-sample cell should outweigh a tiny one
#   3. cross-validated degree selection, capped at 2
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
MAX_DEGREE_RA_SUM = 2
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

# ============================================================================================
# EXTRAPOLATION SAFETY: per wickets row, the trusted cells (sample >= MIN_CELL_SAMPLE_RA_SUM)
# form one contiguous run of ballNumbers - keep the model's own smoothed prediction inside that
# trusted range; outside it, hold the value at the nearest trusted-range boundary instead of
# letting the polynomial keep extrapolating further (matches men's fix).
# ============================================================================================
trustedByRow = RA_sum_wl_br[RA_sum_wl_br['sample'] >= MIN_CELL_SAMPLE_RA_SUM].groupby('totalInningWickets')['inningBallNumber'].agg(['min', 'max'])


def clamp_to_trusted_range(row):
    if row['totalInningWickets'] not in trustedByRow.index:
        return row['predicted_RA_Sum']
    lo, hi = trustedByRow.loc[row['totalInningWickets'], ['min', 'max']]
    b = np.clip(row['inningBallNumber'], lo, hi)
    return predict_RA_sum([row['totalInningWickets']], [b])[0]


RA_sum_wl_br['predicted_RA_Sum'] = RA_sum_wl_br.apply(clamp_to_trusted_range, axis=1)
RA_sum_wl_br = RA_sum_wl_br.loc[:, ['totalInningWickets', 'inningBallNumber', 'predicted_RA_Sum']]

# create year grouping used for prediction
masterLookup['daysGroup'] = masterLookup['year'] - 2015

# duplicate the latest year and relabel as 9.4, this gives us the number we want to match the match market
extraRows = masterLookup.loc[masterLookup['daysGroup'] == masterLookup['daysGroup'].max()].copy()
extraRows['daysGroup'] = 11.6

# append future-year rows back onto master lookup
masterLookup = pd.concat([masterLookup, extraRows], ignore_index=True)

# recreate interaction features for prediction
masterLookup['daysGroup_totalInningWickets'] = masterLookup['daysGroup'] * masterLookup['totalInningWickets']
masterLookup['daysGroup_inningBallNumber'] = masterLookup['daysGroup'] * masterLookup['inningBallNumber']
masterLookup['daysGroup_daysGroup'] = masterLookup['daysGroup'] * masterLookup['daysGroup']
masterLookup['daysGroup_daysGroup_daysGroup'] =  masterLookup['daysGroup'] * masterLookup['daysGroup'] * masterLookup['daysGroup']
masterLookup['daysGroup_daysGroup_daysGroup_daysGroup'] =  masterLookup['daysGroup'] * masterLookup['daysGroup'] * masterLookup['daysGroup'] * masterLookup['daysGroup']
masterLookup['daysGroup_daysGroup_daysGroup_daysGroup_daysGroup'] =  masterLookup['daysGroup'] * masterLookup['daysGroup'] * masterLookup['daysGroup'] * masterLookup['daysGroup'] * masterLookup['daysGroup']

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

# apply predicted year factors to baseline spline values
masterLookup['totalInningRunsToComeSimBiasSplineYearAdj'] = masterLookup['totalInningRunsToComeSimBiasSplineYearRateAdj'] * masterLookup['totalInningRunsToComeSimBiasSpline']
masterLookup['totalInningRunsToComeSimBiasSplineYear'] = masterLookup['totalInningRunsToComeSimBiasSplineYearRate'] * masterLookup['totalInningRunsToComeSimBiasSpline']
masterLookup = masterLookup.sort_values(by=['totalInningWickets', 'inningBallNumber', 'ord', 'daysGroup']).reset_index(drop=True)

# apply predicted RA_sum as recovery work:
masterLookup = masterLookup.merge(RA_sum_wl_br, on=('totalInningWickets', 'inningBallNumber'), how='left')
masterLookup['totalInningRunsToComeSimBiasSplineYearAdj'] = masterLookup['totalInningRunsToComeSimBiasSplineYearAdj'] - masterLookup['predicted_RA_Sum']

# #export final lookup table
masterLookup.to_csv(PROJECT_ROOT / 'women/expBall&runsToCome/outputs/5_masterLookup_w.csv', index=False)

##below is for making an output of the values each daysGroup will give
lookupForInruns = pd.DataFrame({'daysGroup': np.arange(5, 20.1, 0.1)})
# create interaction terms between year trend and game state
lookupForInruns['totalInningWickets'] = 0
lookupForInruns['inningBallNumber'] = 1
lookupForInruns['daysGroup_totalInningWickets'] = lookupForInruns['daysGroup'] * lookupForInruns['totalInningWickets']
lookupForInruns['daysGroup_inningBallNumber'] = lookupForInruns['daysGroup'] * lookupForInruns['inningBallNumber']
lookupForInruns['daysGroup_daysGroup'] = lookupForInruns['daysGroup'] * lookupForInruns['daysGroup']
lookupForInruns['daysGroup_daysGroup_daysGroup'] = lookupForInruns['daysGroup'] * lookupForInruns['daysGroup'] * lookupForInruns['daysGroup']
lookupForInruns['daysGroup_daysGroup_daysGroup_daysGroup'] = lookupForInruns['daysGroup'] * lookupForInruns['daysGroup'] * lookupForInruns['daysGroup'] * lookupForInruns['daysGroup']
lookupForInruns['daysGroup_daysGroup_daysGroup_daysGroup_daysGroup'] = lookupForInruns['daysGroup'] * lookupForInruns['daysGroup'] * lookupForInruns['daysGroup'] * lookupForInruns['daysGroup'] * lookupForInruns['daysGroup']

lookupForInruns = lookupForInruns.merge(masterLookup[(masterLookup.totalInningWickets == 0) & (masterLookup.inningBallNumber == 1)].groupby(['inningBallNumber', 'totalInningWickets'])[['totalInningRunsToComeSimBiasSpline', 'predicted_RA_Sum']].mean().reset_index(), on=('totalInningWickets', 'inningBallNumber'), how='left')
# prediction feature matrix
X_lookup = lookupForInruns[features]
X_lookup_120 = lookupForInruns[['daysGroup']]#, 'daysGroup_daysGroup']]
lookupForInruns['totalInningRunsToComeSimBiasSplineYearRateAdj'] = (model_adj.predict(X_lookup))
lookupForInruns['totalInningRunsToComeSimBiasSplineYearRate'] = (model_raw.predict(X_lookup))
lookupForInruns['totalInningRunsToComeSimBiasSplineYearRate120'] = (model_120.predict(X_lookup_120))
if log_method == 1:
    lookupForInruns['totalInningRunsToComeSimBiasSplineYearRateAdj'] = np.expm1(lookupForInruns['totalInningRunsToComeSimBiasSplineYearRateAdj']) + vsAdjOvrMin
    lookupForInruns['totalInningRunsToComeSimBiasSplineYearRate'] = np.expm1(lookupForInruns['totalInningRunsToComeSimBiasSplineYearRate']) + vsOvrMin
    lookupForInruns['totalInningRunsToComeSimBiasSplineYearRate120'] = np.expm1(lookupForInruns['totalInningRunsToComeSimBiasSplineYearRate120']) + vsAdjOvrMin
lookupForInruns['totalInningRunsToComeSimBiasSplineYearAdj2'] = (lookupForInruns['totalInningRunsToComeSimBiasSplineYearRateAdj'] * lookupForInruns['totalInningRunsToComeSimBiasSpline']) - lookupForInruns['predicted_RA_Sum']
lookupForInruns['totalInningRunsToComeSimBiasSplineYear2'] = lookupForInruns['totalInningRunsToComeSimBiasSplineYearRate'] * lookupForInruns['totalInningRunsToComeSimBiasSpline']
lookupForInruns['totalInningRunsToComeSimBiasSplineYear1202'] = (lookupForInruns['totalInningRunsToComeSimBiasSplineYearRate120'] * lookupForInruns['totalInningRunsToComeSimBiasSpline']) - lookupForInruns['predicted_RA_Sum']
# this is to allow for overall bias in the by year adjust model - lookupForInruns is always at
# wickets=0/ball=1, so the surface normalizer is evaluated directly at that point
lookupForInruns = lookupForInruns.merge(testing_wl, on='totalInningWickets', how='left')
lookupForInruns['totalInningRunsToComeSimBiasSplineYearAdj3'] = lookupForInruns['totalInningRunsToComeSimBiasSplineYearAdj2'] / predict_yf_normalizer(lookupForInruns['totalInningWickets'], lookupForInruns['inningBallNumber'])
lookupForInruns['totalInningRunsToComeSimBiasSplineYear3'] = lookupForInruns['totalInningRunsToComeSimBiasSplineYear2'] / lookupForInruns['yearFactor2']
lookupForInruns['totalInningRunsToComeSimBiasSplineYear1203'] = lookupForInruns['totalInningRunsToComeSimBiasSplineYear1202'] / lookupForInruns['yearFactor120']


lookupForInruns_final = lookupForInruns.loc[:, ['daysGroup', 'totalInningRunsToComeSimBiasSplineYear3', 'totalInningRunsToComeSimBiasSplineYearAdj3', 'totalInningRunsToComeSimBiasSplineYear1203']]

testing_br_year['pred_runsadj'], testing_br_year['pred_runs'] = testing_br_year['totalInningRunsToComeSimBiasSpline'] * testing_br_year['yearFactor'], testing_br_year['totalInningRunsToComeSimBiasSpline'] * testing_br_year['yearFactor2']
comparison_by_year_final = testing_br_year.copy()
comparison_by_year_final = comparison_by_year_final.loc[:,['inningBallNumber', 'totalInningRunsToComeAdj', 'pred_runsadj', 'totalInningRunsToCome', 'pred_runs']]