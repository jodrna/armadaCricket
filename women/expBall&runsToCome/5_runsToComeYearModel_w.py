import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from paths import PROJECT_ROOT
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# import cleaned ball-by-ball data
trainData = pd.read_csv(PROJECT_ROOT / 'women/expBall&runsToCome/data/dataClean_w.csv', parse_dates=['date'])

# import master lookup table from previous modelling step
masterLookup = pd.read_csv(PROJECT_ROOT / 'women/expBall&runsToCome/outputs/4_masterLookup_w.csv')

# only use first innings data
trainData = trainData.loc[trainData['inningNumber'] == 1].copy()

# when wickets > 7, use adj number
trainData['totalInningRunsToComeAdj'] = np.where(
    trainData['totalInningWickets'] > 7,
    trainData['totalInningRunsToCome'],
    trainData['totalInningRunsToComeAdj'])


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
trainData['daysGroup_daysGroup'] = trainData['daysGroup'] #* trainData['daysGroup']


# features used in the regression models
features = [
    'daysGroup',
    'daysGroup_inningBallNumber',
    'daysGroup_totalInningWickets'
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
# target using just 120br runs
trainData120 = trainData[(trainData['inningBallNumber'] == 1) & (trainData['year'] > 2018)]
X120 = trainData120[['daysGroup']]#, 'daysGroup_daysGroup']]
y_120 = trainData120['vsAdjOvr']


# fit model for adjusted runs-to-come
model_adj = LinearRegression()
model_adj.fit(X, y_adj)

# fit model for raw runs-to-come
model_raw = LinearRegression()
model_raw.fit(X, y_raw)

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
testing_wl = trainData.groupby(['totalInningWickets'])[['yearFactor', 'yearFactor2', 'yearFactor120']].mean().reset_index()

trainData = trainData.merge(testing_wl, on='totalInningWickets', how='left', suffixes=('_old', '_wl'))

trainData['yearFactor'] = trainData['yearFactor_old'] / trainData['yearFactor_wl']
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
testing_br_year = trainData.groupby(['inningBallNumber', 'year'])[['yearFactor', 'yearFactor2', 'totalInningRunsToComeAdj', 'totalInningRunsToCome']].mean().reset_index()
testing_wl_2 = trainData.groupby(['totalInningWickets'])[['yearFactor', 'yearFactor2']].mean().reset_index()
testing_wl_br = trainData.groupby(['totalInningWickets', 'inningBallNumber'])[['yearFactor', 'yearFactor2']].mean().reset_index()
testing_br = trainData.groupby(['inningBallNumber'])[['yearFactor', 'yearFactor2']].mean().reset_index()
testing_RA_sum_br = trainData.groupby(['inningBallNumber'])[['RA_Sum']].mean().reset_index()
testing_RA_sum_wl = trainData.groupby(['totalInningWickets'])['RA_Sum'].mean().reset_index()
RA_sum_wl_br = trainData.groupby(['totalInningWickets', 'inningBallNumber'])['RA_Sum'].mean().reset_index()

# model for RA_sum prediction
testing_RA_sum_wl_br = RA_sum_wl_br.dropna()
model_RA_sum = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('reg',  LinearRegression())
])
model_RA_sum.fit(RA_sum_wl_br[['totalInningWickets', 'inningBallNumber']], RA_sum_wl_br['RA_Sum'])
RA_sum_wl_br['predicted_RA_Sum'] = model_RA_sum.predict(RA_sum_wl_br[['totalInningWickets', 'inningBallNumber']])
trainData['predicted_RA_Sum'] = model_RA_sum.predict(trainData[['totalInningWickets', 'inningBallNumber']])
RA_sum_factoring = trainData.groupby(['inningBallNumber'])['predicted_RA_Sum'].mean().reset_index()
RA_sum_wl_br = RA_sum_wl_br.merge(RA_sum_factoring, on='inningBallNumber', suffixes=('', '_factoring'))
RA_sum_wl_br['predicted_RA_Sum'] = RA_sum_wl_br['predicted_RA_Sum'] - RA_sum_wl_br['predicted_RA_Sum_factoring']
RA_sum_wl_br = RA_sum_wl_br.loc[:, ['totalInningWickets', 'inningBallNumber', 'predicted_RA_Sum']]

# create year grouping used for prediction
masterLookup['daysGroup'] = masterLookup['year'] - 2015

# duplicate the latest year and relabel as 9.4, this gives us the number we want to match the match market
extraRows = masterLookup.loc[masterLookup['daysGroup'] == masterLookup['daysGroup'].max()].copy()
extraRows['daysGroup'] = 10.8

# append future-year rows back onto master lookup
masterLookup = pd.concat([masterLookup, extraRows], ignore_index=True)

# recreate interaction features for prediction
masterLookup['daysGroup_totalInningWickets'] = masterLookup['daysGroup'] * masterLookup['totalInningWickets']
masterLookup['daysGroup_inningBallNumber'] = masterLookup['daysGroup'] * masterLookup['inningBallNumber']

# prediction feature matrix
X_master = masterLookup[features]

# predict year adjustment rates
masterLookup['totalInningRunsToComeSimBiasSplineYearRateAdj'] = (model_adj.predict(X_master))
masterLookup['totalInningRunsToComeSimBiasSplineYearRate'] = (model_raw.predict(X_master))
if log_method == 1:
    masterLookup['totalInningRunsToComeSimBiasSplineYearRateAdj'] = np.expm1(masterLookup['totalInningRunsToComeSimBiasSplineYearRateAdj']) + vsAdjOvrMin
    masterLookup['totalInningRunsToComeSimBiasSplineYearRate'] = np.expm1(masterLookup['totalInningRunsToComeSimBiasSplineYearRate']) + vsOvrMin

# this is to allow for overall bias in the by year adjust model, this means the overall adjust for each wicket will be 1.000
masterLookup = masterLookup.merge(testing_wl, on='totalInningWickets', how='left')
masterLookup['totalInningRunsToComeSimBiasSplineYearRateAdj'] = masterLookup['totalInningRunsToComeSimBiasSplineYearRateAdj'] / masterLookup['yearFactor']
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
# this is to allow for overall bias in the by year adjust model, this means the overall adjust for each wicket will be 1.000
lookupForInruns = lookupForInruns.merge(testing_wl, on='totalInningWickets', how='left')
lookupForInruns['totalInningRunsToComeSimBiasSplineYearAdj3'] = lookupForInruns['totalInningRunsToComeSimBiasSplineYearAdj2'] / lookupForInruns['yearFactor']
lookupForInruns['totalInningRunsToComeSimBiasSplineYear3'] = lookupForInruns['totalInningRunsToComeSimBiasSplineYear2'] / lookupForInruns['yearFactor2']
lookupForInruns['totalInningRunsToComeSimBiasSplineYear1203'] = lookupForInruns['totalInningRunsToComeSimBiasSplineYear1202'] / lookupForInruns['yearFactor120']


lookupForInruns_final = lookupForInruns.loc[:, ['daysGroup', 'totalInningRunsToComeSimBiasSplineYear3', 'totalInningRunsToComeSimBiasSplineYearAdj3', 'totalInningRunsToComeSimBiasSplineYear1203']]

comparison_by_year_final = testing_br_year.copy()