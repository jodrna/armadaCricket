import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from paths import PROJECT_ROOT


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

# only train on 2018+ data
trainData = trainData.loc[trainData['year'] > 2018].copy()

# create interaction terms between year trend and game state
trainData['daysGroup_totalInningWickets'] = trainData['daysGroup'] * trainData['totalInningWickets']
trainData['daysGroup_inningBallNumber'] = trainData['daysGroup'] * trainData['inningBallNumber']


# features used in the regression models
features = [
    'daysGroup',
    'daysGroup_inningBallNumber',
    'daysGroup_totalInningWickets'
]

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

# apply year adjustments back onto baseline spline predictions
trainData['totalInningRunsToComeSimBiasSplineYearAdj'] = trainData['totalInningRunsToComeSimBiasSpline'] * trainData['yearFactor']
trainData['totalInningRunsToComeSimBiasSplineYear'] = trainData['totalInningRunsToComeSimBiasSpline'] * trainData['yearFactor2']

# remove rows with nan predictions
trainData = trainData.dropna(subset=['totalInningRunsToComeSimBiasSplineYearAdj', 'totalInningRunsToComeSimBiasSplineYear'])

# print training MAE for adjusted and raw models
print(mean_absolute_error(trainData['totalInningRunsToCome'], trainData['totalInningRunsToComeSimBiasSplineYearAdj']))
print(mean_absolute_error(trainData['totalInningRunsToCome'], trainData['totalInningRunsToComeSimBiasSplineYear']))


# create year grouping used for prediction
masterLookup['daysGroup'] = masterLookup['year'] - 2015

# duplicate the latest year and relabel as 9.4
# this is used as a manually adjusted future-year estimate
extraRows = masterLookup.loc[masterLookup['daysGroup'] == 10].copy()
extraRows['daysGroup'] = 13
# append future-year rows back onto master lookup
masterLookup = pd.concat([masterLookup, extraRows], ignore_index=True)
# this is used as a manually adjusted future-year estimate
extraRows = masterLookup.loc[masterLookup['daysGroup'] == 13].copy()
extraRows['daysGroup'] = 15.87
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

# apply predicted year factors to baseline spline values
masterLookup['totalInningRunsToComeSimBiasSplineYearAdj'] = masterLookup['totalInningRunsToComeSimBiasSplineYearRateAdj'] * masterLookup['totalInningRunsToComeSimBiasSpline']
masterLookup['totalInningRunsToComeSimBiasSplineYear'] = masterLookup['totalInningRunsToComeSimBiasSplineYearRate'] * masterLookup['totalInningRunsToComeSimBiasSpline']
masterLookup = masterLookup.sort_values(by=['totalInningWickets', 'inningBallNumber', 'ord', 'daysGroup']).reset_index(drop=True)

# export final lookup table
masterLookup.to_csv(PROJECT_ROOT / 'women/expBall&runsToCome/outputs/5_masterLookup_w.csv', index=False)



