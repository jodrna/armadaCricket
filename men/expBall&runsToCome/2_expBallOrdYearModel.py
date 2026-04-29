import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import log_loss
from paths import PROJECT_ROOT

# import datasets and filter to 1st innings only
trainData = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/data/dataClean.csv', parse_dates=['date'])
trainData = trainData[trainData['inningNumber'] == 1]
masterLookup = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/outputs/1_masterLookup.csv')

# do some random cleaning and prep
trainData = trainData[(trainData['noballRuns'] == 0) & (trainData['wideRuns'] == 0)]
trainData = trainData[trainData['ord'] <= (trainData['totalInningWickets'] + 2)]
trainData = trainData[(trainData['ord'] > 0) & (trainData['ord'] < 12) & (trainData['inningNumber'] == 1)]
trainData['ord'] = np.where(trainData['overNumber'] > 2, (np.where(trainData['ord'] == 1, 2, trainData['ord'])), trainData['ord'])


# merge exp values in
trainData = trainData.merge(masterLookup.loc[:, ['totalInningWickets', 'inningBallNumber', 'm_batsmanRunsBall', 'm_isWicketBowlerBall']], how='left', on=['totalInningWickets', 'inningBallNumber'])
# get the ratio of runs to exp runs which is what we will runsModel
trainData['modelToRealRunsBall'] = trainData['batsmanRuns'] / trainData['m_batsmanRunsBall']
trainData['modelToRealWicketBall'] = trainData['isWicketBowler'] / trainData['m_isWicketBowlerBall']







# Define the input and target for order adjustment
X = trainData[['ord', 'totalInningWickets']]
y = trainData['modelToRealRunsBall']
# Create the linear regression runsOrdModel with no intercept
runsOrdModel = HistGradientBoostingRegressor(monotonic_cst=[-1, 1], random_state=42, max_depth=2)
# Fit the runsOrdModel
runsOrdModel.fit(X, y)
trainData['m_modelToRealRunsBall'] = runsOrdModel.predict(X)
trainData['m_modelToRealRunsBall'] = np.where((trainData['ord'] == 1) & (trainData['totalInningWickets'] == 1), 0.97557, trainData['m_modelToRealRunsBall'])
trainData['m_batsmanRunsBallOrd'] = trainData['m_modelToRealRunsBall'] * trainData['m_batsmanRunsBall']


# reset model to real runs ball with the order adjustment made
trainData['modelToRealRunsBall'] = trainData['batsmanRuns'] / trainData['m_batsmanRunsBallOrd']

# Define the input and target for year adjustment
X = trainData[['daysGroup', 'inningBallNumber']]
y = trainData['modelToRealRunsBall']
# Create the linear regression runsYearModel with no intercept
runsYearModel = HistGradientBoostingRegressor(monotonic_cst=[1, 0], random_state=42, max_depth=2)
# Fit the runsYearModel
runsYearModel.fit(X, y)
trainData['m_modelToRealRunsBall'] = runsYearModel.predict(X)
trainData['m_batsmanRunsBallOrdYear'] = trainData['m_modelToRealRunsBall'] * trainData['m_batsmanRunsBallOrd']
# in over 1, balls 2-5 the model overrates runs from number 2 and underrates from number 1
trainData['m_batsmanRunsBallOrdYear'] = np.where((trainData['ord'] == 1) & (np.isin(trainData['inningBallNumber'], [2, 3, 4, 5])),
                                                 trainData['m_batsmanRunsBallOrdYear'] / 0.927, trainData['m_batsmanRunsBallOrdYear'])
trainData['m_batsmanRunsBallOrdYear'] = np.where((trainData['ord'] == 2) & (np.isin(trainData['inningBallNumber'], [2, 3, 4, 5])),
                                                 trainData['m_batsmanRunsBallOrdYear'] / 1.11, trainData['m_batsmanRunsBallOrdYear'])


# check biases and performance
runBiases = pd.pivot_table(trainData, index=['inningBallNumber', 'ord'], values=['m_batsmanRunsBallOrdYear', 'batsmanRuns'], aggfunc=['sum', 'count']).reset_index()
# check errors
mse = mean_squared_error(trainData['m_batsmanRunsBall'], trainData['batsmanRuns'])
mseOrd = mean_squared_error(trainData['m_batsmanRunsBallOrd'], trainData['batsmanRuns'])
mseOrdYear = mean_squared_error(trainData['m_batsmanRunsBallOrdYear'], trainData['batsmanRuns'])






# now wickets
X = trainData[['daysGroup', 'inningBallNumber', 'ord', 'totalInningWickets']]
y = trainData['modelToRealWicketBall']
# Create the HistGradientBoostingRegressor model with no intercept and specific parameters
wicketsOrdYearModel = HistGradientBoostingRegressor(monotonic_cst=[0, 0, 0, 0], random_state=42, max_depth=2)
# Use cross_val_predict to perform 5-fold cross-validation and generate out-of-sample predictions
trainData['m_modelToRealWicketBall'] = cross_val_predict(wicketsOrdYearModel, X, y, cv=10)
trainData['m_isWicketBowlerBallOrdYear'] = trainData['m_modelToRealWicketBall'] * trainData['m_isWicketBowlerBall']

# check biases and performance
wicketBiases = pd.pivot_table(trainData, index=['ord', 'overNumber'], values=['m_isWicketBowlerBallOrdYear', 'isWicketBowler'], aggfunc=['sum', 'count']).reset_index()
# check losses
logLossStatic = log_loss(trainData['isWicketBowler'], np.full(len(trainData), 0.0540025)) * 1000
logLoss = log_loss(trainData['isWicketBowler'], trainData['m_isWicketBowlerBall']) * 1000
logLossOrdYear = log_loss(trainData['isWicketBowler'], trainData['m_isWicketBowlerBallOrdYear']) * 1000







# create a list of totalinningwickets 0-9, each value has order, then merge to master which will duplicate rows for each order
rows = []
# Populate the rows based on the specified rules
for i in range(10):
    for j in range(1, i + 4):
        rows.append({'totalInningWickets': i, 'ord': j})

# Create the DataFrame from the list of rows
df = pd.DataFrame(rows)
df = df[df['ord'] < 12]

# by merging on just wickets we dduplicate the rows for order
masterLookup = masterLookup.merge(df, how='left', on=['totalInningWickets'])

# Duplicate each row 12 times, for each year
masterLookup = pd.DataFrame(np.repeat(masterLookup.values, 12, axis=0), columns=masterLookup.columns)
# Add the ID column ranging from 0 to 9 for each group of repeated rows
masterLookup['daysGroup'] = np.tile(np.arange(12), 8880)





# add the adjusted order number
X = masterLookup[['ord', 'totalInningWickets']]
masterLookup['runsOrdRate'] = runsOrdModel.predict(X)
masterLookup['m_batsmanRunsBallOrd'] = masterLookup['m_batsmanRunsBall'] * masterLookup['runsOrdRate']

# add the runs adjusted for year
X = masterLookup[['daysGroup', 'inningBallNumber']]
masterLookup['runsYearRate'] = runsYearModel.predict(X)
masterLookup['m_batsmanRunsBallOrdYear'] = masterLookup['m_batsmanRunsBallOrd'] * masterLookup['runsYearRate']



# now wickets to the masterlookup
X = trainData[['daysGroup', 'inningBallNumber', 'ord', 'totalInningWickets']]
y = trainData['modelToRealWicketBall']
# fit the model with no cv
wicketsOrdYearModel.fit(X, y)
X = masterLookup[['daysGroup', 'inningBallNumber', 'ord', 'totalInningWickets']]
masterLookup['wicketsOrdYearRate'] = wicketsOrdYearModel.predict(X)
masterLookup['m_isWicketBowlerBallOrdYear'] = masterLookup['wicketsOrdYearRate'] * masterLookup['m_isWicketBowlerBall']


# just rename daysgroup as year
masterLookup = masterLookup.rename(columns={'daysGroup': 'year'})
masterLookup['year'] = masterLookup['year'] + 2015

# export
masterLookup.to_csv(PROJECT_ROOT / 'men/expBall&runsToCome/outputs/2_masterLookup.csv', index=False)

