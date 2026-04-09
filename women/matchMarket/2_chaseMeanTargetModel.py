import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
import sklearn.utils
from paths import PROJECT_ROOT

# import dataframes
trainData = pd.read_csv(PROJECT_ROOT / 'women/data/dataClean.csv', parse_dates=['date'])
chaseLookup = pd.read_csv(PROJECT_ROOT / 'women/matchMarket/1_chaseLookup.csv')
chaseLookup = chaseLookup.reset_index(drop=True)

# only look at second innings data
trainData = trainData[trainData['inningNumber'] == 2]

# this model is going to predict the runs scored so far given the situation with wickets lost, balls faced and required, once we have runs scored we add remaining required to get the original target
trainData['runs_per_ball'] = trainData['totalInningRuns'] / trainData['inningBallNumber']
trainData['runs_per_ball'] = np.where(trainData['inningBallNumber'] == 0, 0, trainData['runs_per_ball'])      # this will be infinity otherwise and confuse the model

# shuffle the data and then model using a neural network
trainData = sklearn.utils.shuffle(trainData, random_state=42)       # randomise the order of the data
X = trainData.loc[:, ['totalInningWickets', 'inningBallNumber', 'runsRequired']]       # use wickets, balls faced and required as the x variables
y = trainData.loc[:, ['runs_per_ball']].values.ravel()
model = MLPRegressor(random_state=42, hidden_layer_sizes=(16, 8))       # the model is a regression neural network

model.fit(X, y)

# we want to predict the runs per ball in the innings so far at each point given wickets, balls bowled and required runs, produce dataframe with raw values
chaseLookup['pred_target'] = model.predict(chaseLookup.loc[:, ['totalInningWickets', 'inningBallNumber', 'runsRequired']])      # predict using the variables in the data pivot
chaseLookup['pred_target'] = np.where(chaseLookup['inningBallsRemaining'] == 0, 0, chaseLookup['pred_target'])   # where balls faced is 0 runs scored so far is also 0, state this incase the model predicts something like 0.1
chaseLookup['pred_target'] = (chaseLookup['pred_target'] * chaseLookup['inningBallNumber']) + chaseLookup['runsRequired']   # takes runs scored and add required to get original target
chaseLookup = chaseLookup.drop(labels=['inningBallNumber'], axis=1)    # remove balls faced from the data pivot because it is no longer needed



# # order correctly for illogical situations
# cols = chaseLookup.loc[:, ['totalInningWickets', 'runsRequired', 'inningBallsRemaining', 'pred_target']]
# colsWrong = cols.sort_values(by=['totalInningWickets', 'runsRequired', 'inningBallsRemaining'], axis=0).reset_index(drop=True)
# colsRight = cols.sort_values(by=['totalInningWickets', 'runsRequired', 'pred_target'], axis=0).reset_index(drop=True)
# colsWrong['pred_target'] = colsRight['pred_target']
# colsWrong = colsWrong.sort_values(by=['inningBallsRemaining', 'runsRequired', 'totalInningWickets'], axis=0).reset_index(drop=True)
# chaseLookup['pred_target'] = colsWrong['pred_target']



# export
chaseLookup.to_csv(PROJECT_ROOT / 'women/matchMarket/2_chaseLookup.csv', index=False)


# chaseLookupMen = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/2_chaseLookup.csv')
