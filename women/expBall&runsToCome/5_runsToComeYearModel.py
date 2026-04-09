import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


# import data and the runs to come modelled numbers
trainData = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/Development/women/data/dataClean.csv', parse_dates=['date'])
trainData = trainData[trainData['inningNumber'] == 1]
masterLookup = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/Development/women/expBall&runsToCome/4_masterLookup.csv')

# masterlookup 1 year only
masterLookupSingle = masterLookup.copy()
masterLookupSingle = masterLookupSingle.drop_duplicates(subset=['totalInningWickets', 'inningBallNumber']).reset_index(drop=True)

# use this to see actuals by year etc
# yearnums = pd.pivot_table(trainData, index=['year', 'totalInningWickets', 'inningBallNumber'], aggfunc='mean', values=['totalInningRunsToCome']).reset_index()

# get runs to come by year for reference
runsToComeYear = pd.pivot_table(trainData.drop_duplicates(subset=['matchID']), values=['totalInningRunsEnd'], index=['year'], aggfunc='mean').reset_index()

# runs to come by host for the adjustment
runsToComeHost = pd.pivot_table(trainData, values=['totalInningRunsToCome'], index=['host', 'totalInningWickets', 'inningBallNumber'], aggfunc='mean').reset_index()

# vs HOST Overall
trainData = trainData.merge(runsToComeHost, how='left', on=['host', 'totalInningWickets', 'inningBallNumber'])
trainData['vsHostOverall'] = trainData['totalInningRunsToCome_x'] / trainData['totalInningRunsToCome_y']
trainData = trainData.dropna(subset=['vsHostOverall'])

# Features and target variable
X = trainData[['daysGroup', 'totalInningWickets', 'inningBallNumber']]
y = trainData['vsHostOverall']

# Initialize and train the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
trainData['yearFactor'] = model.predict(X)


# merge the runs to come numbers overall then multiply by the year factor
trainData = trainData.merge(masterLookupSingle.loc[:, ['totalInningWickets', 'inningBallNumber', 'totalInningRunsToComeSimBiasSpline']], how='left', on=['totalInningWickets', 'inningBallNumber'])
trainData['totalInningRunsToComeSimBiasSplineYear'] = trainData['totalInningRunsToComeSimBiasSpline'] * trainData['yearFactor']
trainData = trainData.dropna(axis=0, subset=['totalInningRunsToComeSimBiasSplineYear'])

# check the training error
print(mean_absolute_error(trainData['totalInningRunsToCome_x'], trainData['totalInningRunsToComeSimBiasSplineYear']))



# predict for all years
masterLookup['daysGroup'] = masterLookup['year'] - 2015
out = []
for _, row in masterLookup.iterrows():
    out.append(row)
    if row['daysGroup'] == 10:
        dup = row.copy()
        dup['daysGroup'] = 16
        out.append(dup)

masterLookup = pd.DataFrame(out).reset_index(drop=True)



# # predict for all years in a range
# out = []
# for _, row in masterLookup.iterrows():
#     out.append(row)
#     if 2025 <= row['year'] <= 2031:
#         for val in range(2025, 2032):
#             if val != row['year']:
#                 dup = row.copy()
#                 dup['year'] = val
#                 out.append(dup)
#
# masterLookup = pd.DataFrame(out)
# masterLookup['daysGroup'] = masterLookup['year'] - 2015



# predict for all years
X = masterLookup[['daysGroup', 'totalInningWickets', 'inningBallNumber']]
masterLookup['totalInningRunsToComeSimBiasSplineYearRate'] = model.predict(X)
masterLookup['totalInningRunsToComeSimBiasSplineYear'] = masterLookup['totalInningRunsToComeSimBiasSplineYearRate'] * masterLookup['totalInningRunsToComeSimBiasSpline']



# export
masterLookup.to_csv('/Users/jordan/Documents/ArmadaCricket/Development/women/expBall&runsToCome/5_masterLookup.csv', index=False)



