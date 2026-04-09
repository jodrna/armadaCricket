import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


# import data and the runs to come modelled numbers
trainData = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/Development/men/data/dataClean.csv', parse_dates=['date'])
trainData = trainData[trainData['inningNumber'] == 2]
masterLookup = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/Development/men/expBall&runsToCome/4_masterLookup.csv')


trainData['totalInningRunsToComeAdj2'] = trainData['totalInningRunsToComeAdj']
# trainData['totalInningRunsToComeAdj'] = trainData['totalInningRunsToCome']


# masterlookup 1 year only
masterLookupSingle = masterLookup.copy()
masterLookupSingle = masterLookupSingle.drop_duplicates(subset=['totalInningWickets', 'inningBallNumber']).reset_index(drop=True)

# runs to come adjusted by
# trainData['totalInningRunsToComeAdj'] = trainData['totalInningRunsToCome'] - trainData['RA_Sum']
runsToComeOvr = pd.pivot_table(trainData, values=['totalInningRunsToComeAdj'], index=['totalInningWickets', 'inningBallNumber'], aggfunc='mean').reset_index().rename(columns={'totalInningRunsToComeAdj': 'totalInningRunsToComeAdjOvr'})


# get runs to come by year for reference
runsToComeYear = pd.pivot_table(trainData, values=['totalInningRunsToComeAdj', 'totalInningRunsToCome'], index=['year', 'totalInningWickets', 'inningBallNumber'], aggfunc='mean').reset_index()


# vs Overall adjusted number
trainData = trainData.merge(runsToComeOvr, how='left', on=['totalInningWickets', 'inningBallNumber'])
trainData['vsAdjOvr'] = trainData['totalInningRunsToComeAdj'] / trainData['totalInningRunsToComeAdjOvr']
trainData = trainData.dropna(subset=['vsAdjOvr'])

# Features and target variable
X = trainData[['daysGroup', 'totalInningWickets', 'inningBallNumber']]
y = trainData['vsAdjOvr']

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
print(mean_absolute_error(trainData['totalInningRunsToCome'], trainData['totalInningRunsToComeSimBiasSplineYear']))



# predict for all years
masterLookup['daysGroup'] = masterLookup['year'] - 2015
# Create empty list to collect rows
rows = []

# Loop through each row in masterLookup
for _, row in masterLookup.iterrows():
    rows.append(row)
    if row['daysGroup'] == 11:
        dup = row.copy()
        dup['daysGroup'] = 11.5
        rows.append(dup)

# Rebuild the DataFrame
masterLookup = pd.DataFrame(rows)
X = masterLookup[['daysGroup', 'totalInningWickets', 'inningBallNumber']]
masterLookup['totalInningRunsToComeSimBiasSplineYearRate'] = model.predict(X)
masterLookup['totalInningRunsToComeSimBiasSplineYear'] = masterLookup['totalInningRunsToComeSimBiasSplineYearRate'] * masterLookup['totalInningRunsToComeSimBiasSpline']

# get runs to come by year for reference
runsToComeYear = pd.pivot_table(trainData, values=['totalInningRunsToComeAdj', 'totalInningRunsToComeAdj2', 'totalInningRunsToCome', 'totalInningRunsToComeSimBiasSplineYear', 'daysGroup'],
                                index=['year', 'totalInningWickets', 'inningBallNumber'], aggfunc='mean').reset_index()


# export
masterLookup.to_csv('/Users/jordan/Documents/ArmadaCricket/Development/men/expBall&runsToCome/5_masterLookup.csv', index=False)
trainData.to_csv('/Users/jordan/Documents/ArmadaCricket/Development/men/expBall&runsToCome/5_trainData.csv', index=False)



# only first ball of the innings
plotData = runsToComeYear[runsToComeYear['inningBallNumber'] == 1]

# aggregate by year
plotData = pd.pivot_table(
    plotData,
    values=['totalInningRunsToComeAdj2', 'totalInningRunsToCome', 'totalInningRunsToComeSimBiasSplineYear'],
    index=['year'],
    aggfunc='mean'
).reset_index()

# plot
plt.figure()

plt.plot(plotData['year'], plotData['totalInningRunsToComeAdj2'], label='RunsToComeAdj')
plt.plot(plotData['year'], plotData['totalInningRunsToCome'], label='RunsToCome')
plt.plot(plotData['year'], plotData['totalInningRunsToComeSimBiasSplineYear'], label='Model')

plt.xlabel('Year')
plt.ylabel('Runs To Come (Ball 1)')
plt.title('Runs To Come Comparison (Ball 1)')
plt.legend()

plt.show()


