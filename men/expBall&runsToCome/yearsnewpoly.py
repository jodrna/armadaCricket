import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from pygam import LinearGAM
from pygam import s
from pygam import l
from pygam import f
from paths import PROJECT_ROOT

# import data and the runs to come modelled numbers
trainData = pd.read_csv(PROJECT_ROOT / 'men/data/dataClean.csv', parse_dates=['date'])
trainData = trainData[trainData['inningNumber'] == 1]
masterLookup = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/4_masterLookup.csv')



# masterlookup 1 year only
masterLookupSingle = masterLookup.copy()
masterLookupSingle = masterLookupSingle.drop_duplicates(subset=['totalInningWickets', 'inningBallNumber']).reset_index(drop=True)
trainData = trainData.merge(masterLookupSingle.loc[:, ['totalInningWickets', 'inningBallNumber', 'totalInningRunsToComeSimBiasSpline']], how='left', on=['totalInningWickets', 'inningBallNumber'])
overall_avg_runs_test1 = trainData.groupby(['totalInningWickets', 'inningBallNumber'])[['totalInningRunsToComeAdj', 'totalInningRunsToCome', 'totalInningRunsToComeSimBiasSpline']].mean().reset_index()

# runs to come adjusted by
runsToComeOvr = pd.pivot_table(trainData, values=['totalInningRunsToComeAdj', 'totalInningRunsToCome'], index=['totalInningWickets', 'inningBallNumber'], aggfunc='mean').reset_index().rename(
    columns={'totalInningRunsToComeAdj': 'totalInningRunsToComeAdjOvr', 'totalInningRunsToCome': 'totalInningRunsToComeOvr'})

# get runs to come by year for reference
runsToComeYear = pd.pivot_table(trainData, values=['totalInningRunsToComeAdj', 'totalInningRunsToCome'], index=['year', 'totalInningWickets', 'inningBallNumber'], aggfunc='mean').reset_index()

# vs Overall adjusted number
trainData = trainData.merge(runsToComeOvr, how='left', on=['totalInningWickets', 'inningBallNumber'])
trainData['vsAdjOvr'] = trainData['totalInningRunsToComeAdj'] / trainData['totalInningRunsToComeSimBiasSpline']  # trainData['totalInningRunsToComeAdjOvr']
trainData['vsOvr'] = trainData['totalInningRunsToCome'] / trainData['totalInningRunsToComeSimBiasSpline']  # trainData['totalInningRunsToComeOvr']
trainData = trainData.dropna(subset=['vsAdjOvr'])
trainData = trainData.dropna(subset=['vsOvr'])

trainData = trainData[(trainData.vsAdjOvr > 0) & (trainData.vsOvr > 0)]
trainData = trainData[(trainData.year > 2018)]

###1111111 original model
# # Features and target variable
# X = trainData[['daysGroup', 'totalInningWickets', 'inningBallNumber']]
# y = trainData['vsAdjOvr']
# y2 = trainData['vsOvr']
#
# # Initialize and train the model
# model = LinearRegression()
# model.fit(X, y)
# model2 = LinearRegression()
# model2.fit(X, y2)
#
# # Make predictions
# trainData['yearFactor'] = model.predict(X)
# trainData['yearFactor2'] = model2.predict(X)


##222222222222GAM model
X = trainData[['daysGroup', 'totalInningWickets', 'inningBallNumber']].values

# s() = spline for continuous, f() = factor for categorical/discrete
gam = LinearGAM(l(0) + f(1) + s(2))
gam.fit(X, trainData['vsAdjOvr'])

gam2 = LinearGAM(l(0) + f(1) + s(2))
gam2.fit(X, trainData['vsOvr'])

trainData['yearFactor'] = gam.predict(X)
trainData['yearFactor2'] = gam2.predict(X)

# ###333333333333log simple model
# X = trainData[['daysGroup', 'totalInningWickets', 'inningBallNumber']]
#
# # --- vsAdjOvr ---
# overall_avg = trainData['vsAdjOvr'].mean()
# trainData['logRatio'] = np.log(trainData['vsAdjOvr'] / overall_avg)
#
# model = LinearRegression()
# model.fit(X, trainData['logRatio'])
#
# predicted_log = model.predict(X)
# predicted_log -= predicted_log.mean()
# trainData['yearFactor'] = np.exp(predicted_log) * overall_avg
#
# # --- vsOvr ---
# overall_avg2 = trainData['vsOvr'].mean()
# trainData['logRatio2'] = np.log(trainData['vsOvr'] / overall_avg2)
#
# model2 = LinearRegression()
# model2.fit(X, trainData['logRatio2'])
#
# predicted_log2 = model2.predict(X)
# predicted_log2 -= predicted_log2.mean()
# trainData['yearFactor2'] = np.exp(predicted_log2) * overall_avg2


# ###44444444GAM log model:
# X = trainData[['daysGroup', 'totalInningWickets', 'inningBallNumber']].values
#
# # --- vsAdjOvr ---
# overall_avg = trainData['vsAdjOvr'].mean()
# trainData['logRatio'] = np.log(trainData['vsAdjOvr'] / overall_avg)
#
# gam = LinearGAM(l(0) + f(1) + s(2))
# gam.fit(X, trainData['logRatio'])
#
# predicted_log = gam.predict(X)
# predicted_log -= predicted_log.mean()
# trainData['yearFactor'] = np.exp(predicted_log) * overall_avg
#
# # --- vsOvr ---
# overall_avg2 = trainData['vsOvr'].mean()
# trainData['logRatio2'] = np.log(trainData['vsOvr'] / overall_avg2)
#
# gam2 = LinearGAM(l(0) + f(1) + s(2))
# gam2.fit(X, trainData['logRatio2'])
#
# predicted_log2 = gam2.predict(X)
# predicted_log2 -= predicted_log2.mean()
# trainData['yearFactor2'] = np.exp(predicted_log2) * overall_avg2


# merge the runs to come numbers overall then multiply by the year factor
trainData['totalInningRunsToComeSimBiasSplineYearAdj'] = trainData['totalInningRunsToComeSimBiasSpline'] * trainData['yearFactor']
trainData['totalInningRunsToComeSimBiasSplineYear'] = trainData['totalInningRunsToComeSimBiasSpline'] * trainData['yearFactor2']
trainData = trainData.dropna(axis=0, subset=['totalInningRunsToComeSimBiasSplineYearAdj'])
trainData = trainData.dropna(axis=0, subset=['totalInningRunsToComeSimBiasSplineYear'])

# check the training error
print(mean_absolute_error(trainData['totalInningRunsToCome'], trainData['totalInningRunsToComeSimBiasSplineYearAdj']))

overall_avg_runs_test = trainData.groupby(['totalInningWickets', 'inningBallNumber'])[
    ['totalInningRunsToComeSimBiasSplineYearAdj', 'totalInningRunsToComeAdj', 'totalInningRunsToComeSimBiasSplineYear', 'totalInningRunsToCome', 'totalInningRunsToComeSimBiasSpline']].mean().reset_index()

# predict for all years
masterLookup['daysGroup'] = masterLookup['year'] - 2015
# Create empty list to collect rows
rows = []

# Loop through each row in masterLookup
for _, row in masterLookup.iterrows():
    rows.append(row)
    if row['daysGroup'] == 11:
        dup = row.copy()
        dup['daysGroup'] = 9.6
        rows.append(dup)

# Rebuild the DataFrame
masterLookup = pd.DataFrame(rows)
X = masterLookup[['daysGroup', 'totalInningWickets', 'inningBallNumber']]
masterLookup['totalInningRunsToComeSimBiasSplineYearRateAdj'] = gam.predict(X)
masterLookup['totalInningRunsToComeSimBiasSplineYearRate'] = gam2.predict(X)
masterLookup['totalInningRunsToComeSimBiasSplineYearAdj'] = masterLookup['totalInningRunsToComeSimBiasSplineYearRateAdj'] * masterLookup['totalInningRunsToComeSimBiasSpline']
masterLookup['totalInningRunsToComeSimBiasSplineYear'] = masterLookup['totalInningRunsToComeSimBiasSplineYearRate'] * masterLookup['totalInningRunsToComeSimBiasSpline']

# get runs to come by year for reference
runsToComeYear = pd.pivot_table(trainData, values=['totalInningRunsToComeAdj', 'totalInningRunsToCome', 'totalInningRunsToComeSimBiasSplineYear', 'totalInningRunsToComeSimBiasSplineYearAdj'],
                                index=['year', 'totalInningWickets', 'inningBallNumber'], aggfunc='mean').reset_index().sort_values(by=['inningBallNumber', 'totalInningWickets', 'year'],
                                                                                                                                    ascending=[True, True, True])

# export
masterLookup.to_csv(PROJECT_ROOT / 'men/expBall&runsToCome/5_masterLookup.csv', index=False)
# trainData.to_csv(PROJECT_ROOT / 'men/expBall&runsToCome/5_trainData.csv', index=False)



# only first ball of the innings
plotData = runsToComeYear[runsToComeYear['inningBallNumber'] == 1]

# aggregate by year
plotData = pd.pivot_table(
    plotData,
    values=['totalInningRunsToComeAdj', 'totalInningRunsToCome', 'totalInningRunsToComeSimBiasSplineYear'],
    index=['year'],
    aggfunc='mean'
).reset_index()

# plot
plt.figure()

plt.plot(plotData['year'], plotData['totalInningRunsToComeAdj'], label='RunsToComeAdj')
plt.plot(plotData['year'], plotData['totalInningRunsToCome'], label='RunsToCome')
plt.plot(plotData['year'], plotData['totalInningRunsToComeSimBiasSplineYear'], label='Model')

plt.xlabel('Year')
plt.ylabel('Runs To Come (Ball 1)')
plt.title('Runs To Come Comparison (Ball 1)')
plt.legend()

plt.show()



