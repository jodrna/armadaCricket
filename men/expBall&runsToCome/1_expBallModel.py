import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from paths import PROJECT_ROOT


# import and filter to 1st innings only
trainData = pd.read_csv(PROJECT_ROOT / 'men/data/dataClean.csv', parse_dates=['date'])
trainData = trainData[trainData['inningNumber'] == 1]

# extras averages, must be done here at the start before we remove these for modelling ball by ball
extras = pd.pivot_table(trainData, values=['wideRuns', 'noballRuns', 'isWide', 'isNoball', 'byeRuns'], index=['overNumber'], aggfunc=['sum', 'mean']).reset_index()
extras.columns = ['_'.join([str(elem) for elem in col]).strip() for col in extras.columns.values]
extras['isInvalid'] = extras['mean_isWide'] + extras['mean_isNoball']
extras['avgRunsInvalid'] = (extras['sum_noballRuns'] + extras['sum_wideRuns']) / (extras['sum_isNoball'] + extras['sum_isWide'])
extras = extras.rename(columns={'overNumber_': 'overNumber'})

# combined prob of no ball and wide given over number + avg runs from that, add on avg byeRuns regardless, then prob of wicket if a wide/noball
avgWicketsInvalid = pd.pivot_table(trainData[(trainData['noballRuns'] > 0) | (trainData['wideRuns'] > 0)], values=['isWicket'], index=['overNumber'], aggfunc='mean').reset_index()
avgWicketsRunOut = pd.pivot_table(trainData[(trainData['noballRuns'] == 0) | (trainData['wideRuns'] == 0)], values=['isWicketRunOut'], index=['overNumber'], aggfunc='mean').reset_index()
extras = extras.merge(avgWicketsInvalid, how='left', on=['overNumber'])
extras = extras.merge(avgWicketsRunOut, how='left', on=['overNumber'])
extras = extras.rename(columns={'isInvalid': 'isInvalidOver', 'avgRunsInvalid': 'invalidRunsOver', 'isWicket': 'isWicketInvalidOver', 'mean_byeRuns': 'byeRunsOver', 'isWicketRunOut': 'isWicketRunOutOver'})


# we only model VALID balls so remove wides and noballs
trainData = trainData[(trainData['noballRuns'] == 0) & (trainData['wideRuns'] == 0)]

# Define the range of wickets lost and ball numbers, this will be the beginning of our predict dataframe
wicketsLost = range(0, 10)
ballNumbers = range(1, 121)  # 1 to 120 inclusive
# Create a list to hold tuples of (wicketsLost, ball_number, over_number)
combinations = []
for wickets in wicketsLost:
    for ball in ballNumbers:
        over = np.ceil(ball / 6)   # Calculate over number (1-based index)
        combinations.append((wickets, ball, int(over)))
# Create DataFrame from the list of tuples
masterLookup = pd.DataFrame(combinations, columns=['totalInningWickets', 'inningBallNumber', 'overNumber'])
masterLookup['isPowerplay'] = np.where(masterLookup['inningBallNumber'] <= 36, 1, 0)
masterLookup['inningNumber'] = 1


# sample sizes
sampleSizes = pd.pivot_table(trainData, index=['inningBallNumber', 'totalInningWickets'], values=['sample'], aggfunc='sum').reset_index()
masterLookup = masterLookup.merge(sampleSizes, how='left', on=['inningBallNumber', 'totalInningWickets'])

# work out actuals by the ball number
ballValues = pd.pivot_table(trainData, index=['inningBallNumber', 'totalInningWickets'], values=['batsmanRuns', 'isWicketBowler'], aggfunc='mean').reset_index()
ballValues = ballValues.rename(columns={'batsmanRuns': 'batsmanRunsBall', 'isWicketBowler': 'isWicketBowlerBall'})
# merge actuals into predict dataframe
masterLookup = masterLookup.merge(ballValues.loc[:, ['inningBallNumber', 'batsmanRunsBall', 'totalInningWickets', 'isWicketBowlerBall']], how='left', on=['inningBallNumber', 'totalInningWickets'])
# work out actuals by the over
ballValues = pd.pivot_table(trainData, index=['overNumber', 'totalInningWickets'], values=['batsmanRuns', 'isWicketBowler'], aggfunc='mean').reset_index()
ballValues = ballValues.rename(columns={'batsmanRuns': 'batsmanRunsOver', 'isWicketBowler': 'isWicketBowlerOver'})
# merge actuals into predict dataframe
masterLookup = masterLookup.merge(ballValues.loc[:, ['overNumber', 'batsmanRunsOver', 'totalInningWickets', 'isWicketBowlerOver']], how='left', on=['overNumber', 'totalInningWickets'])


# merge extras probs and avgruns into predict data for the sim
masterLookup = masterLookup.merge(extras.loc[:, ['overNumber', 'isInvalidOver', 'invalidRunsOver', 'isWicketInvalidOver', 'isWicketRunOutOver', 'byeRunsOver']], how='left', on=['overNumber'])




# ball by ball values model for 0 wickets, which is the base of everything, we work out the runs for 0 wickets then for every other value of wicket we just adjust from this base
X = trainData[['inningBallNumber', 'isPowerplay', 'totalInningWickets']]
X = X[X['totalInningWickets'] == 0]
y = trainData[trainData['totalInningWickets'] == 0]['batsmanRuns']
# Create polynomial features
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X)
# Create and fit the polynomial regression model
model = LinearRegression()
model.fit(X_poly, y)
# Predict the runs using the model
X_pred = masterLookup[['inningBallNumber', 'isPowerplay', 'totalInningWickets']]
X_pred = poly.fit_transform(X_pred)
y_pred = model.predict(X_pred)
# this is the prediction for 0 wickets down for all balls
masterLookup['m_batsmanRunsBallBase'] = y_pred



# from our ball predictions work out the average prediction by over
overPreds = pd.pivot_table(masterLookup, values=['m_batsmanRunsBallBase'], index=['overNumber', 'totalInningWickets'], aggfunc='mean')
masterLookup = masterLookup.merge(overPreds, how='left', on=['totalInningWickets', 'overNumber'], suffixes=('', 'Over'))
masterLookup = masterLookup.rename(columns={'m_batsmanRunsBallBaseOver': 'm_batsmanRunsOverBase'})
# work out rates of the different wickets down actuals vs the prediction for 0 wickets (ie base)
masterLookup['rateBall'] = masterLookup['batsmanRunsBall'] / masterLookup['m_batsmanRunsBallBase']
masterLookup['rateOver'] = masterLookup['batsmanRunsOver'] / masterLookup['m_batsmanRunsOverBase']
masterLookup['rateBall'] = np.where(masterLookup['totalInningWickets'] == 0, 1, masterLookup['rateBall'])
masterLookup['rateOver'] = np.where(masterLookup['totalInningWickets'] == 0, 1, masterLookup['rateOver'])


# merge rates into main data for prediction
trainData = trainData.merge(masterLookup.loc[:, ['inningBallNumber', 'totalInningWickets', 'rateOver']], how='left', on=['inningBallNumber', 'totalInningWickets'])

# ball by ball values model, model the adjustment for each wicket value then multiply that by base 0 wickets value
X = trainData[['inningBallNumber', 'totalInningWickets']]
y = trainData['rateOver']
model = HistGradientBoostingRegressor(monotonic_cst=[1, -1], random_state=42)
model.fit(X, y)
X_pred = masterLookup[['inningBallNumber', 'totalInningWickets']]
y_pred = model.predict(X_pred)
masterLookup['m_rate'] = y_pred
masterLookup['m_batsmanRunsBall'] = np.where(masterLookup['totalInningWickets'] > 0, masterLookup['m_rate'] * masterLookup['m_batsmanRunsBallBase'], masterLookup['m_batsmanRunsBallBase'])

# finally drop unnecessary columns
masterLookup = masterLookup.drop(['m_batsmanRunsBallBase', 'm_batsmanRunsOverBase', 'rateBall', 'rateOver', 'm_rate'], axis=1)




# smoothing out the values wicket by wicket using a simple regression
adjustResults = pd.DataFrame()
for x in np.arange(1, 10, 1):
    adjusts = pd.pivot_table(masterLookup, values=['m_batsmanRunsBall'], index=['inningBallNumber', 'totalInningWickets'], aggfunc='mean').reset_index()
    adjusts = adjusts[adjusts['totalInningWickets'] == x]
    adjusts['isPowerplay'] = np.where(adjusts['inningBallNumber'] <= 36, 1, 0)
    # # ball by ball values model
    X = adjusts[['inningBallNumber', 'totalInningWickets', 'isPowerplay']]
    y = adjusts['m_batsmanRunsBall']
    # Create polynomial features
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X)
    # Create and fit the polynomial regression model
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    adjusts['m_batsmanRunsBall'] = y_pred
    adjustResults = pd.concat([adjustResults, adjusts], axis=0)

# we need to adjust for when runs are unrealistically low in bizarre situations, not really important but needs done
adjusts = pd.pivot_table(adjustResults, values=['m_batsmanRunsBall'], index=['inningBallNumber', 'totalInningWickets'], aggfunc='mean').reset_index()
adjusts['m_batsmanRunsBall'] = np.where(adjusts['m_batsmanRunsBall'] < 0.4, 0.4 - ((0.4 - adjusts['m_batsmanRunsBall']) / 5), adjusts['m_batsmanRunsBall'])

# now merge final numbers into predict data
masterLookup = masterLookup.merge(adjusts, how='left', on=['inningBallNumber', 'totalInningWickets'], suffixes=('x', ''))
masterLookup['m_batsmanRunsBall'] = np.where(masterLookup['totalInningWickets'] == 0, masterLookup['m_batsmanRunsBallx'], masterLookup['m_batsmanRunsBall'])
masterLookup = masterLookup.drop(['m_batsmanRunsBallx'], axis=1)




# merge modelled numbers into train
trainData = trainData.merge(masterLookup.loc[:, ['inningBallNumber', 'totalInningWickets', 'm_batsmanRunsBall']], how='left', on=['inningBallNumber', 'totalInningWickets'])
biasRuns = pd.pivot_table(trainData, values=['m_batsmanRunsBall', 'batsmanRuns'], index=['totalInningWickets', 'overNumber'], aggfunc=['sum', 'count', 'mean']).reset_index()



# wickets model
X = trainData[['inningBallNumber', 'totalInningWickets', 'isPowerplay', 'overNumber']]
y = trainData['isWicketBowler']
model = HistGradientBoostingClassifier(monotonic_cst=[1, 0, 0, 1], random_state=42)
model.fit(X, y)
X_pred = masterLookup[['inningBallNumber', 'totalInningWickets', 'isPowerplay', 'overNumber']]
y_pred = model.predict_proba(X_pred)
masterLookup['m_isWicketBowlerBall'] = y_pred[:, 1]

# the total wicket prob for a valid ball, is the prob of a bowler wicket plus the prob of a run out, the run out is worked out by over just not by ball
masterLookup['m_isWicketBall'] = masterLookup['m_isWicketBowlerBall'] + masterLookup['isWicketRunOutOver']



# merge predict data into train data to check biases
trainData = trainData.merge(masterLookup.loc[:, ['inningBallNumber', 'totalInningWickets', 'm_isWicketBowlerBall']], how='left', on=['inningBallNumber', 'totalInningWickets'])
biasWickets = pd.pivot_table(trainData, values=['m_isWicketBowlerBall', 'isWicketBowler'], index=['totalInningWickets', 'overNumber'], aggfunc=['sum', 'count', 'mean']).reset_index()




# train and target
X = trainData[['inningBallNumber', 'm_batsmanRunsBall', 'totalInningWickets']]
y = trainData['batsmanRuns']

# define and fit the model
# model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5,), max_iter=10000, random_state=42)
model = LogisticRegression(max_iter=10000, random_state=42)
model.fit(X, y)

# now predict the masterlookup scenarios
X_pred = masterLookup[['inningBallNumber', 'm_batsmanRunsBall', 'totalInningWickets']]
y_pred = model.predict_proba(X_pred)
X_pred = pd.concat([X_pred, pd.DataFrame(y_pred)], axis=1)
X_pred['sumProdClassRuns'] = sum(col * X_pred[col] for col in X_pred.columns if isinstance(col, int))
masterLookup = masterLookup.merge(X_pred, how='left', on=['totalInningWickets', 'inningBallNumber', 'm_batsmanRunsBall'])


# create pivot table with mean runs so far, merge
pivot = pd.pivot_table(trainData, values='totalInningRuns', index=['inningBallNumber', 'totalInningWickets'], aggfunc='mean').reset_index()
masterLookup = masterLookup.merge(pivot, on=['inningBallNumber', 'totalInningWickets'], how='left')

# work out average runs for any given wickets and ball number
X = trainData[['inningBallNumber', 'totalInningWickets']]
y = trainData['totalInningRuns']
poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)

# predict using trained model
X_lookup = poly.transform(masterLookup[['inningBallNumber', 'totalInningWickets']])
masterLookup['predTotalInningRuns'] = model.predict(X_lookup)
masterLookup['predTotalInningRuns'] = np.where(masterLookup['sample'] > 100, masterLookup['totalInningRuns'], masterLookup['predTotalInningRuns'])
masterLookup['predTotalInningRuns'] = np.where(masterLookup['predTotalInningRuns'] < 0, 0, masterLookup['predTotalInningRuns'])


# export
masterLookup.to_csv(PROJECT_ROOT / 'men/expBall&runsToCome/1_masterLookup.csv', index=False)




# # Create a 10x2 grid of subplots
# fig, axes = plt.subplots(10, 2, figsize=(14, 40))  # Adjust size for readability
# for i, wickets in enumerate(range(10)):  # Loop over totalInningWickets 0-9
#     # Filter data for the given totalInningWickets
#     men_filtered = masterLookup[masterLookup["totalInningWickets"] == wickets]
#
#     # Column 1: m_batsmanRunsBall
#     ax1 = axes[i, 0]
#     sns.lineplot(data=men_filtered, x="inningBallNumber", y="m_batsmanRunsBall", ax=ax1, label="MenPredictedMean", color="black")
#     sns.lineplot(data=men_filtered, x="inningBallNumber", y="sumProdClassRuns", ax=ax1, label="MenPredictedClass", color="black")
#     ax1.set_title(f"Wickets: {wickets} - Runs Per Ball")
#     ax1.set_xlabel("Inning Ball Number")
#     ax1.set_ylabel("Runs Per Ball")
#     ax1.set_ylim(0, 3)
#     ax1.legend()
#
#     # Column 2: m_batsmanRunsBall for women
#     ax2 = axes[i, 1]
#     sns.lineplot(data=men_filtered, x="inningBallNumber", y="m_isWicketBowlerBall", ax=ax2, label="MenPredicted", color="black")
#     ax2.set_title(f"Wickets: {wickets} - isWicketBowler")
#     ax2.set_xlabel("Inning Ball Number")
#     ax2.set_ylabel("isWicketBowler")
#     ax2.set_ylim(0, 0.2)
#     ax2.legend()
#
# # Adjust layout
# plt.tight_layout()
# plt.savefig('1_expBallModelWomenVSMenPredicted.png')
# plt.show()
#
#
