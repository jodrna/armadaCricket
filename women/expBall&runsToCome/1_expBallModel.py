import pandas as pd
import numpy as np
# from optree.version import suffix
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from paths import PROJECT_ROOT

# import and filter to 1st innings only
trainData = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/Development/women/data/dataClean.csv', parse_dates=['date'])
trainData = trainData[trainData['inningNumber'] == 1]
trainDataMen = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/Development/men/data/dataClean.csv', parse_dates=['date'])
trainDataMen = trainDataMen[trainDataMen['inningNumber'] == 1]
trainDataMen = trainDataMen[(trainDataMen['noballRuns'] == 0) & (trainDataMen['wideRuns'] == 0)]



# mens master lookup, for merging into women's data and getting the difference
masterLookupMen = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/Development/men/expBall&runsToCome/1_masterLookup.csv')

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
extras = extras.rename(columns={'isInvalid': 'isInvalidOver', 'avgRunsInvalid': 'invalidRunsOver', 'isWicket': 'isWicketInvalidOver', 'mean_byeRuns': 'byeRunsOver',
                                'isWicketRunOut': 'isWicketRunOutOver'})




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




# we don't include extras when modelling
trainData = trainData[(trainData['noballRuns'] == 0) & (trainData['wideRuns'] == 0)]
# merge the mens numbers, we'll then model the difference between the 2
trainData = trainData.merge(masterLookupMen.loc[:, ['totalInningWickets', 'inningBallNumber', 'm_batsmanRunsBall', 'm_isWicketBowlerBall']], how='left', on=['totalInningWickets', 'inningBallNumber'])
# difference between men and women's
trainData['runMFRate'] = trainData['batsmanRuns'] / trainData['m_batsmanRunsBall']
trainData['wicketMFRate'] = trainData['isWicketBowler'] / trainData['m_isWicketBowlerBall']


# ball by ball values model for 0 wickets, which is the base of everything, we work out the runs for 0 wickets then for every other value of wicket we just adjust from this base
X = trainData[['overNumber', 'isPowerplay', 'totalInningWickets']]
y = trainData['runMFRate']
# Create polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
# Create and fit the polynomial regression model
model = LinearRegression()
model.fit(X_poly, y)
# Predict the runs using the model
X_pred = masterLookupMen[['overNumber', 'isPowerplay', 'totalInningWickets']]
X_pred = poly.fit_transform(X_pred)
y_pred = model.predict(X_pred)
# this is the prediction for 0 wickets down for all balls
masterLookup['m_batsmanRunsBall'] = masterLookupMen['m_batsmanRunsBall']  * y_pred


# ball by ball values model for 0 wickets, which is the base of everything, we work out the runs for 0 wickets then for every other value of wicket we just adjust from this base
X = trainData[['overNumber', 'isPowerplay', 'totalInningWickets']]
y = trainData['wicketMFRate']
# Create polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
# Create and fit the polynomial regression model
model = LinearRegression()
model.fit(X_poly, y)
# Predict the runs using the model
X_pred = masterLookupMen[['overNumber', 'isPowerplay', 'totalInningWickets']]
X_pred = poly.fit_transform(X_pred)
y_pred = model.predict(X_pred)
# this is the prediction for 0 wickets down for all balls
masterLookup['m_isWicketBowlerBall'] = masterLookupMen['m_isWicketBowlerBall'] * y_pred

# get overall wicket ball prob by adding bowler wicket and run out wicket
masterLookup['m_isWicketBall'] = masterLookup['m_isWicketBowlerBall'] + masterLookup['isWicketRunOutOver']




trainData['gender'] = 1
trainData = trainData.drop(columns=['m_batsmanRunsBall', 'm_isWicketBowlerBall', 'runMFRate', 'wicketMFRate'])
trainData = trainData.merge(masterLookup.loc[:, ['m_batsmanRunsBall', 'totalInningWickets', 'inningBallNumber']], how='left', on=['totalInningWickets', 'inningBallNumber'])
trainDataMen['gender'] = 0
trainDataMen = trainDataMen.merge(masterLookupMen.loc[:, ['m_batsmanRunsBall', 'totalInningWickets', 'inningBallNumber']], how='left', on=['totalInningWickets', 'inningBallNumber'])
trainData = pd.concat([trainDataMen, trainData], axis=0)

# train and target
X = trainData[['inningBallNumber', 'm_batsmanRunsBall', 'totalInningWickets', 'gender']]
y = trainData['batsmanRuns']

# define and fit the model
# model = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(8, 4), max_iter=10000, random_state=42)
model = LogisticRegression(max_iter=10000, random_state=42)
model.fit(X, y)

# now predict the masterlookup scenarios
X_pred = masterLookup[['inningBallNumber', 'm_batsmanRunsBall', 'totalInningWickets']]
X_pred['gender'] = 1
y_pred = model.predict_proba(X_pred)
X_pred = pd.concat([X_pred, pd.DataFrame(y_pred)], axis=1)
X_pred['sumProdClassRuns'] = sum(col * X_pred[col] for col in X_pred.columns if isinstance(col, int))
masterLookup = masterLookup.merge(X_pred, how='left', on=['totalInningWickets', 'inningBallNumber', 'm_batsmanRunsBall'])


classBias = pd.pivot_table(masterLookup, values=['sumProdClassRuns', 'm_batsmanRunsBall', 'batsmanRunsBall'], index=['totalInningWickets'], aggfunc='sum').reset_index()




# # create pivot table with mean runs so far, merge
# pivot = pd.pivot_table(trainData, values='totalInningRuns', index=['inningBallNumber', 'totalInningWickets'], aggfunc='mean').reset_index()
# masterLookup = masterLookup.merge(pivot, on=['inningBallNumber', 'totalInningWickets'], how='left')
#
# # work out average runs for any given wickets and ball number
# X = trainData[['inningBallNumber', 'totalInningWickets']]
# y = trainData['totalInningRuns']
# poly = PolynomialFeatures(degree=5)
# X_poly = poly.fit_transform(X)
# model = LinearRegression()
# model.fit(X_poly, y)
#
# # predict using trained model
# X_lookup = poly.transform(masterLookup[['inningBallNumber', 'totalInningWickets']])
# masterLookup['predTotalInningRuns'] = model.predict(X_lookup)
# masterLookup['predTotalInningRuns'] = np.where(masterLookup['sample'] > 100, masterLookup['totalInningRuns'], masterLookup['predTotalInningRuns'])
# masterLookup['predTotalInningRuns'] = np.where(masterLookup['predTotalInningRuns'] < 0, 0, masterLookup['predTotalInningRuns'])



#
# # Create a 10x2 grid of subplots
# fig, axes = plt.subplots(10, 2, figsize=(14, 40))  # Adjust size for readability
# for i, wickets in enumerate(range(10)):  # Loop over totalInningWickets 0-9
#     # Filter data for the given totalInningWickets
#     men_filtered = masterLookup[masterLookup["totalInningWickets"] == wickets]
#
#     # Column 1: m_batsmanRunsBall
#     ax1 = axes[i, 0]
#     sns.lineplot(data=men_filtered, x="inningBallNumber", y="m_batsmanRunsBall", ax=ax1, label="MenPredicted", color="black")
#     # sns.lineplot(data=men_filtered, x="inningBallNumber", y="m_batsmanRunsBallW", ax=ax1, label="WomenPredicted", color="red")
#     ax1.set_title(f"Wickets: {wickets} - Runs Per Ball")
#     ax1.set_xlabel("Inning Ball Number")
#     ax1.set_ylabel("Runs Per Ball")
#     ax1.set_ylim(0, 3)
#     ax1.legend()
#
#     # Column 2: m_batsmanRunsBall for women
#     ax2 = axes[i, 1]
#     sns.lineplot(data=men_filtered, x="inningBallNumber", y="m_isWicketBowlerBall", ax=ax2, label="MenPredicted", color="black")
#     # sns.lineplot(data=men_filtered, x="inningBallNumber", y="m_isWicketBowlerBallW", ax=ax2, label="WomenPredicted", color="red")
#     ax2.set_title(f"Wickets: {wickets} - isWicketBowler")
#     ax2.set_xlabel("Inning Ball Number")
#     ax2.set_ylabel("isWicketBowler")
#     ax2.set_ylim(0, 0.2)
#     ax2.legend()
#
# # Adjust layout
# plt.tight_layout()
# # plt.savefig('1_expBallModelWomenVSMenPredicted.png')
# plt.show()


# # merge modelled numbers into train
# trainData = trainData.merge(masterLookup.loc[:, ['inningBallNumber', 'totalInningWickets', 'm_batsmanRunsBallW', 'm_isWicketBowlerBallW']], how='left', on=['inningBallNumber', 'totalInningWickets'], suffixes=('', 'W'))
# biasRuns = pd.pivot_table(trainData, values=['m_batsmanRunsBallW', 'batsmanRuns', 'm_isWicketBowlerBallW', 'isWicketBowler'], index=['totalInningWickets'], aggfunc=['sum', 'count', 'mean']).reset_index()


# export
masterLookup.to_csv('/Users/jordan/Documents/ArmadaCricket/Development/women/expBall&runsToCome/1_masterLookup.csv', index=False)
