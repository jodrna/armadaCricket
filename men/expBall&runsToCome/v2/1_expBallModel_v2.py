import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from paths import PROJECT_ROOT

# v2 of 1_expBallModel.py - same outputs plus what the discrete sim needs. Changes:
#   - runs distribution: gradient boosted classifier on not out valid balls (v1 logistic regression ran ~18% high at the death)
#   - m_isWicketBall: modelled directly on any valid ball wicket (v1 added a run out rate that included wides/no balls)
#   - new columns for the discrete sim: notOutRuns_0-7, invRuns_1-7 (runs off a wide/no ball), byeRuns_0-5, wktRuns_0-4
#   - '0'-'7' keep v1's meaning (runs distribution for any valid ball), rebuilt from the above
#   - exports to 1_masterLookup_v2.csv (v1's export is commented out)

# import and filter to 1st innings only
trainData = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/data/dataClean.csv', parse_dates=['date'])
trainData = trainData[trainData['inningNumber'] == 1]

# # remove IPL and ILT20 after 2023 because of sub rule
# trainData = trainData[
#     ~(
#         trainData['competition'].isin(['Indian Premier League', 'International League T20']) &
#         (trainData['year'] > 2022)
#     )
# ]
# comps = pd.pivot_table(trainData, values=['batsmanRuns'], index=['competition'], columns=['year'], aggfunc='count').reset_index()

# extras averages, must be done here at the start before we remove these for modelling ball by ball
extras = pd.pivot_table(trainData, values=['wideRuns', 'noballRuns', 'isWide', 'isNoball', 'byeRuns'], index=['overNumber'], aggfunc=['sum', 'mean']).reset_index()
extras.columns = ['_'.join([str(elem) for elem in col]).strip() for col in extras.columns.values]
extras['isInvalid'] = extras['mean_isWide'] + extras['mean_isNoball']
extras['avgRunsInvalid'] = (extras['sum_noballRuns'] + extras['sum_wideRuns']) / (extras['sum_isNoball'] + extras['sum_isWide'])
extras = extras.rename(columns={'overNumber_': 'overNumber'})

# combined prob of no ball and wide given over number + avg runs from that, add on avg byeRuns regardless, then prob of wicket if a wide/noball
avgWicketsInvalid = pd.pivot_table(trainData[(trainData['noballRuns'] > 0) | (trainData['wideRuns'] > 0)], values=['isWicket'], index=['overNumber'], aggfunc='mean').reset_index()
# v2: run outs on valid balls only (v1 used | here, which kept every ball)
avgWicketsRunOut = pd.pivot_table(trainData[(trainData['noballRuns'] == 0) & (trainData['wideRuns'] == 0)], values=['isWicketRunOut'], index=['overNumber'], aggfunc='mean').reset_index()
extras = extras.merge(avgWicketsInvalid, how='left', on=['overNumber'])
extras = extras.merge(avgWicketsRunOut, how='left', on=['overNumber'])
extras = extras.rename(columns={'isInvalid': 'isInvalidOver', 'avgRunsInvalid': 'invalidRunsOver', 'isWicket': 'isWicketInvalidOver', 'mean_byeRuns': 'byeRunsOver', 'isWicketRunOut': 'isWicketRunOutOver'})


# v2: full distributions by over, so the sim can draw exact extras rather than adding averages
# (leg byes are already counted inside batsmanRuns in this data, byes are separate)
def over_distribution(data, values, outcomes, prefix):
    dist = pd.crosstab(data['overNumber'], values.clip(outcomes[0], outcomes[-1]), normalize='index')
    dist = dist.reindex(columns=outcomes, fill_value=0)
    dist.columns = [f'{prefix}{i}' for i in outcomes]
    return dist.reset_index()

invalidBalls = trainData[(trainData['noballRuns'] > 0) | (trainData['wideRuns'] > 0)]
validNotOutDots = trainData[(trainData['noballRuns'] == 0) & (trainData['wideRuns'] == 0) & (trainData['isWicket'] == 0) & (trainData['batsmanRuns'] == 0)]
validWicketBalls = trainData[(trainData['noballRuns'] == 0) & (trainData['wideRuns'] == 0) & (trainData['isWicket'] == 1)]
# total runs off a wide or no ball, penalty included (e.g. a wide to the rope is 5, a no ball hit for six is 7)
invalidRunsDist = over_distribution(invalidBalls, invalidBalls['wideRuns'] + invalidBalls['noballRuns'] + invalidBalls['batsmanRuns'].clip(lower=0) + invalidBalls['byeRuns'], list(range(1, 8)), 'invRuns_')
# byes on a valid ball the batter didn't score off and wasn't out on
byeRunsDist = over_distribution(validNotOutDots, validNotOutDots['byeRuns'], list(range(0, 6)), 'byeRuns_')
# runs off the bat on the ball a wicket falls (mostly 0, run outs can come with completed runs)
wicketBallRunsDist = over_distribution(validWicketBalls, validWicketBalls['batsmanRuns'], list(range(0, 5)), 'wktRuns_')


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

# v2: the total wicket prob for a valid ball is modelled directly on any wicket (bowler or run out) by ball and wickets,
# rather than bowler wickets plus a run out rate by over. min_samples_leaf keeps thin situations from being fit to noise
X = trainData[['inningBallNumber', 'totalInningWickets']]
y = trainData['isWicket']
model = HistGradientBoostingClassifier(min_samples_leaf=200, random_state=42)
model.fit(X, y)
masterLookup['m_isWicketBall'] = model.predict_proba(masterLookup[['inningBallNumber', 'totalInningWickets']])[:, 1]



# merge predict data into train data to check biases
trainData = trainData.merge(masterLookup.loc[:, ['inningBallNumber', 'totalInningWickets', 'm_isWicketBowlerBall']], how='left', on=['inningBallNumber', 'totalInningWickets'])
biasWickets = pd.pivot_table(trainData, values=['m_isWicketBowlerBall', 'isWicketBowler'], index=['totalInningWickets', 'overNumber'], aggfunc=['sum', 'count', 'mean']).reset_index()




# v2: runs off the bat distribution, 0-7, on valid balls where the batter ISN'T out, from a gradient boosted classifier
# on ball and wickets (v1's logistic regression was linear in its inputs and couldn't follow the death overs,
# running ~18% high at 0-2 wickets there). min_samples_leaf keeps thin situations from being fit to noise
runOutcomes = list(range(8))
notOut = trainData[trainData['isWicket'] == 0]
model = HistGradientBoostingClassifier(min_samples_leaf=200, random_state=42)
# 7s off the bat are near non existent (1 ball), folded into 6 so the classifier's validation split works
model.fit(notOut[['inningBallNumber', 'totalInningWickets']], notOut['batsmanRuns'].clip(0, 6))
notOutProbs = pd.DataFrame(model.predict_proba(masterLookup[['inningBallNumber', 'totalInningWickets']]), columns=model.classes_)
notOutProbs = notOutProbs.reindex(columns=runOutcomes, fill_value=0)
for i in runOutcomes:
    masterLookup[f'notOutRuns_{i}'] = notOutProbs[i].to_numpy()

# discrete extras distributions by over, for the sim
masterLookup = masterLookup.merge(invalidRunsDist, how='left', on=['overNumber'])
masterLookup = masterLookup.merge(byeRunsDist, how='left', on=['overNumber'])
masterLookup = masterLookup.merge(wicketBallRunsDist, how='left', on=['overNumber'])

# the unconditional runs distribution for a valid ball (same meaning as v1's '0'-'7' columns, which matchMarket/simModel.py
# also reads) mixes the not out distribution with the runs scored on wicket balls
for i in runOutcomes:
    wktRuns = masterLookup[f'wktRuns_{i}'] if f'wktRuns_{i}' in masterLookup.columns else 0
    masterLookup[str(i)] = (1 - masterLookup['m_isWicketBall']) * masterLookup[f'notOutRuns_{i}'] + masterLookup['m_isWicketBall'] * wktRuns
masterLookup['sumProdClassRuns'] = sum(i * masterLookup[str(i)] for i in runOutcomes)


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


# export - v2 file name so step 2 / the sim can be switched over to it when ready
masterLookup.to_csv(PROJECT_ROOT / 'men/expBall&runsToCome/outputs/1_masterLookup_v2.csv', index=False)




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
