# PROCESS: stage 4 - expected ball model -> outputs/1_masterLookup_w_v2.csv, 1_masterLookupOrd_w_v2.csv (the sim's
#   base). Reads the MEN's 1_masterLookup_v2.csv - run men's step 1 first.
# See PROCESS.txt at the top of prod_model for the whole run order.
# project root = the first folder above this script holding paths.py, so paths / db / shared modules come from THIS
# project wherever it is (OneDrive, GitHub, a copy) - no PYTHONPATH needed (20 Oct 2026)
import sys as _sys
from pathlib import Path as _Path
_root = str(next(p for p in _Path(__file__).resolve().parents if (p / 'paths.py').exists()))
if _root in _sys.path: _sys.path.remove(_root)
_sys.path.insert(0, _root)
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

# v2 of 1_expBallModel_w.py - same outputs plus what the discrete sim needs, mirroring men's v2/1_expBallModel_v2.py:
#   - runs distribution: gradient boosted classifier on women's not out valid balls (v1 pooled men + women in a logistic regression)
#   - m_isWicketBall: modelled directly on any valid ball wicket (v1 added a run out rate that included wides/no balls)
#   - new columns for the discrete sim: notOutRuns_0-7, invRuns_1-7 (runs off a wide/no ball), byeRuns_0-5, wktRuns_0-4
#   - '0'-'7' keep v1's meaning (runs distribution for any valid ball), rebuilt from the above
#   - m_batsmanRunsBall / m_isWicketBowlerBall still modelled as a ratio to men's (step 2 uses them), now off men's v2 lookup
#   - exports to 1_masterLookup_w_v2.csv

# import and filter to 1st innings only
trainData = pd.read_csv(PROJECT_ROOT / 'women/expBall&runsToCome/data/dataClean_w.csv', parse_dates=['date'])
trainData = trainData[trainData['inningNumber'] == 1]
# v2: The Hundred is left out - its innings end at ball 100, so in this 120 ball frame its runs to come, per ball rates by
# ball number and crease states don't match a T20 (it pulled real runs to come down ~2-3 runs before ball 100 and made a
# step at ball 100 that step 4 couldn't remove). Note the joins are on inningBallNumber, which 3_hundredAdjusts_w.py doesn't
# rescale (only inningBallsRemaining), so the Hundred can't simply be mapped onto this 120 ball frame either
trainData = trainData[trainData['competition'] != "The Hundred (Women's Comp)"]
# surge-era WBBL (from WBBL|08, 2022 - user, 15 Oct 2026) out: it must not shape ball-by-ball values. It stays in dataClean_w for 5_runsToComeYearModel only
trainData = trainData[~((trainData['competition'] == "Women's Big Bash League") & (pd.to_datetime(trainData['date']) > pd.Timestamp('2022-06-01')))]



# mens master lookup, for merging into women's data and getting the difference
masterLookupMen = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/outputs/1_masterLookup_v2.csv')

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
extras = extras.rename(columns={'isInvalid': 'isInvalidOver', 'avgRunsInvalid': 'invalidRunsOver', 'isWicket': 'isWicketInvalidOver', 'mean_byeRuns': 'byeRunsOver',
                                'isWicketRunOut': 'isWicketRunOutOver'})


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

# v2: the total wicket prob for a valid ball is modelled directly on any wicket (bowler or run out) by ball and wickets,
# rather than bowler wickets plus a run out rate by over. min_samples_leaf keeps thin situations from being fit to noise
model = HistGradientBoostingClassifier(min_samples_leaf=200, random_state=42)
model.fit(trainData[['inningBallNumber', 'totalInningWickets']], trainData['isWicket'])
masterLookup['m_isWicketBall'] = model.predict_proba(masterLookup[['inningBallNumber', 'totalInningWickets']])[:, 1]




# v2: runs off the bat distribution, 0-7, on valid balls where the batter ISN'T out, from a gradient boosted classifier
# on ball and wickets (v1's logistic regression was linear in its inputs and couldn't follow the death overs).
# min_samples_leaf keeps thin situations from being fit to noise
runOutcomes = list(range(8))
notOut = trainData[trainData['isWicket'] == 0]
model = HistGradientBoostingClassifier(min_samples_leaf=200, random_state=42)
# 7s off the bat are near non existent, folded into 6 so the classifier's validation split works
model.fit(notOut[['inningBallNumber', 'totalInningWickets']], notOut['batsmanRuns'].clip(0, 6))
notOutProbs = pd.DataFrame(model.predict_proba(masterLookup[['inningBallNumber', 'totalInningWickets']]), columns=model.classes_)
notOutProbs = notOutProbs.reindex(columns=runOutcomes, fill_value=0)
for i in runOutcomes:
    masterLookup[f'notOutRuns_{i}'] = notOutProbs[i].to_numpy()

# v2: the same wicket and not out runs models with the striker's batting order added (ord 1-2 grouped as 2, 11+ as 11),
# so the sim can use the rates for whoever's on strike. Held out by match (2026-10-09): runs calibration by order went from
# ~8x noise to noise level; leaf 400 matched or beat 200 and smaller leaves overfit the runs
# position in the over (1-6) as its own input: the 6th ball's real dip in runs (men ~4%, strike farming / taking a single)
# was only half captured from the ball number alone; held out by match it improved men's runs and wickets (women neutral)
trainData['ordK'] = trainData['ord'].clip(2, 11).astype(int)
trainData['ballInOver'] = ((trainData['inningBallNumber'] - 1) % 6 + 1).astype(int)
ordFeatures = ['inningBallNumber', 'totalInningWickets', 'ordK', 'ballInOver']
masterLookupOrd = pd.DataFrame([(b, w, o) for w in range(10) for b in range(1, 121) for o in range(2, 12)], columns=ordFeatures[:3])
masterLookupOrd['ballInOver'] = ((masterLookupOrd['inningBallNumber'] - 1) % 6 + 1).astype(int)
ordWicketModel = HistGradientBoostingClassifier(min_samples_leaf=400, random_state=42)
ordWicketModel.fit(trainData[ordFeatures], trainData['isWicket'])
masterLookupOrd['m_isWicketBall'] = ordWicketModel.predict_proba(masterLookupOrd[ordFeatures])[:, 1]
ordRunsModel = HistGradientBoostingClassifier(min_samples_leaf=400, random_state=42)
ordNotOut = trainData[trainData['isWicket'] == 0]
ordRunsModel.fit(ordNotOut[ordFeatures], ordNotOut['batsmanRuns'].clip(0, 6))
ordNotOutProbs = pd.DataFrame(ordRunsModel.predict_proba(masterLookupOrd[ordFeatures]), columns=ordRunsModel.classes_)
ordNotOutProbs = ordNotOutProbs.reindex(columns=runOutcomes, fill_value=0)
for i in runOutcomes:
    masterLookupOrd[f'notOutRuns_{i}'] = ordNotOutProbs[i].to_numpy()
masterLookupOrd.to_csv(PROJECT_ROOT / 'women/expBall&runsToCome/outputs/1_masterLookupOrd_w_v2.csv', index=False)

# discrete extras distributions by over, for the sim
masterLookup = masterLookup.merge(invalidRunsDist, how='left', on=['overNumber'])
masterLookup = masterLookup.merge(byeRunsDist, how='left', on=['overNumber'])
masterLookup = masterLookup.merge(wicketBallRunsDist, how='left', on=['overNumber'])

# the unconditional runs distribution for a valid ball (same meaning as v1's '0'-'7' columns) mixes the not out
# distribution with the runs scored on wicket balls
for i in runOutcomes:
    wktRuns = masterLookup[f'wktRuns_{i}'] if f'wktRuns_{i}' in masterLookup.columns else 0
    masterLookup[str(i)] = (1 - masterLookup['m_isWicketBall']) * masterLookup[f'notOutRuns_{i}'] + masterLookup['m_isWicketBall'] * wktRuns
masterLookup['sumProdClassRuns'] = sum(i * masterLookup[str(i)] for i in runOutcomes)


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
# export - v2 file name
masterLookup.to_csv(PROJECT_ROOT / 'women/expBall&runsToCome/outputs/1_masterLookup_w_v2.csv', index=False)
