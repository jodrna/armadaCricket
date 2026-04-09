import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import log_loss
from paths import PROJECT_ROOT


# import
trainData = pd.read_csv(PROJECT_ROOT / 'women/data/dataClean.csv', parse_dates=['date'])
masterLookup = pd.read_csv(PROJECT_ROOT / 'women/expBall&runsToCome/5_masterLookup.csv')
chaseSituations = pd.read_csv(PROJECT_ROOT / 'women/matchMarket/chaseSituationBuilder.csv')
chaseLookupLive = pd.read_csv(PROJECT_ROOT / 'women/matchMarket/1_chaseLookupLive.csv')



# drop nans from adj
trainData = trainData.dropna(axis=0, subset=['runsRequiredAdj'])
# now when running for adj simply change std runs required to adj
trainData['runsRequiredStd'] = trainData['runsRequired']
# take out the below 2 lines when running standard runs model
trainData['runsRequired'] = trainData['runsRequiredAdj']
# round to nearest int
trainData['runsRequired'] = trainData['runsRequired'].round()



# Create a new dataframe with expanded rows from the max runs required defined in chase situation builder
chaseSituationsRows = []
for _, row in chaseSituations.iterrows():
    for runs in range(1, row['maxRunsRequired'] + 1):
        chaseSituationsRows.append({
            'inningBallNumber': row['inningBallNumber'],
            'inningBallsRemaining': row['inningBallsRemaining'],
            'totalInningWickets': row['totalInningWickets'],
            'runsRequired': runs
        })
# Create the new dataframe with the expanded out rows
chaseSituations = pd.DataFrame(chaseSituationsRows)
chaseSituations = chaseSituations.sort_values(by=['inningBallsRemaining', 'runsRequired', 'totalInningWickets']).reset_index(drop=True)




# we only want innings 2 for the chase predictions, and shuffle the data
trainData = trainData[trainData['inningNumber'] == 2]
trainData = trainData.sample(frac=1, random_state=42).reset_index(drop=True)
# we need to remove duplicates in runs to come so just select batting order 1
masterLookup = masterLookup[(masterLookup['ord'] == 1) & (masterLookup['daysGroup'] == 16)]


trainData = trainData.merge(masterLookup.loc[:, ['totalInningRunsToComeSimBiasSpline', 'totalInningWickets', 'inningBallNumber', 'totalInningValidBallsFacedToCome', 'bowledOut']], how='left', on=['totalInningWickets', 'inningBallNumber'])

trainData['ratioRequired'] = trainData['runsRequired'] / trainData['totalInningRunsToComeSimBiasSpline']
trainData = trainData.dropna(axis=0, subset=['ratioRequired'])



# # create an empty dataframe
# chaseLookup = pd.pivot_table(trainData, values=['sample', 'chaseWin', 'totalInningRunsToCome', 'totalInningWicketsToCome'],
#                             index=['totalInningWickets', 'inningBallNumber', 'runsRequired'],
#                             aggfunc={'sample': 'sum', 'chaseWin': 'sum', 'totalInningRunsToCome': 'mean', 'totalInningWicketsToCome': 'mean'}).reset_index()
# chaseLookup['chaseWin%'] = chaseLookup['chaseWin'] / chaseLookup['sample']
# chaseLookup = chaseSituations.merge(chaseLookup, how='left', on=['totalInningWickets', 'inningBallNumber', 'runsRequired'])
# chaseLookup = chaseLookup.rename(columns={'sample': 'chaseSample'})
# chaseLookup['daysGroup'] = 16
# chaseLookup['year'] = 2025
#
# chaseLookup = chaseLookup.merge(masterLookup.loc[:, ['totalInningWickets', 'inningBallNumber', 'sample', 'totalInningRunsToComeSimBiasSplineYear', 'totalInningValidBallsFacedToCome', 'bowledOut', 'daysGroup']], how='left', on=['totalInningWickets', 'inningBallNumber', 'daysGroup'])
# chaseLookup = chaseLookup.rename(columns={'sample': 'ballWicketSample'})
# chaseLookup['daysGroup'] = 16
#
# chaseLookup['ratioRequired'] = chaseLookup['runsRequired'] / chaseLookup['totalInningRunsToComeSimBiasSplineYear']
# chaseLookup = chaseLookup.dropna(axis=0, subset=['totalInningRunsToComeSimBiasSplineYear']).reset_index(drop=True)
# chaseLookup['in'] = 1



# create an empty dataframe
chaseLookup = pd.pivot_table(trainData, values=['sample', 'chaseWin', 'totalInningRunsToCome', 'totalInningWicketsToCome', 'runsRequiredStd'],
                            index=['totalInningWickets', 'inningBallNumber', 'runsRequired'],
                            aggfunc={'sample': 'sum', 'chaseWin': 'sum', 'totalInningRunsToCome': 'mean', 'totalInningWicketsToCome': 'mean', 'runsRequiredStd': 'mean'}).reset_index()
chaseLookup['chaseWin%'] = chaseLookup['chaseWin'] / chaseLookup['sample']
chaseLookup = chaseSituations.merge(chaseLookup, how='left', on=['totalInningWickets', 'inningBallNumber', 'runsRequired'])
chaseLookup = chaseLookup.rename(columns={'sample': 'chaseSample'})
chaseLookup = chaseLookup.merge(masterLookup.loc[:, ['totalInningWickets', 'inningBallNumber', 'sample', 'totalInningRunsToComeSimBiasSpline', 'totalInningValidBallsFacedToCome', 'bowledOut']], how='left', on=['totalInningWickets', 'inningBallNumber'])
chaseLookup = chaseLookup.rename(columns={'sample': 'ballWicketSample'})

chaseLookup['ratioRequired'] = chaseLookup['runsRequired'] / chaseLookup['totalInningRunsToComeSimBiasSpline']
chaseLookup['daysGroup'] = 16
chaseLookup = chaseLookup.dropna(axis=0, subset=['totalInningRunsToComeSimBiasSpline']).reset_index(drop=True)
chaseLookup['in'] = 1


# remove chases which are effectively lost
trainData = trainData.merge(chaseLookup.loc[:, ['in', 'totalInningWickets', 'runsRequired', 'inningBallNumber']], how='left', on=['totalInningWickets', 'runsRequired', 'inningBallNumber'])
trainData = trainData[trainData['in'] == 1]






# high wicket model
trainDataMain = trainData.copy()
trainDataMain = trainDataMain[(trainDataMain['inningBallsRemaining'] > 1)]

# prepare the data
y = trainDataMain['chaseWin']
X_std = trainDataMain[['runsRequired', 'ratioRequired', 'totalInningWickets', 'inningBallsRemaining']]
scaler = StandardScaler()
scaler.fit(X_std)
X_std = scaler.transform(X_std)

# build the model
model = MLPClassifier(hidden_layer_sizes=(4, 2), random_state=42, activation='logistic', batch_size='auto', learning_rate='constant', max_iter=5000, early_stopping=False, learning_rate_init=0.001)
model.fit(X_std, y)
trainDataMain['m_chaseWin%'] = model.predict_proba(X_std)[:, 1]



# now predict the chase situations outside of training
chaseLookupMain = chaseLookup.copy()
chaseLookupMain = chaseLookupMain[(chaseLookupMain['inningBallsRemaining'] > 6)]

# chaseLookupMain = chaseLookupMain[(chaseLookupMain['inningBallsRemaining'] == 120) & (chaseLookupMain['totalInningWickets'] == 0) & (chaseLookupMain['runsRequired'] == 1)]
# chaseLookupMain = pd.concat([chaseLookupMain]*12, ignore_index=True).assign(
#     totalInningRunsToComeSimBiasSplineYear=[130.99822, 132.30136, 133.60451, 134.90765, 136.21080, 137.51395,
#         138.81709, 140.12024, 141.42338, 142.72653, 144.02967, 145.33282],
#     daysGroup=range(12)
# )
# chaseLookupMain['runsRequired'] = chaseLookupMain['totalInningRunsToComeSimBiasSplineYear']
# chaseLookupMain['ratioRequired'] = 1

X = chaseLookupMain[['runsRequired', 'ratioRequired', 'totalInningWickets', 'inningBallsRemaining']]
X = scaler.transform(X)
chaseLookupMain['m_chaseWin%'] = model.predict_proba(X)[:, 1]
trainDataMain = trainDataMain[(trainDataMain['inningBallsRemaining'] > 6)]








# last over model
trainDataLastOver = trainData.copy()
trainDataLastOver = trainDataLastOver[(trainDataLastOver['inningBallsRemaining'] < 7)]

# prepare the data
y = trainDataLastOver['chaseWin']
X_std = trainDataLastOver[['runsRequired', 'ratioRequired', 'totalInningWickets', 'daysGroup', 'inningBallsRemaining']]
scaler = StandardScaler()
scaler.fit(X_std)
X_std = scaler.transform(X_std)

# build the model
model = MLPClassifier(hidden_layer_sizes=(8, 4), random_state=42, activation='logistic', batch_size='auto', learning_rate='constant', max_iter=5000, early_stopping=False, learning_rate_init=0.001)
model.fit(X_std, y)
trainDataLastOver['m_chaseWin%'] = model.predict_proba(X_std)[:, 1]

# now predict the chase situations outside of training
chaseLookupLastOver = chaseLookup.copy()
chaseLookupLastOver = chaseLookupLastOver[(chaseLookupLastOver['inningBallsRemaining'] < 7)]
X = chaseLookupLastOver[['runsRequired', 'ratioRequired', 'totalInningWickets', 'daysGroup', 'inningBallsRemaining']]
X = scaler.transform(X)
chaseLookupLastOver['m_chaseWin%'] = model.predict_proba(X)[:, 1]

# Scaling to range [0.0001, 0.9999]
min_val, max_val = 0.0001, 0.9999
chaseLookupLastOver['m_chaseWin%'] = min_val + (chaseLookupLastOver['m_chaseWin%'] - chaseLookupLastOver['m_chaseWin%'].min()) * (max_val - min_val) / (chaseLookupLastOver['m_chaseWin%'].max() - chaseLookupLastOver['m_chaseWin%'].min())



# combine the 2 models
chaseLookup = pd.concat([chaseLookupLastOver, chaseLookupMain], axis=0).reset_index(drop=True)
trainData = pd.concat([trainDataLastOver, trainDataMain], axis=0).reset_index(drop=True)


# order correctly for illogical situations
cols = chaseLookup.loc[:, ['totalInningWickets', 'runsRequired', 'inningBallsRemaining', 'm_chaseWin%']]
colsWrong = cols.sort_values(by=['totalInningWickets', 'runsRequired', 'inningBallsRemaining'], axis=0).reset_index(drop=True)
colsRight = cols.sort_values(by=['totalInningWickets', 'runsRequired', 'm_chaseWin%'], axis=0).reset_index(drop=True)
colsWrong['m_chaseWin%'] = colsRight['m_chaseWin%']
colsWrong = colsWrong.sort_values(by=['inningBallsRemaining', 'runsRequired', 'totalInningWickets'], axis=0).reset_index(drop=True)
chaseLookup['m_chaseWin%'] = colsWrong['m_chaseWin%']


# bias check
bias = pd.pivot_table(trainData, values=['m_chaseWin%', 'chaseWin', 'sample'], aggfunc='sum', index=['totalInningWickets']).reset_index()
bias['bias'] = bias['m_chaseWin%'] / bias['chaseWin']
bias['win%'] = bias['chaseWin'] / bias['sample']

# compare
chaseLookup = chaseLookup.merge(chaseLookupLive.loc[:, ['m_chaseWin%', 'totalInningWickets', 'runsRequired', 'inningBallsRemaining']],
                                how='left', on=['totalInningWickets', 'runsRequired', 'inningBallsRemaining'], suffixes=('', 'Live'))
chaseLookup['m_diff'] = chaseLookup['m_chaseWin%'] - chaseLookup['m_chaseWin%Live']


# chase win % year
years = pd.pivot_table(trainData, index=['totalInningWickets', 'runsRequired', 'inningBallsRemaining'], values=['m_chaseWin%'], aggfunc='mean').reset_index()
chaseLookup = chaseLookup.merge(years, how='left', on=['totalInningWickets', 'runsRequired', 'inningBallsRemaining'], suffixes=('', 'Year'))



# # graph of predictions
# fig, axes = plt.subplots(10, 4, figsize=(20, 40))           # create a figure of dimension 10 (Wickets) by 5 (number of graphs for each wicket)
# for x in np.arange(0, 10, 1):                               # loop 0-10 for wickets
#     graph_data = chaseLookup.copy()
#     graph_data = graph_data[graph_data['totalInningWickets'] == x]       # filter the dataframe for the wicket in question
#     # graph_data['chase_adj%'] = graph_data['blendr_win%'] - graph_data['X_win%']
#     # create tables of the numbers to be plotted
#     actual = pd.pivot_table(graph_data, index='runsRequired', columns='inningBallsRemaining', values='chaseWin%', aggfunc='mean')
#     old = pd.pivot_table(graph_data, index='runsRequired', columns='inningBallsRemaining', values='m_chaseWin%', aggfunc='mean')
#     new = pd.pivot_table(graph_data, index='runsRequired', columns='inningBallsRemaining', values='m_chaseWin%', aggfunc='mean')
#     diff = pd.pivot_table(graph_data, index='runsRequired', columns='inningBallsRemaining', values='m_diff', aggfunc='mean')
#     # plot in a heatmap
#     sns.heatmap(ax=axes[x, 0], data=actual, cmap=plt.cm.get_cmap('PiYG', 1000), vmin=0, vmax=1, center=0.5, xticklabels=10, yticklabels=10)
#     sns.heatmap(ax=axes[x, 1], data=old, cmap=plt.cm.get_cmap('PiYG', 1000), vmin=0, vmax=1, center=0.5, xticklabels=10, yticklabels=10)
#     sns.heatmap(ax=axes[x, 2], data=new, cmap=plt.cm.get_cmap('PiYG', 1000), vmin=0, vmax=1, center=0.5, xticklabels=10, yticklabels=10)
#     sns.heatmap(ax=axes[x, 3], data=diff, cmap=plt.cm.get_cmap('PiYG', 1000), vmin=-0.2, vmax=0.2, center=0, xticklabels=10, yticklabels=10)
#
#     # set titles for each graph
#     title1 = f"actual_win% - {x} wickets lost"
#     axes[x, 0].set_title(title1)
#     title2 = f"old - {x} wickets lost"
#     axes[x, 1].set_title(title2)
#     title3 = f"new {x} wickets lost"
#     axes[x, 2].set_title(title3)
#     title4 = f"diff - {x} wickets lost"
#     axes[x, 3].set_title(title4)
#     # title5 = f"blendr_win%_ - {x} wickets lost"
#     # axes[x, 4].set_title(title4)
# plt.tight_layout()
# plt.show()



# # some year analysis
# trainData = trainData[trainData['inningBallsRemaining'] == 120]
# chasewins = pd.pivot_table(trainData, values=['chaseWin', 'target'], index=['competition'], aggfunc='mean').reset_index()
#
#
# m_chaseWinWeighted = pd.pivot_table(trainData, values=['m_chaseWin%'], index=['inningBallsRemaining', 'totalInningWickets', 'runsRequired'], aggfunc='mean').reset_index()
# chaseLookup = chaseLookup.merge(m_chaseWinWeighted, how='left', on=['inningBallsRemaining', 'totalInningWickets', 'runsRequired'], suffixes=('', 'Year'))

# exports
# chaseLookup.to_csv(PROJECT_ROOT / 'women/matchMarket/1_chaseLookup.csv', index=False)



# now
years2 = pd.pivot_table(trainData, values=['chaseWin'], index='year', aggfunc=['mean', 'count']).reset_index()

rasums = pd.pivot_table(trainData, values=['RA_Sum'], index=['inningBallNumber', 'totalInningWickets'], aggfunc=['mean']).reset_index()
