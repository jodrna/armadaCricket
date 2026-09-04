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
trainData = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/data/dataClean.csv', parse_dates=['date'])
masterLookup = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/outputs/5_masterLookup.csv')
chaseSituations = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/auxiliaries/chaseSituationBuilder.csv')
# chaseLookupLive = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/outputs/1_chaseLookupLive.csv')


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
masterLookup = masterLookup[(masterLookup['ord'] == 1) & (masterLookup['daysGroup'] == 11)]
# merge in runs to come
trainData = trainData.merge(masterLookup.loc[:, ['totalInningRunsToComeSimBiasSplineYear', 'totalInningWickets', 'inningBallNumber', 'totalInningValidBallsFacedToCome', 'bowledOut', 'sample']].rename(columns={'sample': 'ballWicketSample'}), how='left', on=['totalInningWickets', 'inningBallNumber'])
# ballWicketSample = sample size behind the (wickets, ball) state, kept on every row so it can be used to mask out unreliable states when scoring/checking the model (not when training it)
# create a ratio of runs to come to be used as a predictor, drop any nans
trainData['ratioRequired'] = trainData['runsRequired'] / trainData['totalInningRunsToComeSimBiasSplineYear']
trainData = trainData.dropna(axis=0, subset=['ratioRequired'])


# test = trainData.copy()
# test['wickets_group'] = np.round(test['totalInningWickets'] / 3, 0) * 3
# test['year_group'] = np.round(test['year'] / 3, 0) * 3
# test['runsRequired_round'] = np.round(test['runsRequiredAdj'], 0)
# test = pd.pivot_table(test, values=['sample', 'chaseWin'],
#                             index=['wickets_group', 'inningBallNumber', 'runsRequired_round', 'year_group'],
#                             aggfunc={'sample': 'sum', 'chaseWin': 'sum'}).reset_index()
# test['chase_win%'] = test['chaseWin'] / test['sample']
# test = test[(test['wickets_group'] < 7) & (test['inningBallNumber'] == 115)] #



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
chaseLookup['daysGroup'] = 11.5
chaseLookup = chaseLookup.dropna(axis=0, subset=['totalInningRunsToComeSimBiasSpline']).reset_index(drop=True)
chaseLookup['in'] = 1

# remove chases which are effectively lost
trainData = trainData.merge(chaseLookup.loc[:, ['in', 'totalInningWickets', 'runsRequired', 'inningBallNumber']], how='left', on=['totalInningWickets', 'runsRequired', 'inningBallNumber'])
trainData = trainData[trainData['in'] == 1]



# # OLD METHOD
# # over 1-19 model, train on all data but keep only over 1-19
# trainDataMain = trainData.copy()
# trainDataMain = trainDataMain[(trainDataMain['inningBallsRemaining'] > 1)]
#
# # prepare the data
# y = trainDataMain['chaseWin']
# X_std = trainDataMain[['runsRequired', 'ratioRequired', 'totalInningWickets', 'daysGroup']]
# scaler = StandardScaler()
# scaler.fit(X_std)
# X_std = scaler.transform(X_std)
#
# # build the model
# model = MLPClassifier(hidden_layer_sizes=(8, 4), random_state=42, activation='logistic', batch_size='auto', learning_rate='constant', max_iter=5000, early_stopping=False, learning_rate_init=0.001)
# model.fit(X_std, y)
# trainDataMain['m_chaseWin%'] = model.predict_proba(X_std)[:, 1]
#
# # now predict the chase situations outside of training
# chaseLookupMain = chaseLookup.copy()
# chaseLookupMain = chaseLookupMain[(chaseLookupMain['inningBallsRemaining'] > 6)]
# X = chaseLookupMain[['runsRequired', 'ratioRequired', 'totalInningWickets', 'daysGroup']]
# X = scaler.transform(X)
# chaseLookupMain['m_chaseWin%'] = model.predict_proba(X)[:, 1]
# trainDataMain = trainDataMain[(trainDataMain['inningBallsRemaining'] > 6)]
#
#
# # last over model
# trainDataLastOver = trainData.copy()
# trainDataLastOver = trainDataLastOver[(trainDataLastOver['inningBallsRemaining'] < 7)]
#
# # prepare the data
# y = trainDataLastOver['chaseWin']
# X_std = trainDataLastOver[['runsRequired', 'ratioRequired', 'totalInningWickets', 'inningBallsRemaining', 'daysGroup']]
# scaler = StandardScaler()
# scaler.fit(X_std)
# X_std = scaler.transform(X_std)
#
# # build the model
# model = MLPClassifier(hidden_layer_sizes=(8, 4), random_state=42, activation='logistic', batch_size='auto', learning_rate='constant', max_iter=5000, early_stopping=False, learning_rate_init=0.001)
# model.fit(X_std, y)
# trainDataLastOver['m_chaseWin%'] = model.predict_proba(X_std)[:, 1]
#
# # now predict the chase situations outside of training
# chaseLookupLastOver = chaseLookup.copy()
# chaseLookupLastOver = chaseLookupLastOver[(chaseLookupLastOver['inningBallsRemaining'] < 7)]
# X = chaseLookupLastOver[['runsRequired', 'ratioRequired', 'totalInningWickets', 'inningBallsRemaining', 'daysGroup']]
# X = scaler.transform(X)
# chaseLookupLastOver['m_chaseWin%'] = model.predict_proba(X)[:, 1]
#
#
# # Scaling to range [0.0001, 0.9999]
# min_val, max_val = 0.0001, 0.9999
# chaseLookupLastOver['m_chaseWin%'] = min_val + (chaseLookupLastOver['m_chaseWin%'] - chaseLookupLastOver['m_chaseWin%'].min()) * (max_val - min_val) / (chaseLookupLastOver['m_chaseWin%'].max() - chaseLookupLastOver['m_chaseWin%'].min())
#
#
# # combine the 2 models
# chaseLookup = pd.concat([chaseLookupLastOver, chaseLookupMain], axis=0).reset_index(drop=True)
# trainData = pd.concat([trainDataLastOver, trainDataMain], axis=0).reset_index(drop=True)

# NEW METHOD:
# start of innings model
trainDataMain = trainData.copy()

# prepare the data
y = trainDataMain['chaseWin']
X_std = trainDataMain[['runsRequired', 'ratioRequired', 'totalInningWickets', 'daysGroup']]
scaler = StandardScaler()
scaler.fit(X_std)
X_std = scaler.transform(X_std)

# build the model
model = MLPClassifier(hidden_layer_sizes=(64, 32), random_state=42, activation='logistic', batch_size='auto', learning_rate='constant', max_iter=5000, early_stopping=False, learning_rate_init=0.001)
model.fit(X_std, y)
trainDataMain['m_chaseWin%Main'] = model.predict_proba(X_std)[:, 1]

# now predict the chase situations outside of training
chaseLookupMain = chaseLookup.copy()
X = chaseLookupMain[['runsRequired', 'ratioRequired', 'totalInningWickets', 'daysGroup']]
X = scaler.transform(X)
chaseLookupMain['m_chaseWin%Main'] = model.predict_proba(X)[:, 1]

# last over model
trainDataLastOver = trainData.copy()
trainDataLastOver = trainDataLastOver[(trainDataLastOver['inningBallsRemaining'] < 10)] # train on 10 then predict on 7

# prepare the data
y = trainDataLastOver['chaseWin']
X_std = trainDataLastOver[['runsRequired', 'ratioRequired', 'daysGroup', 'inningBallsRemaining']]
# , 'totalInningWickets' don't use wickets as model thinks wickets dictate because of all the times teams are in charge when low wickets
scaler = StandardScaler()
scaler.fit(X_std)
X_std = scaler.transform(X_std)

# build the model
model = MLPClassifier(hidden_layer_sizes=(8, 4), random_state=42, activation='logistic', batch_size='auto', learning_rate='constant', max_iter=5000, early_stopping=False, learning_rate_init=0.001)
model.fit(X_std, y)
trainDataLastOver['m_chaseWin%LO'] = model.predict_proba(X_std)[:, 1]
trainDataLastOver = trainDataLastOver[(trainDataLastOver['inningBallsRemaining'] < 7)] # train on 12 then predict on 7


# now predict the chase situations outside of training
chaseLookupLastOver = chaseLookup.copy()
chaseLookupLastOver = chaseLookupLastOver[(chaseLookupLastOver['inningBallsRemaining'] < 7)]
X = chaseLookupLastOver[['runsRequired', 'ratioRequired', 'daysGroup', 'inningBallsRemaining']]
# , 'totalInningWickets'
# X = X.rename(columns={'runsRequired': 'runsRequiredStd', 'ratioRequired': 'ratioRequiredStd'})
X = scaler.transform(X)
chaseLookupLastOver['m_chaseWin%LO'] = model.predict_proba(X)[:, 1]

# Scaling to range [0.0001, 0.9999]
min_val, max_val = 0.0001, 0.9999
chaseLookupLastOver['m_chaseWin%LO'] = min_val + (chaseLookupLastOver['m_chaseWin%LO'] - chaseLookupLastOver['m_chaseWin%LO'].min()) * (max_val - min_val) / (chaseLookupLastOver['m_chaseWin%LO'].max() - chaseLookupLastOver['m_chaseWin%LO'].min())


# DEATH model
trainDataDeath = trainData.copy()
trainDataDeath = trainDataDeath[(trainDataDeath['inningBallsRemaining'] < 36)]

# prepare the data
y = trainDataDeath['chaseWin']
X_stdDeath = trainDataDeath[['runsRequired', 'ratioRequired', 'daysGroup', 'inningBallsRemaining']]
# , 'totalInningWickets'
scaler = StandardScaler()
scaler.fit(X_stdDeath)
X_stdDeath = scaler.transform(X_stdDeath)

# build the model
modelDeath = MLPClassifier(hidden_layer_sizes=(8, 4), random_state=42, activation='logistic', batch_size='auto', learning_rate='constant', max_iter=5000, early_stopping=False, learning_rate_init=0.001)
modelDeath.fit(X_stdDeath, y)
trainDataDeath['m_chaseWin%_Death'] = modelDeath.predict_proba(X_stdDeath)[:, 1]

# now predict the chase situations outside of training
chaseLookupDeath = chaseLookup.copy()
chaseLookupDeath = chaseLookupDeath[(chaseLookupDeath['inningBallsRemaining'] < 31)]
X = chaseLookupDeath[['runsRequired', 'ratioRequired', 'daysGroup', 'inningBallsRemaining']]
# , 'totalInningWickets'
# X = X.rename(columns={'runsRequired': 'runsRequiredStd', 'ratioRequired': 'ratioRequiredStd'})
X = scaler.transform(X)
chaseLookupDeath['m_chaseWin%_Death'] = modelDeath.predict_proba(X)[:, 1]

# Scaling to range [0.0001, 0.9999]
min_val, max_val = 0.0001, 0.9999
chaseLookupDeath['m_chaseWin%_Death'] = min_val + (chaseLookupDeath['m_chaseWin%_Death'] - chaseLookupDeath['m_chaseWin%_Death'].min()) * (max_val - min_val) / (chaseLookupDeath['m_chaseWin%_Death'].max() - chaseLookupDeath['m_chaseWin%_Death'].min())


# combine the main and the death models - this shows the three models side by side and allows us to blend them or choose between them

chaseLookupMain = chaseLookupMain.merge(chaseLookupDeath.loc[:, ['runsRequired', 'ratioRequired', 'totalInningWickets', 'daysGroup', 'inningBallsRemaining', 'm_chaseWin%_Death']], on=('runsRequired', 'ratioRequired', 'totalInningWickets', 'daysGroup', 'inningBallsRemaining'), how='left')
chaseLookupMain['m_chaseWin%'] = np.where(chaseLookupMain['inningBallsRemaining'] > 30, chaseLookupMain['m_chaseWin%Main'], np.where(chaseLookupMain['inningBallsRemaining'] < 12, chaseLookupMain['m_chaseWin%_Death'], (((chaseLookupMain['inningBallsRemaining'] - 12) / 18) * chaseLookupMain['m_chaseWin%Main']) + ((1 - ((chaseLookupMain['inningBallsRemaining'] - 12) / 18)) * chaseLookupMain['m_chaseWin%_Death'])))
chaseLookup = chaseLookupMain.merge(chaseLookupLastOver.loc[:, ['runsRequired', 'ratioRequired', 'totalInningWickets', 'daysGroup', 'inningBallsRemaining', 'm_chaseWin%LO']], on=('runsRequired', 'ratioRequired', 'totalInningWickets', 'daysGroup', 'inningBallsRemaining'), how='left')
chaseLookup['m_chaseWin%'] = np.where(chaseLookup['inningBallsRemaining'] < 7, chaseLookup[['m_chaseWin%LO', 'm_chaseWin%']].mean(axis=1), chaseLookup['m_chaseWin%'])


trainDataMain = trainDataMain.merge(trainDataDeath.loc[:, ['runsRequired', 'ratioRequired', 'totalInningWickets', 'daysGroup', 'inningBallsRemaining', 'm_chaseWin%_Death']], on=('runsRequired', 'ratioRequired', 'totalInningWickets', 'daysGroup', 'inningBallsRemaining'), how='left')
trainDataMain['m_chaseWin%'] = np.where(trainDataMain['inningBallsRemaining'] > 30, trainDataMain['m_chaseWin%Main'], np.where(trainDataMain['inningBallsRemaining'] < 12, trainDataMain['m_chaseWin%_Death'], (((trainDataMain['inningBallsRemaining'] - 12) / 18) * trainDataMain['m_chaseWin%Main']) + ((1 - ((trainDataMain['inningBallsRemaining'] - 12) / 18)) * trainDataMain['m_chaseWin%_Death'])))
trainData = trainDataMain.merge(trainDataLastOver.loc[:, ['runsRequired', 'ratioRequired', 'totalInningWickets', 'daysGroup', 'inningBallsRemaining', 'm_chaseWin%LO']], on=('runsRequired', 'ratioRequired', 'totalInningWickets', 'daysGroup', 'inningBallsRemaining'), how='left')
trainData['m_chaseWin%'] = np.where(trainData['inningBallsRemaining'] < 7, trainData[['m_chaseWin%LO', 'm_chaseWin%']].mean(axis=1), trainData['m_chaseWin%'])


# order correctly for illogical situations
cols = chaseLookup.loc[:, ['totalInningWickets', 'runsRequired', 'inningBallsRemaining', 'm_chaseWin%']]
colsWrong = cols.sort_values(by=['totalInningWickets', 'runsRequired', 'inningBallsRemaining'], axis=0).reset_index(drop=True)
colsRight = cols.sort_values(by=['totalInningWickets', 'runsRequired', 'm_chaseWin%'], axis=0).reset_index(drop=True)
colsWrong['m_chaseWin%'] = colsRight['m_chaseWin%']
colsWrong = colsWrong.sort_values(by=['inningBallsRemaining', 'runsRequired', 'totalInningWickets'], axis=0).reset_index(drop=True)
chaseLookup['m_chaseWin%'] = colsWrong['m_chaseWin%']

# add in an identifier/lookup column
chaseLookup['state_id'] = (
    chaseLookup['totalInningWickets']
    + (chaseLookup['inningBallsRemaining'] / 1000)
    + (chaseLookup['runsRequired'] / 1_000_000)
).round(6)



# some checks, the below doesn't affect the model
# bias check - scored only on states with enough underlying ball/wicket data to trust (full trainData is still used for fitting above)
bias = pd.pivot_table(trainData[trainData['ballWicketSample'] >= 100], values=['m_chaseWin%', 'chaseWin', 'sample'], aggfunc='sum', index=['totalInningWickets']).reset_index()
bias['bias'] = bias['m_chaseWin%'] / bias['chaseWin']
bias['win%'] = bias['chaseWin'] / bias['sample']

# # compare
# chaseLookup = chaseLookup.merge(chaseLookupLive.loc[:, ['m_chaseWin%', 'totalInningWickets', 'runsRequired', 'inningBallsRemaining']],
#                                 how='left', on=['totalInningWickets', 'runsRequired', 'inningBallsRemaining'], suffixes=('', 'Live'))
# chaseLookup['m_diff'] = chaseLookup['m_chaseWin%'] - chaseLookup['m_chaseWin%Live']


# chase win % year
years = pd.pivot_table(trainData, index=['totalInningWickets', 'runsRequired', 'inningBallsRemaining'], values=['m_chaseWin%'], aggfunc='mean').reset_index()
chaseLookup = chaseLookup.merge(years, how='left', on=['totalInningWickets', 'runsRequired', 'inningBallsRemaining'], suffixes=('', 'Year'))

# insert lookup column for inserting into RAS
col_position = chaseLookup.columns.get_loc('m_chaseWin%')  # gets index of 'B'
chaseLookup.insert(col_position, 'lookup', (chaseLookup['totalInningWickets'] + (chaseLookup['inningBallsRemaining'] / 1000) + (chaseLookup['runsRequired'] / 1000000)).round(6))

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
#     diff = pd.pivot_table(graph_data, index='runsRequired', columns='inningBallsRemaining', values='chaseSample', aggfunc='mean')
#     # plot in a heatmap
#     sns.heatmap(ax=axes[x, 0], data=actual, cmap=plt.cm.get_cmap('PiYG', 1000), vmin=0, vmax=1, center=0.5, xticklabels=10, yticklabels=10)
#     sns.heatmap(ax=axes[x, 1], data=old, cmap=plt.cm.get_cmap('PiYG', 1000), vmin=0, vmax=1, center=0.5, xticklabels=10, yticklabels=10)
#     sns.heatmap(ax=axes[x, 2], data=new, cmap=plt.cm.get_cmap('PiYG', 1000), vmin=0, vmax=1, center=0.5, xticklabels=10, yticklabels=10)
#     sns.heatmap(ax=axes[x, 3], data=diff, cmap=plt.cm.get_cmap('PiYG', 1000), vmin=0, vmax=500, center=62, xticklabels=10, yticklabels=10)
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






# # over/under performance heatmap
# # x = daysGroup, split into 0.04-year steps
# # y = balls remaining
# # colour = actual chase win% - expected chase win%
#
# heat_data = trainData.copy()
# heat_data = heat_data.drop(columns=['m_chaseWin%'])
# heat_data = heat_data.merge(chaseLookup.loc[:, ['inningBallsRemaining', 'totalInningWickets', 'runsRequired', 'm_chaseWin%']], how='left', on=['inningBallsRemaining', 'totalInningWickets', 'runsRequired'])
# heat_data = heat_data.dropna(
#     subset=[
#         'daysGroup',
#         'inningBallsRemaining',
#         'chaseWin',
#         'm_chaseWin%'
#     ]
# )
#
# year_centres = np.arange(
#     np.floor(heat_data['daysGroup'].min()),
#     np.ceil(heat_data['daysGroup'].max()) + 0.04,
#     0.04
# )
#
# ball_centres = np.arange(1, 121, 1)
#
# rows = []
#
# for yc in year_centres:
#     for bc in ball_centres:
#         mask = (
#             (heat_data['daysGroup'].sub(yc).abs() <= 0.5) &
#             (heat_data['inningBallsRemaining'].sub(bc).abs() <= 5)
#         )
#
#         cell = heat_data.loc[mask]
#
#         sample = len(cell)
#
#         if sample < 100:
#             actual = np.nan
#             expected = np.nan
#             over_under = np.nan
#         else:
#             actual = cell['chaseWin'].mean()
#             expected = cell['m_chaseWin%'].mean()
#             over_under = actual - expected
#
#         rows.append({
#             'daysGroupCentre': yc,
#             'yearLabel': 2015 + yc,
#             'inningBallsRemaining': bc,
#             'sample': sample,
#             'actual_chaseWin%': actual,
#             'expected_chaseWin%': expected,
#             'over_under': over_under
#         })
#
# heatmap_df = pd.DataFrame(rows)
#
# heatmap_pivot = heatmap_df.pivot(
#     index='inningBallsRemaining',
#     columns='daysGroupCentre',
#     values='over_under'
# )
#
# plt.figure(figsize=(16, 10))
#
# sns.heatmap(
#     heatmap_pivot,
#     cmap='RdYlGn',
#     center=0,
#     vmin=-0.08,
#     vmax=0.08,
#     linewidths=0,
#     cbar_kws={'label': 'Actual chase win% - expected chase win%'}
# )
#
# plt.title('Women chasing over/under performance by year and balls remaining')
# plt.xlabel('Year')
# plt.ylabel('Balls remaining')
#
# xtick_positions = np.arange(0, len(year_centres), 25)
# xtick_labels = [
#     str(int(2015 + year_centres[pos]))
#     for pos in xtick_positions
# ]
#
# plt.xticks(
#     xtick_positions,
#     xtick_labels,
#     rotation=0
# )
#
# plt.gca().invert_yaxis()
#
# plt.tight_layout()
# plt.show()




# exports
chaseLookup.to_csv(PROJECT_ROOT / 'men/matchMarket/outputs/1_chaseLookup.csv', index=False)
#
# chaseLookupComparison = chaseLookup.loc[:, ['inningBallNumber', 'inningBallsRemaining', 'totalInningWickets', 'runsRequired', 'chaseSample', 'chaseWin%', 'm_chaseWin%Main', 'm_chaseWin%_Death', 'm_chaseWin%LO', 'm_chaseWin%', 'm_chaseWin%Year']]

trainData = trainData.loc[:, ['matchID', 'ID', 'm_chaseWin%']]
trainData.to_csv(PROJECT_ROOT / 'men/matchMarket/outputs/neuralPreds.csv', index=False)


