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
trainData = pd.read_csv(PROJECT_ROOT / 'women/expBall&runsToCome/data/dataClean_w.csv', parse_dates=['date'])
trainData = trainData[trainData['competition'] != "The Hundred (Women's Comp)"]
# trainData = trainData[(trainData['competition'] != "Women's Big Bash League") | (trainData['date'] < pd.Timestamp(2020, 6, 6))]
masterLookup = pd.read_csv(PROJECT_ROOT / 'women/expBall&runsToCome/outputs/5_masterLookup_w.csv')
chaseSituations = pd.read_csv(PROJECT_ROOT / 'women/matchMarket/auxiliaries/chaseSituationBuilder_w.csv')
# chaseLookupLive = pd.read_csv(PROJECT_ROOT / 'women/matchMarket/outputs/1_chaseLookupLive_w.csv')



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
masterLookup = masterLookup[(masterLookup['ord'] == 1) & (masterLookup['daysGroup'] == 10)]
# merge in runs to come
trainData = trainData.merge(masterLookup.loc[:, ['totalInningRunsToComeSimBiasSplineYear', 'totalInningWickets', 'inningBallNumber', 'totalInningValidBallsFacedToCome', 'bowledOut']], how='left', on=['totalInningWickets', 'inningBallNumber'])
# create a ratio of runs to come to be used as a predictor, drop any nans
trainData['ratioRequired'] = trainData['runsRequired'] / trainData['totalInningRunsToComeSimBiasSplineYear'] # changing to include year
trainData['ratioRequiredStd'] = trainData['runsRequiredStd'] / trainData['totalInningRunsToComeSimBiasSplineYear']
trainData = trainData.dropna(axis=0, subset=['ratioRequired'])



# create an empty dataframe
chaseLookup = pd.pivot_table(trainData, values=['sample', 'chaseWin', 'totalInningRunsToCome', 'totalInningWicketsToCome', 'runsRequiredStd'],
                            index=['totalInningWickets', 'inningBallNumber', 'runsRequired'],
                            aggfunc={'sample': 'sum', 'chaseWin': 'sum', 'totalInningRunsToCome': 'mean', 'totalInningWicketsToCome': 'mean', 'runsRequiredStd': 'mean'}).reset_index()
chaseLookup['chaseWin%'] = chaseLookup['chaseWin'] / chaseLookup['sample']
chaseLookup = chaseSituations.merge(chaseLookup, how='left', on=['totalInningWickets', 'inningBallNumber', 'runsRequired'])
chaseLookup = chaseLookup.rename(columns={'sample': 'chaseSample'})
chaseLookup = chaseLookup.merge(masterLookup.loc[:, ['totalInningWickets', 'inningBallNumber', 'sample', 'totalInningRunsToComeSimBiasSplineYear', 'totalInningValidBallsFacedToCome', 'bowledOut']], how='left', on=['totalInningWickets', 'inningBallNumber'])
chaseLookup = chaseLookup.rename(columns={'sample': 'ballWicketSample'})
chaseLookup['ratioRequired'] = chaseLookup['runsRequired'] / chaseLookup['totalInningRunsToComeSimBiasSplineYear'] # changing to include year
chaseLookup['ratioRequiredStd'] = chaseLookup['runsRequiredStd'] / chaseLookup['totalInningRunsToComeSimBiasSplineYear'] # changing to include year
chaseLookup['daysGroup'] = 11.7

chaseLookup = chaseLookup.dropna(axis=0, subset=['totalInningRunsToComeSimBiasSplineYear']).reset_index(drop=True)
chaseLookup['in'] = 1


# #
# medians = pd.pivot_table(trainData, index=['inningBallsRemaining', 'totalInningWickets', 'runsRequired'], values=['daysGroup'], aggfunc='median').reset_index()
# medians = medians.rename(columns={'daysGroup': 'daysGroupMedian'})
# chaseLookup = chaseLookup.merge(medians, how='left', on=['inningBallsRemaining', 'totalInningWickets', 'runsRequired'])
# chaseLookup['daysGroup'] = chaseLookup['daysGroupMedian'].fillna(12)

# remove chases which are effectively lost
trainData = trainData.merge(chaseLookup.loc[:, ['in', 'totalInningWickets', 'runsRequired', 'inningBallNumber']], how='left', on=['totalInningWickets', 'runsRequired', 'inningBallNumber'])
trainData = trainData[trainData['in'] == 1]





# start of innings model
trainDataMain = trainData.copy()
trainDataMain = trainDataMain[(trainDataMain['inningBallsRemaining'] > 1)]

# prepare the data
y = trainDataMain['chaseWin']
X_std = trainDataMain[['runsRequired', 'ratioRequired', 'totalInningWickets', 'daysGroup']]
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
X = chaseLookupMain[['runsRequired', 'ratioRequired', 'totalInningWickets', 'daysGroup']]
X = scaler.transform(X)
chaseLookupMain['m_chaseWin%'] = model.predict_proba(X)[:, 1]
trainDataMain = trainDataMain[(trainDataMain['inningBallsRemaining'] > 6)]








# last over model
trainDataLastOver = trainData.copy()
trainDataLastOver = trainDataLastOver[(trainDataLastOver['inningBallsRemaining'] < 7)]

# prepare the data
y = trainDataLastOver['chaseWin']
X_std = trainDataLastOver[['runsRequiredStd', 'ratioRequiredStd', 'totalInningWickets', 'daysGroup', 'inningBallsRemaining']]
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
X = chaseLookupLastOver[['runsRequiredStd', 'ratioRequiredStd', 'totalInningWickets', 'daysGroup', 'inningBallsRemaining']]
X = scaler.transform(X)
chaseLookupLastOver['m_chaseWin%'] = model.predict_proba(X)[:, 1]

# Scaling to range [0.0001, 0.9999]
min_val, max_val = 0.0001, 0.9999
chaseLookupLastOver['m_chaseWin%'] = min_val + (chaseLookupLastOver['m_chaseWin%'] - chaseLookupLastOver['m_chaseWin%'].min()) * (max_val - min_val) / (chaseLookupLastOver['m_chaseWin%'].max() - chaseLookupLastOver['m_chaseWin%'].min())




# DEATH model
trainDataDeath = trainData.copy()
trainDataDeath = trainDataDeath[(trainDataDeath['inningBallsRemaining'] < 31)]

# prepare the data
y = trainDataDeath['chaseWin']
X_stdDeath = trainDataDeath[['runsRequiredStd', 'ratioRequiredStd', 'totalInningWickets', 'daysGroup', 'inningBallsRemaining']]
scaler = StandardScaler()
scaler.fit(X_stdDeath)
X_stdDeath = scaler.transform(X_stdDeath)

# build the model
modelDeath = MLPClassifier(hidden_layer_sizes=(8, 4), random_state=42, activation='logistic', batch_size='auto', learning_rate='constant', max_iter=5000, early_stopping=False, learning_rate_init=0.001)
modelDeath.fit(X_stdDeath, y)
trainDataDeath['m_chaseWin%'] = modelDeath.predict_proba(X_stdDeath)[:, 1]

# now predict the chase situations outside of training
chaseLookupDeath = chaseLookup.copy()
chaseLookupDeath = chaseLookupDeath[(chaseLookupDeath['inningBallsRemaining'] < 31)]
X = chaseLookupDeath[['runsRequiredStd', 'ratioRequiredStd', 'totalInningWickets', 'daysGroup', 'inningBallsRemaining']]
X = scaler.transform(X)
chaseLookupDeath['m_chaseWin%'] = modelDeath.predict_proba(X)[:, 1]

# Scaling to range [0.0001, 0.9999]
min_val, max_val = 0.0001, 0.9999
chaseLookupDeath['m_chaseWin%'] = min_val + (chaseLookupDeath['m_chaseWin%'] - chaseLookupDeath['m_chaseWin%'].min()) * (max_val - min_val) / (chaseLookupDeath['m_chaseWin%'].max() - chaseLookupDeath['m_chaseWin%'].min())




# combine the 3 models
chaseLookup = pd.concat([chaseLookupLastOver, chaseLookupMain], axis=0).reset_index(drop=True)
trainData = pd.concat([trainDataLastOver, trainDataMain], axis=0).reset_index(drop=True)


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
# bias check
bias = pd.pivot_table(trainData, values=['m_chaseWin%', 'chaseWin', 'sample'], aggfunc='sum', index=['totalInningWickets']).reset_index()
bias['bias'] = bias['m_chaseWin%'] / bias['chaseWin']
bias['win%'] = bias['chaseWin'] / bias['sample']

# # compare
# chaseLookup = chaseLookup.merge(chaseLookupLive.loc[:, ['m_chaseWin%old', 'm_chaseWin%', 'totalInningWickets', 'runsRequired', 'inningBallsRemaining']],
#                                 how='left', on=['totalInningWickets', 'runsRequired', 'inningBallsRemaining'], suffixes=('', 'Live'))
# chaseLookup['m_diff'] = chaseLookup['m_chaseWin%'] - chaseLookup['m_chaseWin%Live']
# chaseLookup['m_diff_old'] = chaseLookup['m_chaseWin%'] - chaseLookup['m_chaseWin%old']


# chase win % year
years = pd.pivot_table(trainData, index=['totalInningWickets', 'runsRequired', 'inningBallsRemaining'], values=['m_chaseWin%'], aggfunc='mean').reset_index()
chaseLookup = chaseLookup.merge(years, how='left', on=['totalInningWickets', 'runsRequired', 'inningBallsRemaining'], suffixes=('', 'Year'))

#insert lookup column for inserting into RAS
col_position = chaseLookup.columns.get_loc('m_chaseWin%')
chaseLookup.insert(col_position, 'lookup', (chaseLookup['totalInningWickets'] + (chaseLookup['inningBallsRemaining'] / 1000) + (chaseLookup['runsRequired'] / 1000000)).round(6))

# # use this to get the target win% in distribs, 51.30%
# yearswins = pd.pivot_table(trainData, index=['totalInningWickets', 'inningBallsRemaining'], values=['chaseWin'], aggfunc='mean').reset_index()



# # # graph of predictions
# # fig, axes = plt.subplots(10, 4, figsize=(20, 40))           # create a figure of dimension 10 (Wickets) by 5 (number of graphs for each wicket)
# # for x in np.arange(0, 10, 1):                               # loop 0-10 for wickets
# #     graph_data = chaseLookup.copy()
# #     graph_data = graph_data[graph_data['totalInningWickets'] == x]       # filter the dataframe for the wicket in question
# #     # graph_data['chase_adj%'] = graph_data['blendr_win%'] - graph_data['X_win%']
# #     # create tables of the numbers to be plotted
# #     actual = pd.pivot_table(graph_data, index='runsRequired', columns='inningBallsRemaining', values='chaseWin%', aggfunc='mean')
# #     old = pd.pivot_table(graph_data, index='runsRequired', columns='inningBallsRemaining', values='m_chaseWin%', aggfunc='mean')
# #     new = pd.pivot_table(graph_data, index='runsRequired', columns='inningBallsRemaining', values='m_chaseWin%', aggfunc='mean')
# #     diff = pd.pivot_table(graph_data, index='runsRequired', columns='inningBallsRemaining', values='m_diff', aggfunc='mean')
# #     # plot in a heatmap
# #     sns.heatmap(ax=axes[x, 0], data=actual, cmap=plt.cm.get_cmap('PiYG', 1000), vmin=0, vmax=1, center=0.5, xticklabels=10, yticklabels=10)
# #     sns.heatmap(ax=axes[x, 1], data=old, cmap=plt.cm.get_cmap('PiYG', 1000), vmin=0, vmax=1, center=0.5, xticklabels=10, yticklabels=10)
# #     sns.heatmap(ax=axes[x, 2], data=new, cmap=plt.cm.get_cmap('PiYG', 1000), vmin=0, vmax=1, center=0.5, xticklabels=10, yticklabels=10)
# #     sns.heatmap(ax=axes[x, 3], data=diff, cmap=plt.cm.get_cmap('PiYG', 1000), vmin=-0.2, vmax=0.2, center=0, xticklabels=10, yticklabels=10)
# #
# #     # set titles for each graph
# #     title1 = f"actual_win% - {x} wickets lost"
# #     axes[x, 0].set_title(title1)
# #     title2 = f"old - {x} wickets lost"
# #     axes[x, 1].set_title(title2)
# #     title3 = f"new {x} wickets lost"
# #     axes[x, 2].set_title(title3)
# #     title4 = f"diff - {x} wickets lost"
# #     axes[x, 3].set_title(title4)
# #     # title5 = f"blendr_win%_ - {x} wickets lost"
# #     # axes[x, 4].set_title(title4)
# # plt.tight_layout()
# # plt.show()
#
# # over/under performance heatmap
# # x = daysGroup, split into 0.04-year steps
# # y = balls remaining
# # colour = actual chase win% - expected chase win%
#
#
# heat_data = trainData.copy()
# heat_data = heat_data.drop(columns=['m_chaseWin%'])
# heat_data = heat_data.merge(chaseLookup.loc[:, ['inningBallsRemaining', 'totalInningWickets', 'runsRequired', 'm_chaseWin%']], how='left', on=['inningBallsRemaining', 'totalInningWickets', 'runsRequired'])
#
#
# # medians = pd.pivot_table(trainData, index=['inningBallsRemaining', 'totalInningWickets', 'runsRequired'], values=['daysGroup'], aggfunc='median').reset_index()
# # medians = medians.rename(columns={'daysGroup': 'daysGroupMedian'})
# # heat_data = heat_data.merge(medians, how='left', on=['inningBallsRemaining', 'totalInningWickets', 'runsRequired'])
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
#             (heat_data['inningBallsRemaining'].sub(bc).abs() <= 10)
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
#     cmap='PiYG',
#     center=0,
#     vmin=-0.2,
#     vmax=0.2,
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
chaseLookup.to_csv(PROJECT_ROOT / 'women/matchMarket/outputs/1_chaseLookup_w.csv', index=False)


