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
trainData = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/data/dataClean1st.csv', parse_dates=['date'])
masterLookup = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/outputs/5_masterLookup.csv')
chaseSituations = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/auxiliaries/chaseSituationBuilder.csv')

# set runs required to efftarget
trainData['runsRequired'] = trainData['effTarget']
trainData = trainData.dropna(axis=0, subset=['runsRequired', 'totalInningWickets', 'inningBallsRemaining'])


# we only want innings 1 for the training and shuffle the data
trainData = trainData[trainData['inningNumber'] == 1]

runs_required_spread = trainData.groupby(['inningBallsRemaining', 'totalInningWickets'], as_index=False).agg(
    minRunsRequired=('runsRequired', 'min'),
    maxRunsRequired=('runsRequired', 'max')
)

runs_required_spread['minRunsRequired'] = np.floor(runs_required_spread['minRunsRequired']).astype(int)
runs_required_spread['maxRunsRequired'] = np.ceil(runs_required_spread['maxRunsRequired']).astype(int)
runs_required_spread['inningBallsRemaining'] = runs_required_spread['inningBallsRemaining'].astype(int)
runs_required_spread['totalInningWickets'] = runs_required_spread['totalInningWickets'].astype(int)

chaseSituationsRows = []

for _, row in runs_required_spread.iterrows():
    for runs in range(int(row['minRunsRequired']), int(row['maxRunsRequired']) + 1):
        chaseSituationsRows.append({
            'inningBallsRemaining': int(row['inningBallsRemaining']),
            'inningBallNumber': int(121 - row['inningBallsRemaining']),
            'totalInningWickets': int(row['totalInningWickets']),
            'runsRequired': runs
        })

chaseSituations = pd.DataFrame(chaseSituationsRows)
chaseSituations = chaseSituations.sort_values(by=['inningBallsRemaining', 'runsRequired', 'totalInningWickets']).reset_index(drop=True)
trainData = trainData.sample(frac=1, random_state=42).reset_index(drop=True)


# we only want innings 1 for the training and shuffle the data
trainData = trainData[trainData['inningNumber'] == 1]
trainData = trainData.sample(frac=1, random_state=42).reset_index(drop=True)
# we need to remove duplicates in runs to come so just select batting order 1 and daysgroup 11
masterLookup = masterLookup[(masterLookup['ord'] == 1) & (masterLookup['daysGroup'] == 11)]

# merge in predicted runs to come
trainData = trainData.merge(masterLookup.loc[:, ['totalInningRunsToComeSimBiasSpline', 'totalInningWickets', 'inningBallNumber', 'totalInningValidBallsFacedToCome', 'bowledOut']], how='left', on=['totalInningWickets', 'inningBallNumber'])




# create an empty dataframe
chaseLookup = pd.pivot_table(trainData, values=['sample', 'result', 'totalInningRunsToCome', 'totalInningWicketsToCome'],
                            index=['totalInningWickets', 'inningBallNumber', 'runsRequired'],
                            aggfunc={'sample': 'sum', 'result': 'sum', 'totalInningRunsToCome': 'mean', 'totalInningWicketsToCome': 'mean'}).reset_index()
chaseLookup['result%'] = chaseLookup['result'] / chaseLookup['sample']
chaseLookup = chaseSituations.merge(chaseLookup, how='left', on=['totalInningWickets', 'inningBallNumber', 'runsRequired'])
chaseLookup = chaseLookup.rename(columns={'sample': 'chaseSample'})
chaseLookup = chaseLookup.merge(masterLookup.loc[:, ['totalInningWickets', 'inningBallNumber', 'sample', 'totalInningRunsToComeSimBiasSpline', 'totalInningValidBallsFacedToCome', 'bowledOut']], how='left', on=['totalInningWickets', 'inningBallNumber'])
chaseLookup = chaseLookup.rename(columns={'sample': 'ballWicketSample'})

chaseLookup['ratioRequired'] = chaseLookup['runsRequired'] / chaseLookup['totalInningRunsToComeSimBiasSpline']
chaseLookup['daysGroup'] = 11
chaseLookup = chaseLookup.dropna(axis=0, subset=['totalInningRunsToComeSimBiasSpline']).reset_index(drop=True)



# model, train on all data
trainDataMain = trainData.copy()

# prepare the data
y = trainDataMain['result']
X_std = trainDataMain[['runsRequired', 'totalInningWickets', 'inningBallsRemaining', 'daysGroup']]
scaler = StandardScaler()
scaler.fit(X_std)
X_std = scaler.transform(X_std)

# build the model
model = MLPClassifier(hidden_layer_sizes=(8, 4), random_state=42, activation='logistic', batch_size='auto', learning_rate='constant', max_iter=5000, early_stopping=False, learning_rate_init=0.001)
model.fit(X_std, y)
trainDataMain['m_result%'] = model.predict_proba(X_std)[:, 1]

# now predict the chase situations outside of training
X = chaseLookup[['runsRequired', 'totalInningWickets', 'inningBallsRemaining', 'daysGroup']]
X = scaler.transform(X)
chaseLookup['m_result%'] = model.predict_proba(X)[:, 1]







# # exports
chaseLookup.to_csv(PROJECT_ROOT / 'men/matchMarket/outputs/1_chaseLookup1st.csv', index=False)





