import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from paths import PROJECT_ROOT


# import
trainData = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/data/dataClean.csv', parse_dates=['date'])
firstInningsLookup = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/outputs/5_masterLookup.csv')


# keep second innings and last 4 overs
trainData = trainData[trainData['inningBallsRemaining'] <= 24]
trainData = trainData[trainData['inningNumber'] == 2]


# required runs per ball
trainData['runsRequiredPerBall'] = trainData['runsRequired'] / trainData['inningBallsRemaining']


# estimated scoring ability by wickets lost
runsPerBall = {
    0: 1.85,
    1: 1.82,
    2: 1.78,
    3: 1.72,
    4: 1.65,
    5: 1.55,
    6: 1.43,
    7: 1.30,
    8: 1.15,
    9: 0.95
}


# create approximately 5,000 chase situations
situations = []

for inningBallsRemaining in range(1, 25):
    for totalInningWickets in range(10):

        centreRunsRequired = inningBallsRemaining * runsPerBall[totalInningWickets]
        halfWidth = 5.5 + (0.35 * inningBallsRemaining)

        minRunsRequired = max(1, int(np.floor(centreRunsRequired - halfWidth)))
        maxRunsRequired = int(np.ceil(centreRunsRequired + halfWidth))

        if inningBallsRemaining <= 6:
            absoluteMaxRunsRequired = inningBallsRemaining * 6
        else:
            absoluteMaxRunsRequired = inningBallsRemaining * 3

        maxRunsRequired = min(maxRunsRequired, absoluteMaxRunsRequired)

        for runsRequired in range(minRunsRequired, maxRunsRequired + 1):
            situations.append({
                'runsRequired': runsRequired,
                'totalInningWickets': totalInningWickets,
                'inningBallsRemaining': inningBallsRemaining
            })


masterLookup = pd.DataFrame(situations)


# create variables for each actual chase situation
masterLookup['inningBallNumber'] = 121 - masterLookup['inningBallsRemaining']
masterLookup['overNumber'] = np.ceil(masterLookup['inningBallNumber'] / 6).astype(int)
masterLookup['inningNumber'] = 2
masterLookup['runsRequiredPerBall'] = masterLookup['runsRequired'] / masterLookup['inningBallsRemaining']


print('Total situations:', len(masterLookup))


# train and target
X = trainData[
    [
        'inningBallNumber',
        'runsRequiredPerBall',
        'totalInningWickets'
    ]
]

y = trainData['batsmanRuns']


# polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)

X_poly = poly.fit_transform(X)


# scale polynomial features
scaler = StandardScaler()

X_poly_scaled = scaler.fit_transform(X_poly)


# define and fit model
model = LogisticRegression(max_iter=10000, random_state=42)

model.fit(X_poly_scaled, y)


# predict every chase situation
X_pred = masterLookup[
    [
        'inningBallNumber',
        'runsRequiredPerBall',
        'totalInningWickets'
    ]
]

X_pred_poly = poly.transform(X_pred)

X_pred_poly_scaled = scaler.transform(X_pred_poly)

y_pred = model.predict_proba(X_pred_poly_scaled)


# add chase outcome probabilities
probabilityColumns = [str(run) for run in model.classes_]

probabilities = pd.DataFrame(
    y_pred,
    columns=probabilityColumns,
    index=masterLookup.index
)

masterLookup = pd.concat([masterLookup, probabilities], axis=1)


# expected batsman runs from chase model
masterLookup['m_batsmanRunsBall'] = sum(
    int(run) * masterLookup[str(run)]
    for run in model.classes_
)


# rename chase probabilities before comparison
chaseRename = {
    '0': 'chase_0',
    '1': 'chase_1',
    '2': 'chase_2',
    '3': 'chase_3',
    '4': 'chase_4',
    '5': 'chase_5',
    '6': 'chase_6',
    '7': 'chase_7',
    'm_batsmanRunsBall': 'chase_m_batsmanRunsBall'
}

masterLookup = masterLookup.rename(columns=chaseRename)


# keep first innings probabilities needed for comparison
firstInningsColumns = [
    'totalInningWickets',
    'inningBallNumber',
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    'm_batsmanRunsBall'
]

firstInningsLookup = firstInningsLookup[firstInningsColumns]
firstInningsLookup = firstInningsLookup.drop_duplicates(subset=['totalInningWickets', 'inningBallNumber'])


# rename first innings probabilities
firstInningsRename = {
    '0': 'first_0',
    '1': 'first_1',
    '2': 'first_2',
    '3': 'first_3',
    '4': 'first_4',
    '5': 'first_5',
    '6': 'first_6',
    '7': 'first_7',
    'm_batsmanRunsBall': 'first_m_batsmanRunsBall'
}

firstInningsLookup = firstInningsLookup.rename(columns=firstInningsRename)


# merge first innings probabilities onto chase situations
masterLookup = masterLookup.merge(
    firstInningsLookup,
    how='left',
    on=['totalInningWickets', 'inningBallNumber']
)


# create comparison columns
for run in model.classes_:
    run = str(run)

    masterLookup[f'diff_{run}'] = masterLookup[f'chase_{run}'] - masterLookup[f'first_{run}']


masterLookup['diff_m_batsmanRunsBall'] = masterLookup['chase_m_batsmanRunsBall'] - masterLookup['first_m_batsmanRunsBall']


# comparison table
comparisonColumns = [
    'runsRequired',
    'totalInningWickets',
    'inningBallsRemaining',
    'runsRequiredPerBall',
    'first_m_batsmanRunsBall',
    'chase_m_batsmanRunsBall',
    'diff_m_batsmanRunsBall'
]

for run in model.classes_:
    run = str(run)

    comparisonColumns += [
        f'first_{run}',
        f'chase_{run}',
        f'diff_{run}'
    ]


comparisonTable = masterLookup[comparisonColumns]


