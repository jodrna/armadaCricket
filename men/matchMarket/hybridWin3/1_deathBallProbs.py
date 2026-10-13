import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
import time
from itertools import combinations
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from paths import PROJECT_ROOT

modelStartTime = time.perf_counter()
THREADS = 4
PARALLEL_FITS = 2
N_FOLDS = 5
FOLD_SEED = 2718
RANDOM_SEED = 42
stateColumns = ['runsRequired', 'totalInningWickets', 'inningBallsRemaining', 'ord', 'nonStrikerOrd']
ballColumns = stateColumns[:4]
rowColumns = ['matchID', 'ID', 'fold']
featureColumns = [
    'runsRequired', 'pressure', 'logPressure', 'inningBallsRemaining',
    'totalInningWickets', 'ord', 'ballsInOver', 'orderGap', 'newestBatter',
    'tailOrder', 'runsPerWicket', 'pressureOrder', 'pressurePerWicket',
    'ballsWickets', 'ballsOrder', 'powerplay', 'death', 'logBalls',
]

# --------------------------------------------------
# ONE DATA SOURCE; THE SAME WHOLE-MATCH FOLDS THROUGHOUT
# --------------------------------------------------
trainData = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/data/dataClean.csv')
trainData = trainData.loc[trainData['inningNumber'].eq(2)].copy()
conflictingMatches = trainData.groupby('matchID')['chaseWin'].nunique()
conflictingMatches = conflictingMatches.index[conflictingMatches.gt(1)]
fractionalMatches = trainData.loc[
    trainData['runsRequired'].notna()
    & ~np.isclose(trainData['runsRequired'], trainData['runsRequired'].round(), atol=1e-8, rtol=0),
    'matchID',
]
excludedMatches = set(conflictingMatches) | set(fractionalMatches)
trainData = trainData.loc[~trainData['matchID'].isin(excludedMatches)].copy()
trainData = trainData.dropna(subset=['matchID', 'ID', 'runsRequired', 'inningBallsRemaining', 'totalInningWickets', 'ord', 'chaseWin'])
trainData = trainData.loc[
    trainData['runsRequired'].gt(0)
    & trainData['inningBallsRemaining'].between(1, 120)
    & trainData['totalInningWickets'].between(0, 9)
    & trainData['ord'].between(1, 11)
    & trainData['chaseWin'].isin([0, 1])
].copy()
trainData = trainData.reset_index(drop=True)
if trainData.duplicated(['matchID', 'ID']).any():
    raise ValueError('dataClean.csv must contain one row per matchID/ID.')
matches = trainData[['matchID', 'chaseWin']].drop_duplicates().sort_values('matchID').reset_index(drop=True)
matches['fold'] = -1
for fold, (_, testMatches) in enumerate(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED).split(matches, matches['chaseWin'])):
    matches.loc[testMatches, 'fold'] = fold
matchFolds = matches.set_index('matchID')['fold']
trainData['fold'] = trainData['matchID'].map(matchFolds).astype(int)
dataHash = hashlib.sha256((PROJECT_ROOT / 'men/matchMarket/hybridWin3/data/dataClean.csv').read_bytes()).hexdigest()

runInfo = pd.DataFrame([{
    'dataHash': dataHash, 'nFolds': N_FOLDS, 'foldSeed': FOLD_SEED,
    'modelSeed': RANDOM_SEED, 'pipelineVersion': 'full_remodel_v3', 'maxBalls': 120,
    'gridHash': hashlib.sha256((PROJECT_ROOT / 'men/matchMarket/hybridWin3/auxiliaries/1_chaseLookup.csv').read_bytes()).hexdigest(), 'ballReady': 0, 'simulationReady': 0,
    'directReady': 0, 'blendReady': 0,
}])
runInfo.to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/cv/runInfo.csv', index=False)

# --------------------------------------------------
# BALL-OUTCOME TRAINING DATA: ALL 20 OVERS
# --------------------------------------------------
# The supplied dataClean.csv already contains legal deliveries only.
trainData = trainData.loc[trainData['inningBallsRemaining'].between(1, 120)].copy()
trainData = trainData.dropna(subset=['batsmanRuns', 'isWicket'])
trainData = trainData.loc[
    trainData['batsmanRuns'].isin(range(7))
    & trainData['isWicket'].isin([0, 1])
    & ~(trainData['batsmanRuns'].eq(6) & trainData['isWicket'].eq(1))
].copy().reset_index(drop=True)
jointEvents = np.array([(runs, wicket) for runs in range(7) for wicket in range(2) if (runs, wicket) != (6, 1)], dtype=np.int64)
jointColumns = [f'pressure_{runs}_' + ('wicket' if wicket else 'noWicket') for runs, wicket in jointEvents]
jointEventLookup = {tuple(event): position for position, event in enumerate(jointEvents)}
trainData['jointTarget'] = [jointEventLookup[(int(runs), int(wicket))] for runs, wicket in zip(trainData['batsmanRuns'], trainData['isWicket'])]

# --------------------------------------------------
# PRESERVE THE FINAL-24-BALL GRID; EXTEND TO 120 BALLS
# --------------------------------------------------
runsPerBall = {
    0: 1.85, 1: 1.82, 2: 1.78, 3: 1.72, 4: 1.65,
    5: 1.55, 6: 1.43, 7: 1.30, 8: 1.15, 9: 0.95,
}
situations = []
for inningBallsRemaining in range(1, 25):
    for totalInningWickets in range(10):
        centreRunsRequired = inningBallsRemaining * runsPerBall[totalInningWickets]
        halfWidth = 5.5 + 0.35 * inningBallsRemaining
        minRunsRequired = max(1, int(np.floor(centreRunsRequired - halfWidth)))
        maxRunsRequired = int(np.ceil(centreRunsRequired + halfWidth))
        absoluteMaxRunsRequired = inningBallsRemaining * (6 if inningBallsRemaining <= 6 else 3)
        maxRunsRequired = min(maxRunsRequired, absoluteMaxRunsRequired)
        for runsRequired in range(minRunsRequired, maxRunsRequired + 1):
            situations.append({
                'runsRequired': runsRequired,
                'totalInningWickets': totalInningWickets,
                'inningBallsRemaining': inningBallsRemaining,
            })
shortGrid = pd.DataFrame(situations)
# Copy the EXACT saved neural grid above 24 balls; probabilities are not features.
neuralGrid = pd.read_csv(
    PROJECT_ROOT / 'men/matchMarket/hybridWin3/auxiliaries/1_chaseLookup.csv',
    usecols=stateColumns[:3],
).drop_duplicates()
neuralGrid = neuralGrid.loc[neuralGrid['inningBallsRemaining'].between(25, 120)].copy()
if neuralGrid.empty or not np.isfinite(neuralGrid.to_numpy(float)).all():
    raise ValueError('Missing or invalid long-innings neural grid.')
if not np.equal(neuralGrid.to_numpy(float), np.round(neuralGrid.to_numpy(float))).all():
    raise ValueError('Neural grid states must be integers.')
neuralGrid = neuralGrid.astype(np.int64)
if not (neuralGrid['runsRequired'].gt(0) & neuralGrid['totalInningWickets'].between(0, 9)).all():
    raise ValueError('Invalid runs/wickets in the neural grid.')
productionGrid = pd.concat([shortGrid, neuralGrid], ignore_index=True).drop_duplicates(stateColumns[:3])
productionGrid['productionGrid'] = True
# Rapid collapses can reach wicket/ball combinations excluded as starting states.
# Supply probabilities there, but do not simulate/export them as starting states.
supportRows = []
for balls, frame in neuralGrid.groupby('inningBallsRemaining'):
    presentWickets = set(frame['totalInningWickets'])
    for wickets in range(10):
        if wickets not in presentWickets:
            supportRows.extend((runs, wickets, int(balls)) for runs in range(1, int(frame['runsRequired'].max()) + 1))
supportGrid = pd.DataFrame(supportRows, columns=stateColumns[:3])
supportGrid['productionGrid'] = False
masterLookup = pd.concat([productionGrid, supportGrid], ignore_index=True)
masterLookup = masterLookup.sort_values(['inningBallsRemaining', 'totalInningWickets', 'runsRequired']).reset_index(drop=True)
masterLookup = masterLookup.loc[masterLookup.index.repeat(masterLookup['totalInningWickets'] + 2)].copy()
masterLookup['ord'] = masterLookup.groupby(['runsRequired', 'totalInningWickets', 'inningBallsRemaining']).cumcount() + 1
masterLookup = masterLookup.reset_index(drop=True)
masterLookup['inningBallNumber'] = 121 - masterLookup['inningBallsRemaining']
masterLookup['overNumber'] = np.ceil(masterLookup['inningBallNumber'] / 6).astype(int)
masterLookup['inningNumber'] = 2
masterLookup['runsRequiredPerBall'] = masterLookup['runsRequired'] / masterLookup['inningBallsRemaining']

for frame in [trainData, masterLookup]:
    frame['powerplay'] = frame['inningBallsRemaining'].ge(85).astype(int)
    frame['death'] = frame['inningBallsRemaining'].le(24).astype(int)
    frame['logBalls'] = np.log(frame['inningBallsRemaining'])
    frame['pressure'] = frame['runsRequired'] / frame['inningBallsRemaining']
    frame['logPressure'] = np.log(frame['pressure'])
    frame['ballsInOver'] = (frame['inningBallsRemaining'] - 1) % 6 + 1
    frame['orderGap'] = frame['totalInningWickets'] + 2 - frame['ord']
    frame['newestBatter'] = frame['orderGap'].eq(0).astype(int)
    frame['tailOrder'] = np.maximum(frame['ord'] - 7, 0)
    frame['runsPerWicket'] = frame['runsRequired'] / (10 - frame['totalInningWickets'])
    frame['pressureOrder'] = frame['pressure'] * frame['ord']
    frame['pressurePerWicket'] = frame['pressure'] / (10 - frame['totalInningWickets'])
    frame['ballsWickets'] = frame['inningBallsRemaining'] * (10 - frame['totalInningWickets'])
    frame['ballsOrder'] = frame['inningBallsRemaining'] * frame['ord']


# One unseen probability vector per eligible historical delivery.
oof = trainData[rowColumns + ballColumns + ['batsmanRuns', 'isWicket', 'jointTarget']].copy()
oof[jointColumns] = np.nan
# Five outer fits, ten reused inner fits, then the full-data production fit.
# An inner fit excludes BOTH listed folds. It can serve either as the outer
# test fold, with the other as an inner validation fold, without refitting.
jobs = [(fold, -1) for fold in range(N_FOLDS)]
jobs += list(combinations(range(N_FOLDS), 2))
jobs += [(-1, -1)]

def run_ball_job(job):
    foldA, foldB = job
    excludedFolds = [fold for fold in (foldA, foldB) if fold >= 0]
    if foldA == -1:
        jobName = 'production'
    elif foldB == -1:
        jobName = f'fold_{foldA}'
    else:
        jobName = f'inner_{foldA}_{foldB}'
    useRows = ~trainData['fold'].isin(excludedFolds)
    print(f'{jobName}: fitting ball model on {int(useRows.sum()):,} deliveries; excluded folds {excludedFolds}', flush=True)
    model = CatBoostClassifier(
        iterations=1200, depth=5, learning_rate=0.035, l2_leaf_reg=100,
        loss_function='MultiClass', random_seed=RANDOM_SEED,
        thread_count=THREADS, verbose=False, allow_writing_files=False,
    )
    checkpoint = PROJECT_ROOT / f'men/matchMarket/hybridWin3/cv/ballModel_{jobName}.cbm'
    checkpointInfo = PROJECT_ROOT / f'men/matchMarket/hybridWin3/cv/ballModel_{jobName}.json'
    recipeKey = hashlib.sha256(json.dumps({'data': dataHash, 'excluded': excludedFolds, 'features': featureColumns, 'foldSeed': FOLD_SEED, 'parameters': model.get_params()}, sort_keys=True).encode()).hexdigest()
    if checkpoint.exists() and checkpointInfo.exists() and json.loads(checkpointInfo.read_text()).get('recipeKey') == recipeKey:
        model.load_model(str(checkpoint))
        print(f'{jobName}: reused matching fitted-model checkpoint', flush=True)
    else:
        model.fit(trainData.loc[useRows, featureColumns].to_numpy(float), trainData.loc[useRows, 'jointTarget'].to_numpy(int))
        model.save_model(str(checkpoint))
        checkpointInfo.write_text(json.dumps({'recipeKey': recipeKey}))

    probabilities = np.zeros((len(masterLookup), len(jointEvents)), dtype=float)
    probabilities[:, model.classes_.astype(int)] = model.predict_proba(masterLookup[featureColumns].to_numpy(float))
    gridOutput = masterLookup[ballColumns + ['inningBallNumber', 'overNumber', 'inningNumber', 'runsRequiredPerBall', 'productionGrid']].copy()
    gridOutput[jointColumns] = probabilities
    if foldA == -1:
        gridOutput['pressure_6_wicket'] = 0.0
        for runs in range(7):
            gridOutput[f'pressure_{runs}'] = probabilities[:, jointEvents[:, 0] == runs].sum(axis=1)
        gridOutput['pressure_wicket'] = probabilities[:, jointEvents[:, 1] == 1].sum(axis=1)
        gridOutput['pressure_m_batsmanRunsBall'] = probabilities @ jointEvents[:, 0]
        gridOutput.to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/deathBallProbs.csv', index=False)
        model.save_model(str(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/deathBallModel.cbm'))
    else:
        gridOutput.to_csv(PROJECT_ROOT / f'men/matchMarket/hybridWin3/cv/deathBallProbs_{jobName}.csv', index=False)

    if foldA >= 0 and foldB == -1:
        testRows = trainData['fold'].eq(foldA)
        rowProbabilities = np.zeros((int(testRows.sum()), len(jointEvents)), dtype=float)
        rowProbabilities[:, model.classes_.astype(int)] = model.predict_proba(trainData.loc[testRows, featureColumns].to_numpy(float))
        return testRows, rowProbabilities

# Independent fits share the available CPU budget; prediction rows remain disjoint.
with ThreadPoolExecutor(max_workers=PARALLEL_FITS) as executor:
    for prediction in executor.map(run_ball_job, jobs):
        if prediction is not None:
            testRows, rowProbabilities = prediction
            oof.loc[testRows, jointColumns] = rowProbabilities

if oof[jointColumns].isna().any().any():
    raise RuntimeError('Some ball rows are missing their out-of-fold prediction.')
for runs in range(7):
    oof[f'pressure_{runs}'] = oof.loc[:, np.array(jointColumns)[jointEvents[:, 0] == runs]].sum(axis=1)
oof['pressure_wicket'] = oof.loc[:, np.array(jointColumns)[jointEvents[:, 1] == 1]].sum(axis=1)
oof['pressure_m_batsmanRunsBall'] = oof[jointColumns].to_numpy(float) @ jointEvents[:, 0]
oof.to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/deathBallProbs_oof.csv', index=False)
runInfo.at[0, 'ballReady'] = 1
runInfo.to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/cv/runInfo.csv', index=False)
print(f'Exported {len(oof):,} unseen ball predictions and {len(masterLookup):,} ball lookup rows (including internal transition rows).')
print(f'Ball-model runtime: {(time.perf_counter() - modelStartTime) / 60:.1f} minutes')
