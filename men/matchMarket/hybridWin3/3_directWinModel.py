# Direct match-win probabilities across all 20 overs.
# Run after 1_deathBallProbs.py and 2_simModel.py.
# Full remodel: richer resource/state splines, anchored to ball outcomes near the finish.
# Both components use only match state and batting order.
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '3')
import hashlib
import json
import time
from paths import PROJECT_ROOT
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from itertools import combinations

startTime = time.perf_counter()
N_FOLDS = 5
FOLD_SEED = 2718
RANDOM_SEED = 42
ENDGAME_ANCHOR_BALLS = 6
ALPHA = 0.0001
PREDICT_BATCH_SIZE = 25000
featureColumns = ['runsRequired', 'inningBallsRemaining', 'totalInningWickets', 'ord', 'nonStrikerOrd']
stateColumns = featureColumns
rowColumns = ['matchID', 'ID', 'fold']
resourceColumns = ['totalInningRunsToComeSimBiasSpline', 'totalInningRunsToComeSimSTD']

# %% READ DATA AND RETAIN THE PHASE 1 WHOLE-MATCH FOLDS
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

# Confirm that the preceding stages used the same data and folds.
runInfo = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/cv/runInfo.csv')
if 'pipelineVersion' not in runInfo or runInfo.at[0, 'pipelineVersion'] != 'full_remodel_v3':
    raise ValueError('These scripts require full-innings outputs. Run scripts 1-5 in order.')
if 'gridHash' not in runInfo or runInfo.at[0, 'gridHash'] != hashlib.sha256((PROJECT_ROOT / 'men/matchMarket/hybridWin3/auxiliaries/1_chaseLookup.csv').read_bytes()).hexdigest():
    raise ValueError('The neural grid changed. Rerun scripts 1-5 in order.')
if runInfo.at[0, 'dataHash'] != dataHash or int(runInfo.at[0, 'nFolds']) != N_FOLDS or int(runInfo.at[0, 'foldSeed']) != FOLD_SEED:
    raise ValueError('Input data or fold settings changed. Run scripts 1-3 in order.')
if int(runInfo.at[0, 'ballReady']) != 1 or int(runInfo.at[0, 'simulationReady']) != 1 or int(runInfo.at[0, 'endgameReady']) != 1:
    raise ValueError('Complete 1_deathBallProbs.py and 2_simModel.py first.')
if runInfo.at[0, 'simulationResourceHash'] != hashlib.sha256((PROJECT_ROOT / 'men/matchMarket/hybridWin3/auxiliaries/6_masterLookup.csv').read_bytes()).hexdigest():
    raise ValueError('The extras/resource lookup changed. Run 2_simModel.py before refitting the direct model.')
runInfo['directReady'] = 0
runInfo['blendReady'] = 0
runInfo.to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/cv/runInfo.csv', index=False)

# %% ELIGIBILITY AUDIT AND BATTING-ORDER RECONSTRUCTION
auditData = trainData.sort_values(['matchID', 'inningBallNumber', 'ID']).copy()
legalWicketsBefore = auditData.groupby('matchID')['isWicket'].cumsum() - auditData['isWicket']
duplicateBalls = auditData.duplicated(['matchID', 'inningBallNumber'], keep=False)
matchWickets = auditData.groupby('matchID')['isWicket'].sum()
ambiguousMatches = set(auditData.loc[duplicateBalls, 'matchID']) | set(matchWickets.index[matchWickets.gt(10)])
previousWickets = auditData.groupby('matchID')['totalInningWickets'].shift()
previousEvent = auditData.groupby('matchID')['isWicket'].shift()
incoherentRows = auditData['totalInningWickets'].lt(legalWicketsBefore) | auditData['totalInningWickets'].lt(previousWickets + previousEvent)
incoherentIDs = set(auditData.loc[incoherentRows, 'ID'])

# Historical names recover batting order within that match only.
# Prediction inputs are exclusively the agreed state/order columns.
if 'nonStrikerOrd' not in trainData.columns:
    trainData['nonStrikerOrd'] = np.nan
newestBatterOrd = trainData['totalInningWickets'] + 2
trainData['nonStrikerOrd'] = trainData['nonStrikerOrd'].fillna(
    pd.Series(np.where(trainData['ord'].ne(newestBatterOrd), newestBatterOrd, np.nan), index=trainData.index)
)
nonstrikerOrder = trainData.groupby(['matchID', 'inningNumber', 'batsmanName'])['ord'].min()
orderKeys = pd.MultiIndex.from_frame(trainData[['matchID', 'inningNumber', 'nonstrikerName']])
trainData['nonStrikerOrd'] = trainData['nonStrikerOrd'].fillna(
    pd.Series(nonstrikerOrder.reindex(orderKeys).to_numpy(), index=trainData.index)
)
# Leave unresolved orders missing rather than inventing a batting position.
validPair = (
    trainData['ord'].le(newestBatterOrd)
    & trainData['nonStrikerOrd'].between(1, 11)
    & trainData['nonStrikerOrd'].le(newestBatterOrd)
    & trainData['ord'].ne(trainData['nonStrikerOrd'])
)

trainingMask = validPair & ~trainData['matchID'].isin(ambiguousMatches) & ~trainData['ID'].isin(incoherentIDs)

# %% JOIN THE FIRST-INNINGS RESOURCE LOOKUP
# These two supplied columns depend only on balls remaining and wickets.
# Validate that duplicates really agree before reducing the source lookup.
resourceLookup = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/auxiliaries/6_masterLookup.csv', usecols=['inningNumber', 'inningBallNumber', 'totalInningWickets'] + resourceColumns)
resourceLookup = resourceLookup.loc[resourceLookup['inningNumber'].eq(1)].copy()
resourceKeys = ['inningBallNumber', 'totalInningWickets']
for column in resourceColumns:
    if resourceLookup.groupby(resourceKeys)[column].nunique(dropna=False).gt(1).any():
        raise ValueError(f'{column} is not uniquely defined by ball and wickets in this lookup.')
resourceLookup = resourceLookup.drop_duplicates(resourceKeys)
resourceLookup['inningBallsRemaining'] = 121 - resourceLookup['inningBallNumber']
resourceKeys = ['inningBallsRemaining', 'totalInningWickets']
resourceLookup = resourceLookup[resourceKeys + resourceColumns]
resourceLookup = resourceLookup.loc[np.isfinite(resourceLookup[resourceColumns]).all(axis=1) & resourceLookup[resourceColumns].gt(0).all(axis=1)].copy()
# Retain every balls/wickets state. Missing resource features use the state-only component.
resourceLookup = pd.MultiIndex.from_product([range(1, 121), range(10)], names=resourceKeys).to_frame(index=False).merge(resourceLookup, on=resourceKeys, how='left', validate='one_to_one')
resourceLookup.to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/resourceLookup.csv', index=False)

trainData['eligibleTraining'] = trainingMask
trainData = trainData.loc[validPair, rowColumns + stateColumns + ['chaseWin', 'eligibleTraining']].copy()
trainData = trainData.merge(resourceLookup, on=resourceKeys, how='left', validate='many_to_one')
trainData['lookupAvailable'] = trainData[resourceColumns].notna().all(axis=1)
trainingMask = trainData['eligibleTraining']
oof = trainData.copy().reset_index(drop=True)
oof['resourceAvailable'] = oof['lookupAvailable']
oof['overNumber'] = ((120 - oof['inningBallsRemaining']) // 6 + 1).astype(int)
for column in ['resourceWinProbability', 'stateWinProbability', 'directWinProbability']:
    oof[column] = np.nan
print(f'Direct-model training: {int(trainingMask.sum()):,} rows; evaluation: {len(oof):,} rows.', flush=True)
print(f'State-only fallback: {int((~trainData["lookupAvailable"]).sum()):,} historical rows without resource features.', flush=True)
print('Five whole-match outer folds, ten paired inner fits, then full-data production; fixed direct-model recipe.', flush=True)

# %% PRODUCTION STATES AND THE SAME OUTER/INNER FOLD JOBS
# The ball grid is generated in script 1 and the chase grid in script 2.
# Above 24 balls, direct production uses exactly the same starting grid as simulation.
# Keep existing short-innings historical states for compatibility; OOF training stays full-innings.
simulationStates = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/chaseSimSituations.csv', usecols=stateColumns)
production = pd.concat([trainData.loc[trainData['inningBallsRemaining'].le(24), stateColumns], simulationStates], ignore_index=True).drop_duplicates(stateColumns)
production = production.merge(resourceLookup, on=resourceKeys, how='inner', validate='many_to_one')
production = production.sort_values(stateColumns).reset_index(drop=True)
probabilityColumns = ['resourceWinProbability', 'stateWinProbability', 'directWinProbability']
print(f'Production direct states: {len(production):,}.', flush=True)

# The direct components themselves have a fixed 50/50 average. Paired inner
# fits are required by script 4, so its learned simulation/direct blend never
# trains on predictions from a base fit that included the outer test matches.
coefficientRows = []
fitRows = []
jobs = [(fold, -1) for fold in range(N_FOLDS)]
jobs += list(combinations(range(N_FOLDS), 2))
jobs += [(-1, -1)]
for foldA, foldB in jobs:
    excludedFolds = [fold for fold in (foldA, foldB) if fold >= 0]
    if foldA == -1:
        jobName = 'production'
    elif foldB == -1:
        jobName = f'fold_{foldA}'
    else:
        jobName = f'inner_{foldA}_{foldB}'
    fitMask = trainingMask & ~trainData['fold'].isin(excludedFolds)
    fitData = trainData.loc[fitMask]
    if foldA == -1:
        requiredStates = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/chaseSimSituations.csv', usecols=stateColumns).drop_duplicates(stateColumns)
    else:
        requiredStates = pd.read_csv(PROJECT_ROOT / f'men/matchMarket/hybridWin3/cv/chaseSimSituations_{jobName}.csv', usecols=stateColumns).drop_duplicates(stateColumns)
    if foldA == -1:
        evaluation = production[stateColumns].copy()
    elif foldB == -1:
        evaluationRows = oof.loc[oof['fold'].eq(foldA)]
        if set(fitData['matchID']) & set(evaluationRows['matchID']):
            raise RuntimeError('Training and evaluation matches overlap.')
        evaluation = pd.concat([evaluationRows[stateColumns], requiredStates], ignore_index=True).drop_duplicates(stateColumns)
    else:
        evaluation = requiredStates.copy()
    evaluation = evaluation.merge(resourceLookup, on=resourceKeys, how='inner', validate='many_to_one').reset_index(drop=True)
    evaluation[probabilityColumns] = np.nan
    print(f'{jobName}: fitting the two direct components on {len(fitData):,} deliveries; excluded folds {excludedFolds}', flush=True)

    # Group identical predictor rows with binomial counts. This is exactly the
    # same per-ball likelihood, with no averaging of different match states.
    grouped = fitData.groupby(stateColumns, sort=True)['chaseWin'].agg(['sum', 'count']).reset_index()
    grouped = grouped.merge(resourceLookup, on=resourceKeys, how='left', validate='many_to_one')
    target = grouped['sum'].to_numpy(float) / grouped['count'].to_numpy(float)
    weights = grouped['count'].to_numpy(float)
    weights /= weights.sum()

    for component in ['resource', 'state']:
        # Build the same basis for training and each prediction batch.
        componentTraining = grouped
        componentEvaluation = evaluation
        if component == 'resource':
            componentTraining = grouped.loc[grouped[resourceColumns].notna().all(axis=1)]
            componentEvaluation = evaluation.loc[evaluation[resourceColumns].notna().all(axis=1)]
        target = componentTraining['sum'].to_numpy(float) / componentTraining['count'].to_numpy(float)
        weights = componentTraining['count'].to_numpy(float)
        weights /= weights.sum()
        tasks = [('fit', componentTraining)]
        tasks += [('predict', componentEvaluation.iloc[start:start + PREDICT_BATCH_SIZE]) for start in range(0, len(componentEvaluation), PREDICT_BATCH_SIZE)]
        completed = 0
        checkpoint = PROJECT_ROOT / f'men/matchMarket/hybridWin3/cv/directCoefficients_{jobName}_{component}.npy'
        checkpointInfo = PROJECT_ROOT / f'men/matchMarket/hybridWin3/cv/directCoefficients_{jobName}_{component}.json'
        resourceHash = hashlib.sha256((PROJECT_ROOT / 'men/matchMarket/hybridWin3/auxiliaries/6_masterLookup.csv').read_bytes()).hexdigest()
        recipeKey = hashlib.sha256(json.dumps({'data': dataHash, 'resource': resourceHash, 'foldSeed': FOLD_SEED, 'excluded': excludedFolds, 'component': component, 'alpha': ALPHA, 'recipe': 'rich_splines_v1'}, sort_keys=True).encode()).hexdigest()
        cachedInfo = json.loads(checkpointInfo.read_text()) if checkpointInfo.exists() else {}
        for task, frame in tasks:
            if task == 'fit' and checkpoint.exists() and cachedInfo.get('recipeKey') == recipeKey:
                coefficients = np.load(checkpoint)
                fitRows.append({key: value for key, value in cachedInfo.items() if key != 'recipeKey'})
                if foldA == -1:
                    coefficientRows.extend({'component': component, 'coefficient': i, 'value': value} for i, value in enumerate(coefficients))
                print(f'  {component}: reused matching fitted coefficients', flush=True)
                continue
            runs = frame['runsRequired'].to_numpy(float)
            balls = frame['inningBallsRemaining'].to_numpy(float)
            wickets = frame['totalInningWickets'].to_numpy(float)
            striker = frame['ord'].to_numpy(float)
            nonstriker = frame['nonStrikerOrd'].to_numpy(float)
            if component == 'resource':
                first = (runs - frame[resourceColumns[0]].to_numpy(float)) / frame[resourceColumns[1]].to_numpy(float)
                firstKnots = np.array([-10., -5., -3., -2., -1., 0., 1., 2., 3., 5., 10., 20.])
            else:
                first = np.log(runs)
                firstKnots = np.log([1., 5., 10., 20., 30., 45., 60., 80., 110., 150., 210., 350.])
            values = [first, np.log(balls), wickets, striker, nonstriker]
            knots = [firstKnots, np.log([1., 3., 6., 12., 24., 48., 72., 84., 96., 120.]),
                     np.array([0., 2., 4., 6., 8., 9.]), np.array([1., 3., 5., 7., 9., 11.]), np.array([1., 3., 5., 7., 9., 11.])]
            bases = []
            for value, knot in zip(values, knots):
                extendedKnots = np.r_[np.repeat(knot[0], 2), knot, np.repeat(knot[-1], 2)]
                bases.append(sparse.csr_matrix(BSpline.design_matrix(np.clip(value, knot[0], knot[-1]), extendedKnots, 2)))
            blocks = [sparse.csr_matrix(np.ones((len(frame), 1)))] + bases
            shapes = [(basis.shape[1],) for basis in bases]
            for axisA, axisB in [(0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (3, 4), (1, 3), (1, 4), (2, 3), (2, 4)]:
                left, right = bases[axisA], bases[axisB]
                leftCount, rightCount = np.diff(left.indptr)[0], np.diff(right.indptr)[0]
                entries = (left.data.reshape(-1, leftCount, 1) * right.data.reshape(-1, 1, rightCount)).reshape(-1)
                indices = (left.indices.reshape(-1, leftCount, 1) * right.shape[1] + right.indices.reshape(-1, 1, rightCount)).reshape(-1)
                indptr = np.arange(len(frame) + 1, dtype=np.int64) * leftCount * rightCount
                blocks.append(sparse.csr_matrix((entries, indices, indptr), shape=(len(frame), left.shape[1] * right.shape[1])))
                shapes.append((left.shape[1], right.shape[1]))
            # Full runs/balls/wickets interaction, with a curvature penalty on each axis.
            left, right, third = bases[:3]
            leftCount, rightCount, thirdCount = [np.diff(basis.indptr)[0] for basis in (left, right, third)]
            entries = (left.data.reshape(-1, leftCount, 1, 1) * right.data.reshape(-1, 1, rightCount, 1) * third.data.reshape(-1, 1, 1, thirdCount)).reshape(-1)
            indices = ((left.indices.reshape(-1, leftCount, 1, 1) * right.shape[1] + right.indices.reshape(-1, 1, rightCount, 1)) * third.shape[1] + third.indices.reshape(-1, 1, 1, thirdCount)).reshape(-1)
            indptr = np.arange(len(frame) + 1, dtype=np.int64) * leftCount * rightCount * thirdCount
            blocks.append(sparse.csr_matrix((entries, indices, indptr), shape=(len(frame), left.shape[1] * right.shape[1] * third.shape[1])))
            shapes.append((left.shape[1], right.shape[1], third.shape[1]))
            ballsInOver = ((balls - 1) % 6 + 1) / 6
            orderDifference = (striker - nonstriker) / 10
            blocks.append(sparse.csr_matrix(np.column_stack([orderDifference * ballsInOver, orderDifference * ballsInOver ** 2])))
            design = sparse.hstack(blocks, format='csr')
            design.eliminate_zeros()

            if task == 'fit':
                # Curvature penalties keep the surfaces smooth, including
                # sparse states; a small ridge stabilizes the coefficients.
                penalties = [sparse.csr_matrix((1, 1))]
                for shape in shapes:
                    size = int(np.prod(shape))
                    blockPenalty = sparse.eye(size, format='csr') * .02
                    for axis, count in enumerate(shape):
                        secondDifference = sparse.diags([np.ones(count - 2), -2 * np.ones(count - 2), np.ones(count - 2)], [0, 1, 2], shape=(count - 2, count))
                        difference = sparse.kron(sparse.eye(int(np.prod(shape[:axis]))), sparse.kron(secondDifference, sparse.eye(int(np.prod(shape[axis + 1:])))), format='csr')
                        blockPenalty = blockPenalty + difference.T @ difference
                    penalties.append(blockPenalty)
                penalties.append(sparse.eye(2, format='csr'))
                penalty = sparse.block_diag(penalties, format='csr')
                transpose = design.T.tocsr()
                initial = np.zeros(design.shape[1])
                initial[0] = np.log(np.dot(weights, target) / (1 - np.dot(weights, target)))
                objective = lambda coefficients: (
                    np.dot(weights, np.logaddexp(0, design @ coefficients) - target * (design @ coefficients)) + ALPHA * np.dot(coefficients, penalty @ coefficients) / 2,
                    transpose @ (weights * (expit(design @ coefficients) - target)) + ALPHA * (penalty @ coefficients),
                )
                fit = minimize(objective, initial, jac=True, method='L-BFGS-B', bounds=[(-30, 30)] * len(initial), options={'maxiter': 700, 'ftol': 1e-10, 'gtol': 1e-6, 'maxcor': 15})
                if not fit.success:
                    raise RuntimeError(f'{jobName} {component} did not converge: {fit.message}')
                coefficients = fit.x.copy()
                fitRows.append({'job': jobName, 'component': component, 'excludedFolds': ','.join(map(str, excludedFolds)), 'trainingMatches': fitData['matchID'].nunique(), 'trainingRows': int(componentTraining['count'].sum()), 'uniqueStates': len(componentTraining), 'parameters': len(coefficients), 'iterations': fit.nit, 'converged': fit.success})
                np.save(checkpoint, coefficients)
                checkpointInfo.write_text(json.dumps(dict(fitRows[-1], recipeKey=recipeKey)))
                if foldA == -1:
                    coefficientRows.extend({'component': component, 'coefficient': i, 'value': value} for i, value in enumerate(coefficients))
            else:
                probability = np.clip(expit(design @ coefficients), 1e-7, 1 - 1e-7)
                evaluation.loc[frame.index, component + 'WinProbability'] = probability
                completed += len(frame)
        print(f'  {component}: converged; predicted {completed:,} states.', flush=True)

    evaluation['directWinProbability'] = evaluation[['resourceWinProbability', 'stateWinProbability']].mean(axis=1)
    evaluation['resourceAvailable'] = evaluation['resourceWinProbability'].notna()
    # The exact endgame model comes from the ball fit with the SAME excluded
    # matches as this direct fit. No outer-test outcomes enter this anchor.
    if foldA == -1:
        endgameLookup = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/endgameWinProbabilities.csv', float_precision='round_trip')
    else:
        endgameLookup = pd.read_csv(PROJECT_ROOT / f'men/matchMarket/hybridWin3/cv/endgameWinProbabilities_{jobName}.csv', float_precision='round_trip')
    evaluation = evaluation.merge(endgameLookup, on=stateColumns, how='left', validate='one_to_one')
    nearFinish = evaluation['inningBallsRemaining'].le(6)
    # Targets above the computed range require more than 12 invalid balls.
    # Their omitted probability is below the bound exported by script 2.
    missingEndgame = nearFinish & evaluation['endgameWinProbability'].isna()
    if (evaluation.loc[missingEndgame, 'runsRequired'] <= endgameLookup['runsRequired'].max()).any():
        raise ValueError('An endgame state within the calculated range is missing.')
    evaluation.loc[missingEndgame, 'endgameWinProbability'] = 0.0
    splineWeight = np.clip((evaluation.loc[nearFinish, 'inningBallsRemaining'] - 1) / (ENDGAME_ANCHOR_BALLS - 1), 0, 1)
    evaluation.loc[nearFinish, 'directWinProbability'] = (
        splineWeight * evaluation.loc[nearFinish, 'directWinProbability']
        + (1 - splineWeight) * evaluation.loc[nearFinish, 'endgameWinProbability']
    )
    evaluation['directWinProbability'] = np.clip(evaluation['directWinProbability'], 1e-7, 1 - 1e-7)
    evaluation = evaluation.drop(columns='endgameWinProbability')
    if foldA == -1:
        production = evaluation
    else:
        gridOutput = requiredStates.merge(evaluation[stateColumns + probabilityColumns], on=stateColumns, how='left', validate='one_to_one')
        if len(gridOutput) != len(requiredStates) or not np.isfinite(gridOutput[['stateWinProbability', 'directWinProbability']].to_numpy()).all():
            raise RuntimeError(f'Incomplete direct prediction coverage for {jobName}.')
        gridOutput.to_csv(PROJECT_ROOT / f'men/matchMarket/hybridWin3/cv/directWinProbabilities_{jobName}.csv', index=False)
        if foldB == -1:
            aligned = evaluationRows[rowColumns + stateColumns].merge(evaluation[stateColumns + probabilityColumns], on=stateColumns, how='left', validate='many_to_one', sort=False)
            if len(aligned) != len(evaluationRows) or aligned[['stateWinProbability', 'directWinProbability']].isna().any().any():
                raise RuntimeError(f'Missing direct OOF probabilities for {jobName}.')
            oof.loc[evaluationRows.index, probabilityColumns] = aligned[probabilityColumns].to_numpy()

pd.DataFrame(coefficientRows).to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/directWinCoefficients.csv', index=False)
pd.DataFrame(fitRows).to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/directWinFitLog.csv', index=False)
production.to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/directWinProbabilities.csv', index=False)
if oof[['stateWinProbability', 'directWinProbability']].isna().any().any() or oof.duplicated(['matchID', 'ID']).any():
    raise RuntimeError('Missing or duplicate OOF predictions.')
neural = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/auxiliaries/1_chaseLookup.csv', usecols=['runsRequired', 'inningBallsRemaining', 'totalInningWickets', 'm_chaseWin%'])
neuralKeys = ['runsRequired', 'inningBallsRemaining', 'totalInningWickets']
if neural.groupby(neuralKeys)['m_chaseWin%'].nunique(dropna=False).gt(1).any():
    raise ValueError('The neural lookup has conflicting probabilities for a state.')
oof = oof.merge(neural.drop_duplicates(neuralKeys), on=neuralKeys, how='left', validate='many_to_one')
oof.to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/directWinProbabilities_oof.csv', index=False)
pd.DataFrame([{'dataHash': dataHash, 'lookupHash': hashlib.sha256((PROJECT_ROOT / 'men/matchMarket/hybridWin3/auxiliaries/6_masterLookup.csv').read_bytes()).hexdigest(), 'nFolds': N_FOLDS, 'foldSeed': FOLD_SEED, 'trainingRows': int(trainingMask.sum()), 'oofRows': len(oof), 'stateOnlyRows': int((~trainData['lookupAvailable']).sum()), 'productionStates': len(production), 'resourceWeight': .5, 'stateWeight': .5, 'alpha': ALPHA, 'endgameAnchorBalls': ENDGAME_ANCHOR_BALLS}]).to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/directWinRunInfo.csv', index=False)

# %% OVERALL, OVER-BY-OVER AND FOLD COMPARISONS
# All models in each scope use exactly the same rows.
# The old neural is an in-sample reference. Phase 2 and Phase 3 architecture
# choices used this historical data. Full five-fold OOF excludes each match
# from fitting but is not an untouched external or model-selection test.
scoreRows = []
overRows = []
foldRows = []
calibrationRows = []
for scope in ['All covered states', 'Common neural states']:
    frame = oof if scope == 'All covered states' else oof.loc[oof['m_chaseWin%'].between(0, 1)]
    models = [('Direct win model', 'directWinProbability')]
    if scope == 'Common neural states':
        models.append(('Old neural (in sample)', 'm_chaseWin%'))
    for modelName, probabilityColumn in models:
        result = frame[['matchID', 'ID', 'fold', 'overNumber', 'chaseWin', probabilityColumn]].copy()
        y = result['chaseWin'].to_numpy(float)
        p = np.clip(result[probabilityColumn].to_numpy(float), 1e-7, 1 - 1e-7)
        result['logLoss'] = -y * np.log(p) - (1 - y) * np.log1p(-p)
        result['brier'] = (p - y) ** 2
        result['probability'] = p
        scoreRows.append({'scope': scope, 'model': modelName, 'rows': len(result), 'matches': result['matchID'].nunique(), 'logLoss': result['logLoss'].mean(), 'brier': result['brier'].mean(), 'equalMatchLogLoss': result.groupby('matchID')['logLoss'].mean().mean(), 'meanProbability': p.mean(), 'winRate': y.mean()})
        for over, part in result.groupby('overNumber'):
            overRows.append({'scope': scope, 'model': modelName, 'over': over, 'rows': len(part), 'matches': part['matchID'].nunique(), 'logLoss': part['logLoss'].mean(), 'brier': part['brier'].mean()})
        for fold, part in result.groupby('fold'):
            foldRows.append({'scope': scope, 'model': modelName, 'fold': fold, 'rows': len(part), 'matches': part['matchID'].nunique(), 'logLoss': part['logLoss'].mean(), 'brier': part['brier'].mean()})
        result['probabilityBin'] = np.minimum((result['probability'] * 20).astype(int), 19)
        for binNumber, part in result.groupby('probabilityBin'):
            calibrationRows.append({'scope': scope, 'model': modelName, 'binLow': binNumber / 20, 'binHigh': (binNumber + 1) / 20, 'rows': len(part), 'meanProbability': part['probability'].mean(), 'winRate': part['chaseWin'].mean()})
scores = pd.DataFrame(scoreRows)
scores.to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/directWinScores.csv', index=False)
pd.DataFrame(overRows).to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/directWinByOver.csv', index=False)
pd.DataFrame(foldRows).to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/directWinByFold.csv', index=False)
pd.DataFrame(calibrationRows).to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/directWinCalibration.csv', index=False)
print('\nFULL-INNINGS RESULTS', flush=True)
print(scores.to_string(index=False, float_format='{:.9f}'.format), flush=True)
overComparison = pd.DataFrame(overRows)
overComparison = overComparison.loc[overComparison['scope'].eq('Common neural states')]
overComparison = overComparison.pivot(index='over', columns='model', values=['logLoss', 'brier'])
print('\nDIRECT VERSUS OLD NEURAL, OVER BY OVER:', flush=True)
print(overComparison.to_string(float_format='{:.9f}'.format), flush=True)
print('The old neural is an in-sample benchmark. Whole-match OOF excludes each test match from parameter fitting; earlier recipe selection and the fixed lookup remain evaluation limitations.', flush=True)
runInfo['directReady'] = 1
runInfo['directResourceHash'] = hashlib.sha256((PROJECT_ROOT / 'men/matchMarket/hybridWin3/auxiliaries/6_masterLookup.csv').read_bytes()).hexdigest()
runInfo['directRecipe'] = 'rich_resource_state_splines_state_fallback_alpha0.0001_endgame_anchor6'
runInfo['directOofRows'] = len(oof)
runInfo['directStateOnlyRows'] = int((~trainData['lookupAvailable']).sum())
runInfo.to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/cv/runInfo.csv', index=False)
print(f'\nFinished in {(time.perf_counter() - startTime) / 60:.1f} minutes. Outputs: {PROJECT_ROOT / "men/matchMarket/hybridWin3/outputs"}', flush=True)


# %% PREDICT ANY NEW STATES WITHOUT TRAINING
# Run just this cell in PyCharm after the model outputs exist.
# Input: hybridWin3/statesToPredict.csv with the five stateColumns below.
# Inference uses the coefficient/resource CSVs plus the endgame probability CSV.
from paths import PROJECT_ROOT
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.special import expit

stateColumns = ['runsRequired', 'inningBallsRemaining', 'totalInningWickets', 'ord', 'nonStrikerOrd']
resourceKeys = ['inningBallsRemaining', 'totalInningWickets']
resourceColumns = ['totalInningRunsToComeSimBiasSpline', 'totalInningRunsToComeSimSTD']
if (PROJECT_ROOT / 'men/matchMarket/hybridWin3/statesToPredict.csv').exists():
    states = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/statesToPredict.csv', usecols=stateColumns)
    if not np.isfinite(states.to_numpy(float)).all() or not np.equal(states.to_numpy(float), np.round(states.to_numpy(float))).all():
        raise ValueError('New state inputs must be finite integers.')
    newest = states['totalInningWickets'] + 2
    valid = (states['runsRequired'].gt(0) & states['inningBallsRemaining'].between(1, 120)
             & states['totalInningWickets'].between(0, 9) & states['ord'].between(1, 11)
             & states['nonStrikerOrd'].between(1, 11) & states['ord'].le(newest)
             & states['nonStrikerOrd'].le(newest) & states['ord'].ne(states['nonStrikerOrd']))
    if not valid.all():
        raise ValueError('New states must be live T20 chases with valid, different batting orders. Terminal/tied matches are outside this model.')
    states.insert(0, 'inputRow', np.arange(len(states)))
    suppliedCount = len(states)
    lookup = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/resourceLookup.csv', float_precision='round_trip')
    states = states.merge(lookup, on=resourceKeys, how='inner', validate='many_to_one')
    states = states.reset_index(drop=True)
    saved = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/directWinCoefficients.csv', float_precision='round_trip')
    states['resourceWinProbability'] = np.nan
    states['stateWinProbability'] = np.nan
    for component in ['resource', 'state']:
        coefficients = saved.loc[saved['component'].eq(component)].sort_values('coefficient')['value'].to_numpy(float)
        for start in range(0, len(states), 25000):
            frame = states.iloc[start:start + 25000]
            if component == 'resource':
                frame = frame.loc[frame[resourceColumns].notna().all(axis=1)]
            if frame.empty:
                continue
            runs = frame['runsRequired'].to_numpy(float)
            balls = frame['inningBallsRemaining'].to_numpy(float)
            wickets = frame['totalInningWickets'].to_numpy(float)
            striker = frame['ord'].to_numpy(float)
            nonstriker = frame['nonStrikerOrd'].to_numpy(float)
            if component == 'resource':
                first = (runs - frame[resourceColumns[0]].to_numpy(float)) / frame[resourceColumns[1]].to_numpy(float)
                firstKnots = np.array([-10., -5., -3., -2., -1., 0., 1., 2., 3., 5., 10., 20.])
            else:
                first = np.log(runs)
                firstKnots = np.log([1., 5., 10., 20., 30., 45., 60., 80., 110., 150., 210., 350.])
            values = [first, np.log(balls), wickets, striker, nonstriker]
            knots = [firstKnots, np.log([1., 3., 6., 12., 24., 48., 72., 84., 96., 120.]),
                     np.array([0., 2., 4., 6., 8., 9.]), np.array([1., 3., 5., 7., 9., 11.]), np.array([1., 3., 5., 7., 9., 11.])]
            bases = []
            for value, knot in zip(values, knots):
                extendedKnots = np.r_[np.repeat(knot[0], 2), knot, np.repeat(knot[-1], 2)]
                bases.append(sparse.csr_matrix(BSpline.design_matrix(np.clip(value, knot[0], knot[-1]), extendedKnots, 2)))
            blocks = [sparse.csr_matrix(np.ones((len(frame), 1)))] + bases
            shapes = [(basis.shape[1],) for basis in bases]
            for axisA, axisB in [(0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (3, 4), (1, 3), (1, 4), (2, 3), (2, 4)]:
                left, right = bases[axisA], bases[axisB]
                leftCount, rightCount = np.diff(left.indptr)[0], np.diff(right.indptr)[0]
                entries = (left.data.reshape(-1, leftCount, 1) * right.data.reshape(-1, 1, rightCount)).reshape(-1)
                indices = (left.indices.reshape(-1, leftCount, 1) * right.shape[1] + right.indices.reshape(-1, 1, rightCount)).reshape(-1)
                indptr = np.arange(len(frame) + 1, dtype=np.int64) * leftCount * rightCount
                blocks.append(sparse.csr_matrix((entries, indices, indptr), shape=(len(frame), left.shape[1] * right.shape[1])))
                shapes.append((left.shape[1], right.shape[1]))
            # Full runs/balls/wickets interaction, with a curvature penalty on each axis.
            left, right, third = bases[:3]
            leftCount, rightCount, thirdCount = [np.diff(basis.indptr)[0] for basis in (left, right, third)]
            entries = (left.data.reshape(-1, leftCount, 1, 1) * right.data.reshape(-1, 1, rightCount, 1) * third.data.reshape(-1, 1, 1, thirdCount)).reshape(-1)
            indices = ((left.indices.reshape(-1, leftCount, 1, 1) * right.shape[1] + right.indices.reshape(-1, 1, rightCount, 1)) * third.shape[1] + third.indices.reshape(-1, 1, 1, thirdCount)).reshape(-1)
            indptr = np.arange(len(frame) + 1, dtype=np.int64) * leftCount * rightCount * thirdCount
            blocks.append(sparse.csr_matrix((entries, indices, indptr), shape=(len(frame), left.shape[1] * right.shape[1] * third.shape[1])))
            shapes.append((left.shape[1], right.shape[1], third.shape[1]))
            ballsInOver = ((balls - 1) % 6 + 1) / 6
            orderDifference = (striker - nonstriker) / 10
            blocks.append(sparse.csr_matrix(np.column_stack([orderDifference * ballsInOver, orderDifference * ballsInOver ** 2])))
            design = sparse.hstack(blocks, format='csr')
            design.eliminate_zeros()
            if len(coefficients) != design.shape[1]:
                raise ValueError('The saved coefficient CSV does not match this script.')
            states.loc[frame.index, component + 'WinProbability'] = np.clip(expit(design @ coefficients), 1e-7, 1 - 1e-7)
    if states.empty:
        states['resourceWinProbability'] = np.nan
        states['stateWinProbability'] = np.nan
    states['directWinProbability'] = states[['resourceWinProbability', 'stateWinProbability']].mean(axis=1)
    states['resourceAvailable'] = states['resourceWinProbability'].notna()
    endgameLookup = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/endgameWinProbabilities.csv', float_precision='round_trip')
    states = states.merge(endgameLookup, on=stateColumns, how='left', validate='many_to_one')
    nearFinish = states['inningBallsRemaining'].le(6)
    missingEndgame = nearFinish & states['endgameWinProbability'].isna()
    if (states.loc[missingEndgame, 'runsRequired'] <= endgameLookup['runsRequired'].max()).any():
        raise ValueError('An endgame state within the calculated range is missing.')
    states.loc[missingEndgame, 'endgameWinProbability'] = 0.0
    splineWeight = np.clip((states.loc[nearFinish, 'inningBallsRemaining'] - 1) / 5, 0, 1)
    states.loc[nearFinish, 'directWinProbability'] = splineWeight * states.loc[nearFinish, 'directWinProbability'] + (1 - splineWeight) * states.loc[nearFinish, 'endgameWinProbability']
    states['directWinProbability'] = np.clip(states['directWinProbability'], 1e-7, 1 - 1e-7)
    states = states.drop(columns='endgameWinProbability')
    states.to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/customDirectWinProbabilities.csv', index=False)
    print(f"Predicted {len(states):,} custom states; {int((~states['resourceAvailable']).sum()):,} use state-only fallback.", flush=True)
else:
    print(f'For new predictions, create {PROJECT_ROOT / "men/matchMarket/hybridWin3/statesToPredict.csv"} and run only the final cell.', flush=True)

