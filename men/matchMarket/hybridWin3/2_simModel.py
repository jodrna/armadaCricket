import os
os.environ.setdefault('NUMBA_NUM_THREADS', '4')

import hashlib
import time
from itertools import combinations
import numpy as np
import pandas as pd
from numba import njit
from numba import prange
from sklearn.model_selection import StratifiedKFold
from scipy.special import betainc
from paths import PROJECT_ROOT

simulationStartTime = time.perf_counter()
NUMBER_OF_SIMS = 10000
BATCH_SIZE = 2000
N_FOLDS = 5
FOLD_SEED = 2718
RANDOM_SEED = 42
stateColumns = ['runsRequired', 'totalInningWickets', 'inningBallsRemaining', 'ord', 'nonStrikerOrd']
ballColumns = stateColumns[:4]
rowColumns = ['matchID', 'ID', 'fold']
extraColumns = ['byeRunsOver', 'isWicketInvalidOver', 'isInvalidOver', 'invalidRunsOver']

# --------------------------------------------------
# READ THE SAME DATA AND RECREATE THE SAME WHOLE-MATCH FOLDS
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
runInfo = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/cv/runInfo.csv')
if 'pipelineVersion' not in runInfo or runInfo.at[0, 'pipelineVersion'] != 'full_remodel_v3':
    raise ValueError('These scripts require full-innings outputs. Run scripts 1-5 in order.')
if 'gridHash' not in runInfo or runInfo.at[0, 'gridHash'] != hashlib.sha256((PROJECT_ROOT / 'men/matchMarket/hybridWin3/auxiliaries/1_chaseLookup.csv').read_bytes()).hexdigest():
    raise ValueError('The neural grid changed. Rerun scripts 1-5 in order.')
if runInfo.at[0, 'dataHash'] != dataHash or int(runInfo.at[0, 'nFolds']) != N_FOLDS or int(runInfo.at[0, 'foldSeed']) != FOLD_SEED:
    raise ValueError('Input data or fold settings changed. Rerun 1_deathBallProbs.py and then the remaining scripts in order.')
if int(runInfo.at[0, 'ballReady']) != 1:
    raise ValueError('Complete 1_deathBallProbs.py before running the simulation.')
runInfo.at[0, 'simulationReady'] = 0
runInfo.at[0, 'directReady'] = 0
runInfo.at[0, 'blendReady'] = 0
runInfo.to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/cv/runInfo.csv', index=False)

# --------------------------------------------------
# ORIGINAL EXTRAS AND GENERATED BALL GRID
# --------------------------------------------------
masterLookup = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/deathBallProbs.csv', float_precision='round_trip')
# Use the expanded production grid; preserve nearest-run transition lookup behavior.
if 'productionGrid' not in masterLookup or not masterLookup['productionGrid'].isin([True, False]).all():
    raise ValueError('Missing production-grid flags. Rerun script 1.')
oldMasterLookup = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/auxiliaries/6_masterLookup.csv')
invalidLookup = oldMasterLookup[['totalInningWickets', 'inningBallNumber'] + extraColumns]
invalidLookup = invalidLookup.drop_duplicates(['totalInningWickets', 'inningBallNumber'])
masterLookup = masterLookup.merge(invalidLookup, on=['totalInningWickets', 'inningBallNumber'], how='left', validate='many_to_one')
masterLookup = masterLookup.drop_duplicates(ballColumns, keep='last').reset_index(drop=True)
extraValues = np.ascontiguousarray(masterLookup[extraColumns].to_numpy(np.float64))
if not np.isfinite(extraValues).all():
    raise ValueError('The original extras lookup is missing values for some grid states.')

# --------------------------------------------------
# BUILD THE FULL-INNINGS CHASE GRID AND LOOKUP IN MEMORY
# --------------------------------------------------
situations = []
for runsRequired, totalInningWickets, inningBallsRemaining in masterLookup.loc[masterLookup['productionGrid'], stateColumns[:3]].drop_duplicates().itertuples(index=False, name=None):
    newestBatterOrd = int(totalInningWickets) + 2
    for otherBatterOrd in range(1, newestBatterOrd):
        situations.append((runsRequired, totalInningWickets, inningBallsRemaining, newestBatterOrd, otherBatterOrd))
        situations.append((runsRequired, totalInningWickets, inningBallsRemaining, otherBatterOrd, newestBatterOrd))
situations = pd.DataFrame(situations, columns=stateColumns).astype(np.int64)

runGrid = np.arange(max(60, int(situations['runsRequired'].max())) + 1)
# The final-over integration need not grow with the 120-ball target cap.
endgameRunGrid = np.arange(61)
lookupIndex = np.full((10, 121, 12, len(runGrid)), -1, dtype=np.int32)
for (wickets, balls, striker), group in masterLookup.groupby(['totalInningWickets', 'inningBallsRemaining', 'ord'], sort=False):
    group = group.sort_values('runsRequired')
    availableRuns = group['runsRequired'].to_numpy()
    nearest = np.abs(availableRuns[:, None] - runGrid[None, :]).argmin(axis=0)
    lookupIndex[int(wickets), int(balls), int(striker), :] = group.index.to_numpy()[nearest]


for wickets in range(10):
    if (lookupIndex[wickets, 1:121, 1:wickets + 3, :] < 0).any():
        raise ValueError('The ball lookup does not cover all reachable transition states.')

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

evaluationData = trainData.loc[validPair & trainData['inningBallsRemaining'].between(1, 120), rowColumns + stateColumns + ['chaseWin']].copy()
eligibleInningsRows = len(evaluationData)
gridIndex = situations.copy()
gridIndex['stateIndex'] = np.arange(len(gridIndex))
evaluationData = evaluationData.merge(gridIndex, on=stateColumns, how='inner', validate='many_to_one')
evaluationData = evaluationData.reset_index(drop=True)
print(f'Historical simulation coverage: {len(evaluationData):,} of {eligibleInningsRows:,} valid-pair full-innings rows are on the expanded grid.', flush=True)
print(f'{eligibleInningsRows - len(evaluationData):,} off-grid rows remain outside simulation/blend coverage; direct OOF includes valid-pair off-grid states.', flush=True)

@njit(cache=True)
def exact_endgame(eventProbabilities, extras, maximumInvalid=12):
    # Integrate the existing extras rules, including their fractional means.
    # No discrete extras distribution is invented. Up to 12 invalid balls are
    # integrated; the omitted path probability is bounded separately.
    bye, wicketInvalid, invalidProbability, invalidRuns = extras
    maximumRuns = eventProbabilities.shape[3] - 1
    pairIndex = np.full((10, 12, 12), -1, dtype=np.int64)
    pairs = np.zeros((440, 3), dtype=np.int64)
    position = 0
    for wickets in range(10):
        for striker in range(1, wickets + 3):
            for nonstriker in range(1, wickets + 3):
                if striker != nonstriker:
                    pairIndex[wickets, striker, nonstriker] = position
                    pairs[position, 0] = wickets
                    pairs[position, 1] = striker
                    pairs[position, 2] = nonstriker
                    position += 1
    legalNext = np.full((440, 13), -1, dtype=np.int64)
    invalidNext = np.full(440, -1, dtype=np.int64)
    for position in range(440):
        wickets, striker, nonstriker = pairs[position]
        if wickets < 9:
            invalidNext[position] = pairIndex[wickets + 1, wickets + 3, nonstriker]
        for event in range(13):
            runs, wicket = jointEvents[event]
            if wickets + wicket >= 10:
                continue
            nextStriker = wickets + 3 if wicket else striker
            nextNonStriker = nonstriker
            if runs % 2:
                nextStriker, nextNonStriker = nextNonStriker, nextStriker
            legalNext[position, event] = pairIndex[wickets + wicket, nextStriker, nextNonStriker]
    result = np.zeros((7, maximumRuns + 1, 440, 4))
    for startingBalls in range(1, 7):
        table = np.zeros((startingBalls + 1, maximumInvalid + 2, maximumRuns + 1, 440, 4))
        for balls in range(startingBalls + 1):
            for invalidCount in range(maximumInvalid + 1, -1, -1):
                extrasSoFar = bye * (startingBalls - balls + invalidCount) + invalidRuns * invalidCount
                for runIndex in range(maximumRuns + 1):
                    required = runIndex - extrasSoFar
                    if required <= 0:
                        table[balls, invalidCount, runIndex, :, 0] = 1.0
                        continue
                    if balls == 0 or invalidCount > maximumInvalid:
                        continue
                    lookupRuns = min(max(int(np.ceil(required)), 1), maximumRuns)
                    for position in range(440):
                        wickets, striker, nonstriker = pairs[position]
                        for invalidWicket in range(2):
                            probability = invalidProbability * (wicketInvalid if invalidWicket else 1 - wicketInvalid)
                            futurePair = invalidNext[position] if invalidWicket else position
                            table[balls, invalidCount, runIndex, position, 1] += probability * (invalidRuns + bye)
                            table[balls, invalidCount, runIndex, position, 3] += probability * invalidWicket
                            if required <= invalidRuns + bye:
                                table[balls, invalidCount, runIndex, position, 0] += probability
                            elif futurePair >= 0:
                                for metric in range(4):
                                    table[balls, invalidCount, runIndex, position, metric] += probability * table[balls, invalidCount + 1, runIndex, futurePair, metric]
                        for event in range(13):
                            probability = (1 - invalidProbability) * eventProbabilities[wickets, balls, striker, lookupRuns, event]
                            if probability == 0:
                                continue
                            runs, wicket = jointEvents[event]
                            table[balls, invalidCount, runIndex, position, 1] += probability * (runs + bye)
                            table[balls, invalidCount, runIndex, position, 2] += probability
                            table[balls, invalidCount, runIndex, position, 3] += probability * wicket
                            if required <= runs + bye:
                                table[balls, invalidCount, runIndex, position, 0] += probability
                            elif balls > 1 and legalNext[position, event] >= 0:
                                for metric in range(4):
                                    table[balls, invalidCount, runIndex, position, metric] += probability * table[balls - 1, invalidCount, runIndex - runs, legalNext[position, event], metric]
        result[startingBalls] = table[startingBalls, 0]
    return result, pairIndex, pairs


@njit(parallel=True, cache=True)
def simulate_batch(states, lookupIndex, eventCdf, jointEvents, extraValues, numberOfSims, stateIndices, randomSeed):
    results = np.zeros((len(states), 4), dtype=np.float64)
    for index in prange(len(states)):
        np.random.seed(randomSeed + stateIndices[index])
        wins = 0
        runsTotal = 0.0
        ballsTotal = 0
        wicketsTotal = 0
        for sim in range(numberOfSims):
            runsRequired = float(states[index, 0])
            totalInningWickets = states[index, 1]
            inningBallsRemaining = states[index, 2]
            strikerOrd = states[index, 3]
            nonStrikerOrd = states[index, 4]
            runsScoredSim = 0.0
            ballsUsedSim = 0
            wicketsStart = totalInningWickets
            while runsRequired > 0 and inningBallsRemaining > 0 and totalInningWickets < 10:
                if lookupIndex.ndim == 4:
                    rowIndex = lookupIndex[totalInningWickets, inningBallsRemaining, strikerOrd, int(np.ceil(runsRequired))]
                else:
                    rowIndex = lookupIndex[totalInningWickets, inningBallsRemaining, strikerOrd, nonStrikerOrd, int(np.ceil(runsRequired))]
                if np.random.rand() < extraValues[rowIndex, 2]:
                    totalRunsBall = extraValues[rowIndex, 3] + extraValues[rowIndex, 0]
                    if np.random.rand() < extraValues[rowIndex, 1]:
                        totalInningWickets += 1
                        if totalInningWickets < 10:
                            strikerOrd = totalInningWickets + 2
                else:
                    draw = np.random.rand()
                    eventIndex = 0
                    while eventIndex < len(jointEvents) - 1 and draw >= eventCdf[rowIndex, eventIndex]:
                        eventIndex += 1
                    batsmanRun = jointEvents[eventIndex, 0]
                    isWicket = jointEvents[eventIndex, 1]
                    totalRunsBall = batsmanRun + extraValues[rowIndex, 0]
                    if isWicket == 1:
                        totalInningWickets += 1
                        if totalInningWickets < 10:
                            strikerOrd = totalInningWickets + 2
                    if batsmanRun % 2 == 1 and totalInningWickets < 10:
                        strikerOrd, nonStrikerOrd = nonStrikerOrd, strikerOrd
                    inningBallsRemaining -= 1
                    ballsUsedSim += 1
                    if inningBallsRemaining > 0 and inningBallsRemaining % 6 == 0 and totalInningWickets < 10:
                        strikerOrd, nonStrikerOrd = nonStrikerOrd, strikerOrd
                runsRequired -= totalRunsBall
                runsScoredSim += totalRunsBall
            wins += int(runsRequired <= 0)
            runsTotal += runsScoredSim
            ballsTotal += ballsUsedSim
            wicketsTotal += totalInningWickets - wicketsStart
        results[index, 0] = wins / numberOfSims
        results[index, 1] = runsTotal / numberOfSims
        results[index, 2] = ballsTotal / numberOfSims
        results[index, 3] = wicketsTotal / numberOfSims
    return results

# --------------------------------------------------
# SIMULATE EACH SET OF EXCLUDED MATCH FOLDS
# --------------------------------------------------
jointEvents = np.array([(runs, wicket) for runs in range(7) for wicket in range(2) if (runs, wicket) != (6, 1)], dtype=np.int64)
jointColumns = [f'pressure_{runs}_' + ('wicket' if wicket else 'noWicket') for runs, wicket in jointEvents]
resultColumns = ['sim_chaseWin%', 'sim_meanRunsScored', 'sim_meanBallsUsed', 'sim_meanWicketsLost']
oofFrames = []
# Five outer fits, ten reused inner fits, then the full-data production fit.
# An inner fit excludes BOTH listed folds. It can serve either as the outer
# test fold, with the other as an inner validation fold, without refitting.
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
    if foldA == -1:
        ballProbabilities = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/deathBallProbs.csv', float_precision='round_trip')
    else:
        ballProbabilities = pd.read_csv(PROJECT_ROOT / f'men/matchMarket/hybridWin3/cv/deathBallProbs_{jobName}.csv', float_precision='round_trip')
    aligned = masterLookup[ballColumns].merge(ballProbabilities[ballColumns + jointColumns], on=ballColumns, how='left', validate='one_to_one', sort=False)
    eventProbs = aligned[jointColumns].to_numpy(np.float64)
    if not np.isfinite(eventProbs).all() or (eventProbs < 0).any() or (eventProbs.sum(axis=1) <= 0).any():
        raise ValueError(f'Missing or invalid ball probabilities for {jobName}.')
    eventCdf = np.ascontiguousarray(np.cumsum(eventProbs / eventProbs.sum(axis=1, keepdims=True), axis=1))
    eventCdf[:, -1] = 1.0
    # The fixed extras lookup is constant throughout the final over. This
    # lets us sum every legal-ball outcome and invalid-delivery branch.
    lastOverExtras = masterLookup.loc[masterLookup['inningBallsRemaining'].between(1, 6), extraColumns].drop_duplicates()
    if len(lastOverExtras) != 1:
        raise ValueError('Exact final-over calculation requires the supplied constant final-over extras rates.')
    endgameExtras = lastOverExtras.iloc[0].to_numpy(float)
    endgameErrorBound = float(betainc(13, 6, endgameExtras[2]))
    if endgameErrorBound > 1e-10:
        raise ValueError('Invalid-delivery truncation is too large for the exact endgame settings.')
    shortProbabilities = np.zeros((10, 7, 12, len(endgameRunGrid), 13))
    for wickets in range(10):
        for balls in range(1, 7):
            for striker in range(1, wickets + 3):
                shortProbabilities[wickets, balls, striker] = eventProbs[lookupIndex[wickets, balls, striker, :len(endgameRunGrid)]]
    endgameValues, endgamePairIndex, endgamePairs = exact_endgame(shortProbabilities, endgameExtras)
    endgameStates = np.array([
        (runs, wickets, balls, striker, nonstriker)
        for balls in range(1, 7)
        for runs in range(1, len(endgameRunGrid))
        for wickets, striker, nonstriker in endgamePairs
    ], dtype=np.int64)
    endgameOutput = pd.DataFrame(endgameStates, columns=stateColumns)
    endRuns, endWickets, endBalls, endStriker, endNonStriker = endgameStates.T
    endgameOutput['endgameWinProbability'] = endgameValues[endBalls, endRuns, endgamePairIndex[endWickets, endStriker, endNonStriker], 0]
    if foldA == -1:
        endgameOutput.to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/endgameWinProbabilities.csv', index=False)
    else:
        endgameOutput.to_csv(PROJECT_ROOT / f'men/matchMarket/hybridWin3/cv/endgameWinProbabilities_{jobName}.csv', index=False)
    print(f'{jobName}: final-over probabilities integrated; omitted path probability at most {endgameErrorBound:.2e}', flush=True)
    if foldA == -1:
        stateIndices = np.arange(len(situations), dtype=np.int64)
    else:
        stateIndices = np.sort(evaluationData.loc[evaluationData['fold'].isin(excludedFolds), 'stateIndex'].unique().astype(np.int64))
    states = np.ascontiguousarray(situations.iloc[stateIndices][stateColumns].to_numpy(np.int64))
    results = np.empty((len(states), 4), dtype=np.float64)
    started = time.perf_counter()
    print(f'{jobName}: {len(states):,} states, {NUMBER_OF_SIMS:,} paths per state', flush=True)
    for startIndex in range(0, len(states), BATCH_SIZE):
        endIndex = min(startIndex + BATCH_SIZE, len(states))
        batchStates = states[startIndex:endIndex]
        shortMask = batchStates[:, 2] <= 6
        batchResults = np.empty((len(batchStates), 4), dtype=np.float64)
        if shortMask.any():
            runs, wickets, balls, striker, nonstriker = batchStates[shortMask].T
            batchResults[shortMask] = endgameValues[balls, runs, endgamePairIndex[wickets, striker, nonstriker]]
        if (~shortMask).any():
            batchResults[~shortMask] = simulate_batch(
                np.ascontiguousarray(batchStates[~shortMask]), lookupIndex, eventCdf, jointEvents,
                extraValues, NUMBER_OF_SIMS, np.ascontiguousarray(stateIndices[startIndex:endIndex][~shortMask]), RANDOM_SEED,
            )
        results[startIndex:endIndex] = batchResults
        elapsed = time.perf_counter() - started
        remaining = elapsed / endIndex * (len(states) - endIndex)
        print(f'{jobName}: {endIndex:,} of {len(states):,} states | {elapsed / 60:.1f} min elapsed | {remaining / 60:.1f} min remaining', flush=True)
    output = situations.iloc[stateIndices].copy()
    output[resultColumns] = results
    output['simSample'] = np.where(output['inningBallsRemaining'].le(6), 0, NUMBER_OF_SIMS)
    output['simulationMethod'] = np.where(output['inningBallsRemaining'].le(6), 'exact final over', 'Monte Carlo')
    if foldA == -1:
        output.to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/chaseSimSituations.csv', index=False)
    else:
        output.to_csv(PROJECT_ROOT / f'men/matchMarket/hybridWin3/cv/chaseSimSituations_{jobName}.csv', index=False)
    if foldA >= 0 and foldB == -1:
        rows = evaluationData.loc[evaluationData['fold'].eq(foldA), rowColumns + stateColumns + ['chaseWin']]
        oofFrames.append(rows.merge(output, on=stateColumns, how='left', validate='many_to_one'))

oof = pd.concat(oofFrames, ignore_index=True).sort_values(['matchID', 'ID']).reset_index(drop=True)
if len(oof) != len(evaluationData) or oof.duplicated(['matchID', 'ID']).any() or oof[resultColumns].isna().any().any():
    raise RuntimeError('The simulation does not have exactly one unseen prediction per eligible evaluation row.')
oof.to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/chaseSimSituations_oof.csv', index=False)
runInfo.at[0, 'simulationReady'] = 1
runInfo['endgameReady'] = 1
runInfo['simulationResourceHash'] = hashlib.sha256((PROJECT_ROOT / 'men/matchMarket/hybridWin3/auxiliaries/6_masterLookup.csv').read_bytes()).hexdigest()
runInfo['endgameErrorBound'] = endgameErrorBound
runInfo['simulationRecipe'] = 'neural_grid_above24_exact_final_over'
runInfo.at[0, 'numberOfSims'] = NUMBER_OF_SIMS
runInfo.at[0, 'simulationSeed'] = RANDOM_SEED
runInfo.at[0, 'eligibleInningsRows'] = eligibleInningsRows
runInfo.at[0, 'hybridOofRows'] = len(evaluationData)
runInfo.to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/cv/runInfo.csv', index=False)
print(f'Exported {len(oof):,} unseen historical predictions and {len(situations):,} production states.')
print(f'Total simulation runtime: {(time.perf_counter() - simulationStartTime) / 60:.1f} minutes')
