import pandas as pd
import numpy as np
from paths import PROJECT_ROOT
import time

simulationStartTime = time.perf_counter()


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
NUMBER_OF_SIMS = 1000

# --------------------------------------------------
# IMPORT
# --------------------------------------------------
masterLookup = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/outputs/deathBallProbs.csv')
oldMasterLookup = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/outputs/1_masterLookup.csv')

# --------------------------------------------------
# ADD INVALID-DELIVERY VARIABLES FROM OLD LOOKUP
# --------------------------------------------------
invalidLookup = oldMasterLookup[['totalInningWickets', 'inningBallNumber', 'byeRunsOver', 'isWicketInvalidOver', 'isInvalidOver', 'invalidRunsOver']]
invalidLookup = invalidLookup.drop_duplicates(subset=['totalInningWickets', 'inningBallNumber'])
masterLookup = masterLookup.merge(invalidLookup, how='left', on=['totalInningWickets', 'inningBallNumber'])

# --------------------------------------------------
# CREATE CHASE SITUATIONS
# --------------------------------------------------
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
            newestBatterOrd = totalInningWickets + 2
            for otherBatterOrd in range(1, newestBatterOrd):
                situations.append({
                    'runsRequired': runsRequired,
                    'totalInningWickets': totalInningWickets,
                    'inningBallsRemaining': inningBallsRemaining,
                    'ord': newestBatterOrd,
                    'nonStrikerOrd': otherBatterOrd
                })
                situations.append({
                    'runsRequired': runsRequired,
                    'totalInningWickets': totalInningWickets,
                    'inningBallsRemaining': inningBallsRemaining,
                    'ord': otherBatterOrd,
                    'nonStrikerOrd': newestBatterOrd
                })
situations = pd.DataFrame(situations)
print('Total situations:', len(situations))

# --------------------------------------------------
# JOINT LEGAL-BALL EVENTS
# --------------------------------------------------
jointEvents = []
jointColumns = []
for batsmanRun in range(8):
    if f'pressure_{batsmanRun}_noWicket' in masterLookup.columns:
        jointEvents.append((batsmanRun, 0))
        jointColumns.append(f'pressure_{batsmanRun}_noWicket')
    if f'pressure_{batsmanRun}_wicket' in masterLookup.columns:
        jointEvents.append((batsmanRun, 1))
        jointColumns.append(f'pressure_{batsmanRun}_wicket')

# --------------------------------------------------
# CREATE LOOKUP
# --------------------------------------------------
lookup = {}
availableRunsLookup = {}
for _, row in masterLookup.iterrows():
    event_probs = row[jointColumns].to_numpy(dtype=float)
    event_probs = event_probs / event_probs.sum()
    runsRequired = int(row['runsRequired'])
    totalInningWickets = int(row['totalInningWickets'])
    inningBallsRemaining = int(row['inningBallsRemaining'])
    ord = int(row['ord'])
    lookup[(runsRequired, totalInningWickets, inningBallsRemaining, ord)] = {
        'event_probs': event_probs,
        'byeRunsOver': float(row['byeRunsOver']),
        'isWicketInvalidOver': float(row['isWicketInvalidOver']),
        'isInvalidOver': float(row['isInvalidOver']),
        'invalidRunsOver': float(row['invalidRunsOver'])
    }
    availableRunsLookup.setdefault((totalInningWickets, inningBallsRemaining, ord), []).append(runsRequired)
for key in availableRunsLookup:
    availableRunsLookup[key] = np.array(sorted(set(availableRunsLookup[key])))

# --------------------------------------------------
# LOOKUP ROW
# --------------------------------------------------
def get_lookup_row(lookup, availableRunsLookup, runsRequired, totalInningWickets, inningBallsRemaining, ord):
    runsRequired = int(np.ceil(runsRequired))
    masterLookupRow = lookup.get((runsRequired, totalInningWickets, inningBallsRemaining, ord))
    if masterLookupRow is not None:
        return masterLookupRow
    availableRuns = availableRunsLookup.get((totalInningWickets, inningBallsRemaining, ord))
    if availableRuns is None:
        return None
    nearestRunsRequired = availableRuns[np.argmin(np.abs(availableRuns - runsRequired))]
    return lookup[(int(nearestRunsRequired), totalInningWickets, inningBallsRemaining, ord)]

# --------------------------------------------------
# SIMULATE ONE CHASE STATE
# --------------------------------------------------
def simulate_chase(lookup, availableRunsLookup, jointEvents, runsRequiredSet, totalInningWicketsSet, inningBallsRemainingSet, ordSet, nonStrikerOrdSet, numberOfSims):
    wins = 0
    runsScored = []
    ballsUsed = []
    wicketsLost = []
    for sim in range(numberOfSims):
        runsRequired = float(runsRequiredSet)
        totalInningWickets = int(totalInningWicketsSet)
        inningBallsRemaining = int(inningBallsRemainingSet)
        strikerOrd = int(ordSet)
        nonStrikerOrd = int(nonStrikerOrdSet)
        runsScoredSim = 0.0
        ballsUsedSim = 0
        wicketsStart = totalInningWickets
        while runsRequired > 0 and inningBallsRemaining > 0 and totalInningWickets < 10:
            masterLookupRow = get_lookup_row(lookup, availableRunsLookup, runsRequired, totalInningWickets, inningBallsRemaining, strikerOrd)
            if masterLookupRow is None:
                break
            totalRunsBall = 0.0
            if np.random.rand() < masterLookupRow['isInvalidOver']:
                invalidRunsBall = masterLookupRow['invalidRunsOver']
                byeRunsBall = masterLookupRow['byeRunsOver']
                totalRunsBall = invalidRunsBall + byeRunsBall
                if np.random.rand() < masterLookupRow['isWicketInvalidOver']:
                    totalInningWickets += 1
                    if totalInningWickets < 10:
                        strikerOrd = totalInningWickets + 2
            else:
                eventIndex = np.random.choice(len(jointEvents), p=masterLookupRow['event_probs'])
                batsmanRun, isWicket = jointEvents[eventIndex]
                byeRunsBall = masterLookupRow['byeRunsOver']
                totalRunsBall = batsmanRun + byeRunsBall
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
            if runsRequired <= 0:
                break
        isWin = int(runsRequired <= 0)
        wins += isWin
        runsScored.append(runsScoredSim)
        ballsUsed.append(ballsUsedSim)
        wicketsLost.append(totalInningWickets - wicketsStart)
    return {
        'sim_chaseWin%': wins / numberOfSims,
        'sim_meanRunsScored': np.mean(runsScored),
        'sim_meanBallsUsed': np.mean(ballsUsed),
        'sim_meanWicketsLost': np.mean(wicketsLost),
        'simSample': numberOfSims
    }

# --------------------------------------------------
# SIMULATE EVERY CHASE SITUATION
# --------------------------------------------------
simulationResults = []
for index, row in situations.iterrows():
    print(index + 1, 'of', len(situations), int(row['runsRequired']), int(row['totalInningWickets']), int(row['inningBallsRemaining']), int(row['ord']), int(row['nonStrikerOrd']))
    result = simulate_chase(
        lookup=lookup,
        availableRunsLookup=availableRunsLookup,
        jointEvents=jointEvents,
        runsRequiredSet=row['runsRequired'],
        totalInningWicketsSet=row['totalInningWickets'],
        inningBallsRemainingSet=row['inningBallsRemaining'],
        ordSet=row['ord'],
        nonStrikerOrdSet=row['nonStrikerOrd'],
        numberOfSims=NUMBER_OF_SIMS
    )
    simulationResults.append({
        'runsRequired': row['runsRequired'],
        'totalInningWickets': row['totalInningWickets'],
        'inningBallsRemaining': row['inningBallsRemaining'],
        'ord': row['ord'],
        'nonStrikerOrd': row['nonStrikerOrd'],
        **result
    })
simulationResults = pd.DataFrame(simulationResults)

# --------------------------------------------------
# EXPORT
# --------------------------------------------------
simulationResults.to_csv(PROJECT_ROOT / 'men/matchMarket/outputs/chaseSimSituations.csv', index=False)



simulationRuntimeSeconds = (
    time.perf_counter() - simulationStartTime
)

print(f'Total simulation runtime: {simulationRuntimeSeconds:.2f} seconds')
print(f'Total simulation runtime: {simulationRuntimeSeconds / 60:.2f} minutes')


