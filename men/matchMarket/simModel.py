import pandas as pd
import numpy as np
from paths import PROJECT_ROOT


# settings
NUMBER_OF_SIMS = 1000


# import
masterLookup = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/outputs/5_masterLookup.csv')



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


# create approximately 5,000 chase situations centred around plausible 20-80% states
situations = []

for inningBallsRemaining in range(1, 25):
    for totalInningWickets in range(10):

        centreRunsRequired = inningBallsRemaining * runsPerBall[totalInningWickets]
        halfWidth = 5.5 + (0.35 * inningBallsRemaining)

        minRunsRequired = max(1, int(np.floor(centreRunsRequired - halfWidth)))
        maxRunsRequired = int(np.ceil(centreRunsRequired + halfWidth))

        # cap extremely unlikely chase states
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

situations = pd.DataFrame(situations)

print('Total situations:', len(situations))




# the sim only uses wickets lost and ball number from the master lookup
masterLookup = masterLookup.drop_duplicates(subset=['totalInningWickets', 'inningBallNumber']).reset_index(drop=True)


# create a dictionary once so we do not repeatedly filter the dataframe inside every simulation ball
lookup = {}

for _, row in masterLookup.iterrows():
    run_probs = row[[str(i) for i in range(8)]].to_numpy(dtype=float)
    run_probs = run_probs / run_probs.sum()

    lookup[(int(row['totalInningWickets']), int(row['inningBallNumber']))] = {
        'run_probs': run_probs,
        'byeRunsOver': float(row['byeRunsOver']),
        'isWicketInvalidOver': float(row['isWicketInvalidOver']),
        'isInvalidOver': float(row['isInvalidOver']),
        'invalidRunsOver': float(row['invalidRunsOver']),
        'm_isWicketBall': float(row['m_isWicketBall'])
    }


# simulate one chase state
# the only starting-state inputs are runsRequired, totalInningWickets and inningBallsRemaining
def simulate_chase(lookup, runsRequiredSet, totalInningWicketsSet, inningBallsRemainingSet, numberOfSims):
    wins = 0
    runsScored = []
    ballsUsed = []
    wicketsLost = []

    for sim in range(numberOfSims):
        runsRequired = float(runsRequiredSet)
        totalInningWickets = int(totalInningWicketsSet)
        inningBallsRemaining = int(inningBallsRemainingSet)

        runsScoredSim = 0.0
        ballsUsedSim = 0
        wicketsStart = totalInningWickets

        while runsRequired > 0 and inningBallsRemaining > 0 and totalInningWickets < 10:
            inningBallNumber = 121 - inningBallsRemaining
            masterLookupRow = lookup.get((totalInningWickets, inningBallNumber))

            if masterLookupRow is None:
                break

            totalRunsBall = 0.0

            # invalid delivery: same logic as the first-innings sim, so the legal-ball count does not fall
            if np.random.rand() < masterLookupRow['isInvalidOver']:
                invalidRunsBall = masterLookupRow['invalidRunsOver']
                byeRunsBall = masterLookupRow['byeRunsOver']
                totalRunsBall = invalidRunsBall + byeRunsBall

                if np.random.rand() < masterLookupRow['isWicketInvalidOver']:
                    totalInningWickets += 1

            # valid delivery
            else:
                batsmanRun = np.random.choice(range(8), p=masterLookupRow['run_probs'])
                byeRunsBall = masterLookupRow['byeRunsOver']
                totalRunsBall = batsmanRun + byeRunsBall

                if np.random.rand() < masterLookupRow['m_isWicketBall']:
                    totalInningWickets += 1

                inningBallsRemaining -= 1
                ballsUsedSim += 1

            runsRequired -= totalRunsBall
            runsScoredSim += totalRunsBall

            # chase ends immediately once the target has been reached
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


# simulate every chase situation
simulationResults = []

for index, row in situations.iterrows():
    print(
        index + 1,
        'of',
        len(situations),
        int(row['runsRequired']),
        int(row['totalInningWickets']),
        int(row['inningBallsRemaining'])
    )

    result = simulate_chase(
        lookup=lookup,
        runsRequiredSet=row['runsRequired'],
        totalInningWicketsSet=row['totalInningWickets'],
        inningBallsRemainingSet=row['inningBallsRemaining'],
        numberOfSims=NUMBER_OF_SIMS
    )

    simulationResults.append({
        'runsRequired': row['runsRequired'],
        'totalInningWickets': row['totalInningWickets'],
        'inningBallsRemaining': row['inningBallsRemaining'],
        **result
    })


simulationResults = pd.DataFrame(simulationResults)


# export simulation results
simulationResults.to_csv(
    PROJECT_ROOT / 'men/matchMarket/outputs/chaseSimSituations.csv',
    index=False
)

