import pandas as pd
import numpy as np
import random
import itertools
from scipy.stats import kurtosis
from paths import PROJECT_ROOT

# faster version of 3_runsToComeSimClassOrd.py - same sim logic, but the per-ball pandas filtering is replaced with
# plain python list lookups built once up front, which is what made the original take days to run

# import files
masterLookup = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/outputs/2_masterLookup.csv')
rateTrajectoryAdjustments = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/auxiliaries/rateTrajectoryAdjustments.csv')
fxbXslw = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/auxiliaries/fxbXslw.csv')

# the master lookup has values for ord and year, we don't want those for the sim, so drop them
masterLookup = masterLookup.drop_duplicates(subset=['totalInningWickets', 'inningBallNumber']).reset_index(drop=True)
masterLookup = masterLookup.merge(rateTrajectoryAdjustments, how='left', on=['inningBallNumber'])

# we start with 100k sims of the whole innings then for 5 wickets and higher do a sim from the inningball number closest to 50 samples
toSim = masterLookup[(masterLookup['sample'] < 50) & (masterLookup['totalInningWickets'] >= 3)].drop_duplicates(subset=['totalInningWickets'], keep='last').reset_index(drop=True)
toSim = toSim[['totalInningWickets', 'inningBallNumber']]


# starting crease states for the top-up sims, taken from real first innings data. At W wickets the recent arrival is ord W+2,
# the survivor is whoever else is still in - survivorOrderByWickets.csv only holds the average survivor ord, so we build the
# full joint distribution here (survivor ord, both batters' balls faced, balls since last wicket, who's on strike) and
# sample whole real states from it, which keeps all of those consistent with each other
realStates = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/data/dataClean.csv',
                         usecols=['matchID', 'inningNumber', 'batsmanName', 'nonstrikerName', 'ord', 'totalInningWickets',
                                  'inningBallNumber', 'isValid', 'batsmanBallsFaced'])
realStates = realStates[realStates['inningNumber'] == 1]
inningKeys = ['matchID', 'inningNumber']
ordLookup = realStates[inningKeys + ['batsmanName', 'ord']].drop_duplicates(subset=inningKeys + ['batsmanName'])
realStates = realStates[realStates['isValid'] == 1].copy()
# batsmanBallsFaced includes the current ball, we want balls faced before it
realStates['strikerBF'] = realStates['batsmanBallsFaced'] - 1
realStates = realStates.merge(ordLookup.rename(columns={'batsmanName': 'nonstrikerName', 'ord': 'nonStrikerOrd'}),
                              how='left', on=inningKeys + ['nonstrikerName'])
# non striker's balls faced = their count as at the last ball they faced before this one
nonStrikerFaced = realStates[inningKeys + ['batsmanName', 'inningBallNumber', 'batsmanBallsFaced']].rename(
    columns={'batsmanName': 'nonstrikerName', 'batsmanBallsFaced': 'nonStrikerBF'}).sort_values('inningBallNumber')
realStates = pd.merge_asof(realStates.sort_values('inningBallNumber'), nonStrikerFaced, on='inningBallNumber',
                           by=inningKeys + ['nonstrikerName'], allow_exact_matches=False)
realStates['nonStrikerBF'] = realStates['nonStrikerBF'].fillna(0)
# valid balls already bowled at this wickets level
realStates['ballsSinceLastWicket'] = realStates['inningBallNumber'] - realStates.groupby(inningKeys + ['totalInningWickets'])['inningBallNumber'].transform('min')
realStates = realStates.dropna(subset=['nonStrikerOrd']).reset_index(drop=True)

arrivalOrd = realStates['totalInningWickets'] + 2
strikerIsArrival = realStates['ord'] == arrivalOrd
realStates['survivorOrd'] = np.where(strikerIsArrival, realStates['nonStrikerOrd'], realStates['ord'])
realStates['arrivalBF'] = np.where(strikerIsArrival, realStates['strikerBF'], realStates['nonStrikerBF'])
# keep rows where exactly one batter is the recent arrival and the rest of the state is internally consistent (drops data noise)
realStates = realStates[(strikerIsArrival | (realStates['nonStrikerOrd'] == arrivalOrd)) &
                        (realStates['survivorOrd'] >= 1) & (realStates['survivorOrd'] <= realStates['totalInningWickets'] + 1) &
                        (realStates['arrivalBF'] <= realStates['ballsSinceLastWicket'])]

startStates = {}
for w, b in toSim.itertuples(index=False):
    w, b = int(w), int(b)
    # widen the ball window around the start ball until we have a decent sample of real states
    for window in [3, 6, 10, 15, 25]:
        s = realStates[(realStates['totalInningWickets'] == w) & (realStates['inningBallNumber'].between(b - window, b + window))]
        if len(s) >= 300:
            break
    startStates[(w, b)] = s[['ord', 'strikerBF', 'nonStrikerOrd', 'nonStrikerBF', 'ballsSinceLastWicket']].to_numpy(dtype=np.int64)
    print(f'start states for {w} wkts, ball {b}: {len(s)} real states (ball window ±{window})')
del realStates, ordLookup, nonStrikerFaced


# build the lookups once: masterLookup indexed [wickets][inningBallNumber] -> tuple of the values the sim needs, None if the situation doesn't exist
mlCols = ['predTotalInningRuns', 'byeRunsOver', 'isWicketInvalidOver', 'isInvalidOver', 'overNumber', 'm_isWicketBall',
          'X_wr_balls_smooth', 'X_rr_balls_smooth', 'invalidRunsOver']
mlLookup = [[None] * 121 for _ in range(10)]
runProbs = masterLookup[[str(i) for i in range(8)]].to_numpy(dtype=float)
runCdfs = np.cumsum(runProbs, axis=1)
runCdfs = runCdfs / runCdfs[:, [-1]]
for i, row in enumerate(masterLookup[['totalInningWickets', 'inningBallNumber'] + mlCols].itertuples(index=False)):
    w, b = int(row[0]), int(row[1])
    mlLookup[w][b] = tuple(row[2:]) + (runCdfs[i].tolist(),)

# fxbXslw indexed by (ord, over, ballsFaced) -> (fxbRuns, fxbWkts, slwRuns, slwWkts)
fxbLookup = {(int(r.ord), int(r.over), int(r.ballsFaced)): (r.fxbRunscurve, r.fxbWktscurve, r.slwRunscurve, r.slwWktscurve)
             for r in fxbXslw.itertuples(index=False)}

# unique sim ids across every call (the original used the ms timestamp, which collides once sims run this fast)
simIDCounter = itertools.count()

simColumns = ['simID', 'inningBallNumber', 'overNumber', 'rateTrajectory', 'm_batsmanRunsBall', 'invalidRunsBall', 'byeRunsOver', 'isWicket', 'isWicketInvalid',
              'totalInningBatRuns', 'totalInningInvalidRuns', 'totalInningByes', 'totalInningWickets',
              'm_isWicketBall', 'isWicketInvalidOver', 'isInvalidOver', 'strikerID', 'nonStrikerID', 'strikerBallsFaced', 'nonStrikerBallsFaced',
              'strikerRuns', 'nonStrikerRuns', 'ballsSinceLastWicket', 'fxbXslwRunsCurve', 'fxbXslwWktsCurve']
intColumns = ['simID', 'inningBallNumber', 'isWicket', 'isWicketInvalid', 'totalInningWickets', 'strikerID', 'nonStrikerID',
              'strikerBallsFaced', 'nonStrikerBallsFaced', 'ballsSinceLastWicket']


def simulate_innings(ballsRemainingSet, totalInningWicketsSet, numberOfSims, startStates=None):
    rand = random.random
    chunks = []
    simulationResults = []
    totalInningWicketsSet = int(totalInningWicketsSet)
    ballsRemainingSet = int(ballsRemainingSet)

    # pre draw every sim's starting crease state here so the sim loop only has to unpack a tuple
    # (strikerOrd, strikerBF, nonStrikerOrd, nonStrikerBF, ballsSinceLastWicket) - openers on 0 balls for a full innings
    if startStates is None:
        simStarts = [(1, 0, 2, 0, 0)] * numberOfSims
    else:
        simStarts = startStates[np.random.randint(len(startStates), size=numberOfSims)].tolist()
    # start part way through the over if the start ball isn't the first ball of one
    validBallsInOverSet = (120 - ballsRemainingSet) % 6

    for sim in range(numberOfSims):
        if sim % 10000 == 0:
            print(sim, totalInningWicketsSet)
        simID = next(simIDCounter)
        strikerOrdStart, strikerBFStart, nonStrikerOrdStart, nonStrikerBFStart, ballsSinceLastWicket = simStarts[sim]

        totalInningBatRuns, totalInningByes, totalInningInvalidRuns = 0, 0, 0
        totalInningWickets, ballsRemaining = totalInningWicketsSet, ballsRemainingSet
        validBallsInOver = validBallsInOverSet

        # new batters come in as max(id) + 1, so with the arrival at W+2 the next one in is W+3 as it should be
        batsmen = [{'id': strikerOrdStart, 'runs': 0, 'ballsFaced': strikerBFStart},
                   {'id': nonStrikerOrdStart, 'runs': 0, 'ballsFaced': nonStrikerBFStart}]
        striker_idx, non_striker_idx = 0, 1

        while ballsRemaining > 0 and totalInningWickets < 10:
            inningBallNumber = 121 - ballsRemaining
            ml = mlLookup[totalInningWickets][inningBallNumber]

            if ml is None:
                break

            (predTotalInningRuns, byeRunsOver, isWicketInvalidOver, isInvalidOver, current_over,
             m_isWicketBall, X_wr, X_rr, invalidRunsOver, runCdf) = ml

            totalInningRuns = totalInningBatRuns + totalInningByes + totalInningInvalidRuns
            # mirror numpy division from the original: x/0 gives nan or inf rather than raising
            if predTotalInningRuns == 0:
                rateTrajectory = float('nan') if totalInningRuns == 0 else float('inf')
            else:
                rateTrajectory = totalInningRuns / predTotalInningRuns

            batsmanRunsBall = 0
            invalidRunsBall = 0
            isWicket = 0
            isInvalidWicket = 0

            striker, non_striker = batsmen[striker_idx], batsmen[non_striker_idx]
            striker_id_before = striker['id']
            striker_balls_before = striker['ballsFaced']
            striker_runs_before = striker['runs']
            non_striker_id = non_striker['id']
            non_striker_balls = non_striker['ballsFaced']
            non_striker_runs = non_striker['runs']

            overKey = int(current_over)
            fxbRow = fxbLookup.get((striker_id_before, overKey, striker_balls_before))
            slwRow = fxbLookup.get((striker_id_before, overKey, ballsSinceLastWicket))

            if fxbRow is None or slwRow is None:
                run_adjust, wkt_adjust = 1, 1
            elif striker_balls_before < non_striker_balls:
                run_adjust = fxbRow[0]
                wkt_adjust = fxbRow[1]
            else:
                run_adjust = fxbRow[0] * slwRow[2]
                wkt_adjust = fxbRow[1] * slwRow[3]

            isWicketBall = wkt_adjust * m_isWicketBall
            isWicketBall = (((rateTrajectory - 1) * X_wr) + 1) * isWicketBall

            if rand() < isInvalidOver:
                invalidRunsBall = (((rateTrajectory - 1) * X_rr) + 1) * invalidRunsOver
                totalInningInvalidRuns += invalidRunsBall
                totalInningByes += byeRunsOver

                if rand() < isWicketInvalidOver:
                    isInvalidWicket = 1
                    totalInningWickets += 1

            else:
                striker['ballsFaced'] += 1
                validBallsInOver += 1

                # inverse cdf draw, same distribution as np.random.choice(range(8), p=run_probs)
                u = rand()
                batsmanRun = 0
                while batsmanRun < 7 and u >= runCdf[batsmanRun]:
                    batsmanRun += 1
                batsmanRunsBall = batsmanRun * run_adjust
                batsmanRunsBall = (((rateTrajectory - 1) * X_rr) + 1) * batsmanRunsBall

                striker['runs'] += batsmanRunsBall
                totalInningBatRuns += batsmanRunsBall
                totalInningByes += byeRunsOver

                if rand() < isWicketBall:
                    isWicket = 1
                    totalInningWickets += 1
                    new_batsman_id = max(b['id'] for b in batsmen) + 1
                    batsmen[striker_idx] = {'id': new_batsman_id, 'runs': 0, 'ballsFaced': 0}
                else:
                    if batsmanRun % 2 == 1:
                        striker_idx, non_striker_idx = non_striker_idx, striker_idx

                ballsRemaining -= 1

                if validBallsInOver == 6:
                    validBallsInOver = 0
                    striker_idx, non_striker_idx = non_striker_idx, striker_idx

                ballsSinceLastWicket = 0 if isWicket else ballsSinceLastWicket + 1

            simulationResults.append((
                simID, inningBallNumber, current_over, rateTrajectory, batsmanRunsBall, invalidRunsBall, byeRunsOver,
                isWicket, isInvalidWicket, totalInningBatRuns, totalInningInvalidRuns, totalInningByes,
                totalInningWickets, isWicketBall, isWicketInvalidOver, isInvalidOver,
                striker_id_before, non_striker_id,
                striker_balls_before, non_striker_balls,
                striker_runs_before, non_striker_runs,
                ballsSinceLastWicket, run_adjust, wkt_adjust
            ))

        # flush to numpy periodically so we don't hold millions of python tuples in memory
        if len(simulationResults) > 500000:
            chunks.append(np.array(simulationResults, dtype=np.float64))
            simulationResults = []

    if simulationResults:
        chunks.append(np.array(simulationResults, dtype=np.float64))

    allSimBalls = pd.DataFrame(np.concatenate(chunks), columns=simColumns)
    allSimBalls[intColumns] = allSimBalls[intColumns].astype(np.int64)
    allSimBalls['totalRunsBall'] = allSimBalls['m_batsmanRunsBall'] + allSimBalls['invalidRunsBall'] + allSimBalls['byeRunsOver']
    allSimBalls['totalInningRuns'] = allSimBalls['totalInningBatRuns'] + allSimBalls['totalInningInvalidRuns'] + allSimBalls['totalInningByes']


    # create a pivot which aggregates the innings, so we can see the end of innings number of runs for every sim
    simInnings = pd.pivot_table(allSimBalls, values=['totalRunsBall'], index=['simID'], aggfunc='sum')
    simInnings.columns = ['totalInningRunsEnd']

    # create a sample column so we can see sample size for situations
    allSimBalls['sample'] = 1
    # merge the end of innings totals to each sim ball, then we simply work out the difference between this ball and end of innings to see runs to come
    allSimBalls = allSimBalls.merge(simInnings, how='left', on=['simID'])

    # is it a valid ball
    allSimBalls['isValid'] = np.where(allSimBalls['m_batsmanRunsBall'] > 0, 1, 0)
    # is it any kind of wicket, basically total wickets from legit and non legit balls
    allSimBalls['isWicketAny'] = allSimBalls['isWicket'] + allSimBalls['isWicketInvalid']


    # basically if we do the calculation as above it will be for AFTER the ball in question, we want it to be before
    allSimBalls['totalInningRunsToCome'] = allSimBalls['totalInningRunsEnd'] - allSimBalls['totalInningRuns'] + allSimBalls['totalRunsBall']
    allSimBalls['totalInningWickets'] = np.where(allSimBalls['isWicketAny'] == True, allSimBalls['totalInningWickets'] - 1, allSimBalls['totalInningWickets'])


    # simply print the mean of this particular sim run
    print(np.mean(simInnings['totalInningRunsEnd']))

    return allSimBalls


# only the columns the summary needs, keeps memory down across ~16m sim balls
keepColumns = ['simID', 'inningBallNumber', 'totalInningWickets', 'totalInningRunsToCome', 'sample']

# first of all do a sim of the whole innings then do the additional
simParts = [simulate_innings(ballsRemainingSet=120, totalInningWicketsSet=0, numberOfSims=100000)[keepColumns]]

# this buffs up situations where we NEED samples but the sim doesn't get into those situations enough, it makes sure any real life situation with over 50 samples has at least 10k sim samples
for index, row in toSim.iterrows():
    simParts.append(simulate_innings(ballsRemainingSet=(121 - row.iloc[1]), totalInningWicketsSet=row.iloc[0], numberOfSims=10000,
                                     startStates=startStates[(int(row.iloc[0]), int(row.iloc[1]))])[keepColumns])
allSimBalls = pd.concat(simParts, axis=0, ignore_index=True)
del simParts


# summarise by situation rather than exporting every sim ball - same stats 4_runsToComeBiasSplineModel.py builds from the raw sims
# (and a superset of what 7_distribs.py needs), so the downstream files can read this directly
# valid balls faced to come and bowled out come from each sim's last valid ball, same as step 4
allSimBalls['totalInningValidBallsFaced'] = allSimBalls.groupby('simID')['inningBallNumber'].transform('max')
allSimBalls['totalInningValidBallsFacedToCome'] = (allSimBalls['totalInningValidBallsFaced'] + 1) - allSimBalls['inningBallNumber']
allSimBalls['bowledOut'] = np.where(allSimBalls['totalInningValidBallsFaced'] < 120, 1, 0)

simSituationRunsToCome = allSimBalls.groupby(['inningBallNumber', 'totalInningWickets']).agg(
    totalInningRunsToCome=('totalInningRunsToCome', 'mean'),
    sample=('sample', 'sum'),
    totalInningValidBallsFacedToCome=('totalInningValidBallsFacedToCome', 'mean'),
    bowledOut=('bowledOut', 'mean'),
    totalInningRunsToComeSimCount=('totalInningRunsToCome', 'count'),
    totalInningRunsToComeSimSTD=('totalInningRunsToCome', 'std'),
    totalInningRunsToComeSimSkew=('totalInningRunsToCome', 'skew'),
    totalInningRunsToComeSimKurt=('totalInningRunsToCome', lambda x: kurtosis(x, fisher=True)),
    totalInningRunsToComeSimMin=('totalInningRunsToCome', 'min'),
    totalInningRunsToComeSimMax=('totalInningRunsToCome', 'max'),
).reset_index()

# export the situation summary - new file name so the downstream files can be switched over to it when ready
simSituationRunsToCome.to_csv(PROJECT_ROOT / 'men/expBall&runsToCome/outputs/ballSimsClassOrdSummary.csv', index=False)
