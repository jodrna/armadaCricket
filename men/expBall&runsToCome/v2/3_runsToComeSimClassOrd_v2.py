import pandas as pd
import numpy as np
import random
import itertools
from scipy.stats import kurtosis
from paths import PROJECT_ROOT

# v2 of 3_runsToComeSimClassOrd_fast.py - a fully discrete sim, every ball is an exact event:
#   - valid ball: wicket drawn first, then runs off the bat from the not out distribution (or the wicket ball runs
#     distribution if out), then byes if the batter didn't score - all whole numbers
#   - wide / no ball: total runs drawn from the real distribution for that over (1 wide, 5 wides, no ball + six etc)
#   - FXB/SLW now scale the chance of scoring off a ball (dot ball probability) rather than
#     scaling the runs, so the runs stay whole numbers and the expected runs are scaled by exactly the same amount
#   - strike changes on odd runs actually run, including byes and runs off wides/no balls
#   - a wicket off a wide/no ball now brings in a new batter (the old sim just added to the wicket count)
#   - no rate trajectory (momentum) adjustment: tested against average team spread (auxiliaries/rateTrajectory_v2.py),
#     no version improved it once team quality was removed and none had the best means, so it's switched off
#   - uses the Aug FXB.csv / SLW.csv curves (adds overs 19-20) instead of the June fxbXslw.csv
# reads step 1 v2 directly - the sim only ever used step 1's columns, step 2's ord/year rows were being dropped

# import files
masterLookup = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/outputs/1_masterLookup_v2.csv')
fxbRaw = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/auxiliaries/FXB.csv')
slwRaw = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/auxiliaries/SLW.csv')

masterLookup = masterLookup.drop_duplicates(subset=['totalInningWickets', 'inningBallNumber']).reset_index(drop=True)

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
mlCols = ['isWicketInvalidOver', 'isInvalidOver', 'overNumber', 'm_isWicketBall']


def cdfs(columns):
    probs = masterLookup[columns].to_numpy(dtype=float)
    cdf = np.cumsum(probs, axis=1)
    return cdf / cdf[:, [-1]]


# runs off the bat when not out, split into the dot ball chance and the distribution of runs given the batter scores (1-7),
# so the run multipliers can scale the chance of scoring while keeping the scoring shots' shape
notOutProbs = masterLookup[[f'notOutRuns_{i}' for i in range(8)]].to_numpy(dtype=float)
notOutProbs = notOutProbs / notOutProbs.sum(axis=1, keepdims=True)
dotProb = notOutProbs[:, 0]
scoringCdfs = np.cumsum(notOutProbs[:, 1:], axis=1)
scoringCdfs = scoringCdfs / scoringCdfs[:, [-1]]
invalidRunsCdfs = cdfs([f'invRuns_{i}' for i in range(1, 8)])     # outcome k -> k+1 runs
byeRunsCdfs = cdfs([f'byeRuns_{i}' for i in range(0, 6)])
wicketRunsCdfs = cdfs([f'wktRuns_{i}' for i in range(0, 5)])

mlLookup = [[None] * 121 for _ in range(10)]
for i, row in enumerate(masterLookup[['totalInningWickets', 'inningBallNumber'] + mlCols].itertuples(index=False)):
    w, b = int(row[0]), int(row[1])
    mlLookup[w][b] = tuple(row[2:]) + (dotProb[i], scoringCdfs[i].tolist(), invalidRunsCdfs[i].tolist(),
                                       byeRunsCdfs[i].tolist(), wicketRunsCdfs[i].tolist())


def draw(cdf, u):
    # inverse cdf draw, returns the outcome index
    k, last = 0, len(cdf) - 1
    while k < last and u >= cdf[k]:
        k += 1
    return k

# FXB (batter's own balls faced) and SLW (balls since last wicket) curves, the Aug versions that replace the June fxbXslw.csv.
# code = ballsFaced*1000 + over + ord/100 + inningNumber/1000, same decoding as auxiliaries/fxbAdjust.py. They cover ord 2-8
# (openers are floored to 2, same as the rest of the pipeline), overs 3-20 and ballsFaced 1-21 (21 = 21+, clamped like fxbAdjust.py).
# FXB splits overs 19-20 by innings - this is a first innings sim so those use inningNumber 1, everything else the universal (0) rows
FXB_MAX_BALLS = 21


def decode_curve(raw):
    curve = raw.copy()
    curve['ballsFaced'] = np.floor(curve['code'] / 1000).astype(int)
    remainder = curve['code'] - curve['ballsFaced'] * 1000
    curve['over'] = np.floor(remainder).astype(int)
    scaled = np.round((remainder - curve['over']) * 1000).astype(int)
    curve['ord'] = scaled // 10
    curve['inningNumber'] = scaled % 10
    curve = curve[(curve['inningNumber'] == 0) | ((curve['inningNumber'] == 1) & (curve['over'] >= 19))]
    # where an over has both, the first innings row wins over the universal one
    curve = curve.sort_values('inningNumber').drop_duplicates(subset=['ord', 'over', 'ballsFaced'], keep='last')
    return {(int(r.ord), int(r.over), int(r.ballsFaced)): (r.runscurve, r.wktscurve) for r in curve.itertuples(index=False)}


fxbLookup = decode_curve(fxbRaw)
slwLookup = decode_curve(slwRaw)

# unique sim ids across every call (the original used the ms timestamp, which collides once sims run this fast)
simIDCounter = itertools.count()

simColumns = ['simID', 'inningBallNumber', 'overNumber', 'm_batsmanRunsBall', 'invalidRunsBall', 'byeRunsBall', 'isWicket', 'isWicketInvalid',
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

            (isWicketInvalidOver, isInvalidOver, current_over, m_isWicketBall,
             dotProbBall, scoringCdf, invalidRunsCdf, byeRunsCdf, wicketRunsCdf) = ml


            batsmanRunsBall = 0
            invalidRunsBall = 0
            byeRunsBall = 0
            isWicket = 0
            isInvalidWicket = 0

            striker, non_striker = batsmen[striker_idx], batsmen[non_striker_idx]
            striker_id_before = striker['id']
            striker_balls_before = striker['ballsFaced']
            striker_runs_before = striker['runs']
            non_striker_id = non_striker['id']
            non_striker_balls = non_striker['ballsFaced']
            non_striker_runs = non_striker['runs']

            # curves are looked up on balls already faced (as the original sim did - tested better than counting the current ball)
            overKey = int(current_over)
            curveOrd = striker_id_before if striker_id_before > 2 else 2
            fxbBalls = striker_balls_before if striker_balls_before < FXB_MAX_BALLS else FXB_MAX_BALLS
            slwBalls = ballsSinceLastWicket if ballsSinceLastWicket < FXB_MAX_BALLS else FXB_MAX_BALLS
            fxbRow = fxbLookup.get((curveOrd, overKey, fxbBalls))
            slwRow = slwLookup.get((curveOrd, overKey, slwBalls))

            if fxbRow is None or slwRow is None:
                run_adjust, wkt_adjust = 1, 1
            elif striker_balls_before < non_striker_balls:
                run_adjust = fxbRow[0]
                wkt_adjust = fxbRow[1]
            else:
                run_adjust = fxbRow[0] * slwRow[0]
                wkt_adjust = fxbRow[1] * slwRow[1]

            isWicketBall = wkt_adjust * m_isWicketBall

            if rand() < isInvalidOver:
                # wide / no ball: total runs off it, penalty included, from the real distribution for this over
                invalidRunsBall = draw(invalidRunsCdf, rand()) + 1
                totalInningInvalidRuns += invalidRunsBall
                # the ends change if the runs actually run (everything but the 1 run penalty) is odd
                if (invalidRunsBall - 1) % 2 == 1:
                    striker_idx, non_striker_idx = non_striker_idx, striker_idx

                if rand() < isWicketInvalidOver:
                    isInvalidWicket = 1
                    totalInningWickets += 1
                    new_batsman_id = max(b['id'] for b in batsmen) + 1
                    batsmen[striker_idx] = {'id': new_batsman_id, 'runs': 0, 'ballsFaced': 0}
                    ballsSinceLastWicket = 0

            else:
                striker['ballsFaced'] += 1
                validBallsInOver += 1

                if rand() < isWicketBall:
                    isWicket = 1
                    totalInningWickets += 1
                    # runs completed on the wicket ball (mostly 0, run outs can come with runs)
                    batsmanRunsBall = draw(wicketRunsCdf, rand())
                    striker['runs'] += batsmanRunsBall
                    totalInningBatRuns += batsmanRunsBall
                    new_batsman_id = max(b['id'] for b in batsmen) + 1
                    batsmen[striker_idx] = {'id': new_batsman_id, 'runs': 0, 'ballsFaced': 0}
                else:
                    # scale the chance of scoring so the expected runs move by exactly the multiplier, runs stay whole numbers
                    runMultiplier = run_adjust
                    dotProbAdj = 1 - runMultiplier * (1 - dotProbBall)
                    if dotProbAdj < 0:
                        dotProbAdj = 0
                    elif dotProbAdj > 1:
                        dotProbAdj = 1
                    u = rand()
                    if u < dotProbAdj:
                        # dot off the bat - byes are possible
                        byeRunsBall = draw(byeRunsCdf, rand())
                    else:
                        batsmanRunsBall = draw(scoringCdf, (u - dotProbAdj) / (1 - dotProbAdj)) + 1

                    striker['runs'] += batsmanRunsBall
                    totalInningBatRuns += batsmanRunsBall
                    totalInningByes += byeRunsBall
                    if (batsmanRunsBall + byeRunsBall) % 2 == 1:
                        striker_idx, non_striker_idx = non_striker_idx, striker_idx

                ballsRemaining -= 1

                if validBallsInOver == 6:
                    validBallsInOver = 0
                    striker_idx, non_striker_idx = non_striker_idx, striker_idx

                ballsSinceLastWicket = 0 if isWicket else ballsSinceLastWicket + 1

            simulationResults.append((
                simID, inningBallNumber, current_over, batsmanRunsBall, invalidRunsBall, byeRunsBall,
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
    allSimBalls['totalRunsBall'] = allSimBalls['m_batsmanRunsBall'] + allSimBalls['invalidRunsBall'] + allSimBalls['byeRunsBall']
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
simSituationRunsToCome.to_csv(PROJECT_ROOT / 'men/expBall&runsToCome/outputs/ballSimsClassOrdSummary_v2.csv', index=False)
