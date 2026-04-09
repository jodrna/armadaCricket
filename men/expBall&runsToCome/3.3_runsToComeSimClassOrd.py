import pandas as pd
import numpy as np
import time
from paths import PROJECT_ROOT

# import files
masterLookup = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/2_masterLookup.csv')
rateTrajectoryAdjustments = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/rateTrajectoryAdjustments.csv')
fxbXslw = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/fxbXslw.csv')

# the master lookup has values for ord and year, we don't want those for the sim, so drop them
masterLookup = masterLookup.drop_duplicates(subset=['totalInningWickets', 'inningBallNumber']).reset_index(drop=True)
masterLookup = masterLookup.merge(rateTrajectoryAdjustments, how='left', on=['inningBallNumber'])

# we start with 100k sims of the whole innings then for 5 wickets and higher do a sim from the inningball number closest to 50 samples
toSim = masterLookup[(masterLookup['sample'] < 50) & (masterLookup['totalInningWickets'] >= 3)].drop_duplicates(subset=['totalInningWickets'], keep='last').reset_index(drop=True)
toSim = toSim[['totalInningWickets', 'inningBallNumber']]


def simulate_innings(masterLookup, fxbXslw, ballsRemainingSet, totalInningWicketsSet, numberOfSims):
    simulationResults = []

    for sim in range(numberOfSims):
        print(sim, totalInningWicketsSet)
        simID = int(time.time() * 1000)

        totalInningBatRuns, totalInningByes, totalInningInvalidRuns = 0, 0, 0
        totalInningWickets, ballsRemaining = totalInningWicketsSet, ballsRemainingSet
        ballsSinceLastWicket, validBallsInOver = 0, 0

        batsmen = [{'id': 1, 'runs': 0, 'ballsFaced': 0}, {'id': 2, 'runs': 0, 'ballsFaced': 0}]
        striker_idx, non_striker_idx = 0, 1

        while ballsRemaining > 0 and totalInningWickets < 10:
            inningBallNumber = 121 - ballsRemaining
            masterLookupRow = masterLookup[
                (masterLookup['totalInningWickets'] == totalInningWickets) &
                (masterLookup['inningBallNumber'] == inningBallNumber)
            ]

            if masterLookupRow.empty:
                break

            totalInningRuns = totalInningBatRuns + totalInningByes + totalInningInvalidRuns
            rateTrajectory = totalInningRuns / masterLookupRow['predTotalInningRuns'].values[0]

            batsmanRunsBall = 0
            invalidRunsBall = 0
            isWicket = 0
            isInvalidWicket = 0

            byeRunsOver = masterLookupRow['byeRunsOver'].values[0]
            isWicketInvalidOver = masterLookupRow['isWicketInvalidOver'].values[0]
            isInvalidOver = masterLookupRow['isInvalidOver'].values[0]

            striker, non_striker = batsmen[striker_idx], batsmen[non_striker_idx]
            striker_id_before = striker['id']
            striker_balls_before = striker['ballsFaced']
            striker_runs_before = striker['runs']
            non_striker_id = non_striker['id']
            non_striker_balls = non_striker['ballsFaced']
            non_striker_runs = non_striker['runs']

            current_over = masterLookupRow['overNumber'].values[0]

            fxbRow = fxbXslw[
                (fxbXslw['ord'] == striker_id_before) &
                (fxbXslw['over'] == current_over) &
                (fxbXslw['ballsFaced'] == striker_balls_before)
            ]
            slwRow = fxbXslw[
                (fxbXslw['ord'] == striker_id_before) &
                (fxbXslw['over'] == current_over) &
                (fxbXslw['ballsFaced'] == ballsSinceLastWicket)
            ]

            if fxbRow.empty or slwRow.empty:
                run_adjust, wkt_adjust = 1, 1
            elif striker_balls_before < non_striker_balls:
                run_adjust = fxbRow['fxbRunscurve'].values[0]
                wkt_adjust = fxbRow['fxbWktscurve'].values[0]
            else:
                run_adjust = fxbRow['fxbRunscurve'].values[0] * slwRow['slwRunscurve'].values[0]
                wkt_adjust = fxbRow['fxbWktscurve'].values[0] * slwRow['slwWktscurve'].values[0]

            isWicketBall = wkt_adjust * masterLookupRow['m_isWicketBall'].values[0]
            isWicketBall = (((rateTrajectory - 1) * masterLookupRow['X_wr_balls_smooth'].values[0]) + 1) * isWicketBall


            if np.random.rand() < isInvalidOver:
                invalidRunsBall = (((rateTrajectory - 1) * masterLookupRow['X_rr_balls_smooth'].values[0]) + 1) * masterLookupRow['invalidRunsOver'].values[0]
                totalInningInvalidRuns += invalidRunsBall
                totalInningByes += byeRunsOver

                if np.random.rand() < isWicketInvalidOver:
                    isInvalidWicket = 1
                    totalInningWickets += 1

            else:
                striker['ballsFaced'] += 1
                validBallsInOver += 1

                run_probs = masterLookupRow[[str(i) for i in range(8)]].values.flatten()
                batsmanRun = np.random.choice(range(8), p=run_probs)
                batsmanRunsBall = batsmanRun * run_adjust
                batsmanRunsBall = (((rateTrajectory - 1) * masterLookupRow['X_rr_balls_smooth'].values[0]) + 1) * batsmanRunsBall

                striker['runs'] += batsmanRunsBall
                totalInningBatRuns += batsmanRunsBall
                totalInningByes += byeRunsOver

                if np.random.rand() < isWicketBall:
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






    allSimBalls = pd.DataFrame(simulationResults, columns=['simID', 'inningBallNumber', 'overNumber', 'rateTrajectory', 'm_batsmanRunsBall', 'invalidRunsBall', 'byeRunsOver', 'isWicket', 'isWicketInvalid',
                                                                    'totalInningBatRuns', 'totalInningInvalidRuns', 'totalInningByes', 'totalInningWickets',
                                                                    'm_isWicketBall', 'isWicketInvalidOver', 'isInvalidOver', 'strikerID', 'nonStrikerID', 'strikerBallsFaced', 'nonStrikerBallsFaced',
                                                           'strikerRuns', 'nonStrikerRuns', 'ballsSinceLastWicket', 'fxbXslwRunsCurve', 'fxbXslwWktsCurve'])
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


# first of all do a sim of the whole innings then do the additional
allSimBalls = simulate_innings(masterLookup, fxbXslw, ballsRemainingSet=120, totalInningWicketsSet=0, numberOfSims=100000)

# this buffs up situations where we NEED samples but the sim doesn't get into those situations enough, it makes sure any real life situation with over 50 samples has at least 10k sim samples
for index, row in toSim.iterrows():
    allSimBalls = pd.concat([allSimBalls, simulate_innings(masterLookup, fxbXslw, ballsRemainingSet=(121 - row.iloc[1]), totalInningWicketsSet=row.iloc[0], numberOfSims=10000)], axis=0)


# export the ball simulations
allSimBalls.to_csv(PROJECT_ROOT / 'men/expBall&runsToCome/ballSimsClassOrd.csv', index=False)



# create a pivot which shows average runs to come by inning ball number and wickets lost
situations = pd.pivot_table(allSimBalls, values=['totalInningRunsToCome', 'sample'], index=['inningBallNumber', 'totalInningWickets'],
                            aggfunc={'totalInningRunsToCome': 'mean', 'sample': 'sum'}).reset_index()






