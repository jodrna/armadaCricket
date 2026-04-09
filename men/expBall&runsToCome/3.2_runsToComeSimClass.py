import pandas as pd
import numpy as np
import time

# import files
masterLookup = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/Development/men/expBall&runsToCome/2_masterLookup.csv')
rateTrajectoryAdjustments = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/Development/men/expBall&runsToCome/rateTrajectoryAdjustments.csv')

# the master lookup has values for ord and year, we don't want those for the sim, so drop them
masterLookup = masterLookup.drop_duplicates(subset=['totalInningWickets', 'inningBallNumber']).reset_index(drop=True)
masterLookup = masterLookup.merge(rateTrajectoryAdjustments, how='left', on=['inningBallNumber'])

# we start with 100k sims of the whole innings then for 5 wickets and higher do a sim from the inningball number closest to 50 samples
toSim = masterLookup[(masterLookup['sample'] < 50) & (masterLookup['totalInningWickets'] >= 3)].drop_duplicates(subset=['totalInningWickets'], keep='last').reset_index(drop=True)
toSim = toSim[['totalInningWickets', 'inningBallNumber']]

#
def simulate_innings(masterLookup, ballsRemainingSet, totalInningWicketsSet, numberOfSims):
    simulationResults = []

    for sim in range(numberOfSims):
        simNumber = int(time.time() * 1000)
        print(simNumber)

        # set totals at 0 which we'll add to as we sim through the innings
        totalInningBatRuns = 0
        totalInningByes = 0
        totalInningInvalidRuns = 0
        totalInningWickets = totalInningWicketsSet
        ballsRemaining = ballsRemainingSet


        # STEP 1, check are there balls and wickets left in the innings, then isolate the numbers from the masterlookup table for balls left and wickets lost, then get all requried numbers
        while ballsRemaining > 0 and totalInningWickets < 10:
            inningBallNumber = 121 - ballsRemaining  # Current ball number in the innings
            current_row = masterLookup[(masterLookup['totalInningWickets'] == totalInningWickets) & (masterLookup['inningBallNumber'] == inningBallNumber)]


            # we need current total inning runs and expected, we can work out if they are ahead or behind the game and adjust accordinly
            totalInningRuns = totalInningBatRuns + totalInningByes + totalInningInvalidRuns
            rateTrajectoryRuns = totalInningRuns / current_row['avgTotalInningRuns'].values[0]
            rateTrajectoryWickets = totalInningWickets / (current_row['avgTotalInningWickets'].values[0] + 0.00000000000000001)


            # you must redeclare these as zero etc before each ball sim because they will be appended, if you don't redeclare isWicket will always be 1 after the sim first sees a wicket
            if not current_row.empty:
                # we declare these as zero because we don't know what they will be yet, it depends on the ball sim if it's a wicket or not, or valid ball or not
                batsmanRunsBall = 0
                invalidRunsBall = 0
                isWicket = 0
                isInvalidWicket = 0

                # these are probabilities to be used below which will decide the above 4 values, except byes, they never change
                byeRunsOver = current_row['byeRunsOver'].values[0]
                isWicketBall = current_row['m_isWicketBall'].values[0]
                isWicketInvalidOver = current_row['isWicketInvalidOver'].values[0]
                isInvalidOver = current_row['isInvalidOver'].values[0]

                # adjust isWicketBall for game state
                isWicketBall = (((rateTrajectoryWickets- 1) * current_row['X_wr_balls_smooth'].values[0]) + 1) * isWicketBall


                # STEP 2, check is the ball valid
                if np.random.rand() < isInvalidOver:
                    # STEP 2A, add the average runs for an invalid ball, adjust first for game state, if they are ahead or behind average
                    invalidRunsBall = current_row['invalidRunsOver'].values[0]
                    invalidRunsBall = (((rateTrajectoryRuns- 1) * current_row['X_rr_balls_smooth'].values[0]) + 1) * invalidRunsBall
                    totalInningInvalidRuns += invalidRunsBall

                    # add the byes which are static for every ball
                    totalInningByes += byeRunsOver

                    # STEP 2B, check for a wicket off a wide/no ball
                    if np.random.rand() < isWicketInvalidOver:
                        isInvalidWicket = 1
                        totalInningWickets += isInvalidWicket  # IF there is a wicket we add 1 to the total

                # STEP 3, we know the ball is valid
                else:
                    # STEP 3A, get runs for a valid ball, adjust first for game state, if they are ahead or behind average
                    batsmanRunsBall = np.random.choice(range(8), p=current_row[[str(i) for i in range(8)]].values.flatten())
                    batsmanRunsBall = (((rateTrajectoryRuns- 1) * current_row['X_rr_balls_smooth'].values[0]) + 1) * batsmanRunsBall
                    totalInningBatRuns += batsmanRunsBall

                    # add the byes which are static for every ball
                    totalInningByes += byeRunsOver

                    # STEP 3B, check for a wicket, if yes add a wicket to wickets lost
                    if np.random.rand() < isWicketBall:
                        isWicket = 1
                        totalInningWickets += isWicket  # IF there is a wicket we add 1 to the total

                    # STEP 3C, take one ball off the total remaining, this is the main difference in process from an invalid ball, we don't take a ball off for a wide/noball
                    ballsRemaining -= 1



            simulationResults.append((simNumber, inningBallNumber, rateTrajectoryRuns, batsmanRunsBall, invalidRunsBall, byeRunsOver, isWicket, isInvalidWicket,
                                       totalInningBatRuns, totalInningInvalidRuns, totalInningByes, totalInningWickets, isWicketBall, isWicketInvalidOver, isInvalidOver))


    allSimBalls = pd.DataFrame(simulationResults, columns=['simNumber', 'inningBallNumber', 'rateTrajectoryRuns', 'm_batsmanRunsBall', 'invalidRunsBall', 'byeRunsOver', 'isWicket', 'isWicketInvalid',
                                                                    'totalInningBatRuns', 'totalInningInvalidRuns', 'totalInningByes', 'totalInningWickets',
                                                                    'm_isWicketBall', 'isWicketInvalidOver', 'isInvalidOver'])
    allSimBalls['totalRunsBall'] = allSimBalls['m_batsmanRunsBall'] + allSimBalls['invalidRunsBall'] + allSimBalls['byeRunsOver']
    allSimBalls['totalInningRuns'] = allSimBalls['totalInningBatRuns'] + allSimBalls['totalInningInvalidRuns'] + allSimBalls['totalInningByes']


    # create a pivot which aggregates the innings, so we can see the end of innings number of runs for every sim
    simInnings = pd.pivot_table(allSimBalls, values=['totalRunsBall'], index=['simNumber'], aggfunc='sum')
    simInnings.columns = ['totalInningRunsEnd']

    # create a sample column so we can see sample size for situations
    allSimBalls['sample'] = 1
    # merge the end of innings totals to each sim ball, then we simply work out the difference between this ball and end of innings to see runs to come
    allSimBalls = allSimBalls.merge(simInnings, how='left', on=['simNumber'])

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
allSimBalls = simulate_innings(masterLookup, ballsRemainingSet=120, totalInningWicketsSet=0, numberOfSims=100000)

# this buffs up situations where we NEED samples but the sim doesn't get into those situations enough, it makes sure any real life situation with over 50 samples has at least 10k sim samples
for index, row in toSim.iterrows():
    allSimBalls = pd.concat([allSimBalls, simulate_innings(masterLookup, ballsRemainingSet=(121 - row.iloc[1]), totalInningWicketsSet=row.iloc[0], numberOfSims=10000)], axis=0)


# export the ball simulations
allSimBalls.to_csv('/Users/jordan/Documents/ArmadaCricket/Development/men/expBall&runsToCome/ballSimsClassRateAdjusted.csv', index=False)



# create a pivot which shows average runs to come by inning ball number and wickets lost
situations = pd.pivot_table(allSimBalls, values=['totalInningRunsToCome', 'sample'], index=['inningBallNumber', 'totalInningWickets'],
                            aggfunc={'totalInningRunsToCome': 'mean', 'sample': 'sum'}).reset_index()



