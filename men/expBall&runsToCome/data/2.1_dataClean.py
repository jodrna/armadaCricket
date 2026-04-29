import pandas as pd
import numpy as np
from paths import PROJECT_ROOT


# import
trainData = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/data/Cleaned_t20bbb3_adjusted_runs_to_come_plus_eff_target_inns1.csv', parse_dates=['date'])
trainData = trainData.rename(columns={'eff_target': 'effTarget'})

# rename columns
trainData = trainData.rename(columns={'innings': 'inningNumber', 'wickets': 'totalInningWickets', 'bowlerwicket': 'isWicketBowler', 'noball': 'noballRuns', 'over': 'overNumber',
                                      'score': 'totalInningRuns', 't_runs': 'totalInningRunsEnd', 'runs_to_come': 'totalInningRunsToCome', 'wicket': 'isWicket',
                                      't_wickets': 'totalInningWicketsEnd', 'runs': 'totalRuns', 'wide': 'wideRuns', 'ballsremaining': 'inningBallsRemaining', 'byes': 'byeRuns'})

# difference in days between the ball and the first ball in the db
def parse_mixed_date(series):
    return pd.to_datetime(
        series.astype(str).str.strip(),
        format='mixed',
        errors='coerce'
    )
trainData['date'] = parse_mixed_date(trainData['date'])
trainData['year'] = trainData['date'].dt.year
trainData['daysGroup'] = (trainData['date'] - (trainData['date'].min())).dt.days / 365

# create some columns
trainData['overBallNumber'] = (trainData['ball'] + 1 - trainData['overNumber']) * 100
trainData['inningBallNumber'] = 121 - trainData['inningBallsRemaining']
trainData['isPowerplay'] = np.where(trainData['inningBallNumber'] <= 36, 1, 0)
trainData['isValid'] = np.where((trainData['wideRuns'] > 0) | (trainData['noballRuns'] > 0), 0, 1)
trainData['isWide'] = np.where(trainData['wideRuns'] > 0, 1, 0)
trainData['isNoball'] = np.where(trainData['noballRuns'] > 0, 1, 0)
trainData['sample'] = 1
trainData['totalInningWicketsToCome'] = trainData['totalInningWicketsEnd'] - trainData['totalInningWickets']
trainData['batsmanRuns'] = trainData['totalRuns'] - trainData['noballRuns'] - trainData['wideRuns'] - trainData['byeRuns']
trainData['isWicketRunOut'] = np.where(trainData['isWicket'] > trainData['isWicketBowler'], 1, 0)
trainData['chaseWin'] = np.where(trainData['totalInningRunsEnd'] >= trainData['effTarget'], 1, 0)
trainData['effRunsRequired'] = trainData['effTarget'] - trainData['totalInningRuns']

# take out big bash power surge and 10 wickets down, keep only matches after 1st Jan 2015
trainData = trainData[(trainData['competition'] != 'Big Bash League') | (trainData['date'] < '06-06-2020')]
# trainData['totalInningWickets'] = np.where(trainData['isWicket'] == True, trainData['totalInningWickets'] - 1, trainData['totalInningWickets'])
trainData = trainData[trainData['totalInningWickets'] <= 9]
trainData = trainData[trainData['totalInningWickets'] >= 0]
trainData = trainData[trainData['date'] >= '01-01-2015']

# isolate the exact columns we need and give them correct names
trainData = trainData.loc[:, ['matchid', 'id', 'tier', 'date', 'year', 'competition', 'venue', 'host', 'home', 'away', 'battingteam', 'inningNumber', 'totalRuns', 'totalInningRuns',
                              'totalInningRunsEnd', 'isWicket', 'totalInningWickets', 'totalInningWicketsEnd', 'inningBallsRemaining', 'effTarget', 'noballRuns', 'wideRuns', 'ord', 'byeRuns', 'legbyes',
                              'innperiod', 'isWicketBowler', 'realexprbat', 'realexpwbat', 'rating_sample_size', 'major_nation', 'batsmanballs', 'ovrexpr', 'ovrexpw', 'batsman', 'nonstriker', 'extra',
                              'true_score', 'comp', 'required', 'totalInningRunsToCome', 'result', 'overNumber', 'daysGroup', 'overBallNumber', 'inningBallNumber', 'isPowerplay', 'isValid',
                              'isWide', 'isNoball', 'sample', 'totalInningWicketsToCome', 'batsmanRuns', 'isWicketRunOut', 'chaseWin', 'effRunsRequired', 'RA_sum']]
trainData.columns = ['matchID', 'ID', 'tier', 'date', 'year', 'competition', 'venue', 'host', 'home', 'away', 'battingTeam', 'inningNumber', 'totalRuns',
                     'totalInningRuns', 'totalInningRunsEnd', 'isWicket', 'totalInningWickets', 'totalInningWicketsEnd', 'inningBallsRemaining', 'effTarget', 'noballRuns', 'wideRuns',
                     'ord', 'byeRuns', 'legbyeRuns', 'inningPhase', 'isWicketBowler', 'realexprbat', 'realexpwbat', 'rating_sample_size', 'major_nation', 'batsmanBallsFaced',
                     'ovrexpr', 'ovrexpw', 'batsmanName', 'nonstrikerName', 'extra', 'true_score', 'comp', 'required', 'totalInningRunsToCome', 'result',
                     'overNumber', 'daysGroup', 'overBallNumber', 'inningBallNumber', 'isPowerplay', 'isValid', 'isWide', 'isNoball', 'sample',
                     'totalInningWicketsToCome', 'batsmanRuns', 'isWicketRunOut', 'chaseWin', 'effRunsRequired', 'RA_Sum']

# adjusted runs, used for the match market and year adjustment
# trainData['runsRequiredAdj'] = trainData['runsRequired'] - trainData['RA_Sum']
# trainData['totalInningRunsToComeAdj'] = trainData['totalInningRunsToCome'] - trainData['RA_Sum']


# export the cleaned data
trainData.to_csv(PROJECT_ROOT / 'men/expBall&runsToCome/data/dataClean1st.csv', index=False)


