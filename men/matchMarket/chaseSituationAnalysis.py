import pandas as pd
import numpy as np

# import file for analysis
chaseLookup = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/Development/matchMarket/2_chaseLookup.csv')
trainData = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/Development/matchMarket/trainDataModelled.csv', parse_dates=['date'])


runsRequired, runsRequiredBuffer = 171, 0
inningBallsRemaining, inningBallsRemainingBuffer = 120, 0
totalInningWickets, totalInningWicketsBuffer = 0, 0

trainData = trainData.merge(chaseLookup.loc[:, ['totalInningWickets', 'runsRequired', 'inningBallsRemaining', 'm_chaseWin%Live']], how='left',
                            on=['totalInningWickets', 'runsRequired', 'inningBallsRemaining'])

trainData = trainData[(trainData['totalInningWickets'] == totalInningWickets) &
                          (trainData['runsRequired'] > (runsRequired - runsRequiredBuffer)) & (trainData['runsRequired'] < (runsRequired + runsRequiredBuffer)) &
                          (trainData['inningBallsRemaining'] > (inningBallsRemaining - inningBallsRemainingBuffer)) & (trainData['inningBallsRemaining'] < (inningBallsRemaining + inningBallsRemainingBuffer))]
trainData = trainData.drop_duplicates(subset=['matchID'], keep='last')

results = pd.pivot_table(trainData, values=['runsRequired', 'inningBallsRemaining', 'chaseWin', 'matchID', 'm_chaseWin%', 'm_chaseWin%Live', 'totalInningRunsToComeSimBiasSpline'], index='year',
                         aggfunc={'runsRequired': 'mean', 'inningBallsRemaining': 'mean', 'chaseWin': 'sum', 'matchID': 'count', 'm_chaseWin%': 'mean',
                                  'totalInningRunsToComeSimBiasSpline': 'mean', 'm_chaseWin%Live': 'mean'}).reset_index()
# results['chaseRate'] = results['chaseWin'] / results['matchID']



