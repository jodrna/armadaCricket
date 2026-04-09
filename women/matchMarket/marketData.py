import pandas as pd
import numpy as np
from paths import PROJECT_ROOT

# import
marketData = pd.read_csv(PROJECT_ROOT / 'Predictions.csv')
marketData = marketData[marketData['providermarketname'] == 'Match Odds']
marketData = marketData[marketData['InningsNo'] == 2]
marketData = marketData[~marketData['Summary'].str.contains('DLS|D/L', na=False)]
marketData = marketData[~marketData['CompetitionName'].str.contains('T10', na=False)]
marketData['reduced'] = np.where(marketData['BallsInnings1'] < 120, np.where(marketData['WicketsInnings1'] < 10, 1, 0), 0)
marketData = marketData[marketData['reduced'] == 0]
marketData = marketData[marketData['TossWinner'] != 0]
marketData = marketData.drop_duplicates()


marketData[['teamA', 'teamB']] = marketData['MatchName'].str.split(' v ', expand=True)
marketData['tossTeam'] = np.where(marketData['TossWinner'] == 1, marketData['teamA'], marketData['teamB'])
marketData['chaseTeam'] = np.where(marketData['TossWinner'] == 1,
                                   (np.where(marketData['TossDecision'] == 'field', marketData['teamA'], marketData['teamB'])),
                                   (np.where(marketData['TossDecision'] == 'field', marketData['teamB'], marketData['teamA'])))

marketData = marketData[marketData['providerrunnername'] == marketData['chaseTeam']]
marketData['runsRequired'] = marketData['Target'] - marketData['CurrentRuns']
marketData['marketChaseProb'] = 1 / marketData['WeightedAveragePrice']


odds = pd.pivot_table(marketData, values=['marketChaseProb'], index=['runsRequired', 'BallNo', 'CurrentWickets'], aggfunc='mean').reset_index()

compcounts = pd.pivot_table(marketData, values=['InningsNo'], index=['CompetitionName'], aggfunc='count').reset_index()
