import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from paths import PROJECT_ROOT


# -------------------------
# Import
# -------------------------
simSituations = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/outputs/chaseSimSituations.csv')
chaseLookup = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/outputs/1_chaseLookup.csv')
trainData = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/data/dataClean.csv', parse_dates=['date'])



# -------------------------
# Create non-striker order
# -------------------------
trainData['newestBatterOrd'] = trainData['totalInningWickets'] + 2
trainData['nonStrikerOrd'] = np.where(trainData['ord'] != trainData['newestBatterOrd'], trainData['newestBatterOrd'], np.nan)

playerInfo = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/auxiliaries/playerInfo.csv')
playerInfo = playerInfo[['playerid', 'cricinfo_id']].drop_duplicates(subset=['cricinfo_id'])
playerInfo = playerInfo.rename(columns={'cricinfo_id': 'nonstrikerid', 'playerid': 'nonStrikerPlayerID'})
trainData = trainData.merge(playerInfo, how='left', on='nonstrikerid')

nonstrikerOrder = trainData.dropna(subset=['batterid', 'ord']).groupby(['matchID', 'inningNumber', 'batterid'], as_index=False)['ord'].min()
nonstrikerOrder = nonstrikerOrder.rename(columns={'batterid': 'nonStrikerPlayerID', 'ord': 'nonStrikerOrdID'})
trainData = trainData.merge(nonstrikerOrder, how='left', on=['matchID', 'inningNumber', 'nonStrikerPlayerID'])
trainData['nonStrikerOrd'] = trainData['nonStrikerOrd'].fillna(trainData['nonStrikerOrdID'])

nonstrikerOrderName = trainData.dropna(subset=['batsmanName', 'ord']).groupby(['matchID', 'inningNumber', 'batsmanName'], as_index=False)['ord'].min()
nonstrikerOrderName = nonstrikerOrderName.rename(columns={'batsmanName': 'nonstrikerName', 'ord': 'nonStrikerOrdName'})
trainData = trainData.merge(nonstrikerOrderName, how='left', on=['matchID', 'inningNumber', 'nonstrikerName'])
trainData['nonStrikerOrd'] = trainData['nonStrikerOrd'].fillna(trainData['nonStrikerOrdName'])

trainData = trainData.drop(columns=['newestBatterOrd', 'nonStrikerPlayerID', 'nonStrikerOrdID', 'nonStrikerOrdName'])

# -------------------------
# Keep last 4 overs
# -------------------------
trainData = trainData[trainData['inningBallsRemaining'] <= 24]
trainData = trainData[trainData['inningNumber'] == 2]


# -------------------------
# Check batting pair coverage
# -------------------------
print('Non-striker order available:', trainData['nonStrikerOrd'].notna().sum(), 'of', len(trainData))

validBattingPair = (trainData[['ord', 'nonStrikerOrd']].max(axis=1) == trainData['totalInningWickets'] + 2) & (trainData['ord'] != trainData['nonStrikerOrd'])

print('Valid batting pair:', validBattingPair.sum(), 'of', len(trainData))


# -------------------------
# Columns used to identify each chase situation
# -------------------------
baseSituationColumns = ['runsRequired', 'totalInningWickets', 'inningBallsRemaining']
simSituationColumns = ['runsRequired', 'totalInningWickets', 'inningBallsRemaining', 'ord', 'nonStrikerOrd']


# -------------------------
# Keep only the neural chase probability from the chase lookup
# -------------------------
chaseLookup = chaseLookup[baseSituationColumns + ['m_chaseWin%']].drop_duplicates(subset=baseSituationColumns)


# -------------------------
# Merge neural chase probability into the simulation situations
# -------------------------
simSituations = simSituations.merge(chaseLookup, how='left', on=baseSituationColumns)


# -------------------------
# Check simulation-state coverage
# -------------------------
baseSituationColumns = ['runsRequired', 'totalInningWickets', 'inningBallsRemaining']
strikerSituationColumns = ['runsRequired', 'totalInningWickets', 'inningBallsRemaining', 'ord']
simSituationColumns = ['runsRequired', 'totalInningWickets', 'inningBallsRemaining', 'ord', 'nonStrikerOrd']

print('Total real-life rows:', len(trainData))
print('Ord available:', trainData['ord'].notna().sum())
print('Non-striker ord available:', trainData['nonStrikerOrd'].notna().sum())

validPair = (trainData['ord'] != trainData['nonStrikerOrd']) & (trainData[['ord', 'nonStrikerOrd']].max(axis=1) == trainData['totalInningWickets'] + 2)
print('Valid batting pair:', validPair.sum())

baseStates = simSituations[baseSituationColumns].drop_duplicates()
baseStates['baseMatch'] = 1
coverage = trainData.merge(baseStates, how='left', on=baseSituationColumns)
print('Base state exists in sim:', coverage['baseMatch'].notna().sum())

strikerStates = simSituations[strikerSituationColumns].drop_duplicates()
strikerStates['strikerMatch'] = 1
coverage = trainData.merge(strikerStates, how='left', on=strikerSituationColumns)
print('Base + striker ord exists:', coverage['strikerMatch'].notna().sum())

pairStates = simSituations[simSituationColumns].drop_duplicates()
pairStates['pairMatch'] = 1
coverage = trainData.merge(pairStates, how='left', on=simSituationColumns)
print('Full batting pair exists:', coverage['pairMatch'].notna().sum())


# -------------------------
# Merge simulation chase probability into the real-life samples
# -------------------------
trainData = trainData.merge(simSituations[simSituationColumns + ['sim_chaseWin%']], how='left', on=simSituationColumns)


# -------------------------
# Merge neural chase probability into the real-life samples
# -------------------------
trainData = trainData.merge(chaseLookup, how='left', on=baseSituationColumns)


trainData = trainData[trainData['m_chaseWin%'].between(0.2, 0.8, inclusive='both')]


# -------------------------
# Model difference
# -------------------------
simSituations['modelDifference'] = simSituations['sim_chaseWin%'] - simSituations['m_chaseWin%']
trainData['modelDifference'] = trainData['sim_chaseWin%'] - trainData['m_chaseWin%']


print('Simulation situations:', len(simSituations))
print('Real-life samples:', len(trainData))
print('Sim situations with both predictions:', simSituations[['sim_chaseWin%', 'm_chaseWin%']].notna().all(axis=1).sum())
print('Real-life samples with both predictions:', trainData[['sim_chaseWin%', 'm_chaseWin%']].notna().all(axis=1).sum())


# -------------------------
# Difference between simulation and neural model
# -------------------------
simSituations['modelDifference'] = simSituations['sim_chaseWin%'] - simSituations['m_chaseWin%']


# -------------------------
# Use the same colour scale across all 10 plots
# -------------------------
maxDifference = simSituations['modelDifference'].abs().max()


# -------------------------
# 5x2 plots, one for each wickets-lost state
# -------------------------
fig, axes = plt.subplots(5, 2, figsize=(20, 28))
axes = axes.flatten()

for wickets in range(10):
    wicketData = simSituations[simSituations['totalInningWickets'] == wickets]

    heatmapData = wicketData.pivot_table(index='runsRequired', columns='inningBallsRemaining', values='modelDifference', aggfunc='mean')
    heatmapData = heatmapData.sort_index(ascending=False)

    sns.heatmap(heatmapData, ax=axes[wickets], cmap='PiYG', center=0, vmin=-maxDifference, vmax=maxDifference, cbar=wickets == 0, cbar_kws={'label': 'Sim - Neural'})

    axes[wickets].set_title(f'{wickets} Wickets Lost')
    axes[wickets].set_xlabel('Balls Remaining')
    axes[wickets].set_ylabel('Runs Required')

plt.suptitle('Simulation vs Neural Chase Win Probability', fontsize=20, y=0.995)
plt.tight_layout()
plt.show()


# -------------------------
# Keep only rows where both models have a prediction
# -------------------------
logLossData = trainData.dropna(subset=['sim_chaseWin%', 'm_chaseWin%', 'chaseWin']).copy()


# -------------------------
# Clip probabilities so we never take log(0)
# -------------------------
epsilon = 1e-15

logLossData['sim_chaseWin%'] = logLossData['sim_chaseWin%'].clip(epsilon, 1 - epsilon)
logLossData['m_chaseWin%'] = logLossData['m_chaseWin%'].clip(epsilon, 1 - epsilon)


# -------------------------
# Log loss for each row
# -------------------------
logLossData['simLogLoss'] = -(logLossData['chaseWin'] * np.log(logLossData['sim_chaseWin%']) + (1 - logLossData['chaseWin']) * np.log(1 - logLossData['sim_chaseWin%']))

logLossData['neuralLogLoss'] = -(logLossData['chaseWin'] * np.log(logLossData['m_chaseWin%']) + (1 - logLossData['chaseWin']) * np.log(1 - logLossData['m_chaseWin%']))


# -------------------------
# Overall log loss
# -------------------------
simLogLoss = logLossData['simLogLoss'].mean()
neuralLogLoss = logLossData['neuralLogLoss'].mean()


print('Log loss samples:', len(logLossData))
print('Neural log loss:', neuralLogLoss)
print('Simulation log loss:', simLogLoss)
print('Difference:', simLogLoss - neuralLogLoss)


# -------------------------
# Model disagreement buckets
# -------------------------
logLossData['modelDifference'] = logLossData['sim_chaseWin%'] - logLossData['m_chaseWin%']

bins = [-np.inf, -0.10, -0.08, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08, 0.10, np.inf]

labels = [
    '< -10%',
    '-10% to -8%',
    '-8% to -6%',
    '-6% to -4%',
    '-4% to -2%',
    '-2% to 0%',
    '0% to +2%',
    '+2% to +4%',
    '+4% to +6%',
    '+6% to +8%',
    '+8% to +10%',
    '> +10%'
]

logLossData['differenceBucket'] = pd.cut(logLossData['modelDifference'], bins=bins, labels=labels, include_lowest=True)


# -------------------------
# Summary by size and direction of disagreement
# -------------------------
logLossComparison = pd.pivot_table(logLossData, index='differenceBucket', values=['simLogLoss', 'neuralLogLoss'], aggfunc=['mean', 'count'], observed=False)

print(logLossComparison)


# -------------------------
# Cleaner dataframe for plotting
# -------------------------
logLossPlot = logLossData.groupby('differenceBucket', observed=False).agg(neuralLogLoss=('neuralLogLoss', 'mean'), simLogLoss=('simLogLoss', 'mean'), samples=('chaseWin', 'size'), meanNeural=('m_chaseWin%', 'mean'), meanSim=('sim_chaseWin%', 'mean'), actualWinRate=('chaseWin', 'mean')).reset_index()

logLossPlot['logLossDifference'] = logLossPlot['simLogLoss'] - logLossPlot['neuralLogLoss']

print(logLossPlot)


# -------------------------
# Plot
# ------------------------
fig, ax = plt.subplots(figsize=(16, 8))

x = np.arange(len(logLossPlot))
width = 0.4

ax.bar(x - width / 2, logLossPlot['neuralLogLoss'], width, label='Neural')
ax.bar(x + width / 2, logLossPlot['simLogLoss'], width, label='Simulation')

ax.set_xticks(x)
ax.set_xticklabels(logLossPlot['differenceBucket'], rotation=45, ha='right')
ax.set_xlabel('Sim - Neural Probability')
ax.set_ylabel('Log Loss')
ax.set_title('Log Loss by Model Disagreement')
ax.legend()

plt.tight_layout()
plt.show()
