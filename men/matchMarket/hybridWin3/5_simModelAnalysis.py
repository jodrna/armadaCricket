import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from paths import PROJECT_ROOT

N_FOLDS = 5
FOLD_SEED = 2718
RANDOM_SEED = 42
stateColumns = ['runsRequired', 'totalInningWickets', 'inningBallsRemaining', 'ord', 'nonStrikerOrd']
ballColumns = stateColumns[:4]
rowColumns = ['matchID', 'ID', 'fold']
dataHash = hashlib.sha256((PROJECT_ROOT / 'men/matchMarket/hybridWin3/data/dataClean.csv').read_bytes()).hexdigest()
runInfo = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/runInfo.csv')
if 'pipelineVersion' not in runInfo or runInfo.at[0, 'pipelineVersion'] != 'full_remodel_v3':
    raise ValueError('These scripts require full-innings outputs. Run scripts 1-5 in order.')
if 'gridHash' not in runInfo or runInfo.at[0, 'gridHash'] != hashlib.sha256((PROJECT_ROOT / 'men/matchMarket/hybridWin3/auxiliaries/1_chaseLookup.csv').read_bytes()).hexdigest():
    raise ValueError('The neural grid changed. Rerun scripts 1-5 in order.')
if runInfo.at[0, 'dataHash'] != dataHash or int(runInfo.at[0, 'nFolds']) != N_FOLDS or int(runInfo.at[0, 'foldSeed']) != FOLD_SEED:
    raise ValueError('Input data or fold settings changed. Rerun 1_deathBallProbs.py and then the remaining scripts in order.')
if any(int(runInfo.at[0, column]) != 1 for column in ['ballReady', 'simulationReady', 'directReady', 'blendReady']):
    raise ValueError('Complete all four preceding scripts on the same data before analysis.')

resourceHash = hashlib.sha256((PROJECT_ROOT / 'men/matchMarket/hybridWin3/auxiliaries/6_masterLookup.csv').read_bytes()).hexdigest()
if 'directResourceHash' not in runInfo or runInfo.at[0, 'directResourceHash'] != resourceHash:
    raise ValueError('The resource lookup changed. Run scripts 3-5 again.')

# --------------------------------------------------
# READ HISTORICAL OOF PREDICTIONS FOR SCORING
# --------------------------------------------------
testData = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/chaseWinProbabilities_oof.csv', float_precision='round_trip')
modelColumns = {
    'Raw simulation': 'sim_chaseWin%',
    'Calibrated simulation': 'sim_chaseWin%Calibrated',
    'Direct win model': 'directWinProbability',
    'Blended model': 'matchWinProbability',
}
values = testData[list(modelColumns.values())].to_numpy(float)
if testData.empty or not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
    raise ValueError('Missing or invalid OOF win probabilities. Run the four scripts in order.')
if testData.duplicated(['matchID', 'ID']).any() or set(testData['fold']) != set(range(N_FOLDS)):
    raise ValueError('Expected one unseen prediction per eligible row across five folds.')
if testData.groupby('matchID')['fold'].nunique().gt(1).any():
    raise ValueError('A match has been split across test folds.')

# --------------------------------------------------
# THE ORIGINAL NEURAL LOOKUP, USED ONLY FOR COMPARISON
# --------------------------------------------------
baseSituationColumns = ['runsRequired', 'totalInningWickets', 'inningBallsRemaining']
chaseLookup = pd.read_csv(
    PROJECT_ROOT / 'men/matchMarket/hybridWin3/auxiliaries/1_chaseLookup.csv',
    usecols=baseSituationColumns + ['m_chaseWin%'],
)
chaseLookup = chaseLookup.drop_duplicates(subset=baseSituationColumns)
testData = testData.merge(chaseLookup, on=baseSituationColumns, how='left', validate='many_to_one')
commonData = testData.loc[testData['m_chaseWin%'].between(0, 1)].copy()
commonModels = dict(modelColumns, **{'Old neural': 'm_chaseWin%'})
if commonData.empty:
    raise ValueError('No common neural rows are available for the win-model comparison.')
samples = [('Common neural rows', commonData, commonModels)]

print(f'Five-fold OOF evaluation on common neural rows: {len(commonData):,} balls from {commonData["matchID"].nunique():,} matches.')
print(f'Excluded {len(testData) - len(commonData):,} balls without a valid neural lookup probability from the win-model comparison.')
print('All five win models are compared on identical common neural rows; probabilities cover the full 0-1 range.')
print('Each outer test match was excluded from its ball model, direct model, calibration and blend fits.')
print('The extras and first-innings mean/STD lookups are fixed; their upstream training provenance is unknown. The old neural is an in-sample benchmark.')
print('Earlier architecture choices used this historical data. These are cross-validated results, not a fresh external test.')
print(f'Simulation grid coverage: {int(runInfo.at[0, "hybridOofRows"]):,} of {int(runInfo.at[0, "eligibleInningsRows"]):,} valid-pair full-innings rows; {len(testData):,} remain for the blend after resource-coverage filtering.')

# --------------------------------------------------
# OVERALL AND FOLD SCORES; MATCH-WEIGHTED LOSS AS WELL AS ROW LOSS
# --------------------------------------------------
scoreRows = []
foldRows = []
segmentRows = []
for sampleName, sample, models in samples:
    subsets = [('all', sample)] + [(int(fold), group) for fold, group in sample.groupby('fold')]
    for fold, frame in subsets:
        target = frame['chaseWin'].to_numpy(int)
        matchWeights = 1 / frame['matchID'].map(frame.groupby('matchID').size()).to_numpy(float)
        for modelName, column in models.items():
            originalProbability = frame[column].to_numpy(float)
            probability = np.clip(originalProbability, 1e-7, 1 - 1e-7)
            rowLoss = -target * np.log(probability) - (1 - target) * np.log1p(-probability)
            row = {
                'Sample': sampleName, 'Model': modelName,
                'Log loss': float(rowLoss.mean()),
                'Brier score': float(np.mean((originalProbability - target) ** 2)),
                'Equal-match log loss': float(np.average(rowLoss, weights=matchWeights)),
                'Mean predicted win probability': float(originalProbability.mean()),
                'Actual win rate': float(target.mean()),
                'Rows': len(frame), 'Matches': int(frame['matchID'].nunique()),
            }
            if fold == 'all':
                scoreRows.append(row)
            else:
                row['Fold'] = fold
                foldRows.append(row)
    sample = sample.copy()
    sample['ballsRemainingBand'] = pd.cut(sample['inningBallsRemaining'], bins=[0, 6, 12, 18, 24, 60, 84, 120], labels=['1-6', '7-12', '13-18', '19-24', '25-60', '61-84', '85-120'])
    sample['overNumber'] = (120 - sample['inningBallsRemaining']) // 6 + 1
    for groupColumn in ['totalInningWickets', 'ballsRemainingBand', 'inningBallsRemaining', 'overNumber']:
        for groupValue, group in sample.groupby(groupColumn, observed=True):
            for modelName, column in models.items():
                segmentRows.append({
                    'Sample': sampleName, 'Group': groupColumn, 'Value': str(groupValue),
                    'Model': modelName, 'Rows': len(group), 'Matches': group['matchID'].nunique(),
                    'Log loss': log_loss(group['chaseWin'], np.clip(group[column], 1e-7, 1 - 1e-7), labels=[0, 1]),
                    'Brier score': float(np.mean((group[column].to_numpy(float) - group['chaseWin'].to_numpy(int)) ** 2)),
                })
scores = pd.DataFrame(scoreRows)
print('\n' + scores.to_string(index=False, float_format='{:.9f}'.format))
scores.to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/modelAnalysis.csv', index=False)
pd.DataFrame(foldRows).to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/foldScores.csv', index=False)
pd.DataFrame(segmentRows).to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/stateScores.csv', index=False)
overComparison = pd.DataFrame(segmentRows)
overComparison = overComparison.loc[overComparison['Group'].eq('overNumber')].copy()
overComparison['Over'] = overComparison['Value'].astype(float).astype(int)
print('\nALL FIVE MODELS BY OVER — IDENTICAL COMMON NEURAL ROWS:')
print(overComparison.pivot(index='Over', columns='Model', values=['Log loss', 'Brier score']).to_string(float_format='{:.9f}'.format))

# --------------------------------------------------
# CALIBRATION: ONE PANEL PER MODEL
# --------------------------------------------------
plotData = commonData
plotModels = commonModels
colours = ['tab:purple', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red']
fig, axes = plt.subplots(5, 1, figsize=(8, 24), sharex=True, sharey=True)
for ax, (modelName, column), colour in zip(axes, plotModels.items(), colours):
    frame = plotData[[column, 'chaseWin']].copy()
    frame['probabilityBin'] = pd.qcut(frame[column], q=10, duplicates='drop')
    calibration = frame.groupby('probabilityBin', observed=True).agg(predicted=(column, 'mean'), observed=('chaseWin', 'mean'))
    ax.plot([0, 1], [0, 1], '--', color='grey', label='Perfect calibration')
    ax.plot(calibration['predicted'], calibration['observed'], 'o-', color=colour, linewidth=2, label=modelName)
    ax.set_title(modelName)
    ax.set_xlabel('Predicted win probability')
    ax.set_ylabel('Observed win rate')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', labelbottom=True)
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
fig.suptitle('Five-fold OOF: common neural rows', fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.98])
# Displayed with the production heatmaps by plt.show() at the end.

# --------------------------------------------------
# BALL MODEL: EVERY ELIGIBLE FULL-INNINGS ROW, INCLUDING OFF-GRID ROWS
# --------------------------------------------------
ballData = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/deathBallProbs_oof.csv', float_precision='round_trip')
jointEvents = np.array([(runs, wicket) for runs in range(7) for wicket in range(2) if (runs, wicket) != (6, 1)], dtype=int)
jointColumns = [f'pressure_{runs}_' + ('wicket' if wicket else 'noWicket') for runs, wicket in jointEvents]
probabilities = ballData[jointColumns].to_numpy(float)
if not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1):
    raise ValueError('Invalid OOF ball probability vectors.')
if ballData.duplicated(['matchID', 'ID']).any():
    raise ValueError('Duplicate OOF ball rows.')
jointLoss = log_loss(ballData['jointTarget'].to_numpy(int), probabilities, labels=np.arange(len(jointEvents)))
ballResults = [{'Category': 'Joint event', 'Log loss': jointLoss, 'Rows': len(ballData)}]
for runs in range(7):
    probability = probabilities[:, jointEvents[:, 0] == runs].sum(axis=1)
    loss = log_loss(ballData['batsmanRuns'].eq(runs).astype(int), probability, labels=[0, 1])
    ballResults.append({'Category': f'{runs} runs', 'Log loss': loss, 'Rows': len(ballData)})
wicketProbability = probabilities[:, jointEvents[:, 1] == 1].sum(axis=1)
ballResults.append({'Category': 'Wicket', 'Log loss': log_loss(ballData['isWicket'].astype(int), wicketProbability, labels=[0, 1]), 'Rows': len(ballData)})
ballResults = pd.DataFrame(ballResults)
ballResults.to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/ballModelAnalysis.csv', index=False)
print('\nUnseen ball-model scores:')
print(ballResults.to_string(index=False, float_format='{:.9f}'.format))
print(f'Mean legal-ball runs: predicted {np.mean(probabilities @ jointEvents[:, 0]):.6f}, actual {ballData["batsmanRuns"].mean():.6f}')
print(f'Analysis exported to {PROJECT_ROOT / "men/matchMarket/hybridWin3/outputs"}')


# --------------------------------------------------
# FULL-INNINGS DIRECT VERSUS OLD NEURAL, PRODUCED BY SCRIPT 3
# --------------------------------------------------
directScores = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/directWinScores.csv')
print('\nFULL-INNINGS DIRECT VERSUS OLD NEURAL:')
print(directScores.loc[directScores['scope'].eq('Common neural states')].to_string(index=False, float_format='{:.9f}'.format))
directOvers = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/directWinByOver.csv')
directOvers = directOvers.loc[directOvers['scope'].eq('Common neural states')]
print('\nFULL-INNINGS COMPARISON BY OVER:')
print(directOvers.pivot(index='over', columns='model', values=['logLoss', 'brier']).to_string(float_format='{:.9f}'.format))


# --------------------------------------------------
# ALL PRODUCTION STATES AND MODEL PROBABILITIES
# --------------------------------------------------
# Keep the full simulation grid, including states without direct/blend coverage.
# These production predictions are for inspection; scores above use OOF rows.
productionSim = pd.read_csv(
    PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/chaseSimSituations.csv',
    usecols=stateColumns + ['sim_chaseWin%'],
    float_precision='round_trip',
)
productionDirect = pd.read_csv(
    PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/directWinProbabilities.csv',
    usecols=stateColumns + ['directWinProbability'],
    float_precision='round_trip',
)
blendColumns = ['sim_chaseWin%Calibrated', 'matchWinProbability', 'directWeight', 'simulationWeight']
productionBlend = pd.read_csv(
    PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/chaseWinProbabilities.csv',
    usecols=stateColumns + blendColumns,
    float_precision='round_trip',
)

situations = productionSim.merge(productionDirect, on=stateColumns, how='left', validate='one_to_one')
situations = situations.merge(productionBlend, on=stateColumns, how='left', validate='one_to_one')
situations = situations.merge(chaseLookup, on=baseSituationColumns, how='left', validate='many_to_one')
productionModelColumns = list(modelColumns.values()) + ['m_chaseWin%']
situations = situations[stateColumns + productionModelColumns + ['directWeight', 'simulationWeight']]
# Counts identify extrapolated production scenarios; they are diagnostics,
# never predictors or weights in model fitting.
historicalDirect = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/directWinProbabilities_oof.csv', usecols=stateColumns + ['matchID', 'ID'])
support = historicalDirect.groupby(stateColumns).agg(historicalRows=('ID', 'size'), historicalMatches=('matchID', 'nunique')).reset_index()
situations = situations.merge(support, on=stateColumns, how='left', validate='one_to_one')
situations[['historicalRows', 'historicalMatches']] = situations[['historicalRows', 'historicalMatches']].fillna(0).astype(int)
situations.to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/productionComparison.csv', index=False)

print(f'\nProduction comparison: {len(situations):,} states in the situations dataframe.')
print('Available production predictions by model:')
print(situations[productionModelColumns].notna().sum().to_string())
print('States without a model prediction remain in the table with a blank CSV value.')
print(f'Exported {PROJECT_ROOT / "men/matchMarket/hybridWin3/outputs/productionComparison.csv"}')



# --------------------------------------------------
# BLENDED PRODUCTION HEATMAPS: 5x2 GRID BY WICKETS
# --------------------------------------------------
# Each panel fixes wickets lost. The opener stays on strike and the
# non-striker is the latest batting order: wickets lost + 2.
# Grey means the state is outside the saved grid or has no model prediction.
# Only saved production predictions are plotted; no models are fitted here.
HEATMAP_STRIKER_ORDER = 1
heatmapModels = {
    'Blended model': 'matchWinProbability',
}

heatmapBallsRemaining = np.arange(1, int(situations['inningBallsRemaining'].max()) + 1)
heatmapRunsRequired = np.arange(1, int(situations['runsRequired'].max()) + 1)
heatmapBallTicks = [1] + list(range(12, int(heatmapBallsRemaining[-1]) + 1, 12))
heatmapRunTicks = [1] + list(range(30, int(heatmapRunsRequired[-1]) + 1, 30))
heatmapColourMap = plt.get_cmap('PiYG').copy()
heatmapColourMap.set_bad('#dddddd')

for heatmapModelName, heatmapModelColumn in heatmapModels.items():
    heatmapFigure, heatmapAxes = plt.subplots(5, 2, figsize=(14, 21), sharex=True, sharey=True)
    heatmapFigure.subplots_adjust(left=0.08, right=0.87, bottom=0.075, top=0.915, wspace=0.18, hspace=0.42)
    heatmapFigure.suptitle(
        f'{heatmapModelName}: production win probabilities',
        fontsize=18, fontweight='bold', y=0.982,
    )
    heatmapFigure.text(
        0.475, 0.956,
        f'Striker batting order {HEATMAP_STRIKER_ORDER} throughout | non-striker order = wickets lost + 2',
        ha='center', fontsize=11,
    )

    for heatmapWickets, heatmapAxis in enumerate(heatmapAxes.flat):
        heatmapNonStrikerOrder = heatmapWickets + 2
        heatmapScenario = situations.loc[
            situations['totalInningWickets'].eq(heatmapWickets)
            & situations['ord'].eq(HEATMAP_STRIKER_ORDER)
            & situations['nonStrikerOrd'].eq(heatmapNonStrikerOrder)
        ]
        heatmapMatrix = heatmapScenario.pivot(
            index='runsRequired',
            columns='inningBallsRemaining',
            values=heatmapModelColumn,
        )
        heatmapMatrix = heatmapMatrix.reindex(index=heatmapRunsRequired, columns=heatmapBallsRemaining)
        heatmapValues = heatmapMatrix.to_numpy(float) * 100
        heatmapImage = heatmapAxis.imshow(
            np.ma.masked_invalid(heatmapValues),
            origin='lower',
            aspect='auto',
            interpolation='nearest',
            extent=[
                heatmapBallsRemaining[0] - 0.5,
                heatmapBallsRemaining[-1] + 0.5,
                heatmapRunsRequired[0] - 0.5,
                heatmapRunsRequired[-1] + 0.5,
            ],
            cmap=heatmapColourMap,
            vmin=0,
            vmax=100,
        )
        heatmapWicketLabel = 'wicket' if heatmapWickets == 1 else 'wickets'
        heatmapAxis.set_title(
            f'{heatmapWickets} {heatmapWicketLabel} | striker {HEATMAP_STRIKER_ORDER}, non-striker {heatmapNonStrikerOrder}',
            fontsize=11, pad=9,
        )
        heatmapAxis.set_xticks(heatmapBallTicks)
        heatmapAxis.set_yticks(heatmapRunTicks)
        heatmapAxis.tick_params(axis='both', labelbottom=True, labelleft=True, labelsize=9)
        for heatmapSpine in heatmapAxis.spines.values():
            heatmapSpine.set_visible(False)

    heatmapFigure.supxlabel('Balls remaining', fontsize=13, x=0.475, y=0.046)
    heatmapFigure.supylabel('Runs required', fontsize=13, x=0.018)
    heatmapColourBarAxis = heatmapFigure.add_axes([0.90, 0.13, 0.018, 0.72])
    heatmapColourBar = heatmapFigure.colorbar(heatmapImage, cax=heatmapColourBarAxis)
    heatmapColourBar.set_label('Win probability (%)', fontsize=11)
    heatmapColourBar.set_ticks([0, 20, 40, 60, 80, 100])
    heatmapFigure.text(
        0.08, 0.018,
        'All panels use the same 0-100% scale. Grey = no saved prediction for that state.\n'
        'The non-striker changes between panels. These are production predictions, not observed win rates.',
        fontsize=9, color='#505050', linespacing=1.6,
    )

# Display the calibration chart and blended production heatmap figure in PyCharm.
# No plot images are exported or saved.
plt.show()

