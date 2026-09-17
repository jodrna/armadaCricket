import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from paths import PROJECT_ROOT

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
NEURAL_HIDDEN_LAYERS = (64, 64, 32)
NEURAL_ALPHA = 0.001
NEURAL_BATCH_SIZE = 512
NEURAL_LEARNING_RATE = 0.001
NEURAL_MAX_ITER = 500
NEURAL_VALIDATION_FRACTION = 0.10
NEURAL_N_ITER_NO_CHANGE = 20
NEURAL_RANDOM_STATE = 42
IMPOSSIBLE_WICKET_RUN_OUTCOMES = [6]

# --------------------------------------------------
# IMPORT
# --------------------------------------------------
trainData = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/data/dataClean.csv', parse_dates=['date'])

# --------------------------------------------------
# KEEP SECOND INNINGS AND LAST 4 OVERS
# --------------------------------------------------
trainData = trainData[trainData['inningBallsRemaining'].between(1, 24)]
trainData = trainData[trainData['inningNumber'] == 2]
trainData = trainData.dropna(subset=['runsRequired', 'inningBallsRemaining', 'totalInningWickets', 'batsmanRuns', 'isWicket'])
trainData = trainData[trainData['runsRequired'] > 0]
trainData = trainData[trainData['totalInningWickets'].between(0, 9)]

# --------------------------------------------------
# KEEP LEGAL DELIVERIES
# --------------------------------------------------
trainData = trainData[(trainData['wideRuns'].fillna(0) == 0) & (trainData['noballRuns'].fillna(0) == 0)]
trainData['batsmanRuns'] = trainData['batsmanRuns'].astype(int)
trainData['isWicket'] = trainData['isWicket'].astype(int)
trainData = trainData[~(trainData['batsmanRuns'].isin(IMPOSSIBLE_WICKET_RUN_OUTCOMES) & (trainData['isWicket'] == 1))]
trainData['runsRequiredPerBall'] = trainData['runsRequired'] / trainData['inningBallsRemaining']

# --------------------------------------------------
# ESTIMATED SCORING ABILITY BY WICKETS LOST
# ONLY USED TO DEFINE THE SIMULATION STATES
# --------------------------------------------------
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

# --------------------------------------------------
# CREATE APPROXIMATELY 5,000 CHASE SITUATIONS
# --------------------------------------------------
situations = []
for inningBallsRemaining in range(1, 25):
    for totalInningWickets in range(10):
        centreRunsRequired = inningBallsRemaining * runsPerBall[totalInningWickets]
        halfWidth = 5.5 + (0.35 * inningBallsRemaining)
        minRunsRequired = max(1, int(np.floor(centreRunsRequired - halfWidth)))
        maxRunsRequired = int(np.ceil(centreRunsRequired + halfWidth))
        if inningBallsRemaining <= 6:
            absoluteMaxRunsRequired = inningBallsRemaining * 6
        else:
            absoluteMaxRunsRequired = inningBallsRemaining * 3
        maxRunsRequired = min(maxRunsRequired, absoluteMaxRunsRequired)
        for runsRequired in range(minRunsRequired, maxRunsRequired + 1):
            situations.append({'runsRequired': runsRequired, 'totalInningWickets': totalInningWickets, 'inningBallsRemaining': inningBallsRemaining})
masterLookup = pd.DataFrame(situations)

# --------------------------------------------------
# MASTER LOOKUP VARIABLES
# --------------------------------------------------
masterLookup['inningBallNumber'] = 121 - masterLookup['inningBallsRemaining']
masterLookup['overNumber'] = np.ceil(masterLookup['inningBallNumber'] / 6).astype(int)
masterLookup['inningNumber'] = 2
masterLookup['runsRequiredPerBall'] = masterLookup['runsRequired'] / masterLookup['inningBallsRemaining']

# --------------------------------------------------
# KEEP ONLY HISTORICAL STATES USED BY THE SIMULATOR
# --------------------------------------------------
situationColumns = ['runsRequired', 'totalInningWickets', 'inningBallsRemaining']
trainData = trainData.merge(masterLookup[situationColumns].drop_duplicates(), how='inner', on=situationColumns)

# --------------------------------------------------
# POSSIBLE BATSMAN-RUN OUTCOMES
# --------------------------------------------------
outcomeValues = sorted(trainData['batsmanRuns'].unique())

# --------------------------------------------------
# PRESSURE
# --------------------------------------------------
# Raw required runs per ball. No smoothed baseline is used.
trainData['pressure'] = trainData['runsRequiredPerBall']
trainData['logPressure'] = np.log(trainData['pressure'])

# --------------------------------------------------
# JOINT RUN/WICKET EVENTS
# --------------------------------------------------
jointEvents = []
for outcome in outcomeValues:
    jointEvents.append((int(outcome), 0))
    if outcome not in IMPOSSIBLE_WICKET_RUN_OUTCOMES:
        jointEvents.append((int(outcome), 1))
numberOfJointEvents = len(jointEvents)
jointEventLookup = {event: position for position, event in enumerate(jointEvents)}
jointTarget = np.array([jointEventLookup[(int(run), int(wicket))] for run, wicket in zip(trainData['batsmanRuns'], trainData['isWicket'])], dtype=int)


def expandJointProbabilities(model, features, numberOfJointEvents):
    predictedProbability = model.predict_proba(features)
    if hasattr(model, 'classes_'):
        modelClasses = model.classes_
    else:
        modelClasses = model.named_steps['classifier'].classes_
    expandedProbability = np.zeros((len(features), numberOfJointEvents), dtype=float)
    for classPosition, jointEventPosition in enumerate(modelClasses):
        expandedProbability[:, int(jointEventPosition)] = predictedProbability[:, classPosition]
    expandedProbability = expandedProbability / expandedProbability.sum(axis=1, keepdims=True)
    return expandedProbability

# --------------------------------------------------
# HISTOGRAM GRADIENT BOOSTING FEATURES
# --------------------------------------------------
# The best same-date model found in shuffled cross-validation. These are all
# pre-ball variables; leg-byes remain included in batsmanRuns by design.
modelTrainingMask = trainData['ord'].notna() & trainData['ord'].between(1, 11)
modelFeatureColumns = ['runsRequired', 'pressure', 'logPressure', 'inningBallsRemaining', 'totalInningWickets', 'ord']
XModel = trainData.loc[modelTrainingMask, modelFeatureColumns].to_numpy(dtype=float)
yModel = jointTarget[modelTrainingMask.to_numpy()]

# --------------------------------------------------
# FIT HISTOGRAM GRADIENT BOOSTING CLASSIFIER
# --------------------------------------------------
ballOutcomeModel = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.04,
    max_leaf_nodes=7,
    min_samples_leaf=100,
    l2_regularization=20.0,
    early_stopping=False,
    random_state=42
)
ballOutcomeModel.fit(XModel, yModel)

# --------------------------------------------------
# FIT NEURAL NETWORK CLASSIFIER
# --------------------------------------------------
neuralOutcomeModel = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', MLPClassifier(
        hidden_layer_sizes=NEURAL_HIDDEN_LAYERS,
        alpha=NEURAL_ALPHA,
        batch_size=NEURAL_BATCH_SIZE,
        learning_rate_init=NEURAL_LEARNING_RATE,
        max_iter=NEURAL_MAX_ITER,
        validation_fraction=NEURAL_VALIDATION_FRACTION,
        n_iter_no_change=NEURAL_N_ITER_NO_CHANGE,
        early_stopping=True,
        random_state=NEURAL_RANDOM_STATE
    ))
])
neuralOutcomeModel.fit(XModel, yModel)

# --------------------------------------------------
# OVERALL JOINT-EVENT LOG LOSS
# --------------------------------------------------
neuralTrainingJointProbabilityArray = expandJointProbabilities(
    neuralOutcomeModel,
    XModel,
    numberOfJointEvents
)
histTrainingJointProbabilityArray = expandJointProbabilities(
    ballOutcomeModel,
    XModel,
    numberOfJointEvents
)
neuralOverallLogLoss = log_loss(
    yModel,
    neuralTrainingJointProbabilityArray,
    labels=np.arange(numberOfJointEvents)
)
histOverallLogLoss = log_loss(
    yModel,
    histTrainingJointProbabilityArray,
    labels=np.arange(numberOfJointEvents)
)
print()
print(f'Overall joint-event log loss on {len(yModel):,} model rows:')
print(f'Neural network: {neuralOverallLogLoss:.6f}')
print(f'Histogram gradient boosting: {histOverallLogLoss:.6f}')

# --------------------------------------------------
# CATEGORY BINARY LOG LOSS
# --------------------------------------------------
categoryOutcomes = [outcome for outcome in [0, 1, 2, 3, 4, 6] if outcome in outcomeValues]
modelTrainingRuns = trainData.loc[modelTrainingMask, 'batsmanRuns'].to_numpy()
modelTrainingWickets = trainData.loc[modelTrainingMask, 'isWicket'].to_numpy()
print('Category binary log loss:')
print(f'{"Category":<12}{"Neural":>14}{"Histogram":>14}')
for outcome in categoryOutcomes:
    eventPositions = [
        eventPosition
        for eventPosition, event in enumerate(jointEvents)
        if event[0] == outcome
    ]
    actualCategory = (modelTrainingRuns == outcome).astype(int)
    neuralCategoryProbability = neuralTrainingJointProbabilityArray[:, eventPositions].sum(axis=1)
    histCategoryProbability = histTrainingJointProbabilityArray[:, eventPositions].sum(axis=1)
    neuralCategoryLogLoss = log_loss(
        actualCategory,
        neuralCategoryProbability,
        labels=[0, 1]
    )
    histCategoryLogLoss = log_loss(
        actualCategory,
        histCategoryProbability,
        labels=[0, 1]
    )
    print(f'{str(outcome) + " runs":<12}{neuralCategoryLogLoss:>14.6f}{histCategoryLogLoss:>14.6f}')

wicketEventPositions = [
    eventPosition
    for eventPosition, event in enumerate(jointEvents)
    if event[1] == 1
]
neuralWicketProbability = neuralTrainingJointProbabilityArray[:, wicketEventPositions].sum(axis=1)
histWicketProbability = histTrainingJointProbabilityArray[:, wicketEventPositions].sum(axis=1)
neuralWicketLogLoss = log_loss(
    modelTrainingWickets,
    neuralWicketProbability,
    labels=[0, 1]
)
histWicketLogLoss = log_loss(
    modelTrainingWickets,
    histWicketProbability,
    labels=[0, 1]
)
print(f'{"Wicket":<12}{neuralWicketLogLoss:>14.6f}{histWicketLogLoss:>14.6f}')
print()

# --------------------------------------------------
# MASTER LOOKUP PRESSURE
# --------------------------------------------------
masterLookup['pressure'] = masterLookup['runsRequiredPerBall']
masterLookup['logPressure'] = np.log(masterLookup['pressure'])

# --------------------------------------------------
# EXPAND MASTER LOOKUP BY BATSMAN ORDER
# --------------------------------------------------
masterLookup = masterLookup.loc[masterLookup.index.repeat(masterLookup['totalInningWickets'] + 2)].copy()
masterLookup['ord'] = masterLookup.groupby(['runsRequired', 'totalInningWickets', 'inningBallsRemaining']).cumcount() + 1

# --------------------------------------------------
# HISTOGRAM GRADIENT BOOSTING MASTER LOOKUP PREDICTIONS
# --------------------------------------------------
XMaster = masterLookup[modelFeatureColumns].to_numpy(dtype=float)
ballOutcomeJointProbabilityArray = expandJointProbabilities(
    ballOutcomeModel,
    XMaster,
    numberOfJointEvents
)

# --------------------------------------------------
# STORE FINAL JOINT PROBABILITIES
# KEEP PRESSURE_* NAMES FOR SIMULATOR COMPATIBILITY
# --------------------------------------------------
pressureJointColumns = []
for eventPosition, event in enumerate(jointEvents):
    outcome = event[0]
    isWicket = event[1]
    if isWicket == 1:
        column = f'pressure_{outcome}_wicket'
    else:
        column = f'pressure_{outcome}_noWicket'
    pressureJointColumns.append(column)
    masterLookup[column] = ballOutcomeJointProbabilityArray[:, eventPosition]
for outcome in IMPOSSIBLE_WICKET_RUN_OUTCOMES:
    masterLookup[f'pressure_{outcome}_wicket'] = 0.0

# --------------------------------------------------
# FINAL RUN MARGINALS
# --------------------------------------------------
for outcome in outcomeValues:
    noWicketColumn = f'pressure_{int(outcome)}_noWicket'
    wicketColumn = f'pressure_{int(outcome)}_wicket'
    if wicketColumn in masterLookup.columns:
        masterLookup[f'pressure_{int(outcome)}'] = masterLookup[noWicketColumn] + masterLookup[wicketColumn]
    else:
        masterLookup[f'pressure_{int(outcome)}'] = masterLookup[noWicketColumn]

# --------------------------------------------------
# FINAL WICKET PROBABILITY
# --------------------------------------------------
pressureWicketColumns = [f'pressure_{event[0]}_wicket' for event in jointEvents if event[1] == 1]
masterLookup['pressure_wicket'] = masterLookup[pressureWicketColumns].sum(axis=1)

# --------------------------------------------------
# FINAL EXPECTED BATSMAN RUNS
# --------------------------------------------------
masterLookup['pressure_m_batsmanRunsBall'] = sum(outcome * masterLookup[f'pressure_{int(outcome)}'] for outcome in outcomeValues)
masterLookup['pressureJointProbabilitySum'] = masterLookup[pressureJointColumns].sum(axis=1)

# --------------------------------------------------
# MODEL RESIDUALS AFTER NEURAL NETWORK MODEL
# --------------------------------------------------
plotOutcomes = [0, 1, 2, 3, 4, 6]
predictionColumns = [f'pressure_{outcome}' for outcome in plotOutcomes] + ['pressure_wicket']
stateColumns = ['runsRequired', 'totalInningWickets', 'inningBallsRemaining', 'ord']
trainData = trainData[trainData['ord'].notna() & trainData['ord'].between(1, 11)]
trainData['ord'] = trainData['ord'].astype(int)
trainData = trainData.merge(masterLookup[stateColumns + predictionColumns], how='left', on=stateColumns)
trainData['pressureBucket'] = (trainData['pressure'] / 0.2).round() * 0.2
for outcome in plotOutcomes:
    trainData[f'actual_{outcome}'] = (trainData['batsmanRuns'] == outcome).astype(int)
trainData['actual_wicket'] = trainData['isWicket']
aggregation = {'sample': ('batsmanRuns', 'size')}
for outcome in plotOutcomes:
    aggregation[f'actual_{outcome}'] = (f'actual_{outcome}', 'mean')
    aggregation[f'pred_{outcome}'] = (f'pressure_{outcome}', 'mean')
aggregation['actual_wicket'] = ('actual_wicket', 'mean')
aggregation['pred_wicket'] = ('pressure_wicket', 'mean')
wicketPressurePivot = trainData.groupby(['totalInningWickets', 'pressureBucket'], as_index=False).agg(**aggregation)
for outcome in plotOutcomes:
    wicketPressurePivot[f'diff_{outcome}'] = wicketPressurePivot[f'actual_{outcome}'] - wicketPressurePivot[f'pred_{outcome}']
wicketPressurePivot['diff_wicket'] = wicketPressurePivot['actual_wicket'] - wicketPressurePivot['pred_wicket']

# --------------------------------------------------
# PLOT
# --------------------------------------------------
fig, axes = plt.subplots(5, 2, figsize=(16, 20))
axes = axes.flatten()
for totalInningWickets in range(10):
    ax = axes[totalInningWickets]
    plotData = wicketPressurePivot[wicketPressurePivot['totalInningWickets'] == totalInningWickets]
    for outcome in plotOutcomes:
        ax.plot(plotData['pressureBucket'], plotData[f'diff_{outcome}'], marker='o', label=str(outcome))
    ax.plot(plotData['pressureBucket'], plotData['diff_wicket'], marker='o', linewidth=2, label='Wicket')
    ax.axhline(0, linewidth=1)
    ax.set_title(f'{totalInningWickets} Wickets Lost')
    ax.set_xlabel('Pressure')
    ax.set_ylabel('Actual - Model Probability')
    ax.grid(True, alpha=0.3)
    if totalInningWickets == 0:
        ax.legend()
plt.suptitle('Ball-Outcome Residuals After Histogram Gradient Boosting Model', fontsize=16)
plt.tight_layout()
plt.show()



masterLookup.to_csv(PROJECT_ROOT / 'men/expBall&runsToCome/outputs/deathBallProbs.csv', index=False)


