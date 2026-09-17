import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from paths import PROJECT_ROOT

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
RUN_POLYNOMIAL_DEGREE = 3
WICKET_POLYNOMIAL_DEGREE = 3
CONDITIONAL_WICKET_POLYNOMIAL_DEGREE = 2
MIN_PROBABILITY = 1e-8
WICKET_END_WEIGHT_STRENGTH = 3.0
WICKET_END_WEIGHT_POWER = 6
# pressure adjustment
PRESSURE_L2 = 0.05
PRESSURE_MAX_ITER = 1000
# batsman order adjustment
ORDER_POLYNOMIAL_DEGREE = 3
ORDER_L2 = 0.05
ORDER_MAX_ITER = 1000
# wicket resource adjustment
RESOURCE_L2 = 0.05
RESOURCE_MAX_ITER = 1000
# wicket combinations that cannot happen
IMPOSSIBLE_WICKET_RUN_OUTCOMES = [6]

# --------------------------------------------------
# IMPORT
# --------------------------------------------------
trainData = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/data/dataClean.csv', parse_dates=['date'])

# --------------------------------------------------
# KEEP SECOND INNINGS AND LAST 4 OVERS
# --------------------------------------------------
trainData = trainData[ trainData['inningBallsRemaining'].between(1, 24) ]
trainData = trainData[ trainData['inningNumber'] == 2 ]
trainData = trainData.dropna( subset=[ 'runsRequired', 'inningBallsRemaining', 'totalInningWickets', 'batsmanRuns', 'isWicket' ] )
trainData = trainData[ trainData['runsRequired'] > 0 ]
trainData = trainData[ trainData['totalInningWickets'].between(0, 9) ]

# --------------------------------------------------
# KEEP LEGAL DELIVERIES
# --------------------------------------------------
#
# Invalid deliveries are handled separately by
# the simulator.

# --------------------------------------------------
trainData = trainData[ ( trainData['wideRuns'].fillna(0) == 0 ) & ( trainData['noballRuns'].fillna(0) == 0 ) ]
trainData['batsmanRuns'] = ( trainData['batsmanRuns'] .astype(int) )
trainData['isWicket'] = ( trainData['isWicket'] .astype(int) )
# remove impossible 6 + wicket combinations
trainData = trainData[ ~( trainData['batsmanRuns'].isin( IMPOSSIBLE_WICKET_RUN_OUTCOMES ) & ( trainData['isWicket'] == 1 ) ) ]
trainData['runsRequiredPerBall'] = ( trainData['runsRequired'] / trainData['inningBallsRemaining'] )

# --------------------------------------------------
# ESTIMATED SCORING ABILITY BY WICKETS LOST
# --------------------------------------------------
#
# ONLY used to define the simulation states.
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
        centreRunsRequired = ( inningBallsRemaining * runsPerBall[ totalInningWickets ] )
        halfWidth = ( 5.5 + ( 0.35 * inningBallsRemaining ) )
        minRunsRequired = max( 1, int( np.floor( centreRunsRequired - halfWidth ) ) )
        maxRunsRequired = int( np.ceil( centreRunsRequired + halfWidth ) )
        if inningBallsRemaining <= 6:
            absoluteMaxRunsRequired = ( inningBallsRemaining * 6 )
        else:
            absoluteMaxRunsRequired = ( inningBallsRemaining * 3 )
        maxRunsRequired = min( maxRunsRequired, absoluteMaxRunsRequired )
        for runsRequired in range( minRunsRequired, maxRunsRequired + 1 ):
            situations.append({
                'runsRequired': runsRequired,
                'totalInningWickets': totalInningWickets,
                'inningBallsRemaining': inningBallsRemaining
            })
masterLookup = pd.DataFrame(situations)

# --------------------------------------------------
# MASTER LOOKUP VARIABLES
# --------------------------------------------------
masterLookup['inningBallNumber'] = ( 121 - masterLookup['inningBallsRemaining'] )
masterLookup['overNumber'] = np.ceil( masterLookup[ 'inningBallNumber' ] / 6 ).astype(int)
masterLookup['inningNumber'] = 2
masterLookup['runsRequiredPerBall'] = ( masterLookup['runsRequired'] / masterLookup['inningBallsRemaining'] )

# --------------------------------------------------
# KEEP ONLY HISTORICAL STATES USED BY THE SIMULATOR
# --------------------------------------------------
situationColumns = [
    'runsRequired',
    'totalInningWickets',
    'inningBallsRemaining'
]
trainData = trainData.merge(masterLookup[situationColumns].drop_duplicates(), how='inner', on=situationColumns)

# --------------------------------------------------
# POSSIBLE BATSMAN-RUN OUTCOMES
# --------------------------------------------------
outcomeValues = sorted( trainData[ 'batsmanRuns' ].unique() )

# --------------------------------------------------
# ACTUAL RUN COUNTS BY BALLS REMAINING
# --------------------------------------------------
actualRunCounts = trainData.groupby(['inningBallsRemaining', 'batsmanRuns']).size().unstack(fill_value=0)
actualRunCounts = actualRunCounts.reindex(index=range(1, 25), fill_value=0).reindex(columns=outcomeValues, fill_value=0)

# --------------------------------------------------
# SAMPLE SIZE
# --------------------------------------------------
actualSample = actualRunCounts.sum(axis=1)

# --------------------------------------------------
# RAW RUN PROBABILITIES
# --------------------------------------------------
actualRunProbabilities = actualRunCounts.div(actualSample, axis=0)

# --------------------------------------------------
# ACTUAL WICKET PROBABILITY
# --------------------------------------------------
actualWicketCounts = trainData.groupby('inningBallsRemaining')['isWicket'].sum().reindex(range(1, 25), fill_value=0)
actualWicketProbability = ( actualWicketCounts / actualSample )

# --------------------------------------------------
# CHASE PROGRESS
# --------------------------------------------------
#
# 0 = 24 balls remaining
# 1 = 1 ball remaining

# --------------------------------------------------
ballsRemainingIndex = ( actualRunCounts.index .to_numpy( dtype=float ) )
chaseProgress = ( 24 - ballsRemainingIndex ) / 23

# --------------------------------------------------
# BASIC SMOOTHING WEIGHTS
# --------------------------------------------------
smoothingWeights = np.sqrt( actualSample.to_numpy( dtype=float ) )

# --------------------------------------------------
# SMOOTH RUN PROBABILITIES
# --------------------------------------------------
actualRunProbabilityArray = ( actualRunProbabilities .to_numpy( dtype=float ) )
smoothedLogRunProbabilities = np.zeros_like( actualRunProbabilityArray, dtype=float )
for columnPosition, outcome in enumerate( outcomeValues ):
    probability = ( actualRunProbabilityArray[ :, columnPosition ] )
    logProbability = np.log( np.clip( probability, MIN_PROBABILITY, None ) )
    coefficients = np.polyfit( chaseProgress, logProbability, deg=RUN_POLYNOMIAL_DEGREE, w=smoothingWeights )
    smoothedLogRunProbabilities[ :, columnPosition ] = np.polyval( coefficients, chaseProgress )
smoothedRunProbabilityArray = np.exp( smoothedLogRunProbabilities )
smoothedRunProbabilityArray = ( smoothedRunProbabilityArray / smoothedRunProbabilityArray.sum( axis=1, keepdims=True ) )
smoothedRunProbabilities = pd.DataFrame(smoothedRunProbabilityArray, index=actualRunProbabilities.index, columns=actualRunProbabilities.columns)

# --------------------------------------------------
# SMOOTH OVERALL WICKET PROBABILITY
# --------------------------------------------------
actualWicketProbabilityArray = ( actualWicketProbability .to_numpy( dtype=float ) )
actualWicketProbabilityClipped = np.clip( actualWicketProbabilityArray, MIN_PROBABILITY, 1 - MIN_PROBABILITY )
actualWicketLogit = np.log( actualWicketProbabilityClipped / ( 1 - actualWicketProbabilityClipped ) )
wicketEndWeight = ( 1 + ( WICKET_END_WEIGHT_STRENGTH * np.power( chaseProgress, WICKET_END_WEIGHT_POWER ) ) )
wicketSmoothingWeights = ( smoothingWeights * wicketEndWeight )
wicketCoefficients = np.polyfit( chaseProgress, actualWicketLogit, deg=WICKET_POLYNOMIAL_DEGREE, w=wicketSmoothingWeights )
smoothedWicketLogit = np.polyval( wicketCoefficients, chaseProgress )
smoothedWicketProbability = ( 1 / ( 1 + np.exp( -smoothedWicketLogit ) ) )

# --------------------------------------------------
# CONDITIONAL WICKET DATA
# --------------------------------------------------
#
# P(wicket | batsman runs, balls remaining)
# --------------------------------------------------
conditionalWicketSample = trainData.groupby(['inningBallsRemaining', 'batsmanRuns']).size().unstack(fill_value=0)
conditionalWicketSample = conditionalWicketSample.reindex(index=range(1, 25), fill_value=0).reindex(columns=outcomeValues, fill_value=0)
conditionalWicketCounts = trainData.groupby(['inningBallsRemaining', 'batsmanRuns'])['isWicket'].sum().unstack(fill_value=0)
conditionalWicketCounts = conditionalWicketCounts.reindex(index=range(1, 25), fill_value=0).reindex(columns=outcomeValues, fill_value=0)
actualConditionalWicketProbability = ( conditionalWicketCounts / conditionalWicketSample.replace( 0, np.nan ) )

# --------------------------------------------------
# SMOOTH CONDITIONAL WICKET PROBABILITIES
# --------------------------------------------------
smoothedConditionalWicketProbability = pd.DataFrame(index=range(1, 25), columns=outcomeValues, dtype=float)
for outcome in outcomeValues:
    if outcome in IMPOSSIBLE_WICKET_RUN_OUTCOMES:
        smoothedConditionalWicketProbability[ outcome ] = 0.0
        continue
    probability = ( actualConditionalWicketProbability[ outcome ] .to_numpy( dtype=float ) )
    outcomeSample = ( conditionalWicketSample[ outcome ] .to_numpy( dtype=float ) )
    validMask = ( np.isfinite( probability ) & ( outcomeSample > 0 ) )
    probabilityValid = np.clip( probability[ validMask ], MIN_PROBABILITY, 1 - MIN_PROBABILITY )
    outcomeLogit = np.log( probabilityValid / ( 1 - probabilityValid ) )
    outcomeWeights = np.sqrt( outcomeSample[ validMask ] )
    outcomeChaseProgress = ( chaseProgress[ validMask ] )
    if validMask.sum() > ( CONDITIONAL_WICKET_POLYNOMIAL_DEGREE ):
        coefficients = np.polyfit( outcomeChaseProgress, outcomeLogit, deg=CONDITIONAL_WICKET_POLYNOMIAL_DEGREE, w=outcomeWeights )
        fittedLogit = np.polyval( coefficients, chaseProgress )
        fittedProbability = ( 1 / ( 1 + np.exp( -fittedLogit ) ) )
    else:
        pooledWickets = ( conditionalWicketCounts[ outcome ].sum() )
        pooledBalls = ( conditionalWicketSample[ outcome ].sum() )
        pooledProbability = ( pooledWickets / pooledBalls )
        fittedProbability = np.repeat( pooledProbability, 24 )
    smoothedConditionalWicketProbability[ outcome ] = ( fittedProbability )

# --------------------------------------------------
# CALIBRATE CONDITIONAL WICKET PROBABILITIES
# --------------------------------------------------
#
# Force:
#
# sum(
#     P(run) *
#     P(wicket | run)
# )
#
# =
#
# smooth overall wicket probability
# --------------------------------------------------
calibratedConditionalWicketProbability = pd.DataFrame(index=range(1, 25), columns=outcomeValues, dtype=float)
for rowPosition in range(24):
    runProbabilityVector = ( smoothedRunProbabilityArray[ rowPosition ] )
    targetWicketProbability = ( smoothedWicketProbability[ rowPosition ] )
    conditionalProbabilityVector = ( smoothedConditionalWicketProbability .iloc[ rowPosition ] .to_numpy( dtype=float ) )
    impossibleMask = np.array( [ outcome in IMPOSSIBLE_WICKET_RUN_OUTCOMES for outcome in outcomeValues ] )
    possibleMask = ( ~impossibleMask )
    clippedConditionalProbability = np.clip( conditionalProbabilityVector[ possibleMask ], MIN_PROBABILITY, 1 - MIN_PROBABILITY )
    baseConditionalLogit = np.log( clippedConditionalProbability / ( 1 - clippedConditionalProbability ) )
    lowShift = -20.0
    highShift = 20.0
    for _ in range(100):
        middleShift = ( lowShift + highShift ) / 2
        shiftedProbability = ( 1 / ( 1 + np.exp( -( baseConditionalLogit + middleShift ) ) ) )
        impliedWicketProbability = np.sum( runProbabilityVector[ possibleMask ] * shiftedProbability )
        if impliedWicketProbability < ( targetWicketProbability ):
            lowShift = ( middleShift )
        else:
            highShift = ( middleShift )
    finalShift = ( lowShift + highShift ) / 2
    calibratedProbabilityVector = np.zeros( len( outcomeValues ), dtype=float )
    calibratedProbabilityVector[ possibleMask ] = ( 1 / ( 1 + np.exp( -( baseConditionalLogit + finalShift ) ) ) )
    calibratedProbabilityVector[ impossibleMask ] = 0.0
    calibratedConditionalWicketProbability.iloc[ rowPosition ] = calibratedProbabilityVector

# --------------------------------------------------
# CREATE BASELINE JOINT EVENTS
# --------------------------------------------------
jointEvents = []
for outcome in outcomeValues:
    jointEvents.append( ( int(outcome), 0 ) )
    if outcome not in IMPOSSIBLE_WICKET_RUN_OUTCOMES:
        jointEvents.append( ( int(outcome), 1 ) )
numberOfJointEvents = len( jointEvents )
baselineJointProbabilityArray = np.zeros( ( 24, numberOfJointEvents ), dtype=float )
for eventPosition, event in enumerate( jointEvents ):
    outcome = event[0]
    isWicket = event[1]
    outcomePosition = ( outcomeValues.index( outcome ) )
    runProbability = ( smoothedRunProbabilityArray[ :, outcomePosition ] )
    wicketGivenRun = ( calibratedConditionalWicketProbability[ outcome ].to_numpy( dtype=float ) )
    if isWicket == 1:
        baselineJointProbabilityArray[ :, eventPosition ] = ( runProbability * wicketGivenRun )
    else:
        baselineJointProbabilityArray[ :, eventPosition ] = ( runProbability * ( 1 - wicketGivenRun ) )
# normalise for numerical protection
baselineJointProbabilityArray = ( baselineJointProbabilityArray / baselineJointProbabilityArray.sum( axis=1, keepdims=True ) )

# --------------------------------------------------
# BASELINE EXPECTED BATSMAN RUNS
# --------------------------------------------------
baseMeanRunsByBalls = np.zeros( 24 )
for outcomePosition, outcome in enumerate( outcomeValues ):
    baseMeanRunsByBalls += ( outcome * smoothedRunProbabilityArray[ :, outcomePosition ] )

# --------------------------------------------------
# ADD BASELINE VARIABLES TO TRAIN DATA
# --------------------------------------------------
trainBallsIndex = ( trainData[ 'inningBallsRemaining' ] .astype(int) .to_numpy() - 1 )
trainData[ 'base_m_batsmanRunsBall' ] = ( baseMeanRunsByBalls[ trainBallsIndex ] )

# --------------------------------------------------
# PRESSURE
# --------------------------------------------------
#
# 1.0:
# required scoring rate equals the baseline
# expected batsman scoring rate.
# --------------------------------------------------
trainData[ 'pressure' ] = ( trainData[ 'runsRequiredPerBall' ] / trainData[ 'base_m_batsmanRunsBall' ] )
trainData[ 'logPressure' ] = np.log( trainData[ 'pressure' ] )
trainData[ 'chaseProgress' ] = ( 24 - trainData[ 'inningBallsRemaining' ] ) / 23

# --------------------------------------------------
# PRESSURE FEATURE MATRIX
# --------------------------------------------------
#
# No intercept.
#
# Therefore when:
#
# pressure = 1
# logPressure = 0
#
# every pressure adjustment becomes zero and the
# model returns exactly to the baseline.

# --------------------------------------------------
logPressure = ( trainData[ 'logPressure' ].to_numpy( dtype=float ) )
trainChaseProgress = ( trainData[ 'chaseProgress' ].to_numpy( dtype=float ) )
XPressure = np.column_stack(
    [
        logPressure,
        np.square(
            logPressure
        ),
        np.power(
            logPressure,
            3
        ),
        (
            logPressure *
            trainChaseProgress
        ),
        (
            np.square(
                logPressure
            ) *
            trainChaseProgress
        )
    ]
)
pressureFeatureNames = [
    'logPressure',
    'logPressureSquared',
    'logPressureCubed',
    'logPressure_x_chaseProgress',
    'logPressureSquared_x_chaseProgress'
]
numberOfPressureFeatures = ( XPressure.shape[1] )

# --------------------------------------------------
# CREATE ACTUAL JOINT EVENT TARGET
# --------------------------------------------------
jointEventLookup = {
    event: position
    for position, event
    in enumerate(
        jointEvents
    )
}
jointTarget = np.array( [ jointEventLookup[ ( int(run), int(wicket) ) ] for run, wicket in zip( trainData[ 'batsmanRuns' ], trainData[ 'isWicket' ] ) ], dtype=int )

# --------------------------------------------------
# BASELINE JOINT PROBABILITIES FOR TRAINING ROWS
# --------------------------------------------------
trainingBaselineJointProbability = ( baselineJointProbabilityArray[ trainBallsIndex ] )
trainingBaselineLogProbability = np.log( np.clip( trainingBaselineJointProbability, MIN_PROBABILITY, None ) )

# --------------------------------------------------
# PRESSURE MODEL
# --------------------------------------------------
#
# Multinomial softmax with the baseline probabilities
# acting as fixed offsets.
#
#
# score(event) =
#
# log(base probability)
#
# +
#
# pressure adjustment
#
#
# One event is used as the reference event so that
# the model is identifiable.

# --------------------------------------------------
referenceEventPosition = 0
nonReferenceEventPositions = [
    position
    for position
    in range(
        numberOfJointEvents
    )
    if position != referenceEventPosition
]
numberOfFreeEvents = len( nonReferenceEventPositions )
def pressureObjective( parameters ):
    coefficients = parameters.reshape( ( numberOfFreeEvents, numberOfPressureFeatures ) )
    fullCoefficients = np.zeros( ( numberOfJointEvents, numberOfPressureFeatures ) )
    fullCoefficients[ nonReferenceEventPositions, : ] = coefficients
    adjustment = ( XPressure @ fullCoefficients.T )
    scores = ( trainingBaselineLogProbability + adjustment )
    scores = ( scores - scores.max( axis=1, keepdims=True ) )
    expScores = np.exp( scores )
    probabilities = ( expScores / expScores.sum( axis=1, keepdims=True ) )
    rowIndex = np.arange( len( jointTarget ) )
    negativeLogLikelihood = -np.mean( np.log( probabilities[ rowIndex, jointTarget ] + MIN_PROBABILITY ) )
    regularisation = ( 0.5 * PRESSURE_L2 * np.sum( np.square( coefficients ) ) )
    objective = ( negativeLogLikelihood + regularisation )
    error = probabilities.copy()
    error[ rowIndex, jointTarget ] -= 1
    gradientFull = ( error.T @ XPressure ) / len( jointTarget )
    gradient = ( gradientFull[ nonReferenceEventPositions, : ] + ( PRESSURE_L2 * coefficients ) )
    return ( objective, gradient.ravel() )

# --------------------------------------------------
# FIT PRESSURE MODEL
# --------------------------------------------------
initialParameters = np.zeros( numberOfFreeEvents * numberOfPressureFeatures )
pressureFitResult = minimize( fun=lambda parameters: pressureObjective( parameters ), x0=initialParameters, method='L-BFGS-B', jac=True, options={ 'maxiter': PRESSURE_MAX_ITER } )
if not pressureFitResult.success:
    raise RuntimeError( pressureFitResult.message )
pressureCoefficientsArray = ( pressureFitResult.x.reshape( ( numberOfFreeEvents, numberOfPressureFeatures ) ) )
fullPressureCoefficients = np.zeros( ( numberOfJointEvents, numberOfPressureFeatures ) )
fullPressureCoefficients[ nonReferenceEventPositions, : ] = pressureCoefficientsArray

# --------------------------------------------------
# PRESSURE COEFFICIENT TABLE
# --------------------------------------------------
pressureCoefficients = pd.DataFrame(fullPressureCoefficients, columns=pressureFeatureNames)
pressureCoefficients[ 'batsmanRuns' ] = [ event[0] for event in jointEvents ]
pressureCoefficients[ 'isWicket' ] = [ event[1] for event in jointEvents ]

# --------------------------------------------------
# PRESSURE-ADJUSTED JOINT PROBABILITIES FOR TRAINING ROWS
# --------------------------------------------------
trainingPressureAdjustment = ( XPressure @ fullPressureCoefficients.T )
trainingPressureScores = ( trainingBaselineLogProbability + trainingPressureAdjustment )
trainingPressureScores = ( trainingPressureScores - trainingPressureScores.max( axis=1, keepdims=True ) )
trainingPressureExpScores = np.exp( trainingPressureScores )
trainingPressureJointProbability = ( trainingPressureExpScores / trainingPressureExpScores.sum( axis=1, keepdims=True ) )
trainingPressureLogProbability = np.log( np.clip( trainingPressureJointProbability, MIN_PROBABILITY, None ) )

# --------------------------------------------------
# BATSMAN ORDER FEATURE MATRIX
# --------------------------------------------------
orderTrainingMask = ( trainData['ord'].notna() & trainData['ord'].between(1, 11) )
orderTrainingMaskArray = orderTrainingMask.to_numpy()
orderScaled = ( ( trainData.loc[orderTrainingMask, 'ord'].to_numpy( dtype=float ) - 6.0 ) / 5.0 )
orderLogPressure = trainData.loc[orderTrainingMask, 'logPressure'].to_numpy( dtype=float )
orderPolynomialFeatures = [ np.power( orderScaled, degree ) for degree in range( 1, ORDER_POLYNOMIAL_DEGREE + 1 ) ]
orderRawFeatures = np.column_stack( orderPolynomialFeatures + [ feature * orderLogPressure for feature in orderPolynomialFeatures ] + [ feature * np.square( orderLogPressure ) for feature in orderPolynomialFeatures[:2] ] )
orderFeatureMeans = orderRawFeatures.mean( axis=0, keepdims=True )
XOrder = ( orderRawFeatures - orderFeatureMeans )
orderFeatureNames = [ f'orderDegree{degree}' for degree in range( 1, ORDER_POLYNOMIAL_DEGREE + 1 ) ] + [ f'orderDegree{degree}_x_logPressure' for degree in range( 1, ORDER_POLYNOMIAL_DEGREE + 1 ) ] + [ f'orderDegree{degree}_x_logPressureSquared' for degree in range( 1, 3 ) ]
numberOfOrderFeatures = ( XOrder.shape[1] )
orderJointTarget = jointTarget[ orderTrainingMaskArray ]
orderTrainingPressureLogProbability = trainingPressureLogProbability[ orderTrainingMaskArray ]

# --------------------------------------------------
# BATSMAN ORDER MODEL
# --------------------------------------------------
def orderObjective( parameters ):
    coefficients = parameters.reshape( ( numberOfFreeEvents, numberOfOrderFeatures ) )
    fullCoefficients = np.zeros( ( numberOfJointEvents, numberOfOrderFeatures ) )
    fullCoefficients[ nonReferenceEventPositions, : ] = coefficients
    adjustment = ( XOrder @ fullCoefficients.T )
    scores = ( orderTrainingPressureLogProbability + adjustment )
    scores = ( scores - scores.max( axis=1, keepdims=True ) )
    expScores = np.exp( scores )
    probabilities = ( expScores / expScores.sum( axis=1, keepdims=True ) )
    rowIndex = np.arange( len( orderJointTarget ) )
    negativeLogLikelihood = -np.mean( np.log( probabilities[ rowIndex, orderJointTarget ] + MIN_PROBABILITY ) )
    regularisation = ( 0.5 * ORDER_L2 * np.sum( np.square( coefficients ) ) )
    objective = ( negativeLogLikelihood + regularisation )
    error = probabilities.copy()
    error[ rowIndex, orderJointTarget ] -= 1
    gradientFull = ( error.T @ XOrder ) / len( orderJointTarget )
    gradient = ( gradientFull[ nonReferenceEventPositions, : ] + ( ORDER_L2 * coefficients ) )
    return ( objective, gradient.ravel() )

# --------------------------------------------------
# FIT BATSMAN ORDER MODEL
# --------------------------------------------------
initialOrderParameters = np.zeros( numberOfFreeEvents * numberOfOrderFeatures )
orderFitResult = minimize( fun=lambda parameters: orderObjective( parameters ), x0=initialOrderParameters, method='L-BFGS-B', jac=True, options={ 'maxiter': ORDER_MAX_ITER } )
if not orderFitResult.success:
    raise RuntimeError( orderFitResult.message )
orderCoefficientsArray = ( orderFitResult.x.reshape( ( numberOfFreeEvents, numberOfOrderFeatures ) ) )
fullOrderCoefficients = np.zeros( ( numberOfJointEvents, numberOfOrderFeatures ) )
fullOrderCoefficients[ nonReferenceEventPositions, : ] = orderCoefficientsArray

# --------------------------------------------------
# BATSMAN ORDER COEFFICIENT TABLE
# --------------------------------------------------
orderCoefficients = pd.DataFrame(fullOrderCoefficients, columns=orderFeatureNames)
orderCoefficients[ 'batsmanRuns' ] = [ event[0] for event in jointEvents ]
orderCoefficients[ 'isWicket' ] = [ event[1] for event in jointEvents ]

# --------------------------------------------------
# BATSMAN-ORDER-ADJUSTED JOINT PROBABILITIES FOR TRAINING ROWS
# --------------------------------------------------
trainingOrderAdjustment = ( XOrder @ fullOrderCoefficients.T )
trainingOrderScores = ( orderTrainingPressureLogProbability + trainingOrderAdjustment )
trainingOrderScores = ( trainingOrderScores - trainingOrderScores.max( axis=1, keepdims=True ) )
trainingOrderExpScores = np.exp( trainingOrderScores )
trainingOrderJointProbability = ( trainingOrderExpScores / trainingOrderExpScores.sum( axis=1, keepdims=True ) )
trainingOrderLogProbability = np.log( np.clip( trainingOrderJointProbability, MIN_PROBABILITY, None ) )

# --------------------------------------------------
# WICKET RESOURCE FEATURE MATRIX
# --------------------------------------------------
wicketsRemaining = ( 10.0 - trainData.loc[orderTrainingMask, 'totalInningWickets'].to_numpy( dtype=float ) )
resource = ( wicketsRemaining / 10.0 )
resourceLogPressure = trainData.loc[orderTrainingMask, 'logPressure'].to_numpy( dtype=float )
resourceRawFeatures = np.column_stack([resource, np.square(resource), resource * resourceLogPressure, np.square(resource) * resourceLogPressure])
resourceFeatureMeans = resourceRawFeatures.mean( axis=0, keepdims=True )
XResource = ( resourceRawFeatures - resourceFeatureMeans )
resourceFeatureNames = ['resource', 'resourceSquared', 'resource_x_logPressure', 'resourceSquared_x_logPressure']
numberOfResourceFeatures = ( XResource.shape[1] )
resourceJointTarget = orderJointTarget

# --------------------------------------------------
# WICKET RESOURCE MODEL
# --------------------------------------------------
def resourceObjective( parameters ):
    coefficients = parameters.reshape( ( numberOfFreeEvents, numberOfResourceFeatures ) )
    fullCoefficients = np.zeros( ( numberOfJointEvents, numberOfResourceFeatures ) )
    fullCoefficients[ nonReferenceEventPositions, : ] = coefficients
    adjustment = ( XResource @ fullCoefficients.T )
    scores = ( trainingOrderLogProbability + adjustment )
    scores = ( scores - scores.max( axis=1, keepdims=True ) )
    expScores = np.exp( scores )
    probabilities = ( expScores / expScores.sum( axis=1, keepdims=True ) )
    rowIndex = np.arange( len( resourceJointTarget ) )
    negativeLogLikelihood = -np.mean( np.log( probabilities[ rowIndex, resourceJointTarget ] + MIN_PROBABILITY ) )
    regularisation = ( 0.5 * RESOURCE_L2 * np.sum( np.square( coefficients ) ) )
    objective = ( negativeLogLikelihood + regularisation )
    error = probabilities.copy()
    error[ rowIndex, resourceJointTarget ] -= 1
    gradientFull = ( error.T @ XResource ) / len( resourceJointTarget )
    gradient = ( gradientFull[ nonReferenceEventPositions, : ] + ( RESOURCE_L2 * coefficients ) )
    return ( objective, gradient.ravel() )

# --------------------------------------------------
# FIT WICKET RESOURCE MODEL
# --------------------------------------------------
initialResourceParameters = np.zeros( numberOfFreeEvents * numberOfResourceFeatures )
resourceFitResult = minimize( fun=lambda parameters: resourceObjective( parameters ), x0=initialResourceParameters, method='L-BFGS-B', jac=True, options={ 'maxiter': RESOURCE_MAX_ITER } )
if not resourceFitResult.success:
    raise RuntimeError( resourceFitResult.message )
resourceCoefficientsArray = ( resourceFitResult.x.reshape( ( numberOfFreeEvents, numberOfResourceFeatures ) ) )
fullResourceCoefficients = np.zeros( ( numberOfJointEvents, numberOfResourceFeatures ) )
fullResourceCoefficients[ nonReferenceEventPositions, : ] = resourceCoefficientsArray

# --------------------------------------------------
# WICKET RESOURCE COEFFICIENT TABLE
# --------------------------------------------------
resourceCoefficients = pd.DataFrame(fullResourceCoefficients, columns=resourceFeatureNames)
resourceCoefficients[ 'batsmanRuns' ] = [ event[0] for event in jointEvents ]
resourceCoefficients[ 'isWicket' ] = [ event[1] for event in jointEvents ]

# --------------------------------------------------
# BASELINE VALUES FOR MASTER LOOKUP
# --------------------------------------------------
masterBallsIndex = ( masterLookup[ 'inningBallsRemaining' ] .astype(int) .to_numpy() - 1 )
masterLookup[ 'base_m_batsmanRunsBall' ] = ( baseMeanRunsByBalls[ masterBallsIndex ] )
masterLookup[ 'base_wicket' ] = ( smoothedWicketProbability[ masterBallsIndex ] )

# --------------------------------------------------
# BASELINE RUN PROBABILITIES
# --------------------------------------------------
for outcomePosition, outcome in enumerate( outcomeValues ):
    masterLookup[ f'base_{int(outcome)}' ] = ( smoothedRunProbabilityArray[ masterBallsIndex, outcomePosition ] )

# --------------------------------------------------
# BASELINE JOINT PROBABILITIES
# --------------------------------------------------
for eventPosition, event in enumerate( jointEvents ):
    outcome = event[0]
    isWicket = event[1]
    if isWicket == 1:
        column = ( f'base_{outcome}_wicket' )
    else:
        column = ( f'base_{outcome}_noWicket' )
    masterLookup[ column ] = ( baselineJointProbabilityArray[ masterBallsIndex, eventPosition ] )
# impossible wicket outcomes remain explicitly zero
for outcome in IMPOSSIBLE_WICKET_RUN_OUTCOMES:
    masterLookup[ f'base_{outcome}_wicket' ] = 0.0

# --------------------------------------------------
# MASTER LOOKUP PRESSURE FEATURES
# --------------------------------------------------
masterLookup[ 'pressure' ] = ( masterLookup[ 'runsRequiredPerBall' ] / masterLookup[ 'base_m_batsmanRunsBall' ] )
masterLookup[ 'logPressure' ] = np.log( masterLookup[ 'pressure' ] )
masterLookup[ 'chaseProgress' ] = ( 24 - masterLookup[ 'inningBallsRemaining' ] ) / 23
masterLogPressure = ( masterLookup[ 'logPressure' ].to_numpy( dtype=float ) )
masterChaseProgress = ( masterLookup[ 'chaseProgress' ].to_numpy( dtype=float ) )
XMasterPressure = np.column_stack(
    [
        masterLogPressure,
        np.square(
            masterLogPressure
        ),
        np.power(
            masterLogPressure,
            3
        ),
        (
            masterLogPressure *
            masterChaseProgress
        ),
        (
            np.square(
                masterLogPressure
            ) *
            masterChaseProgress
        )
    ]
)

# --------------------------------------------------
# PRESSURE-ADJUSTED JOINT PROBABILITIES
# --------------------------------------------------
masterBaselineJointProbability = ( baselineJointProbabilityArray[ masterBallsIndex ] )
masterBaselineLogProbability = np.log( np.clip( masterBaselineJointProbability, MIN_PROBABILITY, None ) )
masterPressureAdjustment = ( XMasterPressure @ fullPressureCoefficients.T )
masterPressureScores = ( masterBaselineLogProbability + masterPressureAdjustment )
masterPressureScores = ( masterPressureScores - masterPressureScores.max( axis=1, keepdims=True ) )
masterPressureExpScores = np.exp( masterPressureScores )
pressureJointProbabilityArray = ( masterPressureExpScores / masterPressureExpScores.sum( axis=1, keepdims=True ) )

# --------------------------------------------------
# STORE PRESSURE-ADJUSTED JOINT PROBABILITIES
# --------------------------------------------------
for eventPosition, event in enumerate( jointEvents ):
    outcome = event[0]
    isWicket = event[1]
    if isWicket == 1:
        column = ( f'pressure_{outcome}_wicket' )
    else:
        column = ( f'pressure_{outcome}_noWicket' )
    masterLookup[ column ] = ( pressureJointProbabilityArray[ :, eventPosition ] )
# impossible events stay explicitly zero
for outcome in IMPOSSIBLE_WICKET_RUN_OUTCOMES:
    masterLookup[ f'pressure_{outcome}_wicket' ] = 0.0


# --------------------------------------------------
# PRESSURE-ADJUSTED RUN MARGINALS
# --------------------------------------------------
for outcome in outcomeValues:
    noWicketColumn = ( f'pressure_{int(outcome)}_noWicket' )
    wicketColumn = ( f'pressure_{int(outcome)}_wicket' )
    if wicketColumn in masterLookup.columns:
        masterLookup[ f'pressure_{int(outcome)}' ] = ( masterLookup[ noWicketColumn ] + masterLookup[ wicketColumn ] )
    else:
        masterLookup[ f'pressure_{int(outcome)}' ] = ( masterLookup[ noWicketColumn ] )


# --------------------------------------------------
# PRESSURE-ADJUSTED WICKET PROBABILITY
# --------------------------------------------------
pressureWicketColumns = [
    f'pressure_{event[0]}_wicket'
    for event in jointEvents
    if event[1] == 1
]
masterLookup[ 'pressure_wicket' ] = ( masterLookup[ pressureWicketColumns ].sum( axis=1 ) )


# --------------------------------------------------
# PRESSURE-ADJUSTED EXPECTED BATSMAN RUNS
# --------------------------------------------------
masterLookup[ 'pressure_m_batsmanRunsBall' ] = sum( outcome * masterLookup[ f'pressure_{int(outcome)}' ] for outcome in outcomeValues )


# --------------------------------------------------
# PROBABILITY SUM CHECK
# --------------------------------------------------
pressureJointColumns = []
for event in jointEvents:
    outcome = event[0]
    isWicket = event[1]
    if isWicket == 1:
        pressureJointColumns.append( f'pressure_{outcome}_wicket' )
    else:
        pressureJointColumns.append( f'pressure_{outcome}_noWicket' )
masterLookup[ 'pressureJointProbabilitySum' ] = ( masterLookup[ pressureJointColumns ].sum( axis=1 ) )


# --------------------------------------------------
# EXPAND MASTER LOOKUP BY BATSMAN ORDER
# --------------------------------------------------
masterLookup = masterLookup.loc[masterLookup.index.repeat(masterLookup['totalInningWickets'] + 2)].copy()
masterLookup['ord'] = masterLookup.groupby(['runsRequired', 'totalInningWickets', 'inningBallsRemaining']).cumcount() + 1

# --------------------------------------------------
# MASTER LOOKUP BATSMAN ORDER FEATURES
# --------------------------------------------------
masterOrderScaled = ( ( masterLookup['ord'].to_numpy( dtype=float ) - 6.0 ) / 5.0 )
masterOrderLogPressure = masterLookup['logPressure'].to_numpy( dtype=float )
masterOrderPolynomialFeatures = [ np.power( masterOrderScaled, degree ) for degree in range( 1, ORDER_POLYNOMIAL_DEGREE + 1 ) ]
masterOrderRawFeatures = np.column_stack( masterOrderPolynomialFeatures + [ feature * masterOrderLogPressure for feature in masterOrderPolynomialFeatures ] + [ feature * np.square( masterOrderLogPressure ) for feature in masterOrderPolynomialFeatures[:2] ] )
XMasterOrder = ( masterOrderRawFeatures - orderFeatureMeans )

# --------------------------------------------------
# BATSMAN-ORDER-ADJUSTED JOINT PROBABILITIES
# --------------------------------------------------
masterPressureJointProbability = masterLookup[pressureJointColumns].to_numpy( dtype=float )
masterPressureLogProbability = np.log( np.clip( masterPressureJointProbability, MIN_PROBABILITY, None ) )
masterOrderAdjustment = ( XMasterOrder @ fullOrderCoefficients.T )
masterOrderScores = ( masterPressureLogProbability + masterOrderAdjustment )
masterOrderScores = ( masterOrderScores - masterOrderScores.max( axis=1, keepdims=True ) )
masterOrderExpScores = np.exp( masterOrderScores )
pressureJointProbabilityArray = ( masterOrderExpScores / masterOrderExpScores.sum( axis=1, keepdims=True ) )

# --------------------------------------------------
# MASTER LOOKUP WICKET RESOURCE FEATURES
# --------------------------------------------------
masterWicketsRemaining = ( 10.0 - masterLookup['totalInningWickets'].to_numpy( dtype=float ) )
masterResource = ( masterWicketsRemaining / 10.0 )
masterResourceLogPressure = masterLookup['logPressure'].to_numpy( dtype=float )
masterResourceRawFeatures = np.column_stack([masterResource, np.square(masterResource), masterResource * masterResourceLogPressure, np.square(masterResource) * masterResourceLogPressure])
XMasterResource = ( masterResourceRawFeatures - resourceFeatureMeans )

# --------------------------------------------------
# WICKET-RESOURCE-ADJUSTED JOINT PROBABILITIES
# --------------------------------------------------
masterOrderLogProbability = np.log( np.clip( pressureJointProbabilityArray, MIN_PROBABILITY, None ) )
masterResourceAdjustment = ( XMasterResource @ fullResourceCoefficients.T )
masterResourceScores = ( masterOrderLogProbability + masterResourceAdjustment )
masterResourceScores = ( masterResourceScores - masterResourceScores.max( axis=1, keepdims=True ) )
masterResourceExpScores = np.exp( masterResourceScores )
pressureJointProbabilityArray = ( masterResourceExpScores / masterResourceExpScores.sum( axis=1, keepdims=True ) )

# --------------------------------------------------
# STORE FINAL JOINT PROBABILITIES
# --------------------------------------------------
for eventPosition, event in enumerate( jointEvents ):
    outcome = event[0]
    isWicket = event[1]
    if isWicket == 1:
        column = ( f'pressure_{outcome}_wicket' )
    else:
        column = ( f'pressure_{outcome}_noWicket' )
    masterLookup[ column ] = ( pressureJointProbabilityArray[ :, eventPosition ] )
for outcome in IMPOSSIBLE_WICKET_RUN_OUTCOMES:
    masterLookup[ f'pressure_{outcome}_wicket' ] = 0.0

# --------------------------------------------------
# FINAL RUN MARGINALS
# --------------------------------------------------
for outcome in outcomeValues:
    noWicketColumn = ( f'pressure_{int(outcome)}_noWicket' )
    wicketColumn = ( f'pressure_{int(outcome)}_wicket' )
    if wicketColumn in masterLookup.columns:
        masterLookup[ f'pressure_{int(outcome)}' ] = ( masterLookup[ noWicketColumn ] + masterLookup[ wicketColumn ] )
    else:
        masterLookup[ f'pressure_{int(outcome)}' ] = ( masterLookup[ noWicketColumn ] )

# --------------------------------------------------
# FINAL WICKET PROBABILITY
# --------------------------------------------------
masterLookup[ 'pressure_wicket' ] = ( masterLookup[ pressureWicketColumns ].sum( axis=1 ) )

# --------------------------------------------------
# FINAL EXPECTED BATSMAN RUNS
# --------------------------------------------------
masterLookup[ 'pressure_m_batsmanRunsBall' ] = sum( outcome * masterLookup[ f'pressure_{int(outcome)}' ] for outcome in outcomeValues )

# --------------------------------------------------
# PROBABILITY SUM CHECK
# --------------------------------------------------
masterLookup[ 'pressureJointProbabilitySum' ] = ( masterLookup[ pressureJointColumns ].sum( axis=1 ) )


# --------------------------------------------------
# CREATE BALLS-REMAINING BASELINE PIVOT
# --------------------------------------------------
actualByBallsRemainingPivot = pd.DataFrame(index=range(1, 25))
actualByBallsRemainingPivot.index.name = ( 'inningBallsRemaining' )
for outcome in outcomeValues:
    actualByBallsRemainingPivot[ f'actual_{int(outcome)}' ] = ( actualRunProbabilities[ outcome ] )
    actualByBallsRemainingPivot[ f'smooth_{int(outcome)}' ] = ( smoothedRunProbabilities[ outcome ] )
actualByBallsRemainingPivot[ 'actualSample' ] = ( actualSample )
actualByBallsRemainingPivot[ 'actual_wicket' ] = ( actualWicketProbability )
actualByBallsRemainingPivot[ 'smooth_wicket' ] = ( smoothedWicketProbability )
actualByBallsRemainingPivot[ 'actual_m_batsmanRunsBall' ] = sum( outcome * actualByBallsRemainingPivot[ f'actual_{int(outcome)}' ] for outcome in outcomeValues )
actualByBallsRemainingPivot[ 'smooth_m_batsmanRunsBall' ] = sum( outcome * actualByBallsRemainingPivot[ f'smooth_{int(outcome)}' ] for outcome in outcomeValues )
actualByBallsRemainingPivot = actualByBallsRemainingPivot.reset_index().sort_values('inningBallsRemaining', ascending=False).reset_index(drop=True)




# --------------------------------------------------
# PLOT BASELINE:
# ACTUAL VS SMOOTHED BY BALLS REMAINING
# --------------------------------------------------
plotData = actualByBallsRemainingPivot.sort_values('inningBallsRemaining').copy()
runOutcomesToPlot = [
    0,
    1,
    2,
    3,
    4,
    6
]
fig, axes = plt.subplots( 4, 2, figsize=(14, 16) )
axes = axes.flatten()
for i, outcome in enumerate( runOutcomesToPlot ):
    ax = axes[i]
    ax.plot( plotData[ 'inningBallsRemaining' ], plotData[ f'actual_{outcome}' ], marker='o', label='Actual' )
    ax.plot( plotData[ 'inningBallsRemaining' ], plotData[ f'smooth_{outcome}' ], linewidth=2, label='Smoothed' )
    ax.set_title( f'Outcome {outcome}' )
    ax.set_xlabel( 'Balls Remaining' )
    ax.set_ylabel( 'Probability' )
    ax.grid( True, alpha=0.3 )
    ax.invert_xaxis()
    if i == 0:
        ax.legend()
ax = axes[6]
ax.plot( plotData[ 'inningBallsRemaining' ], plotData[ 'actual_wicket' ], marker='o', label='Actual' )
ax.plot( plotData[ 'inningBallsRemaining' ], plotData[ 'smooth_wicket' ], linewidth=2, label='Smoothed' )
ax.set_title( 'Wicket' )
ax.set_xlabel( 'Balls Remaining' )
ax.set_ylabel( 'Probability' )
ax.grid( True, alpha=0.3 )
ax.invert_xaxis()
axes[7].axis( 'off' )
plt.suptitle( 'Actual vs Smoothed Outcome Probabilities by Balls Remaining', fontsize=16 )
plt.tight_layout()
plt.show()
# masterLookup.to_csv(PROJECT_ROOT / 'men/expBall&runsToCome/outputs/deathBallProbs.csv', index=False)


# --------------------------------------------------
# MODEL RESIDUALS AFTER CONTROLLING FOR PRESSURE, BATSMAN ORDER AND WICKET RESOURCES
# --------------------------------------------------
plotOutcomes = [0, 1, 2, 3, 4, 6]
predictionColumns = [f'pressure_{outcome}' for outcome in plotOutcomes] + ['pressure_wicket']
stateColumns = ['runsRequired', 'totalInningWickets', 'inningBallsRemaining', 'ord']
trainData = trainData[ trainData['ord'].notna() & trainData['ord'].between(1, 11) ]
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
plt.suptitle( 'Ball-Outcome Residuals After Controlling for Chase Pressure, Batsman Order and Wicket Resources', fontsize=16 )
plt.tight_layout()
plt.show()


