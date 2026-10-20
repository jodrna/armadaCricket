import hashlib
import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits
from paths import PROJECT_ROOT

blendStartTime = time.perf_counter()
N_FOLDS = 5
FOLD_SEED = 2718
RANDOM_SEED = 42
stateColumns = ['runsRequired', 'totalInningWickets', 'inningBallsRemaining', 'ord', 'nonStrikerOrd']
ballColumns = stateColumns[:4]
rowColumns = ['matchID', 'ID', 'fold']
CALIBRATION_PENALTY = 0.003
BLEND_PENALTY = 0.0003
DIRECT_TAPER_BALLS = 12
META_MATCH_WEIGHT_POWER = 0.1
dataHash = hashlib.sha256((PROJECT_ROOT / 'men/matchMarket/hybridWin3/data/dataClean.csv').read_bytes()).hexdigest()
runInfo = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/cv/runInfo.csv')
if 'pipelineVersion' not in runInfo or runInfo.at[0, 'pipelineVersion'] != 'full_remodel_v3':
    raise ValueError('These scripts require full-innings outputs. Run scripts 1-5 in order.')
if 'gridHash' not in runInfo or runInfo.at[0, 'gridHash'] != hashlib.sha256((PROJECT_ROOT / 'men/matchMarket/hybridWin3/auxiliaries/1_chaseLookup.csv').read_bytes()).hexdigest():
    raise ValueError('The neural grid changed. Rerun scripts 1-5 in order.')
if runInfo.at[0, 'dataHash'] != dataHash or int(runInfo.at[0, 'nFolds']) != N_FOLDS or int(runInfo.at[0, 'foldSeed']) != FOLD_SEED:
    raise ValueError('Input data or fold settings changed. Rerun 1_deathBallProbs.py and then the remaining scripts in order.')
if any(int(runInfo.at[0, column]) != 1 for column in ['ballReady', 'simulationReady', 'directReady']):
    raise ValueError('Complete 1_deathBallProbs.py, 2_simModel.py and 3_directWinModel.py first.')
resourceHash = hashlib.sha256((PROJECT_ROOT / 'men/matchMarket/hybridWin3/auxiliaries/6_masterLookup.csv').read_bytes()).hexdigest()
if 'directResourceHash' not in runInfo or runInfo.at[0, 'directResourceHash'] != resourceHash:
    raise ValueError('The direct-model resource lookup changed or script 3 has not been run. Run 3_directWinModel.py first.')
runInfo.at[0, 'blendReady'] = 0
runInfo.to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/cv/runInfo.csv', index=False)

# --------------------------------------------------
# OUTER OOF BASE PREDICTIONS: EACH MATCH WAS EXCLUDED
# --------------------------------------------------
simulationOof = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/chaseSimSituations_oof.csv', float_precision='round_trip')
directOof = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/directWinProbabilities_oof.csv', float_precision='round_trip')
resourceKeys = ['inningBallsRemaining', 'totalInningWickets']
coveredResources = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/resourceLookup.csv', usecols=resourceKeys).drop_duplicates(resourceKeys)
simulationRowCount = len(simulationOof)
simulationOof = simulationOof.merge(coveredResources, on=resourceKeys, how='inner', validate='many_to_one')
droppedResourceRows = simulationRowCount - len(simulationOof)
print(f'Blend: dropped {droppedResourceRows:,} historical rows without the direct-model mean/STD lookup.', flush=True)
joinColumns = rowColumns + stateColumns + ['chaseWin']
baseOof = simulationOof.merge(
    directOof[joinColumns + ['directWinProbability']],
    on=joinColumns, how='left', validate='one_to_one',
)
if baseOof.empty or baseOof.duplicated(['matchID', 'ID']).any():
    raise ValueError('Unseen base predictions must have one row per eligible matchID/ID.')
if not np.isfinite(baseOof[['sim_chaseWin%', 'directWinProbability']].to_numpy(float)).all():
    raise ValueError('Base predictions do not align. Rerun scripts 1-3 on the same data.')
if set(baseOof['fold']) != set(range(N_FOLDS)):
    raise ValueError('Expected historical predictions from all five outer folds.')

oofFrames = []
coefficientRows = []
for outerFold in list(range(N_FOLDS)) + [-1]:
    if outerFold >= 0:
        jobName = f'fold_{outerFold}'
        innerFrames = []
        for innerFold in range(N_FOLDS):
            if innerFold == outerFold:
                continue
            foldA, foldB = sorted([outerFold, innerFold])
            # This base fit excluded BOTH folds. Do not substitute global OOF
            # predictions here: their fits could have included the outer fold.
            innerName = f'inner_{foldA}_{foldB}'
            innerSim = pd.read_csv(PROJECT_ROOT / f'men/matchMarket/hybridWin3/cv/chaseSimSituations_{innerName}.csv', float_precision='round_trip')
            innerDirect = pd.read_csv(PROJECT_ROOT / f'men/matchMarket/hybridWin3/cv/directWinProbabilities_{innerName}.csv', float_precision='round_trip')
            innerRows = baseOof.loc[baseOof['fold'].eq(innerFold), joinColumns].copy()
            innerRows = innerRows.merge(
                innerSim[stateColumns + ['sim_chaseWin%']],
                on=stateColumns, how='left', validate='many_to_one',
            ).merge(
                innerDirect[stateColumns + ['directWinProbability']],
                on=stateColumns, how='left', validate='many_to_one',
            )
            innerFrames.append(innerRows)
        metaTrain = pd.concat(innerFrames, ignore_index=True)
        applyData = baseOof.loc[baseOof['fold'].eq(outerFold)].copy()
        if metaTrain['fold'].eq(outerFold).any() or set(metaTrain['matchID']) & set(applyData['matchID']):
            raise RuntimeError('An outer test match reached the calibration/blend training data.')
    else:
        jobName = 'production'
        # Final calibration/blend learns from all OUTER OOF predictions.
        # It is then applied to base models fitted on all eligible data.
        metaTrain = baseOof.copy()
        productionSim = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/chaseSimSituations.csv', float_precision='round_trip')
        productionDirect = pd.read_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/directWinProbabilities.csv', float_precision='round_trip')
        productionSim = productionSim.merge(coveredResources, on=resourceKeys, how='inner', validate='many_to_one')
        applyData = productionSim.merge(productionDirect[stateColumns + ['directWinProbability']], on=stateColumns, how='left', validate='one_to_one')

    for frame in [metaTrain, applyData]:
        values = frame[['sim_chaseWin%', 'directWinProbability']].to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError(f'Missing or invalid base probabilities for {jobName}.')
    target = metaTrain['chaseWin'].to_numpy(float)
    # Mildly reduce the influence of matches that contribute many rows.
    # Power 0 gives ordinary row weighting; power 1 gives equal-match weighting.
    matchCounts = metaTrain.groupby('matchID').size()
    metaWeights = 1 / metaTrain['matchID'].map(matchCounts).to_numpy(float) ** META_MATCH_WEIGHT_POWER
    metaWeights /= metaWeights.sum()
    if len(np.unique(target)) != 2:
        raise ValueError(f'{jobName} requires both match outcomes for calibration and blending.')
    print(f'{jobName}: fitting calibration and blend on {len(metaTrain):,} unseen base predictions from {metaTrain["matchID"].nunique():,} matches', flush=True)

    # --------------------------------------------------
    # LOGISTIC SIMULATION CALIBRATION, FIT INSIDE THIS FOLD
    # --------------------------------------------------
    calibrationFeatures = np.column_stack([
        np.ones(len(metaTrain)),
        logit(np.clip(metaTrain['sim_chaseWin%'].to_numpy(float), 1e-6, 1 - 1e-6)),
        np.maximum(12 - metaTrain['inningBallsRemaining'], 0) / 11,
        np.maximum(12 - metaTrain['inningBallsRemaining'], 0) / 11 * logit(np.clip(metaTrain['sim_chaseWin%'].to_numpy(float), 1e-6, 1 - 1e-6)),
        (metaTrain['inningBallsRemaining'] - 1) / 119,
        (metaTrain['inningBallsRemaining'] - 1) / 119 * logit(np.clip(metaTrain['sim_chaseWin%'].to_numpy(float), 1e-6, 1 - 1e-6)),
    ])
    calibrationPrior = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    calibrationLoss = lambda coefficients: (
        np.dot(metaWeights, np.logaddexp(0, calibrationFeatures @ coefficients) - target * (calibrationFeatures @ coefficients))
        + 0.5 * CALIBRATION_PENALTY * np.sum((coefficients - calibrationPrior) ** 2)
    )
    calibrationGradient = lambda coefficients: (
        calibrationFeatures.T @ (metaWeights * (expit(calibrationFeatures @ coefficients) - target))
        + CALIBRATION_PENALTY * (coefficients - calibrationPrior)
    )
    with threadpool_limits(limits=3):
        calibrationFit = minimize(
            calibrationLoss, calibrationPrior, jac=calibrationGradient,
            method='L-BFGS-B', bounds=[(-5, 5), (0.5, 3), (-2, 2), (-0.4, 0.4), (-2, 2), (-0.4, 0.4)],
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8},
        )
    if not calibrationFit.success or not np.isfinite(calibrationFit.x).all():
        raise RuntimeError(f'Calibration did not converge for {jobName}: {calibrationFit.message}')
    calibrationCoefficients = calibrationFit.x
    calibratedSimulation = expit(calibrationFeatures @ calibrationCoefficients)

    # --------------------------------------------------
    # THE SAME STATE-DEPENDENT CONVEX BLEND, NOW REFITTED
    # --------------------------------------------------
    # State/order gate, with the value of spare wickets and the non-striker
    # reduced near the finish. The direct weight reaches zero on the final
    # ball, where both predictors already share the same ball-outcome model.
    blendFeatures = np.column_stack([
        np.ones(len(metaTrain)),
        np.clip(np.log(metaTrain['runsRequired'] / metaTrain['inningBallsRemaining']), -2, 2),
        (metaTrain['inningBallsRemaining'] - 60) / 60,
        (metaTrain['totalInningWickets'] - 5) / 5 * (metaTrain['inningBallsRemaining'] - 1) / (metaTrain['inningBallsRemaining'] + 5),
        (metaTrain['ord'] - 6) / 5,
        (metaTrain['nonStrikerOrd'] - 6) / 5 * (metaTrain['inningBallsRemaining'] - 1) / (metaTrain['inningBallsRemaining'] + 5),
        np.log(metaTrain['inningBallsRemaining']) / np.log(120) - 0.5,
        np.maximum(6 - metaTrain['inningBallsRemaining'], 0) / 5,
    ])
    blendPrior = np.array([logit(0.25), 0, 0, 0, 0, 0, 0, 0], dtype=float)
    directTaper = np.clip((metaTrain['inningBallsRemaining'].to_numpy(float) - 1) / DIRECT_TAPER_BALLS, 0, 1)
    difference = metaTrain['directWinProbability'].to_numpy(float) - calibratedSimulation
    blendedProbability = lambda coefficients: np.clip(
        calibratedSimulation + directTaper * expit(blendFeatures @ coefficients) * difference,
        1e-7, 1 - 1e-7,
    )
    # Set the shrinkage centre from these training matches only, rather than
    # forcing every fold toward a fixed 25% direct-model weight.
    constantBlendLoss = lambda weight: -np.dot(
        metaWeights,
        target * np.log(np.clip(calibratedSimulation + directTaper * weight * difference, 1e-7, 1 - 1e-7))
        + (1 - target) * np.log1p(-np.clip(calibratedSimulation + directTaper * weight * difference, 1e-7, 1 - 1e-7)),
    )
    constantBlendFit = minimize_scalar(constantBlendLoss, bounds=(0.01, 0.99), method='bounded')
    if not constantBlendFit.success:
        raise RuntimeError(f'Constant blend prior did not converge for {jobName}.')
    blendPrior[0] = logit(constantBlendFit.x)
    blendLoss = lambda coefficients: (
        -np.dot(metaWeights, target * np.log(blendedProbability(coefficients)) + (1 - target) * np.log1p(-blendedProbability(coefficients)))
        + 0.5 * BLEND_PENALTY * np.sum((coefficients - blendPrior) ** 2)
    )
    with threadpool_limits(limits=3):
        blendFit = minimize(
            blendLoss, blendPrior, method='L-BFGS-B',
            bounds=[(-5, 5)] + [(-3, 3)] * 7,
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 40},
        )
    if not blendFit.success or not np.isfinite(blendFit.x).all():
        raise RuntimeError(f'Blend did not converge for {jobName}: {blendFit.message}')
    blendCoefficients = blendFit.x

    applyCalibrationFeatures = np.column_stack([
        np.ones(len(applyData)),
        logit(np.clip(applyData['sim_chaseWin%'].to_numpy(float), 1e-6, 1 - 1e-6)),
        np.maximum(12 - applyData['inningBallsRemaining'], 0) / 11,
        np.maximum(12 - applyData['inningBallsRemaining'], 0) / 11 * logit(np.clip(applyData['sim_chaseWin%'].to_numpy(float), 1e-6, 1 - 1e-6)),
        (applyData['inningBallsRemaining'] - 1) / 119,
        (applyData['inningBallsRemaining'] - 1) / 119 * logit(np.clip(applyData['sim_chaseWin%'].to_numpy(float), 1e-6, 1 - 1e-6)),
    ])
    applyBlendFeatures = np.column_stack([
        np.ones(len(applyData)),
        np.clip(np.log(applyData['runsRequired'] / applyData['inningBallsRemaining']), -2, 2),
        (applyData['inningBallsRemaining'] - 60) / 60,
        (applyData['totalInningWickets'] - 5) / 5 * (applyData['inningBallsRemaining'] - 1) / (applyData['inningBallsRemaining'] + 5),
        (applyData['ord'] - 6) / 5,
        (applyData['nonStrikerOrd'] - 6) / 5 * (applyData['inningBallsRemaining'] - 1) / (applyData['inningBallsRemaining'] + 5),
        np.log(applyData['inningBallsRemaining']) / np.log(120) - 0.5,
        np.maximum(6 - applyData['inningBallsRemaining'], 0) / 5,
    ])
    applyData['sim_chaseWin%Calibrated'] = expit(applyCalibrationFeatures @ calibrationCoefficients)
    applyData['directWeight'] = np.clip((applyData['inningBallsRemaining'] - 1) / DIRECT_TAPER_BALLS, 0, 1) * expit(applyBlendFeatures @ blendCoefficients)
    applyData['simulationWeight'] = 1 - applyData['directWeight']
    applyData['matchWinProbability'] = (
        applyData['simulationWeight'] * applyData['sim_chaseWin%Calibrated']
        + applyData['directWeight'] * applyData['directWinProbability']
    )
    coefficientRows.append({
        'stage': jobName, 'outerFold': outerFold,
        'trainingRows': len(metaTrain), 'trainingMatches': metaTrain['matchID'].nunique(),
        'calibrationIntercept': calibrationCoefficients[0], 'calibrationSlope': calibrationCoefficients[1],
        'calibrationLate': calibrationCoefficients[2], 'calibrationLateSlope': calibrationCoefficients[3],
        'calibrationHorizon': calibrationCoefficients[4], 'calibrationHorizonSlope': calibrationCoefficients[5],
        'blendBallsCentre': 60, 'blendBallsScale': 60, 'blendLogBallsMaximum': 120,
        'blendIntercept': blendCoefficients[0], 'blendLogPressure': blendCoefficients[1],
        'blendBalls': blendCoefficients[2], 'blendWickets': blendCoefficients[3],
        'blendStrikerOrder': blendCoefficients[4], 'blendNonStrikerOrder': blendCoefficients[5],
        'blendLogBalls': blendCoefficients[6], 'blendFinalOver': blendCoefficients[7],
        'directTaperBalls': DIRECT_TAPER_BALLS,
        'trainingDirectWeightPrior': constantBlendFit.x,
        'metaMatchWeightPower': META_MATCH_WEIGHT_POWER,
        'calibrationPenalty': CALIBRATION_PENALTY, 'blendPenalty': BLEND_PENALTY,
    })
    if outerFold >= 0:
        oofFrames.append(applyData)
    else:
        applyData.to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/chaseWinProbabilities.csv', index=False)

oof = pd.concat(oofFrames, ignore_index=True).sort_values(['matchID', 'ID']).reset_index(drop=True)
if len(oof) != len(baseOof) or oof.duplicated(['matchID', 'ID']).any() or not np.isfinite(oof['matchWinProbability']).all():
    raise RuntimeError('The blend does not have exactly one unseen probability per eligible evaluation row.')
oof.to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/chaseWinProbabilities_oof.csv', index=False)
pd.DataFrame(coefficientRows).to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/blendCoefficients.csv', index=False)
runInfo.at[0, 'blendReady'] = 1
runInfo['calibrationRecipe'] = 'full_innings_horizon_and_late_intercept_slope'
runInfo['blendRecipe'] = 'full_innings_adaptive_prior_horizon_gate_taper12'
runInfo['metaMatchWeightPower'] = META_MATCH_WEIGHT_POWER
runInfo['blendOofRows'] = len(oof)
runInfo['blendDroppedResourceRows'] = droppedResourceRows
runInfo['blendProductionStates'] = len(applyData)
runInfo.to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/cv/runInfo.csv', index=False)
runInfo.to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin3/outputs/runInfo.csv', index=False)
print(f'Exported {len(oof):,} unseen blended predictions and {len(applyData):,} production scenarios.')
print('Use matchWinProbability: _oof.csv for evaluation, chaseWinProbabilities.csv for production.')
print(f'Calibration/blend runtime: {(time.perf_counter() - blendStartTime) / 60:.1f} minutes')
