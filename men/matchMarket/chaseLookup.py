# Men's chase lookup (20 Oct 2026, user: one file per gender; was the shared root chaseLookup.py). Default gender men. Needs
#   modelSweepPipeline.py + catalogueHelper.py + architectureTesting.py at the project root.
# PROCESS: stage 8 - THE LIVE CHASE MODEL, men (user 20 Oct 2026): tapered Start/Main/Death neural nets ->
#   men/matchMarket/outputs/1_chaseLookup_v2.csv, 1_chaseLookupNew_v2.csv, 1_chaseLookupCombined_v2.csv (read by
#   7_distribs). Usage: python men/matchMarket/chaseLookup.py  (old chase scripts are in men/matchMarket/old/)
# See PROCESS.txt at the top of prod_model for the whole run order.
import time
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

from modelSweepPipeline import load_training_data
from paths import PROJECT_ROOT

# Builds the v2 production-style lookup outputs for the finalised Start+Main+Death tapered model
# (2026-10-16). Three files per gender, same grid-building logic as the production scripts
# (1_chaseModelLocal*.py), named like the current production outputs with a _v2 suffix:
#   1_chaseLookup(_w)_v2.csv          - Start+Main blended (taper over ballsRemaining 106-120)
#   1_chaseLookupNew(_w)_v2.csv       - Death alone, new_wkts features, RA_Sum scenario grid
#   1_chaseLookupCombined(_w)_v2.csv  - all three tapered together (Start/Main/Death)
#
# Does not touch any existing production file - all three are new filenames.

OLD_FEATURES = ['runsRequired', 'ratioRequired', 'daysGroup', 'inningBallsRemaining']
NEW_WKTS_FEATURES = ['runsRequiredStd', 'RA_Sum', 'ratioRequiredStd', 'daysGroup', 'inningBallsRemaining', 'totalInningWickets']

CONFIGS = {
    'men': dict(start_arch=(64,), main_arch=(96, 48), death_arch=(64,)),
    'women': dict(start_arch=(16,), main_arch=(48,), death_arch=(16,)),
}

OUTPUT_SUFFIX = {'men': '', 'women': '_w'}

START_LO, START_HI = 106, 120
START_TAPER_BALLS = 14
DEATH_TRAIN_CUTOFF = 36
DEATH_TAPER_LO, DEATH_TAPER_HI = 12, 30

RA_SUM_BUFFER = 2
RA_SUM_STEP_COARSE = 2
RA_SUM_STEP_FINE = 1
RA_SUM_FINE_BALLS_THRESHOLD = 12


def fit_model_and_scaler(data, features, architecture, seed=42):
    sc = StandardScaler()
    Xf = sc.fit_transform(data[features])
    m = MLPClassifier(hidden_layer_sizes=architecture, random_state=seed, activation='logistic', batch_size='auto',
                       learning_rate='constant', max_iter=5000, early_stopping=False, learning_rate_init=0.001)
    m.fit(Xf, data['chaseWin'])
    return m, sc


def predict_with(model, scaler, df, features):
    X = scaler.transform(df[features])
    return model.predict_proba(X)[:, 1]


def build_ra_sum_grid(trainData):
    raSumBounds = trainData[(trainData['inningBallsRemaining'] >= 1) & (trainData['inningBallsRemaining'] <= 30)] \
        .groupby('inningBallsRemaining')['RA_Sum'].quantile([0.01, 0.99]).unstack()
    raSumBounds.columns = ['p01', 'p99']
    raSumRows = []
    for ballsRemaining, bounds in raSumBounds.iterrows():
        lo = np.floor(bounds['p01'] - RA_SUM_BUFFER)
        hi = np.ceil(bounds['p99'] + RA_SUM_BUFFER)
        if ballsRemaining <= RA_SUM_FINE_BALLS_THRESHOLD:
            step = RA_SUM_STEP_FINE
        else:
            step = RA_SUM_STEP_COARSE
            lo = 2 * np.floor(lo / 2)
            hi = 2 * np.ceil(hi / 2)
        for v in np.arange(lo, hi + step / 2, step):
            raSumRows.append({'inningBallsRemaining': ballsRemaining, 'RA_Sum': v})
    return pd.DataFrame(raSumRows)


def build_v2_outputs(gender):
    t0 = time.time()
    cfg = CONFIGS[gender]
    suffix = OUTPUT_SUFFIX[gender]
    outDir = PROJECT_ROOT / f'{gender}/matchMarket/outputs'

    trainData, chaseLookup = load_training_data(gender, return_lookup=True)
    br = trainData['inningBallsRemaining']
    startMask = (br >= START_LO) & (br <= START_HI)
    deathMask = br < DEATH_TRAIN_CUTOFF
    startData = trainData[startMask]
    deathData = trainData[deathMask]

    print(f'[{gender}] fitting Start: old features, arch={cfg["start_arch"]}, n={len(startData)}')
    startModel, startScaler = fit_model_and_scaler(startData, OLD_FEATURES, cfg['start_arch'])
    print(f'[{gender}] fitting Main: old features, arch={cfg["main_arch"]}, n={len(trainData)}')
    mainModel, mainScaler = fit_model_and_scaler(trainData, OLD_FEATURES, cfg['main_arch'])
    print(f'[{gender}] fitting Death: new_wkts features, arch={cfg["death_arch"]}, n={len(deathData)}')
    deathModel, deathScaler = fit_model_and_scaler(deathData, NEW_WKTS_FEATURES, cfg['death_arch'])

    startRawTrain = predict_with(startModel, startScaler, startData, OLD_FEATURES)
    mainRawTrain = predict_with(mainModel, mainScaler, trainData, OLD_FEATURES)
    deathRawTrain = predict_with(deathModel, deathScaler, deathData, NEW_WKTS_FEATURES)

    startIso = IsotonicRegression(out_of_bounds='clip').fit(startRawTrain, startData['chaseWin'])
    mainIso = IsotonicRegression(out_of_bounds='clip').fit(mainRawTrain, trainData['chaseWin'])
    deathIso = IsotonicRegression(out_of_bounds='clip').fit(deathRawTrain, deathData['chaseWin'])

    # ---- File A: Start+Main blended (1_chaseLookup{suffix}_v2.csv) ----
    gridA = chaseLookup.copy()
    gridA['m_chaseWin%Main'] = mainIso.predict(predict_with(mainModel, mainScaler, gridA, OLD_FEATURES))
    startMaskGrid = (gridA['inningBallsRemaining'] >= START_LO) & (gridA['inningBallsRemaining'] <= START_HI)
    gridA['m_chaseWin%Start'] = np.nan
    gridA.loc[startMaskGrid, 'm_chaseWin%Start'] = startIso.predict(
        predict_with(startModel, startScaler, gridA[startMaskGrid], OLD_FEATURES))
    startWeightA = ((gridA['inningBallsRemaining'] - START_LO) / START_TAPER_BALLS).clip(0, 1)
    gridA['m_chaseWin%'] = startWeightA * gridA['m_chaseWin%Start'].fillna(0) + (1 - startWeightA) * gridA['m_chaseWin%Main']

    fileACols = [c for c in chaseLookup.columns] + ['m_chaseWin%Main', 'm_chaseWin%Start', 'm_chaseWin%']
    pathA = outDir / f'1_chaseLookup{suffix}_v2.csv'
    gridA[fileACols].to_csv(pathA, index=False)
    print(f'[{gender}] saved Start+Main blended lookup to {pathA} ({len(gridA)} rows)')

    # ---- File B: Death alone, RA_Sum scenario grid (1_chaseLookupNew{suffix}_v2.csv) ----
    raSumGrid = build_ra_sum_grid(trainData)
    gridB = chaseLookup[(chaseLookup['inningBallsRemaining'] >= 1) & (chaseLookup['inningBallsRemaining'] <= 30)].copy()
    gridB = gridB.merge(raSumGrid, on='inningBallsRemaining', how='left')
    gridB['runsRequiredStd'] = gridB['runsRequired']
    gridB['ratioRequiredStd'] = gridB['ratioRequired']
    gridB['m_chaseWin%_DeathNew'] = deathIso.predict(predict_with(deathModel, deathScaler, gridB, NEW_WKTS_FEATURES))

    lastCols = ['totalInningWickets', 'inningBallsRemaining', 'runsRequired', 'RA_Sum', 'm_chaseWin%_DeathNew']
    gridB = gridB[[c for c in gridB.columns if c not in lastCols] + lastCols]
    pathB = outDir / f'1_chaseLookupNew{suffix}_v2.csv'
    gridB.to_csv(pathB, index=False)
    print(f'[{gender}] saved Death lookup to {pathB} ({len(gridB)} rows)')

    # ---- File C: Combined (1_chaseLookupCombined{suffix}_v2.csv) ----
    # Death's RA_Sum is a real scenario axis chaseLookup's generic states don't have, so the combined
    # grid takes Death's prediction at RA_Sum nearest zero (neutral) per state - same reasoning
    # production uses for why DeathNew isn't blended into chaseLookup today.
    gridBNeutral = gridB.copy()
    gridBNeutral['absRA'] = (gridBNeutral['RA_Sum']).abs()
    neutralIdx = gridBNeutral.groupby(['totalInningWickets', 'inningBallsRemaining', 'runsRequired'])['absRA'].idxmin()
    deathNeutral = gridBNeutral.loc[neutralIdx, ['totalInningWickets', 'inningBallsRemaining', 'runsRequired', 'm_chaseWin%_DeathNew']]

    gridC = gridA[['inningBallNumber', 'inningBallsRemaining', 'totalInningWickets', 'runsRequired', 'm_chaseWin%']].copy()
    gridC = gridC.rename(columns={'m_chaseWin%': 'm_chaseWin%_StartMain'})
    gridC = gridC.merge(deathNeutral, on=['totalInningWickets', 'inningBallsRemaining', 'runsRequired'], how='left')

    deathWeightC = ((DEATH_TAPER_HI - gridC['inningBallsRemaining']) / (DEATH_TAPER_HI - DEATH_TAPER_LO)).clip(0, 1)
    gridC['m_chaseWin%'] = (deathWeightC * gridC['m_chaseWin%_DeathNew'].fillna(0)
                            + (1 - deathWeightC) * gridC['m_chaseWin%_StartMain'])

    pathC = outDir / f'1_chaseLookupCombined{suffix}_v2.csv'
    gridC.to_csv(pathC, index=False)
    print(f'[{gender}] saved Combined (Start+Main+Death tapered) lookup to {pathC} ({len(gridC)} rows)')

    print(f'[{gender}] v2 outputs done, total time={time.time()-t0:.1f}s')
    return {'gender': gender, 'fileA': str(pathA), 'fileB': str(pathB), 'fileC': str(pathC)}


if __name__ == '__main__':
    import sys
    gender = sys.argv[1] if len(sys.argv) > 1 else 'men'
    print(build_v2_outputs(gender))
