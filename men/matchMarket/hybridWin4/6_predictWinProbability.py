from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import time
import pandas as pd
from paths import PROJECT_ROOT
from prediction import load_models, predict

script_start = time.perf_counter()

inputs = pd.DataFrame([{
    'matchDate': '2026-10-22',
    'runsRequired': 37,
    'inningBallsRemaining': 18,
    'totalInningWickets': 4,
    'ord': 4,
    'nonStrikerOrd': 6,
    'striker_run_rating_3': 1.0,
    'striker_wkt_rating_3': 1.0,
    'nonstriker_run_rating_3': 1.0,
    'nonstriker_wkt_rating_3': 1.0,
    'bowler_run_rating_3': 1.0,
    'bowler_wkt_rating_3': 1.0,
    'bowlertype_3': 'right_seam_fast',
}])

NUMBER_OF_SIMS = 10000
RANDOM_SEED = 42
INCLUDE_SIMULATION_DIAGNOSTICS = True

load_start = time.perf_counter()
models = load_models()
load_seconds = time.perf_counter() - load_start

predict_start = time.perf_counter()
results = predict(
    inputs,
    models=models,
    number_of_sims=NUMBER_OF_SIMS,
    seed=RANDOM_SEED,
    include_simulation=INCLUDE_SIMULATION_DIAGNOSTICS,
)
predict_seconds = time.perf_counter() - predict_start

for _, row in results.iterrows():
    price = 1 / row['matchWinProbability'] if row['matchWinProbability'] > 0 else float('inf')
    print(f'Direct probability:         {row["directProbability"]:.2%}')
    print(f'Calibrated sim probability: {row["calibratedSimulationProbability"]:.2%}')
    print(f'Direct weight:              {row["directWeight"]:.2%}')
    print(f'Sim weight:                 {row["simulationWeight"]:.2%}')
    print(f'Final probability:          {row["matchWinProbability"]:.2%}')
    print(f'Chasing team price:         {price:.4f}')

# results.to_csv(PROJECT_ROOT / 'men/matchMarket/hybridWin4/outputs/livePredictions.csv', index=False)

total_seconds = time.perf_counter() - script_start
print(f'Model loading time:         {load_seconds:.3f} seconds')
print(f'Prediction time:            {predict_seconds:.3f} seconds')
print(f'Total run time:             {total_seconds:.3f} seconds')
