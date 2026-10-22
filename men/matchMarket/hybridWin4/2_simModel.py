"""Generate CV simulation probabilities with the fixed ball_recent model."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from fixed_training import STAGES, ball_recipe
from run_simulations import run_plan

NUMBER_OF_SIMS = 1000
WORKERS = 2

if __name__ == '__main__':
    name = ball_recipe()['name']
    run_plan([(name, stage) for stage in STAGES], sims=NUMBER_OF_SIMS, workers=WORKERS, batch_size=32)
