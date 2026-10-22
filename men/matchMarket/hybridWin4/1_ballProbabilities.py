"""Prepare data and fit ball_recent only, for CV and production."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from prepare_data import main
from evaluation_sampling import prepare_samples
from fixed_training import fit_ball_models

if __name__ == '__main__':
    main()
    prepare_samples()
    fit_ball_models()
