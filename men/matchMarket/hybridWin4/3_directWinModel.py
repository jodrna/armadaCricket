"""Fit only the four selected direct components, for CV and production."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from fixed_training import fit_direct_models

if __name__ == '__main__':
    fit_direct_models()
