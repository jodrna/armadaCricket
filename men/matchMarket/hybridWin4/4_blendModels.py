"""Refit the fixed direct mixture and smooth blend; no recipe selection."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from blend import fit_fixed_pipeline

if __name__ == '__main__':
    fit_fixed_pipeline()
