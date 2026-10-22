"""Resumable simulation batches with explicit source/model fingerprints."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import hashlib
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from paths import PROJECT_ROOT
from era_utils import stage_data, atomic_csv
from ball_models import load_ball, predict_ball
from simulation import simulate

BASE = PROJECT_ROOT / 'men/matchMarket/hybridWin4'
CACHE = {}


def fingerprint(name, stage, sims, batch_size, sample_kind):
    digest = hashlib.sha256()
    files = [BASE / name for name in ['data/training.csv', 'data/evaluationStates.csv',
        'simulation.py', 'ball_models.py', 'features.py', 'date_models.py', 'run_simulations.py',
        'auxiliaries/resourceExtras.csv']]
    files += sorted((BASE / 'cv').glob(f'{name}_{stage}*'))
    for file in files:
        if file.is_file():
            digest.update(file.name.encode())
            with file.open('rb') as handle:
                for block in iter(lambda: handle.read(1048576), b''):
                    digest.update(block)
    digest.update(f'{sims},{batch_size},{sample_kind},42'.encode())
    return digest.hexdigest()[:20]


def run_batch(job):
    name, stage, sims, states, start, sig, sample_kind = job
    output = BASE / f'cv/simpart_{name}_{stage}_{sims}_{sample_kind}_{sig}_{start}.csv'
    keys = states[['matchID', 'ID']].reset_index(drop=True)
    if output.exists():
        saved = pd.read_csv(output)
        if saved[['matchID', 'ID']].equals(keys) and saved.rawSimulationProbability.between(0, 1).all():
            return str(output)
    if (name, stage) not in CACHE:
        model = load_ball(BASE / f'cv/{name}_{stage}')
        priors = pd.read_csv(BASE / f'cv/{name}_{stage}_priors.csv')
        extras = pd.read_csv(BASE / 'auxiliaries/resourceExtras.csv')
        CACHE[(name, stage)] = model, priors, extras
    model, priors, extras = CACHE[(name, stage)]
    p = simulate(model, states, priors, extras, n_sims=sims, seed=42+start,
                 predictor=predict_ball)
    result = keys.copy()
    result['rawSimulationProbability'] = p
    result['simProbability'] = p
    atomic_csv(result, output)
    return str(output)


def run_plan(plan, sims=1000, workers=2, batch_size=32, sample_kind='all'):
    sample = pd.read_csv(BASE / 'data/evaluationStates.csv')
    jobs = []
    expected = {}
    signatures = {}
    suffix = '' if sample_kind == 'all' else '_'+sample_kind
    for name, stage in plan:
        _, states = stage_data(sample, stage)
        states = states.reset_index(drop=True)
        sig = fingerprint(name, stage, sims, batch_size, sample_kind)
        final = BASE / f'cv/simulation_{name}_{stage}_{sims}{suffix}.csv'
        marker = final.with_suffix('.signature.sha256')
        if final.exists() and marker.exists() and marker.read_text() == sig:
            print('Already complete', name, stage, sims, sample_kind, flush=True)
            continue
        expected[(name, stage)] = states
        signatures[(name, stage)] = sig
        jobs += [(name, stage, sims, states.iloc[start:start+batch_size].copy(), start, sig, sample_kind)
            for start in range(0, len(states), batch_size)]
    parts = {}
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = {pool.submit(run_batch, job): (job[0], job[1]) for job in jobs}
        for count, future in enumerate(as_completed(futures), 1):
            key = futures[future]
            parts.setdefault(key, []).append(future.result())
            if count % 10 == 0 or count == len(futures):
                print(f'Simulations {count}/{len(futures)} batches; elapsed {time.perf_counter()-started:.1f}s', flush=True)
    for (name, stage), paths in parts.items():
        frames = [pd.read_csv(path) for path in paths]
        result = expected[(name, stage)][['matchID', 'ID']].merge(pd.concat(frames), on=['matchID', 'ID'], validate='one_to_one')
        if result.rawSimulationProbability.isna().any():
            raise ValueError('Missing simulation states')
        final = BASE / f'cv/simulation_{name}_{stage}_{sims}{suffix}.csv'
        atomic_csv(result, final)
        final.with_suffix('.signature.sha256').write_text(signatures[(name, stage)])
        print('Completed', name, stage, len(result), flush=True)
