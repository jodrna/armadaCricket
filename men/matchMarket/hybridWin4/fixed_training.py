"""Fit the saved best recipe only; no alternatives or parameter searches."""
import hashlib
import time
import numpy as np
import pandas as pd
from paths import PROJECT_ROOT
from era_utils import atomic_csv, stage_data
from ball_models import fit_ball, predict_ball, save_ball, legal_frame, EVENTS
from direct_portable import fit_direct, predict_direct, save_direct
from simulation import build_priors

BASE = PROJECT_ROOT / 'men/matchMarket/hybridWin4'
STAGES = ('p50', 'p60', 'p70', 'p80', 'p90')


def ball_recipe():
    return pd.read_csv(BASE / 'ballRecipe.csv').iloc[0].to_dict()


def direct_recipes():
    return pd.read_csv(BASE / 'directRecipe.csv').to_dict('records')


def source_signature(recipe_file):
    digest = hashlib.sha256()
    files = [BASE / 'data/training.csv', BASE / recipe_file]
    files += [BASE / name for name in ['fixed_training.py', 'features.py', 'date_models.py',
              'ball_models.py', 'direct_models.py', 'direct_portable.py', 'era_utils.py', 'simulation.py']]
    for path in files:
        digest.update(path.name.encode())
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1048576), b''):
                digest.update(block)
    return digest.hexdigest()


def fit_ball_models():
    data = pd.read_csv(BASE / 'data/training.csv')
    configuration = ball_recipe()
    name = configuration['name']
    signature = source_signature('ballRecipe.csv')
    for stage in (*STAGES, 'production'):
        train, valid = stage_data(data, stage)
        train = legal_frame(train)
        valid = legal_frame(valid)
        stem = BASE / ('models/' + name if stage == 'production' else f'cv/{name}_{stage}')
        marker = BASE / f'cv/{name}_{stage}_fit.csv'
        priors = stem.parent / (stem.name + '_priors.csv')
        if (marker.exists() and stem.with_suffix('.cbm').exists() and stem.with_suffix('.csv').exists()
                and priors.exists() and pd.read_csv(marker).iloc[0]['signature'] == signature):
            print('RESUME', name, stage, flush=True)
            continue
        print('FIT', name, stage, 'rows', len(train), flush=True)
        started = time.perf_counter()
        model = fit_ball(train, configuration)
        save_ball(model, stem)
        atomic_csv(build_priors(train), priors)
        record = dict(name=name, stage=stage, signature=signature, trainingRows=len(train),
                      trainingMatches=train.matchID.nunique(), trainingMaximumDate=train.date.max())
        if len(valid):
            probability = predict_ball(model, valid)
            target = valid.jointTarget.to_numpy(int)
            loss = -np.log(np.maximum(probability[np.arange(len(valid)), target], 1e-12))
            record['logLoss'] = loss.mean()
            categories = []
            for index, (runs, wicket) in enumerate(EVENTS):
                selected = target == index
                categories.append(dict(runs=runs, wicket=wicket, count=int(selected.sum()),
                                       observedRate=selected.mean(), meanProbability=probability[:, index].mean(),
                                       conditionalLogLoss=loss[selected].mean() if selected.any() else np.nan))
            atomic_csv(pd.DataFrame(categories), BASE / f'cv/{name}_{stage}_categories.csv')
        record['seconds'] = time.perf_counter() - started
        atomic_csv(pd.DataFrame([record]), marker)
        print('DONE', name, stage, 'seconds', round(record['seconds'], 2),
              'ball log loss', record.get('logLoss', 'production fit'), flush=True)


def fit_direct_models():
    data = pd.read_csv(BASE / 'data/training.csv')
    signature = source_signature('directRecipe.csv')
    for configuration in direct_recipes():
        name = configuration['name']
        for stage in (*STAGES, 'production'):
            train, valid = stage_data(data, stage)
            stem = BASE / ('models/' + name if stage == 'production' else f'cv/{name}_{stage}')
            marker = BASE / f'cv/{name}_{stage}_fit.csv'
            predictions = BASE / f'cv/{name}_{stage}_predictions.csv'
            suffix = '.cbm' if configuration['family'] == 'cat' else '.npz'
            if (marker.exists() and stem.with_suffix(suffix).exists() and stem.with_suffix('.csv').exists()
                    and (stage == 'production' or predictions.exists())
                    and pd.read_csv(marker).iloc[0]['signature'] == signature):
                print('RESUME', name, stage, flush=True)
                continue
            print('FIT', name, stage, 'rows', len(train), flush=True)
            started = time.perf_counter()
            model = fit_direct(train, configuration)
            save_direct(model, stem)
            record = dict(name=name, stage=stage, signature=signature, trainingRows=len(train),
                          trainingMatches=train.matchID.nunique(), trainingMaximumDate=train.date.max())
            if len(valid):
                probability = predict_direct(model, valid)
                result = valid[['matchID', 'ID']].copy()
                result[name] = probability
                atomic_csv(result, predictions)
                p = np.clip(probability, 1e-7, 1-1e-7)
                y = valid.chaseWin.to_numpy(float)
                record['logLoss'] = np.mean(-y*np.log(p)-(1-y)*np.log1p(-p))
            record['seconds'] = time.perf_counter() - started
            atomic_csv(pd.DataFrame([record]), marker)
            print('DONE', name, stage, 'seconds', round(record['seconds'], 2),
                  'win log loss', record.get('logLoss', 'production fit'), flush=True)
