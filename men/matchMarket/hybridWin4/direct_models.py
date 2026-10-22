"""The four fixed direct-model components of the selected ensemble."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import date_models
from features import features as state_features
from era_utils import atomic_csv, recency_weights


def _true(value):
    return str(value).lower() in ['true', '1', '1.0']


def numeric_features(frame, configuration, compact=False):
    x = state_features(frame, ability=True, ball=False)
    if _true(configuration.get('monotone', False)):
        x = x.drop(columns=['pressure', 'ballsInOver', 'powerplay', 'death',
                            'orderGap', 'tailOrder', 'ballsWickets',
                            'batRunDifference', 'batWktDifference',
                            'runMatchup', 'wktMatchup'], errors='ignore')
    if _true(configuration.get('no_order', False)):
        x = x.drop(columns=['ord', 'nonStrikerOrd', 'orderGap', 'tailOrder'], errors='ignore')
    if configuration['family'] != 'cat':
        x = x.drop(columns='bowlertype_3')
    if compact:
        x['logRuns'] = np.log(frame.runsRequired.clip(lower=1))
        x['wicketFraction'] = frame.totalInningWickets / 9
        x['requiredRateSquared'] = np.square(np.clip(frame.runsRequired / frame.inningBallsRemaining, 0, 8))
        x['logRunBallInteraction'] = x.logRuns * x.logBalls
        x['wicketLogBalls'] = frame.totalInningWickets * x.logBalls
        x['resourceZSquared'] = np.square(x.resourceZ.clip(-5, 5))
        x['resourceZLogBalls'] = x.resourceZ.clip(-5, 5) * x.logBalls
        x['strikerRunHorizon'] = x.strikerRun * x.logBalls
        x['strikerWktHorizon'] = x.strikerWkt * x.logBalls
        x['nonStrikerRunHorizon'] = x.nonStrikerRun * x.logBalls
        x['nonStrikerWktHorizon'] = x.nonStrikerWkt * x.logBalls
        x['date'] = (np.clip(frame.eraYear, configuration['minimumYear'], configuration['maximumYear']) - 2015.) / 10.
        x['dateHorizon'] = x.date * x.logBalls
        x['datePressure'] = x.date * x.logPressure
    if configuration['family'] in ['logistic', 'mlp']:
        x = x.replace([np.inf, -np.inf], np.nan).fillna(0.)
    return x


def fit_direct(train, configuration):
    configuration = dict(configuration)
    family = configuration['family']
    if family == 'spline':
        with threadpool_limits(limits=1):
            return date_models.fit_model(train, configuration)
    configuration.update(minimumYear=float(train.eraYear.min()), maximumYear=float(train.eraYear.max()),
                         minimumDate=str(train.date.min()), maximumDate=str(train.date.max()),
                         trainingRows=len(train), trainingMatches=train.matchID.nunique())
    x = numeric_features(train, configuration, compact=family in ['logistic', 'mlp'])
    weights = recency_weights(train, float(configuration['half_life']))
    y = train.chaseWin.to_numpy(int)
    if family == 'cat':
        options = dict(iterations=int(configuration['iterations']), depth=int(configuration['depth']),
                       learning_rate=float(configuration['learning_rate']), l2_leaf_reg=float(configuration['l2']),
                       loss_function='Logloss', random_seed=42, thread_count=2, verbose=False,
                       allow_writing_files=False)
        if _true(configuration['monotone']):
            directions = {'runsRequired':-1, 'inningBallsRemaining':1, 'totalInningWickets':-1,
                          'logPressure':-1, 'logBalls':1, 'resourceZ':-1, 'strikerRun':1,
                          'strikerWkt':-1, 'nonStrikerRun':1, 'nonStrikerWkt':-1,
                          'bowlerRun':1, 'bowlerWkt':-1}
            options['monotone_constraints'] = {k:v for k,v in directions.items() if k in x}
        fitted = CatBoostClassifier(**options)
        fitted.fit(x, y, cat_features=['bowlertype_3'], sample_weight=weights)
    elif family == 'logistic':
        fitted = make_pipeline(StandardScaler(), LogisticRegression(C=1./float(configuration['l2']),
                               max_iter=int(configuration['iterations']), solver='lbfgs', tol=1e-7))
        with threadpool_limits(limits=2):
            fitted.fit(x, y, logisticregression__sample_weight=weights)
    elif family == 'mlp':
        # Fixed-epoch training: no randomly split row-level early-stopping set.
        fitted = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,16),
                    activation='relu', alpha=float(configuration['alpha']), batch_size=2048,
                    learning_rate_init=.001, max_iter=int(configuration['iterations']),
                    early_stopping=False, n_iter_no_change=int(configuration['iterations'])+1,
                    random_state=42, tol=1e-8))
        with threadpool_limits(limits=2):
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter('always')
                fitted.fit(x, y)
            configuration['fitWarning'] = ' | '.join(str(w.message) for w in observed)
    else:
        raise ValueError('Unknown direct family: '+family)
    return dict(configuration=configuration, fitted=fitted)


def predict_direct(model, frame):
    config = model['configuration']
    if config['family'] == 'spline':
        with threadpool_limits(limits=1):
            return date_models.predict_model(model, frame)
    x = numeric_features(frame, config, compact=config['family'] in ['logistic', 'mlp'])
    if config['family'] == 'cat':
        return model['fitted'].predict_proba(x, thread_count=2)[:,1]
    with threadpool_limits(limits=2):
        return model['fitted'].predict_proba(x)[:,1]


def save_direct(model, stem):
    configuration = model['configuration']
    if configuration['family'] == 'spline':
        temporary = stem.with_suffix('.npz.partial')
        with temporary.open('wb') as stream:
            np.savez(stream, coefficients=model['fitted'])
        temporary.replace(stem.with_suffix('.npz'))
    else:
        temporary = stem.with_suffix('.cbm.partial')
        model['fitted'].save_model(str(temporary), format='cbm')
        temporary.replace(stem.with_suffix('.cbm'))
    atomic_csv(pd.DataFrame([configuration]), stem.with_suffix('.csv'))

def load_direct(stem):
    configuration = pd.read_csv(stem.with_suffix('.csv')).iloc[0].to_dict()
    if configuration['family'] == 'spline':
        with np.load(stem.with_suffix('.npz')) as saved:
            fitted = saved['coefficients'].copy()
    else:
        fitted = CatBoostClassifier()
        fitted.load_model(str(stem.with_suffix('.cbm')))
    return {'configuration': configuration, 'fitted': fitted}

