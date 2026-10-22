"""Fit/load/predict the single selected ball model: ball_recent."""
import os
import re
import sys
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from catboost import CatBoostClassifier
from features import EVENTS
from era_utils import recency_weights, atomic_csv
from date_models import curve_design

def _boolean(value):
    return str(value).lower() in ['true', '1', '1.0']

def _features(frame, config):
    # Construct once: simulation calls this hundreds of times per batch.
    runs = frame.runsRequired.to_numpy(float)
    balls = frame.inningBallsRemaining.to_numpy(float)
    wickets = frame.totalInningWickets.to_numpy(float)
    order = frame.ord.to_numpy(float)
    pressure = runs/balls
    columns = dict(runsRequired=runs, inningBallsRemaining=balls,
                   totalInningWickets=wickets, ord=order, pressure=pressure,
                   logPressure=np.log(np.maximum(pressure, 1e-6)), logBalls=np.log(balls),
                   ballsInOver=(balls-1)%6+1, powerplay=(balls>=85).astype(int),
                   death=(balls<=24).astype(int), orderGap=wickets+2-order,
                   tailOrder=np.maximum(order-7, 0), ballsWickets=balls*(10-wickets))
    if _boolean(config.get('drop_order', False)):
        for key in ['ord', 'orderGap', 'tailOrder']:
            del columns[key]
    for col in ['strikerRun', 'strikerWkt', 'bowlerRun', 'bowlerWkt']:
        rating = frame[col].to_numpy(float)
        columns[col] = np.log(np.maximum(rating, 1e-4))
        columns[col+'Missing'] = np.isnan(rating).astype(int)
    columns['bowlertype_3'] = frame.bowlertype_3.fillna('unknown').astype(str).to_numpy()
    columns['runMatchup'] = columns['strikerRun']+columns['bowlerRun']
    columns['wktMatchup'] = columns['strikerWkt']+columns['bowlerWkt']
    if _boolean(config.get('partner_context', False)):
        columns['nonStrikerOrd'] = frame.nonStrikerOrd.to_numpy(float)
        columns['nonStrikerTail'] = np.maximum(columns['nonStrikerOrd']-7., 0.)
        for col in ['nonStrikerRun', 'nonStrikerWkt']:
            rating = frame[col].to_numpy(float)
            columns[col] = np.log(np.maximum(rating, 1e-4))
            columns[col+'Missing'] = np.isnan(rating).astype(int)
        columns['partnerRunGap'] = columns['strikerRun']-columns['nonStrikerRun']
        columns['partnerWktGap'] = columns['strikerWkt']-columns['nonStrikerWkt']
    return pd.DataFrame(columns, index=frame.index)

def legal_frame(frame):
    d = frame.loc[frame.batsmanRuns.isin(range(7)) & frame.isWicket.isin([0, 1])
                  & ~(frame.batsmanRuns.eq(6) & frame.isWicket.eq(1))].copy()
    mapping = {tuple(pair): i for i, pair in enumerate(EVENTS)}
    d['jointTarget'] = [mapping[(int(r), int(w))] for r, w in zip(d.batsmanRuns, d.isWicket)]
    return d

def _estimator(config, classes=None):
    args = dict(iterations=int(config['iterations']), depth=int(config['depth']),
                learning_rate=float(config['learning_rate']), l2_leaf_reg=float(config['l2']),
                loss_function='Logloss' if classes is None else 'MultiClass',
                random_seed=42, thread_count=4, verbose=False, allow_writing_files=False,
                one_hot_max_size=32)
    if classes is not None:
        args['classes_count'] = classes
    return CatBoostClassifier(**args)

def raw_predict(model, frame):
    return model['fitted'].predict_proba(_features(frame, model['configuration']), thread_count=1)

def _fit_curve(train, model, weights):
    config = model['configuration']
    config['timeCenter'] = float(np.average(train.eraYear, weights=weights))
    design = curve_design(train, config, fitting=True, weights=weights)
    log_base = np.log(np.maximum(raw_predict(model, train), 1e-12))
    targets = train.jointTarget.to_numpy(int)
    actual = EVENTS[targets]
    penalty = float(config['curve_penalty'])
    def objective(coefficients):
        offsets = design @ coefficients.reshape(design.shape[1], 2)
        logits = log_base + offsets @ EVENTS.T
        norms = logsumexp(logits, axis=1)
        probs = np.exp(logits-norms[:, None])
        loss = np.mean(weights*(norms-logits[np.arange(len(train)), targets]))
        gradient = design.T @ (weights[:, None]*(probs @ EVENTS-actual))/len(train)
        return loss+penalty*np.dot(coefficients, coefficients)/2, gradient.ravel()+penalty*coefficients
    fitted = minimize(objective, np.zeros(design.shape[1]*2), jac=True, method='L-BFGS-B',
                      options={'maxiter':250, 'ftol':1e-10, 'gtol':1e-7})
    if not fitted.success:
        raise RuntimeError('Ball era curve failed: '+str(fitted.message))
    config['curveCount'] = len(fitted.x)
    config.update({f'curveCoefficient_{i}': float(v) for i, v in enumerate(fitted.x)})

def _training_log(message):
    # Unobserved joint outcomes are expected; preserve every other diagnostic.
    message = re.sub(
        r'Found only \d+ unique classes in the data, but have defined \d+ classes\. '
        r'Probably something is wrong with data\.\r?\n?', '', message)
    if message:
        sys.stderr.write(message)


def fit_ball(train, configuration):
    train = legal_frame(train) if 'jointTarget' not in train else train
    config = dict(configuration)
    config.update(minimumYear=float(train.eraYear.min()), maximumYear=float(train.eraYear.max()),
                  minimumDate=str(train.date.min()), maximumDate=str(train.date.max()),
                  trainingRows=len(train), trainingMatches=train.matchID.nunique())
    weights = recency_weights(train, float(config['half_life']))
    fitted = _estimator(config, len(EVENTS))
    fitted.fit(_features(train, config), train.jointTarget, sample_weight=weights, cat_features=['bowlertype_3'],
               log_cout=_training_log, log_cerr=_training_log)
    model = dict(configuration=config, fitted=fitted)
    _fit_curve(train, model, weights)
    return model

def predict_ball(model, frame):
    probability = raw_predict(model, frame)
    config = model['configuration']
    if config['era_mode'] != 'none':
        coefficients = np.array([float(config[f'curveCoefficient_{i}']) for i in range(int(config['curveCount']))]).reshape(-1, 2)
        logits = np.log(np.maximum(probability, 1e-12)) + (curve_design(frame, config) @ coefficients) @ EVENTS.T
        probability = np.exp(logits-logsumexp(logits, axis=1)[:, None])
    return probability

def save_ball(model, stem):
    temporary = stem.with_suffix('.cbm.partial')
    model['fitted'].save_model(str(temporary), format='cbm')
    temporary.replace(stem.with_suffix('.cbm'))
    atomic_csv(pd.DataFrame([model['configuration']]), stem.with_suffix('.csv'))


def load_ball(stem):
    config = pd.read_csv(stem.with_suffix('.csv')).iloc[0].to_dict()
    fitted = CatBoostClassifier()
    fitted.load_model(str(stem.with_suffix('.cbm')))
    return dict(configuration=config, fitted=fitted)
