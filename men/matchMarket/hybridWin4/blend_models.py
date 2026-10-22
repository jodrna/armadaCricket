"""The selected identity calibrations and regularised smooth probability blend."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar
from scipy.special import expit
from scipy.special import logit

EPSILON = 1e-7

def _probability(value):
    result = np.asarray(value, dtype=float).reshape(-1)
    if not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError('Probabilities must be finite and between zero and one.')
    return result

def _weights(weights, count):
    result = np.ones(count) if weights is None else np.asarray(weights, dtype=float)
    if result.shape != (count,) or not np.isfinite(result).all():
        raise ValueError('Sample weights must be a finite vector matching the rows.')
    if np.any(result < 0) or result.sum() <= 0:
        raise ValueError('Sample weights must be nonnegative with positive total.')
    return result / result.sum()

def _target(values, count):
    result = np.asarray(values, dtype=float).reshape(-1)
    if len(result) != count or not np.isin(result, [0, 1]).all():
        raise ValueError('Binary targets must match the number of predictions.')
    return result

def weighted_log_loss(y, probability, weights=None):
    p = np.clip(_probability(probability), EPSILON, 1 - EPSILON)
    target = _target(y, len(p))
    weight = _weights(weights, len(p))
    return float(np.dot(weight, -target * np.log(p) - (1 - target) * np.log1p(-p)))

def predict_calibrator(model, probability, states):
    if model['configuration']['kind'] != 'identity':
        raise ValueError('This fixed recipe uses identity calibration only.')
    return _probability(probability).copy()


def _gate_design(frame, direct, simulation, kind='smooth'):
    balls = np.clip(frame.inningBallsRemaining.to_numpy(float), 1, 120)
    h = np.log(balls) / np.log(120) - .5
    wickets = (frame.totalInningWickets.to_numpy(float) - 4.5) / 4.5
    pressure = np.clip(np.log(frame.runsRequired.to_numpy(float) / balls), -2, 2)
    late = np.maximum(24 - balls, 0) / 23
    return np.column_stack([np.ones(len(frame)), h, h*h, wickets, pressure, late])


def fit_blend(frame, direct_probability, simulation_probability, recipe, weights=None):
    configuration = dict(recipe)
    direct = _probability(direct_probability)
    simulation = _probability(simulation_probability)
    target = _target(frame.chaseWin, len(frame))
    sample_weight = _weights(weights, len(frame))

    def constant_objective(weight):
        probability = np.clip(simulation + weight*(direct-simulation), EPSILON, 1-EPSILON)
        return float(np.dot(sample_weight, -target*np.log(probability)-(1-target)*np.log1p(-probability)))

    fitted_constant = minimize_scalar(constant_objective, bounds=(0., 1.), method='bounded', options={'xatol':1e-10})
    constant = min([1., 0., float(fitted_constant.x)], key=constant_objective)
    design = _gate_design(frame, direct, simulation)
    prior = np.zeros(design.shape[1])
    prior[0] = float(logit(np.clip(constant, .02, .98)))
    penalty = float(configuration['penalty'])
    penalty_scale = np.ones(len(prior))
    penalty_scale[0] = .1

    def objective(coefficient):
        weight = expit(design @ coefficient)
        probability = np.clip(simulation + weight*(direct-simulation), EPSILON, 1-EPSILON)
        delta = coefficient-prior
        value = np.dot(sample_weight, -target*np.log(probability)-(1-target)*np.log1p(-probability))
        value += .5*penalty*np.dot(penalty_scale*delta, delta)
        derivative = sample_weight*(probability-target)/(probability*(1-probability))
        derivative *= (direct-simulation)*weight*(1-weight)
        return float(value), design.T@derivative + penalty*penalty_scale*delta

    fitted = minimize(objective, prior, jac=True, method='L-BFGS-B', bounds=[(-7.,7.)]*len(prior),
                      options={'maxiter':600,'ftol':1e-12,'gtol':1e-8})
    if not fitted.success:
        raise RuntimeError('Blend fitting failed: '+str(fitted.message))
    configuration.update(trainingRows=len(frame), converged=True, constantReference=constant)
    return {'configuration':configuration, 'coefficients':fitted.x}


def predict_blend(model, frame, direct_probability, simulation_probability):
    direct = _probability(direct_probability)
    simulation = _probability(simulation_probability)
    direct_weight = expit(_gate_design(frame, direct, simulation) @ model['coefficients'])
    simulation_weight = 1-direct_weight
    probability = direct_weight*direct + simulation_weight*simulation
    return probability, direct_weight, simulation_weight

def save_meta_model(model, stem):
    """Save a calibrator or blend as portable CSV metadata plus numeric NPZ."""
    stem = Path(stem)
    records = []
    for key, value in model['configuration'].items():
        if isinstance(value, (bool, np.bool_)):
            kind = 'bool'
        elif isinstance(value, (int, np.integer)):
            kind = 'int'
        elif isinstance(value, (float, np.floating)):
            kind = 'float'
        elif isinstance(value, str):
            kind = 'str'
        else:
            raise TypeError('Configuration values must be scalar: ' + key)
        records.append({'key': key, 'type': kind, 'value': str(value)})
    pd.DataFrame(records).to_csv(stem.with_suffix('.csv'), index=False)
    np.savez(stem.with_suffix('.npz'), coefficients=np.asarray(model['coefficients'], dtype=float))

def load_meta_model(stem):
    stem = Path(stem)
    configuration = {}
    conversion = {'int': int, 'float': float, 'str': str,
                  'bool': lambda value: value.lower() == 'true'}
    for row in pd.read_csv(stem.with_suffix('.csv'), dtype=str, keep_default_na=False).itertuples(index=False):
        configuration[row.key] = conversion[row.type](row.value)
    with np.load(stem.with_suffix('.npz'), allow_pickle=False) as saved:
        coefficients = saved['coefficients'].copy()
    return {'configuration': configuration, 'coefficients': coefficients}

