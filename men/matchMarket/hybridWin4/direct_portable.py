"""Portable production inference for the selected direct ensemble components.

Logistic and neural models use plain NumPy arrays, not a pickled sklearn
estimator. Spline coefficients and CatBoost models retain their native formats.
Fitting uses the fixed configuration in directRecipe.csv.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
import direct_models as training
from era_utils import atomic_csv

fit_direct = training.fit_direct


def load_direct(stem):
    stem = Path(stem)
    configuration = pd.read_csv(stem.with_suffix('.csv')).iloc[0].to_dict()
    if configuration['family'] in ['logistic', 'mlp'] and stem.with_suffix('.npz').exists():
        with np.load(stem.with_suffix('.npz'), allow_pickle=False) as saved:
            payload = {key:saved[key].copy() for key in saved.files}
        if str(payload['portableFormat']) != 'dense_binary_v1':
            raise ValueError('Unknown direct-model array format')
        return dict(configuration=configuration, fitted=payload, portable=True)
    return training.load_direct(stem)


def predict_direct(model, frame):
    if not model.get('portable', False):
        return training.predict_direct(model, frame)
    config = model['configuration']
    payload = model['fitted']
    x = training.numeric_features(frame, config, compact=True)
    names = payload['featureNames'].tolist()
    x = x.loc[:, names].to_numpy(dtype=float)
    x = (x-payload['scalerMean'])/payload['scalerScale']
    with threadpool_limits(limits=2):
        if config['family'] == 'logistic':
            return expit(x@payload['coefficient'].T+payload['intercept']).ravel()
        layers = int(payload['layers'])
        for layer in range(layers):
            x = x@payload[f'coefficient_{layer}']+payload[f'intercept_{layer}']
            if layer < layers-1:
                x = np.maximum(x, 0.)
        return expit(x).ravel()


def save_direct(model, stem, validation_frame=None):
    """Write a portable production model and optionally verify probabilities."""
    stem = Path(stem)
    config = dict(model['configuration'])
    family = config['family']
    if family not in ['logistic', 'mlp']:
        training.save_direct(model, stem)
        return
    pipeline = model['fitted']
    scaler = pipeline.steps[0][1]
    estimator = pipeline.steps[1][1]
    if not np.array_equal(estimator.classes_, np.array([0,1])):
        raise ValueError('Portable direct inference requires binary classes 0 and 1')
    payload = dict(portableFormat=np.array('dense_binary_v1'),
                   scalerMean=scaler.mean_, scalerScale=scaler.scale_,
                   featureNames=np.asarray(scaler.feature_names_in_, dtype=str))
    if family == 'logistic':
        payload.update(coefficient=estimator.coef_, intercept=estimator.intercept_)
    else:
        if estimator.activation != 'relu' or estimator.out_activation_ != 'logistic':
            raise ValueError('Unexpected neural activation function')
        payload['layers'] = np.array(len(estimator.coefs_))
        for layer, (coefficient, intercept) in enumerate(zip(estimator.coefs_, estimator.intercepts_)):
            payload[f'coefficient_{layer}'] = coefficient
            payload[f'intercept_{layer}'] = intercept
    temporary = stem.with_suffix('.npz.partial')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(stem.with_suffix('.npz'))
    config['serialization'] = 'numpy_dense_binary_v1'
    config['effectiveDateTreatment'] = 'continuous_clipped_interactions'
    atomic_csv(pd.DataFrame([config]), stem.with_suffix('.csv'))
    if validation_frame is not None:
        expected = training.predict_direct(model, validation_frame)
        actual = predict_direct(load_direct(stem), validation_frame)
        if not np.allclose(actual, expected, rtol=1e-12, atol=1e-12):
            raise AssertionError('Portable inference disagrees with training estimator')
