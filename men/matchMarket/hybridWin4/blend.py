"""Fit or load one fixed four-component direct mixture plus smooth simulation blend."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from paths import PROJECT_ROOT
from era_utils import atomic_csv, stage_data
from fixed_training import STAGES, direct_recipes, ball_recipe
from blend_models import fit_blend, predict_blend, predict_calibrator, save_meta_model, load_meta_model

BASE = PROJECT_ROOT / 'men/matchMarket/hybridWin4'
KEYS = ['matchID', 'ID']


def load_production_bundle(stem=None):
    stem = BASE / 'models/production_meta' if stem is None else Path(stem)
    weights = pd.read_csv(Path(str(stem) + '_directWeights.csv'))
    route = pd.read_csv(Path(str(stem) + '_route.csv')).iloc[0].to_dict()
    result = {'directNames': weights.name.tolist(), 'directWeights': weights.weight.to_numpy(float),
              'ballName': route['ballName'], 'route': route}
    for name in ['directCalibrator', 'simulationCalibrator', 'blend']:
        result[name] = load_meta_model(Path(str(stem) + '_' + name))
    return result


def apply_bundle(bundle, states, direct_predictions, raw_simulation=None):
    if isinstance(direct_predictions, (pd.DataFrame, dict)):
        matrix = pd.DataFrame(direct_predictions)[bundle['directNames']].to_numpy(float)
    else:
        matrix = np.asarray(direct_predictions, dtype=float)
    raw_direct = matrix @ bundle['directWeights']
    direct = predict_calibrator(bundle['directCalibrator'], raw_direct, states)
    if raw_simulation is None:
        raise ValueError('The fixed production blend requires simulation probabilities.')
    raw_simulation = np.asarray(raw_simulation, dtype=float)
    simulation = predict_calibrator(bundle['simulationCalibrator'], raw_simulation, states)
    probability, direct_weight, simulation_weight = predict_blend(bundle['blend'], states, direct, simulation)
    return pd.DataFrame({'rawDirectProbability': raw_direct, 'directProbability': direct,
                         'rawSimulationProbability': raw_simulation,
                         'calibratedSimulationProbability': simulation,
                         'directWeight': direct_weight, 'simulationWeight': simulation_weight,
                         'matchWinProbability': probability}, index=states.index)


def simulation_required(bundle):
    return True


def fit_direct_weights(frame, names, penalty):
    probability = frame[names].to_numpy(float)
    target = frame.chaseWin.to_numpy(float)
    prior = np.full(len(names), 1/len(names))

    def objective(weights):
        p = np.clip(probability @ weights, 1e-7, 1-1e-7)
        loss = np.mean(-target*np.log(p)-(1-target)*np.log1p(-p))
        loss += penalty*np.sum((weights-prior)**2)/2
        gradient = probability.T @ ((p-target)/(p*(1-p)))/len(target)
        return loss, gradient + penalty*(weights-prior)

    fitted = minimize(objective, prior, jac=True, method='SLSQP', bounds=[(0.,1.)]*len(names),
                      constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(len(w))},
                      options={'maxiter':400,'ftol':1e-11})
    if not fitted.success:
        raise RuntimeError('Direct mixture fitting failed: '+str(fitted.message))
    weights = np.maximum(fitted.x, 0.)
    return weights/weights.sum()


def save_bundle(bundle, stem, route):
    pd.DataFrame({'name':bundle['directNames'],'weight':bundle['directWeights']}).to_csv(
        Path(str(stem)+'_directWeights.csv'), index=False)
    for name in ['directCalibrator','simulationCalibrator','blend']:
        save_meta_model(bundle[name], Path(str(stem)+'_'+name))
    atomic_csv(pd.DataFrame([route]), Path(str(stem)+'_route.csv'))


def fit_fixed_pipeline():
    names = [recipe['name'] for recipe in direct_recipes()]
    ball_name = ball_recipe()['name']
    recipe = pd.read_csv(BASE / 'blendRecipe.csv').iloc[0].to_dict()
    training = pd.read_csv(BASE / 'data/training.csv')
    sample = pd.read_csv(BASE / 'data/evaluationStates.csv',
                         usecols=KEYS+['overSample','legacySample','evaluationWeight'])
    full_frames = {}
    sample_frames = {}
    for stage in STAGES:
        _, frame = stage_data(training, stage)
        for name in names:
            prediction = pd.read_csv(BASE / f'cv/{name}_{stage}_predictions.csv', usecols=KEYS+[name])
            frame = frame.merge(prediction, on=KEYS, how='left', validate='one_to_one')
        if frame[names].isna().any().any():
            raise ValueError('Incomplete direct CV predictions: '+stage)
        sampled = frame.merge(sample, on=KEYS, how='inner', validate='one_to_one')
        simulation = pd.read_csv(BASE / f'cv/simulation_{ball_name}_{stage}_1000.csv',
                                 usecols=KEYS+['rawSimulationProbability'])
        sampled = sampled.merge(simulation, on=KEYS, how='left', validate='one_to_one')
        if sampled.rawSimulationProbability.isna().any():
            raise ValueError('Incomplete simulation CV predictions: '+stage)
        over = sampled.loc[sampled.overSample]
        if not np.isclose(over.evaluationWeight.sum(), len(frame)):
            raise ValueError('CV sample does not represent every delivery: '+stage)
        frame['period'] = stage
        sampled['period'] = stage
        full_frames[stage] = frame
        sample_frames[stage] = sampled

    evaluations = []
    direct_evaluations = []
    routes = []
    for evaluation, count in [('p70',2),('p80',3),('p90',4),('production',5)]:
        previous = STAGES[:count]
        fitting_full = pd.concat([full_frames[s] for s in previous], ignore_index=True)
        fitting_sample = pd.concat([sample_frames[s] for s in previous], ignore_index=True)
        fitting_sample = fitting_sample.loc[fitting_sample.overSample].copy()
        weights = fit_direct_weights(fitting_full, names, float(recipe['directWeightPenalty']))
        direct = fitting_sample[names].to_numpy(float) @ weights
        identity = {'configuration':{'name':'identity','kind':'identity'},'coefficients':np.array([])}
        gate_recipe = dict(name=recipe['name'], kind=recipe['kind'], penalty=float(recipe['penalty']))
        gate = fit_blend(fitting_sample, direct, fitting_sample.rawSimulationProbability,
                         gate_recipe, weights=fitting_sample.evaluationWeight)
        bundle = dict(directNames=names, directWeights=weights, ballName=ball_name,
                      directCalibrator=identity, simulationCalibrator=identity, blend=gate)
        _, _, sim_weights = predict_blend(gate, fitting_sample, direct, fitting_sample.rawSimulationProbability)
        route = dict(evaluation=evaluation, refitStages=','.join(previous), fixedRecipe=True,
                     ballName=ball_name, directRecipe='learned_mixture', directCalibration='identity',
                     simulationCalibration='identity', blendRecipe=recipe['name'], blendKind=recipe['kind'],
                     refitMeanSimulationWeight=np.average(sim_weights, weights=fitting_sample.evaluationWeight))
        stem = BASE / ('models/production_meta' if evaluation=='production' else f'cv/meta_{evaluation}')
        save_bundle(bundle, stem, route)
        routes.append(route)
        if evaluation=='production':
            continue
        test = sample_frames[evaluation].copy()
        result = apply_bundle(bundle, test, test[names], test.rawSimulationProbability)
        for column in result:
            test[column] = result[column]
        evaluations.append(test)
        full = full_frames[evaluation].copy()
        full['rawDirectProbability'] = full[names].to_numpy(float) @ weights
        full['directProbability'] = full.rawDirectProbability
        direct_evaluations.append(full)
    atomic_csv(pd.concat(evaluations, ignore_index=True), BASE / 'outputs/evaluationPredictions.csv')
    atomic_csv(pd.concat(direct_evaluations, ignore_index=True), BASE / 'outputs/directEvaluationPredictions.csv')
    atomic_csv(pd.DataFrame(routes), BASE / 'outputs/fixedModelRoutes.csv')
    print('Fixed-model CV predictions and production blend saved. No model or recipe search performed.', flush=True)
