"""Read-only reporting for completed hybridWin4 chronological predictions.

Run after scripts 1-4 complete. This script computes scores, uncertainty,
plots from saved predictions; it fits no model and runs no simulations.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from paths import PROJECT_ROOT


BASE = PROJECT_ROOT / 'men/matchMarket/hybridWin4'
MODEL_COLUMNS = {
    'Direct model': 'directProbability',
    'Raw simulation': 'rawSimulationProbability',
    'Calibrated simulation': 'calibratedSimulationProbability',
    'Selected blend': 'matchWinProbability',
}
N_BOOTSTRAP = 2500


def flag(series):
    if series.dtype == bool:
        return series.to_numpy()
    return series.astype(str).str.lower().isin(['true', '1', '1.0']).to_numpy()


def scoped_samples(predictions):
    for scope, selector in [('weighted_over_sample', 'overSample'),
                            ('one_state_per_match', 'legacySample')]:
        sample = predictions.loc[flag(predictions[selector])].copy()
        sample['_weight'] = sample.evaluationWeight.astype(float) if selector == 'overSample' else 1.0
        if sample.empty or sample._weight.le(0).any() or not np.isfinite(sample._weight).all():
            raise ValueError('Invalid evaluation sample/weights: '+scope)
        yield scope, sample


def losses(target, probability):
    p = np.asarray(probability, float)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Probabilities must be finite and between zero and one')
    p = np.clip(p, 1e-7, 1-1e-7)
    y = np.asarray(target, float)
    return -y*np.log(p)-(1-y)*np.log1p(-p), (p-y)**2


def metrics(frame, probability):
    y = frame.chaseWin.to_numpy(float)
    p = np.asarray(probability, float)
    w = frame._weight.to_numpy(float)
    ll, brier = losses(y, p)
    grouped = pd.DataFrame({'matchID': frame.matchID.to_numpy(), 'weight': w,
                            'weightedLoss': w*ll}).groupby('matchID').sum()
    return dict(states=len(frame), matches=frame.matchID.nunique(), representedDeliveries=float(w.sum()),
                logLoss=float(np.average(ll, weights=w)), brier=float(np.average(brier, weights=w)),
                auc=float(roc_auc_score(y, p, sample_weight=w)) if np.unique(y).size == 2 else np.nan,
                meanProbability=float(np.average(p, weights=w)),
                observedWinRate=float(np.average(y, weights=w)),
                calibrationBias=float(np.average(p-y, weights=w)),
                equalMatchLogLoss=float((grouped.weightedLoss/grouped.weight).mean()))


def common_metrics(predictions):
    records = []
    over_records = []
    weight_records = []
    for scope, sample in scoped_samples(predictions):
        for period, group in list(sample.groupby('period', sort=True))+[('pooled', sample)]:
            for model, column in MODEL_COLUMNS.items():
                if column not in group:
                    continue
                records.append(dict(scope=scope, period=period, model=model,
                                    **metrics(group, group[column])))
                if scope == 'weighted_over_sample':
                    for over, part in group.groupby('overNumber', sort=True):
                        over_records.append(dict(scope=scope, period=period, over=int(over), model=model,
                                                 **metrics(part, part[column])))
            if scope == 'weighted_over_sample':
                for over, part in group.groupby('overNumber', sort=True):
                    weight_records.append(dict(period=period, over=int(over), states=len(part),
                                               matches=part.matchID.nunique(),
                                               representedDeliveries=part._weight.sum(),
                                               meanSimulationWeight=np.average(part.simulationWeight, weights=part._weight),
                                               minimumSimulationWeight=part.simulationWeight.min(),
                                               maximumSimulationWeight=part.simulationWeight.max()))
    return pd.DataFrame(records), pd.DataFrame(over_records), pd.DataFrame(weight_records)


def cluster_bootstrap(frame, first_column, second_column, draws=N_BOOTSTRAP, seed=482917):
    """Paired model2-minus-model1 differences, resampling whole matches.

    Period-stratified for pooled scores, preserving the original number of matches
    in each chronological test window. Frozen predictions/recipes are conditioned
    on; uncertainty from training, recipe selection and MC resimulation is omitted.
    """
    if draws < 2000:
        raise ValueError('Use at least 2000 cluster bootstrap resamples')
    first_ll, first_brier = losses(frame.chaseWin, frame[first_column])
    second_ll, second_brier = losses(frame.chaseWin, frame[second_column])
    weights = frame._weight.to_numpy(float)
    aggregated = frame[['period', 'matchID']].copy()
    aggregated['weight'] = weights
    aggregated['lossNumerator'] = weights*(second_ll-first_ll)
    aggregated['brierNumerator'] = weights*(second_brier-first_brier)
    aggregated = aggregated.groupby(['period', 'matchID'], sort=True).sum().reset_index()
    if frame.groupby('matchID').period.nunique().gt(1).any():
        raise ValueError('A test match appears in more than one period')
    rng = np.random.default_rng(seed)
    denominator = np.zeros(draws)
    numerator = np.zeros((draws, 2))
    equal_match_loss_sum = np.zeros(draws)
    for _, period in aggregated.groupby('period', sort=True):
        values = period[['weight', 'lossNumerator', 'brierNumerator']].to_numpy(float)
        for start in range(0, draws, 100):
            stop = min(start+100, draws)
            selected = rng.integers(0, len(values), size=(stop-start, len(values)))
            sampled = values[selected]
            sums = sampled.sum(axis=1)
            denominator[start:stop] += sums[:, 0]
            numerator[start:stop] += sums[:, 1:]
            equal_match_loss_sum[start:stop] += (sampled[:, :, 1]/sampled[:, :, 0]).sum(axis=1)
    boot = numerator/denominator[:, None]
    boot_equal = equal_match_loss_sum/len(aggregated)
    observed = [np.average(second_ll-first_ll, weights=weights),
                np.average(second_brier-first_brier, weights=weights),
                (aggregated.lossNumerator/aggregated.weight).mean()]
    records = []
    for name, point, values in zip(['logLoss', 'brier', 'equalMatchLogLoss'], observed,
                                   [boot[:, 0], boot[:, 1], boot_equal]):
        lower, upper = np.quantile(values, [.025, .975])
        records.append(dict(metric=name, difference=float(point), lower95=float(lower), upper95=float(upper),
                            bootstrapFractionBelowZero=float(np.mean(values < 0)), draws=draws,
                            clusters=len(aggregated), states=len(frame),
                            interpretation='Second minus first; negative favours second. Frozen-model, period-stratified match bootstrap.'))
    return records


def uncertainty_report(predictions, draws=N_BOOTSTRAP):
    records = []
    for scope, sample in scoped_samples(predictions):
        for period, group in list(sample.groupby('period', sort=True))+[('pooled', sample)]:
            for second, column in [('Selected blend', 'matchWinProbability'),
                                   ('Calibrated simulation', 'calibratedSimulationProbability')]:
                if column not in group:
                    continue
                for row in cluster_bootstrap(group, 'directProbability', column, draws=draws):
                    records.append(dict(scope=scope, period=period, firstModel='Direct model',
                                        secondModel=second, **row))
    return pd.DataFrame(records)


def reliability_report(predictions):
    rows = []
    for scope, sample in scoped_samples(predictions):
        for model, column in MODEL_COLUMNS.items():
            if column not in sample:
                continue
            bins = np.minimum((sample[column].to_numpy(float)*10).astype(int), 9)
            for index in range(10):
                part = sample.loc[bins == index]
                if not len(part):
                    continue
                rows.append(dict(scope=scope, model=model, bin=index, lower=index/10, upper=(index+1)/10,
                                 states=len(part), matches=part.matchID.nunique(),
                                 representedDeliveries=float(part._weight.sum()),
                                 meanProbability=float(np.average(part[column], weights=part._weight)),
                                 observedWinRate=float(np.average(part.chaseWin, weights=part._weight))))
    return pd.DataFrame(rows)


def opening_direct_predictions(base=BASE):
    """Recover all eligible opening-state direct OOF predictions, without inference.

    The period-specific saved ensemble weights/calibrator are applied to saved
    base-model OOF probabilities. This is arithmetic over completed predictions,
    not a refit or a production-model retrospective prediction.
    """
    full_path = base/'outputs/directEvaluationPredictions.csv'
    if full_path.exists():
        full = pd.read_csv(full_path)
        starts = full.loc[full.inningBallsRemaining.eq(120) & full.totalInningWickets.eq(0)].copy()
        if starts.matchID.duplicated().any() or starts.directProbability.isna().any():
            raise ValueError('Invalid full saved opening-state direct predictions')
        starts['_weight'] = 1.
        return starts
    raise FileNotFoundError('Run 4_blendModels.py to generate directEvaluationPredictions.csv first.')


def opening_calibration(frame):
    records = []
    intervals = [(-np.inf, 139, '<=139'), (140, 159, '140-159'), (160, 179, '160-179'),
                 (180, 199, '180-199'), (200, 219, '200-219'), (220, np.inf, '220+'),
                 (170, 190, '170-190 (around 180)'), (180, 180, 'Exactly 180')]
    for period, group in list(frame.groupby('period', sort=True))+[('pooled', frame)]:
        for lower, upper, label in intervals:
            part = group.loc[group.runsRequired.between(lower, upper)]
            if not len(part):
                continue
            n = len(part)
            rate = float(part.chaseWin.mean())
            z = 1.959963984540054
            denominator = 1+z*z/n
            center = (rate+z*z/(2*n))/denominator
            radius = z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/denominator
            records.append(dict(period=period, targetBin=label, model='Direct model',
                                **metrics(part, part.directProbability),
                                observedRateLower95=center-radius, observedRateUpper95=center+radius,
                                intervalNote='Wilson interval on observed win rate; not on the calibration error. Historical ratings, not all-average hypothetical players.'))
    return pd.DataFrame(records)


def figure_report(reliability, by_over, weights, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'Direct model': '#276FBF', 'Calibrated simulation': '#D46B32', 'Selected blend': '#17966F'}
    plt.rcParams.update({'font.size': 11, 'axes.titlesize': 13, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 180})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for model, color in colors.items():
        r = reliability.loc[reliability.scope.eq('weighted_over_sample') & reliability.model.eq(model)]
        axes[0].plot(r.meanProbability, r.observedWinRate, marker='o', markersize=4,
                     linewidth=1.8, label=model, color=color)
        o = by_over.loc[by_over.period.eq('pooled') & by_over.model.eq(model)].sort_values('over')
        axes[1].plot(o.over, o.logLoss, marker='o', markersize=3, linewidth=1.8,
                     label=model, color=color)
    axes[0].plot([0, 1], [0, 1], color='#777777', linestyle='--', linewidth=1)
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean predicted win probability',
                ylabel='Observed win rate', title='Calibration across three test periods')
    axes[0].legend(fontsize=9, loc='upper left')
    axes[1].set(xlim=(1, 20), xticks=[1, 4, 8, 12, 16, 20], xlabel='Over',
                ylabel='Log loss (lower is better)', title='Performance by over')
    w = weights.loc[weights.period.eq('pooled')].sort_values('over')
    axes[2].bar(w.over, w.meanSimulationWeight*100, color='#17966F', width=.78)
    axes[2].set(xlim=(.5, 20.5), ylim=(0, 100), xticks=[1, 4, 8, 12, 16, 20],
                xlabel='Over', ylabel='Mean simulation weight (%)', title='Simulation weights in rolling tests')
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('hybridWin4 — chronological validation', fontsize=17, fontweight='bold')
    fig.savefig(destination, facecolor='white')
    plt.close(fig)


def run_reports(base=BASE, draws=N_BOOTSTRAP):
    predictions = pd.read_csv(base/'outputs/evaluationPredictions.csv')
    if predictions.duplicated(['matchID', 'ID']).any():
        raise ValueError('Duplicate evaluation rows')
    report, by_over, weights = common_metrics(predictions)
    uncertainty = uncertainty_report(predictions, draws=draws)
    reliability = reliability_report(predictions)
    opening_predictions = opening_direct_predictions(base)
    openings = opening_calibration(opening_predictions)
    for frame, name in [(report, 'reportMetrics.csv'), (by_over, 'reportMetricsByOver.csv'),
                        (weights, 'simulationWeightsByOver.csv'), (uncertainty, 'pairedClusterUncertainty.csv'),
                        (reliability, 'reliabilityBins.csv'), (opening_predictions, 'openingDirectPredictions.csv'),
                        (openings, 'openingTargetCalibration.csv')]:
        frame.to_csv(base/'outputs'/name, index=False)
    figure_report(reliability, by_over, weights, base/'outputs/modelDiagnostics.png')
    print(report.loc[report.period.eq('pooled')].to_string(index=False), flush=True)
    print('Reporting complete; no models fitted and no simulations run.', flush=True)
    return report

if __name__ == '__main__':
    run_reports()
