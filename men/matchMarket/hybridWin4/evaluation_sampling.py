"""Outcome-independent per-over samples plus legacy comparison states."""
import pandas as pd
from paths import PROJECT_ROOT
from era_utils import atomic_csv

BASE = PROJECT_ROOT / 'men/matchMarket/hybridWin4'


def prepare_samples():
    data = pd.read_csv(BASE / 'data/training.csv')
    legacy = data.groupby('matchID', group_keys=False).sample(n=1, random_state=1907)[['matchID', 'ID']]
    legacy['legacySample'] = True
    sample = data.groupby(['matchID', 'overNumber'], group_keys=False).sample(n=1, random_state=8719)
    counts = data.groupby(['matchID', 'overNumber']).size().rename('evaluationWeight').reset_index()
    sample = sample.merge(counts, on=['matchID', 'overNumber'], validate='one_to_one')
    sample['overSample'] = True
    union = pd.concat([sample, data.merge(legacy, on=['matchID', 'ID'], validate='one_to_one')], ignore_index=True)
    union = union.drop_duplicates(['matchID', 'ID']).drop(columns='legacySample', errors='ignore')
    union = union.merge(legacy, on=['matchID', 'ID'], how='left', validate='one_to_one')
    union['legacySample'] = union.legacySample.eq(True)
    union['overSample'] = union.overSample.eq(True)
    union['evaluationWeight'] = union.evaluationWeight.fillna(0)
    union = union.sort_values(['matchRank', 'overNumber', 'ID']).reset_index(drop=True)
    atomic_csv(union, BASE / 'data/evaluationStates.csv')
    summary = union.groupby('fold').agg(states=('ID', 'size'), overStates=('overSample', 'sum'),
        matches=('matchID', 'nunique'), legacyStates=('legacySample', 'sum'), representedRows=('evaluationWeight', 'sum')).reset_index()
    atomic_csv(summary, BASE / 'outputs/evaluationSampling.csv')
    print(summary.to_string(index=False), flush=True)
    return union


if __name__ == '__main__':
    prepare_samples()
