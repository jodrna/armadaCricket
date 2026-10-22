"""Whole-match expanding windows and continuous date coordinates."""
import numpy as np
import pandas as pd


def decimal_year(dates):
    dates = pd.to_datetime(dates, errors='raise', utc=True)
    return 2015 + np.asarray((dates-pd.Timestamp('2015-01-01', tz='UTC')).dt.total_seconds())/(365.25*86400)


def recency_weights(frame, half_life=0, reference_year=None):
    years = frame.eraYear.to_numpy(float)
    if not half_life or float(half_life) <= 0:
        return np.ones(len(frame))
    reference = years.max() if reference_year is None else reference_year
    weight = np.exp2(-(reference-years)/float(half_life))
    return weight/weight.mean()


def atomic_csv(frame, path):
    temporary = path.with_suffix(path.suffix+'.partial')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def assign_splits(frame):
    matches = frame.groupby('matchID').agg(date=('date','min'), chaseWin=('chaseWin','first')).reset_index()
    matches = matches.sort_values(['date','matchID']).reset_index(drop=True)
    matches['matchRank'] = np.arange(len(matches))
    matches['fold'] = -1
    for fold, start in enumerate([70,80,90]):
        lo, hi = len(matches)*start//100, len(matches)*(start+10)//100
        matches.loc[matches.matchRank.ge(lo)&matches.matchRank.lt(hi),'fold'] = fold
    matches['split'] = np.where(matches.fold.ge(0),'test','initial_history')
    return matches


def stage_data(frame, stage):
    if stage == 'production':
        return frame.copy(), frame.iloc[:0].copy()
    percent = int(stage[1:])
    count = frame.matchID.nunique()
    lo, hi = count*percent//100, count*(percent+10)//100
    train = frame.loc[frame.matchRank.lt(lo)].copy()
    valid = frame.loc[frame.matchRank.ge(lo)&frame.matchRank.lt(hi)].copy()
    assert train.date.max() <= valid.date.min()
    assert not set(train.matchID).intersection(valid.matchID)
    return train, valid
