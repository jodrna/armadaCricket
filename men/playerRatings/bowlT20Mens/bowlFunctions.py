import numpy as np
import pandas as pd
from numba import njit
from collections import defaultdict

def qualityMethodBins(
    df: pd.DataFrame,
    bin_size: int = 100,
    rating_col: str = "run_rating",
    player_col: str = "playerid",
    match_col: str = "matchid",
    out_col: str = "binid",
    sort_first: bool = True,
) -> pd.DataFrame:
    df = (df.sort_values(rating_col, ascending=False, kind="mergesort").reset_index(drop=True)
          if sort_first else df.reset_index(drop=True))

    n = len(df)
    p_codes, _ = pd.factorize(df[player_col], sort=False)
    m_codes, _ = pd.factorize(df[match_col], sort=False)
    p_codes = p_codes.astype(np.int64)
    m_codes = m_codes.astype(np.int64)

    bins = np.full(n, -1, dtype=np.int32)
    fill = []

    last_bin_for_match = defaultdict(dict)   # pid -> {mid: last_bin}
    player_bin_match = defaultdict(dict)     # pid -> {bin: mid}

    for i in range(n):
        pid = int(p_codes[i])
        mid = int(m_codes[i])
        b = last_bin_for_match[pid].get(mid, i // bin_size)

        while True:
            if b >= len(fill):
                fill.extend([0] * (b + 1 - len(fill)))
            if fill[b] < bin_size:
                existing = player_bin_match[pid].get(b)
                if existing is None or existing == mid:
                    bins[i] = b
                    fill[b] += 1
                    player_bin_match[pid].setdefault(b, mid)
                    last_bin_for_match[pid][mid] = b
                    break
            b += 1

    df[out_col] = bins
    return df


def newMethodBins(df, bin_size_2=60):
    df['inningsGroup'] = np.ceil(df['ballsremaining'] / bin_size_2)
    df['binid'] = (
        df['inningsGroup'].astype('string')
        .str.cat(df['matchid'].astype('string'), sep='|')
        .str.cat(df['innings'].astype('string'), sep='|')
    )
    return df


def buildRunRatingsOriginalPlayer(param, lookbacks_player):
    k, c, h, r = (
        param[0], param[1], param[2], param[3]
    )

    # run ratingsT20
    lookbacks_player_r = lookbacks_player.copy()

    # initial weight based on days ago and then multiply based on same comp, host, region etc
    lookbacks_player_r['recency_weight'] = (1 - k ** lookbacks_player_r['days_ago'])
    lookbacks_player_r['comp_enc'] = np.where(lookbacks_player_r['competition'] == lookbacks_player_r['competition_2'],
                                              c, 1)
    lookbacks_player_r['host_enc'] = np.where(lookbacks_player_r['host'] == lookbacks_player_r['host_2'], h, 1)

    lookbacks_player_r['host_region_enc'] = np.where(lookbacks_player_r['host'] == lookbacks_player_r['host_2'], 1,
                                                     np.where(lookbacks_player_r['host_region'] == lookbacks_player_r[
                                                         'host_region_2'], r, 1))

    lookbacks_player_r['weight'] = lookbacks_player_r['recency_weight'] * lookbacks_player_r['comp_enc'] * lookbacks_player_r['host_enc'] * lookbacks_player_r[
        'host_region_enc']  # * lookbacks_player_r['home_away_enc']
    # weight the runs and xruns
    lookbacks_player_r['weight_runs'] = lookbacks_player_r['weight'] * lookbacks_player_r['runs_2']
    lookbacks_player_r['weight_exprbowl'] = lookbacks_player_r['weight'] * lookbacks_player_r['realexprbowl']
    # create a pivot to sum the weights then divide for the rating
    ratings_player_r = pd.pivot_table(lookbacks_player_r,
                                      values=['weight_runs', 'weight_exprbowl', 'ord_2', 'balls_bowled_2', 'runs_2', 'realexprbowl_2'],
                                      index=['date', 'matchid', 'playerid', 'bowler', 'host', 'competition', 'bowlertype_2'],
                                      aggfunc={'weight_runs': 'sum', 'weight_exprbowl': 'sum', 'balls_bowled_2': 'sum', 'ord_2': 'mean', 'runs_2': 'sum', 'realexprbowl_2': 'sum'})
    ratings_player_r['run_rating'] = ratings_player_r['weight_runs'] / ratings_player_r['weight_exprbowl']
    ratings_player_r = ratings_player_r.reset_index()
    ratings_player_r['z_run_ratio'] = ratings_player_r['runs_2'] / ratings_player_r['realexprbowl_2']


    return ratings_player_r, lookbacks_player_r


def buildWktRatingsOriginalPlayer(param, lookbacks_player):
    k, c, h, r = (
        param[0], param[1], param[2], param[3]
    )

    # wkt ratingsT20, same as above
    lookbacks_player_w = lookbacks_player.copy()

    # initial weight based on days ago and then multiply based on same comp, host, region etc
    lookbacks_player_w['recency_weight'] = (1 - k ** lookbacks_player_w['days_ago'])
    lookbacks_player_w['comp_enc'] = np.where(lookbacks_player_w['competition'] == lookbacks_player_w['competition_2'],
                                              c, 1)
    lookbacks_player_w['host_enc'] = np.where(lookbacks_player_w['host'] == lookbacks_player_w['host_2'], h, 1)
    lookbacks_player_w['host_region_enc'] = np.where(lookbacks_player_w['host'] == lookbacks_player_w['host_2'], 1,
                                                     np.where(lookbacks_player_w['host_region'] == lookbacks_player_w[
                                                         'host_region_2'], r, 1))

    lookbacks_player_w['weight'] = lookbacks_player_w['recency_weight'] * lookbacks_player_w['comp_enc'] * lookbacks_player_w['host_enc'] * lookbacks_player_w[
        'host_region_enc']  # * lookbacks_player_w['home_away_enc']

    # weight the wkts and xwkts and then divide for the rating
    lookbacks_player_w['weight_wkt'] = lookbacks_player_w['weight'] * lookbacks_player_w['wkt_2']
    lookbacks_player_w['weight_expwbowl'] = lookbacks_player_w['weight'] * lookbacks_player_w['realexpwbowl']
    # create a pivot to sum the weights then divide for the rating
    ratings_player_w = pd.pivot_table(lookbacks_player_w,
                                      values=['weight_wkt', 'weight_expwbowl', 'ord_2', 'balls_bowled_2', 'wkt_2', 'realexpwbowl_2'],
                                      index=['date', 'matchid', 'playerid', 'bowler', 'host', 'competition', 'bowlertype_2'],
                                      aggfunc={'weight_wkt': 'sum', 'weight_expwbowl': 'sum', 'balls_bowled_2': 'sum', 'ord_2': 'mean', 'wkt_2': 'sum', 'realexpwbowl_2': 'sum'})
    ratings_player_w['wkt_rating'] = ratings_player_w['weight_wkt'] / ratings_player_w['weight_expwbowl']
    ratings_player_w = ratings_player_w.reset_index()  # move this??
    ratings_player_w['z_wkt_ratio'] = ratings_player_w['wkt_2'] / ratings_player_w['realexpwbowl_2']

    return ratings_player_w, lookbacks_player_w




def buildRunRatingsOriginalInning(param, lookbacks_player):
    """
    Builds type-aware run outputs for bowlers using different weights for seam vs spin.
    Expects `run_params` dict with keys: k_sm, h_sm, r_sm, c_sm, k_s, h_s, r_s, c_s.
    """
    # run ratingsT20
    lookbacks_player_r = lookbacks_player.copy()

    def get_var(v):
        # if bowlertype_2 == 'spin' use *_s, otherwise use *_sm (seam/medium)
        return np.where(
            lookbacks_player_r['bowlertype_2'] == 'spin',
            param[f"{v}_s"],
            param[f"{v}_sm"]
        )

    # initial weight based on days ago and then multiply based on same comp, host, region etc
    lookbacks_player_r['recency_weight'] = ((1 - get_var('k')) ** lookbacks_player_r['days_ago'])
    lookbacks_player_r['comp_enc'] = np.where(
        lookbacks_player_r['competition'] == lookbacks_player_r['competition_2'],
        get_var('c'),
        1
    )
    lookbacks_player_r['host_enc'] = np.where(
        lookbacks_player_r['host'] == lookbacks_player_r['host_2'],
        get_var('h'),
        1
    )

    lookbacks_player_r['host_region_enc'] = np.where(
        lookbacks_player_r['host'] == lookbacks_player_r['host_2'],
        1,
        np.where(
            lookbacks_player_r['host_region'] == lookbacks_player_r['host_region_2'],
            get_var('r'),
            1
        )
    )

    # final per-row weight
    lookbacks_player_r['weight'] = (
        lookbacks_player_r['recency_weight'] *
        lookbacks_player_r['comp_enc'] *
        lookbacks_player_r['host_enc'] *
        lookbacks_player_r['host_region_enc']
    )  # * lookbacks_player_r['home_away_enc']  # optional

    # weight the runs and expected runs vs bowler
    lookbacks_player_r['weight_runs'] = lookbacks_player_r['weight'] * lookbacks_player_r['runs_2']
    lookbacks_player_r['weight_exprbowl'] = lookbacks_player_r['weight'] * lookbacks_player_r['realexprbowl']

    # aggregate to per-innings outputs
    ratings_player_r = pd.pivot_table(
        lookbacks_player_r,
        values=[
            'weight_runs', 'weight_exprbowl', 'ord_2', 'balls_bowled_2', 'runs_2', 'realexprbowl_2'
        ],
        index=['date', 'playerid', 'bowler', 'host', 'competition', 'bowlertype_2'],
        aggfunc={
            'weight_runs': 'sum',
            'weight_exprbowl': 'sum',
            'balls_bowled_2': 'sum',
            'ord_2': 'mean',
            'runs_2': 'sum',
            'realexprbowl_2': 'sum',
        }
    )
    ratings_player_r['run_rating'] = ratings_player_r['weight_runs'] / ratings_player_r['weight_exprbowl']
    ratings_player_r = ratings_player_r.reset_index()
    ratings_player_r['z_run_ratio'] = ratings_player_r['runs_2'] / ratings_player_r['realexprbowl_2']

    return ratings_player_r, lookbacks_player_r



def buildWktRatingsOriginalInning(param, lookbacks_player):
    """
    Builds type-aware wicket outputs for bowlers using different weights for seam vs spin.
    Expects `param` dict with keys: k_sm, h_sm, r_sm, c_sm, k_s, h_s, r_s, c_s.
    """

    # Copy first so the inner function can reference this frame
    lookbacks_player_w = lookbacks_player.copy()

    def get_var(v):
        # if bowlertype_2 == 'spin' use *_s, otherwise use *_sm (seam/medium)
        return np.where(
            lookbacks_player_w['bowlertype_2'] == 'spin',
            param[f"{v}_s"],
            param[f"{v}_sm"]
        )

    # initial weight based on days ago and then multiply based on same comp, host, region etc
    lookbacks_player_w['recency_weight'] = ((1 - get_var('k')) ** lookbacks_player_w['days_ago'])
    lookbacks_player_w['comp_enc'] = np.where(
        lookbacks_player_w['competition'] == lookbacks_player_w['competition_2'],
        get_var('c'),
        1
    )
    lookbacks_player_w['host_enc'] = np.where(
        lookbacks_player_w['host'] == lookbacks_player_w['host_2'],
        get_var('h'),
        1
    )
    lookbacks_player_w['host_region_enc'] = np.where(
        lookbacks_player_w['host'] == lookbacks_player_w['host_2'],
        1,
        np.where(
            lookbacks_player_w['host_region'] == lookbacks_player_w['host_region_2'],
            get_var('r'),
            1
        )
    )

    lookbacks_player_w['weight'] = (
        lookbacks_player_w['recency_weight'] *
        lookbacks_player_w['comp_enc'] *
        lookbacks_player_w['host_enc'] *
        lookbacks_player_w['host_region_enc']
    )  # * lookbacks_player_w['home_away_enc']

    # weight the wkts and xwkts and then divide for the rating
    lookbacks_player_w['weight_wkt'] = lookbacks_player_w['weight'] * lookbacks_player_w['wkt_2']
    lookbacks_player_w['weight_expwbowl'] = lookbacks_player_w['weight'] * lookbacks_player_w['realexpwbowl']

    # aggregate
    ratings_player_w = pd.pivot_table(
        lookbacks_player_w,
        values=['weight_wkt', 'weight_expwbowl', 'ord_2', 'balls_bowled_2', 'wkt_2', 'realexpwbowl_2'],
        index=['date', 'playerid', 'bowler', 'host', 'competition', 'bowlertype_2'],
        aggfunc={
            'weight_wkt': 'sum',
            'weight_expwbowl': 'sum',
            'balls_bowled_2': 'sum',
            'ord_2': 'mean',
            'wkt_2': 'sum',
            'realexpwbowl_2': 'sum',
        }
    )
    ratings_player_w['wkt_rating'] = ratings_player_w['weight_wkt'] / ratings_player_w['weight_expwbowl']
    ratings_player_w = ratings_player_w.reset_index()
    ratings_player_w['z_wkt_ratio'] = ratings_player_w['wkt_2'] / ratings_player_w['realexpwbowl_2']

    return ratings_player_w, lookbacks_player_w


def buildRunRatingsOriginal(param, lookbacks_player):
    """
    Type-aware run outputs for bowlers with priority mapping:
      same_comp > is_t20 > is_odi1 > is_odi2
    Then apply location boosts: same host, else same region.
    Spin and seam/medium use separate parameters via the `param` dict.

    Expected keys in `param` (per type suffix: _s for spin, _sm for seam/medium):
      - k_*   : recency decay (per day) for (1 - k) ** days_ago
      - c_*   : same competition multiplier
      - h_*   : same host multiplier
      - r_*   : same region (host differs) multiplier
      - t20_* : prior format T20 (different competition)
      - odi1_*: prior format ODI1
      - odi2_*: prior format ODI2
    """
    lookbacks_player_r = lookbacks_player.copy()

    def get_var(v):
        # if bowlertype_2 == 'spin' use *_s, otherwise use *_sm (seam/medium)
        return np.where(
            np.isin(lookbacks_player_r['bowlertype_2'], ['f_spin', 'w_spin']),
            param[f"{v}_s"],
            param[f"{v}_sm"]
        )

    # Flags
    same_comp = (lookbacks_player_r['competition'] == lookbacks_player_r['competition_2'])
    # Determine prior format buckets
    prior_is_odi1 = (lookbacks_player_r['competition_2'] == 'ODI1')
    prior_is_odi2 = (lookbacks_player_r['competition_2'] == 'ODI2')
    prior_is_t20  = ~(prior_is_odi1 | prior_is_odi2)

    # Location flags
    host_same = (lookbacks_player_r['host'] == lookbacks_player_r['host_2'])
    region_same = (~host_same) & (lookbacks_player_r['host_region'] == lookbacks_player_r['host_region_2'])

    # Recency
    lookbacks_player_r['recency_weight'] = ((1 - get_var('k')) ** lookbacks_player_r['days_ago'])

    # Primary multiplier by priority: same_comp > t20 > odi1 > odi2
    lookbacks_player_r['primary_weight'] = np.where(
        same_comp,
        get_var('c'),
        np.where(
            prior_is_t20,
            get_var('t20'),
            np.where(prior_is_odi1, get_var('odi1'),
                     np.where(prior_is_odi2, get_var('odi2'), 1.0))
        )
    )

    # Location multipliers
    lookbacks_player_r['host_enc'] = np.where(host_same, get_var('h'), 1.0)
    lookbacks_player_r['region_enc'] = np.where((~host_same) & region_same, get_var('r'), 1.0)

    # location per-row weight
    lookbacks_player_r['weightOne'] = (
        lookbacks_player_r['primary_weight'] *
        lookbacks_player_r['host_enc'] *
        lookbacks_player_r['region_enc']
    )

    lookbacks_player_r['weight'] = lookbacks_player_r['weightOne'] * lookbacks_player_r['recency_weight']

    # weight the runs and expected runs vs bowler
    lookbacks_player_r['weight_runs'] = lookbacks_player_r['weight'] * lookbacks_player_r['runs_2']
    lookbacks_player_r['weight_exprbowl'] = lookbacks_player_r['weight'] * lookbacks_player_r['adj_realexprbowl']
    lookbacks_player_r['weight_exprbowl_unadjusted'] = lookbacks_player_r['weight'] * lookbacks_player_r['realexprbowl_2']


    # aggregate to per-innings outputs
    ratings_player_r = pd.pivot_table(
        lookbacks_player_r,
        values=[
            'weight_runs', 'weight_exprbowl', 'weight_exprbowl_unadjusted', 'ord_2', 'balls_bowled_2', 'runs_2', 'realexprbowl_2'
        ],
        index=['date', 'playerid', 'bowler', 'host', 'competition', 'bowlertype_2', 'matchid'],
        aggfunc={
            'weight_runs': 'sum',
            'weight_exprbowl': 'sum',
            'weight_exprbowl_unadjusted': 'sum',
            'balls_bowled_2': 'sum',
            'ord_2': 'mean',
            'runs_2': 'sum',
            'realexprbowl_2': 'sum',
        }
    )
    ratings_player_r['run_rating_0'] = ratings_player_r['weight_runs'] / ratings_player_r['weight_exprbowl_unadjusted']
    ratings_player_r['run_rating'] = ratings_player_r['weight_runs'] / ratings_player_r['weight_exprbowl']
    ratings_player_r = ratings_player_r.reset_index()
    ratings_player_r['z_run_ratio'] = ratings_player_r['runs_2'] / ratings_player_r['realexprbowl_2']

    return ratings_player_r, lookbacks_player_r



def buildWktRatingsOriginal(param, lookbacks_player):
    """
    Type-aware wkt outputs for bowlers with priority mapping:
      same_comp > is_t20 > is_odi1 > is_odi2
    Then apply location boosts: same host, else same region.
    Spin and seam/medium use separate parameters via the `param` dict.

    Expected keys in `param` (per type suffix: _s for spin, _sm for seam/medium):
      - k_*   : recency decay (per day) for (1 - k) ** days_ago
      - c_*   : same competition multiplier
      - h_*   : same host multiplier
      - r_*   : same region (host differs) multiplier
      - t20_* : prior format T20 (different competition)
      - odi1_*: prior format ODI1
      - odi2_*: prior format ODI2
    """
    lookbacks_player_w = lookbacks_player.copy()

    def get_var(v):
        # if bowlertype_2 == 'spin' use *_s, otherwise use *_sm (seam/medium)
        return np.where(
            np.isin(lookbacks_player_w['bowlertype_2'], ['f_spin', 'w_spin']),
            param[f"{v}_s"],
            param[f"{v}_sm"]
        )

    # Flags
    same_comp = (lookbacks_player_w['competition'] == lookbacks_player_w['competition_2'])
    # Determine prior format buckets
    prior_is_odi1 = (lookbacks_player_w['competition_2'] == 'ODI1')
    prior_is_odi2 = (lookbacks_player_w['competition_2'] == 'ODI2')
    prior_is_t20  = ~(prior_is_odi1 | prior_is_odi2)

    # Location flags
    host_same = (lookbacks_player_w['host'] == lookbacks_player_w['host_2'])
    region_same = (~host_same) & (lookbacks_player_w['host_region'] == lookbacks_player_w['host_region_2'])

    # Recency
    lookbacks_player_w['recency_weight'] = ((1 - get_var('k')) ** lookbacks_player_w['days_ago'])

    # Primary multiplier by priority: same_comp > t20 > odi1 > odi2
    lookbacks_player_w['primary_weight'] = np.where(
        same_comp,
        get_var('c'),
        np.where(
            prior_is_t20,
            get_var('t20'),
            np.where(prior_is_odi1, get_var('odi1'),
                     np.where(prior_is_odi2, get_var('odi2'), 1.0))
        )
    )

    # Location multipliers
    lookbacks_player_w['host_enc'] = np.where(host_same, get_var('h'), 1.0)
    lookbacks_player_w['region_enc'] = np.where((~host_same) & region_same, get_var('r'), 1.0)

    # location per-row weight
    lookbacks_player_w['weightOne'] = (
        lookbacks_player_w['primary_weight'] *
        lookbacks_player_w['host_enc'] *
        lookbacks_player_w['region_enc']
    )

    lookbacks_player_w['weight'] = lookbacks_player_w['weightOne'] * lookbacks_player_w['recency_weight']

    # weight the wkt and expected wkt vs bowler
    lookbacks_player_w['weight_wkt'] = lookbacks_player_w['weight'] * lookbacks_player_w['wkt_2']
    lookbacks_player_w['weight_expwbowl'] = lookbacks_player_w['weight'] * lookbacks_player_w['adj_realexpwbowl']
    lookbacks_player_w['weight_expwbowl_unadjusted'] = lookbacks_player_w['weight'] * lookbacks_player_w['realexpwbowl_2']

    # aggregate to per-innings outputs
    ratings_player_w = pd.pivot_table(
        lookbacks_player_w,
        values=['weight_wkt', 'weight_expwbowl', 'weight_expwbowl_unadjusted', 'ord_2', 'balls_bowled_2', 'wkt_2', 'realexpwbowl_2'],
        index=['date', 'playerid', 'bowler', 'host', 'competition', 'bowlertype_2', 'matchid'],
        aggfunc={'weight_wkt': 'sum',
                'weight_expwbowl': 'sum',
                'weight_expwbowl_unadjusted': 'sum',
                'balls_bowled_2': 'sum',
                'ord_2': 'mean',
                'wkt_2': 'sum',
                'realexpwbowl_2': 'sum'}
    )
    ratings_player_w['wkt_rating_0'] = ratings_player_w['weight_wkt'] / ratings_player_w['weight_expwbowl_unadjusted']
    ratings_player_w['wkt_rating'] = ratings_player_w['weight_wkt'] / ratings_player_w['weight_expwbowl']
    ratings_player_w = ratings_player_w.reset_index()
    ratings_player_w['z_wkt_ratio'] = ratings_player_w['wkt_2'] / ratings_player_w['realexpwbowl_2']

    return ratings_player_w, lookbacks_player_w


