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





def buildRunRatingsMapOneLog(param, lookbacks_player):
    b_same_comp = np.log(param[1])                        # cd baseline for same competition (non-T20I)
    b_same_comp_t20i = np.log(param[2] / param[1])        # extra weight if same competition is T20I: ci/cd
    b_recent_same_tournament = np.log(param[8])           # f for same-comp non-T20I and <90 days

    b_host_same = np.log(param[3])                        # h (same host)
    b_region_same = np.log(param[4])                      # r (same region, only when host differs)

    # Formats only apply when competition differs
    b_format_t20 = np.log(param[5])
    b_format_odi1 = np.log(param[6])
    b_format_odi2 = np.log(param[7])

    # Exponential recency decay parameter
    lambda_days = param[0]

    lookbacks_player_r = lookbacks_player.copy()

    # Recency weight
    lookbacks_player_r['recency_weight'] = np.exp(-lambda_days * lookbacks_player_r['days_ago'])

    # Indicators
    same_comp = (lookbacks_player_r['competition'] == lookbacks_player_r['competition_2'])
    same_comp_t20i = same_comp & (lookbacks_player_r['competition'] == 'T20I')
    same_comp_non_t20i = same_comp & (lookbacks_player_r['competition'] != 'T20I')
    recent_same_tournament = same_comp_non_t20i & (lookbacks_player_r['days_ago'] < 90)
    host_same = (lookbacks_player_r['host'] == lookbacks_player_r['host_2'])
    region_same = (~host_same) & (lookbacks_player_r['host_region'] == lookbacks_player_r['host_region_2'])

    # Prior format flags, but only when competition differs
    prior_is_t20 = (~same_comp) & (~lookbacks_player_r['competition_2'].isin(['ODI1', 'ODI2']))
    prior_is_odi1 = (~same_comp) & (lookbacks_player_r['competition_2'] == 'ODI1')
    prior_is_odi2 = (~same_comp) & (lookbacks_player_r['competition_2'] == 'ODI2')

    # Expose component weights (to show how location_weight is built)
    lookbacks_player_r['comp_weight'] = np.where(
        same_comp_t20i, np.exp(b_same_comp + b_same_comp_t20i),
        np.where(same_comp_non_t20i, np.exp(b_same_comp), 1.0)
    )
    lookbacks_player_r['tournament_weight'] = np.where(recent_same_tournament, np.exp(b_recent_same_tournament), 1.0)
    lookbacks_player_r['host_weight'] = np.where(host_same, np.exp(b_host_same), 1.0)
    lookbacks_player_r['region_weight'] = np.where(region_same, np.exp(b_region_same), 1.0)
    lookbacks_player_r['format_weight'] = np.where(
        prior_is_t20, np.exp(b_format_t20),
        np.where(prior_is_odi1, np.exp(b_format_odi1),
                 np.where(prior_is_odi2, np.exp(b_format_odi2), 1.0))
    )

    # get final weight
    log_location_weight = (
        b_same_comp * same_comp.astype(float) +
        b_same_comp_t20i * same_comp_t20i.astype(float) +
        b_recent_same_tournament * recent_same_tournament.astype(float) +
        b_host_same * host_same.astype(float) +
        b_region_same * region_same.astype(float) +
        b_format_t20 * prior_is_t20.astype(float) +
        b_format_odi1 * prior_is_odi1.astype(float) +
        b_format_odi2 * prior_is_odi2.astype(float)
    )
    lookbacks_player_r['location_weight_from_log'] = np.exp(log_location_weight)

    # Location weight as product of all
    lookbacks_player_r['location_weight'] = (
        lookbacks_player_r['comp_weight'] *
        lookbacks_player_r['tournament_weight'] *
        lookbacks_player_r['host_weight'] *
        lookbacks_player_r['region_weight'] *
        lookbacks_player_r['format_weight']
    )

    # Total weight
    lookbacks_player_r['weight'] = lookbacks_player_r['recency_weight'] * lookbacks_player_r['location_weight']

    # Apply weight to components used in rating
    lookbacks_player_r['weight_runs'] = lookbacks_player_r['weight'] * lookbacks_player_r['runs_2']
    lookbacks_player_r['weight_exprbat'] = lookbacks_player_r['weight'] * lookbacks_player_r['realexprbat_2']
    lookbacks_player_r['weight_ord_r'] = lookbacks_player_r['weight'] * lookbacks_player_r['ord_2']
    lookbacks_player_r['weight_balls_r'] = lookbacks_player_r['weight'] * lookbacks_player_r['balls_faced_2']

    # Aggregate to per-innings outputs
    ratings_player_r = pd.pivot_table(
        lookbacks_player_r,
        values=['weight', 'weight_runs', 'weight_exprbat', 'weight_ord_r',
                'balls_faced_2', 'runs_2', 'realexprbat_2', 'weight_balls_r'],
        index=['date', 'matchid', 'playerid', 'batsman', 'host', 'competition'],
        aggfunc={'weight': 'sum', 'weight_runs': 'sum', 'weight_exprbat': 'sum',
                 'balls_faced_2': 'sum', 'weight_ord_r': 'sum', 'runs_2': 'sum',
                 'realexprbat_2': 'sum', 'weight_balls_r': 'sum'}
    ).reset_index()

    ratings_player_r['run_rating'] = ratings_player_r['weight_runs'] / ratings_player_r['weight_exprbat']
    ratings_player_r['z_run_ratio'] = ratings_player_r['runs_2'] / ratings_player_r['realexprbat_2']
    ratings_player_r['ord_2_r'] = ratings_player_r['weight_ord_r'] / ratings_player_r['weight']

    return ratings_player_r, lookbacks_player_r






def buildRunRatingsOriginal(param, lookbacks_player):
    k, ci, cd, h, r, t20, odi1, odi2, f = (
        param[0], param[1], param[2], param[3], param[4], param[5], param[6], param[7], param[8]
    )
    lookbacks_player_r = lookbacks_player.copy()

    # Recency weight with k scaled by career balls
    lookbacks_player_r['k'] = k * np.where(
        lookbacks_player_r['balls_faced_career'] > 750,
        1,
        0.5 + (0.5 * (lookbacks_player_r['balls_faced_career']) / 750)
    )
    lookbacks_player_r['recency_weight'] = ((1 - lookbacks_player_r['k']) ** lookbacks_player_r['days_ago'])

    # Format encoder (prior inning)
    lookbacks_player_r['format_enc'] = np.where(
        lookbacks_player_r['competition_2'] == 'ODI1', odi1,
        np.where(lookbacks_player_r['competition_2'] == 'ODI2', odi2, t20)
    )

    # Tournament boost (same comp, non-T20I, within 90 days)
    lookbacks_player_r['tournament_boost'] = np.where(
        (lookbacks_player_r['days_ago'] < 90) &
        (lookbacks_player_r['competition'] == lookbacks_player_r['competition_2']) &
        (lookbacks_player_r['competition'] != 'T20I'),
        f,
        1
    )

    # Competition encoder
    lookbacks_player_r['comp_enc'] = np.where(
        lookbacks_player_r['competition'] == lookbacks_player_r['competition_2'],
        np.where(lookbacks_player_r['competition'] == 'T20I', ci, cd),
        1
    )

    # Host and region encoders
    lookbacks_player_r['host_enc'] = np.where(
        lookbacks_player_r['host'] == lookbacks_player_r['host_2'], h, 1
    )
    lookbacks_player_r['host_region_enc'] = np.where(
        lookbacks_player_r['host'] == lookbacks_player_r['host_2'], 1,
        np.where(lookbacks_player_r['host_region'] == lookbacks_player_r['host_region_2'], r, 1)
    )

    # Location weight (multiplicative mapping)
    lookbacks_player_r['location_weight'] = (
        lookbacks_player_r['comp_enc'] *
        lookbacks_player_r['host_enc'] *
        lookbacks_player_r['host_region_enc'] *
        lookbacks_player_r['format_enc'] *
        lookbacks_player_r['tournament_boost']
    )

    # Total per-row weight
    lookbacks_player_r['weight'] = lookbacks_player_r['recency_weight'] * lookbacks_player_r['location_weight']

    # Apply weights to components used in rating
    lookbacks_player_r['weight_runs'] = lookbacks_player_r['weight'] * lookbacks_player_r['runs_2']
    lookbacks_player_r['weight_exprbat'] = lookbacks_player_r['weight'] * lookbacks_player_r['realexprbat_2']
    lookbacks_player_r['weight_ord_r'] = lookbacks_player_r['weight'] * lookbacks_player_r['ord_2']
    lookbacks_player_r['weight_balls_r'] = lookbacks_player_r['weight'] * lookbacks_player_r['balls_faced_2']

    # Aggregate to per-innings outputs
    ratings_player_r = pd.pivot_table(
        lookbacks_player_r,
        values=[
            'weight', 'weight_runs', 'weight_exprbat', 'weight_ord_r',
            'balls_faced_2', 'runs_2', 'realexprbat_2', 'weight_balls_r'
        ],
        index=['date', 'matchid', 'playerid', 'batsman', 'host', 'competition'],
        aggfunc={
            'weight': 'sum', 'weight_runs': 'sum', 'weight_exprbat': 'sum',
            'balls_faced_2': 'sum', 'weight_ord_r': 'sum', 'runs_2': 'sum',
            'realexprbat_2': 'sum', 'weight_balls_r': 'sum'
        }
    ).reset_index()

    ratings_player_r['run_rating'] = ratings_player_r['weight_runs'] / ratings_player_r['weight_exprbat']
    ratings_player_r['z_run_ratio'] = ratings_player_r['runs_2'] / ratings_player_r['realexprbat_2']
    ratings_player_r['ord_2_r'] = ratings_player_r['weight_ord_r'] / ratings_player_r['weight']

    return ratings_player_r, lookbacks_player_r




def buildRunRatingsMapTwoLog(param, lookbacks_player):
      # 0: m_same_domestic        (lift for same competition, non-T20I)
      # 1: m_same_T20I_extra      (extra lift when same competition is T20I; stacks on m_same_domestic)
      # 2: m_recent_tournament    (same domestic comp & days_ago < 90)
      # 3: m_host_same            (same host)
      # 4: m_region_same          (same region, only if host differs)
      # 5: m_format_T20_diff      (prior format T20 when competition differs)
      # 6: m_format_ODI2_diff     (prior format ODI2 when competition differs; ODI1 is BASELINE)
      # 7: lambda_days

    beta_same_domestic     = np.log(param[0])
    beta_same_T20I         = np.log(param[1])
    beta_recent_tournament = np.log(param[2])
    beta_host_same         = np.log(param[3])
    beta_region_same       = np.log(param[4])
    beta_format_T20_diff   = np.log(param[5])
    beta_format_ODI2_diff  = np.log(param[6])
    lambda_days            = param[7]

    lookbacks_player_r = lookbacks_player.copy()

    # Recency (exponential, independent)
    lookbacks_player_r['recency_weight'] = np.exp(-lambda_days * lookbacks_player_r['days_ago'])

    # Flags
    same_comp = (lookbacks_player_r['competition'] == lookbacks_player_r['competition_2'])
    same_comp_t20i = same_comp & (lookbacks_player_r['competition'] == 'T20I')
    same_comp_domestic = same_comp & (lookbacks_player_r['competition'] != 'T20I')

    # Tournament recency (only for same domestic comp)
    recent_same_tournament = same_comp_domestic & (lookbacks_player_r['days_ago'] < 90)

    # Location flags
    host_same = (lookbacks_player_r['host'] == lookbacks_player_r['host_2'])
    region_same = (~host_same) & (lookbacks_player_r['host_region'] == lookbacks_player_r['host_region_2'])

    # Prior format flags for different competition only (ODI1 is baseline)
    diff_comp = ~same_comp
    prior_format_T20 = diff_comp & (~lookbacks_player_r['competition_2'].isin(['ODI1', 'ODI2']))
    prior_format_ODI2 = diff_comp & (lookbacks_player_r['competition_2'] == 'ODI2')

    # Components
    lookbacks_player_r['comp_weight'] = np.where(
        same_comp_t20i, np.exp(beta_same_domestic + beta_same_T20I),
        np.where(same_comp_domestic, np.exp(beta_same_domestic), 1.0)
    )
    lookbacks_player_r['tournament_weight'] = np.where(recent_same_tournament, np.exp(beta_recent_tournament), 1.0)
    lookbacks_player_r['host_weight'] = np.where(host_same, np.exp(beta_host_same), 1.0)
    lookbacks_player_r['region_weight'] = np.where(region_same, np.exp(beta_region_same), 1.0)
    lookbacks_player_r['format_weight'] = np.where(
        prior_format_T20, np.exp(beta_format_T20_diff),
        np.where(prior_format_ODI2, np.exp(beta_format_ODI2_diff), 1.0)
    )

    # Final weights
    lookbacks_player_r['location_weight'] = (
        lookbacks_player_r['comp_weight'] *
        lookbacks_player_r['tournament_weight'] *
        lookbacks_player_r['host_weight'] *
        lookbacks_player_r['region_weight'] *
        lookbacks_player_r['format_weight']
    )
    lookbacks_player_r['weight'] = lookbacks_player_r['recency_weight'] * lookbacks_player_r['location_weight']

    # Apply weights for rating
    lookbacks_player_r['weight_runs'] = lookbacks_player_r['weight'] * lookbacks_player_r['runs_2']
    lookbacks_player_r['weight_exprbat'] = lookbacks_player_r['weight'] * lookbacks_player_r['realexprbat_2']
    lookbacks_player_r['weight_ord_r'] = lookbacks_player_r['weight'] * lookbacks_player_r['ord_2']
    lookbacks_player_r['weight_balls_r'] = lookbacks_player_r['weight'] * lookbacks_player_r['balls_faced_2']

    # Aggregate to per-innings outputs
    ratings_player_r = pd.pivot_table(
        lookbacks_player_r,
        values=['weight', 'weight_runs', 'weight_exprbat', 'weight_ord_r',
                'balls_faced_2', 'runs_2', 'realexprbat_2', 'weight_balls_r'],
        index=['date', 'matchid', 'playerid', 'batsman', 'host', 'competition'],
        aggfunc={'weight': 'sum', 'weight_runs': 'sum', 'weight_exprbat': 'sum',
                 'balls_faced_2': 'sum', 'weight_ord_r': 'sum', 'runs_2': 'sum',
                 'realexprbat_2': 'sum', 'weight_balls_r': 'sum'}
    ).reset_index()

    ratings_player_r['run_rating'] = ratings_player_r['weight_runs'] / ratings_player_r['weight_exprbat']
    ratings_player_r['z_run_ratio'] = ratings_player_r['runs_2'] / ratings_player_r['realexprbat_2']
    ratings_player_r['ord_2_r'] = ratings_player_r['weight_ord_r'] / ratings_player_r['weight']

    return ratings_player_r, lookbacks_player_r



def buildRunRatingsMapOne(param, lookbacks_player):
    # param order: [k, ci, cd, h, r, t20, odi1, odi2, f]
    k, ci, cd, h, r, t20, odi1, odi2, f = (
        param[0], param[1], param[2], param[3], param[4], param[5], param[6], param[7], param[8]
    )

    lookbacks_player_r = lookbacks_player.copy()

    # Recency weight
    lookbacks_player_r['recency_weight'] = ((1 - k) ** lookbacks_player_r['days_ago'])

    # Indicators
    same_comp = (lookbacks_player_r['competition'] == lookbacks_player_r['competition_2'])
    same_comp_t20i = same_comp & (lookbacks_player_r['competition'] == 'T20I')
    same_comp_non_t20i = same_comp & (lookbacks_player_r['competition'] != 'T20I')
    recent_same_tournament = same_comp_non_t20i & (lookbacks_player_r['days_ago'] < 90)

    host_same = (lookbacks_player_r['host'] == lookbacks_player_r['host_2'])
    region_same = (~host_same) & (lookbacks_player_r['host_region'] == lookbacks_player_r['host_region_2'])

    # Prior format flags (apply only when competition differs)
    prior_is_t20 = (~same_comp) & (~lookbacks_player_r['competition_2'].isin(['ODI1', 'ODI2']))
    prior_is_odi1 = (~same_comp) & (lookbacks_player_r['competition_2'] == 'ODI1')
    prior_is_odi2 = (~same_comp) & (lookbacks_player_r['competition_2'] == 'ODI2')


    # Competition component
    lookbacks_player_r['comp_weight'] = np.where(
        same_comp_t20i, ci,
        np.where(same_comp_non_t20i, cd, 1.0)
    )
    # Tournament boost
    lookbacks_player_r['tournament_weight'] = np.where(recent_same_tournament, f, 1.0)
    # host component
    lookbacks_player_r['host_weight'] = np.where(host_same, h, 1.0)
    lookbacks_player_r['region_weight'] = np.where(region_same, r, 1.0)
    # Format component
    lookbacks_player_r['format_weight'] = np.where(
        prior_is_t20, t20,
        np.where(prior_is_odi1, odi1,
                 np.where(prior_is_odi2, odi2, 1.0))
    )

    # Final location weight (product of components)
    lookbacks_player_r['location_weight'] = (
        lookbacks_player_r['comp_weight'] *
        lookbacks_player_r['tournament_weight'] *
        lookbacks_player_r['host_weight'] *
        lookbacks_player_r['region_weight'] *
        lookbacks_player_r['format_weight']
    )

    # Total weight
    lookbacks_player_r['weight'] = lookbacks_player_r['recency_weight'] * lookbacks_player_r['location_weight']

    # Apply weights for rating
    lookbacks_player_r['weight_runs'] = lookbacks_player_r['weight'] * lookbacks_player_r['runs_2']
    lookbacks_player_r['weight_exprbat'] = lookbacks_player_r['weight'] * lookbacks_player_r['realexprbat_2']
    lookbacks_player_r['weight_ord_r'] = lookbacks_player_r['weight'] * lookbacks_player_r['ord_2']
    lookbacks_player_r['weight_balls_r'] = lookbacks_player_r['weight'] * lookbacks_player_r['balls_faced_2']

    # Aggregate to per-innings outputs
    ratings_player_r = pd.pivot_table(
        lookbacks_player_r,
        values=['weight', 'weight_runs', 'weight_exprbat', 'weight_ord_r',
                'balls_faced_2', 'runs_2', 'realexprbat_2', 'weight_balls_r'],
        index=['date', 'matchid', 'playerid', 'batsman', 'host', 'competition'],
        aggfunc={'weight': 'sum', 'weight_runs': 'sum', 'weight_exprbat': 'sum',
                 'balls_faced_2': 'sum', 'weight_ord_r': 'sum', 'runs_2': 'sum',
                 'realexprbat_2': 'sum', 'weight_balls_r': 'sum'}
    ).reset_index()

    ratings_player_r['run_rating'] = ratings_player_r['weight_runs'] / ratings_player_r['weight_exprbat']
    ratings_player_r['z_run_ratio'] = ratings_player_r['runs_2'] / ratings_player_r['realexprbat_2']
    ratings_player_r['ord_2_r'] = ratings_player_r['weight_ord_r'] / ratings_player_r['weight']

    return ratings_player_r, lookbacks_player_r



def buildRunRatingsMapTwo(param, lookbacks_player):
    #   0: m_same_domestic        (lift for same competition, non-T20I)
    #   1: m_same_T20I_extra      (additional lift when same competition is T20I; stacks on m_same_domestic)
    #   2: m_same_tournament    (same domestic comp & days_ago < 90)
    #   3: m_host_same            (same host)
    #   4: m_region_same          (same region, host differs)
    #   5: m_format_T20_diff      (prior format T20 when competition differs; ODI1 is BASELINE)
    #   6: m_format_ODI2_diff     (prior format ODI2 when competition differs; ODI1 is BASELINE)
    #   7: k                      (recency decay rate for (1 - k) ** days_ago, 0 <= k < 1)

    # Work on a copy named consistently
    lookbacks_player_r = lookbacks_player.copy()

    # Recency weight using classic (1 - k) ** days_ago
    lookbacks_player_r['recency_weight'] = (1.0 - param[7]) ** lookbacks_player_r['days_ago']

    # Flags
    same_comp = (lookbacks_player_r['competition'] == lookbacks_player_r['competition_2'])
    same_comp_t20i = same_comp & (lookbacks_player_r['competition'] == 'T20I')
    same_comp_domestic = same_comp & (lookbacks_player_r['competition'] != 'T20I')

    # Tournament recency (only for same domestic comp)
    recent_same_tournament = same_comp_domestic & (lookbacks_player_r['days_ago'] < 90)

    # location flag, same host country or region
    host_same = (lookbacks_player_r['host'] == lookbacks_player_r['host_2'])
    region_same = (~host_same) & (lookbacks_player_r['host_region'] == lookbacks_player_r['host_region_2'])

    # Prior format flags for different competition only (ODI1 is BASELINE)
    diff_comp = ~same_comp
    prior_format_T20 = diff_comp & (~lookbacks_player_r['competition_2'].isin(['ODI1', 'ODI2']))
    prior_format_ODI2 = diff_comp & (lookbacks_player_r['competition_2'] == 'ODI2')
    # prior_format_ODI1 == baseline.... (no multiplier)

    # Component weights, all mutually exclusive
    # Competition component
    lookbacks_player_r['comp_weight'] = np.where(
        same_comp_t20i, param[1],
        np.where(same_comp_domestic, param[0], 1.0)
    )

    # Tournament recency boost (same domestic comp within 90 days)
    lookbacks_player_r['tournament_weight'] = np.where(recent_same_tournament, param[2], 1.0)

    # location country, host vs region vs baseline
    lookbacks_player_r['host_weight'] = np.where(host_same, param[3], 1.0)
    lookbacks_player_r['region_weight'] = np.where(region_same, param[4], 1.0)

    # Format (only when competition differs), ODI1 is baseline
    lookbacks_player_r['format_weight'] = np.where(
        prior_format_T20, param[5],
        np.where(prior_format_ODI2, param[6], 1.0)
    )

    # Final location weight
    lookbacks_player_r['location_weight'] = (
        lookbacks_player_r['comp_weight'] *
        lookbacks_player_r['tournament_weight'] *
        lookbacks_player_r['host_weight'] *
        lookbacks_player_r['region_weight'] *
        lookbacks_player_r['format_weight']
    )

    # Total weight with recency
    lookbacks_player_r['weight'] = lookbacks_player_r['recency_weight'] * lookbacks_player_r['location_weight']

    # Apply weights for rating
    lookbacks_player_r['weight_runs'] = lookbacks_player_r['weight'] * lookbacks_player_r['runs_2']
    lookbacks_player_r['weight_exprbat'] = lookbacks_player_r['weight'] * lookbacks_player_r['realexprbat_2']
    lookbacks_player_r['weight_ord_r'] = lookbacks_player_r['weight'] * lookbacks_player_r['ord_2']
    lookbacks_player_r['weight_balls_r'] = lookbacks_player_r['weight'] * lookbacks_player_r['balls_faced_2']

    # Aggregate to per-innings outputs
    ratings_player_r = pd.pivot_table(
        lookbacks_player_r,
        values=['weight', 'weight_runs', 'weight_exprbat', 'weight_ord_r',
                'balls_faced_2', 'runs_2', 'realexprbat_2', 'weight_balls_r'],
        index=['date', 'matchid', 'playerid', 'batsman', 'host', 'competition'],
        aggfunc={'weight': 'sum', 'weight_runs': 'sum', 'weight_exprbat': 'sum',
                 'balls_faced_2': 'sum', 'weight_ord_r': 'sum', 'runs_2': 'sum',
                 'realexprbat_2': 'sum', 'weight_balls_r': 'sum'}
    ).reset_index()

    ratings_player_r['run_rating'] = ratings_player_r['weight_runs'] / ratings_player_r['weight_exprbat']
    ratings_player_r['z_run_ratio'] = ratings_player_r['runs_2'] / ratings_player_r['realexprbat_2']
    ratings_player_r['ord_2_r'] = ratings_player_r['weight_ord_r'] / ratings_player_r['weight']

    return ratings_player_r, lookbacks_player_r





def buildRunRatingsMapPriority(param, lookbacks_player):
    """
    Priority mapping (no logs/exp).
    param (len=10) = [t, cd, ci, t20, odi2, odi1, dh, h, r, k]
      - Primary (pick first that matches): t -> cd -> ci -> t20 -> odi2 -> odi1
      - Host/region application:
        * For t, cd: host weight = dh if host differs else 1; region weight = 1
        * For ci, t20, odi2, odi1:
            host weight = h if same host else 1
            region weight = r if (host differs and region same) else 1
    """
    t, cd, ci, t20, odi2, odi1, dh, h, r, k = param

    lookbacks_player_r = lookbacks_player.copy()


    # Flags
    same_comp = (lookbacks_player_r['competition'] == lookbacks_player_r['competition_2'])
    same_comp_t20i = same_comp & (lookbacks_player_r['competition'] == 'T20I')
    same_comp_domestic = same_comp & (lookbacks_player_r['competition'] != 'T20I')
    # Only treat "same tournament" for non-T20I (prevents dh being applied for T20I)
    recent_same_tournament = same_comp_domestic & (lookbacks_player_r['days_ago'] < 90)
    host_same = (lookbacks_player_r['host'] == lookbacks_player_r['host_2'])
    region_same = (~host_same) & (lookbacks_player_r['host_region'] == lookbacks_player_r['host_region_2'])
    diff_comp = ~same_comp
    prior_is_t20 = diff_comp & (~lookbacks_player_r['competition_2'].isin(['ODI1', 'ODI2']))
    prior_is_odi2 = diff_comp & (lookbacks_player_r['competition_2'] == 'ODI2')

    # Primary case selection with priority (0=t, 1=cd, 2=ci, 3=t20, 4=odi2, 5=odi1)
    case_codes = np.select(
        condlist=[recent_same_tournament, same_comp_domestic, same_comp_t20i, prior_is_t20, prior_is_odi2],
        choicelist=[0, 1, 2, 3, 4],
        default=5
    )

    # Column 1: primary weight from first 6 param
    lookbacks_player_r['w_primary'] = np.select(
        condlist=[case_codes == 0, case_codes == 1, case_codes == 2, case_codes == 3, case_codes == 4],
        choicelist=[t, cd, ci, t20, odi2],
        default=odi1
    )

    # Column 2: host weight
    # For t/cd -> dh if host differs else 1; others -> h if host same else 1
    w_host_tc = np.where(host_same, 1.0, dh)
    w_host_std = np.where(host_same, h, 1.0)
    lookbacks_player_r['w_host'] = np.where((case_codes == 0) | (case_codes == 1), w_host_tc, w_host_std)

    # Column 3: region weight
    # For t/cd -> always 1; others -> r if (host differs and region same) else 1
    w_region_std = np.where((~host_same) & region_same, r, 1.0)
    lookbacks_player_r['w_region'] = np.where((case_codes == 0) | (case_codes == 1), 1.0, w_region_std)

    # Location weight
    lookbacks_player_r['location_weight'] = (
        lookbacks_player_r['w_primary'] * lookbacks_player_r['w_host'] * lookbacks_player_r['w_region']
    )
    # recency weight, the shape of k goes from small to large as career balls goes up (found using optimiser), I cap it at the average for all balls, but in reality it should get bigger than average above 750 balls
    lookbacks_player_r['k'] = k * np.where(
        lookbacks_player_r['balls_faced_career'] > 750,
        1,
        0.5 + (0.5 * (lookbacks_player_r['balls_faced_career']) / 750)
    )
    lookbacks_player_r['recency_weight'] = (1.0 - lookbacks_player_r['k']) ** lookbacks_player_r['days_ago']

    # # these are optional
    # lookbacks_player_r['location_weight_adjust'] = np.where(lookbacks_player_r['avg_ord'] <= 7, 1, np.where(lookbacks_player_r['avg_ord'] >= 9, 0.05 / 2.15, np.where(lookbacks_player_r['avg_ord'] >= 8, ((lookbacks_player_r['avg_ord'] - 8) * (0.05 / 2.15)) + ((1 - (lookbacks_player_r['avg_ord'] - 8)) * (0.81 / 2.15)), ((lookbacks_player_r['avg_ord'] - 7) * (0.81 / 2.15)) + ((1 - (lookbacks_player_r['avg_ord'] - 7)) * (1)))))
    # lookbacks_player_r['location_weight'] = ((lookbacks_player_r['location_weight'] - 1) * lookbacks_player_r['location_weight_adjust']) + 1

    # final weight
    lookbacks_player_r['weight'] = lookbacks_player_r['recency_weight'] * lookbacks_player_r['location_weight']

    # Apply weights for rating
    # lookbacks_player_r['weight_runs'] = lookbacks_player_r['weight'] * lookbacks_player_r['runs_2']
    # lookbacks_player_r['weight_exprbat'] = lookbacks_player_r['weight'] * lookbacks_player_r['realexprbat_2']
    # lookbacks_player_r['weight_ord_r'] = lookbacks_player_r['weight'] * lookbacks_player_r['ord_2']
    # lookbacks_player_r['weight_balls_r'] = lookbacks_player_r['weight'] * lookbacks_player_r['balls_faced_2']

    lookbacks_player_r['weight_runs'] = lookbacks_player_r['weight'] * lookbacks_player_r['runs_2']
    lookbacks_player_r['weight_exprbat'] = lookbacks_player_r['weight'] * lookbacks_player_r['adj_realexprbat']
    lookbacks_player_r['weight_exprbat_unadjusted'] = lookbacks_player_r['weight'] * lookbacks_player_r['realexprbat_2']
    lookbacks_player_r['weight_ord_r'] = lookbacks_player_r['weight'] * lookbacks_player_r['ord_2']
    lookbacks_player_r['weight_balls_r'] = lookbacks_player_r['weight'] * lookbacks_player_r['balls_faced_2']


    ratings = pd.pivot_table(
        lookbacks_player_r,
        values=['weight', 'weight_runs', 'weight_exprbat', 'weight_exprbat_unadjusted', 'weight_ord_r',
                'balls_faced_2', 'runs_2', 'realexprbat_2', 'weight_balls_r'],
        index=['date', 'matchid', 'playerid', 'batsman', 'host', 'competition'],
        aggfunc={'weight': 'sum', 'weight_runs': 'sum', 'weight_exprbat': 'sum', 'weight_exprbat_unadjusted': 'sum',
                 'balls_faced_2': 'sum', 'weight_ord_r': 'sum', 'runs_2': 'sum',
                 'realexprbat_2': 'sum', 'weight_balls_r': 'sum'}
    ).reset_index()

    ratings['run_rating_0'] = ratings['weight_runs'] / ratings['weight_exprbat_unadjusted']
    ratings['run_rating'] = ratings['weight_runs'] / ratings['weight_exprbat']
    ratings['z_run_ratio'] = ratings['runs_2'] / ratings['realexprbat_2']
    ratings['ord_2_r'] = ratings['weight_ord_r'] / ratings['weight']

    return ratings, lookbacks_player_r




def buildWktRatingsMapPriority(param, lookbacks_player):
    """
    Priority mapping (no logs/exp).
    param (len=10) = [t, cd, ci, t20, odi2, odi1, dh, h, r, k]
      - Primary (pick first that matches): t -> cd -> ci -> t20 -> odi2 -> odi1
      - Host/region application:
        * For t, cd: host weight = dh if host differs else 1; region weight = 1
        * For ci, t20, odi2, odi1:
            host weight = h if same host else 1
            region weight = r if (host differs and region same) else 1
    """
    t, cd, ci, t20, odi2, odi1, dh, h, r, k = param

    lookbacks_player_w = lookbacks_player.copy()

    # Flags
    same_comp = (lookbacks_player_w['competition'] == lookbacks_player_w['competition_2'])
    same_comp_t20i = same_comp & (lookbacks_player_w['competition'] == 'T20I')
    same_comp_domestic = same_comp & (lookbacks_player_w['competition'] != 'T20I')
    # Only treat "same tournament" for non-T20I (prevents dh being applied for T20I)
    recent_same_tournament = same_comp_domestic & (lookbacks_player_w['days_ago'] < 90)
    host_same = (lookbacks_player_w['host'] == lookbacks_player_w['host_2'])
    region_same = (~host_same) & (lookbacks_player_w['host_region'] == lookbacks_player_w['host_region_2'])
    diff_comp = ~same_comp
    prior_is_t20 = diff_comp & (~lookbacks_player_w['competition_2'].isin(['ODI1', 'ODI2']))
    prior_is_odi2 = diff_comp & (lookbacks_player_w['competition_2'] == 'ODI2')

    # Primary case selection with priority (0=t, 1=cd, 2=ci, 3=t20, 4=odi2, 5=odi1)
    case_codes = np.select(
        condlist=[recent_same_tournament, same_comp_domestic, same_comp_t20i, prior_is_t20, prior_is_odi2],
        choicelist=[0, 1, 2, 3, 4],
        default=5
    )

    # Column 1: primary weight from first 6 param
    lookbacks_player_w['w_primary'] = np.select(
        condlist=[case_codes == 0, case_codes == 1, case_codes == 2, case_codes == 3, case_codes == 4],
        choicelist=[t, cd, ci, t20, odi2],
        default=odi1
    )

    # Column 2: host weight
    # For t/cd -> dh if host differs else 1; others -> h if host same else 1
    w_host_tc = np.where(host_same, 1.0, dh)
    w_host_std = np.where(host_same, h, 1.0)
    lookbacks_player_w['w_host'] = np.where((case_codes == 0) | (case_codes == 1), w_host_tc, w_host_std)

    # Column 3: region weight
    # For t/cd -> always 1; others -> r if (host differs and region same) else 1
    w_region_std = np.where((~host_same) & region_same, r, 1.0)
    lookbacks_player_w['w_region'] = np.where((case_codes == 0) | (case_codes == 1), 1.0, w_region_std)

    # Location and recency weights
    lookbacks_player_w['location_weight'] = (
        lookbacks_player_w['w_primary'] * lookbacks_player_w['w_host'] * lookbacks_player_w['w_region']
    )

    # the shape of k goes from small to large as career balls goes up (found using optimiser), I cap it at the average for all balls, but in reality it should get bigger than average above 750 balls
    lookbacks_player_w['k'] = k * np.where(
        lookbacks_player_w['balls_faced_career'] > 750,
        1,
        0.5 + (0.5 * (lookbacks_player_w['balls_faced_career']) / 750)
    )
    lookbacks_player_w['recency_weight'] = (1.0 - lookbacks_player_w['k']) ** lookbacks_player_w['days_ago']

    # # these are optional
    # lookbacks_player_w['location_weight_adjust'] = np.where(lookbacks_player_w['avg_ord'] <= 7, 1, np.where(lookbacks_player_w['avg_ord'] >= 9, 0.05 / 2.15, np.where(lookbacks_player_w['avg_ord'] >= 8, ((lookbacks_player_w['avg_ord'] - 8) * (0.05 / 2.15)) + ((1 - (lookbacks_player_w['avg_ord'] - 8)) * (0.81 / 2.15)), ((lookbacks_player_w['avg_ord'] - 7) * (0.81 / 2.15)) + ((1 - (lookbacks_player_w['avg_ord'] - 7)) * (1)))))
    # lookbacks_player_w['location_weight'] = ((lookbacks_player_w['location_weight'] - 1) * lookbacks_player_w['location_weight_adjust']) + 1

    # final weight
    lookbacks_player_w['weight'] = lookbacks_player_w['recency_weight'] * lookbacks_player_w['location_weight']

    # Apply weights for rating
    # lookbacks_player_w['weight_wkt'] = lookbacks_player_w['weight'] * lookbacks_player_w['wkt_2']
    # lookbacks_player_w['weight_expwbat'] = lookbacks_player_w['weight'] * lookbacks_player_w['realexpwbat_2']
    # lookbacks_player_w['weight_ord_w'] = lookbacks_player_w['weight'] * lookbacks_player_w['ord_2']
    # lookbacks_player_w['weight_balls_w'] = lookbacks_player_w['weight'] * lookbacks_player_w['balls_faced_2']

    lookbacks_player_w['weight_wkt'] = lookbacks_player_w['weight'] * lookbacks_player_w['wkt_2']
    lookbacks_player_w['weight_expwbat'] = lookbacks_player_w['weight'] * lookbacks_player_w['adj_realexpwbat']
    lookbacks_player_w['weight_expwbat_unadjusted'] = lookbacks_player_w['weight'] * lookbacks_player_w['realexpwbat_2']
    lookbacks_player_w['weight_ord_w'] = lookbacks_player_w['weight'] * lookbacks_player_w['ord_2']
    lookbacks_player_w['weight_balls_w'] = lookbacks_player_w['weight'] * lookbacks_player_w['balls_faced_2']

    ratings = pd.pivot_table(
        lookbacks_player_w,
        values=['weight', 'weight_wkt', 'weight_expwbat', 'weight_expwbat_unadjusted', 'weight_ord_w',
                'balls_faced_2', 'wkt_2', 'realexpwbat_2', 'weight_balls_w'],
        index=['date', 'matchid', 'playerid', 'batsman', 'host', 'competition'],
        aggfunc={'weight': 'sum', 'weight_wkt': 'sum', 'weight_expwbat': 'sum', 'weight_expwbat_unadjusted': 'sum',
                 'balls_faced_2': 'sum', 'weight_ord_w': 'sum', 'wkt_2': 'sum',
                 'realexpwbat_2': 'sum', 'weight_balls_w': 'sum'}
    ).reset_index()

    ratings['wkt_rating_0'] = ratings['weight_wkt'] / ratings['weight_expwbat_unadjusted']
    ratings['wkt_rating'] = ratings['weight_wkt'] / ratings['weight_expwbat']
    ratings['z_wkt_ratio'] = ratings['wkt_2'] / ratings['realexpwbat_2']
    ratings['ord_2_w'] = ratings['weight_ord_w'] / ratings['weight']

    return ratings, lookbacks_player_w





def buildWktRatingsOriginal(param, lookbacks_player):
    k, ci, cd, h, r, t20, odi1, odi2, f = param[0], param[1], param[2], param[3], param[4], param[5], param[6], param[7], param[8]

    lookbacks_player_w = lookbacks_player.copy()

    lookbacks_player_w['k'] = k * np.where(lookbacks_player_w['balls_faced_career'] > 750, 1, 0.5 + (0.5 * (lookbacks_player_w['balls_faced_career']) / 750)) #the shape of k goes from small to large as career balls goes up (found using optimiser), I cap it at the average for all balls, but inreality it should get bigger than average above 750 balls, but I'm wary of making it even more form dependant)
    lookbacks_player_w['recency_weight'] = ((1 - lookbacks_player_w['k']) ** lookbacks_player_w['days_ago'])

    # the first weight looks at format, the 2nd at comp, if it's the same comp and it's an international it gets CI, if it's the same domestic comp it gets CD, if not the same comp just gets 1
    lookbacks_player_w['format_enc'] = np.where(lookbacks_player_w['competition_2'] == 'ODI1', odi1,
                                                np.where(lookbacks_player_w['competition_2'] == 'ODI2', odi2, t20))
    # same current tournament
    lookbacks_player_w['comp_enc'] = np.where(
        (lookbacks_player_w['days_ago'] < 90) &
        (lookbacks_player_w['competition'] == lookbacks_player_w['competition_2']) &
        (lookbacks_player_w['competition'] != 'T20I'),
        f,
        1
    )

    lookbacks_player_w['comp_enc'] = np.where(lookbacks_player_w['competition'] == lookbacks_player_w['competition_2'], np.where(lookbacks_player_w['competition'] == 'T20I', ci, cd), 1)
    lookbacks_player_w['host_enc'] = np.where(lookbacks_player_w['host'] == lookbacks_player_w['host_2'], h, 1)
    # for region, if host already the same leave it to avoid double weighting, then if not same host but same region give r, else give 1
    lookbacks_player_w['host_region_enc'] = np.where(lookbacks_player_w['host'] == lookbacks_player_w['host_2'], 1,
                                                         np.where(lookbacks_player_w['host_region'] == lookbacks_player_w['host_region_2'], r, 1))

    # multiply the weights
    lookbacks_player_w['location_weight'] = lookbacks_player_w['comp_enc'] * lookbacks_player_w['host_enc'] * lookbacks_player_w['host_region_enc'] #* lookbacks_player_w['home_away_enc'])
    lookbacks_player_w['location_weight_adjust'] = np.where(lookbacks_player_w['avg_ord'] <= 7, 1, np.where(lookbacks_player_w['avg_ord'] >= 9, 0.05 / 2.15, np.where(lookbacks_player_w['avg_ord'] >= 8, ((lookbacks_player_w['avg_ord'] - 8) * (0.05 / 2.15)) + ((1 - (lookbacks_player_w['avg_ord'] - 8)) * (0.81 / 2.15)), ((lookbacks_player_w['avg_ord'] - 7) * (0.81 / 2.15)) + ((1 - (lookbacks_player_w['avg_ord'] - 7)) * (1)))))
    lookbacks_player_w['location_weight'] = ((lookbacks_player_w['location_weight'] - 1) * lookbacks_player_w['location_weight_adjust']) + 1
    lookbacks_player_w['weight'] = lookbacks_player_w['recency_weight'] * lookbacks_player_w['location_weight']



    # Apply weights to components used in rating
    lookbacks_player_w['weight_wkt'] = lookbacks_player_w['weight'] * lookbacks_player_w['wkt_2']
    lookbacks_player_w['weight_expwbat'] = lookbacks_player_w['weight'] * lookbacks_player_w['realexpwbat_2']
    lookbacks_player_w['weight_ord_w'] = lookbacks_player_w['weight'] * lookbacks_player_w['ord_2']
    lookbacks_player_w['weight_balls_w'] = lookbacks_player_w['weight'] * lookbacks_player_w['balls_faced_2']
    # create a pivot to sum the weights then divide for the rating
    ratings = pd.pivot_table(lookbacks_player_w,
                                      values=['weight', 'weight_wkt', 'weight_expwbat', 'weight_ord_w',
                                              'balls_faced_2', 'wkt_2', 'realexpwbat_2', 'weight_balls_w'],
                                      index=['date', 'matchid', 'playerid', 'batsman', 'host', 'competition'],
                                      aggfunc={'weight': 'sum', 'weight_wkt': 'sum', 'weight_expwbat': 'sum',
                                               'balls_faced_2': 'sum', 'weight_ord_w': 'sum', 'wkt_2': 'sum',
                                               'realexpwbat_2': 'sum', 'weight_balls_w': 'sum'})
    ratings['wkt_rating'] = ratings['weight_wkt'] / ratings['weight_expwbat']
    ratings['z_wkt_ratio'] = ratings['wkt_2'] / ratings['realexpwbat_2']  # this is the actual ratio over the period of balls the rating is based on
    ratings['ord_2_w'] = ratings['weight_ord_w'] / ratings['weight']
    ratings = ratings.reset_index()

    return ratings, lookbacks_player_w
