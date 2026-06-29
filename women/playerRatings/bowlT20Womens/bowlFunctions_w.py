import numpy as np
import pandas as pd
from numba import njit
from collections import defaultdict
from paths import PROJECT_ROOT

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


def buildRunRatings(param, lookbacks_player):
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
      - odi1_*: prior format ODI1 (first 36 overs)
      - odi2_*: prior format ODI2 (last 14 overs)
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
    lookbacks_player_r['location_weight'] = (
        lookbacks_player_r['primary_weight'] *
        lookbacks_player_r['host_enc'] *
        lookbacks_player_r['region_enc']
    )

    lookbacks_player_r['weight'] = lookbacks_player_r['location_weight'] * lookbacks_player_r['recency_weight']

    # weight the runs and expected runs vs bowler
    lookbacks_player_r['weight_runs'] = lookbacks_player_r['weight'] * lookbacks_player_r['runs_2']
    lookbacks_player_r['weight_exprbowl'] = lookbacks_player_r['weight'] * lookbacks_player_r['adj_realexprbowl']
    lookbacks_player_r['weight_exprbowl_unadjusted'] = lookbacks_player_r['weight'] * lookbacks_player_r['realexprbowl_2']
    lookbacks_player_r['weight_balls_r'] = lookbacks_player_r['weight'] * lookbacks_player_r['balls_bowled_2']


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



def buildWktRatings(param, lookbacks_player):
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
      - odi1_*: prior format ODI1 (first 36 overs)
      - odi2_*: prior format ODI2 (last 14 overs)
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
    lookbacks_player_w['location_weight'] = (
        lookbacks_player_w['primary_weight'] *
        lookbacks_player_w['host_enc'] *
        lookbacks_player_w['region_enc']
    )

    lookbacks_player_w['weight'] = lookbacks_player_w['location_weight'] * lookbacks_player_w['recency_weight']

    # weight the wkt and expected wkt vs bowler
    lookbacks_player_w['weight_wkt'] = lookbacks_player_w['weight'] * lookbacks_player_w['wkt_2']
    lookbacks_player_w['weight_expwbowl'] = lookbacks_player_w['weight'] * lookbacks_player_w['adj_realexpwbowl']
    lookbacks_player_w['weight_expwbowl_unadjusted'] = lookbacks_player_w['weight'] * lookbacks_player_w['realexpwbowl_2']
    lookbacks_player_w['weight_balls_w'] = lookbacks_player_w['weight'] * lookbacks_player_w['balls_bowled_2']

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




def build_rating_debug_tables(debug_config, ratings, lookbacks_player_r, lookbacks_player_w):
    debug_model = debug_config['model']
    debug_type = debug_config['type']
    debug_bowler = debug_config['bowler']
    debug_host = debug_config['host']
    debug_competition = debug_config['comp']
    debug_matchid = debug_config['matchid']

    if debug_type == 'run':
        debug_lookbacks_source = lookbacks_player_r
        rating_col_0, rating_col, z_col = 'run_rating_0', 'run_rating', 'z_run_ratio'
        weight_balls_col, actual_col, expected_col = 'weight_balls_r', 'runs_2', 'realexprbowl_2'
        weight_actual_col, weight_expected_col = 'weight_runs', 'weight_exprbowl'

    elif debug_type == 'wkt':
        debug_lookbacks_source = lookbacks_player_w
        rating_col_0, rating_col, z_col = 'wkt_rating_0', 'wkt_rating', 'z_wkt_ratio'
        weight_balls_col, actual_col, expected_col = 'weight_balls_w', 'wkt_2', 'realexpwbowl_2'
        weight_actual_col, weight_expected_col = 'weight_wkt', 'weight_expwbowl'

    else:
        return {
            'model': debug_model,
            'type': debug_type,
            'rating': pd.DataFrame(),
            'lookbacks': pd.DataFrame(),
            'comp_summary': pd.DataFrame(),
            'recency_summary': pd.DataFrame()
        }

    debug_rating = ratings[
        (ratings['bowler'] == debug_bowler) &
        (ratings['host'] == debug_host) &
        (ratings['competition'] == debug_competition) &
        (ratings['matchid'] == debug_matchid)
    ].copy()

    debug_lookbacks = debug_lookbacks_source[
        (debug_lookbacks_source['bowler'] == debug_bowler) &
        (debug_lookbacks_source['host'] == debug_host) &
        (debug_lookbacks_source['competition'] == debug_competition) &
        (debug_lookbacks_source['matchid'] == debug_matchid)
    ].copy()

    if len(debug_rating) == 0 or len(debug_lookbacks) == 0:
        return {
            'model': debug_model,
            'type': debug_type,
            'rating': debug_rating,
            'lookbacks': debug_lookbacks,
            'comp_summary': pd.DataFrame(),
            'recency_summary': pd.DataFrame()
        }

    debug_lookbacks['rating_weight_pct'] = debug_lookbacks[weight_expected_col] / debug_lookbacks[weight_expected_col].sum()

    comp_summary = debug_lookbacks.groupby(['competition_2', 'host_2'], dropna=False).agg(
        innings=('matchid_2', 'count'),
        balls_bowled=('balls_bowled_2', 'sum'),
        actual=(actual_col, 'sum'),
        expected=(expected_col, 'sum'),
        location_weight=('location_weight', 'mean'),
        recency_weight=('recency_weight', 'mean'),
        weight_balls=(weight_balls_col, 'sum'),
        weight_actual=(weight_actual_col, 'sum'),
        weight_expected=(weight_expected_col, 'sum'),
        rating_share=('rating_weight_pct', 'sum')
    ).reset_index()

    comp_summary['rating'] = comp_summary['actual'] / comp_summary['expected']
    comp_summary['effective_multiplier'] = comp_summary['weight_balls'] / comp_summary['balls_bowled']
    comp_summary['effective_balls'] = comp_summary['weight_balls'] * comp_summary['balls_bowled'].sum() / comp_summary['weight_balls'].sum()
    comp_summary['weighted_rating'] = comp_summary['weight_actual'] / comp_summary['weight_expected']
    comp_summary = comp_summary.rename(columns={'competition_2': 'competition', 'host_2': 'host'})
    comp_summary = comp_summary.sort_values('rating_share', ascending=False).reset_index(drop=True)
    comp_summary = comp_summary[['competition', 'host', 'innings', 'balls_bowled', 'actual', 'expected', 'rating', 'location_weight', 'recency_weight', 'effective_multiplier', 'effective_balls', 'weighted_rating', 'rating_share']]
    comp_summary = comp_summary[comp_summary['rating_share'] > 0.01].reset_index(drop=True)

    debug_lookbacks['recency'] = pd.cut(
        debug_lookbacks['days_ago'],
        bins=[-1, 90, 180, 365, 730, np.inf],
        labels=['0-90', '91-180', '181-365', '1-2 years', '2+ years']
    )

    recency_summary = debug_lookbacks.groupby('recency', observed=False).agg(
        innings=('matchid_2', 'count'),
        balls_bowled=('balls_bowled_2', 'sum'),
        actual=(actual_col, 'sum'),
        expected=(expected_col, 'sum'),
        location_weight=('location_weight', 'mean'),
        recency_weight=('recency_weight', 'mean'),
        weight_balls=(weight_balls_col, 'sum'),
        weight_actual=(weight_actual_col, 'sum'),
        weight_expected=(weight_expected_col, 'sum'),
        rating_share=('rating_weight_pct', 'sum')
    ).reset_index()

    recency_summary['rating'] = recency_summary['actual'] / recency_summary['expected']
    recency_summary['effective_multiplier'] = recency_summary['weight_balls'] / recency_summary['balls_bowled']
    recency_summary['effective_balls'] = recency_summary['weight_balls'] * recency_summary['balls_bowled'].sum() / recency_summary['weight_balls'].sum()
    recency_summary['weighted_rating'] = recency_summary['weight_actual'] / recency_summary['weight_expected']
    recency_summary = recency_summary[['recency', 'innings', 'balls_bowled', 'actual', 'expected', 'rating', 'location_weight', 'recency_weight', 'effective_multiplier', 'effective_balls', 'weighted_rating', 'rating_share']]
    recency_summary = recency_summary[~((recency_summary['rating_share'] < 0.01) & (recency_summary['balls_bowled'] < 100))].reset_index(drop=True)

    return {
        'model': debug_model,
        'type': debug_type,
        'rating': debug_rating,
        'lookbacks': debug_lookbacks,
        'comp_summary': comp_summary,
        'recency_summary': recency_summary
    }






def build_replacement_debug_tables(debug_config, ratings, X_run_r, X_wkt_r, X_run_train, X_wkt_train, run_params, wkt_params):
    debug_type = debug_config['type']
    debug_bowler = debug_config['bowler']
    debug_host = debug_config['host']
    debug_competition = debug_config['comp']
    debug_matchid = debug_config['matchid']

    debug_mask = (
        (ratings['bowler'] == debug_bowler) &
        (ratings['competition'] == debug_competition) &
        (ratings['host'] == debug_host) &
        (ratings['matchid'] == debug_matchid)
    )

    debug_row = ratings.loc[debug_mask, :].reset_index(drop=True)

    if len(debug_row) == 0:
        return {
            'type': debug_type,
            'debug_row': debug_row,
            'breakdown': pd.DataFrame(),
            'factor_breakdown': pd.DataFrame()
        }

    if debug_type == 'run':
        debug_X = X_run_r.loc[debug_mask, :].reset_index(drop=True)
        train_X = X_run_train
        params = run_params
        total_col = 'rep_run_ratio'
        factor_col = 'run_factor'
    elif debug_type == 'wkt':
        debug_X = X_wkt_r.loc[debug_mask, :].reset_index(drop=True)
        train_X = X_wkt_train
        params = wkt_params
        total_col = 'rep_wkt_ratio'
        factor_col = 'wkt_factor'
    else:
        return {
            'type': debug_type,
            'debug_row': debug_row,
            'breakdown': pd.DataFrame(),
            'factor_breakdown': pd.DataFrame()
        }

    def category_contrib(X, params):
        contrib = X.mul(params, axis=1)

        nationality_contrib = contrib.filter(like='wt20i_nat__').sum(axis=1)
        bowlertype_3_contrib = contrib.filter(like='bowlertype_3__').sum(axis=1)

        return pd.Series({
            'const': contrib['const'].mean(),
            'competition': contrib.filter(like='competition__').sum(axis=1).mean(),
            'bowlertype_3': bowlertype_3_contrib.mean(),
            'nationality': nationality_contrib.loc[nationality_contrib.abs() > 1e-12].mean(),
            'ballspermatch': contrib['ballspermatch'].mean(),
            'overseas_pct': contrib[['overseas_pct_x', 'overseas_pct_x^2']].sum(axis=1).mean(),
            'experience': contrib['experience'].mean()
        })

    avg_contrib = category_contrib(train_X, params)

    breakdown = pd.DataFrame({
        'feature': debug_X.columns,
        'model_value': debug_X.iloc[0].to_numpy(),
        'coef': params.to_numpy()
    })

    breakdown['contrib'] = breakdown['model_value'] * breakdown['coef']
    breakdown['raw_value'] = breakdown['model_value']

    breakdown.loc[breakdown['feature'] == 'experience', 'raw_value'] = debug_row['balls_bowled_career'].iloc[0]

    overseas_pct_contrib = breakdown.loc[
        breakdown['feature'].isin(['overseas_pct_x', 'overseas_pct_x^2']),
        'contrib'
    ].sum()

    overseas_pct_raw_value = debug_row['overseas_pct'].iloc[0]

    overseas_pct_model_value = breakdown.loc[
        breakdown['feature'] == 'overseas_pct_x',
        'model_value'
    ].iloc[0]

    overseas_pct_coef = overseas_pct_contrib / overseas_pct_model_value if overseas_pct_model_value != 0 else np.nan

    overseas_pct_row = pd.DataFrame([{
        'feature': 'overseas_pct',
        'raw_value': overseas_pct_raw_value,
        'model_value': overseas_pct_raw_value,
        'coef': overseas_pct_coef,
        'contrib': overseas_pct_contrib
    }])

    breakdown = breakdown.loc[
        ~breakdown['feature'].isin(['overseas_pct_x', 'overseas_pct_x^2']),
        :
    ].copy()

    breakdown = pd.concat([breakdown, overseas_pct_row], axis=0, ignore_index=True)

    breakdown.loc[
        breakdown['feature'].str.startswith('competition__', na=False),
        'feature'
    ] = 'competition'

    breakdown.loc[
        breakdown['feature'].str.startswith('bowlertype_3__', na=False),
        'feature'
    ] = 'bowlertype_3'

    breakdown.loc[
        breakdown['feature'].str.startswith('wt20i_nat__', na=False),
        'feature'
    ] = 'nationality'

    breakdown = breakdown.groupby('feature', as_index=False).agg({
        'raw_value': 'sum',
        'model_value': 'sum',
        'coef': 'sum',
        'contrib': 'sum'
    })

    breakdown['avg_contrib'] = breakdown['feature'].map(avg_contrib)
    breakdown['contrib_diff'] = breakdown['contrib'] - breakdown['avg_contrib']

    const = breakdown.loc[breakdown['feature'] == 'const', :].copy()

    breakdown = breakdown.loc[(breakdown['feature'] != 'const'), :].copy()

    breakdown = breakdown.sort_values('contrib', key=lambda z: z.abs(), ascending=False)
    breakdown = pd.concat([const, breakdown], axis=0).reset_index(drop=True)

    breakdown['rolling_sum'] = breakdown['contrib'].cumsum()

    breakdown = breakdown.loc[:, [
        'feature',
        'raw_value',
        'model_value',
        'coef',
        'avg_contrib',
        'contrib',
        'contrib_diff',
        'rolling_sum'
    ]]

    factor_breakdown = pd.DataFrame()

    if factor_col in debug_row.columns:
        rep_value = debug_row[total_col].iloc[0]
        factor_value = debug_row[factor_col].iloc[0]

        factor_breakdown = pd.DataFrame([{
            'rep_value': rep_value / factor_value,
            factor_col: factor_value,
            'final_rep_value': rep_value
        }])

    return {
        'type': debug_type,
        'debug_row': debug_row,
        'breakdown': breakdown,
        'factor_breakdown': factor_breakdown,
        'total_col': total_col
    }



