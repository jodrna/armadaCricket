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
    Priority mapping.
    param = [t, cd, ci, t20, odi2, odi1, dh, h, r, k]

    Step 1:
      1. Same tournament < 90 days
      2. Same tournament >= 90 days
      3. Same competition and T20I/WT20I
      4. Same format T20
      5. ODI2
      6. ODI1

    Step 2:
      - bins 1/2: same host or different host
      - bins 3/4/5/6: same host, same region, or different host
    """
    t, cd, ci, t20, odi2, odi1, dh, h, r, k = param

    lookbacks_player_r = lookbacks_player.copy()

    same_comp = lookbacks_player_r['competition'] == lookbacks_player_r['competition_2']
    intl_t20 = lookbacks_player_r['competition'].isin(['T20I', 'WT20I'])

    same_tournament = same_comp & (~intl_t20)
    same_tournament_recent = same_tournament & (lookbacks_player_r['days_ago'] < 90)
    same_tournament_old = same_tournament & (lookbacks_player_r['days_ago'] >= 90)
    same_comp_t20i = same_comp & intl_t20

    host_same = lookbacks_player_r['host'] == lookbacks_player_r['host_2']
    region_same = (~host_same) & (lookbacks_player_r['host_region'] == lookbacks_player_r['host_region_2'])

    diff_comp = ~same_comp
    prior_is_t20 = diff_comp & (~lookbacks_player_r['competition_2'].isin(['ODI1', 'ODI2']))
    prior_is_odi2 = diff_comp & (lookbacks_player_r['competition_2'] == 'ODI2')
    prior_is_odi1 = diff_comp & (lookbacks_player_r['competition_2'] == 'ODI1')

    case_codes = np.select(
        condlist=[
            same_tournament_recent,
            same_tournament_old,
            same_comp_t20i,
            prior_is_t20,
            prior_is_odi2,
            prior_is_odi1
        ],
        choicelist=[0, 1, 2, 3, 4, 5],
        default=5
    )

    lookbacks_player_r['w_primary'] = np.select(
        condlist=[
            case_codes == 0,
            case_codes == 1,
            case_codes == 2,
            case_codes == 3,
            case_codes == 4,
            case_codes == 5
        ],
        choicelist=[t, cd, ci, t20, odi2, odi1],
        default=odi1
    )

    w_host_tc = np.where(host_same, 1.0, dh)
    w_host_std = np.where(host_same, h, 1.0)
    lookbacks_player_r['w_host'] = np.where((case_codes == 0) | (case_codes == 1), w_host_tc, w_host_std)

    w_region_std = np.where(region_same, r, 1.0)
    lookbacks_player_r['w_region'] = np.where((case_codes == 0) | (case_codes == 1), 1.0, w_region_std)

    lookbacks_player_r['location_weight'] = lookbacks_player_r['w_primary'] * lookbacks_player_r['w_host'] * lookbacks_player_r['w_region']

    lookbacks_player_r['k'] = k * np.where(
        lookbacks_player_r['balls_faced_career'] > 750,
        1,
        0.5 + (0.5 * lookbacks_player_r['balls_faced_career'] / 750)
    )

    lookbacks_player_r['recency_weight'] = (1.0 - lookbacks_player_r['k']) ** lookbacks_player_r['days_ago']
    lookbacks_player_r['weight'] = lookbacks_player_r['recency_weight'] * lookbacks_player_r['location_weight']

    lookbacks_player_r['weight_runs'] = lookbacks_player_r['weight'] * lookbacks_player_r['runs_2']
    lookbacks_player_r['weight_exprbat'] = lookbacks_player_r['weight'] * lookbacks_player_r['adj_realexprbat']
    lookbacks_player_r['weight_exprbat_unadjusted'] = lookbacks_player_r['weight'] * lookbacks_player_r['realexprbat_2']
    lookbacks_player_r['weight_ord_r'] = lookbacks_player_r['weight'] * lookbacks_player_r['ord_2']
    lookbacks_player_r['weight_balls_r'] = lookbacks_player_r['weight'] * lookbacks_player_r['balls_faced_2']

    ratings = pd.pivot_table(
        lookbacks_player_r,
        values=[
            'weight',
            'weight_runs',
            'weight_exprbat',
            'weight_exprbat_unadjusted',
            'weight_ord_r',
            'balls_faced_2',
            'runs_2',
            'realexprbat_2',
            'weight_balls_r'
        ],
        index=['date', 'matchid', 'playerid', 'batsman', 'host', 'competition'],
        aggfunc={
            'weight': 'sum',
            'weight_runs': 'sum',
            'weight_exprbat': 'sum',
            'weight_exprbat_unadjusted': 'sum',
            'balls_faced_2': 'sum',
            'weight_ord_r': 'sum',
            'runs_2': 'sum',
            'realexprbat_2': 'sum',
            'weight_balls_r': 'sum'
        }
    ).reset_index()

    ratings['run_rating_0'] = ratings['weight_runs'] / ratings['weight_exprbat_unadjusted']
    ratings['run_rating'] = ratings['weight_runs'] / ratings['weight_exprbat']
    ratings['z_run_ratio'] = ratings['runs_2'] / ratings['realexprbat_2']
    ratings['ord_2_r'] = ratings['weight_ord_r'] / ratings['weight']

    return ratings, lookbacks_player_r




def buildWktRatings(param, lookbacks_player):
    """
    Priority mapping.
    param = [t, cd, ci, t20, odi2, odi1, dh, h, r, k]
    """
    t, cd, ci, t20, odi2, odi1, dh, h, r, k = param

    lookbacks_player_w = lookbacks_player.copy()

    same_comp = lookbacks_player_w['competition'] == lookbacks_player_w['competition_2']
    intl_t20 = lookbacks_player_w['competition'].isin(['T20I', 'WT20I'])

    same_tournament = same_comp & (~intl_t20)
    same_tournament_recent = same_tournament & (lookbacks_player_w['days_ago'] < 90)
    same_tournament_old = same_tournament & (lookbacks_player_w['days_ago'] >= 90)
    same_comp_t20i = same_comp & intl_t20

    host_same = lookbacks_player_w['host'] == lookbacks_player_w['host_2']
    region_same = (~host_same) & (lookbacks_player_w['host_region'] == lookbacks_player_w['host_region_2'])

    diff_comp = ~same_comp
    prior_is_t20 = diff_comp & (~lookbacks_player_w['competition_2'].isin(['ODI1', 'ODI2']))
    prior_is_odi2 = diff_comp & (lookbacks_player_w['competition_2'] == 'ODI2')
    prior_is_odi1 = diff_comp & (lookbacks_player_w['competition_2'] == 'ODI1')

    case_codes = np.select(
        condlist=[
            same_tournament_recent,
            same_tournament_old,
            same_comp_t20i,
            prior_is_t20,
            prior_is_odi2,
            prior_is_odi1
        ],
        choicelist=[0, 1, 2, 3, 4, 5],
        default=5
    )

    lookbacks_player_w['w_primary'] = np.select(
        condlist=[
            case_codes == 0,
            case_codes == 1,
            case_codes == 2,
            case_codes == 3,
            case_codes == 4,
            case_codes == 5
        ],
        choicelist=[t, cd, ci, t20, odi2, odi1],
        default=odi1
    )

    w_host_tc = np.where(host_same, 1.0, dh)
    w_host_std = np.where(host_same, h, 1.0)
    lookbacks_player_w['w_host'] = np.where((case_codes == 0) | (case_codes == 1), w_host_tc, w_host_std)

    w_region_std = np.where(region_same, r, 1.0)
    lookbacks_player_w['w_region'] = np.where((case_codes == 0) | (case_codes == 1), 1.0, w_region_std)

    lookbacks_player_w['location_weight'] = lookbacks_player_w['w_primary'] * lookbacks_player_w['w_host'] * lookbacks_player_w['w_region']

    lookbacks_player_w['k'] = k * np.where(
        lookbacks_player_w['balls_faced_career'] > 750,
        1,
        0.5 + (0.5 * lookbacks_player_w['balls_faced_career'] / 750)
    )

    lookbacks_player_w['recency_weight'] = (1.0 - lookbacks_player_w['k']) ** lookbacks_player_w['days_ago']
    lookbacks_player_w['weight'] = lookbacks_player_w['recency_weight'] * lookbacks_player_w['location_weight']

    lookbacks_player_w['weight_wkt'] = lookbacks_player_w['weight'] * lookbacks_player_w['wkt_2']
    lookbacks_player_w['weight_expwbat'] = lookbacks_player_w['weight'] * lookbacks_player_w['adj_realexpwbat']
    lookbacks_player_w['weight_expwbat_unadjusted'] = lookbacks_player_w['weight'] * lookbacks_player_w['realexpwbat_2']
    lookbacks_player_w['weight_ord_w'] = lookbacks_player_w['weight'] * lookbacks_player_w['ord_2']
    lookbacks_player_w['weight_balls_w'] = lookbacks_player_w['weight'] * lookbacks_player_w['balls_faced_2']

    ratings = pd.pivot_table(
        lookbacks_player_w,
        values=[
            'weight',
            'weight_wkt',
            'weight_expwbat',
            'weight_expwbat_unadjusted',
            'weight_ord_w',
            'balls_faced_2',
            'wkt_2',
            'realexpwbat_2',
            'weight_balls_w'
        ],
        index=['date', 'matchid', 'playerid', 'batsman', 'host', 'competition'],
        aggfunc={
            'weight': 'sum',
            'weight_wkt': 'sum',
            'weight_expwbat': 'sum',
            'weight_expwbat_unadjusted': 'sum',
            'balls_faced_2': 'sum',
            'weight_ord_w': 'sum',
            'wkt_2': 'sum',
            'realexpwbat_2': 'sum',
            'weight_balls_w': 'sum'
        }
    ).reset_index()

    ratings['wkt_rating_0'] = ratings['weight_wkt'] / ratings['weight_expwbat_unadjusted']
    ratings['wkt_rating'] = ratings['weight_wkt'] / ratings['weight_expwbat']
    ratings['z_wkt_ratio'] = ratings['wkt_2'] / ratings['realexpwbat_2']
    ratings['ord_2_w'] = ratings['weight_ord_w'] / ratings['weight']

    return ratings, lookbacks_player_w




def build_rating_debug_tables(debug_config, ratings, lookbacks_player_r, lookbacks_player_w):
    debug_model = debug_config['model']
    debug_type = debug_config['type']
    debug_batsman = debug_config['batsman']
    debug_host = debug_config['host']
    debug_competition = debug_config['comp']
    debug_matchid = debug_config['matchid']

    if debug_type == 'run':
        debug_lookbacks_source = lookbacks_player_r
        rating_col_0, rating_col, z_col = 'run_rating_0', 'run_rating', 'z_run_ratio'
        weight_balls_col, actual_col, expected_col = 'weight_balls_r', 'runs_2', 'realexprbat_2'
        weight_actual_col, weight_expected_col = 'weight_runs', 'weight_exprbat'

    elif debug_type == 'wkt':
        debug_lookbacks_source = lookbacks_player_w
        rating_col_0, rating_col, z_col = 'wkt_rating_0', 'wkt_rating', 'z_wkt_ratio'
        weight_balls_col, actual_col, expected_col = 'weight_balls_w', 'wkt_2', 'realexpwbat_2'
        weight_actual_col, weight_expected_col = 'weight_wkt', 'weight_expwbat'

    else:
        raise ValueError("debug_type must be either 'run' or 'wkt'")

    debug_rating = ratings[
        (ratings['batsman'] == debug_batsman) &
        (ratings['host'] == debug_host) &
        (ratings['competition'] == debug_competition) &
        (ratings['matchid'] == debug_matchid)
    ].copy()

    debug_lookbacks = debug_lookbacks_source[
        (debug_lookbacks_source['batsman'] == debug_batsman) &
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
        balls_faced=('balls_faced_2', 'sum'),
        runs=(actual_col, 'sum'),
        xruns=(expected_col, 'sum'),
        location_weight=('location_weight', 'mean'),
        recency_weight=('recency_weight', 'mean'),
        weight_balls=(weight_balls_col, 'sum'),
        weight_actual=(weight_actual_col, 'sum'),
        weight_expected=(weight_expected_col, 'sum'),
        rating_share=('rating_weight_pct', 'sum')
    ).reset_index()

    comp_summary['rating'] = comp_summary['runs'] / comp_summary['xruns']
    comp_summary['effective_multiplier'] = comp_summary['weight_balls'] / comp_summary['balls_faced']
    comp_summary['effective_balls'] = comp_summary['weight_balls'] * comp_summary['balls_faced'].sum() / comp_summary['weight_balls'].sum()
    comp_summary['weighted_rating'] = comp_summary['weight_actual'] / comp_summary['weight_expected']
    comp_summary = comp_summary.rename(columns={'competition_2': 'competition', 'host_2': 'host'})
    comp_summary = comp_summary.sort_values('rating_share', ascending=False).reset_index(drop=True)
    comp_summary = comp_summary[['competition', 'host', 'innings', 'balls_faced', 'runs', 'xruns', 'rating', 'location_weight', 'recency_weight', 'effective_multiplier', 'effective_balls', 'weighted_rating', 'rating_share']]
    comp_summary = comp_summary[~((comp_summary['rating_share'] < 0.01) & (comp_summary['balls_faced'] < 100))].reset_index(drop=True)

    debug_lookbacks['recency'] = pd.cut(
        debug_lookbacks['days_ago'],
        bins=[-1, 90, 180, 365, 730, np.inf],
        labels=['0-90', '91-180', '181-365', '1-2 years', '2+ years']
    )

    recency_summary = debug_lookbacks.groupby('recency', observed=False).agg(
        innings=('matchid_2', 'count'),
        balls_faced=('balls_faced_2', 'sum'),
        runs=(actual_col, 'sum'),
        xruns=(expected_col, 'sum'),
        location_weight=('location_weight', 'mean'),
        recency_weight=('recency_weight', 'mean'),
        weight_balls=(weight_balls_col, 'sum'),
        weight_actual=(weight_actual_col, 'sum'),
        weight_expected=(weight_expected_col, 'sum'),
        rating_share=('rating_weight_pct', 'sum')
    ).reset_index()

    recency_summary['rating'] = recency_summary['runs'] / recency_summary['xruns']
    recency_summary['effective_multiplier'] = recency_summary['weight_balls'] / recency_summary['balls_faced']
    recency_summary['effective_balls'] = recency_summary['weight_balls'] * recency_summary['balls_faced'].sum() / recency_summary['weight_balls'].sum()
    recency_summary['weighted_rating'] = recency_summary['weight_actual'] / recency_summary['weight_expected']
    recency_summary = recency_summary[['recency', 'innings', 'balls_faced', 'runs', 'xruns', 'rating', 'location_weight', 'recency_weight', 'effective_multiplier', 'effective_balls', 'weighted_rating', 'rating_share']]


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
    debug_batsman = debug_config['batsman']
    debug_host = debug_config['host']
    debug_competition = debug_config['comp']
    debug_matchid = debug_config['matchid']

    debug_mask = ((ratings['batsman'] == debug_batsman) & (ratings['competition'] == debug_competition) & (ratings['host'] == debug_host) & (ratings['matchid'] == debug_matchid))
    debug_row = ratings.loc[debug_mask, :].reset_index(drop=True)

    if len(debug_row) == 0:
        return {'type': debug_type, 'debug_row': debug_row, 'breakdown': pd.DataFrame(), 'factor_breakdown': pd.DataFrame()}

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
        return {'type': debug_type, 'debug_row': debug_row, 'breakdown': pd.DataFrame(), 'factor_breakdown': pd.DataFrame()}

    def category_contrib(X, params):
        contrib = X.mul(params, axis=1)

        nat_contrib = contrib.filter(like='wt20i_nat__').sum(axis=1)

        return pd.Series({
            'const': contrib['const'].mean(),
            'competition': contrib.filter(like='competition__').sum(axis=1).mean(),
            'nationality': nat_contrib.loc[nat_contrib.abs() > 1e-12].mean(),
            'age': contrib[['age_x', 'age_x^2']].sum(axis=1).mean(),
            'experience': contrib['experience'].mean(),
            'order': contrib[['order_x', 'order_x^2']].sum(axis=1).mean(),
            'oppo': contrib['oppo'].mean()
        })

    avg_contrib = category_contrib(train_X, params)

    breakdown = pd.DataFrame({
        'feature': debug_X.columns,
        'model_value': debug_X.iloc[0].to_numpy(),
        'coef': params.to_numpy()
    })

    breakdown['contrib'] = breakdown['model_value'] * breakdown['coef']
    breakdown['raw_value'] = breakdown['model_value']

    breakdown.loc[breakdown['feature'] == 'experience', 'raw_value'] = debug_row['balls_faced_career'].iloc[0]

    age_contrib = breakdown.loc[breakdown['feature'].isin(['age_x', 'age_x^2']), 'contrib'].sum()
    age_raw_value = debug_row['age'].iloc[0]
    age_model_value = breakdown.loc[breakdown['feature'] == 'age_x', 'model_value'].iloc[0]
    age_coef = age_contrib / age_model_value if age_model_value != 0 else np.nan

    age_row = pd.DataFrame([{
        'feature': 'age',
        'raw_value': age_raw_value,
        'model_value': age_raw_value,
        'coef': age_coef,
        'contrib': age_contrib
    }])

    ord_col = 'ord_r' if 'ord_r' in debug_row.columns else ('ord_w' if 'ord_w' in debug_row.columns else 'ord')

    order_contrib = breakdown.loc[breakdown['feature'].isin(['order_x', 'order_x^2']), 'contrib'].sum()
    order_raw_value = debug_row[ord_col].iloc[0]
    order_model_value = breakdown.loc[breakdown['feature'] == 'order_x', 'model_value'].iloc[0]
    order_coef = order_contrib / order_model_value if order_model_value != 0 else np.nan

    order_row = pd.DataFrame([{
        'feature': 'order',
        'raw_value': order_raw_value,
        'model_value': order_raw_value,
        'coef': order_coef,
        'contrib': order_contrib
    }])

    breakdown = breakdown.loc[~breakdown['feature'].isin(['age_x', 'age_x^2', 'order_x', 'order_x^2']), :].copy()
    breakdown = pd.concat([breakdown, age_row, order_row], axis=0, ignore_index=True)

    breakdown.loc[breakdown['feature'].str.startswith('competition__', na=False), 'feature'] = 'competition'
    breakdown.loc[breakdown['feature'].str.startswith('wt20i_nat__', na=False), 'feature'] = 'nationality'

    breakdown = breakdown.groupby('feature', as_index=False).agg({
        'raw_value': 'sum',
        'model_value': 'sum',
        'coef': 'sum',
        'contrib': 'sum'
    })

    breakdown['avg_contrib'] = breakdown['feature'].map(avg_contrib)
    breakdown['contrib_diff'] = breakdown['contrib'] - breakdown['avg_contrib']

    const = breakdown.loc[breakdown['feature'] == 'const', :].copy()

    breakdown = breakdown.loc[(breakdown['feature'] != 'const') & (breakdown['contrib'] != 0), :].copy()
    breakdown = breakdown.sort_values('contrib', key=lambda z: z.abs(), ascending=False)
    breakdown = pd.concat([const, breakdown], axis=0).reset_index(drop=True)

    breakdown['rolling_sum'] = breakdown['contrib'].cumsum()

    breakdown = breakdown.loc[:, ['feature', 'raw_value', 'model_value', 'coef', 'avg_contrib', 'contrib', 'contrib_diff', 'rolling_sum']]

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
