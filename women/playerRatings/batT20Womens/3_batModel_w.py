import pandas as pd
import numpy as np
from batFunctions_w import buildRunRatings, buildWktRatings, build_rating_debug_tables
from paths import PROJECT_ROOT
DEBUG_CONFIG = globals().get('DEBUG_CONFIG', None)
BAT_MODEL_DEBUG_TABLES = None


# -------------------------
# Known limitations
# -------------------------
# players who play their first 2 games on the same day (e.g. finals day) will have 2 ratings missing instead of 1
# players who only ever play ODI will appear in ratings (from dummy values) but have no t20 data to rate them
# players who only played ODI before their first t20 will have a rating for that first t20
# players who play ODI first but only in the opening overs will have a wicket rating but not a run rating
# outputs are longer than ratings_player_r because they include a player's first innings (assigned rating of 1)


# -------------------------
# Imports
# -------------------------
bat_data = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/data/batDataCombinedClean_w.csv', parse_dates=['date', 'dob'])
n2h_factors = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/auxiliaries/batN2HFactors_w.csv')[['nationality', 'host_2', 'host', 'run_factor', 'wkt_factor']]
n2h_grad = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/auxiliaries/batN2HFactorsGradient_w.csv').rename(columns={'balls_faced_host_mean_y': 'balls_faced_host'})
bat_weightings = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/auxiliaries/batWeightings_w.csv')


# -------------------------
# Test one batsman
# -------------------------
# bat_data = bat_data[(bat_data['batsman'] == 'Alana King')]
# bat_data = bat_data[(bat_data['playerid'] == 489889)]


# -------------------------
# Basic preprocessing
# -------------------------
bat_data['competition'] = np.where(bat_data['competition'] == 'WODI', np.where(bat_data['ballsremaining'] < 84, 'ODI2', 'ODI1'), bat_data['competition'])

bat_data['format'] = bat_data['format'].fillna('t20')

bat_data = bat_data.merge(bat_weightings, on='balls_faced_career', how='left')
bat_data['runs_weight_curve'] = bat_data['runs_weight_curve'].fillna(1)
bat_data['wkts_weight_curve'] = bat_data['wkts_weight_curve'].fillna(1)


# -------------------------
# Build innings table
# -------------------------
innings_info = bat_data.loc[:, ['date', 'matchid', 'playerid', 'batsman', 'nationality', 'competition', 'host', 'host_region', 'balls_faced_career', 'balls_faced_host', 'H/A_competition', 'ord']].drop_duplicates(['matchid', 'playerid', 'date', 'host', 'competition'])

innings_perf = (
    pd.pivot_table(
        bat_data,
        values=['balls_faced', 'runs', 'realexprbat', 'wkt', 'realexpwbat'],
        index=['date', 'matchid', 'playerid', 'competition', 'host', 'ord'],
        aggfunc='sum'
    )
    .reset_index()
)

innings = innings_info.merge(
    innings_perf,
    how='left',
    left_on=['date', 'matchid', 'playerid', 'competition', 'host', 'ord'],
    right_on=['date', 'matchid', 'playerid', 'competition', 'host', 'ord']
)


# -------------------------
# Player lookbacks
# -------------------------
lookbacks_player = (
    innings.set_index('playerid')
    .merge(innings.set_index('playerid'), how='left', left_index=True, right_index=True, suffixes=('', '_2'))
    .reset_index()
)

lookbacks_player = lookbacks_player[lookbacks_player['date'] > lookbacks_player['date_2']]
lookbacks_player = lookbacks_player[~lookbacks_player['competition'].isin(['ODI1', 'ODI2'])]

lookbacks_player['date'] = pd.to_datetime(lookbacks_player['date'])
lookbacks_player['date_2'] = pd.to_datetime(lookbacks_player['date_2'])
lookbacks_player['days_ago'] = (lookbacks_player['date'] - lookbacks_player['date_2']).dt.days
lookbacks_player['balls_ago'] = lookbacks_player['balls_faced_career'] - lookbacks_player['balls_faced_career_2']

avg_ord = bat_data.groupby(['playerid', 'batsman'])['ord'].mean().reset_index()
avg_ord.rename(columns={'ord': 'avg_ord'}, inplace=True)
lookbacks_player = lookbacks_player.merge(avg_ord, on=('playerid', 'batsman'), how='left')





# -------------------------
# n2h adjustments
# -------------------------
lookbacks_player = lookbacks_player.merge(
    n2h_factors.loc[:, ['nationality', 'host', 'run_factor', 'wkt_factor']],
    how='left',
    on=['nationality', 'host']
)

n2h_factors_2 = n2h_factors.drop('host_2', axis=1).rename(columns={'host': 'host_2'})
lookbacks_player = lookbacks_player.merge(
    n2h_factors_2.loc[:, ['nationality', 'host_2', 'run_factor', 'wkt_factor']],
    how='left',
    on=['nationality', 'host_2'],
    suffixes=('', '_2')
)

home_pred = lookbacks_player['H/A_competition'] == 'Home'
home_hist = lookbacks_player['H/A_competition_2'] == 'Home'

m = lookbacks_player['run_factor'].isna()
lookbacks_player.loc[m, 'run_factor'] = np.where(home_pred[m], 1, 0.9882)

m = lookbacks_player['run_factor_2'].isna()
lookbacks_player.loc[m, 'run_factor_2'] = np.where(home_hist[m], 1, 0.9882)

m = lookbacks_player['wkt_factor'].isna()
lookbacks_player.loc[m, 'wkt_factor'] = np.where(home_pred[m], 1, 1.0146)

m = lookbacks_player['wkt_factor_2'].isna()
lookbacks_player.loc[m, 'wkt_factor_2'] = np.where(home_hist[m], 1, 1.0146)

lookbacks_player = lookbacks_player.merge(n2h_grad, on='balls_faced_host', how='left')
away_pred = lookbacks_player['H/A_competition'] == 'Away'
lookbacks_player.loc[away_pred, 'run_factor'] = lookbacks_player.loc[away_pred, 'run_factor'] * lookbacks_player.loc[away_pred, 'run_factor_smooth']
lookbacks_player.loc[away_pred, 'wkt_factor'] = lookbacks_player.loc[away_pred, 'wkt_factor'] * lookbacks_player.loc[away_pred, 'wkt_factor_smooth']

n2h_grad_2 = n2h_grad.rename(columns={'balls_faced_host': 'balls_faced_host_2', 'run_factor_smooth': 'run_factor_smooth_2', 'wkt_factor_smooth': 'wkt_factor_smooth_2'})
lookbacks_player = lookbacks_player.merge(n2h_grad_2, on='balls_faced_host_2', how='left')
away_hist = lookbacks_player['H/A_competition_2'] == 'Away'
lookbacks_player.loc[away_hist, 'run_factor_2'] = lookbacks_player.loc[away_hist, 'run_factor_2'] * lookbacks_player.loc[away_hist, 'run_factor_smooth_2']
lookbacks_player.loc[away_hist, 'wkt_factor_2'] = lookbacks_player.loc[away_hist, 'wkt_factor_2'] * lookbacks_player.loc[away_hist, 'wkt_factor_smooth_2']

lookbacks_player['adj_realexprbat'] = lookbacks_player['realexprbat_2'] / (lookbacks_player['run_factor'] / lookbacks_player['run_factor_2'])
lookbacks_player['adj_realexpwbat'] = lookbacks_player['realexpwbat_2'] / (lookbacks_player['wkt_factor'] / lookbacks_player['wkt_factor_2'])


# -------------------------
# Build outputs for jungle and rasoi
# -------------------------
for x in np.arange(0, 2, 1):
    if x == 0:
        model_name = 'jungle'
        param_r_dict = {
            't': 15.17348002,
            'cd': 8.89753,
            'ci': 12.88380525,
            't20': 6.307514689,
            'odi2': 1.692501798,
            'odi1': 1.000755457,
            'dh': 0.621652589,
            'h': 1.241297181,
            'r': 1.091979717,
            'k': 0.000496309
        }
        param_w_dict = {
            't': 4.293158205,
            'cd': 3.979704835,
            'ci': 2.20296017,
            't20': 1.870824639,
            'odi2': 1.518191934,
            'odi1': 1.071059076,
            'dh': 0.97542588,
            'h': 1.477165773,
            'r': 1,
            'k': 0.000499531
        }

    else:
        model_name = 'rasoi'
        param_r_dict = {
            't': 19.999999998,
            'cd': 12.594571572,
            'ci': 17.460781925,
            't20': 7.338651890,
            'odi2': 2.728047680,
            'odi1': 1.000000000,
            'dh': 0.800003439,
            'h': 1.199992058,
            'r': 1.050241154,
            'k': 0.001515314,
        }
        param_w_dict = {
            't': 13.393558084,
            'cd': 17.022545492,
            'ci': 10.585991537,
            't20': 5.242233553,
            'odi2': 2.434296227,
            'odi1': 1,
            'dh': 0.8,
            'h': 1.4,
            'r': 1.2,
            'k': 0.000649046,
        }

    param_r = list(param_r_dict.values())
    param_w = list(param_w_dict.values())

    # build the ratings from functions in batfunctions
    ratings_player_r, lookbacks_player_r = buildRunRatings(param_r, lookbacks_player)
    ratings_player_w, lookbacks_player_w = buildWktRatings(param_w, lookbacks_player)

    # use only t20 from now on
    bat_data_t20 = bat_data[bat_data['format'] == 't20'].copy()

    # drop duplicates, this happens when players have 2 games in 1 day, like t20 finals day, it causes duplicates down the line with 2 identical ratings on 1 day, we keep 1 rating because they're the same
    ratings_player_r = ratings_player_r.drop_duplicates(subset=['date', 'playerid', 'batsman', 'host', 'competition'])
    ratings_player_w = ratings_player_w.drop_duplicates(subset=['date', 'playerid', 'batsman', 'host', 'competition'])

    # merge the run and wkt ratings
    ratings_player = pd.merge(
        ratings_player_r.drop(labels=['realexprbat_2', 'runs_2', 'weight_exprbat', 'weight_runs'], axis=1),
        ratings_player_w.drop(labels=['realexpwbat_2', 'wkt_2', 'weight_expwbat', 'weight_wkt'], axis=1),
        how='left',
        on=['date', 'playerid', 'batsman', 'host', 'competition'],
        suffixes=('_r', '_w'))

    # merge the innings performance, we'll use later for error measurement and more
    innings_perf_out = (
        pd.pivot_table(
            bat_data_t20,
            values=['balls_faced', 'balls_faced_career', 'balls_faced_host', 'runs', 'wkt', 'realexprbat', 'realexpwbat', 'ord'],
            index=['date', 'playerid', 'matchid', 'batsman', 'host', 'competition'],
            aggfunc={
                'balls_faced': 'sum',
                'balls_faced_career': 'min',
                'balls_faced_host': 'min',
                'runs': 'sum',
                'wkt': 'sum',
                'realexprbat': 'sum',
                'realexpwbat': 'sum',
                'ord': 'mean'
            }).reset_index())
    innings_perf_out['i_run_ratio'] = innings_perf_out['runs'] / innings_perf_out['realexprbat']
    innings_perf_out['i_wkt_ratio'] = innings_perf_out['wkt'] / innings_perf_out['realexpwbat']

    ratings_info = bat_data_t20.loc[:, [
        'date', 'matchid', 'battingteam', 'playerid', 'batsman', 'age',
        'nationality', 'home_region', 'host', 'host_region', 'H/A_competition',
        'H/A_country', 'H/A_region', 'competition', 'overseas_pct', 'careerT20MatchNumber'
    ]].drop_duplicates(subset=['date', 'matchid', 'playerid', 'host', 'competition'])

    ratings = innings_perf_out.merge(
        ratings_info,
        how='left',
        on=['date', 'matchid', 'playerid', 'batsman', 'host', 'competition']
    )

    ratings = ratings.merge(
        ratings_player,
        how='left',
        on=['date', 'playerid', 'batsman', 'host', 'competition']
    )

    ratings = ratings[~ratings['competition'].isin(['ODI1', 'ODI2'])]

    ratings = ratings.loc[:, [
        'date', 'matchid', 'battingteam', 'playerid', 'batsman', 'age',
        'nationality', 'home_region', 'host', 'host_region', 'H/A_competition',
        'H/A_country', 'H/A_region', 'competition', 'careerT20MatchNumber',
        'balls_faced_career', 'balls_faced_host', 'overseas_pct', 'balls_faced_2_r',
        'ord_2_r', 'z_run_ratio', 'run_rating', 'run_rating_0', 'weight_balls_r', 'balls_faced_2_w',
        'ord_2_w', 'z_wkt_ratio', 'wkt_rating', 'wkt_rating_0', 'weight_balls_w', 'balls_faced',
        'ord', 'realexprbat', 'runs', 'i_run_ratio', 'realexpwbat', 'wkt', 'i_wkt_ratio'
    ]]

    ratings = ratings.rename(columns={
        'balls_faced_2_r': 'balls_faced_r',
        'ord_2_r': 'ord_r',
        'balls_faced_2_w': 'balls_faced_w',
        'ord_2_w': 'ord_w',
        'balls_faced': 'i_balls_faced',
        'ord': 'i_ord',
        'realexprbat': 'i_realexprbat',
        'runs': 'i_runs',
        'realexpwbat': 'i_realexpwbat',
        'wkt': 'i_wkt'
    })

    ratings['i_ord'] = ratings['i_ord'].round(0)

    ratings['run_rating_0'] = ratings['run_rating_0'].fillna(1)
    ratings['wkt_rating_0'] = ratings['wkt_rating_0'].fillna(1)
    ratings['run_rating'] = ratings['run_rating'].fillna(1)
    ratings['wkt_rating'] = ratings['wkt_rating'].fillna(1)
    ratings['balls_faced_r'] = ratings['balls_faced_r'].fillna(1)
    ratings['balls_faced_w'] = ratings['balls_faced_w'].fillna(1)
    ratings['ord_r'] = ratings['ord_r'].fillna(ratings['i_ord'])
    ratings['ord_w'] = ratings['ord_w'].fillna(ratings['i_ord'])


    # -------------------------
    # debug
    # -------------------------
    if DEBUG_CONFIG is not None and DEBUG_CONFIG['model'] == model_name:
        BAT_MODEL_DEBUG_TABLES = build_rating_debug_tables(
            DEBUG_CONFIG,
            ratings,
            lookbacks_player_r,
            lookbacks_player_w
        )

    # -------------------------
    # Recencies + export
    # -------------------------
    if x == 0:
        recencies_r = lookbacks_player_r[(lookbacks_player_r['competition'] == 'WT20I') & (lookbacks_player_r['host'] == 'West Indies') & (lookbacks_player_r['date'] == lookbacks_player_r['date'].max())].loc[:, ['playerid', 'matchid_2', 'recency_weight', 'balls_faced_2']]
        recencies_r['recency_weight_match_sum'] = recencies_r['recency_weight'] * recencies_r['balls_faced_2']
        recencies_t = pd.pivot_table(recencies_r, index=['playerid'], values=['recency_weight_match_sum'], aggfunc='sum').reset_index()
        recencies_r = recencies_r.merge(recencies_t, how='left', on=['playerid'])
        recencies_r['recency_weight_bbb_runs'] = recencies_r['recency_weight_match_sum_x'] / recencies_r['recency_weight_match_sum_y'] / recencies_r['balls_faced_2']


        recencies_w = lookbacks_player_w[(lookbacks_player_w['competition'] == 'WT20I') & (lookbacks_player_w['host'] == 'West Indies') & (lookbacks_player_r['date'] == lookbacks_player_r['date'].max())].loc[:, ['playerid', 'matchid_2', 'recency_weight', 'balls_faced_2']]
        recencies_w['recency_weight_match_sum'] = recencies_w['recency_weight'] * recencies_w['balls_faced_2']
        recencies_t = pd.pivot_table(recencies_w, index=['playerid'], values=['recency_weight_match_sum'], aggfunc='sum').reset_index()
        recencies_w = recencies_w.merge(recencies_t, how='left', on=['playerid'])
        recencies_w['recency_weight_bbb_wkt'] = recencies_w['recency_weight_match_sum_x'] / recencies_w['recency_weight_match_sum_y'] / recencies_w['balls_faced_2']

        recencies = pd.merge(recencies_r.loc[:, ['matchid_2', 'playerid', 'recency_weight_bbb_runs']], recencies_w.loc[:, ['matchid_2', 'playerid', 'recency_weight_bbb_wkt']], how='outer')
        recencies.to_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/outputs/batRecencies_w.csv', index=False)
        ratings.to_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/outputs/batRatingsJungle_w.csv', index=False)

    else:
        ratings.to_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/outputs/batRatingsRasoi_w.csv', index=False)

print(np.mean(ratings['wkt_rating']))
print(np.mean(ratings['run_rating']))

