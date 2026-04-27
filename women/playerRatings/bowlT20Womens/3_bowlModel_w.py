import pandas as pd
import numpy as np
from bowlFunctions_w import buildRunRatingsOriginal, buildWktRatingsOriginal
from paths import PROJECT_ROOT


# -------------------------
# Imports
# -------------------------
bowl_data = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/data/bowlDataCombinedClean_w.csv', parse_dates=['date', 'dob'])
bowl_weightings = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/auxiliaries/bowlWeightings_w.csv')
n2h_factors_seam = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/auxiliaries/bowlN2HFactorsSeam_w.csv')[['nationality', 'host_2', 'host', 'run_factor', 'wkt_factor']]
n2h_factors_spin = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/auxiliaries/bowlN2HFactorsSpin_w.csv')[['nationality', 'host_2', 'host', 'run_factor', 'wkt_factor']]


# -------------------------
# Test one bowler
# -------------------------
# bowl_data = bowl_data[(bowl_data['bowler'] == 'Alana King')]


# -------------------------
# Basic preprocessing
# -------------------------
bowl_data['competition'] = np.where(
    bowl_data['competition'].str.contains('ODI', na=False),
    np.where(bowl_data['ballsremaining'] < 84, 'ODI2', 'ODI1'),
    bowl_data['competition']
)

bowl_data['format'] = bowl_data['format'].fillna('t20')

bowl_data = bowl_data.merge(bowl_weightings, on='balls_bowled_career', how='left')
bowl_data['runs_weight_curve'] = bowl_data['runs_weight_curve'].fillna(1)
bowl_data['wkts_weight_curve'] = bowl_data['wkts_weight_curve'].fillna(1)


# -------------------------
# Build innings table
# -------------------------
innings_info = bowl_data.loc[:, [
    'date', 'matchid', 'playerid', 'bowler', 'bowlertype_1', 'bowlertype_2',
    'bowler_arm', 'bowler_pace', 'nationality', 'competition', 'host',
    'host_region', 'balls_bowled_career', 'balls_bowled_host'
]].drop_duplicates(['matchid', 'playerid', 'date', 'host', 'competition', 'bowlertype_2'])

innings_perf = (
    pd.pivot_table(
        bowl_data,
        values=['balls_bowled', 'runs', 'realexprbowl', 'wkt', 'realexpwbowl', 'ord'],
        index=['date', 'matchid', 'playerid', 'competition', 'host'],
        aggfunc='sum'
    )
    .reset_index()
)

innings = innings_info.merge(
    innings_perf,
    how='left',
    on=['date', 'matchid', 'playerid', 'competition', 'host']
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
lookbacks_player['balls_ago'] = lookbacks_player['balls_bowled_career'] - lookbacks_player['balls_bowled_career_2']

types = pd.pivot_table(lookbacks_player, values='date', index='bowlertype_2', aggfunc='count')


# -------------------------
# n2h adjustments
# -------------------------
n2h_factors_seam['bowlertype_1'] = 'seam'
n2h_factors_spin['bowlertype_1'] = 'spin'
n2h_factors = pd.concat([n2h_factors_seam, n2h_factors_spin], axis=0)

lookbacks_player = lookbacks_player.merge(
    n2h_factors.loc[:, ['nationality', 'bowlertype_1', 'host', 'run_factor', 'wkt_factor']],
    how='left',
    on=['nationality', 'bowlertype_1', 'host']
)

n2h_factors_2 = n2h_factors.drop('host_2', axis=1).rename(columns={'host': 'host_2'})

lookbacks_player = lookbacks_player.merge(
    n2h_factors_2.loc[:, ['nationality', 'bowlertype_1', 'host_2', 'run_factor', 'wkt_factor']],
    how='left',
    on=['nationality', 'bowlertype_1', 'host_2'],
    suffixes=('', '_2')
)

cols = ['run_factor', 'run_factor_2', 'wkt_factor', 'wkt_factor_2']
lookbacks_player[cols] = lookbacks_player[cols].fillna(1)

lookbacks_player['adj_realexprbowl'] = lookbacks_player['realexprbowl_2'] / (lookbacks_player['run_factor'] / lookbacks_player['run_factor_2'])
lookbacks_player['adj_realexpwbowl'] = lookbacks_player['realexpwbowl_2'] / (lookbacks_player['wkt_factor'] / lookbacks_player['wkt_factor_2'])


# -------------------------
# Build outputs twice
# -------------------------
for x in np.arange(0, 2, 1):

    if x == 0:
        param_r = {
            'k_sm': 0.000512352,
            'c_sm': 19.89691679,
            'h_sm': 2.303435144,
            'r_sm': 2.538328874,
            't20_sm': 16.57185942,
            'odi1_sm': 1.057447715,
            'odi2_sm': 10.49537199,
            'k_s': 0.001338298,
            'c_s': 8.223711839,
            'h_s': 1.534319621,
            'r_s': 1.000038601,
            't20_s': 13.68554581,
            'odi1_s': 7.140800727,
            'odi2_s': 4.46645703
        }

        param_w = {
            'k_sm': 1.21E-09,
            'c_sm': 3.192616928,
            'h_sm': 1.485641469,
            'r_sm': 1,
            't20_sm': 2.583371789,
            'odi1_sm': 5.400937248,
            'odi2_sm': 2.239620195,
            'k_s': 1.16E-05,
            'c_s': 9.606481836,
            'h_s': 2.053202863,
            'r_s': 5.440643787,
            't20_s': 9.63349785,
            'odi1_s': 19.64893579,
            'odi2_s': 3.611793325
        }

    else:
        param_r = {
            'k_sm': 0.000840494,
            'c_sm': 19.65001961,
            'h_sm': 1.000000818,
            'r_sm': 1,
            't20_sm': 5.224663699,
            'odi1_sm': 1.355607939,
            'odi2_sm': 1.264666101,
            'k_s': 1.53E-06,
            'c_s': 19.99755548,
            'h_s': 1.000008368,
            'r_s': 2.347171475,
            't20_s': 1.159316676,
            'odi1_s': 1,
            'odi2_s': 3.677859499
        }

        param_w = {
            'k_sm': 0.000133246,
            'c_sm': 11.85188111,
            'h_sm': 1.244917215,
            'r_sm': 1,
            't20_sm': 5.239237345,
            'odi1_sm': 19.43121248,
            'odi2_sm': 1.01309911,
            'k_s': 1.91E-05,
            'c_s': 3.533558665,
            'h_s': 2.121826747,
            'r_s': 10.97941704,
            't20_s': 4.510981226,
            'odi1_s': 19.22741794,
            'odi2_s': 1.085666865
        }

    ratings_player_r, lookbacks_player_r = buildRunRatingsOriginal(param_r, lookbacks_player)
    ratings_player_w, lookbacks_player_w = buildWktRatingsOriginal(param_w, lookbacks_player)

    bowl_data_t20 = bowl_data[bowl_data['format'] == 't20'].copy()

    rating_key = ['date', 'playerid', 'bowler', 'host', 'competition']

    ratings_player_r = ratings_player_r.drop_duplicates(subset=rating_key)
    ratings_player_w = ratings_player_w.drop_duplicates(subset=rating_key)

    ratings_player = pd.merge(
        ratings_player_r.drop(labels=['realexprbowl_2', 'runs_2', 'weight_exprbowl', 'weight_runs'], axis=1),
        ratings_player_w.drop(labels=['realexpwbowl_2', 'wkt_2', 'weight_expwbowl', 'weight_wkt'], axis=1),
        how='left',
        on=rating_key,
        suffixes=('_r', '_w')
    )

    innings_perf_out = (
        pd.pivot_table(
            bowl_data_t20,
            values=['balls_bowled', 'balls_bowled_career', 'balls_bowled_host', 'runs', 'wkt', 'realexprbowl', 'realexpwbowl'],
            index=['date', 'matchid', 'playerid', 'bowler', 'host', 'competition'],
            aggfunc={
                'balls_bowled': 'sum',
                'balls_bowled_career': 'min',
                'balls_bowled_host': 'min',
                'runs': 'sum',
                'wkt': 'sum',
                'realexprbowl': 'sum',
                'realexpwbowl': 'sum'
            }
        )
        .reset_index()
    )

    innings_perf_out['i_run_ratio'] = innings_perf_out['runs'] / innings_perf_out['realexprbowl']
    innings_perf_out['i_wkt_ratio'] = innings_perf_out['wkt'] / innings_perf_out['realexpwbowl']

    ratings_info = bowl_data_t20.loc[:, [
        'date', 'matchid', 'battingteam', 'playerid', 'bowler', 'bowlertype_2',
        'bowler_arm', 'bowler_pace', 'bowler_level', 'ballspermatch', 'age',
        'nationality', 'home_region', 'host', 'host_region', 'H/A_competition',
        'H/A_country', 'H/A_region', 'competition', 'overseas_pct'
    ]].drop_duplicates(['date', 'matchid', 'playerid', 'bowler', 'host', 'competition'])

    ratings = innings_perf_out.merge(
        ratings_info,
        how='left',
        on=['date', 'matchid', 'playerid', 'bowler', 'host', 'competition']
    )

    ratings = ratings.merge(
        ratings_player,
        how='left',
        on=rating_key
    )

    ratings = ratings[~ratings['competition'].isin(['ODI1', 'ODI2'])]

    ratings = ratings.loc[:, [
        'date', 'matchid', 'battingteam', 'playerid', 'bowler', 'bowlertype_2',
        'bowler_arm', 'bowler_pace', 'bowler_level', 'ballspermatch', 'age',
        'nationality', 'home_region', 'host', 'host_region', 'H/A_competition',
        'H/A_country', 'H/A_region', 'competition', 'balls_bowled_career',
        'balls_bowled_host', 'overseas_pct', 'balls_bowled_2_r', 'ord_2_r',
        'z_run_ratio', 'run_rating_0', 'run_rating', 'balls_bowled_2_w',
        'ord_2_w', 'z_wkt_ratio', 'wkt_rating_0', 'wkt_rating', 'balls_bowled',
        'realexprbowl', 'runs', 'i_run_ratio', 'realexpwbowl', 'wkt', 'i_wkt_ratio'
    ]]

    ratings = ratings.rename(columns={
        'balls_bowled_2_r': 'balls_bowled_r',
        'ord_2_r': 'ord_r',
        'balls_bowled_2_w': 'balls_bowled_w',
        'ord_2_w': 'ord_w',
        'balls_bowled': 'i_balls_bowled',
        'realexprbowl': 'i_realexprbowl',
        'runs': 'i_runs',
        'realexpwbowl': 'i_realexpwbowl',
        'wkt': 'i_wkt'
    })

    ratings['run_rating_0'] = ratings['run_rating_0'].fillna(1)
    ratings['wkt_rating_0'] = ratings['wkt_rating_0'].fillna(1)
    ratings['run_rating'] = ratings['run_rating'].fillna(1)
    ratings['wkt_rating'] = ratings['wkt_rating'].fillna(1)


    if x == 0:
        recencies_r = lookbacks_player_r[((lookbacks_player_r['competition'] == 'WT20I') | (lookbacks_player_r['competition'] == 'tier_2')) & (lookbacks_player_r['host'] == 'West Indies') & (lookbacks_player_r['date'] == lookbacks_player_r['date'].max())].loc[:, ['playerid', 'matchid_2', 'recency_weight', 'balls_bowled_2']]
        recencies_r['recency_weight_match_sum'] = recencies_r['recency_weight'] * recencies_r['balls_bowled_2']
        recencies_t = pd.pivot_table(recencies_r, index=['playerid'], values=['recency_weight_match_sum'], aggfunc='sum').reset_index()
        recencies_r = recencies_r.merge(recencies_t, how='left', on=['playerid'])
        recencies_r['recency_weight_bbb_runs'] = recencies_r['recency_weight_match_sum_x'] / recencies_r['recency_weight_match_sum_y'] / recencies_r['balls_bowled_2']

        recencies_w = lookbacks_player_w[((lookbacks_player_w['competition'] == 'WT20I') | (lookbacks_player_w['competition'] == 'tier_2')) & (lookbacks_player_w['host'] == 'West Indies') & (lookbacks_player_w['date'] == lookbacks_player_w['date'].max())].loc[:, ['playerid', 'matchid_2', 'recency_weight', 'balls_bowled_2']]
        recencies_w['recency_weight_match_sum'] = recencies_w['recency_weight'] * recencies_w['balls_bowled_2']
        recencies_t = pd.pivot_table(recencies_w, index=['playerid'], values=['recency_weight_match_sum'], aggfunc='sum').reset_index()
        recencies_w = recencies_w.merge(recencies_t, how='left', on=['playerid'])
        recencies_w['recency_weight_bbb_wkt'] = recencies_w['recency_weight_match_sum_x'] / recencies_w['recency_weight_match_sum_y'] / recencies_w['balls_bowled_2']

        recencies = pd.merge(recencies_r.loc[:, ['matchid_2', 'playerid', 'recency_weight_bbb_runs']], recencies_w.loc[:, ['matchid_2', 'playerid', 'recency_weight_bbb_wkt']], how='outer')
        recencies.to_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/outputs/bowlRecencies_w.csv', index=False)
        ratings.to_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/outputs/bowlRatingsJungle_w.csv', index=False)

    else:
        ratings.to_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/outputs/bowlRatingsRasoi_w.csv', index=False)

