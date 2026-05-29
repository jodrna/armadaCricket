import pandas as pd
import numpy as np
from batFunctions_w import buildRunRatingsMapPriority, buildWktRatingsMapPriority
from paths import PROJECT_ROOT


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
bat_weightings = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/auxiliaries/batWeightings.csv')



# -------------------------
# Test one batsman
# -------------------------
# bat_data = bat_data[(bat_data['batsman'] == 'Alana King')]
# bat_data = bat_data[(bat_data['playerid'] == 489889)]


# -------------------------
# Basic preprocessing
# -------------------------
bat_data['competition'] = np.where(
    bat_data['competition'] == 'WODI',
    np.where(bat_data['ballsremaining'] < 84, 'ODI2', 'ODI1'),
    bat_data['competition']
)

bat_data['format'] = bat_data['format'].fillna('t20')

bat_weightings = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/auxiliaries/batWeightings_w.csv')
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
# n2h: nationality -> host adjustments
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

lookbacks_player = lookbacks_player[lookbacks_player['batsman'] == 'Alice Capsey']


# odi_career_ratings = pd.pivot_table(lookbacks_player[lookbacks_player['format'] != 't20'], values=['runs', 'wkt'],
#                                     index=['playerid', 'batsman'], aggfunc={'runs': 'sum', 'wkt': 'sum'}).reset_index()
#
# # -------------------------
# # Build outputs for jungle and rasoi
# # -------------------------
# for x in np.arange(0, 1, 1):
#
#     if x == 0:
#         param_r = [15.17348002, 6.89753, 12.88380525, 6.307514689, 1.692501798, 1.000755457, 0.621652589, 1.241297181, 1.381979717, 0.000496309]
#         param_w = [1.293158205, 3.979704835, 1.80296017, 2.070824639, 1.518191934, 1.071059076, 0.97542588, 1.477165773, 1, 0.000499531]
#     else:
#         param_r = [20, 12.59457633, 17.46079646, 7.338761994, 2.72804768, 1, 0.469010748, 1, 1.444896715, 0.000802191]
#         param_w = [10.5339096, 20, 5.233032976, 7.782822175, 1, 4.423330281, 0.977275005, 1.589376541, 1.080660285, 0.000850588]
#
#     ratings_player_r, lookbacks_player_r = buildRunRatingsMapPriority(param_r, lookbacks_player)
#     ratings_player_w, lookbacks_player_w = buildWktRatingsMapPriority(param_w, lookbacks_player)
#
#
#     # key numbers
#     bat_data['date'] = pd.to_datetime(bat_data['date'])
#     one_year_cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=365)
#     three_year_cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=1095)
#     non_odi = bat_data.loc[bat_data['format'] != 'ODI']
#     career = non_odi.groupby('batsman').agg(runs=('runs', 'sum'), exp=('realexprbat', 'sum'))
#     career['t20_career_run_ratio'] = career['runs'] / career['exp']
#     one_year = non_odi.loc[non_odi['date'] >= one_year_cutoff].groupby('batsman').agg(runs=('runs', 'sum'), exp=('realexprbat', 'sum'))
#     one_year['t20_1yr_run_ratio'] = one_year['runs'] / one_year['exp']
#     three_year = non_odi.loc[non_odi['date'] >= three_year_cutoff].groupby('batsman').agg(runs=('runs', 'sum'), exp=('realexprbat', 'sum'))
#     three_year['t20_3yr_run_ratio'] = three_year['runs'] / three_year['exp']
#     odi = bat_data.loc[bat_data['format'] == 'ODI'].groupby('batsman').agg(runs=('runs', 'sum'), exp=('realexprbat', 'sum'))
#     odi['odi_career_run_ratio'] = odi['runs'] / odi['exp']
#     player_ratings = career[['t20_career_run_ratio']].join(one_year[['t20_1yr_run_ratio']], how='left').join(three_year[['t20_3yr_run_ratio']], how='left').join(odi[['odi_career_run_ratio']],
#                                                                                                                                                                  how='left').reset_index()
#
#     player_ratings = player_ratings[['batsman', 't20_career_run_ratio', 't20_1yr_run_ratio', 't20_3yr_run_ratio', 'odi_career_run_ratio']]
#
#     # use only t20 from now on
#     bat_data_t20 = bat_data[bat_data['format'] == 't20'].copy()
#
#     # drop some
#     ratings_player_r = ratings_player_r.drop_duplicates(subset=['date', 'matchid', 'playerid', 'batsman', 'host', 'competition'])
#     ratings_player_w = ratings_player_w.drop_duplicates(subset=['date', 'matchid', 'playerid', 'batsman', 'host', 'competition'])
#
#     ratings_player = pd.merge(
#         ratings_player_r.drop(labels=['realexprbat_2', 'runs_2', 'weight_exprbat', 'weight_runs'], axis=1),
#         ratings_player_w.drop(labels=['realexpwbat_2', 'wkt_2', 'weight_expwbat', 'weight_wkt'], axis=1),
#         how='left',
#         on=['date', 'matchid', 'playerid', 'batsman', 'host', 'competition'],
#         suffixes=('_r', '_w')
#     )
#
#     innings_perf_out = (
#         pd.pivot_table(
#             bat_data_t20,
#             values=['balls_faced', 'balls_faced_career', 'balls_faced_host', 'runs', 'wkt', 'realexprbat', 'realexpwbat', 'ord'],
#             index=['date', 'playerid', 'matchid', 'batsman', 'host', 'competition'],
#             aggfunc={
#                 'balls_faced': 'sum',
#                 'balls_faced_career': 'min',
#                 'balls_faced_host': 'min',
#                 'runs': 'sum',
#                 'wkt': 'sum',
#                 'realexprbat': 'sum',
#                 'realexpwbat': 'sum',
#                 'ord': 'mean'
#             }
#         )
#         .reset_index()
#     )
#
#     innings_perf_out['i_run_ratio'] = innings_perf_out['runs'] / innings_perf_out['realexprbat']
#     innings_perf_out['i_wkt_ratio'] = innings_perf_out['wkt'] / innings_perf_out['realexpwbat']
#
#     ratings_info = bat_data_t20.loc[:, [
#         'date', 'matchid', 'battingteam', 'playerid', 'batsman', 'age',
#         'nationality', 'home_region', 'host', 'host_region', 'H/A_competition',
#         'H/A_country', 'H/A_region', 'competition', 'overseas_pct', 'careerT20MatchNumber'
#     ]].drop_duplicates(subset=['date', 'matchid', 'playerid', 'host', 'competition'])
#
#     ratings = innings_perf_out.merge(
#         ratings_info,
#         how='left',
#         on=['date', 'matchid', 'playerid', 'batsman', 'host', 'competition']
#     )
#
#     ratings = ratings.merge(
#         ratings_player,
#         how='left',
#         on=['date', 'matchid', 'playerid', 'batsman', 'host', 'competition']
#     )
#
#     ratings = ratings[~ratings['competition'].isin(['ODI1', 'ODI2'])]
#
#     ratings = ratings.loc[:, [
#         'date', 'matchid', 'battingteam', 'playerid', 'batsman', 'age',
#         'nationality', 'home_region', 'host', 'host_region', 'H/A_competition',
#         'H/A_country', 'H/A_region', 'competition', 'careerT20MatchNumber',
#         'balls_faced_career', 'balls_faced_host', 'overseas_pct', 'balls_faced_2_r',
#         'ord_2_r', 'z_run_ratio', 'run_rating', 'run_rating_0', 'weight_balls_r', 'balls_faced_2_w',
#         'ord_2_w', 'z_wkt_ratio', 'wkt_rating', 'wkt_rating_0', 'weight_balls_w', 'balls_faced',
#         'ord', 'realexprbat', 'runs', 'i_run_ratio', 'realexpwbat', 'wkt', 'i_wkt_ratio'
#     ]]
#
#     ratings = ratings.rename(columns={
#         'balls_faced_2_r': 'balls_faced_r',
#         'ord_2_r': 'ord_r',
#         'balls_faced_2_w': 'balls_faced_w',
#         'ord_2_w': 'ord_w',
#         'balls_faced': 'i_balls_faced',
#         'ord': 'i_ord',
#         'realexprbat': 'i_realexprbat',
#         'runs': 'i_runs',
#         'realexpwbat': 'i_realexpwbat',
#         'wkt': 'i_wkt'
#     })
#
#     ratings['i_ord'] = ratings['i_ord'].round(0)
#
#     ratings['run_rating_0'] = ratings['run_rating_0'].fillna(1)
#     ratings['wkt_rating_0'] = ratings['wkt_rating_0'].fillna(1)
#     ratings['run_rating'] = ratings['run_rating'].fillna(1)
#     ratings['wkt_rating'] = ratings['wkt_rating'].fillna(1)
#     ratings['balls_faced_r'] = ratings['balls_faced_r'].fillna(1)
#     ratings['balls_faced_w'] = ratings['balls_faced_w'].fillna(1)
#     ratings['ord_r'] = ratings['ord_r'].fillna(ratings['i_ord'])
#     ratings['ord_w'] = ratings['ord_w'].fillna(ratings['i_ord'])
#
#
#
#     # -------------------------
#     # Recencies + export
#     # -------------------------
#     if x == 0:
#         recencies_r = lookbacks_player_r[(lookbacks_player_r['competition'] == 'WT20I') & (lookbacks_player_r['host'] == 'West Indies') & (lookbacks_player_r['date'] == lookbacks_player_r['date'].max())].loc[:, ['playerid', 'matchid_2', 'recency_weight', 'balls_faced_2']]
#         recencies_r['recency_weight_match_sum'] = recencies_r['recency_weight'] * recencies_r['balls_faced_2']
#         recencies_t = pd.pivot_table(recencies_r, index=['playerid'], values=['recency_weight_match_sum'], aggfunc='sum').reset_index()
#         recencies_r = recencies_r.merge(recencies_t, how='left', on=['playerid'])
#         recencies_r['recency_weight_bbb_runs'] = recencies_r['recency_weight_match_sum_x'] / recencies_r['recency_weight_match_sum_y'] / recencies_r['balls_faced_2']
#
#         recencies_w = lookbacks_player_w[(lookbacks_player_w['competition'] == 'WT20I') & (lookbacks_player_w['host'] == 'West Indies') & (lookbacks_player_r['date'] == lookbacks_player_r['date'].max())].loc[:, ['playerid', 'matchid_2', 'recency_weight', 'balls_faced_2']]
#         recencies_w['recency_weight_match_sum'] = recencies_w['recency_weight'] * recencies_w['balls_faced_2']
#         recencies_t = pd.pivot_table(recencies_w, index=['playerid'], values=['recency_weight_match_sum'], aggfunc='sum').reset_index()
#         recencies_w = recencies_w.merge(recencies_t, how='left', on=['playerid'])
#         recencies_w['recency_weight_bbb_wkt'] = recencies_w['recency_weight_match_sum_x'] / recencies_w['recency_weight_match_sum_y'] / recencies_w['balls_faced_2']
#
#         recencies = pd.merge(recencies_r.loc[:, ['matchid_2', 'playerid', 'recency_weight_bbb_runs']], recencies_w.loc[:, ['matchid_2', 'playerid', 'recency_weight_bbb_wkt']], how='outer')
#         recencies.to_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/outputs/batRecencies_w.csv', index=False)
#         ratings.to_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/outputs/batRatingsJungle_w.csv', index=False)
#
#     else:
#         ratings.to_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/outputs/batRatingsRasoi_w.csv', index=False)
#
#
#
#
#     # -------------------------
#     # Debug rating breakdown
#     # -------------------------
#     debug_type = 'run'  # 'run' or 'wkt'
#     debug_model = 'jungle'
#     debug_batsman = 'Yastika Bhatia'
#     debug_host = 'England'
#     debug_competition = 'WT20I'
#     debug_matchid = 101
#
#     if (debug_model == 'jungle' and x == 0) or (debug_model == 'rasoi' and x == 1):
#         if debug_type == 'run':
#             debug_lookbacks_source = lookbacks_player_r
#             rating_col_0, rating_col, z_col = 'run_rating_0', 'run_rating', 'z_run_ratio'
#             weight_balls_col, actual_col, expected_col = 'weight_balls_r', 'runs_2', 'realexprbat_2'
#             weight_actual_col, weight_expected_col, weighted_rating_col = 'weight_runs', 'weight_exprbat', 'weighted_run_rating'
#
#         elif debug_type == 'wkt':
#             debug_lookbacks_source = lookbacks_player_w
#             rating_col_0, rating_col, z_col = 'wkt_rating_0', 'wkt_rating', 'z_wkt_ratio'
#             weight_balls_col, actual_col, expected_col = 'weight_balls_w', 'wkt_2', 'realexpwbat_2'
#             weight_actual_col, weight_expected_col, weighted_rating_col = 'weight_wkt', 'weight_expwbat', 'weighted_wkt_rating'
#
#         else:
#             raise ValueError("debug_type must be either 'run' or 'wkt'")
#
#         debug_rating = ratings[(ratings['batsman'] == debug_batsman) & (ratings['host'] == debug_host) & (ratings['competition'] == debug_competition) & (ratings['matchid'] == debug_matchid)].copy()
#         debug_lookbacks = debug_lookbacks_source[(debug_lookbacks_source['batsman'] == debug_batsman) & (debug_lookbacks_source['host'] == debug_host) & (debug_lookbacks_source['competition'] == debug_competition) & (debug_lookbacks_source['matchid'] == debug_matchid)].copy()
#
#         if len(debug_rating) > 0 and len(debug_lookbacks) > 0:
#             debug_lookbacks['rating_weight_pct'] = debug_lookbacks[weight_expected_col] / debug_lookbacks[weight_expected_col].sum()
#
#             comp_summary = debug_lookbacks.groupby(['competition_2', 'host_2'], dropna=False).agg(innings=('matchid_2', 'count'),
#                                                                                                   balls_faced_2=('balls_faced_2', 'sum'),
#                                                                                                   avg_location_weight=('location_weight', 'mean'),
#                                                                                                   avg_recency_weight=('recency_weight', 'mean'),
#                                                                                                   weight_balls=(weight_balls_col, 'sum'),
#                                                                                                   actual=(actual_col, 'sum'),
#                                                                                                   expected=(expected_col, 'sum'),
#                                                                                                   weight_actual=(weight_actual_col, 'sum'),
#                                                                                                   weight_expected=(weight_expected_col, 'sum'),
#                                                                                                   rating_contribution_pct=('rating_weight_pct', 'sum')).reset_index()
#             comp_summary['effective_multiplier'] = comp_summary['weight_balls'] / comp_summary['balls_faced_2']
#             comp_summary['effective_balls'] = comp_summary['weight_balls'] * comp_summary['balls_faced_2'].sum() / comp_summary['weight_balls'].sum()
#             comp_summary[weighted_rating_col] = comp_summary['weight_actual'] / comp_summary['weight_expected']
#             comp_summary = comp_summary.sort_values('rating_contribution_pct', ascending=False).reset_index(drop=True)
#             comp_summary = comp_summary[['competition_2', 'host_2', 'innings', 'balls_faced_2', 'avg_location_weight', 'avg_recency_weight', 'effective_multiplier', 'effective_balls', 'actual', 'expected',
#                                          'weight_actual', 'weight_expected', weighted_rating_col, 'rating_contribution_pct']]
#
#             debug_lookbacks['recency_bucket'] = pd.cut(debug_lookbacks['days_ago'], bins=[-1, 90, 180, 365, 730, np.inf], labels=['0-90', '91-180', '181-365', '1-2 years', '2+ years'])
#
#             recency_summary = debug_lookbacks.groupby('recency_bucket', observed=False).agg(innings=('matchid_2', 'count'),
#                                                                                             balls_faced_2=('balls_faced_2', 'sum'),
#                                                                                             avg_location_weight=('location_weight', 'mean'),
#                                                                                             avg_recency_weight=('recency_weight', 'mean'),
#                                                                                             weight_balls=(weight_balls_col, 'sum'),
#                                                                                             actual=(actual_col, 'sum'),
#                                                                                             expected=(expected_col, 'sum'),
#                                                                                             weight_actual=(weight_actual_col, 'sum'),
#                                                                                             weight_expected=(weight_expected_col, 'sum'),
#                                                                                             rating_contribution_pct=('rating_weight_pct', 'sum')).reset_index()
#             recency_summary['effective_multiplier'] = recency_summary['weight_balls'] / recency_summary['balls_faced_2']
#             recency_summary['effective_balls'] = recency_summary['weight_balls'] * recency_summary['balls_faced_2'].sum() / recency_summary['weight_balls'].sum()
#             recency_summary[weighted_rating_col] = recency_summary['weight_actual'] / recency_summary['weight_expected']
#             recency_summary = recency_summary[['recency_bucket', 'innings', 'balls_faced_2', 'avg_location_weight', 'avg_recency_weight', 'effective_multiplier', 'effective_balls', 'actual', 'expected',
#                                                'weight_actual', 'weight_expected', weighted_rating_col, 'rating_contribution_pct']]
#
#
