import pandas as pd
import numpy as np
from batFunctions import buildRunRatingsMapPriority, buildWktRatingsMapPriority
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from pathlib import Path
from paths import PROJECT_ROOT


# -------------------------
# Imports
# -------------------------
bat_data = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/data/combinedBatDataClean.csv', parse_dates=['date', 'dob'])
n2h_factors = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/auxiliaries/batN2HFactors.csv')[['nationality', 'host_2', 'host', 'run_factor', 'wkt_factor']]
n2h_grad = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/auxiliaries/batN2HFactorsGradient.csv').rename(columns={'balls_faced_host_mean_y': 'balls_faced_host'})
bat_weightings = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/auxiliaries/batWeightings.csv')


# -------------------------
# Test one batsman
# -------------------------
# bat_data = bat_data[(bat_data['batsman'] == 'Liam Livingstone')]


# -------------------------
# Basic preprocessing
# -------------------------
bat_data['competition'] = np.where(
    bat_data['competition'].str.contains('ODI', na=False),
    np.where(bat_data['ballsremaining'] < 84, 'ODI2', 'ODI1'),
    bat_data['competition']
)
bat_data['format'] = bat_data['format'].fillna('t20')

bat_data = bat_data.merge(bat_weightings, on='balls_faced_career', how='left')
bat_data['runs_weight_curve'] = bat_data['runs_weight_curve'].fillna(1)
bat_data['wkts_weight_curve'] = bat_data['wkts_weight_curve'].fillna(1)


# -------------------------
# Build innings table (includes dummy innings)
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
# Player lookbacks (self-merge per player)
# suffix _2 = historical innings; non-suffix = innings being predicted
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


# -------------------------
# Avg order per batter (used by rating funcs)
# -------------------------
avg_ord = bat_data.groupby(['playerid', 'batsman'])['ord'].mean().reset_index()
avg_ord.rename(columns={'ord': 'avg_ord'}, inplace=True)
lookbacks_player = lookbacks_player.merge(avg_ord, on=['playerid', 'batsman'], how='left')


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


# -------------------------
# Build outputs twice (x=0=jungle, x=1=rasoi)
# -------------------------
for x in np.arange(0, 2, 1):

    if x == 0:
        param_r_dict = {
            't': 15.17348002,
            'cd': 6.89753,
            'ci': 12.88380525,
            't20': 6.307514689,
            'odi2': 1.692501798,
            'odi1': 1.000755457,
            'dh': 0.621652589,
            'h': 1.241297181,
            'r': 1.381979717,
            'k': 0.000496309
        }

        param_w_dict = {
            't': 3.41440778574228,
            'cd': 3.8550667414475166,
            'ci': 3.173504179099174,
            't20': 2.8307723622651193,
            'odi2': 1.768518342724164,
            'odi1': 1.8861942674443977,
            'dh': 0.9999987012055916,
            'h': 1.0424860935622535,
            'r': 1.000036085364987,
            'k': 0.0005218027196937207
        }

    else:
        param_r_dict = {
            't': 19.999999999996337,
            'cd': 12.703358637592292,
            'ci': 17.706963771657964,
            't20': 7.190821082886796,
            'odi2': 2.4484659358603778,
            'odi1': 1.0000000000001392,
            'dh': 0.44620977429947556,
            'h': 1.000000000000151,
            'r': 1.5560721081512878,
            'k': 0.0007808593485848481
        }

        param_w_dict = {
            't': 10.665839292300744,
            'cd': 19.69562611112414,
            'ci': 7.303091969673036,
            't20': 8.71748836667601,
            'odi2': 1.0000000000012472,
            'odi1': 5.143348793195919,
            'dh': 0.996138316983375,
            'h': 1.2366377661922816,
            'r': 1.0049889542275614,
            'k': 0.0008133348754199519
        }

    param_r = list(param_r_dict.values())
    param_w = list(param_w_dict.values())

    # -------------------------
    # Build outputs
    # -------------------------
    ratings_player_r, lookbacks_player_r = buildRunRatingsMapPriority(param_r, lookbacks_player)
    ratings_player_w, lookbacks_player_w = buildWktRatingsMapPriority(param_w, lookbacks_player)
    bat_data_t20 = bat_data[bat_data['format'] == 't20'].copy()

    # -------------------------
    # Merge run + wkt outputs
    # -------------------------
    ratings_player = pd.merge(
        ratings_player_r.drop(labels=['realexprbat_2', 'runs_2', 'weight_exprbat', 'weight_runs'], axis=1),
        ratings_player_w.drop(labels=['realexpwbat_2', 'wkt_2', 'weight_expwbat', 'weight_wkt'], axis=1),
        how='left',
        on=['date', 'matchid', 'playerid', 'batsman', 'host', 'competition'],
        suffixes=('_r', '_w')
    )

    # -------------------------
    # Ratings info merge
    # -------------------------
    ratings_player = ratings_player.merge(
        bat_data_t20.loc[:, ['date', 'matchid', 'battingteam', 'playerid', 'batsman', 'age', 'nationality', 'home_region', 'host', 'host_region', 'H/A_competition', 'H/A_country', 'H/A_region', 'competition', 'overseas_pct', 'careerT20MatchNumber']].drop_duplicates(subset=['date', 'matchid', 'playerid', 'host', 'competition']),
        how='outer',
        on=['date', 'matchid', 'playerid', 'batsman', 'host', 'competition']
    )

    # -------------------------
    # Innings performance merge
    # -------------------------
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
            }
        )
        .reset_index()
    )

    innings_perf_out['i_run_ratio'] = innings_perf_out['runs'] / innings_perf_out['realexprbat']
    innings_perf_out['i_wkt_ratio'] = innings_perf_out['wkt'] / innings_perf_out['realexpwbat']

    # -------------------------
    # Final outputs table
    # -------------------------
    ratings = innings_perf_out.merge(ratings_player, how='outer', on=['date', 'matchid', 'playerid', 'batsman', 'host', 'competition'])
    ratings = ratings[~ratings['competition'].isin(['ODI1', 'ODI2'])]

    ratings = ratings.loc[:, [
        'date', 'matchid', 'battingteam', 'playerid', 'batsman', 'age', 'nationality', 'home_region', 'host', 'host_region', 'H/A_competition',
        'H/A_country', 'H/A_region', 'competition', 'careerT20MatchNumber', 'balls_faced_career', 'balls_faced_host', 'overseas_pct', 'balls_faced_2_r',
        'ord_2_r', 'z_run_ratio', 'run_rating_0', 'run_rating', 'weight_balls_r', 'balls_faced_2_w', 'ord_2_w', 'z_wkt_ratio', 'wkt_rating_0',
        'wkt_rating', 'weight_balls_w', 'balls_faced', 'ord', 'realexprbat', 'runs', 'i_run_ratio', 'realexpwbat', 'wkt', 'i_wkt_ratio'
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
    # Exports
    # -------------------------
    if x == 0:
        recencies_r = lookbacks_player_r[(lookbacks_player_r['competition'] == 'T20I') & (lookbacks_player_r['host'] == 'West Indies') & (lookbacks_player_r['date'] == lookbacks_player_r['date'].max())].loc[:, ['playerid', 'matchid_2', 'recency_weight', 'balls_faced_2']]
        recencies_r['recency_weight_match_sum'] = recencies_r['recency_weight'] * recencies_r['balls_faced_2']
        recencies_t = pd.pivot_table(recencies_r, index=['playerid'], values=['recency_weight_match_sum'], aggfunc='sum').reset_index()
        recencies_r = recencies_r.merge(recencies_t, how='left', on=['playerid'])
        recencies_r['recency_weight_bbb_runs'] = recencies_r['recency_weight_match_sum_x'] / recencies_r['recency_weight_match_sum_y'] / recencies_r['balls_faced_2']

        recencies_w = lookbacks_player_w[(lookbacks_player_w['competition'] == 'T20I') & (lookbacks_player_w['host'] == 'West Indies') & (lookbacks_player_w['date'] == lookbacks_player_w['date'].max())].loc[:, ['playerid', 'matchid_2', 'recency_weight', 'balls_faced_2']]
        recencies_w['recency_weight_match_sum'] = recencies_w['recency_weight'] * recencies_w['balls_faced_2']
        recencies_t = pd.pivot_table(recencies_w, index=['playerid'], values=['recency_weight_match_sum'], aggfunc='sum').reset_index()
        recencies_w = recencies_w.merge(recencies_t, how='left', on=['playerid'])
        recencies_w['recency_weight_bbb_wkt'] = recencies_w['recency_weight_match_sum_x'] / recencies_w['recency_weight_match_sum_y'] / recencies_w['balls_faced_2']

        recencies = pd.merge(recencies_r.loc[:, ['matchid_2', 'playerid', 'recency_weight_bbb_runs']], recencies_w.loc[:, ['matchid_2', 'playerid', 'recency_weight_bbb_wkt']], how='outer')
        recencies.to_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/outputs/recencies.csv', index=False)
        ratings.to_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/outputs/batRatingsJungle.csv', index=False)

    else:
        ratings.to_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/outputs/batRatingsRasoi.csv', index=False)

# -------------------------
# Report filter
# -------------------------
player_name = 'Khawaja Nafay'
competition_filter = 'T20I'
host_filter = 'Sri Lanka'
match_id = 101

lookbacks_player_r = lookbacks_player_r[(lookbacks_player_r['batsman'] == player_name) &
                                        (lookbacks_player_r['competition'] == competition_filter) &
                                        (lookbacks_player_r['host'] == host_filter) &
                                        (lookbacks_player_r['matchid'] == match_id)]


# -------------------------
# Breakdown dfs for run rating
# -------------------------
lookbacks_breakdown = lookbacks_player_r.copy()

lookbacks_breakdown['year'] = pd.to_datetime(lookbacks_breakdown['date_2']).dt.year
lookbacks_breakdown['n2h_value'] = lookbacks_breakdown['adj_realexprbat'] / lookbacks_breakdown['realexprbat_2']

lookbacks_breakdown['format_group'] = np.where(
    lookbacks_breakdown['competition_2'] == 'ODI1',
    'odi1',
    np.where(
        lookbacks_breakdown['competition_2'] == 'ODI2',
        'odi2',
        't20'
    )
)

total_weight_runs = lookbacks_breakdown['weight_runs'].sum()
total_weight_exprbat = lookbacks_breakdown['weight_exprbat'].sum()

final_run_rating = total_weight_runs / total_weight_exprbat


# -------------------------
# DF 1: by year
# -------------------------
rating_breakdown_year = (
    lookbacks_breakdown.groupby('year', as_index=False)
    .agg(
        runs=('runs_2', 'sum'),
        xruns=('adj_realexprbat', 'sum'),
        balls_faced=('balls_faced_2', 'sum'),
        weight_runs=('weight_runs', 'sum'),
        weight_xruns=('weight_exprbat', 'sum')
    )
)

rating_breakdown_year['raw_rating'] = rating_breakdown_year['runs'] / rating_breakdown_year['xruns']
rating_breakdown_year['weight'] = rating_breakdown_year['weight_xruns'] / total_weight_exprbat
rating_breakdown_year['contribution'] = rating_breakdown_year['weight_runs'] / total_weight_exprbat
rating_breakdown_year['row_rating'] = rating_breakdown_year['weight_runs'] / rating_breakdown_year['weight_xruns']
rating_breakdown_year['final_run_rating'] = final_run_rating

rating_breakdown_year = rating_breakdown_year.loc[:, [
    'year',
    'balls_faced',
    'runs',
    'xruns',
    'raw_rating',
    'weight',
    'contribution',
    'row_rating',
    'final_run_rating'
]]

total_row_year = pd.DataFrame([{
    'year': 'TOTAL',
    'balls_faced': rating_breakdown_year['balls_faced'].sum(),
    'runs': rating_breakdown_year['runs'].sum(),
    'xruns': rating_breakdown_year['xruns'].sum(),
    'raw_rating': rating_breakdown_year['runs'].sum() / rating_breakdown_year['xruns'].sum(),
    'weight': rating_breakdown_year['weight'].sum(),
    'contribution': rating_breakdown_year['contribution'].sum(),
    'row_rating': final_run_rating,
    'final_run_rating': final_run_rating
}])

rating_breakdown_year = pd.concat([rating_breakdown_year, total_row_year], ignore_index=True)


# -------------------------
# DF 2: by host and competition
# -------------------------
rating_breakdown_host_comp = (
    lookbacks_breakdown.groupby(['host_2', 'competition_2'], as_index=False)
    .agg(
        runs=('runs_2', 'sum'),
        xruns=('adj_realexprbat', 'sum'),
        balls_faced=('balls_faced_2', 'sum'),
        weight_runs=('weight_runs', 'sum'),
        weight_xruns=('weight_exprbat', 'sum')
    )
)

n2h_host_comp = (
    lookbacks_breakdown.groupby(['host_2', 'competition_2'])
    .apply(lambda x: np.average(x['n2h_value'], weights=x['weight_exprbat']))
    .reset_index(name='avg_n2h_value')
)

rating_breakdown_host_comp = rating_breakdown_host_comp.merge(
    n2h_host_comp,
    how='left',
    on=['host_2', 'competition_2']
)

rating_breakdown_host_comp['raw_rating'] = rating_breakdown_host_comp['runs'] / rating_breakdown_host_comp['xruns']
rating_breakdown_host_comp['weight'] = rating_breakdown_host_comp['weight_xruns'] / total_weight_exprbat
rating_breakdown_host_comp['contribution'] = rating_breakdown_host_comp['weight_runs'] / total_weight_exprbat
rating_breakdown_host_comp['row_rating'] = rating_breakdown_host_comp['weight_runs'] / rating_breakdown_host_comp['weight_xruns']
rating_breakdown_host_comp['final_run_rating'] = final_run_rating

rating_breakdown_host_comp = rating_breakdown_host_comp.rename(columns={
    'host_2': 'host',
    'competition_2': 'competition'
})

rating_breakdown_host_comp = rating_breakdown_host_comp.loc[:, [
    'host',
    'balls_faced',
    'competition',
    'avg_n2h_value',
    'runs',
    'xruns',
    'raw_rating',
    'weight',
    'contribution',
    'row_rating',
    'final_run_rating'
]]

total_row_host_comp = pd.DataFrame([{
    'host': 'TOTAL',
    'balls_faced': rating_breakdown_host_comp['balls_faced'].sum(),
    'competition': '',
    'avg_n2h_value': np.average(
        lookbacks_breakdown['n2h_value'],
        weights=lookbacks_breakdown['weight_exprbat']
    ),
    'runs': rating_breakdown_host_comp['runs'].sum(),
    'xruns': rating_breakdown_host_comp['xruns'].sum(),
    'raw_rating': rating_breakdown_host_comp['runs'].sum() / rating_breakdown_host_comp['xruns'].sum(),
    'weight': rating_breakdown_host_comp['weight'].sum(),
    'contribution': rating_breakdown_host_comp['contribution'].sum(),
    'row_rating': final_run_rating,
    'final_run_rating': final_run_rating
}])

rating_breakdown_host_comp = pd.concat([rating_breakdown_host_comp, total_row_host_comp], ignore_index=True)


# -------------------------
# DF 3: by format
# -------------------------
rating_breakdown_format = (
    lookbacks_breakdown.groupby('format_group', as_index=False)
    .agg(
        runs=('runs_2', 'sum'),
        xruns=('adj_realexprbat', 'sum'),
        balls_faced=('balls_faced_2', 'sum'),
        weight_runs=('weight_runs', 'sum'),
        weight_xruns=('weight_exprbat', 'sum')
    )
)

n2h_format = (
    lookbacks_breakdown.groupby('format_group')
    .apply(lambda x: np.average(x['n2h_value'], weights=x['weight_exprbat']))
    .reset_index(name='avg_n2h_value')
)

rating_breakdown_format = rating_breakdown_format.merge(
    n2h_format,
    how='left',
    on='format_group'
)

rating_breakdown_format['raw_rating'] = rating_breakdown_format['runs'] / rating_breakdown_format['xruns']
rating_breakdown_format['weight'] = rating_breakdown_format['weight_xruns'] / total_weight_exprbat
rating_breakdown_format['contribution'] = rating_breakdown_format['weight_runs'] / total_weight_exprbat
rating_breakdown_format['row_rating'] = rating_breakdown_format['weight_runs'] / rating_breakdown_format['weight_xruns']
rating_breakdown_format['final_run_rating'] = final_run_rating

format_order = {'t20': 0, 'odi1': 1, 'odi2': 2}
rating_breakdown_format['format_sort'] = rating_breakdown_format['format_group'].map(format_order)
rating_breakdown_format = rating_breakdown_format.sort_values('format_sort').drop(columns='format_sort').reset_index(drop=True)

rating_breakdown_format = rating_breakdown_format.rename(columns={'format_group': 'format'})

rating_breakdown_format = rating_breakdown_format.loc[:, [
    'format',
    'balls_faced',
    'avg_n2h_value',
    'runs',
    'xruns',
    'raw_rating',
    'weight',
    'contribution',
    'row_rating',
    'final_run_rating'
]]

total_row_format = pd.DataFrame([{
    'format': 'TOTAL',
    'balls_faced': rating_breakdown_format['balls_faced'].sum(),
    'avg_n2h_value': np.average(
        lookbacks_breakdown['n2h_value'],
        weights=lookbacks_breakdown['weight_exprbat']
    ),
    'runs': rating_breakdown_format['runs'].sum(),
    'xruns': rating_breakdown_format['xruns'].sum(),
    'raw_rating': rating_breakdown_format['runs'].sum() / rating_breakdown_format['xruns'].sum(),
    'weight': rating_breakdown_format['weight'].sum(),
    'contribution': rating_breakdown_format['contribution'].sum(),
    'row_rating': final_run_rating,
    'final_run_rating': final_run_rating
}])

rating_breakdown_format = pd.concat([rating_breakdown_format, total_row_format], ignore_index=True)


# -------------------------
# Round all numeric columns to 2dp
# -------------------------
def round_df(df):
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].round(2)
    return df

rating_breakdown_year = round_df(rating_breakdown_year)
rating_breakdown_host_comp = round_df(rating_breakdown_host_comp)
rating_breakdown_format = round_df(rating_breakdown_format)


# -------------------------
# Sort by contribution with TOTAL last
# -------------------------
def sort_with_total_last(df, label_col, sort_col):
    df_main = df[df[label_col] != 'TOTAL'].copy()
    df_total = df[df[label_col] == 'TOTAL'].copy()

    df_main = df_main.sort_values(sort_col, ascending=False)

    return pd.concat([df_main, df_total], ignore_index=True)

rating_breakdown_host_comp = sort_with_total_last(rating_breakdown_host_comp, 'host', 'contribution')
rating_breakdown_format = sort_with_total_last(rating_breakdown_format, 'format', 'contribution')


# exports
rating_breakdown_year.to_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/outputs/rating_breakdown_year.csv', index=False)
rating_breakdown_host_comp.to_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/outputs/rating_breakdown_host_comp.csv', index=False)
rating_breakdown_format.to_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/outputs/rating_breakdown_format.csv', index=False)
