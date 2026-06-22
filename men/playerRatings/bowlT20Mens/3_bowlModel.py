import pandas as pd
import numpy as np
from bowlFunctions import buildRunRatingsOriginal, buildWktRatingsOriginal, build_rating_debug_tables
from paths import PROJECT_ROOT
DEBUG_CONFIG = globals().get('DEBUG_CONFIG', None)
BOWL_MODEL_DEBUG_TABLES = None


# -------------------------
# Imports
# -------------------------
bowl_data = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/bowlT20Mens/data/bowlDataCombinedClean.csv', parse_dates=['date', 'dob'])
bowl_weightings = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/bowlT20Mens/auxiliaries/bowlWeightings.csv')
n2h_factors_seam = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/bowlT20Mens/auxiliaries/bowlN2HFactorsSeam.csv')[['nationality', 'host_2', 'host', 'run_factor', 'wkt_factor']]
n2h_factors_spin = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/bowlT20Mens/auxiliaries/bowlN2HFactorsSpin.csv')[['nationality', 'host_2', 'host', 'run_factor', 'wkt_factor']]

# -------------------------
# Test one bowler
# -------------------------
# bowl_data = bowl_data[(bowl_data['bowler'] == 'Rashid Khan')]


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
# Build innings table (includes dummy innings)
# -------------------------
innings_info = bowl_data.loc[:, ['date', 'matchid', 'playerid', 'bowler', 'bowlertype_1', 'bowlertype_2', 'bowlertype_3', 'nationality', 'competition', 'host', 'host_region', 'balls_bowled_career',
                                 'balls_bowled_host']].drop_duplicates(['matchid', 'playerid', 'date', 'host', 'competition', 'bowlertype_2'])

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
lookbacks_player['balls_ago'] = lookbacks_player['balls_bowled_career'] - lookbacks_player['balls_bowled_career_2']

types = pd.pivot_table(lookbacks_player, values='date', index='bowlertype_2', aggfunc='count')


# -------------------------
# n2h: nationality -> host adjustments
# -------------------------
n2h_factors_seam['bowlertype_1'] = 'seam'
n2h_factors_spin['bowlertype_1'] = 'spin'
n2h_factors = pd.concat([n2h_factors_seam, n2h_factors_spin], axis=0)

# merge factors for the match being predicted on (nationality, host) (y)
lookbacks_player = lookbacks_player.merge(
    n2h_factors.loc[:, ['nationality', 'bowlertype_1', 'host', 'run_factor', 'wkt_factor']],
    how='left',
    on=['nationality', 'bowlertype_1', 'host']
)

# merge factors for the game we are using as a predictor (x)
n2h_factors_2 = n2h_factors.drop('host_2', axis=1).rename(columns={'host': 'host_2'})
lookbacks_player = lookbacks_player.merge(
    n2h_factors_2.loc[:, ['nationality', 'bowlertype_1', 'host_2', 'run_factor', 'wkt_factor']],
    how='left',
    on=['nationality', 'bowlertype_1', 'host_2'],
    suffixes=('', '_2')
)

# if the matchup does not exist in n2h we set a value of 1
cols = ['run_factor', 'run_factor_2', 'wkt_factor', 'wkt_factor_2']
lookbacks_player[cols] = lookbacks_player[cols].fillna(1)

# adjust the xruns and xwkts
lookbacks_player['adj_realexprbowl'] = lookbacks_player['realexprbowl_2'] / (lookbacks_player['run_factor'] / lookbacks_player['run_factor_2'])
lookbacks_player['adj_realexpwbowl'] = lookbacks_player['realexpwbowl_2'] / (lookbacks_player['wkt_factor'] / lookbacks_player['wkt_factor_2'])


# -------------------------
# Build outputs twice (x=0=jungle, x=1=rasoi)
# -------------------------
for x in np.arange(0, 2, 1):

    # -------------------------
    # Params
    # -------------------------
    if x == 0:
        model_name = 'jungle'
        param_r = {
            'k_sm': 0.0005008276228889989,
            'c_sm': 7.29840353607683,
            't20_sm': 7.7372167697739265,
            'odi2_sm': 2.4294379752824353,
            'odi1_sm': 1.446303641838873,
            'h_sm': 1.0114692993691115,
            'r_sm': 1.043945731972048,
            'k_s': 0.00027871471614363506,
            'c_s': 14.923183476220958,
            't20_s': 14.603647049760088,
            'odi2_s': 1.7957572339991006,
            'odi1_s': 1.5553567852506576,
            'h_s': 1.1267633615674064,
            'r_s': 1.2521822101207563
        }

        param_w = {
            'k_sm': 1.42331275124588E-05,
            'c_sm': 3.47533966945173,
            't20_sm': 1.56008026180969,
            'odi2_sm': 1.0012817205809,
            'odi1_sm': 3.37376205521305,
            'h_sm': 1.03595437463297,
            'r_sm': 1,
            'k_s': 1.34260736114398E-37,
            'c_s': 1.9041603982315,
            't20_s': 1.01955708661337,
            'odi2_s': 7.34490267562728,
            'odi1_s': 3.45421466548782,
            'h_s': 1.58142850948847,
            'r_s': 1
        }

    else:
        model_name = 'rasoi'
        param_r = {
            'k_sm': 0.0011382609267270565,
            'c_sm': 16.953378400432086,
            't20_sm': 12.177669532266274,
            'odi2_sm': 5.744193498143377,
            'odi1_sm': 1.1590533206158844,
            'h_sm': 1.2710488606389592,
            'r_sm': 1.6768547749481717,
            'k_s': 0.0005191664823675481,
            'c_s': 14.48415344195518,
            't20_s': 9.182168920911959,
            'odi2_s': 1.430452173375036,
            'odi1_s': 3.3650636465902326,
            'h_s': 1.0000010236429717,
            'r_s': 1.0000000000000002
        }

        param_w = {
            'k_sm': 0.000454733281725276,
            'c_sm': 19.9999999999999,
            't20_sm': 8.9818576218138,
            'odi2_sm': 1,
            'odi1_sm': 5.36104365455827,
            'h_sm': 1,
            'r_sm': 1,
            'k_s': 0.000568710279419475,
            'c_s': 19.9999995896568,
            't20_s': 4.34771795586107,
            'odi2_s': 1,
            'odi1_s': 9.20359746630065,
            'h_s': 1.40063771403683,
            'r_s': 1.53738619957142
        }

    # -------------------------
    # Build outputs
    # -------------------------
    ratings_player_r, lookbacks_player_r = buildRunRatingsOriginal(param_r, lookbacks_player)
    ratings_player_w, lookbacks_player_w = buildWktRatingsOriginal(param_w, lookbacks_player)
    bowl_data_t20 = bowl_data[bowl_data['format'] == 't20'].copy()

    # -------------------------
    # Drop duplicate same-day ratings before merging
    # -------------------------
    ratings_player_r = ratings_player_r.drop_duplicates(subset=['date', 'playerid', 'bowler', 'host', 'competition'])
    ratings_player_w = ratings_player_w.drop_duplicates(subset=['date', 'playerid', 'bowler', 'host', 'competition'])

    # -------------------------
    # Merge run + wkt outputs
    # -------------------------
    ratings_player = pd.merge(
        ratings_player_r.drop(labels=['realexprbowl_2', 'runs_2', 'weight_exprbowl', 'weight_runs'], axis=1),
        ratings_player_w.drop(labels=['realexpwbowl_2', 'wkt_2', 'weight_expwbowl', 'weight_wkt'], axis=1),
        how='left',
        on=['date', 'playerid', 'bowler', 'host', 'competition'],
        suffixes=('_r', '_w')
    )

    # -------------------------
    # Innings performance merge
    # -------------------------
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

    # -------------------------
    # Ratings info merge
    # -------------------------
    ratings_info = bowl_data_t20.loc[:, [
        'date', 'matchid', 'battingteam', 'playerid', 'bowler', 'bowlertype_1',
        'bowlertype_2', 'bowlertype_3', 'bowler_level', 'ballspermatch', 'age',
        'nationality', 'home_region', 'host', 'host_region', 'H/A_competition',
        'H/A_country', 'H/A_region', 'competition', 'overseas_pct'
    ]].drop_duplicates(['date', 'matchid', 'playerid', 'bowler', 'host', 'competition'])

    # -------------------------
    # Final outputs table
    # -------------------------
    ratings = innings_perf_out.merge(
        ratings_info,
        how='left',
        on=['date', 'matchid', 'playerid', 'bowler', 'host', 'competition']
    )

    ratings = ratings.merge(
        ratings_player,
        how='left',
        on=['date', 'playerid', 'bowler', 'host', 'competition']
    )

    ratings = ratings[~ratings['competition'].isin(['ODI1', 'ODI2'])]

    ratings = ratings.loc[:, [
        'date', 'matchid', 'battingteam', 'playerid', 'bowler', 'bowlertype_1', 'bowlertype_2', 'bowlertype_3', 'bowler_level', 'ballspermatch', 'age',
        'nationality', 'home_region', 'host', 'host_region', 'H/A_competition', 'H/A_country', 'H/A_region', 'competition', 'balls_bowled_career',
        'balls_bowled_host', 'overseas_pct', 'balls_bowled_2_r', 'ord_2_r', 'z_run_ratio', 'run_rating_0', 'run_rating', 'balls_bowled_2_w',
        'ord_2_w', 'z_wkt_ratio', 'wkt_rating_0', 'wkt_rating', 'balls_bowled', 'realexprbowl', 'runs', 'i_run_ratio', 'realexpwbowl', 'wkt', 'i_wkt_ratio'
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

    # -------------------------
    # Debug
    # -------------------------
    if DEBUG_CONFIG is not None and DEBUG_CONFIG['model'] == model_name:
        BOWL_MODEL_DEBUG_TABLES = build_rating_debug_tables(
            DEBUG_CONFIG,
            ratings,
            lookbacks_player_r,
            lookbacks_player_w
        )

    # -------------------------
    # Exports
    # -------------------------
    if x == 0:
        recencies_r = lookbacks_player_r[((lookbacks_player_r['competition'] == 'T20I') | (lookbacks_player_r['competition'] == 'tier_2')) & (lookbacks_player_r['host'] == 'West Indies') & (lookbacks_player_r['date'] == lookbacks_player_r['date'].max())].loc[:, ['playerid', 'matchid_2', 'recency_weight', 'balls_bowled_2']]
        recencies_r['recency_weight_match_sum'] = recencies_r['recency_weight'] * recencies_r['balls_bowled_2']
        recencies_t = pd.pivot_table(recencies_r, index=['playerid'], values=['recency_weight_match_sum'], aggfunc='sum').reset_index()
        recencies_r = recencies_r.merge(recencies_t, how='left', on=['playerid'])
        recencies_r['recency_weight_bbb_runs'] = recencies_r['recency_weight_match_sum_x'] / recencies_r['recency_weight_match_sum_y'] / recencies_r['balls_bowled_2']

        recencies_w = lookbacks_player_w[((lookbacks_player_w['competition'] == 'T20I') | (lookbacks_player_w['competition'] == 'tier_2')) & (lookbacks_player_w['host'] == 'West Indies') & (lookbacks_player_w['date'] == lookbacks_player_w['date'].max())].loc[:, ['playerid', 'matchid_2', 'recency_weight', 'balls_bowled_2']]
        recencies_w['recency_weight_match_sum'] = recencies_w['recency_weight'] * recencies_w['balls_bowled_2']
        recencies_t = pd.pivot_table(recencies_w, index=['playerid'], values=['recency_weight_match_sum'], aggfunc='sum').reset_index()
        recencies_w = recencies_w.merge(recencies_t, how='left', on=['playerid'])
        recencies_w['recency_weight_bbb_wkt'] = recencies_w['recency_weight_match_sum_x'] / recencies_w['recency_weight_match_sum_y'] / recencies_w['balls_bowled_2']

        recencies = pd.merge(recencies_r.loc[:, ['matchid_2', 'playerid', 'recency_weight_bbb_runs']], recencies_w.loc[:, ['matchid_2', 'playerid', 'recency_weight_bbb_wkt']], how='outer')
        recencies.to_csv(PROJECT_ROOT / 'men/playerRatings/bowlT20Mens/outputs/bowlRecencies.csv', index=False)
        ratings.to_csv(PROJECT_ROOT / 'men/playerRatings/bowlT20Mens/outputs/bowlRatingsJungle.csv', index=False)
    else:
        ratings.to_csv(PROJECT_ROOT / 'men/playerRatings/bowlT20Mens/outputs/bowlRatingsRasoi.csv', index=False)



