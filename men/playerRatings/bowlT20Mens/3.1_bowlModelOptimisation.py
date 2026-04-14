import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from bowlFunctions import newMethodBins, buildRunRatingsOriginal, buildWktRatingsOriginal, qualityMethodBins
from paths import PROJECT_ROOT


# -------------------------
# Imports
# -------------------------
bowl_data = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/bowlT20Mens/data/combinedBowlDataClean.csv', parse_dates=['date', 'dob'])
bowl_weightings = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/bowlT20Mens/auxiliaries/bowlWeightings.csv')
n2h_factors_seam = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/bowlT20Mens/auxiliaries/bowlN2HFactorsSeam.csv')[['nationality', 'host_2', 'host', 'run_factor', 'wkt_factor']]
n2h_factors_spin = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/bowlT20Mens/auxiliaries/bowlN2HFactorsSpin.csv')[['nationality', 'host_2', 'host', 'run_factor', 'wkt_factor']]
current_ratings = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/bowlT20Mens/outputs/bowlRatingsJungle3.csv', parse_dates=['date'])


# -------------------------
# Test one bowler
# -------------------------
# bowl_data = bowl_data[(bowl_data['bowler'] == 'Rashid Khan')]


# -------------------------
# Basic preprocessing
# -------------------------
bowl_data['competition'] = np.where(bowl_data['format'] == 'odi', np.where(bowl_data['ballsremaining'] < 84, 'ODI2', 'ODI1'), bowl_data['competition'])
bowl_data['format'] = bowl_data['format'].fillna('t20')

bowl_data = bowl_data.merge(bowl_weightings, on='balls_bowled_career', how='left')
bowl_data['runs_weight_curve'] = bowl_data['runs_weight_curve'].fillna(1)
bowl_data['wkts_weight_curve'] = bowl_data['wkts_weight_curve'].fillna(1)


# -------------------------
# Build innings table (includes dummy innings)
# -------------------------
innings_info = bowl_data.loc[:, ['date', 'matchid', 'playerid', 'bowler', 'bowlertype_1', 'bowlertype_2', 'bowler_arm', 'bowler_pace', 'nationality', 'competition', 'host', 'host_region', 'balls_bowled_career', 'balls_bowled_host']].drop_duplicates(['matchid', 'playerid', 'date', 'host', 'competition', 'bowlertype_2'])
innings_perf = pd.pivot_table(bowl_data, values=['balls_bowled', 'runs', 'realexprbowl', 'wkt', 'realexpwbowl', 'ord'], index=['date', 'matchid', 'playerid', 'competition', 'host'], aggfunc='sum').reset_index()
innings = innings_info.merge(innings_perf, how='left', left_on=['date', 'matchid', 'playerid', 'competition', 'host'], right_on=['date', 'matchid', 'playerid', 'competition', 'host'])


# -------------------------
# Player lookbacks (self-merge per player)
# suffix _2 = historical innings; non-suffix = innings being predicted
# -------------------------
lookbacks_player = innings.set_index('playerid').merge(innings.set_index('playerid'), how='left', left_index=True, right_index=True, suffixes=('', '_2')).reset_index()
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
lookbacks_player = lookbacks_player.merge(n2h_factors.loc[:, ['nationality', 'bowlertype_1', 'host', 'run_factor', 'wkt_factor']], how='left', on=['nationality', 'bowlertype_1', 'host'])

# merge factors for the game we are using as a predictor (x)
n2h_factors_2 = n2h_factors.drop('host_2', axis=1).rename(columns={'host': 'host_2'})
lookbacks_player = lookbacks_player.merge(n2h_factors_2.loc[:, ['nationality', 'bowlertype_1', 'host_2', 'run_factor', 'wkt_factor']], how='left', on=['nationality', 'bowlertype_1', 'host_2'], suffixes=('', '_2'))

# if the matchup does not exist in n2h we set a value of 1 when home, for away we set 1 (bowling uses 1 baseline)
cols = ['run_factor', 'run_factor_2', 'wkt_factor', 'wkt_factor_2']
lookbacks_player[cols] = lookbacks_player[cols].fillna(1)

# adjust the xruns and xwkts
lookbacks_player['adj_realexprbowl'] = lookbacks_player['realexprbowl_2'] / (lookbacks_player['run_factor'] / lookbacks_player['run_factor_2'])
lookbacks_player['adj_realexpwbowl'] = lookbacks_player['realexpwbowl_2'] / (lookbacks_player['wkt_factor'] / lookbacks_player['wkt_factor_2'])


# -------------------------
# Merge current outputs into row-level bowl_data (for binning / error measurement)
# -------------------------
bowl_data = bowl_data.merge(current_ratings.loc[:, ['playerid', 'matchid', 'run_rating_3', 'wkt_rating_3', 'host', 'competition']], on=['playerid', 'matchid', 'host', 'competition'], how='left')
bowl_data = bowl_data[(bowl_data['competition'] != 'ODI') & (bowl_data['balls_bowled'] > 0)].dropna(subset=['run_rating_3', 'wkt_rating_3'])


# -------------------------
# Tables for inspection
# -------------------------
ratingsOuter = []
pivotOuter = []
bowl_dataOuter = []
lookbacksOuter = []
opt_history = pd.DataFrame(columns=['rmse', 'sse_mean', 'sse_total'] + [f'param{i}' for i in range(14)])


def optimise_params(param, lookbacks_player, build_fn, rating_col, exp_col, actual_col, weight_curve_col, out_pred_col, bin_col='binid'):
    param_dict = dict(zip(PARAM_KEYS, param)) if isinstance(param, (list, np.ndarray)) else param
    ratings_i, lookbacks_i = build_fn(param_dict, lookbacks_player)

    bowl_data_i = bowl_data.copy()
    bowl_data_i = bowl_data_i.merge(ratings_i.loc[:, ['playerid', 'matchid', 'host', 'date', 'competition', rating_col]], how='left', on=['playerid', 'host', 'date', 'competition', 'matchid'])

    bowl_data_i[out_pred_col] = bowl_data_i[exp_col] * bowl_data_i[rating_col]
    bowl_data_i = bowl_data_i.dropna(subset=[rating_col, actual_col, exp_col, weight_curve_col])

    bowl_data_i = bowl_data_i.assign(pred_w=lambda d: d[out_pred_col] * d[weight_curve_col], act_w=lambda d: d[actual_col] * d[weight_curve_col])

    pivot = bowl_data_i.groupby(bin_col, as_index=False).agg(rating_avg=(rating_col, 'mean'),
                                                            balls_bowled=('balls_bowled', 'sum'),
                                                            exp_sum=(exp_col, 'sum'),
                                                            pred_sum=(out_pred_col, 'sum'),
                                                            actual_sum=(actual_col, 'sum'),
                                                            weight_curve_avg=(weight_curve_col, 'mean'),
                                                            pred_w=('pred_w', 'sum'),
                                                            act_w=('act_w', 'sum')).assign(bin_residual=lambda df: df['pred_w'] - df['act_w']).sort_values(bin_col)
    pivot = pivot[pivot['balls_bowled'] > 30]

    residual = pivot['bin_residual'].to_numpy()
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    sse_mean = float(np.mean(residual ** 2))
    sse_total = float(np.sum(residual ** 2))
    print(f"RMSE={rmse:.6f}, mean_SSE={sse_mean:.6f}")

    ratingsOuter.clear(); ratingsOuter.append(ratings_i)
    pivotOuter.clear(); pivotOuter.append(pivot)
    bowl_dataOuter.clear(); bowl_dataOuter.append(bowl_data_i)
    lookbacksOuter.clear(); lookbacksOuter.append(lookbacks_i)

    row = {'rmse': rmse, 'sse_mean': sse_mean, 'sse_total': sse_total, **{f'param{i}': float(p) for i, p in enumerate(param)}}
    opt_history.loc[len(opt_history)] = row

    return residual



# -------------------------
# Params (dict -> list(values) to match the other script)
# -------------------------
# # params optimised for runs using quality grouping
# param0_dict = {
#             # seam
#             'k_sm': 0.00240593698166117,
#             'c_sm': 6.776715141,
#             'h_sm': 2.284969437,
#             'r_sm': 1,
#             't20_sm': 1.022065556,
#             'odi1_sm': 2.422278827,
#             'odi2_sm': 3.920148079,
#             # spin
#             'k_s': 0.002498944,
#             'c_s': 20,
#             'h_s': 4.392599392,
#             'r_s': 2.771822072,
#             't20_s': 2.964639797,
#             'odi1_s': 1.00,
#             'odi2_s': 7.569004655
#         }

# # params optimised for wkts using quality grouping
# param0_dict = {
#             # seam
#             'k_sm': 1.42331275124588E-05,
#             'c_sm': 3.47533966945173,
#             'h_sm': 1.03595437463297,
#             'r_sm': 1,
#             't20_sm': 1.56008026180969,
#             'odi1_sm': 3.37376205521305,
#             'odi2_sm': 1.0012817205809,
#
#             # spin
#             'k_s': 1.34260736114398E-37,
#             'c_s': 1.9041603982315,
#             'h_s': 1.58142850948847,
#             'r_s': 1,
#             't20_s': 1.01955708661337,
#             'odi1_s': 3.45421466548782,
#             'odi2_s': 7.34490267562728
#         }

# # params optimised for runs using innings grouping
# param0_dict = {
#             # seam
#             'k_sm': 0.000848309,
#             'c_sm': 7.989737242,
#             'h_sm': 2.384964402,
#             'r_sm': 1,
#             't20_sm': 1.02215819,
#             'odi1_sm': 2.49029014,
#             'odi2_sm': 1.537421599,
#
#             # spin
#             'k_s': 0.001902713,
#             'c_s': 20,
#             'h_s': 1.29602795,
#             'r_s': 1.41236992,
#             't20_s': 3.052102008,
#             'odi1_s': 1.00,
#             'odi2_s': 1.00
#         }


# params optimised for wkts using innings grouping
param0_dict = {
            # seam
            'k_sm': 0.000454733281725276,
            'c_sm': 19.9999999999999,
            'h_sm': 1,
            'r_sm': 1,
            't20_sm': 8.9818576218138,
            'odi1_sm': 5.36104365455827,
            'odi2_sm': 1,
            # spin
            'k_s': 0.000568710279419475,
            'c_s': 19.9999995896568,
            'h_s': 1.40063771403683,
            'r_s': 1.53738619957142,
            't20_s': 4.34771795586107,
            'odi1_s': 9.20359746630065,
            'odi2_s': 1
        }
param0 = list(param0_dict.values())
PARAM_KEYS = list(param0_dict.keys())


# -------------------------
# Bounds (unchanged)
# -------------------------
lower_dict = {'k_sm': 0.0, 'c_sm': 1, 'h_sm': 1, 'r_sm': 1, 't20_sm': 1, 'odi1_sm': 1, 'odi2_sm': 1,
              'k_s': 0.0, 'c_s': 1, 'h_s': 1, 'r_s': 1, 't20_s': 1, 'odi1_s': 1, 'odi2_s': 1}
upper_dict = {'k_sm': 0.02, 'c_sm': 20, 'h_sm': 20, 'r_sm': 20, 't20_sm': 20, 'odi1_sm': 20, 'odi2_sm': 20,
              'k_s': 0.02, 'c_s': 20, 'h_s': 20, 'r_s': 20, 't20_s': 20, 'odi1_s': 20, 'odi2_s': 20}
lower = list(lower_dict.values()); upper = list(upper_dict.values())


# -------------------------
# Choose grouping method (bins)
# -------------------------
# bowl_data = qualityMethodBins(bowl_data, bin_size=40, rating_col='run_rating_3', out_col='binid')
bowl_data = newMethodBins(bowl_data)


# -------------------------
# Optimiser config + objective
# -------------------------
cfg = dict(lookbacks_player=lookbacks_player,
           build_fn=buildWktRatingsOriginal,
           rating_col='wkt_rating',
           exp_col='realexpwbowl',
           actual_col='wkt',
           weight_curve_col='wkts_weight_curve',
           out_pred_col='wkts_pred',
           bin_col='binid')
obj_fn = lambda p: optimise_params(p, **cfg)


# -------------------------
# Optimisation
# -------------------------
optimise = least_squares(obj_fn, param0, ftol=1e-4, bounds=(lower, upper))


# -------------------------
# Test evaluation of params
# -------------------------
test = obj_fn(param0)

