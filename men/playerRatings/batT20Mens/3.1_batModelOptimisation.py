import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from batFunctions import qualityMethodBins, newMethodBins, buildRunRatingsOriginal, buildRunRatingsMapOne, buildRunRatingsMapTwo, buildRunRatingsMapPriority, buildWktRatingsMapPriority, buildWktRatingsOriginal
from paths import PROJECT_ROOT


# -------------------------
# Imports
# -------------------------
bat_data = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/data/batDataCombinedClean.csv', parse_dates=['date', 'dob'])
n2h_factors = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/auxiliaries/batN2HFactors.csv')[['nationality', 'host_2', 'host', 'run_factor', 'wkt_factor']]
n2h_grad = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/auxiliaries/batN2HFactorsGradient.csv').rename(columns={'balls_faced_host_mean_y': 'balls_faced_host'})
current_ratings = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/outputs/batRatingsJungle3.csv')
bat_weightings = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/auxiliaries/batWeightings.csv')


# -------------------------
# Basic preprocessing
# -------------------------
bat_data['competition'] = np.where(bat_data['competition'] == 'ODI', np.where(bat_data['ballsremaining'] < 84, 'ODI2', 'ODI1'), bat_data['competition'])
bat_data['format'] = bat_data['format'].fillna('t20')

bat_data = bat_data.merge(bat_weightings, on='balls_faced_career', how='left')
bat_data['runs_weight_curve'] = bat_data['runs_weight_curve'].fillna(1)
bat_data['wkts_weight_curve'] = bat_data['wkts_weight_curve'].fillna(1)


# -------------------------
# Build innings table (includes dummy innings)
# -------------------------
innings_info = bat_data.loc[:, ['date', 'matchid', 'playerid', 'batsman', 'nationality', 'competition', 'host', 'host_region', 'balls_faced_career', 'balls_faced_host', 'H/A_competition', 'ord']].drop_duplicates(['matchid', 'playerid', 'date', 'host', 'competition'])
innings_perf = pd.pivot_table(bat_data, values=['balls_faced', 'runs', 'realexprbat', 'wkt', 'realexpwbat'], index=['date', 'matchid', 'playerid', 'competition', 'host', 'ord'], aggfunc='sum').reset_index()
innings = innings_info.merge(innings_perf, how='left', left_on=['date', 'matchid', 'playerid', 'competition', 'host', 'ord'], right_on=['date', 'matchid', 'playerid', 'competition', 'host', 'ord'])


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
# merge factors for the match being predicted on (nationality, host) (y)
lookbacks_player = lookbacks_player.merge(n2h_factors.loc[:, ['nationality', 'host', 'run_factor', 'wkt_factor']], how='left', on=['nationality', 'host'])

# merge factors for the game we are using as a predictor (x)
n2h_factors_2 = n2h_factors.drop('host_2', axis=1).rename(columns={'host': 'host_2'})
lookbacks_player = lookbacks_player.merge(n2h_factors_2.loc[:, ['nationality', 'host_2', 'run_factor', 'wkt_factor']], how='left', on=['nationality', 'host_2'], suffixes=('', '_2'))

# if the matchup does not exist in n2h we set a value of 1 when home, for away we set 0.9882 for runs and 1.0146 for wkts
home_pred = lookbacks_player['H/A_competition'] == 'Home'
home_hist = lookbacks_player['H/A_competition_2'] == 'Home'
m = lookbacks_player['run_factor'].isna();    lookbacks_player.loc[m, 'run_factor']    = np.where(home_pred[m], 1, 0.9882)
m = lookbacks_player['run_factor_2'].isna();  lookbacks_player.loc[m, 'run_factor_2']  = np.where(home_hist[m], 1, 0.9882)
m = lookbacks_player['wkt_factor'].isna();    lookbacks_player.loc[m, 'wkt_factor']    = np.where(home_pred[m], 1, 1.0146)
m = lookbacks_player['wkt_factor_2'].isna();  lookbacks_player.loc[m, 'wkt_factor_2']  = np.where(home_hist[m], 1, 1.0146)

# Experience smoothing (apply only when away), first adjust the y games
lookbacks_player = lookbacks_player.merge(n2h_grad, on='balls_faced_host', how='left')
away_pred = lookbacks_player['H/A_competition'] == 'Away'
lookbacks_player.loc[away_pred, 'run_factor'] = lookbacks_player.loc[away_pred, 'run_factor'] * lookbacks_player.loc[away_pred, 'run_factor_smooth']
lookbacks_player.loc[away_pred, 'wkt_factor'] = lookbacks_player.loc[away_pred, 'wkt_factor'] * lookbacks_player.loc[away_pred, 'wkt_factor_smooth']

# then adjust the x games
n2h_grad_2 = n2h_grad.rename(columns={'balls_faced_host': 'balls_faced_host_2', 'run_factor_smooth': 'run_factor_smooth_2', 'wkt_factor_smooth': 'wkt_factor_smooth_2'})
lookbacks_player = lookbacks_player.merge(n2h_grad_2, on='balls_faced_host_2', how='left')
away_hist = lookbacks_player['H/A_competition_2'] == 'Away'
lookbacks_player.loc[away_hist, 'run_factor_2'] = lookbacks_player.loc[away_hist, 'run_factor_2'] * lookbacks_player.loc[away_hist, 'run_factor_smooth_2']
lookbacks_player.loc[away_hist, 'wkt_factor_2'] = lookbacks_player.loc[away_hist, 'wkt_factor_2'] * lookbacks_player.loc[away_hist, 'wkt_factor_smooth_2']

# adjust the xruns and xwkts
lookbacks_player['adj_realexprbat'] = lookbacks_player['realexprbat_2'] / (lookbacks_player['run_factor'] / lookbacks_player['run_factor_2'])
lookbacks_player['adj_realexpwbat'] = lookbacks_player['realexpwbat_2'] / (lookbacks_player['wkt_factor'] / lookbacks_player['wkt_factor_2'])


# -------------------------
# Merge current outputs into row-level bat_data (for binning / error measurement)
# -------------------------
bat_data = bat_data.merge(current_ratings.loc[:, ['playerid', 'matchid', 'run_rating_3', 'wkt_rating_3', 'host', 'competition']], on=['playerid', 'matchid', 'host', 'competition'], how='left')
bat_data = bat_data[(bat_data['competition'] != 'ODI') & (bat_data['balls_faced'] > 0)].dropna(subset=['run_rating_3', 'wkt_rating_3'])


# -------------------------
# Tables for inspection
# -------------------------
ratingsOuter = []
pivotOuter = []
bat_dataOuter = []
lookbacksOuter = []
opt_history = pd.DataFrame(columns=['rmse', 'sse_mean', 'sse_total', 'param0', 'param1', 'param2', 'param3', 'param4', 'param5', 'param6', 'param7', 'param8', 'param9'])


def optimise_params(param, lookbacks_player, build_fn, rating_col, exp_col, actual_col, weight_curve_col, out_pred_col, bin_col='binid'):
    ratings_i, lookbacks_i = build_fn(param, lookbacks_player)

    bat_data_i = bat_data.copy()
    bat_data_i = bat_data_i.merge(ratings_i.loc[:, ['playerid', 'matchid', 'host', 'date', 'competition', rating_col]], how='left', on=['playerid', 'host', 'date', 'competition', 'matchid'])

    bat_data_i[out_pred_col] = bat_data_i[exp_col] * bat_data_i[rating_col]
    bat_data_i = bat_data_i.dropna(subset=[rating_col, actual_col, exp_col, weight_curve_col])

    bat_data_i = bat_data_i.assign(pred_w=lambda d: d[out_pred_col] * d[weight_curve_col], act_w=lambda d: d[actual_col] * d[weight_curve_col])

    pivot = bat_data_i.groupby(bin_col, as_index=False).agg(rating_avg=(rating_col, 'mean'),
                                                            balls_faced=('balls_faced', 'sum'),
                                                            exp_sum=(exp_col, 'sum'),
                                                            pred_sum=(out_pred_col, 'sum'),
                                                            actual_sum=(actual_col, 'sum'),
                                                            weight_curve_avg=(weight_curve_col, 'mean'),
                                                            pred_w=('pred_w', 'sum'),
                                                            act_w=('act_w', 'sum')).assign(bin_residual=lambda df: df['pred_w'] - df['act_w']).sort_values(bin_col)
    pivot = pivot[pivot['balls_faced'] > 30]

    residual = pivot['bin_residual'].to_numpy()
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    sse_mean = float(np.mean(residual ** 2))
    sse_total = float(np.sum(residual ** 2))
    print(f"RMSE={rmse:.6f}, mean_SSE={sse_mean:.6f}")

    ratingsOuter.clear(); ratingsOuter.append(ratings_i)
    pivotOuter.clear(); pivotOuter.append(pivot)
    bat_dataOuter.clear(); bat_dataOuter.append(bat_data_i)
    lookbacksOuter.clear(); lookbacksOuter.append(lookbacks_i)

    row = {'rmse': rmse, 'sse_mean': sse_mean, 'sse_total': sse_total, **{f'param{i}': float(p) for i, p in enumerate(param)}}
    opt_history.loc[len(opt_history)] = row

    return residual





# -------------------------
# Params (dict -> list(values) to match the other script)
# -------------------------
# # params optimised for runs using quality grouping
# param0_dict = {
#             't': 14.378529018603198,
#             'cd': 7.216926617554105,
#             'ci': 12.707149487260645,
#             't20': 7.329201869901505,
#             'odi2': 2.3432102672470165,
#             'odi1': 1.0475674672311073,
#             'dh': 0.9436118924235586,
#             'h': 1.0014649111276281,
#             'r': 1.2825266178392976,
#             'k': 0.0005820512705756658
#         }

# # params optimised for wkts using quality grouping
# param0_dict =  {
#             't': 3.41440778574228,
#             'cd': 3.8550667414475166,
#             'ci': 3.173504179099174,
#             't20': 2.8307723622651193,
#             'odi2': 1.768518342724164,
#             'odi1': 1.8861942674443977,
#             'dh': 0.9999987012055916,
#             'h': 1.0424860935622535,
#             'r': 1.000036085364987,
#             'k': 0.0005218027196937207
#         }



# params optimised for runs using innings grouping
param0_dict = {
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

# # params optimised for wkts using innings grouping
# param0_dict = {
#             't': 10.665839292300744,
#             'cd': 19.69562611112414,
#             'ci': 7.303091969673036,
#             't20': 8.71748836667601,
#             'odi2': 1.0000000000012472,
#             'odi1': 5.143348793195919,
#             'dh': 0.996138316983375,
#             'h': 1.2366377661922816,
#             'r': 1.0049889542275614,
#             'k': 0.0008133348754199519
#         }

param0 = list(param0_dict.values())


# -------------------------
# Bounds (unchanged)
# -------------------------
lower_dict = {'t': 1, 'cd': 1, 'ci': 1, 't20': 1, 'odi2': 1, 'odi1': 1, 'dh': 0, 'h': 1, 'r': 1, 'k': 0}
upper_dict = {'t': 20, 'cd': 20, 'ci': 20, 't20': 20, 'odi2': 20, 'odi1': 20, 'dh': 1, 'h': 20, 'r': 20, 'k': 0.01}
lower = list(lower_dict.values()); upper = list(upper_dict.values())


# -------------------------
# Choose grouping method (bins)
# -------------------------
# bat_data = qualityMethodBins(bat_data, bin_size=40, rating_col='wkt_rating_3', out_col='binid')
bat_data = newMethodBins(bat_data)


# -------------------------
# Optimiser config + objective
# -------------------------
cfg = dict(lookbacks_player=lookbacks_player,
           build_fn=buildRunRatingsMapPriority,
           rating_col='run_rating',
           exp_col='realexprbat',
           actual_col='runs',
           weight_curve_col='runs_weight_curve',
           out_pred_col='runs_pred',
           bin_col='binid')
obj_fn = lambda p: optimise_params(p, **cfg)


# -------------------------
# Optimisation
# -------------------------
optimise = least_squares(obj_fn, param0, ftol=1e-5, bounds=(lower, upper))


# -------------------------
# Test evaluation of params
# -------------------------
# test = obj_fn(param0)
