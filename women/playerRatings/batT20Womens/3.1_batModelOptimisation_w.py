import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from batFunctions_w import qualityMethodBins, newMethodBins, buildRunRatings, buildWktRatings
from paths import PROJECT_ROOT


# -------------------------
# Configure
# -------------------------
model_name = 'rasoi'     # jungle / rasoi
target = 'wkts'          # runs / wkts
mode = 'optimise'        # test / optimise


# -------------------------
# Imports
# -------------------------
bat_data = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/data/batDataCombinedClean_w.csv', parse_dates=['date', 'dob'])
n2h_factors = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/auxiliaries/batN2HFactors_w.csv')[['nationality', 'host_2', 'host', 'run_factor', 'wkt_factor']]
n2h_grad = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/auxiliaries/batN2HFactorsGradient_w.csv').rename(columns={'balls_faced_host_mean_y': 'balls_faced_host'})
current_ratings_path = PROJECT_ROOT / f"women/playerRatings/batT20Womens/outputs/batRatings{model_name.capitalize()}3_w.csv"
current_ratings = pd.read_csv(current_ratings_path)
bat_weightings = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/auxiliaries/batWeightings_w.csv')


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

# experience smoothing (apply only when away), first adjust the y games
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
# Tables for debugging
# -------------------------
ratingsOuter = []
pivotOuter = []
bat_dataOuter = []
lookbacksOuter = []


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

    row = {'rmse': rmse, 'sse_mean': sse_mean, 'sse_total': sse_total, **{param_names[i]: float(p) for i, p in enumerate(param)}}
    opt_history.loc[len(opt_history)] = row

    return residual





# -------------------------
# Target config
# -------------------------
target_cfg = {
    'runs': {
        'build_fn': buildRunRatings,
        'rating_col': 'run_rating',
        'rating_col_3': 'run_rating_3',
        'exp_col': 'realexprbat',
        'actual_col': 'runs',
        'weight_curve_col': 'runs_weight_curve',
        'out_pred_col': 'runs_pred'
    },
    'wkts': {
        'build_fn': buildWktRatings,
        'rating_col': 'wkt_rating',
        'rating_col_3': 'wkt_rating_3',
        'exp_col': 'realexpwbat',
        'actual_col': 'wkt',
        'weight_curve_col': 'wkts_weight_curve',
        'out_pred_col': 'wkts_pred'
    }
}

build_fn = target_cfg[target]['build_fn']
rating_col = target_cfg[target]['rating_col']
rating_col_3 = target_cfg[target]['rating_col_3']
exp_col = target_cfg[target]['exp_col']
actual_col = target_cfg[target]['actual_col']
weight_curve_col = target_cfg[target]['weight_curve_col']
out_pred_col = target_cfg[target]['out_pred_col']


# -------------------------
# Params
# -------------------------
params = {
    'jungle': {
        'runs': {
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
        },
        'wkts': {
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
    },
    'rasoi': {
        'runs': {
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
        },
        'wkts': {
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
    }
}

param0_dict = params[model_name][target]
param_names = list(param0_dict.keys())
param0 = list(param0_dict.values())
opt_history = pd.DataFrame(columns=['rmse', 'sse_mean', 'sse_total'] + param_names)


# -------------------------
# Bounds
# -------------------------
lower_dict = {'t': 1, 'cd': 1, 'ci': 1, 't20': 1, 'odi2': 1, 'odi1': 1, 'dh': 0, 'h': 1, 'r': 1, 'k': 0}
upper_dict = {'t': 20, 'cd': 20, 'ci': 20, 't20': 20, 'odi2': 20, 'odi1': 20, 'dh': 1, 'h': 20, 'r': 20, 'k': 0.01}

lower = list(lower_dict.values())
upper = list(upper_dict.values())


# -------------------------
# Choose grouping method, jungle = new, rasoi = quality, also merge in current ratings to decide bins
# -------------------------
if model_name == 'jungle':
    bat_data = qualityMethodBins(bat_data)

else:
    bat_data = bat_data.merge(current_ratings.loc[:, ['playerid', 'matchid', 'host', 'competition', rating_col_3]], on=['playerid', 'matchid', 'host', 'competition'], how='left')
    bat_data = bat_data[(bat_data['competition'] != 'ODI') & (bat_data['balls_faced'] > 0)].dropna(subset=[rating_col_3])
    bat_data = newMethodBins(bat_data, bin_size=40, rating_col=rating_col_3, out_col='binid')





# -------------------------
# Optimiser config + objective
# -------------------------
optimiser_cfg = dict(
    lookbacks_player=lookbacks_player,
    build_fn=build_fn,
    rating_col=rating_col,
    exp_col=exp_col,
    actual_col=actual_col,
    weight_curve_col=weight_curve_col,
    out_pred_col=out_pred_col,
    bin_col='binid'
)

obj_fn = lambda p: optimise_params(p, **optimiser_cfg)



# -------------------------
# Optimisation/Testing
# -------------------------
if mode == 'optimise':
    result = least_squares(obj_fn, param0, ftol=1e-5, bounds=(lower, upper))
    # print optimum parameters for pasting into the model
    print('{')
    for name, value in zip(param_names, result.x):
        print(f"    '{name}': {value:.9f},")
    print('}')

else:
    obj_fn(param0)
