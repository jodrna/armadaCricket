import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from bowlFunctions_w import qualityMethodBins, newMethodBins
from paths import PROJECT_ROOT


# -------------------------
# Configure
# -------------------------
model_name = 'jungle'     # jungle / rasoi
target = 'runs'           # runs / wkts
mode = 'optimise'         # test / optimise


# -------------------------
# Imports
# -------------------------
bowl_data = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/data/bowlDataCombinedClean_w.csv', parse_dates=['date', 'dob'])

if model_name == 'jungle':
    ratings = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/outputs/bowlRatingsJungle2_w.csv', parse_dates=['date'])
    current_ratings = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/outputs/bowlRatingsJungle3_w.csv')

else:
    ratings = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/outputs/bowlRatingsRasoi2_w.csv', parse_dates=['date'])
    current_ratings = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/outputs/bowlRatingsRasoi3_w.csv')


# -------------------------
# Tailender replacement adjustment
# -------------------------
ratings['rep_run_ratio'] = np.where(ratings['ord_r'] > 8, (((1 - ratings['rep_run_ratio']) / 2) * np.minimum(2, np.abs(ratings['ord_r'] - 8))) + ratings['rep_run_ratio'], ratings['rep_run_ratio'])
ratings['rep_wkt_ratio'] = np.where(ratings['ord_w'] > 8, (((1 - ratings['rep_wkt_ratio']) / 2) * np.minimum(2, np.abs(ratings['ord_w'] - 8))) + ratings['rep_wkt_ratio'], ratings['rep_wkt_ratio'])


# -------------------------
# Merge ratings into bowl data
# -------------------------
bowl_data = bowl_data[(bowl_data['balls_bowled'] > 0) & (bowl_data['balls_bowled_career'] > 1)]

rating_cols = ['playerid', 'date', 'matchid', 'host', 'competition', 'balls_bowled_r', 'run_rating', 'rep_run_ratio', 'balls_bowled_w', 'wkt_rating', 'rep_wkt_ratio']
bowl_data = bowl_data.merge(ratings[ratings['i_balls_bowled'] > 0].loc[:, rating_cols], how='left', on=['playerid', 'date', 'matchid', 'host', 'competition'])
bowl_data = bowl_data[(bowl_data['run_rating'] >= 0)]
bowl_data = bowl_data[(bowl_data['wkt_rating'] >= 0)]
bowl_data = bowl_data.dropna(subset=['rep_run_ratio'])
bowl_data = bowl_data.dropna(subset=['rep_wkt_ratio'])


# -------------------------
# Target config
# -------------------------
target_cfg = {
    'runs': {
        'balls_for_weight_col': 'balls_bowled_r',
        'rep_ratio_col': 'rep_run_ratio',
        'weight_col': 'rep_run_weight',
        'base_rating_col': 'run_rating',
        'rating_col_3': 'run_rating_3',
        'exp_col': 'realexprbowl',
        'actual_col': 'runs',
        'pred_col': 'runs_pred'
    },
    'wkts': {
        'balls_for_weight_col': 'balls_bowled_w',
        'rep_ratio_col': 'rep_wkt_ratio',
        'weight_col': 'rep_wkt_weight',
        'base_rating_col': 'wkt_rating',
        'rating_col_3': 'wkt_rating_3',
        'exp_col': 'realexpwbowl',
        'actual_col': 'wkt',
        'pred_col': 'wkts_pred'
    }
}

balls_for_weight_col = target_cfg[target]['balls_for_weight_col']
rep_ratio_col = target_cfg[target]['rep_ratio_col']
weight_col = target_cfg[target]['weight_col']
base_rating_col = target_cfg[target]['base_rating_col']
rating_col_3 = target_cfg[target]['rating_col_3']
exp_col = target_cfg[target]['exp_col']
actual_col = target_cfg[target]['actual_col']
pred_col = target_cfg[target]['pred_col']


# -------------------------
# Params
# -------------------------
params = {
    'jungle': {
        'runs': {
            'k': 0.1057993,
            'a': 0.75375,
            'x': 0.00015,
            'y': 0.1
        },
        'wkts': {
            'k': 0.0045298,
            'a': 0.88573,
            'x': 0.000075,
            'y': 0.03
        }
    },
    'rasoi': {
        'runs': {
            'k': 0.1057993,
            'a': 0.75375,
            'x': 0.00015,
            'y': 0.1
        },
        'wkts': {
            'k': 0.0045298,
            'a': 0.88573,
            'x': 0.000075,
            'y': 0.03
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
lower_dict = {'k': 0, 'a': 0, 'x': 0, 'y': 0}
upper_dict = {'k': 1, 'a': 1, 'x': 1, 'y': 1}

lower = [lower_dict[name] for name in param_names]
upper = [upper_dict[name] for name in param_names]


# -------------------------
# Choose grouping method, jungle = quality, rasoi = new
# -------------------------
bowl_data = bowl_data.merge(current_ratings.loc[:, ['playerid', 'matchid', 'run_rating_3', 'wkt_rating_3', 'host', 'competition']], on=['playerid', 'matchid', 'host', 'competition'], how='left')
bowl_data = bowl_data[(bowl_data['competition'] != 'ODI') & (bowl_data['balls_bowled'] > 0)].dropna(subset=['run_rating_3', 'wkt_rating_3'])

if model_name == 'jungle':
    bowl_data = qualityMethodBins(bowl_data, bin_size=40, rating_col=rating_col_3, out_col='binid')

else:
    bowl_data = newMethodBins(bowl_data, bin_size=40, rating_col=rating_col_3, out_col='binid')

bowl_data = bowl_data.drop(columns=['run_rating_3', 'wkt_rating_3'])


# -------------------------
# Tables for debugging
# -------------------------
ratingsOuter = []
pivotOuter = []
bowl_dataOuter = []


def optimise_params(param, ratings, bowl_data, bin_col='binid'):
    k = param[0]
    a = param[1]
    x = param[2]
    y = param[3]

    ratings_i = ratings.copy()
    ratings_i[weight_col] = np.maximum(y, np.maximum((1 - k) ** ratings_i[balls_for_weight_col], a - (x * ratings_i[balls_for_weight_col])))
    ratings_i[rating_col_3] = (ratings_i[rep_ratio_col] * ratings_i[weight_col]) + ((1 - ratings_i[weight_col]) * ratings_i[base_rating_col])

    bowl_data_i = bowl_data.copy()
    bowl_data_i = bowl_data_i.merge(ratings_i.loc[:, ['playerid', 'matchid', 'host', 'date', 'competition', rating_col_3]], how='left', on=['playerid', 'host', 'date', 'competition', 'matchid'])
    bowl_data_i[pred_col] = bowl_data_i[exp_col] * bowl_data_i[rating_col_3]
    bowl_data_i = bowl_data_i.dropna(subset=[rating_col_3, actual_col, exp_col])

    pivot = bowl_data_i.groupby(bin_col, as_index=False).agg(rating_avg=(rating_col_3, 'mean'),
                                                             balls_bowled=('balls_bowled', 'sum'),
                                                             exp_sum=(exp_col, 'sum'),
                                                             pred_sum=(pred_col, 'sum'),
                                                             actual_sum=(actual_col, 'sum')).assign(bin_residual=lambda df: df['pred_sum'] - df['actual_sum']).sort_values(bin_col)
    pivot = pivot[pivot['balls_bowled'] > 30]

    residual = pivot['bin_residual'].to_numpy()
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    sse_mean = float(np.mean(residual ** 2))
    sse_total = float(np.sum(residual ** 2))
    print(f"RMSE={rmse:.6f}, mean_SSE={sse_mean:.6f}")

    ratingsOuter.clear(); ratingsOuter.append(ratings_i)
    pivotOuter.clear(); pivotOuter.append(pivot)
    bowl_dataOuter.clear(); bowl_dataOuter.append(bowl_data_i)

    row = {'rmse': rmse, 'sse_mean': sse_mean, 'sse_total': sse_total, **{param_names[i]: float(p) for i, p in enumerate(param)}}
    opt_history.loc[len(opt_history)] = row

    return residual


# -------------------------
# Optimiser config + objective
# -------------------------
optimiser_cfg = dict(
    ratings=ratings,
    bowl_data=bowl_data,
    bin_col='binid'
)

obj_fn = lambda p: optimise_params(p, **optimiser_cfg)


# -------------------------
# Optimisation/Testing
# -------------------------
if mode == 'optimise':
    result = least_squares(obj_fn, param0, ftol=1e-8, bounds=(lower, upper))

    print('{')
    for name, value in zip(param_names, result.x):
        print(f"    '{name}': {value:.9f},")
    print('}')

else:
    obj_fn(param0)
