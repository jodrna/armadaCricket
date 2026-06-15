import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from batFunctions_w import qualityMethodBins, newMethodBins
from paths import PROJECT_ROOT


# -------------------------
# Configure
# -------------------------
model_name = 'jungle'     # jungle / rasoi
target = 'wkts'           # runs / wkts
mode = 'optimise'         # test / optimise


# -------------------------
# Imports
# -------------------------
bat_data = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/data/batDataCombinedClean_w.csv', parse_dates=['date', 'dob'])

if model_name == 'jungle':
    ratings = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/outputs/batRatingsJungle2_w.csv', parse_dates=['date'])
    current_ratings = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/outputs/batRatingsJungle3_w.csv')

else:
    ratings = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/outputs/batRatingsRasoi2_w.csv', parse_dates=['date'])
    current_ratings = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/outputs/batRatingsRasoi3_w.csv')


# -------------------------
# Tailender replacement adjustment
# -------------------------
ratings['rep_run_ratio'] = np.where(ratings['ord_r'] > 8, (((1 - ratings['rep_run_ratio']) / 2) * np.minimum(2, np.abs(ratings['ord_r'] - 8))) + ratings['rep_run_ratio'], ratings['rep_run_ratio'])
ratings['rep_wkt_ratio'] = np.where(ratings['ord_w'] > 8, (((1 - ratings['rep_wkt_ratio']) / 2) * np.minimum(2, np.abs(ratings['ord_w'] - 8))) + ratings['rep_wkt_ratio'], ratings['rep_wkt_ratio'])


# -------------------------
# Merge ratings into bat data
# -------------------------
bat_data = bat_data[(bat_data['balls_faced'] > 0) & (bat_data['balls_faced_career'] > 1)]

rating_cols = ['playerid', 'date', 'matchid', 'balls_faced_r', 'run_rating', 'run_rating_2', 'rep_run_ratio', 'weight_balls_r', 'balls_faced_w', 'wkt_rating', 'wkt_rating_2', 'rep_wkt_ratio', 'weight_balls_w']
bat_data = bat_data.merge(ratings[ratings['i_balls_faced'] > 0].loc[:, rating_cols], how='left', on=['playerid', 'date', 'matchid'])
bat_data = bat_data[(bat_data['run_rating'] >= 0)]
bat_data = bat_data[(bat_data['wkt_rating'] >= 0)]
bat_data = bat_data.dropna(subset=['rep_run_ratio', 'weight_balls_r'])
bat_data = bat_data.dropna(subset=['rep_wkt_ratio', 'weight_balls_w'])


# -------------------------
# Career balls model
# -------------------------
poly_features = PolynomialFeatures(degree=3)

X = bat_data[['weight_balls_r']]
X2 = bat_data[['weight_balls_w']]
X = poly_features.fit_transform(X)
X2 = poly_features.fit_transform(X2)
y = bat_data.loc[:, ['balls_faced_career']].values.ravel()

model = LinearRegression()
model = model.fit(X, y)

model2 = LinearRegression()
model2 = model2.fit(X2, y)

bat_data['balls_faced_career_exp_r'] = model.predict(X)
ratings.dropna(subset=['weight_balls_r'], inplace=True)
Xp = ratings[['weight_balls_r']]
Xp = poly_features.fit_transform(Xp)
ratings['balls_faced_career_exp_r'] = model.predict(Xp)

bat_data['balls_faced_career_exp_w'] = model2.predict(X2)
ratings.dropna(subset=['weight_balls_w'], inplace=True)
X2p = ratings[['weight_balls_w']]
X2p = poly_features.fit_transform(X2p)
ratings['balls_faced_career_exp_w'] = model2.predict(X2p)

test_r = bat_data.groupby(['weight_balls_r'])['balls_faced_career_exp_r'].mean().reset_index()
test_r.sort_values(by=['weight_balls_r'], ascending=[True], inplace=True)

test_w = bat_data.groupby(['weight_balls_w'])['balls_faced_career_exp_w'].mean().reset_index()
test_w.sort_values(by=['weight_balls_w'], ascending=[True], inplace=True)


# -------------------------
# Balls for reversion weight
# -------------------------
bat_data['balls_for_weight_r'] = np.minimum(bat_data['balls_faced_career'], np.maximum(0, bat_data['balls_faced_career_exp_r'] + np.minimum(400, np.maximum(0, (bat_data['weight_balls_r'] * 0.857143) - 214.2857))))
bat_data['balls_for_weight_w'] = np.minimum(bat_data['balls_faced_career'], np.maximum(0, bat_data['balls_faced_career_exp_w'] + np.minimum(400, np.maximum(0, (bat_data['weight_balls_w'] * 0.857143) - 214.2857))))

ratings['balls_for_weight_r'] = np.minimum(ratings['balls_faced_career'], np.maximum(0, ratings['balls_faced_career_exp_r'] + np.minimum(400, np.maximum(0, (ratings['weight_balls_r'] * 0.857143) - 214.2857))))
ratings['balls_for_weight_w'] = np.minimum(ratings['balls_faced_career'], np.maximum(0, ratings['balls_faced_career_exp_w'] + np.minimum(400, np.maximum(0, (ratings['weight_balls_w'] * 0.857143) - 214.2857))))


# -------------------------
# Target config
# -------------------------
target_cfg = {
    'runs': {
        'balls_for_weight_col': 'balls_for_weight_r',
        'rep_ratio_col': 'rep_run_ratio',
        'weight_col': 'rep_run_weight',
        'base_rating_col': 'run_rating',
        'rating_col_3': 'run_rating_3',
        'exp_col': 'realexprbat',
        'actual_col': 'runs',
        'pred_col': 'runs_pred'
    },
    'wkts': {
        'balls_for_weight_col': 'balls_for_weight_w',
        'rep_ratio_col': 'rep_wkt_ratio',
        'weight_col': 'rep_wkt_weight',
        'base_rating_col': 'wkt_rating',
        'rating_col_3': 'wkt_rating_3',
        'exp_col': 'realexpwbat',
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
            'k': 0.002597,
            'a': 0.611901,
            'x': 0.000757,
            'y': 0.02
        },
        'wkts': {
            'k': 0.000942,
            'a': 0.8874,
            'x': 0.000957,
            'y': 0.02
        }
    },
    'rasoi': {
        'runs': {
            'k': 0.001296,
            'a': 0.611901,
            'x': 0.000757,
            'y': 0.02
        },
        'wkts': {
            'k': 0.000942,
            'a': 0.8874,
            'x': 0.000957,
            'y': 0.02
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
bat_data = bat_data.merge(current_ratings.loc[:, ['playerid', 'matchid', 'run_rating_3', 'wkt_rating_3', 'host', 'competition']], on=['playerid', 'matchid', 'host', 'competition'], how='left')
bat_data = bat_data[(bat_data['competition'] != 'ODI') & (bat_data['balls_faced'] > 0)].dropna(subset=['run_rating_3', 'wkt_rating_3'])

if model_name == 'jungle':
    bat_data = qualityMethodBins(bat_data, bin_size=40, rating_col=rating_col_3, out_col='binid')

else:
    bat_data = newMethodBins(bat_data, bin_size=40, rating_col=rating_col_3, out_col='binid')

bat_data = bat_data.drop(columns=['run_rating_3', 'wkt_rating_3'])


# -------------------------
# Tables for debugging
# -------------------------
ratingsOuter = []
pivotOuter = []
bat_dataOuter = []


def optimise_params(param, ratings, bat_data, bin_col='binid'):
    k = param[0]
    a = param[1]
    x = param[2]
    y = param[3]

    ratings_i = ratings.copy()
    ratings_i[weight_col] = np.maximum(y, np.maximum((1 - k) ** ratings_i[balls_for_weight_col], a - (x * ratings_i[balls_for_weight_col])))
    ratings_i[rating_col_3] = (ratings_i[rep_ratio_col] * ratings_i[weight_col]) + ((1 - ratings_i[weight_col]) * ratings_i[base_rating_col])

    bat_data_i = bat_data.copy()
    bat_data_i = bat_data_i.merge(ratings_i.loc[:, ['playerid', 'matchid', 'host', 'date', 'competition', rating_col_3]], how='left', on=['playerid', 'host', 'date', 'competition', 'matchid'])
    bat_data_i[pred_col] = bat_data_i[exp_col] * bat_data_i[rating_col_3]
    bat_data_i = bat_data_i.dropna(subset=[rating_col_3, actual_col, exp_col])

    pivot = bat_data_i.groupby(bin_col, as_index=False).agg(rating_avg=(rating_col_3, 'mean'),
                                                            balls_faced=('balls_faced', 'sum'),
                                                            exp_sum=(exp_col, 'sum'),
                                                            pred_sum=(pred_col, 'sum'),
                                                            actual_sum=(actual_col, 'sum')).assign(bin_residual=lambda df: df['pred_sum'] - df['actual_sum']).sort_values(bin_col)
    pivot = pivot[pivot['balls_faced'] > 30]

    residual = pivot['bin_residual'].to_numpy()
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    sse_mean = float(np.mean(residual ** 2))
    sse_total = float(np.sum(residual ** 2))
    print(f"RMSE={rmse:.6f}, mean_SSE={sse_mean:.6f}")

    ratingsOuter.clear(); ratingsOuter.append(ratings_i)
    pivotOuter.clear(); pivotOuter.append(pivot)
    bat_dataOuter.clear(); bat_dataOuter.append(bat_data_i)

    row = {'rmse': rmse, 'sse_mean': sse_mean, 'sse_total': sse_total, **{param_names[i]: float(p) for i, p in enumerate(param)}}
    opt_history.loc[len(opt_history)] = row

    return residual


# -------------------------
# Optimiser config + objective
# -------------------------
optimiser_cfg = dict(
    ratings=ratings,
    bat_data=bat_data,
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


