import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from batFunctions_w import qualityMethodBins, newMethodBins
from paths import PROJECT_ROOT


# read data and ratingsT20
bat_data = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/data/combinedBatDataClean.csv', parse_dates=['date', 'dob'])
ratings = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/outputs/batRatingsJungle2.csv', parse_dates=['date'])
current_ratings = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/outputs/batRatingsJungle3.csv')



# revert the rep values for tailenders
ratings['rep_run_ratio'] = np.where(ratings['ord_r'] > 8, (((1 - ratings['rep_run_ratio']) / 2) * np.minimum(2, np.abs(ratings['ord_r'] - 8))) + ratings['rep_run_ratio'], ratings['rep_run_ratio'])
ratings['rep_wkt_ratio'] = np.where(ratings['ord_w'] > 8, (((1 - ratings['rep_wkt_ratio']) / 2) * np.minimum(2, np.abs(ratings['ord_w'] - 8))) + ratings['rep_wkt_ratio'], ratings['rep_wkt_ratio'])

# filter out bat data dummies and first innings of careers
bat_data = bat_data[(bat_data['balls_faced'] > 0) & (bat_data['balls_faced_career'] > 1)]

# merge ratingsT20 into bat data and remove balls for which there is no rating
bat_data = bat_data.merge(ratings[ratings['i_balls_faced'] > 0].loc[:, ['playerid', 'date', 'matchid', 'balls_faced_r', 'run_rating', 'run_rating_2', 'rep_run_ratio', 'weight_balls_r', 'balls_faced_w', 'wkt_rating',
                                                                        'wkt_rating_2', 'rep_wkt_ratio', 'weight_balls_w']],
                                                                        how='left', on=['playerid', 'date', 'matchid'])
bat_data = bat_data[(bat_data['run_rating'] >= 0)]
bat_data = bat_data[(bat_data['wkt_rating'] >= 0)]
# drop nan's or we get an error in the model, this nan for weight balls occurs for players who play ODI before t20
bat_data = bat_data.dropna(subset=['rep_run_ratio', 'weight_balls_r'])
bat_data = bat_data.dropna(subset=['rep_wkt_ratio', 'weight_balls_w'])

# new model with more degrees, more accurate
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

bat_data['balls_for_weight_r'] = np.minimum(bat_data['balls_faced_career'], np.maximum(0, bat_data['balls_faced_career_exp_r'] + np.minimum(400, np.maximum(0, (bat_data['weight_balls_r'] * 0.857143) - 214.2857)))) # this last clause is so we get more strict application of rep values where weighted balls are low but career balls are high, the decimals were worked out by trial and error
bat_data['balls_for_weight_w'] = np.minimum(bat_data['balls_faced_career'], np.maximum(0, bat_data['balls_faced_career_exp_w'] + np.minimum(400, np.maximum(0, (bat_data['weight_balls_w'] * 0.857143) - 214.2857))))
ratings['balls_for_weight_r'] = np.minimum(ratings['balls_faced_career'], np.maximum(0, ratings['balls_faced_career_exp_r'] + np.minimum(400, np.maximum(0, (ratings['weight_balls_r'] * 0.857143) - 214.2857))))
ratings['balls_for_weight_w'] = np.minimum(ratings['balls_faced_career'], np.maximum(0, ratings['balls_faced_career_exp_w'] + np.minimum(400, np.maximum(0, (ratings['weight_balls_w'] * 0.857143) - 214.2857))))



# merge the first set of outputs into bat data, we'll use these to group the bins for the error measurement
bat_data = bat_data.merge(current_ratings.loc[:, ['playerid', 'matchid', 'run_rating_3', 'wkt_rating_3', 'host', 'competition']], on=['playerid', 'matchid', 'host', 'competition'], how='left')
# we'll now use bat data for error measurement, so drop odi and invalid balls, drop nans
bat_data = bat_data[(bat_data['competition'] != 'ODI') & (bat_data['balls_faced'] > 0)].dropna(subset=['run_rating_3', 'wkt_rating_3'])




# tables for appending to from within the function
ratingsOuter = []
pivotOuter = []
bat_dataOuter = []
opt_history = pd.DataFrame(columns=['rmse', 'sse_mean', 'sse_total', 'param0','param1','param2','param3'])






def optimize_weights(kind, param):
    # kind: 'r' for runs, 'w' for wickets
    if kind not in ('r', 'w'):
        raise ValueError("kind must be 'r' (runs) or 'w' (wickets)")

    k, a, x, yi = param[0], param[1], param[2], param[3]

    balls_for_weight_col = f"balls_for_weight_{kind}"
    rep_ratio_col = "rep_run_ratio" if kind == 'r' else "rep_wkt_ratio"
    base_rating_col = "run_rating" if kind == 'r' else "wkt_rating"
    rating3_col = "run_rating_3" if kind == 'r' else "wkt_rating_3"
    exp_col = "realexprbat" if kind == 'r' else "realexpwbat"
    pred_col = "runs_pred" if kind == 'r' else "wkt_pred"
    actual_col = "runs" if kind == 'r' else "wkt"

    # Weight and blended rating on outputs (same in-place behavior as originals)
    ratings['weight'] = np.maximum(
        yi,
        np.maximum((1 - k) ** ratings[balls_for_weight_col], a - (x * ratings[balls_for_weight_col]))
    )
    ratings[rating3_col] = (ratings[rep_ratio_col] * ratings['weight']) + ((1 - ratings['weight']) * ratings[base_rating_col])
    ratings_i = ratings.copy()

    # Merge outputs to row level
    bat_data_i = bat_data.copy()
    bat_data_i = bat_data_i.merge(
        ratings_i.loc[:, ['playerid', 'matchid', 'host', 'date', 'competition', rating3_col]],
        how='left',
        on=['playerid', 'host', 'date', 'competition', 'matchid']
    )

    # Prediction
    bat_data_i[pred_col] = bat_data_i[exp_col] * bat_data_i[rating3_col]

    # Keep rows for evaluation
    bat_data_i = bat_data_i.dropna(subset=[rating3_col, actual_col, exp_col])

    # Bin-level pivot and residuals
    pivot = (
        bat_data_i.groupby('binid', as_index=False)
        .agg(
            rating_avg=(rating3_col, 'mean'),
            balls_faced=('balls_faced', 'sum'),
            exp_sum=(exp_col, 'sum'),
            pred_sum=(pred_col, 'sum'),
            actual_sum=(actual_col, 'sum'),
        )
        .assign(bin_residual=lambda df: df['pred_sum'] - df['actual_sum'])
        .sort_values('binid')
    )
    pivot = pivot[pivot['balls_faced'] > 30]

    # Metrics
    residual = pivot['bin_residual'].to_numpy()
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    sse_mean = float(np.mean(residual ** 2))
    sse_total = float(np.sum(residual ** 2))
    print(f"RMSE={rmse:.6f}, mean_SSE={sse_mean:.6f}")

    # Side outputs for inspection (preserve original behavior)
    ratingsOuter.clear(); ratingsOuter.append(ratings_i)
    pivotOuter.clear(); pivotOuter.append(pivot)
    bat_dataOuter.clear(); bat_dataOuter.append(bat_data_i)

    # Log to opt_history (DataFrame defined outside)
    row = {
        'rmse': rmse,
        'sse_mean': sse_mean,
        'sse_total': sse_total,
        **{f'param{i}': float(p) for i, p in enumerate(param)}
    }
    opt_history.loc[len(opt_history)] = row

    # Append tables from within the function to outer tables for investigation (same as originals)
    ratingsOuter.clear(); ratingsOuter.append(ratings)
    pivotOuter.clear(); pivotOuter.append(pivot)
    bat_dataOuter.clear(); bat_dataOuter.append(bat_data_i)

    return residual






# original params and result
# param0 = [0.002597, 0.611901, 0.000757, 0.02]   # jungle runs
# param0 = [0.001296, 0.611901, 0.000757, 0.02]   # rasoi runs
# 0.003979750202396137,0.3407820032023097,0.000757,0.18511756258555656

# param0 = [0.000942, 0.8874, 0.000957, 0.02]   # jungle wickets
# param0 = [0.000942, 0.8874, 0.000957, 0.02]   # rasoi wickets


# choose the method of grouping for measuring error
# bat_data = qualityMethodBins(bat_data, bin_size=40, rating_col='run_rating_3', out_col='binid')
bat_data = newMethodBins(bat_data)
bat_data = bat_data.drop(columns=['run_rating_3', 'wkt_rating_3'])



# optimisation, if you want to do runs write 'r', for wickets write 'w'
optimise = least_squares(
    lambda p: optimize_weights('w', p),
    param0,
    ftol=1e-8,
    bounds=([0, 0, 0, 0], [1, 1, 1, 1]),
)

# # test evaluation of params
# test_r = optimize_weights_r(param0)

