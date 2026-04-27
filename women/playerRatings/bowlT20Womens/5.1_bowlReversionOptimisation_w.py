import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from bowlFunctions_w import qualityMethodBins, newMethodBins
from paths import PROJECT_ROOT


# read data and ratingsT20
bowl_data = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/data/bowlDataCombinedClean_w.csv', parse_dates=['date', 'dob'])
ratings = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/outputs/bowlRatingsJungle2_w.csv', parse_dates=['date'])
current_ratings = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/outputs/bowlRatingsJungle3_w.csv')


# revert the rep values for tailenders
ratings['rep_run_ratio'] = np.where(ratings['ord_r'] > 8, (((1 - ratings['rep_run_ratio']) / 2) * np.minimum(2, np.abs(ratings['ord_r'] - 8))) + ratings['rep_run_ratio'], ratings['rep_run_ratio'])
ratings['rep_wkt_ratio'] = np.where(ratings['ord_w'] > 8, (((1 - ratings['rep_wkt_ratio']) / 2) * np.minimum(2, np.abs(ratings['ord_w'] - 8))) + ratings['rep_wkt_ratio'], ratings['rep_wkt_ratio'])

# filter out bowl data dummies and first innings of careers
bowl_data = bowl_data[(bowl_data['balls_bowled'] > 0) & (bowl_data['balls_bowled_career'] > 1)]

# merge ratingsT20 into bowl data and remove balls for which there is no rating
bowl_data = bowl_data.merge(ratings[ratings['i_balls_bowled'] > 0].loc[:, ['playerid', 'date', 'matchid', 'host', 'competition', 'balls_bowled_r', 'run_rating', 'rep_run_ratio', 'balls_bowled_w', 'wkt_rating', 'rep_wkt_ratio']],
                                                                        how='left', on=['playerid', 'date', 'matchid', 'host', 'competition'])


bowl_data = bowl_data[(bowl_data['run_rating'] >= 0)]
bowl_data = bowl_data[(bowl_data['wkt_rating'] >= 0)]
# drop nan's or we get an error in the model, this nan for weight balls occurs for players who play ODI before t20
bowl_data = bowl_data.dropna(subset=['rep_run_ratio'])
bowl_data = bowl_data.dropna(subset=['rep_wkt_ratio'])



# merge the first set of outputs into bowl data, we'll use these to group the bins for the error measurement
bowl_data = bowl_data.merge(current_ratings.loc[:, ['playerid', 'matchid', 'run_rating_3', 'wkt_rating_3', 'host', 'competition']], on=['playerid', 'matchid', 'host', 'competition'], how='left')
# we'll now use bowl data for error measurement, so drop odi and invalid balls, drop nans
bowl_data = bowl_data[(bowl_data['competition'] != 'ODI') & (bowl_data['balls_bowled'] > 0)].dropna(subset=['run_rating_3', 'wkt_rating_3'])




# tables for appending to from within the function
ratingsOuter = []
pivotOuter = []
bowl_dataOuter = []
opt_history = pd.DataFrame(columns=['rmse', 'sse_mean', 'sse_total', 'param0','param1','param2','param3'])




def optimize_weights(kind, param):
    # kind: 'r' for runs, 'w' for wickets
    if kind not in ('r', 'w'):
        raise ValueError("kind must be 'r' (runs) or 'w' (wickets)")

    k, a, x, yi = param[0], param[1], param[2], param[3]

    balls_for_weight_col = 'balls_bowled_r' if kind == 'r' else 'balls_bowled_w'
    rep_ratio_col = 'rep_run_ratio' if kind == 'r' else 'rep_wkt_ratio'
    base_rating_col = 'run_rating' if kind == 'r' else 'wkt_rating'
    rating3_col = 'run_rating_3' if kind == 'r' else 'wkt_rating_3'
    exp_col = 'realexprbowl' if kind == 'r' else 'realexpwbowl'
    pred_col = 'runs_pred' if kind == 'r' else 'wkt_pred'
    actual_col = 'runs' if kind == 'r' else 'wkt'

    # Weight and blended rating on outputs (same in-place behavior as originals)
    ratings['weight'] = np.maximum(
        yi,
        np.maximum((1 - k) ** ratings[balls_for_weight_col], a - (x * ratings[balls_for_weight_col]))
    )
    ratings[rating3_col] = (ratings[rep_ratio_col] * ratings['weight']) + ((1 - ratings['weight']) * ratings[base_rating_col])
    ratings_i = ratings.copy()

    # Merge outputs to row level
    bowl_data_i = bowl_data.copy()
    bowl_data_i = bowl_data_i.merge(
        ratings_i.loc[:, ['playerid', 'matchid', 'host', 'date', 'competition', rating3_col]],
        how='left',
        on=['playerid', 'host', 'date', 'competition', 'matchid']
    )

    # Prediction
    bowl_data_i[pred_col] = bowl_data_i[exp_col] * bowl_data_i[rating3_col]

    # Keep rows for evaluation
    bowl_data_i = bowl_data_i.dropna(subset=[rating3_col, actual_col, exp_col])

    # Bin-level pivot and residuals
    pivot = (
        bowl_data_i.groupby('binid', as_index=False)
        .agg(
            rating_avg=(rating3_col, 'mean'),
            balls_bowled=('balls_bowled', 'sum'),
            exp_sum=(exp_col, 'sum'),
            pred_sum=(pred_col, 'sum'),
            actual_sum=(actual_col, 'sum'),
        )
        .assign(bin_residual=lambda df: df['pred_sum'] - df['actual_sum'])
        .sort_values('binid')
    )
    pivot = pivot[pivot['balls_bowled'] > 30]

    # Metrics
    residual = pivot['bin_residual'].to_numpy()
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    sse_mean = float(np.mean(residual ** 2))
    sse_total = float(np.sum(residual ** 2))
    print(f"RMSE={rmse:.6f}, mean_SSE={sse_mean:.6f}")

    # Side outputs for inspection (preserve original behavior)
    ratingsOuter.clear(); ratingsOuter.append(ratings_i)
    pivotOuter.clear(); pivotOuter.append(pivot)
    bowl_dataOuter.clear(); bowl_dataOuter.append(bowl_data_i)

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
    bowl_dataOuter.clear(); bowl_dataOuter.append(bowl_data_i)

    return residual






# original params and result
param0 = [.1057993, .75375, .00015, 0.1]    # runs
# param0 = [.0045298, .88573, .000075, 0.03]  # wickets

# choose the method of grouping for measuring error
bowl_data = qualityMethodBins(bowl_data, bin_size=40, rating_col='run_rating_3', out_col='binid')
# bowl_data = newMethodBins(bowl_data)
bowl_data = bowl_data.drop(columns=['run_rating_3', 'wkt_rating_3'])



# optimisation, if you want to do runs write 'r', for wickets write 'w'
optimise = least_squares(
    lambda p: optimize_weights('r', p),
    param0,
    ftol=1e-4,
    bounds=([0, 0, 0, 0], [1, 1, 1, 1]),
)

# # test evaluation of params
# test_r = optimize_weights('r', param0)


