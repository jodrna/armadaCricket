import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from batFunctions_w import qualityMethodBins, newMethodBins, buildRunRatingsOriginal, buildRunRatingsMapOne, buildRunRatingsMapTwo, buildRunRatingsMapPriority, buildWktRatingsMapPriority, buildWktRatingsOriginal
from paths import PROJECT_ROOT

bat_data = pd.read_csv(PROJECT_ROOT / 'OneDrive - Decimal Data Services Ltd/player_ratings/bat_t20_womens/all/data/combinedBatDataClean.csv', parse_dates=['date', 'dob'])
# n2h_factors = pd.read_csv(PROJECT_ROOT / 'OneDrive - Decimal Data Services Ltd/player_ratings/bat_t20_womens/all/auxiliaries/batN2HFactors.csv')
# n2h_factors = n2h_factors.loc[:, ['nationality', 'host_2', 'host', 'run_factor', 'wkt_factor']]
# coeff_adjust = pd.read_csv(PROJECT_ROOT / 'OneDrive - Decimal Data Services Ltd/player_ratings/bat_t20_womens/all/auxiliaries/batN2HFactorsCoeffs.csv')
# coeff_values = coeff_adjust.mean().to_dict()  # this makes a dictionary of values, the mean of each column (there is only one value for each column) named the name of each column. I can call on these values later
# current_ratings = pd.read_csv(PROJECT_ROOT / 'OneDrive - Decimal Data Services Ltd/player_ratings/bat_t20_womens/all/outputs/batRatingsPlayer3.csv')

# split odi innings
bat_data['competition'] = np.where(bat_data['competition'] == 'ODI', np.where(bat_data['ballsremaining'] < 84, 'ODI2', 'ODI1'), bat_data['competition'])

# make format in the dummy games 'T20'
bat_data['format'] = bat_data['format'].fillna('t20')

# we create 2 dataframes and merge so we have all innings, we need 2 because creating a pivot excludes the dummy innings, so to get them do a remove duplicates on bat data
innings_info = bat_data.loc[:, ['date', 'matchid', 'playerid', 'batsman', 'nationality', 'competition', 'host', 'host_region', 'balls_faced_career',
                                'balls_faced_host']].drop_duplicates(['matchid', 'playerid', 'date', 'host', 'competition'])
innings_perf = pd.pivot_table(bat_data, values=['balls_faced', 'runs', 'realexprbat', 'wkt', 'realexpwbat'], index=['date', 'matchid', 'playerid', 'competition', 'host', 'ord'], aggfunc='sum').reset_index()
innings = innings_info.merge(innings_perf, how='left', left_on=['date', 'matchid', 'playerid', 'competition', 'host'], right_on=['date', 'matchid', 'playerid', 'competition', 'host'])

# now we have df of all innings, we merge them to themselves, so for every inning's we have all other innings also, and then drop the ones after the date of the predicting inning
# lookbacks is every innings a player has played duplicated *x, x being the number of innings he has played total
lookbacks_player = innings.set_index('playerid').merge(innings.set_index('playerid'), how='left', left_index=True, right_index=True, suffixes=('', '_2')).reset_index()
lookbacks_player = lookbacks_player[lookbacks_player['date'] > lookbacks_player['date_2']]  # date is the date of the innings we are predicting, drop innings after it
lookbacks_player = lookbacks_player[~lookbacks_player['competition'].isin(['ODI1', 'ODI2'])]


# Calculate the difference in days and create a new column
lookbacks_player['date'] = pd.to_datetime(lookbacks_player['date'])
lookbacks_player['date_2'] = pd.to_datetime(lookbacks_player['date_2'])
lookbacks_player['days_ago'] = (lookbacks_player['date'] - lookbacks_player['date_2']).dt.days
lookbacks_player['balls_ago'] = lookbacks_player['balls_faced_career'] - lookbacks_player['balls_faced_career_2']

bat_weightings = pd.read_csv(PROJECT_ROOT / 'OneDrive - Decimal Data Services Ltd/player_ratings/bat_t20_womens/all/auxiliaries/batWeightings.csv')
bat_data = bat_data.merge(bat_weightings, on='balls_faced_career', how='left')
bat_data['runs_weight_curve'] = bat_data['runs_weight_curve'].fillna(1)
bat_data['wkts_weight_curve'] = bat_data['wkts_weight_curve'].fillna(1)


avg_ord = bat_data.groupby(['playerid', 'batsman'])['ord'].mean().reset_index()
avg_ord.rename(columns={'ord': 'avg_ord'}, inplace=True)
lookbacks_player = lookbacks_player.merge(avg_ord, on=('playerid', 'batsman'), how='left')


# # merge the first set of outputs into bat data, we'll use these to group the bins for the error measurement
# bat_data = bat_data.merge(current_ratings.loc[:, ['playerid', 'matchid', 'run_rating_3', 'wkt_rating_3', 'host', 'competition']], on=['playerid', 'matchid', 'host', 'competition'], how='left')
# # we'll now use bat data for error measurement, so drop odi and invalid balls, drop nans
# bat_data = bat_data[(bat_data['competition'] != 'ODI') & (bat_data['balls_faced'] > 0)].dropna(subset=['run_rating_3', 'wkt_rating_3'])



# tables for appending to from within the function
ratingsOuter = []
pivotOuter = []
bat_dataOuter = []
lookbacksOuter = []
opt_history = pd.DataFrame(columns=['rmse', 'sse_mean', 'sse_total', 'param0','param1','param2','param3','param4','param5','param6','param7', 'param8', 'param9'])

def optimise_params(param, lookbacks_player, build_fn,
                              rating_col, exp_col, actual_col, weight_curve_col,
                              out_pred_col, bin_col='binid'):
    # Build outputs/lookbacks with the provided builder
    ratings_i, lookbacks_i = build_fn(param, lookbacks_player)

    # Merge outputs to row level
    bat_data_i = bat_data.copy()
    bat_data_i = bat_data_i.merge(
        ratings_i.loc[:, ['playerid', 'matchid', 'host', 'date', 'competition', rating_col]],
        how='left',
        on=['playerid', 'host', 'date', 'competition', 'matchid']
    )

    # Prediction
    bat_data_i[out_pred_col] = bat_data_i[exp_col] * bat_data_i[rating_col]

    # Keep rows for evaluation
    bat_data_i = bat_data_i.dropna(subset=[rating_col, actual_col, exp_col, weight_curve_col])

    # Scale at row level, then compute bin-level residuals from weighted sums
    bat_data_i = bat_data_i.assign(
        pred_w=lambda d: d[out_pred_col] * d[weight_curve_col],
        act_w=lambda d: d[actual_col] * d[weight_curve_col]
    )

    pivot = (
        bat_data_i.groupby(bin_col, as_index=False)
          .agg(
            rating_avg=(rating_col, 'mean'),
            balls_faced=('balls_faced', 'sum'),
            exp_sum=(exp_col, 'sum'),
            pred_sum=(out_pred_col, 'sum'),
            actual_sum=(actual_col, 'sum'),
            weight_curve_avg=(weight_curve_col, 'mean'),
            pred_w=('pred_w', 'sum'),
            act_w=('act_w', 'sum'),

        )
    .assign(bin_residual=lambda df: df['pred_w'] - df['act_w'])
          .sort_values(bin_col)
    )
    pivot = pivot[pivot['balls_faced'] > 30]

    # Metrics
    residual = pivot['bin_residual'].to_numpy()
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    sse_mean = float(np.mean(residual ** 2))
    sse_total = float(np.sum(residual ** 2))
    print(f"RMSE={rmse:.6f}, mean_SSE={sse_mean:.6f}")

    # Side outputs for inspection
    ratingsOuter.clear(); ratingsOuter.append(ratings_i)
    pivotOuter.clear(); pivotOuter.append(pivot)
    bat_dataOuter.clear(); bat_dataOuter.append(bat_data_i)
    lookbacksOuter.clear(); lookbacksOuter.append(lookbacks_i.drop_duplicates(subset=['location_weight'], keep='first'))

    # Log to opt_history (DataFrame defined outside)
    row = {
        'rmse': rmse,
        'sse_mean': sse_mean,
        'sse_total': sse_total,
        **{f'param{i}': float(p) for i, p in enumerate(param)}
    }
    opt_history.loc[len(opt_history)] = row

    return residual





# # params optimised for runs using quality grouping.....same tournament, same comp, same comp t20i, t20, odi2, odi1, dh, h, r, k
# param0 = [15.17348002,	6.89753,	12.88380525,	6.307514689,	1.692501798,	1.000755457,	0.621652589,	1.241297181,	1.381979717,	0.000496309]

# # params optimised for wkts using quality grouping.....same tournament, same comp, same comp t20i, t20, odi2, odi1, dh, h, r, k
# param0 = [1.293158205,	3.979704835,	1.80296017,	2.070824639,	1.518191934,	1.071059076,	0.97542588,	1.477165773,	1,	0.000499531]


# # params optimised for runs using innings grouping.....same tournament, same comp, same comp t20i, t20, odi2, odi1, dh, h, r, k
# param0 = [20,	12.59457633,	17.46079646,	7.338761994,	2.72804768,	1,	0.469010748,	1,	1.444896715,	0.000802191]

# # params optimised for wkts using innings grouping.....same tournament, same comp, same comp t20i, t20, odi2, odi1, dh, h, r, k
# param0 = [10.5339096,	20,	5.233032976,	7.782822175,	1,	4.423330281,	0.977275005,	1.589376541,	1.080660285,	0.000850588]



# # params for original method on runs, based on 'new' grouping, k, ci, cd, h, r, t20, odi1, odi2, t
# param0 = [0.000900, 1.637170, 1.377430, 1.247840, 1.099910, 1.184380, 0, 1.000000, 1.000000]
# # params for original method on wickets, based on 'new' grouping, k, ci, cd, h, r, t20, odi1, odi2, t
# param0 = [0.000680, 1.100830, 1.787410, 1.361030, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000]


# standard starting params
param0 = [1, 1, 1, 1, 1, 1, 0, 1, 1, 0]
# bounds for optimization
lower = [
    1,    # t
    1,    # cd
    1,    # ci
    1,    # t20
    1,    # odi2
    1,    # odi1
    0,    # dh
    1,    # h
    1,    # r
    0     # k
]
upper = [
    20,    # t
    20,    # cd
    20,    # ci
    20,    # t20
    20,    # odi2
    20,    # odi1
    1,     # dh
    20,    # h
    20,    # r
    0.01   # k
]


# choose the method of grouping for measuring error
# bat_data = qualityMethodBins(bat_data, bin_size=40, rating_col='run_rating_3', out_col='binid')
bat_data = newMethodBins(bat_data)


# One line, used for both the test run and the optimizer
cfg = dict(
    lookbacks_player=lookbacks_player,
    build_fn=buildWktRatingsMapPriority,                    # the method of creating the outputs
    rating_col='wkt_rating',
    exp_col='realexpwbat',
    actual_col='wkt',
    weight_curve_col='wkts_weight_curve',
    out_pred_col='wkts_pred',
    bin_col='binid'
)
obj_fn = lambda p: optimise_params(p, **cfg)


# optimisation
optimise = least_squares(
obj_fn,
    param0,
    ftol=1e-4,
    bounds=(lower, upper)
)


# # test evaluation of params
# test = obj_fn(param0)

