import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from paths import PROJECT_ROOT


# -------------------------
# Reversion function
# -------------------------
def rep_weight(faced, rating, rep_ratio, param_dict):
    k = param_dict['k']
    a = param_dict['a']
    x = param_dict['x']
    y = param_dict['y']

    weight = np.maximum(y, np.maximum((1 - k) ** faced, a - (x * faced)))
    rating_2 = (rep_ratio * weight) + ((1 - weight) * rating)

    return weight, rating_2


# -------------------------
# Build outputs for jungle and rasoi
# -------------------------
for x in np.arange(0, 2, 1):
    if x == 0:
        model_name = 'jungle'
        param_r_dict = {
            'k': 0.002597,
            'a': 0.611901,
            'x': 0.000757,
            'y': 0.02
        }
        param_w_dict = {
            'k': 0.000942,
            'a': 0.8874,
            'x': 0.000957,
            'y': 0.02
        }

    else:
        model_name = 'rasoi'
        param_r_dict = {
            'k': 0.001296,
            'a': 0.611901,
            'x': 0.000757,
            'y': 0.02
        }
        param_w_dict = {
            'k': 0.000942,
            'a': 0.8874,
            'x': 0.000957,
            'y': 0.02
        }


    if x == 0:
        ratings = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/outputs/batRatingsJungle2_w.csv', parse_dates=['date'])

    else:
        ratings = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/outputs/batRatingsRasoi2_w.csv', parse_dates=['date'])

    # -------------------------
    # Tailender replacement adjustment
    # -------------------------
    ratings['rep_run_ratio'] = np.where(ratings['ord_r'] > 8, (((1 - ratings['rep_run_ratio']) / 2) * np.minimum(2, np.abs(ratings['ord_r'] - 8))) + ratings['rep_run_ratio'], ratings['rep_run_ratio'])
    ratings['rep_wkt_ratio'] = np.where(ratings['ord_w'] > 8, (((1 - ratings['rep_wkt_ratio']) / 2) * np.minimum(2, np.abs(ratings['ord_w'] - 8))) + ratings['rep_wkt_ratio'], ratings['rep_wkt_ratio'])


    # -------------------------
    # Import bat data, must do within the loop because it changes depending on jungle or rasoi
    # -------------------------
    bat_data = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/data/batDataCombinedClean_w.csv', parse_dates=['date', 'dob'])


    career_ratings = pd.pivot_table(bat_data, values=['runs', 'realexprbat', 'wkt', 'realexpwbat'], index=['playerid', 'batsman', 'format'], aggfunc='mean').reset_index()
    career_ratings['run_rating'] = career_ratings['runs'] / career_ratings['realexprbat']
    career_ratings['wkt_rating'] = career_ratings['wkt'] / career_ratings['realexpwbat']

    career_ratings = pd.pivot_table(career_ratings, values=['run_rating', 'wkt_rating'], index=['playerid', 'batsman'], columns='format', aggfunc='mean').reset_index()

    career_ratings.columns = [
        f'career_{col[1]}_{col[0]}' if col[1] else col[0]
        for col in career_ratings.columns]

    # now only t20 going forward
    bat_data = bat_data[bat_data['format'] == 't20']

    # -------------------------
    # Merge ratings into bat data
    # -------------------------
    bat_data = bat_data[(bat_data['balls_faced'] > 0) & (bat_data['balls_faced_career'] > 1)]

    rating_cols = ['playerid', 'date', 'balls_faced_r', 'run_rating', 'run_rating_2', 'rep_run_ratio', 'weight_balls_r', 'balls_faced_w', 'wkt_rating', 'wkt_rating_2', 'rep_wkt_ratio', 'weight_balls_w']

    bat_data = bat_data.merge(ratings[ratings['i_balls_faced'] > 0].loc[:, rating_cols], how='left', on=['playerid', 'date'])
    bat_data = bat_data[(bat_data['run_rating'] >= 0)]
    bat_data = bat_data[(bat_data['wkt_rating'] >= 0)]

    bat_data = bat_data.dropna(subset=['rep_run_ratio', 'weight_balls_r'])
    bat_data = bat_data.dropna(subset=['rep_wkt_ratio', 'weight_balls_w'])

    # -------------------------
    # Career balls model
    # -------------------------
    X = bat_data[['weight_balls_r']]
    X2 = bat_data[['weight_balls_w']]
    y = pd.DataFrame(bat_data['balls_faced_career'])

    model = LinearRegression()
    model.fit(X, y)

    model2 = LinearRegression()
    model2.fit(X2, y)

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
    # Apply reversion
    # -------------------------
    bat_data.insert(bat_data.columns.get_loc("rep_run_ratio") + 1, 'rep_run_weight', rep_weight(bat_data['balls_for_weight_r'], bat_data['run_rating'], bat_data['rep_run_ratio'], param_r_dict)[0])
    bat_data.insert(bat_data.columns.get_loc("rep_run_weight") + 1, 'run_rating_3', rep_weight(bat_data['balls_for_weight_r'], bat_data['run_rating'], bat_data['rep_run_ratio'], param_r_dict)[1])
    bat_data.insert(bat_data.columns.get_loc("rep_wkt_ratio") + 1, 'rep_wkt_weight', rep_weight(bat_data['balls_for_weight_w'], bat_data['wkt_rating'], bat_data['rep_wkt_ratio'], param_w_dict)[0])
    bat_data.insert(bat_data.columns.get_loc("rep_wkt_weight") + 1, 'wkt_rating_3', rep_weight(bat_data['balls_for_weight_w'], bat_data['wkt_rating'], bat_data['rep_wkt_ratio'], param_w_dict)[1])

    ratings.insert(ratings.columns.get_loc("rep_run_ratio") + 1, 'rep_run_weight', rep_weight(ratings['balls_for_weight_r'], ratings['run_rating'], ratings['rep_run_ratio'], param_r_dict)[0])
    ratings.insert(ratings.columns.get_loc("rep_run_weight") + 1, 'run_rating_3', rep_weight(ratings['balls_for_weight_r'], ratings['run_rating'], ratings['rep_run_ratio'], param_r_dict)[1])
    ratings.insert(ratings.columns.get_loc("rep_wkt_ratio") + 1, 'rep_wkt_weight', rep_weight(ratings['balls_for_weight_w'], ratings['wkt_rating'], ratings['rep_wkt_ratio'], param_w_dict)[0])
    ratings.insert(ratings.columns.get_loc("rep_wkt_weight") + 1, 'wkt_rating_3', rep_weight(ratings['balls_for_weight_w'], ratings['wkt_rating'], ratings['rep_wkt_ratio'], param_w_dict)[1])
    ratings = ratings.merge(career_ratings[['playerid', 'batsman', 'career_t20_run_rating', 'career_t20_wkt_rating', 'career_odi_run_rating', 'career_odi_wkt_rating']], on=['playerid', 'batsman'], how='left')

    # -------------------------
    # SQL upload table
    # -------------------------
    sql_upload = ratings[ratings['date'] == ratings['date'].max()]

    batter_names = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/data/batDataCombinedClean_w.csv', parse_dates=['date']).loc[:, ['playerid', 'batsman']].drop_duplicates()
    sql_upload = sql_upload.merge(batter_names, how='left', left_on=['playerid'], right_on=['playerid'])

    sql_upload = sql_upload.loc[:, ['batsman_y', 'playerid', 'host', 'ord_r', 'balls_faced_r', 'run_rating', 'wkt_rating', 'competition', 'rep_run_weight', 'run_rating_3', 'rep_wkt_weight', 'wkt_rating_3']]
    sql_upload.insert(sql_upload.columns.get_loc("wkt_rating") + 1, 'external_rating', 28)
    sql_upload.columns = ['batter', 'playerid', 'host', 'order', 'balls_faced', 'run_rating', 'wkt_rating', 'external_rating', 'competition', 'rep_run_weight', 'run_rating_2', 'rep_wkt_weight', 'wkt_rating_2']

    # -------------------------
    # Exports
    # -------------------------
    if x == 0:
        ratings.to_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/outputs/batRatingsJungle3_w.csv', index=False)
        sql_upload.to_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/outputs/sqlUploadJungle_w.csv', index=False)

    else:
        ratings.to_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/outputs/batRatingsRasoi3_w.csv', index=False)
        sql_upload.to_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/outputs/sqlUploadRasoi_w.csv', index=False)

