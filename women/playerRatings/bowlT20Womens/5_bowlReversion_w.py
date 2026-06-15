import pandas as pd
import numpy as np
from paths import PROJECT_ROOT


# -------------------------
# Reversion function
# -------------------------
def rep_weight(bowled, rating, rep_ratio, param_dict):
    k = param_dict['k']
    a = param_dict['a']
    x = param_dict['x']
    y = param_dict['y']

    weight = np.maximum(y, np.maximum((1 - k) ** bowled, a - (x * bowled)))
    rating_2 = (rep_ratio * weight) + ((1 - weight) * rating)

    return weight, rating_2


# -------------------------
# Build outputs for jungle and rasoi
# -------------------------
for x in np.arange(0, 2, 1):
    if x == 0:
        model_name = 'jungle'
        ratings = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/outputs/bowlRatingsJungle2_w.csv', parse_dates=['date'])

    else:
        model_name = 'rasoi'
        ratings = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/outputs/bowlRatingsRasoi2_w.csv', parse_dates=['date'])

    param_r_dict = {
        'k': 0.1057993,
        'a': 0.75375,
        'x': 0.00015,
        'y': 0.1
    }

    param_w_dict = {
        'k': 0.0045298,
        'a': 0.88573,
        'x': 0.000075,
        'y': 0.03
    }

    # -------------------------
    # Import bowl data
    # -------------------------
    bowl_data = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/data/bowlDataCombinedClean_w.csv', parse_dates=['date', 'dob'])

    # -------------------------
    # Career ratings
    # -------------------------
    career_ratings = pd.pivot_table(bowl_data, values=['runs', 'realexprbowl', 'wkt', 'realexpwbowl'], index=['playerid', 'bowler', 'format'], aggfunc='mean').reset_index()
    career_ratings['run_rating'] = career_ratings['runs'] / career_ratings['realexprbowl']
    career_ratings['wkt_rating'] = career_ratings['wkt'] / career_ratings['realexpwbowl']

    career_ratings = pd.pivot_table(career_ratings, values=['run_rating', 'wkt_rating'], index=['playerid', 'bowler'], columns='format', aggfunc='mean').reset_index()

    career_ratings.columns = [
        f'career_{col[1]}_{col[0]}' if col[1] else col[0]
        for col in career_ratings.columns]

    # now only t20 going forward
    bowl_data = bowl_data[bowl_data['format'] == 't20']

    # -------------------------
    # Merge ratings into bowl data
    # -------------------------
    bowl_data = bowl_data[bowl_data['balls_bowled'] > 0]

    rating_cols = ['playerid', 'date', 'balls_bowled_r', 'run_rating', 'rep_run_ratio', 'balls_bowled_w', 'wkt_rating', 'rep_wkt_ratio']

    bowl_data = bowl_data.merge(ratings[ratings['i_balls_bowled'] > 0].loc[:, rating_cols], how='left', on=['playerid', 'date'])
    bowl_data = bowl_data[bowl_data['run_rating'] >= 0]
    bowl_data = bowl_data[bowl_data['wkt_rating'] >= 0]

    # -------------------------
    # Apply reversion
    # -------------------------
    bowl_data.insert(bowl_data.columns.get_loc('rep_run_ratio') + 1, 'rep_run_weight', rep_weight(bowl_data['balls_bowled_r'], bowl_data['run_rating'], bowl_data['rep_run_ratio'], param_r_dict)[0])
    bowl_data.insert(bowl_data.columns.get_loc('rep_run_weight') + 1, 'run_rating_3', rep_weight(bowl_data['balls_bowled_r'], bowl_data['run_rating'], bowl_data['rep_run_ratio'], param_r_dict)[1])
    bowl_data.insert(bowl_data.columns.get_loc('rep_wkt_ratio') + 1, 'rep_wkt_weight', rep_weight(bowl_data['balls_bowled_w'], bowl_data['wkt_rating'], bowl_data['rep_wkt_ratio'], param_w_dict)[0])
    bowl_data.insert(bowl_data.columns.get_loc('rep_wkt_weight') + 1, 'wkt_rating_3', rep_weight(bowl_data['balls_bowled_w'], bowl_data['wkt_rating'], bowl_data['rep_wkt_ratio'], param_w_dict)[1])

    ratings.insert(ratings.columns.get_loc('rep_run_ratio') + 1, 'rep_run_weight', rep_weight(ratings['balls_bowled_r'], ratings['run_rating'], ratings['rep_run_ratio'], param_r_dict)[0])
    ratings.insert(ratings.columns.get_loc('rep_run_weight') + 1, 'run_rating_3', rep_weight(ratings['balls_bowled_r'], ratings['run_rating'], ratings['rep_run_ratio'], param_r_dict)[1])
    ratings.insert(ratings.columns.get_loc('rep_wkt_ratio') + 1, 'rep_wkt_weight', rep_weight(ratings['balls_bowled_w'], ratings['wkt_rating'], ratings['rep_wkt_ratio'], param_w_dict)[0])
    ratings.insert(ratings.columns.get_loc('rep_wkt_weight') + 1, 'wkt_rating_3', rep_weight(ratings['balls_bowled_w'], ratings['wkt_rating'], ratings['rep_wkt_ratio'], param_w_dict)[1])

    ratings = ratings.merge(career_ratings[['playerid', 'bowler', 'career_t20_run_rating', 'career_t20_wkt_rating', 'career_odi_run_rating', 'career_odi_wkt_rating']], on=['playerid', 'bowler'], how='left')

    # -------------------------
    # SQL upload table
    # -------------------------
    sql_upload = ratings.loc[ratings['date'] == ratings['date'].max()].copy()
    sql_upload.loc[:, 'last_match_date'] = ratings.loc[ratings['matchid'] != 101, 'date'].max()
    # read raw combined names to make sure every playerid keeps a bowler name even if naming differs after cleaning
    bowler_names = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/data/bowlDataCombined_w.csv', parse_dates=['date']).loc[:, ['bowlerid', 'bowler']].drop_duplicates()
    sql_upload = sql_upload.merge(bowler_names, how='left', left_on=['playerid'], right_on=['bowlerid'])

    sql_upload = sql_upload.loc[:, ['last_match_date', 'bowler_y', 'playerid', 'host', 'ord_w', 'balls_bowled_w', 'run_rating', 'wkt_rating', 'competition', 'rep_run_weight', 'run_rating_3', 'rep_wkt_weight', 'wkt_rating_3']]
    sql_upload.insert(sql_upload.columns.get_loc('wkt_rating') + 1, 'external_rating', 28)
    sql_upload.columns = ['last_match_date', 'bowler', 'playerid', 'host', 'order', 'balls_bowled', 'run_rating', 'wkt_rating', 'external_rating', 'competition', 'rep_run_weight', 'run_rating_2', 'rep_wkt_weight', 'wkt_rating_2']

    # -------------------------
    # Exports
    # -------------------------
    if model_name == 'jungle':
        ratings.to_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/outputs/bowlRatingsJungle3_w.csv', index=False)
        sql_upload.to_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/outputs/sqlUploadJungle_w.csv', index=False)

    else:
        ratings.to_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/outputs/bowlRatingsRasoi3_w.csv', index=False)
        sql_upload.to_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/outputs/sqlUploadRasoi_w.csv', index=False)



