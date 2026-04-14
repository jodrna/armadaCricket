import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from paths import PROJECT_ROOT


for x in np.arange(0, 2, 1):
    # read data
    bowl_data = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/bowlT20Mens/data/combinedBowlDataClean.csv', parse_dates=['date', 'dob'])
    bowl_data = bowl_data[bowl_data['format'] == 't20']

    # read ratingsT20
    if x == 0:
        ratings = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/bowlT20Mens/outputs/bowlRatingsJungle2.csv', parse_dates=['date'])
    else:
        ratings = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/bowlT20Mens/outputs/bowlRatingsRasoi2.csv', parse_dates=['date'])

    # filter out bowl data dummies
    bowl_data = bowl_data[(bowl_data['balls_bowled'] > 0)]

    # merge ratingsT20 into bowl data
    bowl_data = bowl_data.merge(ratings[ratings['i_balls_bowled'] > 0].loc[:, ['playerid', 'date', 'balls_bowled_r', 'run_rating', 'rep_run_ratio', 'balls_bowled_w', 'wkt_rating', 'rep_wkt_ratio']],
                                how='left', on=['playerid', 'date'])
    bowl_data = bowl_data[(bowl_data['run_rating'] >= 0)]
    bowl_data = bowl_data[(bowl_data['wkt_rating'] >= 0)]


    # this function takes the optimised parameters (which are produced in another script), the replacement value, the pre reversion rating, balls faced, and outputs the final reverted rating
    if x == 0:
        def rep_weight_r(bowled, rating, rep_ratio):
            k, a, x, y = .1057993, .75375, .00015, 0.1
            weight = np.maximum(y, np.maximum((1 - k) ** bowled, a - (x * bowled)))
            rating_2 = (rep_ratio * weight) + ((1 - weight) * rating)
            return weight, rating_2
    else:
        def rep_weight_r(bowled, rating, rep_ratio):
            k, a, x, y = .1057993, .75375, .00015, 0.1
            weight = np.maximum(y, np.maximum((1 - k) ** bowled, a - (x * bowled)))
            rating_2 = (rep_ratio * weight) + ((1 - weight) * rating)
            return weight, rating_2

    if x == 0:
        def rep_weight_w(bowled, rating, rep_ratio):
            k, a, x, y = .0045298, .88573, .000075, 0.03
            weight = np.maximum(y, np.maximum((1 - k) ** bowled, a - (x * bowled)))
            rating_2 = (rep_ratio * weight) + ((1 - weight) * rating)
            return weight, rating_2

    else:
        def rep_weight_w(bowled, rating, rep_ratio):
            k, a, x, y = .0045298, .88573, .000075, 0.03
            weight = np.maximum(y, np.maximum((1 - k) ** bowled, a - (x * bowled)))
            rating_2 = (rep_ratio * weight) + ((1 - weight) * rating)
            return weight, rating_2

    # now insert rating 3, the rating reverted to replacement value
    bowl_data.insert(bowl_data.columns.get_loc("rep_run_ratio") + 1, 'rep_run_weight', rep_weight_r(bowl_data['balls_bowled_r'], bowl_data['run_rating'], bowl_data['rep_run_ratio'])[0])
    bowl_data.insert(bowl_data.columns.get_loc("rep_run_weight") + 1, 'run_rating_3', rep_weight_r(bowl_data['balls_bowled_r'], bowl_data['run_rating'], bowl_data['rep_run_ratio'])[1])
    bowl_data.insert(bowl_data.columns.get_loc("rep_wkt_ratio") + 1, 'rep_wkt_weight', rep_weight_w(bowl_data['balls_bowled_w'], bowl_data['wkt_rating'], bowl_data['rep_wkt_ratio'])[0])
    bowl_data.insert(bowl_data.columns.get_loc("rep_wkt_weight") + 1, 'wkt_rating_3', rep_weight_w(bowl_data['balls_bowled_w'], bowl_data['wkt_rating'], bowl_data['rep_wkt_ratio'])[1])

    ratings.insert(ratings.columns.get_loc("rep_run_ratio") + 1, 'rep_run_weight', rep_weight_r(ratings['balls_bowled_r'], ratings['run_rating'], ratings['rep_run_ratio'])[0])
    ratings.insert(ratings.columns.get_loc("rep_run_weight") + 1, 'run_rating_3', rep_weight_r(ratings['balls_bowled_r'], ratings['run_rating'], ratings['rep_run_ratio'])[1])
    ratings.insert(ratings.columns.get_loc("rep_wkt_ratio") + 1, 'rep_wkt_weight', rep_weight_w(ratings['balls_bowled_w'], ratings['wkt_rating'], ratings['rep_wkt_ratio'])[0])
    ratings.insert(ratings.columns.get_loc("rep_wkt_weight") + 1, 'wkt_rating_3', rep_weight_w(ratings['balls_bowled_w'], ratings['wkt_rating'], ratings['rep_wkt_ratio'])[1])

    # create a table for simple sql upload
    sql_upload = ratings.loc[ratings['date'] == ratings['date'].max()].copy()
    sql_upload.loc[:, 'last_match_date'] = ratings.loc[ratings['matchid'] != 101, 'date'].max()
    bowler_names = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/bowlT20Mens/data/combinedBowlData.csv', parse_dates=['date']).loc[:, ['bowlerid', 'bowler']].drop_duplicates()  # this is done to make sure all names have ratingsT20, even if an id is matched to two names like Shaheen
    sql_upload = sql_upload.merge(bowler_names, how='left', left_on=['playerid'], right_on=['bowlerid'])
    sql_upload = sql_upload.loc[:, ['last_match_date', 'bowler_y', 'playerid', 'host', 'ord_w', 'balls_bowled_w', 'run_rating', 'wkt_rating', 'competition', 'rep_run_weight', 'run_rating_3', 'rep_wkt_weight', 'wkt_rating_3']]
    sql_upload.insert(sql_upload.columns.get_loc("wkt_rating") + 1, 'external_rating', 28)
    sql_upload.columns = ['last_match_date', 'bowler', 'playerid', 'host', 'order', 'balls_bowled', 'run_rating', 'wkt_rating', 'external_rating', 'competition', 'rep_run_weight', 'run_rating_2', 'rep_wkt_weight', 'wkt_rating_2']


    # export the detailed ratingsT20 and table for sql upload
    if x == 0:
        ratings.to_csv(PROJECT_ROOT / 'men/playerRatings/bowlT20Mens/outputs/bowlRatingsJungle3.csv', index=False)
        sql_upload.to_csv(PROJECT_ROOT / 'men/playerRatings/bowlT20Mens/outputs/sqlUploadJungle.csv', index=False)
    else:
        ratings.to_csv(PROJECT_ROOT / 'men/playerRatings/bowlT20Mens/outputs/bowlRatingsRasoi3.csv', index=False)
        sql_upload.to_csv(PROJECT_ROOT / 'men/playerRatings/bowlT20Mens/outputs/sqlUploadRasoi.csv', index=False)


