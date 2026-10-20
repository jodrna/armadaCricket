# PROCESS: stage 1 - bowler ratings step 5: reversion -> outputs/bowlRatingsJungle3 / Rasoi3 (final) + sqlUpload files
#   (BOWL_RUNS_BASIS=bat -> _batRuns).
# See PROCESS.txt at the top of prod_model for the whole run order.
# v2 (2026-10-12): starts as an exact copy of 5_bowlReversion.py; the original is kept in _v1/. To be reworked, starting with
# recency weighting and experience (ratings looked systematically out by experience in the ball outcome work).
# project root = the first folder above this script holding paths.py, so paths / db / shared modules come from THIS
# project wherever it is (OneDrive, GitHub, a copy) - no PYTHONPATH needed (20 Oct 2026)
import sys as _sys
from pathlib import Path as _Path
_root = str(next(p for p in _Path(__file__).resolve().parents if (p / 'paths.py').exists()))
if _root in _sys.path: _sys.path.remove(_root)
_sys.path.insert(0, _root)
import pandas as pd
import numpy as np
from paths import PROJECT_ROOT
import os
# Bowler runs basis (armadacricket-75, user 19 Oct): 'all' (default) = runs include the bowler's wides / no-balls, as the live
# ratings tables and RA_Sum use; 'bat' (env BOWL_RUNS_BASIS=bat) = batters' runs off legal balls only, for the live sim, which
# adds the bowler's wides / no-balls separately (with 'all' the sim counted them twice: seamers too dear, spinners too cheap).
# 'bat' reads / writes *_batRuns* files only; the normal files are untouched.
RUNS_TAG = '_batRuns' if os.environ.get('BOWL_RUNS_BASIS', 'all') == 'bat' else ''


# -------------------------
# Reversion function
# -------------------------
def rep_weight(bowled, rating, rep_ratio, param_dict):
    k = param_dict['k']
    x = param_dict['x']
    y = param_dict['y']
    # v2: the starting weight on replacement depends on whether the bowler's own record is better or worse than his
    # replacement level (bowlReversionCurveTest_v2.py): a poor start off a few overs says ~nothing, a good one says a little
    good = (rating < rep_ratio) if param_dict['good_is_low'] else (rating > rep_ratio)
    a = np.where(good, param_dict['a_good'], param_dict['a_bad'])

    weight = np.maximum(y, np.maximum((1 - k) ** bowled, a - (x * bowled)))
    rating_2 = (rep_ratio * weight) + ((1 - weight) * rating)

    return weight, rating_2


# -------------------------
# v2 extra reversion (taperReversionTest_v2.py, tuned 2019-22, tested 2023+, 2026-10-12)
# The reversion floor (y) leaves established players almost untouched, but their ratings were too spread: players a little
# above their replacement level fell short and those a little below beat theirs. A flat higher floor fixed that but
# over-pulled genuinely exceptional players, so the extra pull fades as the gap to replacement grows:
#     extra = max(0, pull - w) / (1 - w) * gap * exp(-(gap / size) ** 2),  gap = rating_3 - replacement
# with separate pull strengths for ratings above (pulled down) and below (pulled up) their replacement level.
# -------------------------
def taper_reversion(df, rating_col, rep_col, weight_col, pull_down, pull_up, size):
    cur, rep, w = df[rating_col].to_numpy(), df[rep_col].to_numpy(), df[weight_col].to_numpy()
    gap = cur - rep
    pull = np.where(gap > 0, pull_down, pull_up)
    extra = np.clip(pull - w, 0, None) / np.clip(1 - w, 1e-9, None)
    df[rating_col] = cur - extra * gap * np.exp(-(gap / size) ** 2)
    return df


# -------------------------
# Build outputs for jungle and rasoi
# -------------------------
for x in np.arange(0, 2, 1):
    if x == 0:
        model_name = 'jungle'
        ratings = pd.read_csv(PROJECT_ROOT / f'men/playerRatings/bowlT20Mens/outputs/bowlRatingsJungle2{RUNS_TAG}.csv', parse_dates=['date'])

    else:
        model_name = 'rasoi'
        ratings = pd.read_csv(PROJECT_ROOT / f'men/playerRatings/bowlT20Mens/outputs/bowlRatingsRasoi2{RUNS_TAG}.csv', parse_dates=['date'])

    # v2 (bowlReversionCurveTest_v2.py, tuned 2019-22 / tested 2023+): v1 was runs a 0.75375, x 0.00015, y 0.1 and
    # wkts a 0.88573, x 0.000075, y 0.03. Runs: tiny samples now lean (almost) fully on replacement, reaching their own
    # record by ~1600 balls, and no flat floor (the tapering pull below does the extra reversion)
    param_r_dict = {
        'k': 0.1057993,
        'a_good': 0.95,
        'a_bad': 1.0,
        'x': 0.0006,
        'y': 0.0,
        'good_is_low': True    # fewer runs than replacement = good
    }

    param_w_dict = {
        'k': 0.0045298,
        'a_good': 0.95,
        'a_bad': 1.0,
        'x': 0.000075,
        'y': 0.1,
        'good_is_low': False   # more wickets than replacement = good
    }

    # -------------------------
    # Import bowl data
    # -------------------------
    bowl_data = pd.read_csv(PROJECT_ROOT / f'men/playerRatings/bowlT20Mens/data/bowlDataCombinedClean{RUNS_TAG}.csv', parse_dates=['date', 'dob'])

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
    # v2: part-timers (under 6 balls per match) concede ~3% less than the replacement model says (captains pick their spots)
    # -------------------------
    for frame in (bowl_data, ratings):
        frame['rep_run_ratio'] = frame['rep_run_ratio'] * np.where(frame['ballspermatch'] < 6, 0.97, 1.0)

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


    # -------------------------
    # v2: tapering extra reversion, and x1.03 runs conceded in a bowler's first T20 spell (rust after long gaps was
    # tested and dropped: it showed in 2023+ but not 2019-22)
    # -------------------------
    for frame in (bowl_data, ratings):
        taper_reversion(frame, 'run_rating_3', 'rep_run_ratio', 'rep_run_weight', pull_down=0.5, pull_up=0.3, size=0.2)
        taper_reversion(frame, 'wkt_rating_3', 'rep_wkt_ratio', 'rep_wkt_weight', pull_down=0.5, pull_up=0.7, size=0.3)
    real = ratings['matchid'] != 101
    first_spell = real & (ratings[real].sort_values(['date', 'matchid']).groupby('playerid').cumcount() == 0).reindex(ratings.index, fill_value=False)
    ratings.loc[first_spell, 'run_rating_3'] = ratings.loc[first_spell, 'run_rating_3'] * 1.03

    ratings = ratings.merge(career_ratings[['playerid', 'bowler', 'career_t20_run_rating', 'career_t20_wkt_rating', 'career_odi_run_rating', 'career_odi_wkt_rating']], on=['playerid', 'bowler'], how='left')

    # -------------------------
    # SQL upload table
    # -------------------------
    sql_upload = ratings.loc[ratings['date'] == ratings['date'].max()].copy()
    sql_upload.loc[:, 'last_match_date'] = ratings.loc[ratings['matchid'] != 101, 'date'].max()
    # read raw combined names to make sure every playerid keeps a bowler name even if naming differs after cleaning
    bowler_names = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/bowlT20Mens/data/bowlDataCombined.csv', parse_dates=['date']).loc[:, ['bowlerid', 'bowler']].drop_duplicates()
    sql_upload = sql_upload.merge(bowler_names, how='left', left_on=['playerid'], right_on=['bowlerid'])

    sql_upload = sql_upload.loc[:, ['last_match_date', 'bowler_y', 'playerid', 'host', 'ord_w', 'balls_bowled_w', 'run_rating', 'wkt_rating', 'competition', 'rep_run_weight', 'run_rating_3', 'rep_wkt_weight', 'wkt_rating_3']]
    sql_upload.insert(sql_upload.columns.get_loc('wkt_rating') + 1, 'external_rating', 28)
    sql_upload.columns = ['last_match_date', 'bowler', 'playerid', 'host', 'order', 'balls_bowled', 'run_rating', 'wkt_rating', 'external_rating', 'competition', 'rep_run_weight', 'run_rating_2', 'rep_wkt_weight', 'wkt_rating_2']

    # -------------------------
    # Exports
    # -------------------------
    if model_name == 'jungle':
        ratings.to_csv(PROJECT_ROOT / f'men/playerRatings/bowlT20Mens/outputs/bowlRatingsJungle3{RUNS_TAG}.csv', index=False)
        sql_upload.to_csv(PROJECT_ROOT / f'men/playerRatings/bowlT20Mens/outputs/sqlUploadJungle{RUNS_TAG}.csv', index=False)

    else:
        ratings.to_csv(PROJECT_ROOT / f'men/playerRatings/bowlT20Mens/outputs/bowlRatingsRasoi3{RUNS_TAG}.csv', index=False)
        sql_upload.to_csv(PROJECT_ROOT / f'men/playerRatings/bowlT20Mens/outputs/sqlUploadRasoi{RUNS_TAG}.csv', index=False)

comps = pd.pivot_table(ratings, values='bowler', index='competition', aggfunc='count').reset_index()
