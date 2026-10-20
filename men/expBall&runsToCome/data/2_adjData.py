# PROCESS: stage 3 - RA_Sum: ball data + pre-match ratings + ground + FXB/SLW + form + splits -> data/dataClean.csv
#   (surge-era BBL -> dataClean_bblSurge.csv). Imports simToDataClean + biasGrid at the end.
# See PROCESS.txt at the top of prod_model for the whole run order.
# project root = the first folder above this script holding paths.py, so paths / db / shared modules come from THIS
# project wherever it is (OneDrive, GitHub, a copy) - no PYTHONPATH needed (20 Oct 2026)
import sys as _sys
from pathlib import Path as _Path
_root = str(next(p for p in _Path(__file__).resolve().parents if (p / 'paths.py').exists()))
if _root in _sys.path: _sys.path.remove(_root)
_sys.path.insert(0, _root)
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib as mpl
mpl.use('TkAgg')
from pathlib import Path
user_name = Path.home()
from db import engine
from paths import PROJECT_ROOT
from datetime import date

######## need to run lineups_filler before running this!!!!!!!!!!!

# RA_Sum is built twice in one run: RA_Sum uses the day's reverted ground values (ground_table reverted_runs/wkts - the
# match's own scoring, reverted), RA_Sum_nonDay uses the ground's normal level (runsratio_ground/wktsratio_ground).
# Everything else is shared. The _nonDay columns replace the old ground toggle and its *_non_today_groundRASum file

connection = engine.connect()
raw_data_og = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/data/Cleaned_t20bbb3_new.csv', parse_dates=['date'])
# raw_data_og_max_date = raw_data_og['date'].max()
raw_data_og = raw_data_og.drop_duplicates(subset=['ID'])
# ---- surge-era BBL (armadacricket-d0, 2026-10-15) ----
# 1_getData keeps BBL after 2019 again (with isPowerSurge). Surge-era BBL (date > 2020-06-06) gets its own surge
# handling below (as WBBL in 2_adjData_w), is left OUT of the fits of the 'average situation' curves (er_tc_avg_now /
# er_tc_smooth_basic, expr_pred, er_tc_ground / model12 / model22 and their dnb tables) so surge overs don't distort
# them - those curves are still applied to every ball - and is written to Data/dataClean_bblSurge.csv instead of
# dataClean.csv at the end. Everything downstream reads dataClean.csv, so surge-era BBL never reaches the v2 chain,
# ball values, FXB_all, outcomeShapes or the chase models
if 'isPowerSurge' not in raw_data_og.columns:   # Cleaned_t20bbb3_new.csv from before the 1_getData change
    raw_data_og['isPowerSurge'] = 0

# # raw_data_og = raw_data_og[raw_data_og.matchID == 1233979]
bowler_data = pd.read_sql_query("""select bowler, playerid as bowlerid, matchid as "matchID", competition, host, date, run_rating as run_rating_pr_bo, wkt_rating as wkt_rating_pr_bo from player_ratings.bowler_ratings_historic""", con=connection)
batter_data = pd.read_sql_query("""select batter as "batsmanName", playerid as batterid, matchid as "matchID", competition, host, date, run_rating as run_rating_pr_ba, wkt_rating as wkt_rating_pr_ba, balls_faced as balls_faced_pr_ba, ord as ord_pr_ba from player_ratings.batter_ratings_historic""", con=connection)
ground_data = pd.read_sql_query("""select matchid as "matchID", venue, innperiod as "inningPhase", reverted_runs, reverted_wkts, reverted_runs_2nd, reverted_wkts_2nd, runsratio_ground, wktsratio_ground from player_ratings.ground_table""", con=connection)
bblvalues = pd.read_sql_query("""select over as "overNumber", runs_ratio as bbl_runs_ratio, wkts_ratio as bbl_wkts_ratio from player_ratings.bbl_csv""", con=connection)
wktvalues = pd.read_sql_query("""select overno as "overNumber", wktslost as "totalInningWickets", wktvalue from player_ratings.wkt_values""", con=connection)
#
# bowler_data.to_csv(fr'{user_name}\Documents\Tempdata\bowldataformatch.csv', index=False)
# batter_data.to_csv(fr'{user_name}\Documents\Tempdata\batdataformatch.csv', index=False)
# ground_data.to_csv(fr'{user_name}\Documents\Tempdata\grounddataformatch.csv', index=False)
# wktvalues.to_csv(fr'{user_name}\Documents\Tempdata\wktvalueformatch.csv', index=False)
# #
# bowler_data = pd.read_csv(fr'{user_name}\Documents\Tempdata\bowldataformatch.csv', parse_dates=['date'])
# bowler_data['date'] = bowler_data['date'].dt.date
# batter_data = pd.read_csv(fr'{user_name}\Documents\Tempdata\batdataformatch.csv', parse_dates=['date'])
# batter_data['date'] = batter_data['date'].dt.date
# ground_data = pd.read_csv(fr'{user_name}\Documents\Tempdata\grounddataformatch.csv')
# wktvalues = pd.read_csv(fr'{user_name}\Documents\Tempdata\wktvalueformatch.csv')



raw_data = raw_data_og.copy()
raw_data['date'] = raw_data['date'].dt.date
# raw_data = raw_data[raw_data.inningNumber == 2]
# preserve the TRUE batting position (1-11) before the next line floors it at 2 for the
# expected-runs-to-come-by-ord smoothing below - the order/balls-faced adjustments further down
# need the real value, including ord=1 (openers), which that floored column no longer has
raw_data['ord_real'] = raw_data['ord']
raw_data['ord'] = np.maximum(2, raw_data['ord'])
raw_data['count1'] = 1
# 'ID' is NOT a reliable chronological order within a match/inningNumber - roughly 9% of inningNumber have
# at least one ball whose ID puts it out of true delivery order (median displacement ~7 deliveries,
# up to 19), and ~9% of matches even have inningNumber 1's ID range overlapping/exceeding inningNumber 2's.
# Every rolling "rest of the inningNumber" sum below (er_tc/ew_tc/rar_bowl_sum/raw_bowl_sum etc.) depends
# on row order matching true chronological order, so an out-of-place ID can make a ball look like it
# has the whole rest of the inningNumber still ahead of it (or none of it). 'inningBallNumber' is a clean,
# strictly-incrementing per-ball counter (unlike 'ball'/'overNumber', which legitimately repeats their
# label across a wide/no-ball) - only 2 rows in the whole dataset share a (matchID, inningNumber,
# inningBallNumber), vs thousands for ID - so it's used as the ordering key instead, with inningNumber sorted
# explicitly first since inningBallNumber resets to 0 each inningNumber.
# UPDATE 2026-10-14: in the current data inningBallNumber is NO longer unique - 42,576 rows share it, a wide / no-ball
# taking the same number as the legal ball that completes it (as the women's data does). So the same tiebreak as
# 2_adjData_w: extras before the legal ball, then ID. Since 2025 IDs are often out of playing order (71% of 2026
# innings), so ID is only ever the last tiebreak
raw_data['is_legal'] = np.where(raw_data['isValid'] == 1, 1, 0)
raw_data.sort_values(by=['matchID', 'inningNumber', 'inningBallNumber', 'is_legal', 'ID'], ascending=[True, True, True, True, True], inplace=True)
raw_data = raw_data.drop(columns=['is_legal'])
raw_data = raw_data.reset_index(drop=True)
raw_data = raw_data.loc[:, ['matchID', 'venue', 'inningNumber', 'date', 'host', 'competition', 'ID', 'inningBallNumber', 'nonstrikerName', 'battingTeam', 'overNumber', 'inningBallsRemaining', 'totalInningWickets', 'ord', 'ord_real', 'batsmanBallsFaced', 'batsmanName', 'bowler', 'batterid', 'bowlerid', 'inningPhase', 'realexprbat', 'realexpwbat', 'realexprbowl', 'realexpwbowl', 'ovrexpr', 'ovrexpw', 'isPowerSurge', 'isValid', 'batsmanRuns', 'totalRuns', 'byeRuns', 'noballRuns', 'wideRuns']]
# his_max_date0 = raw_data['date'].max()
##get recovery values as back up for the historic player ratings, in case historic values aren't available for specifc games
# Fallbacks WITHOUT future data (2026-10-14). ratings_historic rows are PRE-match (a match's row is the rating going into
# it), so the (matchID, id) joins below are right. The fallbacks used to be each player's / ground's mean over ALL their
# rows - including matches played after this one. Now: the latest row dated strictly before this match (as live_sim's
# pre_match()); no earlier row -> 1.0 for ratings and ground (order: left missing, as when there was no row at all).
def latest_before(keys, table, by, cols):
    # per (matchID, *by) in keys: the `by` group's latest table row with date < the match date (a match's earliest date
    # if the ball data gives it more than one - 3 men's matches)
    k = keys.loc[:, ['matchID'] + by].drop_duplicates().merge(keys.groupby('matchID', as_index=False)['date'].min(), on='matchID', how='left')
    t = table.loc[:, ['date'] + by + cols].copy()
    for frame in (k, t):
        frame['date'] = pd.to_datetime(frame['date']).astype('datetime64[ns]')
        for col in by:
            frame[f'_{col}'] = frame[col].astype('float64') if col != 'venue' else frame[col].astype(str)
    k, t = k.dropna(subset=['date']), t.dropna(subset=['date'] + cols, how='any').drop(columns=by)
    byKey = [f'_{col}' for col in by]
    out = pd.merge_asof(k.sort_values('date'), t.sort_values('date'), on='date', by=byKey, allow_exact_matches=False)
    return out.drop(columns=['date'] + byKey)


matchDates = raw_data.groupby('matchID', as_index=False)['date'].min()
recoveries_bat = latest_before(raw_data, batter_data, ['batterid'], ['run_rating_pr_ba', 'wkt_rating_pr_ba'])
recoveries_bat.rename(columns={'run_rating_pr_ba': 'oppo_bat_runs_old', 'wkt_rating_pr_ba': 'oppo_bat_wkts_old'}, inplace=True)
recoveries_ord = latest_before(raw_data, batter_data, ['batterid'], ['ord_pr_ba'])
recoveries_ord.rename(columns={'ord_pr_ba': 'ord_pr_ba_old'}, inplace=True)
recoveries_bowl = latest_before(raw_data, bowler_data, ['bowlerid'], ['run_rating_pr_bo', 'wkt_rating_pr_bo'])
recoveries_bowl.rename(columns={'run_rating_pr_bo': 'oppo_bowl_runs_old', 'wkt_rating_pr_bo': 'oppo_bowl_wkts_old'}, inplace=True)
# ground normal by phase: the venue's latest earlier ground_table row (ground_table has no date - matches get theirs from
# the ball data, so rows for matches not in it can't serve as an earlier row)
groundDated = ground_data.merge(matchDates, on='matchID', how='inner')
recoveries_ground = latest_before(raw_data, groundDated, ['venue', 'inningPhase'], ['runsratio_ground', 'wktsratio_ground'])
recoveries_ground.rename(columns={'runsratio_ground': 'ground_runs', 'wktsratio_ground': 'ground_wkts'}, inplace=True)

raw_data = raw_data.merge(recoveries_bat, on=['matchID', 'batterid'], how='left')
raw_data = raw_data.merge(recoveries_ord, on=['matchID', 'batterid'], how='left')
raw_data = raw_data.merge(recoveries_bowl, on=['matchID', 'bowlerid'], how='left')
raw_data = raw_data.merge(recoveries_ground, on=['matchID', 'venue', 'inningPhase'], how='left')
raw_data[['oppo_bat_runs_old', 'oppo_bat_wkts_old', 'oppo_bowl_runs_old', 'oppo_bowl_wkts_old', 'ground_runs', 'ground_wkts']] = \
    raw_data[['oppo_bat_runs_old', 'oppo_bat_wkts_old', 'oppo_bowl_runs_old', 'oppo_bowl_wkts_old', 'ground_runs', 'ground_wkts']].fillna(1.0)

# his_max_date1 = raw_data['date'].max()
##merge in the bowler batter and ground ratings
# joined on (matchID, bowlerid), as the batters are below - the old (date, bowlerid, competition, host) join hit only
# 91% of men's balls (71% in 2025) and 60% of women's, vs 99% / 88% on matchID (2026-10-13)
bowler_data = bowler_data.drop_duplicates(subset=['matchID', 'bowlerid'])
raw_data = raw_data.merge(bowler_data.loc[:, ['matchID', 'bowlerid', 'run_rating_pr_bo', 'wkt_rating_pr_bo']], on=('matchID', 'bowlerid'), how='left')
raw_data['oppo_bowl_runs'] = raw_data['run_rating_pr_bo'].fillna(raw_data['oppo_bowl_runs_old'])
raw_data['oppo_bowl_wkts'] = raw_data['wkt_rating_pr_bo'].fillna(raw_data['oppo_bowl_wkts_old'])
# his_max_date2 = raw_data['date'].max()
#
# joined on (matchID, batterid) - NOT (date, batterid, competition, host). Verified on the women's
# data (same table structure/pipeline): the old date/competition/host join only matched 62.1% of
# rows, because (a) smaller competitions get bucketed under generic labels in this table (e.g.
# 'tier_2', country names) that don't match the raw data's real competition names, and (b) it
# requires an exact same-day rating snapshot, which doesn't exist for most matches (ratings aren't
# refreshed daily). matchID+batterid is the precise, unambiguous key both tables actually share -
# joining on it directly recovered 88.8% coverage on the women's side instead.
batter_data = batter_data.drop_duplicates(subset=['batterid', 'matchID'])
raw_data = raw_data.merge(batter_data.loc[:, ['batterid', 'matchID', 'run_rating_pr_ba', 'wkt_rating_pr_ba', 'balls_faced_pr_ba', 'ord_pr_ba']], on=('matchID', 'batterid'), how='left')
raw_data['run_rating_pr_ba'] = raw_data['run_rating_pr_ba'].fillna(raw_data['oppo_bat_runs_old'])
raw_data['wkt_rating_pr_ba'] = raw_data['wkt_rating_pr_ba'].fillna(raw_data['oppo_bat_wkts_old'])
# every batter uses their own pre-match rating (the old rule sent batters under 150 balls to their career mean, which
# included future matches; the ratings already shrink small samples - 2026-10-14)
raw_data['oppo_bat_runs'] = raw_data['run_rating_pr_ba']
raw_data['oppo_bat_wkts'] = raw_data['wkt_rating_pr_ba']
# expected_ord fallback: if this batter wasn't individually rated in THIS match (the (matchID,
# batterid) join missed), fall back to their own average rated order across every other match -
# the same rescue oppo_bat_runs/oppo_bat_wkts already get above, just newly added for order since
# nothing analogous existed for it before.
raw_data['ord_pr_ba'] = raw_data['ord_pr_ba'].fillna(raw_data['ord_pr_ba_old'])

# ---- order adjustment (men/expBall&runsToCome/auxiliaries/byOrderAdjusts.py) ----
# oppo_bat_runs/oppo_bat_wkts above are the batter's rating-implied performance multiplier, from
# their historical run/wicket rating alone - they don't know the position the batter actually
# bats at today versus the position their rating was computed at (ord_pr_ba). byOrderAdjusts.csv
# (runsAdjustRatio/wktsAdjustRatio by (expected_ord, ord)) corrects for that. expected_ord is
# rounded to match the lookup's whole-number grid; ord/expected_ord 1-4 already collapse to
# identical rows within that grid (see byOrderAdjusts.py's ord_group()), so no extra bucketing is
# needed on this side. A batter batting exactly where expected (ord == expected_ord) always gets
# exactly 1.0 - no adjustment. Missing (expected_ord, ord) pairs default to 1.0 rather than being
# silently dropped.
#
# This is safe to apply here (per real ball, before any per-batter deduplication) because ord and
# expected_ord don't change across a batter's own inningNumber - every one of their balls gets the same
# adjustment regardless of which ball later survives the dedup down to one row per (matchID,
# batsmanName). The BALLS-FACED adjustment does NOT have this property (it genuinely varies ball to
# ball) and is applied separately, inside `his`, after the batting order/lineup context is
# available - see below (search "balls-faced adjustment").
orderAdjusts = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/outputs/byOrderAdjusts.csv')
raw_data['expected_ord_rounded'] = raw_data['ord_pr_ba'].round()
raw_data = raw_data.merge(
    orderAdjusts.rename(columns={'expected_ord': 'expected_ord_rounded', 'ord': 'ord_real'}),
    on=['expected_ord_rounded', 'ord_real'], how='left'
)
raw_data['runsAdjustRatio'] = raw_data['runsAdjustRatio'].fillna(1.0)
raw_data['wktsAdjustRatio'] = raw_data['wktsAdjustRatio'].fillna(1.0)

# preadj = the clean, pre-order-adjustment values. oppo_bat_runs/oppo_bat_wkts get order-adjusted
# in place from here on (used internally to compute RA_sum via `his` below), but are reset back to
# these _preadj values right before export at the end of the script - see the comment there for why.
raw_data['oppo_bat_runs_preadj'] = raw_data['oppo_bat_runs']
raw_data['oppo_bat_wkts_preadj'] = raw_data['oppo_bat_wkts']
raw_data['oppo_bat_runs'] = raw_data['oppo_bat_runs'] * raw_data['runsAdjustRatio']
raw_data['oppo_bat_wkts'] = raw_data['oppo_bat_wkts'] * raw_data['wktsAdjustRatio']


###need to merge the ground stuff into separate columns for each inningPhase, this makes it easier to predict the amount of runs added by inningPhase later on:
ground_data = ground_data.drop_duplicates(subset=['matchID', 'inningPhase'])
ground_data1 = ground_data.copy()
ground_data1 = ground_data1[ground_data1.inningPhase == 1]
ground_data1.rename(columns={'reverted_runs': 'ground_runs_1', 'reverted_wkts': 'ground_wkts_1'}, inplace=True)
ground_data2 = ground_data.copy()
ground_data2 = ground_data2[ground_data2.inningPhase == 2]
ground_data2.rename(columns={'reverted_runs': 'ground_runs_2', 'reverted_wkts': 'ground_wkts_2'}, inplace=True)
ground_data3 = ground_data.copy()
ground_data3 = ground_data3[ground_data3.inningPhase == 3]
ground_data3.rename(columns={'reverted_runs': 'ground_runs_3', 'reverted_wkts': 'ground_wkts_3'}, inplace=True)
raw_data = raw_data.merge(ground_data1.loc[:,['matchID', 'ground_runs_1', 'ground_wkts_1']], on='matchID', how='left')
raw_data = raw_data.merge(ground_data2.loc[:,['matchID', 'ground_runs_2', 'ground_wkts_2']], on='matchID', how='left')
raw_data = raw_data.merge(ground_data3.loc[:,['matchID', 'ground_runs_3', 'ground_wkts_3']], on='matchID', how='left')
raw_data['ground_runs_1'] = raw_data['ground_runs_1'].fillna(raw_data['ground_runs'])
raw_data['ground_wkts_1'] = raw_data['ground_wkts_1'].fillna(raw_data['ground_wkts'])
raw_data['ground_runs_2'] = raw_data['ground_runs_2'].fillna(raw_data['ground_runs'])
raw_data['ground_wkts_2'] = raw_data['ground_wkts_2'].fillna(raw_data['ground_wkts'])
raw_data['ground_runs_3'] = raw_data['ground_runs_3'].fillna(raw_data['ground_runs'])
raw_data['ground_wkts_3'] = raw_data['ground_wkts_3'].fillna(raw_data['ground_wkts'])
# 2nd innings only gets a share of the first innings' day deviation (SECOND_INNS_DAY_WEIGHT in
# ground_table_update) - swap in the damped reverted_*_2nd values for inningNumber 2
for phase in [1, 2, 3]:
    ground2nd = ground_data[ground_data.inningPhase == phase].rename(columns={'reverted_runs_2nd': f'ground_runs_{phase}_2nd', 'reverted_wkts_2nd': f'ground_wkts_{phase}_2nd'})
    raw_data = raw_data.merge(ground2nd.loc[:, ['matchID', f'ground_runs_{phase}_2nd', f'ground_wkts_{phase}_2nd']], on='matchID', how='left')
    inns2 = raw_data['inningNumber'] == 2
    raw_data[f'ground_runs_{phase}'] = np.where(inns2, raw_data[f'ground_runs_{phase}_2nd'].fillna(raw_data['ground_runs']), raw_data[f'ground_runs_{phase}'])
    raw_data[f'ground_wkts_{phase}'] = np.where(inns2, raw_data[f'ground_wkts_{phase}_2nd'].fillna(raw_data['ground_wkts']), raw_data[f'ground_wkts_{phase}'])
    raw_data = raw_data.drop(columns=[f'ground_runs_{phase}_2nd', f'ground_wkts_{phase}_2nd'])
# the same by inningPhase factors at the ground's normal level (non-day), for RA_Sum_nonDay
for phase in [1, 2, 3]:
    groundNonDay = ground_data[ground_data.inningPhase == phase].rename(columns={'runsratio_ground': f'ground_runs_{phase}_nonDay', 'wktsratio_ground': f'ground_wkts_{phase}_nonDay'})
    raw_data = raw_data.merge(groundNonDay.loc[:, ['matchID', f'ground_runs_{phase}_nonDay', f'ground_wkts_{phase}_nonDay']], on='matchID', how='left')
    raw_data[f'ground_runs_{phase}_nonDay'] = raw_data[f'ground_runs_{phase}_nonDay'].fillna(raw_data['ground_runs'])
    raw_data[f'ground_wkts_{phase}_nonDay'] = raw_data[f'ground_wkts_{phase}_nonDay'].fillna(raw_data['ground_wkts'])

#merge in wkt_values and make the wkt_value * exp_wkts values so we have a value in runs for exp totalInningWickets
wktvalues = wktvalues.drop_duplicates(subset=['overNumber', 'totalInningWickets'])
raw_data = raw_data.merge(wktvalues, on=('overNumber', 'totalInningWickets'), how='left')
raw_data['bat_exp_wktvalue'] = raw_data['wktvalue'] * raw_data['realexpwbat']
raw_data['bowl_exp_wktvalue'] = raw_data['wktvalue'] * raw_data['realexpwbowl']
raw_data['ground_exp_wktvalue'] = raw_data['wktvalue'] * raw_data['ovrexpw']
# surge BBL values (as 2_adjData_w): overs 1-10 always carry the bbl_csv ratio in the surge era, overs 11-20 only while
# the surge is on; ovrexpr_bbl / ovrexpw_bbl are the de-surged exposures
bblSurge = ((raw_data['competition'] == 'Big Bash League') & (raw_data['date'] > date(2020, 6, 6))).to_numpy()
raw_data = raw_data.merge(bblvalues, on='overNumber', how='left')
bblOn = bblSurge & ((raw_data['overNumber'] < 11) | (raw_data['isPowerSurge'] == 1)).to_numpy()
raw_data['bbl_runs_ratio'] = np.where(bblOn, raw_data['bbl_runs_ratio'], 1)
raw_data['bbl_wkts_ratio'] = np.where(bblOn, raw_data['bbl_wkts_ratio'], 1)
raw_data['ovrexpr_bbl'] = raw_data['ovrexpr'] / raw_data['bbl_runs_ratio']
raw_data['ovrexpw_bbl'] = raw_data['ovrexpw'] / raw_data['bbl_wkts_ratio']
raw_data['bbl_exp_wktvalue'] = raw_data['wktvalue'] * raw_data['ovrexpw_bbl']

###now we can work out the ra_bowl already for each ball, we'll sum them later
# ---- phase / type / hand rating splits (prod_model/raSumSplits.py, armadacricket-d0 2026-10-15; rules in its docstring) ----
# bowlers: each actual ball x runs_mult[bowler, 5-phase] x runs_factor[bowler, striker's hand] (wickets the same). For now the
# real bowlers and real strikers (hindsight, as rar_bowl_sum always was) - the bowling plan replaces them when it exists
import raSumSplits as rss
splitStyle, splitHand = rss.player_styles(connection, 'men')
bowlSplitRuns, bowlSplitWkts = rss.bowler_multipliers(raw_data, splitStyle, splitHand, 'men')
# ---- take the match's own day ground value out of the bowler exposures (armadacricket-d0, 2026-10-15) ----
# realexprbowl / realexpwbowl come from the realexp SQL, which multiplies in ground_table reverted_runs / reverted_wkts (the
# match's own post-match day value; runsratio_ground where that's missing), joined on phase only. Swap it for the ground's
# NORMAL level (runsratio_ground / wktsratio_ground): exposure x normal / day. The pitch so far is applied to the rest-of-
# innings sum below (rar_bowl_sum x groundCondSoFar at the ball - one estimate for all the balls still to come, never each
# future ball's own). Wickets stay at the normal level, as the pitch blend. Still in realexp: the actual BBL surge factor.
_gd = ground_data.drop_duplicates(subset=['matchID', 'inningPhase'])[['matchID', 'inningPhase', 'reverted_runs', 'reverted_wkts', 'runsratio_ground', 'wktsratio_ground']]
_gb = raw_data[['matchID', 'inningPhase']].merge(_gd, on=['matchID', 'inningPhase'], how='left')
assert len(_gb) == len(raw_data)
_normR, _normW = _gb['runsratio_ground'].to_numpy(), _gb['wktsratio_ground'].to_numpy()
_dayR, _dayW = _gb['reverted_runs'].fillna(_gb['runsratio_ground']).to_numpy(), _gb['reverted_wkts'].fillna(_gb['wktsratio_ground']).to_numpy()
dayOutRuns = np.where((_dayR > 0) & ~np.isnan(_normR), _normR / np.where(_dayR > 0, _dayR, 1), 1.0)
dayOutWkts = np.where((_dayW > 0) & ~np.isnan(_normW), _normW / np.where(_dayW > 0, _dayW, 1), 1.0)
raw_data['rar_bowl'] = (raw_data['oppo_bowl_runs'] * bowlSplitRuns - 1) * raw_data['realexprbowl'] * dayOutRuns
raw_data['raw_bowl'] = (1 - raw_data['oppo_bowl_wkts'] * bowlSplitWkts) * raw_data['bowl_exp_wktvalue'] * dayOutWkts
raw_data['rar_bbl'] = (raw_data['bbl_runs_ratio'] - 1) * raw_data['ovrexpr_bbl']
raw_data['raw_bbl'] = (1 - raw_data['bbl_wkts_ratio']) * raw_data['bbl_exp_wktvalue']

##now work out the sums of various things for the rest of the inningNumber
er_tc = raw_data.iloc[::-1].groupby(['batsmanName', 'matchID', 'inningNumber'], sort=False)['realexprbat'].transform(lambda x: x.rolling(150, min_periods=1, closed='right').sum()).iloc[::-1].reset_index().fillna(0)
er_tc_ground = raw_data.iloc[::-1].groupby(['inningPhase', 'matchID', 'inningNumber'], sort=False)['ovrexpr'].transform(lambda x: x.rolling(150, min_periods=1, closed='right').sum()).iloc[::-1].reset_index().fillna(0)
ew_tc_bat = raw_data.iloc[::-1].groupby(['batsmanName', 'matchID', 'inningNumber'], sort=False)['bat_exp_wktvalue'].transform(lambda x: x.rolling(150, min_periods=1, closed='right').sum()).iloc[::-1].reset_index().fillna(0)
ew_tc_ground = raw_data.iloc[::-1].groupby(['inningPhase', 'matchID', 'inningNumber'], sort=False)['ground_exp_wktvalue'].transform(lambda x: x.rolling(150, min_periods=1, closed='right').sum()).iloc[::-1].reset_index().fillna(0)
rar_bowl = raw_data.iloc[::-1].groupby(['matchID', 'inningNumber'], sort=False)['rar_bowl'].transform(lambda x: x.rolling(150, min_periods=1, closed='right').sum()).iloc[::-1].reset_index().fillna(0)
raw_bowl = raw_data.iloc[::-1].groupby(['matchID', 'inningNumber'], sort=False)['raw_bowl'].transform(lambda x: x.rolling(150, min_periods=1, closed='right').sum()).iloc[::-1].reset_index().fillna(0)
raw_data['er_tc'] = er_tc['realexprbat']
raw_data['er_tc_ground'] = er_tc_ground['ovrexpr']
raw_data['ew_tc'] = ew_tc_bat['bat_exp_wktvalue']
raw_data['ew_tc_ground'] = ew_tc_ground['ground_exp_wktvalue']
raw_data['rar_bowl_sum'] = rar_bowl['rar_bowl']
raw_data['raw_bowl_sum'] = raw_bowl['raw_bowl']
# batter splits: each striker's expected runs to come by 5-phase, and the remaining balls' bowler-type mix by phase
rss.batter_phase_runs(raw_data)
splitIds, splitCnt, splitTot = rss.type_mix(raw_data)
rar_bbl = raw_data.iloc[::-1].groupby(['matchID', 'inningNumber'], sort=False)['rar_bbl'].transform(lambda x: x.rolling(150, min_periods=1, closed='right').sum()).iloc[::-1].reset_index().fillna(0)
raw_bbl = raw_data.iloc[::-1].groupby(['matchID', 'inningNumber'], sort=False)['raw_bbl'].transform(lambda x: x.rolling(150, min_periods=1, closed='right').sum()).iloc[::-1].reset_index().fillna(0)
raw_data['rar_bbl_sum'] = rar_bbl['rar_bbl']
raw_data['raw_bbl_sum'] = raw_bbl['raw_bbl']

# ---- BBL power surge to come: the CHANCE it's taken in each remaining over (armadacricket-d0 spec, 2026-10-15) ----
# rar_bbl_sum / raw_bbl_sum above use the overs where the surge was ACTUALLY taken, which aren't known live. For
# surge-era BBL (date > 2020-06-06) they are replaced here, per ball:
#   - surge already started in this innings (including the current over - known once the over has begun): its two
#     overs are known, so the actual-ratio sums above stand
#   - otherwise: overs 1-10 always carry their bbl_csv ratio (actual sums, unchanged); for each later over k after the
#     current one, P_k = P(start = k) + P(start = k - 1) from PythonData/bbl_surge/remaining_smooth_men.csv at
#     over_end = overs completed, wickets now, with a start in the current over ruled out (it isn't on) and the rest
#     renormalised. Expected ratio = 1 + P_k x (ratio_k - 1), runs and wickets separately, applied to that over's
#     de-surged exposure (ovrexpr_bbl / bbl_exp_wktvalue of the balls actually bowled in it)
# KNOWN GAP: realexprbat / er_tc / ovrexpr come from the SQL functions, which use the ACTUAL surge (competition factor x
# surge flag) - same class of issue as the day ground value. Not fixed here (SQL left alone).
SURGE_FIRST, SURGE_LAST = 11, 19   # possible start overs in the table
surgeTable = pd.read_csv(PROJECT_ROOT / 'bblSurge/outputs/remaining_smooth_men.csv')
surgeEra = bblSurge
surgeOver = raw_data['overNumber'].to_numpy().astype(int)
surgeInn = raw_data.groupby(['matchID', 'inningNumber'], sort=False).ngroup().to_numpy()
surgeStart = raw_data['overNumber'].where(raw_data['isPowerSurge'] == 1).groupby([raw_data['matchID'], raw_data['inningNumber']]).transform('min').to_numpy()
surgeOpen = surgeEra & ~(surgeStart <= surgeOver)   # NaN start (never taken) compares False -> still open
# start probabilities at (overs completed, wickets now); a start in the current over is ruled out
startCols = [str(k) for k in range(SURGE_FIRST, SURGE_LAST + 1)]
tbl = surgeTable.set_index(['over_end', 'wickets'])
key = pd.MultiIndex.from_arrays([surgeOver - 1, np.clip(raw_data['totalInningWickets'].to_numpy(), 0, 9)])
Q = tbl.reindex(key)[startCols + ['never']].to_numpy()          # rows past over_end 18 are NaN -> no future surge overs
Q[:, :len(startCols)] = np.where(np.arange(SURGE_FIRST, SURGE_LAST + 1)[None, :] == surgeOver[:, None], 0.0, Q[:, :len(startCols)])
Q = np.nan_to_num(Q / Q.sum(1, keepdims=True))
Qs = Q[:, :len(startCols)]
P = np.zeros((len(raw_data), 10))                                 # P(surge on) in overs 11..20
P[:, :9] += Qs
P[:, 1:] += Qs
P = np.where(np.arange(11, 21)[None, :] > surgeOver[:, None], P, 0.0)   # later overs only
bblRatios = bblvalues.set_index('overNumber').reindex(range(11, 21))
runsUp = bblRatios['bbl_runs_ratio'].to_numpy() - 1
wktsDown = 1 - bblRatios['bbl_wkts_ratio'].to_numpy()
# de-surged exposure of each innings' overs 11..20 (balls actually bowled)
late = surgeOver >= 11
expo = pd.DataFrame({'inn': surgeInn[late], 'k': surgeOver[late] - 11, 'r': raw_data['ovrexpr_bbl'].to_numpy()[late], 'w': raw_data['bbl_exp_wktvalue'].to_numpy()[late]})
nInn = surgeInn.max() + 1
Eruns, Ewkts = np.zeros((nInn, 10)), np.zeros((nInn, 10))
np.add.at(Eruns, (expo['inn'], expo['k']), expo['r'].fillna(0))
np.add.at(Ewkts, (expo['inn'], expo['k']), expo['w'].fillna(0))
# actual sums over overs 1-10 still to come (ratios there are fixed)
early = pd.DataFrame({'r': np.where(surgeOver <= 10, raw_data['rar_bbl'].fillna(0), 0.0), 'w': np.where(surgeOver <= 10, raw_data['raw_bbl'].fillna(0), 0.0)})
earlyRuns = early['r'].iloc[::-1].groupby(surgeInn[::-1]).cumsum().iloc[::-1].to_numpy()
earlyWkts = early['w'].iloc[::-1].groupby(surgeInn[::-1]).cumsum().iloc[::-1].to_numpy()
expRuns = earlyRuns + (P * runsUp[None, :] * Eruns[surgeInn]).sum(1)
expWkts = earlyWkts + (P * wktsDown[None, :] * Ewkts[surgeInn]).sum(1)
raw_data['rar_bbl_sum'] = np.where(surgeOpen, expRuns, raw_data['rar_bbl_sum'])
raw_data['raw_bbl_sum'] = np.where(surgeOpen, expWkts, raw_data['raw_bbl_sum'])

# raw_data.to_csv(fr'{user_name}\Documents\Tempdata\raw_data_mmrra.csv', index=False)
# raw_data = pd.read_csv(fr'{user_name}\Documents\Tempdata\raw_data_mmrra.csv')
#
#
#
# # ##################need from here to mkae his ########################################################
# raw_data = pd.read_csv(fr'{user_name}\Documents\Tempdata\raw_data_mmrra.csv')

###EXPECTED RUNS TO COME SMOOTHED, just for current batters
fitData = raw_data[~bblSurge]   # 'average situation' curves fitted without surge-era BBL, applied to all balls
er_tc_avg_now = pd.pivot_table(fitData, values=['er_tc', 'ew_tc'], index=['inningBallsRemaining', 'totalInningWickets', 'ord'], aggfunc={'er_tc': ['mean', 'count'], 'ew_tc': ['count', 'mean']}).reset_index()
er_tc_avg_now.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in er_tc_avg_now.columns.to_flat_index()]
er_tc_avg_now_ground = pd.pivot_table(fitData, values=['er_tc_ground', 'ew_tc_ground'], index=['inningBallsRemaining', 'totalInningWickets', 'inningPhase'], aggfunc={'er_tc_ground': ['mean', 'count'], 'ew_tc_ground': ['count', 'mean']}).reset_index()
er_tc_avg_now_ground.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in er_tc_avg_now_ground.columns.to_flat_index()]


X = er_tc_avg_now[['inningBallsRemaining', 'totalInningWickets', 'ord']]  # Independent variables
y = er_tc_avg_now['er_tc_mean']  # Dependent variable
weights = er_tc_avg_now['er_tc_count']  # Weights
poly = PolynomialFeatures(degree=3)  # Choose your degree
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y, sample_weight=weights)

y2 = er_tc_avg_now['ew_tc_mean']  # Dependent variable
weights2 = er_tc_avg_now['ew_tc_count']  # Weights
model2 = LinearRegression()
model2.fit(X_poly, y2, sample_weight=weights2)

X_predict = raw_data[['inningBallsRemaining', 'totalInningWickets', 'ord']]
X_predict = poly.fit_transform(X_predict)
raw_data['er_tc_smooth_basic'] = model.predict(X_predict)
raw_data['er_tc_smooth_basic'] = np.maximum(0, raw_data['er_tc_smooth_basic'])
raw_data['ew_tc_smooth_basic'] = model2.predict(X_predict)
raw_data['ew_tc_smooth_basic'] = np.maximum(0, raw_data['ew_tc_smooth_basic'])

##### calc for balls faced per ord per situation:
raw_data_inns1 = fitData.copy()   # without surge-era BBL (see fitData)
raw_data_inns1 = raw_data_inns1[raw_data_inns1.inningNumber == 1]
max = raw_data_inns1.groupby(['matchID', 'inningNumber'])['ord'].max().reset_index()

match_ids = raw_data_inns1[['matchID', 'inningNumber']].drop_duplicates()

dnb = pd.DataFrame({
    'matchID': np.repeat(max['matchID'].values, 9),
    'inningNumber': np.repeat(max['inningNumber'].values, 9),
    'ord': np.tile(range(3, 12), len(max))
})

dnb = dnb.merge(max, how='left', on=('matchID', 'inningNumber'), suffixes= ('', '_max'))

dnb = dnb[dnb.ord > dnb.ord_max]
dnb['er_tc'] = 0
dnb['ew_tc'] = 0
dnb.drop(['ord_max'], axis=1, inplace=True)
dnb['inningBallsRemaining'] = 0
dnb['totalInningWickets'] = 0

PHASE_COLS = [f'er_tc_p{p}' for p in range(1, 6)]
for c in PHASE_COLS:
    dnb[c] = 0
test = raw_data_inns1.loc[:, ['matchID', 'inningNumber', 'totalInningWickets', 'ord', 'er_tc', 'ew_tc', 'inningBallsRemaining'] + PHASE_COLS]
test = pd.concat([test, dnb], ignore_index=True)


expr_pred = pd.DataFrame()

for BR in range(120, 0, -1):
    for WL in range(10):

# BR = 120
# WL = 0

        test2 = test[test.inningBallsRemaining == BR]
        test2 = test2[test2.totalInningWickets == WL]
        test3 = test[test.inningBallsRemaining <= BR]

        test2['include'] = 1

        test4 = test3.merge(test2.loc[:, ['matchID', 'inningNumber', 'include']], on=('matchID', 'inningNumber'), how='left')
        test4['include'] = test4['include'].fillna(0)
        test4 = test4[test4.include == 1]
        test4 = test4.sort_values(by=['matchID', 'inningNumber', 'ord', 'inningBallsRemaining'], ascending=[True, True, True, False])
        test4 = test4.drop_duplicates(subset=['matchID', 'inningNumber', 'ord'], keep='first')
        test4 = test4.groupby(['ord'])[['er_tc', 'ew_tc'] + PHASE_COLS].mean().reset_index()
        test4['inningBallsRemaining'] = BR
        test4['totalInningWickets'] = WL

        expr_pred = pd.concat([expr_pred, test4], ignore_index=True)
#
# phase shares of a batter's runs to come: striker (now) and everyone else (first-appearance averages, as expr_pred)
splitShareNow, splitShareCard = rss.share_tables(raw_data_inns1, expr_pred)
###smooth the above finding for er_tc for all totalInningWickets in all situations
X = expr_pred[['inningBallsRemaining', 'ord', 'totalInningWickets']]  # Independent variables
y = expr_pred['er_tc']
y2 = expr_pred['ew_tc']# Dependent variable
poly = PolynomialFeatures(degree=4)  # Choose your degree
X = poly.fit_transform(X)
model = LinearRegression()
model.fit(X, y)
model2 = LinearRegression()
model2.fit(X, y2)
expr_pred['er_tc_smooth'] = model.predict(X)
expr_pred['er_tc_smooth'] = np.maximum(expr_pred['er_tc_smooth'], 0)
expr_pred['ew_tc_smooth'] = model2.predict(X)
expr_pred['ew_tc_smooth'] = np.maximum(expr_pred['ew_tc_smooth'], 0)
# expr_pred.to_csv(fr'{user_name}\OneDrive - Decimal Data Services Ltd\PythonData\MatchMarket\er_tc_smooth.csv', index=False)
# ### calc for balls faced per ord per situation ground:

max2 = raw_data_inns1.groupby(['matchID', 'inningNumber'])['inningPhase'].max().reset_index()

match_ids2 = raw_data_inns1[['matchID', 'inningNumber']].drop_duplicates()

dnb2 = pd.DataFrame({
    'matchID': np.repeat(max2['matchID'].values, 2),
    'inningNumber': np.repeat(max2['inningNumber'].values, 2),
    'inningPhase': np.tile(range(2, 4), len(max2))
})

dnb2 = dnb2.merge(max2, how='left', on=('matchID', 'inningNumber'), suffixes= ('', '_max'))

dnb2 = dnb2[dnb2.inningPhase > dnb2.inningPhase_max]
dnb2['er_tc_ground'] = 0
dnb2['ew_tc_ground'] = 0
dnb2.drop(['inningPhase_max'], axis=1, inplace=True)
dnb2['inningBallsRemaining'] = 0
dnb2['totalInningWickets'] = 0

test12 = raw_data_inns1.loc[:, ['matchID', 'inningNumber', 'totalInningWickets', 'inningPhase', 'er_tc_ground', 'ew_tc_ground', 'inningBallsRemaining']]
test12 = pd.concat([test12, dnb2], ignore_index=True)


expr_pred2 = pd.DataFrame()

for BR in range(120, 0, -1):
    for WL in range(10):

# BR = 120
# WL = 0

        test22 = test12[test12.inningBallsRemaining == BR]
        test22= test22[test22.totalInningWickets == WL]
        test32 = test12[test12.inningBallsRemaining <= BR]

        test22['include'] = 1

        test42 = test32.merge(test22.loc[:, ['matchID', 'inningNumber', 'include']], on=('matchID', 'inningNumber'), how='left')
        test42['include'] = test42['include'].fillna(0)
        test42 = test42[test42.include == 1]
        test42 = test42.sort_values(by=['matchID', 'inningNumber', 'inningPhase', 'inningBallsRemaining'], ascending=[True, True, True, False])
        test42 = test42.drop_duplicates(subset=['matchID', 'inningNumber', 'inningPhase'], keep='first')
        test42 = test42.groupby(['inningPhase'])[['er_tc_ground', 'ew_tc_ground']].mean().reset_index()
        test42['inningBallsRemaining'] = BR
        test42['totalInningWickets'] = WL

        expr_pred2 = pd.concat([expr_pred2, test42], ignore_index=True)
# #
###smooth the above finding for er_tc for all totalInningWickets in all situations
X12 = expr_pred2[['inningBallsRemaining', 'inningPhase', 'totalInningWickets']]  # Independent variables
y12 = expr_pred2['er_tc_ground']
y22 = expr_pred2['ew_tc_ground']# Dependent variable
poly12 = PolynomialFeatures(degree=4)  # Choose your degree
X12 = poly12.fit_transform(X12)
model12 = LinearRegression()
model12.fit(X12, y12)
model22 = LinearRegression()
model22.fit(X12, y22)
# expr_pred2['er_tc_ground_smooth'] = model12.predict(X12)
# expr_pred2['er_tc_ground_smooth'] = np.maximum(expr_pred['er_tc_ground_smooth'], 0)
# expr_pred2['ew_tc_ground_smooth'] = model22.predict(X12)
# expr_pred2['ew_tc_ground_smooth'] = np.maximum(expr_pred['ew_tc_ground_smooth'], 0)

raw_data['1'] = 1
raw_data['2'] = 2
raw_data['3'] = 3
X_predict22 = raw_data.loc[:,['inningBallsRemaining', '1', 'totalInningWickets']]
X_predict22 = poly12.fit_transform(X_predict22)
raw_data['er_tc_1'] = model12.predict(X_predict22)
raw_data['er_tc_1'] = np.where(raw_data['inningPhase'] > 1, 0, np.maximum(raw_data['er_tc_1'], 0))
raw_data['ew_tc_1'] = model22.predict(X_predict22)
raw_data['ew_tc_1'] = np.where(raw_data['inningPhase'] > 1, 0, np.maximum(raw_data['ew_tc_1'], 0))
X_predict23 = raw_data.loc[:,['inningBallsRemaining', '2', 'totalInningWickets']]
X_predict23 = poly12.fit_transform(X_predict23)
raw_data['er_tc_2'] = model12.predict(X_predict23)
raw_data['er_tc_2'] = np.where(raw_data['inningPhase'] > 2, 0, np.maximum(raw_data['er_tc_2'], 0))
raw_data['ew_tc_2'] = model22.predict(X_predict23)
raw_data['ew_tc_2'] = np.where(raw_data['inningPhase'] > 2, 0, np.maximum(raw_data['ew_tc_2'], 0))
X_predict24 = raw_data.loc[:,['inningBallsRemaining', '3', 'totalInningWickets']]
X_predict24 = poly12.fit_transform(X_predict24)
raw_data['er_tc_3'] = model12.predict(X_predict24)
raw_data['er_tc_3'] = np.maximum(raw_data['er_tc_3'], 0)
raw_data['ew_tc_3'] = model22.predict(X_predict24)
raw_data['ew_tc_3'] = np.maximum(raw_data['ew_tc_3'], 0)

# ---- pitch on the day: conditions so far (auxiliaries/conditionsSoFar.py and conditionsSoFar2nd.py, armadacricket-d0) ----
# The day version of RA_Sum used the match's full post-match reverted day value (ground_table reverted_runs / _2nd) from
# ball 1, i.e. knew the whole innings' scoring in advance. That post-match value (ground_table_update's W120 / the 0.57
# second innings share) must never feed a live prediction; this blend replaces it, using only the balls bowled so far:
#   ratio so far = sum actual / sum expected over the non-wide balls BEFORE this one in this innings (1 when n = 0)
#       actual   = totalRuns - byeRuns - noballRuns
#       expected = ovrexpr x the ground's normal level (ground_runs, by phase - no day value; not realexprbat, which
#                  holds the match's own reverted day value)
#       order    = inningBallNumber, then extras first, then ID (the sort at the top - IDs aren't in playing order from 2025)
#       n        = number of those balls
#   innings 1: factor = 1 + wThisRuns[n] x (ratio so far - 1)                                 conditionsSoFar.csv, inn 1 rows
#   innings 2: s = startShareRuns (0.57 x W120) x (whole innings 1 ratio - 1)
#              x = sum actual so far / (sum expected so far x (1 + s)) - 1 (0 when n = 0)
#              factor = (1 + s) x (1 + wThisRuns[n] x x)                                      conditionsSoFar2nd.csv
#   ground runs for each phase still to come = the ground's normal level for that phase (ground_runs_p_nonDay) x factor
#   Wickets: factor 1 in both innings (no carry-over beat the ground normal held out). The Hundred: factor 1 (not fitted).
#   An innings with any non-wide ball missing its expectation gets ratio 1 (and s = 0 if it's innings 1).
# Held out, rest-of-innings error vs the ground normal: men 0.975 / 0.970, women 0.977 / 0.964 (inn 1 / inn 2).
# This replaces the reverted / reverted_*_2nd day values merged above (2026-10-14). Worked example to reproduce:
# men/expBall&runsToCome/auxiliaries/conditionsSoFar_worked_example.txt
condSoFar1 = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/auxiliaries/conditionsSoFar.csv')
condSoFar1 = condSoFar1.loc[condSoFar1['inningNumber'] == 1].set_index('ballsSoFar')['wThisRuns']
condSoFar2 = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/auxiliaries/conditionsSoFar2nd.csv').set_index('ballsSoFar')
csCounted = (raw_data['wideRuns'] == 0).to_numpy()
cs = raw_data.loc[:, ['matchID', 'inningNumber']].copy()  # raw_data is in playing order here (see the sort at the top)
cs['A'] = np.where(csCounted, (raw_data['totalRuns'] - raw_data['byeRuns'] - raw_data['noballRuns']).clip(lower=0), 0.0)
cs['E'] = np.where(csCounted, raw_data['ovrexpr'] * raw_data['ground_runs'].fillna(1.0), 0.0)
cs['n'] = csCounted.astype(float)
cs['missing'] = csCounted & cs['E'].isna().to_numpy()
csInnings = cs.groupby(['matchID', 'inningNumber'], sort=False)
csIncomplete = csInnings['missing'].transform('any').to_numpy()
cs['E'] = cs['E'].fillna(0.0)
csSums = csInnings[['A', 'E', 'n']].cumsum() - cs[['A', 'E', 'n']]  # BEFORE this ball
csOk = (csSums['E'] > 0).to_numpy() & ~csIncomplete
csN = csSums['n'].astype(int)
# innings 1
csW1 = condSoFar1.reindex(csN.clip(upper=condSoFar1.index.max())).fillna(0).to_numpy()
csFactor1 = 1 + csW1 * (np.where(csOk, csSums['A'] / csSums['E'].where(csSums['E'] > 0), 1.0) - 1)
# innings 2: start from the share of the whole first innings the ratings carry over, then blend in this innings so far
csInn1 = cs.loc[(cs['inningNumber'] == 1).to_numpy() & ~csIncomplete].groupby('matchID')[['A', 'E']].sum()
csInn1Ratio = (csInn1['A'] / csInn1['E'].where(csInn1['E'] > 0)).reindex(cs['matchID']).fillna(1.0).to_numpy()
csS = condSoFar2['startShareRuns'].iloc[0] * (csInn1Ratio - 1)
csW2 = condSoFar2['wThisRuns'].reindex(csN.clip(upper=condSoFar2.index.max())).fillna(0).to_numpy()
csX = np.where(csOk, csSums['A'] / (csSums['E'].where(csSums['E'] > 0) * (1 + csS)) - 1, 0.0)
csFactor2 = (1 + csS) * (1 + csW2 * csX)
csInning = raw_data['inningNumber'].to_numpy()
raw_data['groundCondSoFar'] = np.where(csInning == 1, csFactor1, np.where(csInning == 2, csFactor2, 1.0))
raw_data.loc[raw_data['competition'].fillna('').str.startswith('The Hundred'), 'groundCondSoFar'] = 1.0
for phase in [1, 2, 3]:
    raw_data[f'ground_runs_{phase}'] = raw_data[f'ground_runs_{phase}_nonDay'] * raw_data['groundCondSoFar']
    raw_data[f'ground_wkts_{phase}'] = raw_data[f'ground_wkts_{phase}_nonDay']

# bowler runs still to come: at the pitch so far as it stands at this ball (see 'take the match's own day ground value out')
raw_data['rar_bowl_sum'] = raw_data['rar_bowl_sum'] * raw_data['groundCondSoFar']

raw_data['rar_ground_sum'] = ((raw_data['ground_runs_1'] - 1) * raw_data['er_tc_1']) + ((raw_data['ground_runs_2'] - 1) * raw_data['er_tc_2']) + ((raw_data['ground_runs_3'] - 1) * raw_data['er_tc_3'])
raw_data['raw_ground_sum'] = -(((raw_data['ground_wkts_1'] - 1) * raw_data['ew_tc_1']) + ((raw_data['ground_wkts_2'] - 1) * raw_data['ew_tc_2']) + ((raw_data['ground_wkts_3'] - 1) * raw_data['ew_tc_3']))
# same formulas and exposures at the ground's normal level, for RA_Sum_nonDay
raw_data['rar_ground_sum_nonDay'] = ((raw_data['ground_runs_1_nonDay'] - 1) * raw_data['er_tc_1']) + ((raw_data['ground_runs_2_nonDay'] - 1) * raw_data['er_tc_2']) + ((raw_data['ground_runs_3_nonDay'] - 1) * raw_data['er_tc_3'])
raw_data['raw_ground_sum_nonDay'] = -(((raw_data['ground_wkts_1_nonDay'] - 1) * raw_data['ew_tc_1']) + ((raw_data['ground_wkts_2_nonDay'] - 1) * raw_data['ew_tc_2']) + ((raw_data['ground_wkts_3_nonDay'] - 1) * raw_data['ew_tc_3']))

# raw_data.to_csv(fr'{user_name}\Documents\Tempdata\raw_data_mmrra2.csv', index=False)

lineups = pd.read_sql_query("""select matchid as "matchID", player as batter, playerid as batterid, team as "battingTeam", carded from player_ratings.t20_lineups_updated""", con=connection)
# lineup ids (2026-10-14): the men's t20_lineups_updated holds players.playerid up to 2024 and match-data ids (= players.newid)
# from 2025; the two systems overlap (the same number can be two different players). Per (match, team) use whichever
# system matches more of the ids seen in the ball data for that match. Women's lineups are match-data ids throughout,
# so this leaves them as they are
playerIds = pd.read_sql_query("""select playerid, newid from players_teams.players where newid is not null""", con=connection).drop_duplicates(subset=['playerid'])
seenIds = set(zip(raw_data['matchID'], raw_data['batterid'])) | set(zip(raw_data['matchID'], raw_data['bowlerid']))
lineups['_new'] = lineups['batterid'].map(playerIds.set_index('playerid')['newid'])
lineups['_hitRaw'] = [(m, p) in seenIds for m, p in zip(lineups['matchID'], lineups['batterid'])]
lineups['_hitNew'] = [(m, p) in seenIds for m, p in zip(lineups['matchID'], lineups['_new'])]
lineupHits = lineups.groupby(['matchID', 'battingTeam'])[['_hitRaw', '_hitNew']].transform('mean')
lineups['batterid'] = np.where(lineupHits['_hitNew'] > lineupHits['_hitRaw'], lineups['_new'], lineups['batterid'])
lineups = lineups.drop(columns=['_new', '_hitRaw', '_hitNew'])
lineups = lineups.drop_duplicates(subset=['matchID', 'batterid'])

# fallback for innings with no t20_lineups_updated rows: without it the inner join in `his` below drops every ball of
# that innings, so it gets no RA_sum at all (~8% of men's balls, ~20% in 2015-17). Uses the batting order seen in the
# ball data (each striker's ord) - batters who never faced a ball are missing from it
seenLineups = raw_data.loc[raw_data['batsmanName'].notna(), ['matchID', 'battingTeam', 'batsmanName', 'batterid', 'ord_real']].drop_duplicates(subset=['matchID', 'battingTeam', 'batsmanName'])
seenLineups = seenLineups.rename(columns={'batsmanName': 'batter', 'ord_real': 'carded'})
seenLineups = seenLineups.merge(lineups.loc[:, ['matchID', 'battingTeam']].drop_duplicates().assign(_hasLineup=1), on=['matchID', 'battingTeam'], how='left')
lineups = pd.concat([lineups, seenLineups[seenLineups['_hasLineup'].isna()].drop(columns=['_hasLineup'])], ignore_index=True)
# lineups.to_csv(fr'{user_name}\Documents\Tempdata\lineups.csv', index=False)



# expr_pred = pd.read_csv(fr'{user_name}\OneDrive - Decimal Data Services Ltd\PythonData\MatchMarket\er_tc_smooth.csv')
# raw_data = pd.read_csv(fr'{user_name}\Documents\Tempdata\raw_data_mmrra2.csv')
# lineups = pd.read_csv(fr'{user_name}\Documents\Tempdata\lineups.csv')

####### applying to the data (both inningNumber):
# non-striker's own live balls-faced-so-far isn't directly available per ball - raw_data's
# 'batsmanBallsFaced' column only ever reflects whoever is STRIKING that specific ball. Build it via an
# as-of lookup: for every ball, find that player's own most recent batsmanBallsFaced checkpoint (the
# last ball where THEY were striking), within the same match/inningNumber. Used below (with the
# striker's own batsmanBallsFaced) for the balls-faced adjustment, applied only to the two batters
# actually at the crease.
raw_data['_origIdx'] = raw_data.index
# ordered by 'inningBallNumber', not 'ID' - see the note above the earlier sort_values for why ID can't be
# trusted as chronological order; merge_asof needs its 'on' key properly ordered to find the right
# "most recent" checkpoint
ballCheckpoints = raw_data.loc[:, ['matchID', 'inningNumber', 'inningBallNumber', 'batsmanName', 'batsmanBallsFaced']].rename(columns={'batsmanName': 'player'}).sort_values('inningBallNumber')
nonstrikerLookup = raw_data.loc[:, ['matchID', 'inningNumber', 'inningBallNumber', 'nonstrikerName', '_origIdx']].rename(columns={'nonstrikerName': 'player'}).sort_values('inningBallNumber')
nonstrikerBalls = pd.merge_asof(nonstrikerLookup, ballCheckpoints, on='inningBallNumber', by=['matchID', 'inningNumber', 'player'], direction='backward')
nonstrikerBalls = nonstrikerBalls.set_index('_origIdx').sort_index()
raw_data['nonstriker_batsmanballs'] = nonstrikerBalls['batsmanBallsFaced'].fillna(0)
raw_data = raw_data.drop(columns=['_origIdx'])

# ---- batter form (auxiliaries/in_live_ratings.py -> auxiliaries/batterFormAdjust.csv) ----
# Each in-batter's runs so far vs expected so far (whole innings, and last 10 balls) at this ball, built exactly as
# in_live_ratings builds it: legal balls BEFORE this one, expected runs per ball = realexprbat x the batter's clean
# rating (oppo_bat_runs_preadj, 1 if missing) x FXB (newest/other) x SLW (non-newest only), from fxbSlwLookup.csv.
# formRunsToComeAdjust then multiplies that batter's expected runs for the rest of the innings (applied in `his`
# below, after the fxb balls-faced ratio). Runs only - form showed no effect on dismissals. No adjustment below 10
# balls faced, or for an innings with any ball missing its rated expectation (in_live_ratings drops those)
FORM_MIN_BALLS, FORM_MAX_BALLS, FORM_LAST_BALLS, FORM_LABEL_STEP, FORM_LABEL_MAX = 10, 50, 10, 0.1, 3.0
fxbSlwLookup = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/auxiliaries/fxbSlwLookup.csv')
formAdjust = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/auxiliaries/batterFormAdjust.csv')
curveCap = int(fxbSlwLookup['balls'].max())  # the top balls row means that number and above

raw_data['_pos'] = np.arange(len(raw_data))  # raw_data is in true delivery order here (see the sort at the top)
formBalls = raw_data.loc[raw_data['isValid'] == 1, ['_pos', 'matchID', 'inningNumber', 'batsmanName', 'ord_real', 'overNumber', 'totalInningWickets',
                                                    'batsmanBallsFaced', 'batsmanRuns', 'realexprbat', 'oppo_bat_runs_preadj']].copy()
formBalls['batsmanRuns'] = formBalls['batsmanRuns'].clip(lower=0)
# legal balls since the last wicket, including this one (SLW's clock)
formBalls['ballsSinceWicket'] = formBalls.groupby(['matchID', 'inningNumber', 'totalInningWickets']).cumcount() + 1
formIsNewest = ((formBalls['totalInningWickets'] == 0) | (formBalls['ord_real'] == formBalls['totalInningWickets'] + 2)).to_numpy()
curveKeys = pd.DataFrame({'ord': formBalls['ord_real'].fillna(2).clip(1, 11).astype(int).to_numpy(),
                          'over': formBalls['overNumber'].clip(1, 20).astype(int).to_numpy(),
                          'innings': formBalls['inningNumber'].astype(int).to_numpy()})


def curve_runs_mult(curve, balls):
    keys = curveKeys.assign(curve=curve, balls=np.clip(balls, 1, curveCap).astype(int))
    return keys.merge(fxbSlwLookup.loc[:, ['curve', 'ord', 'over', 'innings', 'balls', 'runsMult']], on=['curve', 'ord', 'over', 'innings', 'balls'], how='left')['runsMult'].fillna(1.0).to_numpy()


fxbRunsMult = curve_runs_mult(np.where(formIsNewest, 'FXB_newest', 'FXB_other'), formBalls['batsmanBallsFaced'].to_numpy())
slwRunsMult = np.where(formIsNewest, 1.0, curve_runs_mult('SLW', formBalls['ballsSinceWicket'].to_numpy()))
formBalls['expRuns'] = formBalls['realexprbat'] * formBalls['oppo_bat_runs_preadj'].fillna(1.0) * fxbRunsMult * slwRunsMult
formInnings = formBalls.groupby(['matchID', 'inningNumber', 'batsmanName'], sort=False)
formBalls['formIncomplete'] = formInnings['expRuns'].transform(lambda x: x.isna().any())
# running totals INCLUDING each legal ball - looked up below from the batter's last legal ball BEFORE the current one
formBalls['runsSoFar'] = formInnings['batsmanRuns'].cumsum()
formBalls['expRunsSoFar'] = formInnings['expRuns'].cumsum()
formBalls['runsLast'] = formInnings['batsmanRuns'].transform(lambda x: x.rolling(FORM_LAST_BALLS, min_periods=1).sum())
formBalls['expRunsLast'] = formInnings['expRuns'].transform(lambda x: x.rolling(FORM_LAST_BALLS, min_periods=1).sum())
formBalls = formBalls.rename(columns={'batsmanName': 'player'}).loc[:, ['_pos', 'matchID', 'inningNumber', 'player', 'batsmanBallsFaced', 'runsSoFar', 'expRunsSoFar',
                                                                       'runsLast', 'expRunsLast', 'formIncomplete']].sort_values('_pos')


def form_runs_to_come(playerCol):
    look = raw_data.loc[:, ['_pos', 'matchID', 'inningNumber', playerCol]].rename(columns={playerCol: 'player'}).sort_values('_pos')
    state = pd.merge_asof(look, formBalls, on='_pos', by=['matchID', 'inningNumber', 'player'], direction='backward', allow_exact_matches=False)
    state = state.set_index('_pos').reindex(raw_data['_pos'])
    ratioSoFar = state['runsSoFar'] / state['expRunsSoFar']
    ratioLast = (state['runsLast'] / state['expRunsLast']).fillna(ratioSoFar)
    key = pd.DataFrame({
        'batsmanBallsFaced': (state['batsmanBallsFaced'].fillna(0) + 1).clip(upper=FORM_MAX_BALLS).to_numpy(),  # the ball they face next
        'ratioSoFarLabel': np.minimum(np.round(ratioSoFar / FORM_LABEL_STEP) * FORM_LABEL_STEP, FORM_LABEL_MAX).round(1).to_numpy(),
        'ratioLast10Label': np.minimum(np.round(ratioLast / FORM_LABEL_STEP) * FORM_LABEL_STEP, FORM_LABEL_MAX).round(1).to_numpy()})
    mult = key.merge(formAdjust.loc[:, ['batsmanBallsFaced', 'ratioSoFarLabel', 'ratioLast10Label', 'formRunsToComeAdjust']],
                     on=['batsmanBallsFaced', 'ratioSoFarLabel', 'ratioLast10Label'], how='left')['formRunsToComeAdjust'].to_numpy()
    usable = (key['batsmanBallsFaced'].to_numpy() >= FORM_MIN_BALLS) & (state['expRunsSoFar'].to_numpy() > 0) & (state['formIncomplete'].to_numpy() == False)
    return np.where(usable, np.nan_to_num(mult, nan=1.0), 1.0)


raw_data['form_striker'] = form_runs_to_come('batsmanName')
raw_data['form_nonstriker'] = form_runs_to_come('nonstrikerName')
raw_data = raw_data.drop(columns=['_pos'])

his = raw_data.copy()
# his_max_date = his['date'].max()
# his = his[his.matchID == 80457]
his['date'] = pd.to_datetime(his['date'])
his['year'] = his['date'].dt.year
his = his.loc[:, ['competition', 'inningBallsRemaining', 'totalInningWickets', 'matchID', 'batsmanName', 'nonstrikerName', 'battingTeam', 'ID', 'inningBallNumber', 'er_tc_smooth_basic',  'ew_tc_smooth_basic', 'rar_ground_sum', 'raw_ground_sum', 'rar_ground_sum_nonDay', 'raw_ground_sum_nonDay', 'rar_bowl_sum', 'raw_bowl_sum', 'rar_bbl_sum', 'raw_bbl_sum', 'year', 'inningNumber', 'batsmanBallsFaced', 'nonstriker_batsmanballs', 'form_striker', 'form_nonstriker']]
his = his.merge(lineups, on=('matchID', 'battingTeam')) #
his['in_now'] = np.where((his['batsmanName'] == his['batter']) | (his['nonstrikerName'] == his['batter']), 1, 0)
his['out'] = np.where((his['carded'] >= his['totalInningWickets'] + 2) | (his['in_now'] == 1), 1, 0)
his['wkts_till_bat'] = np.maximum((his['carded'] - his['totalInningWickets']) - 2, 0)
bat_ratings_game = raw_data.loc[:, ['matchID', 'batsmanName', 'oppo_bat_runs', 'oppo_bat_wkts', 'oppo_bat_runs_preadj', 'oppo_bat_wkts_preadj', 'expected_ord_rounded']].drop_duplicates(subset=['matchID', 'batsmanName'])
bat_ratings_game.columns = ['matchID', 'batter', 'oppo_bat_runs', 'oppo_bat_wkts', 'oppo_bat_runs_preadj', 'oppo_bat_wkts_preadj', 'expected_ord_rounded']
his = his.merge(bat_ratings_game, how='left', on=('matchID', 'batter'))
# batters on the card who haven't faced a ball in this match yet: their own pre-match row for this match, else their
# latest row before it, else 1 below (was their career mean by name / by competition, which included future matches)
cardKeys = his.loc[:, ['matchID', 'batterid']].drop_duplicates().merge(matchDates, on='matchID', how='left')
cardRatings = cardKeys.merge(batter_data.loc[:, ['matchID', 'batterid', 'run_rating_pr_ba', 'wkt_rating_pr_ba']], on=['matchID', 'batterid'], how='left')
cardEarlier = latest_before(cardKeys, batter_data, ['batterid'], ['run_rating_pr_ba', 'wkt_rating_pr_ba'])
cardRatings = cardRatings.merge(cardEarlier, on=['matchID', 'batterid'], how='left', suffixes=('', '_earlier'))
cardRatings['oppo_bat_runs_card'] = cardRatings['run_rating_pr_ba'].fillna(cardRatings['run_rating_pr_ba_earlier'])
cardRatings['oppo_bat_wkts_card'] = cardRatings['wkt_rating_pr_ba'].fillna(cardRatings['wkt_rating_pr_ba_earlier'])
his = his.merge(cardRatings.loc[:, ['matchID', 'batterid', 'oppo_bat_runs_card', 'oppo_bat_wkts_card']], on=['matchID', 'batterid'], how='left')
his['oppo_bat_runs'] = his['oppo_bat_runs'].fillna(his['oppo_bat_runs_card'])
his['oppo_bat_wkts'] = his['oppo_bat_wkts'].fillna(his['oppo_bat_wkts_card'])
his['oppo_bat_runs'] = his['oppo_bat_runs'].fillna(1) #just setting oppo_bat_runs to br_tc for now so they are same as above, trying to work out where difference comes from
his['oppo_bat_wkts'] = his['oppo_bat_wkts'].fillna(1)

# ---- survivor order substitution (men/expBall&runsToCome/auxiliaries/byOrderAdjusts.py) ----
# For the two batters actually at the crease (in_now==1): the recent arrival's own order is always
# exactly totalInningWickets+2 by construction (each wicket brings in exactly the next man), needing no
# substitution. The survivor's own real order varies unpredictably depending on who's lasted -
# substitute the AVERAGE survivor order for this totalInningWickets level instead of their own real order when
# recomputing the order-adjustment ratio. Recomputed from oppo_bat_runs_preadj (the clean,
# pre-any-order-adjustment baseline) rather than stacked on top of the existing ratio, since this
# fully replaces the real-own-order-based adjustment for the survivor, not adds to it.
survivorOrderByWickets = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/outputs/survivorOrderByWickets.csv')
survivorOrderByWickets = survivorOrderByWickets.rename(columns={'wickets': 'totalInningWickets'})
his = his.merge(survivorOrderByWickets, on='totalInningWickets', how='left')
his['is_survivor'] = (his['in_now'] == 1) & (his['carded'] != his['totalInningWickets'] + 2)

# avgSurvivorOrd is continuous (e.g. 2.50, 3.28) but byOrderAdjusts.csv is gridded on whole numbers
# only - interpolate between its floor/ceiling rather than rounding and losing information (the same
# linear-interpolation pattern already used below for eff_ord/not-yet-arrived batters).
his['survivor_ord_lower'] = np.clip(np.floor(his['avgSurvivorOrd']), 2, 11)
his['survivor_ord_upper'] = np.clip(np.ceil(his['avgSurvivorOrd']), 2, 11)
his['survivor_ord_upper_prop'] = his['avgSurvivorOrd'] - his['survivor_ord_lower']
his['survivor_ord_lower_prop'] = 1 - his['survivor_ord_upper_prop']

survivorOrderAdjusts_lower = orderAdjusts.rename(columns={'expected_ord': 'expected_ord_rounded', 'ord': 'survivor_ord_lower', 'runsAdjustRatio': 'survivorRunsRatio_lower', 'wktsAdjustRatio': 'survivorWktsRatio_lower'})
his = his.merge(survivorOrderAdjusts_lower.loc[:, ['expected_ord_rounded', 'survivor_ord_lower', 'survivorRunsRatio_lower', 'survivorWktsRatio_lower']], on=['expected_ord_rounded', 'survivor_ord_lower'], how='left')
survivorOrderAdjusts_upper = orderAdjusts.rename(columns={'expected_ord': 'expected_ord_rounded', 'ord': 'survivor_ord_upper', 'runsAdjustRatio': 'survivorRunsRatio_upper', 'wktsAdjustRatio': 'survivorWktsRatio_upper'})
his = his.merge(survivorOrderAdjusts_upper.loc[:, ['expected_ord_rounded', 'survivor_ord_upper', 'survivorRunsRatio_upper', 'survivorWktsRatio_upper']], on=['expected_ord_rounded', 'survivor_ord_upper'], how='left')

his['survivorRunsRatio'] = (his['survivorRunsRatio_upper'] * his['survivor_ord_upper_prop']) + (his['survivorRunsRatio_lower'] * his['survivor_ord_lower_prop'])
his['survivorWktsRatio'] = (his['survivorWktsRatio_upper'] * his['survivor_ord_upper_prop']) + (his['survivorWktsRatio_lower'] * his['survivor_ord_lower_prop'])
his['survivorRunsRatio'] = his['survivorRunsRatio'].fillna(1.0)
his['survivorWktsRatio'] = his['survivorWktsRatio'].fillna(1.0)

his['oppo_bat_runs'] = np.where(his['is_survivor'], his['oppo_bat_runs_preadj'] * his['survivorRunsRatio'], his['oppo_bat_runs'])
his['oppo_bat_wkts'] = np.where(his['is_survivor'], his['oppo_bat_wkts_preadj'] * his['survivorWktsRatio'], his['oppo_bat_wkts'])

# ---- balls-faced adjustment (men/expBall&runsToCome/auxiliaries/fxbAdjust.py) ----
# Applied only to the two batters ACTUALLY at the crease for this ball (in_now) - not to the other
# ~9 potential-lineup rows 'his' carries per ball, since we only have a real, live balls-faced
# count for whoever is actually out there. carded is this row's potential batter's real lineup
# position (their 'ord'). R19 is that specific batter's own balls faced before their own next
# delivery, capped at the lookup grid's top R19 row.
#
# The striker and non-striker need different treatment here, because 'batsmanBallsFaced' is inclusive of
# the CURRENT row's own ball (confirmed against real data - a batter's very first row already shows
# batsmanBallsFaced=1, or =0 if that first ball is a wide, which doesn't increment it) - so for the
# striker we subtract 1 to land on "balls faced before this delivery" (floored at 0, which also
# correctly handles the wide-first-ball case: clip(0-1,0,10)=0).
#
# nonstriker_batsmanballs is different: it's copied via merge_asof from the non-striker's OWN last
# row on strike, which is already a completed, inclusive-of-that-past-ball count - nothing has
# changed for them since (they aren't facing THIS row's delivery), so that value already equals
# their true "balls faced so far" with no further subtraction needed. Applying the same -1 here
# double-subtracts and undercounts every non-striker who has faced at least one ball by exactly 1
# (only looked correct for a brand-new batter, where clip's floor rescued the 0-ball case too).
# FXB-only table (no SLW): SLW in RA_Sum is on hold until rechecked (user). fxbBallsFacedAdjust.csv is now the SLW version
# (extra BSW0 key, up to 16 rows per state) - joining it here on these keys copied every in-batter row up to 16x (found
# 15 Oct, armadacricket-d0). _fxbOnly is the FXB-only table on the final curves (R19 0-40), as RA_Sum used before.
fxbAdjusts = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/auxiliaries/fxbBallsFacedAdjust_fxbOnly.csv')
# R19 is capped at the lookup's own top balls-faced row (10 originally, 21 / 40 as fxbAdjust widens it with the FXB curves)
r19Cap = int(fxbAdjusts['R19'].max())
his['R19'] = np.where(
    his['batsmanName'] == his['batter'], np.clip(his['batsmanBallsFaced'] - 1, 0, r19Cap),
    np.where(his['nonstrikerName'] == his['batter'], np.clip(his['nonstriker_batsmanballs'], 0, r19Cap), np.nan)
)
fxbAdjusts = fxbAdjusts.rename(columns={'ord': 'carded'}).loc[:, ['carded', 'totalInningWickets', 'inningBallsRemaining', 'R19', 'inningNumber', 'fxbRunsAdjustRatio', 'fxbWktsAdjustRatio']]
_nHis = len(his)
his = his.merge(fxbAdjusts, on=['carded', 'totalInningWickets', 'inningBallsRemaining', 'R19', 'inningNumber'], how='left')
assert len(his) == _nHis, 'fxbBallsFacedAdjust join duplicated rows - wrong table (SLW version has a BSW0 key)'
# not-in_now rows (R19 is NaN, so nothing can match above) and any other missing combination
# default to 1.0 - no adjustment - rather than being silently dropped
# By design (2026-10-15): batters not at the crease get no FXB/SLW here - their rows keep 1.0. SLW in RA_Sum
# (fxbBallsFacedAdjust's SLW part) is on hold until pycharmprojects-1f reports on how fxbAdjust projects FXB + SLW
# over the rest of the innings.
his['fxbRunsAdjustRatio'] = his['fxbRunsAdjustRatio'].fillna(1.0)
his['fxbWktsAdjustRatio'] = his['fxbWktsAdjustRatio'].fillna(1.0)
# (removed 2026-10-14: fxb used to be forced to 1 at 0 wickets down. The final FXB curves treat both openers as newest
# batters, and fxbBallsFacedAdjust has real values there - men mean 1.019, 0.72-1.20)
his['oppo_bat_runs'] = his['oppo_bat_runs'] * his['fxbRunsAdjustRatio']
his['oppo_bat_wkts'] = his['oppo_bat_wkts'] * his['fxbWktsAdjustRatio']

# batter form on top of the fxb ratio, runs only (built above; its baseline already has rating x FXB x SLW in it)
his['formRunsToComeAdjust'] = np.where(his['batsmanName'] == his['batter'], his['form_striker'],
                                       np.where(his['nonstrikerName'] == his['batter'], his['form_nonstriker'], 1.0))
his['oppo_bat_runs'] = his['oppo_bat_runs'] * his['formRunsToComeAdjust']

his['extra_ord'] = ((1 / his['oppo_bat_wkts']) - 1) * his['out']
# sorted by (inningNumber, inningBallNumber), not 'ID' - this rolling window is grouped only by matchID (not
# inningNumber), so it also depends on inningNumber 1 sorting entirely before inningNumber 2, which ID cannot be
# trusted to do (see the note above the raw_data sort_values near the top of the script)
his.sort_values(by=['matchID', 'inningNumber', 'inningBallNumber', 'carded'], ascending=[True, True, True, True], inplace=True)
ord_total = his.groupby(['matchID'], sort=False)['extra_ord'].transform(lambda x: x.rolling(150, min_periods=1, closed='left').mean()).reset_index().fillna(0)
his['extra_ord_tot'] = ord_total['extra_ord']
his['carded'] = np.maximum(2, his['carded'])
his['eff_ord'] = np.minimum(11, np.maximum(2, np.where(his['totalInningWickets'] == his['carded'] - 2, his['carded'], his['carded'] + his['extra_ord_tot'])))
his['ord_upper'] = np.maximum(2, np.minimum(11, np.ceil(his['eff_ord'])))
his['ord_lower'] = np.maximum(2, np.minimum(11, np.floor(his['eff_ord'])))
his['ord_upper_prop'] = his['eff_ord'] - his['ord_lower']
his['ord_lower_prop'] = 1 - his['ord_upper_prop']
expr_pred.rename(columns={'ord': 'ord_upper', 'er_tc_smooth': 'er_tc_smooth_upper', 'ew_tc_smooth': 'ew_tc_smooth_upper'}, inplace=True)
his = his.merge(expr_pred.loc[:, ['ord_upper', 'inningBallsRemaining', 'totalInningWickets', 'er_tc_smooth_upper', 'ew_tc_smooth_upper']], on=('ord_upper', 'inningBallsRemaining', 'totalInningWickets'), how='left')
expr_pred.rename(columns={'ord_upper': 'ord_lower', 'er_tc_smooth_upper': 'er_tc_smooth_lower', 'ew_tc_smooth_upper': 'ew_tc_smooth_lower'}, inplace=True)
his = his.merge(expr_pred.loc[:, ['ord_lower', 'inningBallsRemaining', 'totalInningWickets', 'er_tc_smooth_lower', 'ew_tc_smooth_lower']], on=('ord_lower', 'inningBallsRemaining', 'totalInningWickets'), how='left')
his['er_tc_smooth'] = (his['er_tc_smooth_upper'] * his['ord_upper_prop']) + (his['er_tc_smooth_lower'] * his['ord_lower_prop']) #this is exp runs for the batter, given their effective batting position, we only use this for people who aren't in yet
his['ew_tc_smooth'] = (his['ew_tc_smooth_upper'] * his['ord_upper_prop']) + (his['ew_tc_smooth_lower'] * his['ord_lower_prop']) #this is exp runs for the batter, given their effective batting position, we only use this for people who aren't in yet
expr_pred.rename(columns={'ord_lower': 'ord', 'er_tc_smooth_lower': 'er_tc_smooth', 'ew_tc_smooth_lower': 'ew_tc_smooth'}, inplace=True) #returing this to how it was before in case I want to use it again
his['er_tc_smooth'] = np.where(his['batsmanName'] == his['batter'], his['er_tc_smooth_basic'], his['er_tc_smooth']) ###we do this to use the er_tc which is derived from just in now data, which will be more accurate for in batters than the model which looks at batters who aren't in yet as well.
his['ew_tc_smooth'] = np.where(his['batsmanName'] == his['batter'], his['ew_tc_smooth_basic'], his['ew_tc_smooth']) ###we do this to use the er_tc which is derived from just in now data, which will be more accurate for in batters than the model which looks at batters who aren't in yet as well.

# apply 'out' here - previously only used for extra_ord/eff_ord above, never to stop a batter's
# own expected FUTURE contribution from accumulating for the rest of the inningNumber once they're out
# batter splits (runs only): sum over phases of share x runs_mult[batter, phase] x the remaining balls' type mix x runs_factor
his['batSplitRuns'] = rss.batter_runs_multiplier(his, raw_data, splitIds, splitCnt, splitTot, splitShareNow, splitShareCard, 'men')
his['oppo_bat_runs'] = his['oppo_bat_runs'] * his['batSplitRuns']
his['rar_bat'] = (his['oppo_bat_runs'] - 1) * his['er_tc_smooth'] * his['out']
his['raw_bat'] = (1 - his['oppo_bat_wkts']) * his['ew_tc_smooth'] * his['out']
his = his.sort_values(by='year', ascending=False)

new_data = his.copy()
new_data = pd.pivot_table(new_data, values=['rar_bat', 'raw_bat',  'rar_ground_sum', 'raw_ground_sum', 'rar_ground_sum_nonDay', 'raw_ground_sum_nonDay', 'rar_bowl_sum', 'raw_bowl_sum', 'rar_bbl_sum', 'raw_bbl_sum'], index=['ID'], aggfunc={'rar_bat': 'sum', 'raw_bat': 'sum', 'rar_ground_sum': 'mean', 'raw_ground_sum': 'mean', 'rar_ground_sum_nonDay': 'mean', 'raw_ground_sum_nonDay': 'mean', 'rar_bowl_sum': 'mean', 'raw_bowl_sum': 'mean', 'rar_bbl_sum': 'mean', 'raw_bbl_sum': 'mean'}).reset_index()

new_data['RA_sum'] = new_data['rar_bat'] + new_data['raw_bat'] + new_data['rar_ground_sum'] + new_data['raw_ground_sum'] + new_data['rar_bowl_sum'] + new_data['raw_bowl_sum'] + new_data['rar_bbl_sum'] + new_data['raw_bbl_sum']
new_data['RA_sum_nonDay'] = new_data['rar_bat'] + new_data['raw_bat'] + new_data['rar_ground_sum_nonDay'] + new_data['raw_ground_sum_nonDay'] + new_data['rar_bowl_sum'] + new_data['raw_bowl_sum'] + new_data['rar_bbl_sum'] + new_data['raw_bbl_sum']

raw_data_og = raw_data_og.merge(new_data.loc[:, ['ID', 'rar_bat', 'raw_bat',  'rar_ground_sum', 'raw_ground_sum', 'rar_ground_sum_nonDay', 'raw_ground_sum_nonDay', 'rar_bowl_sum', 'raw_bowl_sum', 'RA_sum', 'RA_sum_nonDay']], on='ID', how='left')

# oppo_bat_runs/oppo_bat_wkts get exported CLEAN (pre order-adjustment), not the order-adjusted
# version used internally above to compute RA_sum. byOrderAdjusts.py (and fxbAdjust.py) read this
# exported column back in as the baseline to compare real performance against, to DERIVE
# runsAdjustRatio/wktsAdjustRatio in the first place - exporting the already-adjusted version would
# make that circular (re-deriving the adjustment from data that already has it applied). RA_sum
# itself, which reflects both adjustments, is unaffected by this - it was already computed above
# from oppo_bat_runs/oppo_bat_wkts before they get reset here.
raw_data['oppo_bat_runs'] = raw_data['oppo_bat_runs_preadj']
raw_data['oppo_bat_wkts'] = raw_data['oppo_bat_wkts_preadj']

raw_data_og = raw_data_og.merge(raw_data.loc[:,['ID', 'oppo_bowl_runs', 'oppo_bowl_wkts', 'oppo_bat_runs', 'oppo_bat_wkts', 'ground_runs', 'ground_wkts', 'ord_pr_ba']], on='ID', how='left')

dataClean_w = raw_data_og.copy() # .drop(columns=['rar_bat', 'raw_bat',  'rar_ground_sum', 'raw_ground_sum', 'rar_bowl_sum', 'raw_bowl_sum']))
dataClean_w['totalInningRunsToComeAdj'] = dataClean_w['totalInningRunsToCome'] - dataClean_w['RA_sum']
dataClean_w['runsRequiredAdj'] = dataClean_w['runsRequired'] - dataClean_w['RA_sum']
dataClean_w['totalInningRunsToComeAdj_nonDay'] = dataClean_w['totalInningRunsToCome'] - dataClean_w['RA_sum_nonDay']
dataClean_w['runsRequiredAdj_nonDay'] = dataClean_w['runsRequired'] - dataClean_w['RA_sum_nonDay']
dataClean_w = dataClean_w.rename(columns={'RA_sum': 'RA_Sum', 'RA_sum_nonDay': 'RA_Sum_nonDay', 'ord_pr_ba': 'expected_ord'})

# surge-era BBL out of dataClean.csv (it's only needed up to RA_Sum) and into its own file
isBblSurge = (dataClean_w['competition'] == 'Big Bash League') & (pd.to_datetime(dataClean_w['date']) > pd.Timestamp(2020, 6, 6))
dataClean_w.loc[isBblSurge].to_csv(PROJECT_ROOT / 'men/expBall&runsToCome/Data/dataClean_bblSurge.csv', index=False)
dataClean_w = dataClean_w.loc[~isBblSurge]
assert not ((dataClean_w['competition'] == 'Big Bash League') & (pd.to_datetime(dataClean_w['date']) > pd.Timestamp(2020, 6, 6))).any(), 'surge-era BBL rows left in dataClean.csv'
# live sim results (runs to come at ball 1 and every 12 balls) kept in dataClean - prod_model/simToDataClean.py
import simToDataClean
dataClean_w = simToDataClean.add(dataClean_w, 'men')
dataClean_w.to_csv(PROJECT_ROOT / 'men/expBall&runsToCome/Data/dataClean.csv', index=False)

# runs-to-come bias grids (raw t20_inruns baseline, and RA_Sum + adjusted baseline) - prod_model/biasGrid.py
import biasGrid
biasGrid.run(dataClean_w, 'men')

# ##testing:
# raw_data_og = pd.read_csv(fr'{user_name}\OneDrive - Decimal Data Services Ltd\PythonData\Cleaned_t20bbb3_adjusted_runs_to_come_{for_match}.csv')
# raw_data_og = raw_data_og.sort_values(by=['year'])
# # x = 10
# # raw_data_og['RA_sum'] = 50 # np.where(raw_data_og['RA_sum'] > x, x, np.where(raw_data_og['RA_sum'] < -x, -x, raw_data_og['RA_sum']))
# # raw_data_og['RA_sum'] = raw_data_og['RA_sum'].clip(lower=-x, upper=x)
# raw_data_og['RA_sum'] = raw_data_og['RA_sum'] #- (raw_data_og['rar_bat'] + raw_data_og['raw_bat'] + raw_data_og['rar_bowl_sum'] + raw_data_og['raw_bowl_sum']) #raw_data_og['RA_sum']
#
# raw_data_og['required_adjusted'] = np.where(raw_data_og['inningNumber'] == 1, raw_data_og['required'], raw_data_og['required'] - raw_data_og['RA_sum']) # (raw_data_og['rar_ground_sum'] + raw_data_og['raw_ground_sum']) #raw_data_og['RA_sum']
# raw_data_og['adjust_side'] = np.where(raw_data_og['RA_sum'] > 0, 1, 0) #raw_data_og['RA_sum']
# #raw_data_og = raw_data_og.sort_values(by=['RA_sum'])
#
# raw_data_og_test = raw_data_og.copy()
# raw_data_og_test = raw_data_og_test.dropna(subset=['result', 'required_adjusted', 'required'])
# raw_data_og_test = raw_data_og_test[(raw_data_og_test.inningBallsRemaining == 120) & (raw_data_og_test.inningNumber == 2)]# & (raw_data_og_test.RA_sum < 5) & (raw_data_og_test.RA_sum > -5)]
# raw_data_og_test = raw_data_og_test.drop_duplicates(subset=['matchID'])
# raw_data_og_test["required_adjusted_bin"] = (raw_data_og_test["required_adjusted"] // 10) * 10 #np.where(raw_data_og_test["required_adjusted"] > 167, 1, 0)       #(raw_data_og_test["required_adjusted"] // 5) * 5 #round(raw_data_og_test["required_adjusted"] / 10) * 10
# raw_data_og_test["required_bin"] = (raw_data_og_test["required"] // 10) * 10 # np.where(raw_data_og_test["required"] > 167, 1, 0) #(raw_data_og_test["required"] // 5) * 5 #round(raw_data_og_test["required"] / 10) * 10
#
# raw_data_og_test0 = raw_data_og_test.groupby(['required_adjusted_bin', 'adjust_side']).agg({"matchID": "count", "result": "mean", "required": "mean", "required_adjusted": "mean"}).reset_index()
# raw_data_og_test0 = raw_data_og_test0.sort_values(by=['adjust_side', 'required_adjusted_bin'])
# raw_data_og_test02 = raw_data_og_test.groupby(['required_bin', 'adjust_side']).agg({"matchID": "count", "result": "mean", "required": "mean", "required_adjusted": "mean"}).reset_index()
# raw_data_og_test02 = raw_data_og_test02.sort_values(by=['adjust_side', 'required_bin'])
# raw_data_og_test1 = raw_data_og_test.groupby(['required_adjusted_bin']).agg({"matchID": "count", "result": "mean", "required": "mean", "required_adjusted": "mean"}).reset_index()
# raw_data_og_test2 = raw_data_og_test.groupby("required_bin").agg({"matchID": "count", "result": "mean", "required": "mean", "required_adjusted": "mean"})  # or sum, count etc.
# # raw_data_og_test3 = raw_data_og_test.groupby("inningNumber").agg({"matchID": "count", "result": "mean", "required": "mean", "required_adjusted": "mean", 'RA_sum': 'mean'})  # or sum, count etc.
# #
# raw_data_og_test00 = raw_data_og_test0[raw_data_og_test0.adjust_side == 0].loc[:,['required_adjusted_bin', 'matchID', 'result']].merge(raw_data_og_test0[raw_data_og_test0.adjust_side == 1].loc[:,['required_adjusted_bin', 'result']], how='left', on='required_adjusted_bin', suffixes=('_up', '_down'))
# raw_data_og_test002 = raw_data_og_test02[raw_data_og_test02.adjust_side == 0].loc[:,['required_bin', 'matchID', 'result']].merge(raw_data_og_test02[raw_data_og_test02.adjust_side == 1].loc[:,['required_bin', 'result']], how='left', on='required_bin', suffixes=('_up', '_down'))
# dive = raw_data_og.copy()
# dive = dive[(dive.inningBallsRemaining == 120) & (dive.inningNumber == 2)]
# dive = dive[(dive.required_adjusted >= 150) & (dive.required_adjusted < 155)]
# dive = dive.sort_values(by=['RA_sum'])
