from pathlib import Path
from datetime import timedelta, date
import pandas as pd
import numpy as np
from db import engine
from paths import PROJECT_ROOT
user_name = Path.home()
connection = engine.connect()


# -------------------------
# Update settings
# -------------------------
# 1 = complete update of all data
# 2 = daily update
run_type = 1


# -------------------------
# Get last downloaded date
# -------------------------
last_date_data = pd.read_csv(PROJECT_ROOT / 'Women/expBall&runsToCome/auxiliaries/latest_data_clean_w.csv', parse_dates=['date'])

if last_date_data['date_of_run'].max() == pd.Timestamp(date.today()):
    exit()

last_date = last_date_data['date'].max() - timedelta(days=10)

format_date_old = '12/31/2014'
format_date_new = last_date.strftime('%m/%d/%Y')

if run_type == 1:
    format_date = format_date_old
else:
    format_date = format_date_new


# -------------------------
# Get ball-by-ball data
# -------------------------
sql_query = '''
select matchid as "matchID", id as "ID", tier, date, year, competition, venue, host, home, away, battingteam as "battingTeam", innings as "inningNumber",
       ball, delivery2, runs as "totalRuns", score as "totalInningRuns", t_runs as "totalInningRunsEnd", wicket as "isWicket",
       wickets as "totalInningWickets", t_wickets as "totalInningWicketsEnd", ballsremaining as "inningBallsRemaining", target, reduced, max_balls,
       noball as "noballRuns", wide as "wideRuns", ord, byes as "byeRuns", legbyes as "legbyeRuns", innperiod as "inningPhase",
       bowlerwicket as "isWicketBowler", realexprbat, realexpwbat, realexprbowl, realexpwbowl, rating_sample_size, major_nation,
       batsmanballs as "batsmanBallsFaced", ovrexpr, ovrexpw, batsman as "batsmanName", nonstriker as "nonstrikerName", 
       batterid, bowler, bowlerid, av_runs_bat, av_wkts_bat, style_new, power_surge as "isPowerSurge"
from match_data.w_t20_bbb
where tier < 3
and date > %s
and reduced is not true
order by matchid, innings, id
'''

allData = pd.read_sql_query(sql_query, con=connection, params=(format_date,))
allData['date'] = pd.to_datetime(allData['date'], errors='raise')
allData = allData.sort_values(by=['matchID', 'inningNumber', 'delivery2']).reset_index(drop=True)


# -------------------------
# Initial cleaning
# -------------------------
allData['target'] = np.where(allData['inningNumber'] == 1, np.nan, allData['target'])
allData['reduced'] = allData['reduced'].fillna(False)
allData = allData.dropna(subset=['battingTeam'])
allData['venue'] = allData['venue'].replace({'R.Premadasa Stadium': 'R Premadasa Stadium'})


# -------------------------
# Recalculate second innings targets
# -------------------------
targets = pd.pivot_table(allData[allData['inningNumber'] == 1], values='totalInningRunsEnd', index='matchID', aggfunc='mean').reset_index()
targets['target_new'] = targets['totalInningRunsEnd'] + 1
targets['inningNumber'] = 2
targets = targets[['matchID', 'inningNumber', 'target_new']]

allData = allData.merge(targets, how='left', on=['matchID', 'inningNumber'])
allData['target'] = np.where(allData['inningNumber'] == 2, allData['target_new'], allData['target'])
allData = allData.drop(columns=['target_new'])


# -------------------------
# Create over and extras variables
# -------------------------
allData['over_number'] = np.floor(allData['delivery2'])
allData['overNumber'] = allData['over_number'] + 1
allData['extra'] = np.where((allData['wideRuns'] + allData['noballRuns']) > 0, 1, 0)


# -------------------------
# Separate Hundred and T20 data
# -------------------------
hundredData = allData[allData['competition'] == 'The Hundred (Women\'s Comp)'].copy()
t20Data = allData[allData['competition'] != 'The Hundred (Women\'s Comp)'].copy()





# -------------------------
# CLEAN T20 DATA
# -------------------------
# -------------------------
# Identify reduced T20 games that aren't already flagged on SQL as reduced
# -------------------------
pivot = pd.pivot_table(
    t20Data,
    values=['totalInningRunsEnd', 'totalInningWicketsEnd', 'max_balls', 'ball', 'noballRuns', 'wideRuns', 'target'],
    index=['matchID', 'inningNumber', 'reduced'],
    aggfunc={'totalInningRunsEnd': 'max', 'totalInningWicketsEnd': 'max', 'max_balls': 'min', 'ball': 'count', 'noballRuns': 'sum', 'wideRuns': 'sum', 'target': 'max'}
).reset_index()


# -------------------------
# Identify 1st innings reduced games
# -------------------------
pivot_1 = pivot[pivot['inningNumber'] == 1].copy()
pivot_1['innings_balls'] = pivot_1['ball'] - pivot_1['wideRuns'] - pivot_1['noballRuns']
pivot_1['remove'] = 0

# Known reduced innings
pivot_1['remove'] = np.where((pivot_1['max_balls'] < 120) & (pivot_1['max_balls'] > 0), 1, pivot_1['remove'])

# Max balls says 120 but innings was flagged as reduced and stopped early
pivot_1['remove'] = np.where((pivot_1['max_balls'] == 120) & pivot_1['reduced'] & (pivot_1['innings_balls'] < 115), 1, pivot_1['remove'])

# max_balls = 0 is unreliable, so use actual balls bowled
pivot_1['remove'] = np.where((pivot_1['max_balls'] == 0) & (pivot_1['innings_balls'] > 117), 0, pivot_1['remove'])
pivot_1['remove'] = np.where((pivot_1['max_balls'] == 0) & (pivot_1['innings_balls'] < 118) & (pivot_1['totalInningWicketsEnd'] < 10), 1, pivot_1['remove'])


# -------------------------
# Identify 2nd innings reduced games
# -------------------------
pivot_2 = pivot[pivot['inningNumber'] == 2].copy()
pivot_2['innings_balls'] = pivot_2['ball'] - pivot_2['wideRuns'] - pivot_2['noballRuns']
pivot_2['remove'] = 0

# Known reduced innings
pivot_2['remove'] = np.where(pivot_2['reduced'], 1, pivot_2['remove'])

# Reduced maximum innings length
pivot_2['remove'] = np.where((pivot_2['max_balls'] < 120) & (pivot_2['max_balls'] > 0), 1, pivot_2['remove'])

# Short innings which did not reach the target or lose 10 wickets
pivot_2['remove'] = np.where(~pivot_2['reduced'] & (pivot_2['max_balls'] == 120) & (pivot_2['innings_balls'] < 114) & (pivot_2['totalInningRunsEnd'] < pivot_2['target']) & (pivot_2['totalInningWicketsEnd'] < 10), 1, pivot_2['remove'])
pivot_2['remove'] = np.where(~pivot_2['reduced'] & (pivot_2['max_balls'] == 0) & (pivot_2['innings_balls'] < 114) & (pivot_2['totalInningRunsEnd'] < pivot_2['target']) & (pivot_2['totalInningWicketsEnd'] < 10), 1, pivot_2['remove'])


# -------------------------
# Remove reduced games
# -------------------------
pivot = pd.concat([pivot_1, pivot_2], ignore_index=True)
t20Data = t20Data.merge(pivot[['matchID', 'inningNumber', 'remove']], how='left', on=['matchID', 'inningNumber'])
t20Data = t20Data[t20Data['remove'] == 0].copy()
t20Data = t20Data.drop(columns=['reduced', 'remove', 'max_balls'])

# -------------------------
# Fix ball number
# -------------------------
t20Data['extra_before'] = t20Data.groupby(['matchID', 'over_number', 'inningNumber'], sort=False)['extra'].cumsum() - t20Data['extra']
t20Data['ball'] = t20Data['delivery2'] - (t20Data['extra_before'] / 100)
t20Data['inningBallsRemaining'] = np.round(120 - ((np.floor(t20Data['ball']) * 6) + ((t20Data['ball'] - np.floor(t20Data['ball'])) * 100) - 1), 0)

# -------------------------
# Fix score
# -------------------------
runs_comp = pd.pivot_table(t20Data, values=['totalRuns', 'totalInningRunsEnd'], index=['matchID', 'inningNumber'], aggfunc={'totalRuns': 'sum', 'totalInningRunsEnd': 'mean'}).reset_index()
runs_comp['comp'] = runs_comp['totalRuns'] - runs_comp['totalInningRunsEnd']
t20Data['true_score'] = t20Data.groupby(['matchID', 'inningNumber'], sort=False)['totalRuns'].cumsum() - t20Data['totalRuns']
t20Data = t20Data.merge(runs_comp[['matchID', 'inningNumber', 'comp']], how='left', on=['matchID', 'inningNumber'])
t20Data['totalInningRuns'] = np.where(t20Data['comp'] != 0, t20Data['totalInningRuns'], t20Data['true_score'])
t20Data['inningBallNumber'] = 121 - t20Data['inningBallsRemaining']
t20Data['isPowerplay'] = np.where(t20Data['inningBallNumber'] <= 36, 1, 0)






# -------------------------
# CLEAN HUNDRED DATA
# -------------------------
# -------------------------
# Fix ball number
# -------------------------
hundredData['extra_before'] = hundredData.groupby(['matchID', 'over_number', 'inningNumber'], sort=False)['extra'].cumsum() - hundredData['extra']
hundredData['ball'] = round(hundredData['delivery2'] - (hundredData['extra_before'] / 100), 2)
hundredData['inningBallsRemaining'] = np.round(100 - ((np.floor(hundredData['ball']) * 5) + ((hundredData['ball'] - np.floor(hundredData['ball'])) * 100) - 1), 0)

# -------------------------
# Fix score
# -------------------------
runs_comp = pd.pivot_table(hundredData, values=['totalRuns', 'totalInningRunsEnd'], index=['matchID', 'inningNumber'], aggfunc={'totalRuns': 'sum', 'totalInningRunsEnd': 'mean'}).reset_index()
runs_comp['comp'] = runs_comp['totalRuns'] - runs_comp['totalInningRunsEnd']
hundredData['true_score'] = hundredData.groupby(['matchID', 'inningNumber'], sort=False)['totalRuns'].cumsum() - hundredData['totalRuns']
hundredData = hundredData.merge(runs_comp[['matchID', 'inningNumber', 'comp']], how='left', on=['matchID', 'inningNumber'])
hundredData['totalInningRuns'] = np.where(hundredData['comp'] != 0, hundredData['totalInningRuns'], hundredData['true_score'])
hundredData['inningBallNumber'] = 101 - hundredData['inningBallsRemaining']
hundredData['isPowerplay'] = np.where(hundredData['inningBallNumber'] <= 75, 1, 0)




# -------------------------
# Combine Hundred and T20 data again
# -------------------------
allData = pd.concat([t20Data, hundredData], ignore_index=True)



# -------------------------
# adding columns
# -------------------------
allData['totalInningWickets'] = allData['totalInningWickets'] - allData['isWicket']
allData['totalInningWickets'] = np.where(allData['totalInningWickets'] == -1, 0, allData['totalInningWickets'])
allData['totalInningRunsToCome'] = allData['totalInningRunsEnd'] - allData['totalInningRuns']
allData['result'] = np.where(allData['inningNumber'] == 1, np.nan, np.where(allData['totalInningRunsEnd'] >= allData['target'], 1, 0))
allData['year'] = allData['date'].dt.year
allData['daysGroup'] = (allData['date'] - allData['date'].min()).dt.days / 365
allData['overBallNumber'] = (allData['ball'] + 1 - allData['overNumber']) * 100
allData['isValid'] = np.where((allData['wideRuns'] > 0) | (allData['noballRuns'] > 0), 0, 1)
allData['isWide'] = np.where(allData['wideRuns'] > 0, 1, 0)
allData['isNoball'] = np.where(allData['noballRuns'] > 0, 1, 0)
allData['sample'] = 1
allData['totalInningWicketsToCome'] = allData['totalInningWicketsEnd'] - allData['totalInningWickets']
allData['batsmanRuns'] = allData['totalRuns'] - allData['noballRuns'] - allData['wideRuns'] - allData['byeRuns']
allData['isWicketRunOut'] = np.where(allData['isWicket'] > allData['isWicketBowler'], 1, 0)
allData['chaseWin'] = np.where(allData['totalInningRunsEnd'] >= allData['target'], 1, 0)
allData['runsRequired'] = allData['target'] - allData['totalInningRuns']


# -------------------------
# removing some nonsense rows
# -------------------------
allData = allData[allData['totalInningWickets'] < 10]
allData = allData[(allData['runsRequired'] > 0) | (allData['inningNumber'] == 1)]
allData = allData[allData['inningBallsRemaining'] > 0]
allData = allData[allData['totalInningRuns'] > -1]


# -------------------------
# import merge and export wicket stuff
# -------------------------
wkt_value_sum = pd.read_csv(PROJECT_ROOT / 'Women/expBall&runsToCome/auxiliaries/wkt_sum_mean_w.csv')
wkt_value_sum = wkt_value_sum.rename(columns={'matchid': 'matchID', 'innings': 'inningNumber', 'wickets': 'totalInningWickets', 'ballsremaining': 'inningBallsRemaining'})
allData = allData.merge(wkt_value_sum, how='left')
allData = allData.drop_duplicates(subset=['ID']).reset_index(drop=True)



# -------------------------
# Order columns
# -------------------------
allData = allData.loc[:, [
    'matchID',
    'ID',
    'tier',
    'date',
    'year',
    'competition',
    'venue',
    'host',
    'home',
    'away',
    'battingTeam',
    'inningNumber',
    'totalRuns',
    'totalInningRuns',
    'totalInningRunsEnd',
    'isWicket',
    'totalInningWickets',
    'totalInningWicketsEnd',
    'inningBallsRemaining',
    'target',
    'noballRuns',
    'wideRuns',
    'ord',
    'byeRuns',
    'legbyeRuns',
    'inningPhase',
    'isWicketBowler',
    'realexprbat',
    'realexpwbat',
    'realexprbowl',
    'realexpwbowl',
    'rating_sample_size',
    'major_nation',
    'batsmanBallsFaced',
    'ovrexpr',
    'ovrexpw',
    'batsmanName',
    'bowler',
    'batterid',
    'bowlerid',
    'nonstrikerName',
    'comp',
    'totalInningRunsToCome',
    'result',
    'overNumber',
    'daysGroup',
    'overBallNumber',
    'inningBallNumber',
    'isPowerplay',
    'isPowerSurge',
    'isValid',
    'isWide',
    'isNoball',
    'sample',
    'totalInningWicketsToCome',
    'batsmanRuns',
    'isWicketRunOut',
    'chaseWin',
    'runsRequired'
]]



allDataOld = allData.copy().rename(
    columns={
        'matchID': 'matchid',
        'ID': 'id',
        'battingTeam': 'battingteam',
        'legbyeRuns': 'legbyes',
        'inningPhase': 'innperiod',
        'batsmanBallsFaced': 'batsmanballs',
        'batsmanName': 'batsman',
        'nonstrikerName': 'nonstriker',
        'RA_Sum': 'RA_sum',
    }
)

# -------------------------
# Export
# -------------------------
if run_type == 1:
    allData.to_csv(PROJECT_ROOT / 'women/expBall&runsToCome/data/Cleaned_t20bbb3_w.csv', index=False)
    # allDataOld.to_csv(PROJECT_ROOT / 'women/expBall&runsToCome/data/Cleaned_t20bbb3_w.csv', index=False)


#     # allData = pd.read_csv(fr'{user_name}\OneDrive - Decimal Data Services Ltd\PythonData\Cleaned_t20bbb3_w.csv')
#     sqlupload = allData.loc[:, ['id', 'ball', 'score', 'ballsremaining', 'wickets', 'target', 'ord', 'required']]
#     sqlupload.columns = ['id_clean_a', 'ball2_clean_a', 'score_clean_a', 'ballsremaining_clean_a', 'wickets_clean_a', 'target_clean_a', 'ord_clean_a', 'required_clean_a']
#
#     with connection.begin():
#         connection.execute(text("TRUNCATE TABLE player_ratings.w_t20_bbb_clean"))
#         sqlupload.to_sql(
#             "w_t20_bbb_clean",
#             con=connection,
#             schema="player_ratings",
#             if_exists='append',
#             index=False
#         )
#
#         connection.execute(text("""
#             UPDATE match_data.w_t20_bbb a
#             SET ballsremaining         = COALESCE(t.ballsremaining_clean_a, a.ballsremaining),
#                 score                  = COALESCE(t.score_clean_a, a.score),
#                 target                 = COALESCE(t.target_clean_a, a.target),
#                 ord                    = COALESCE(t.ord_clean_a, a.ord),
#                 id_clean_a             = t.id_clean_a,
#                 ball2_clean_a          = t.ball2_clean_a,
#                 score_clean_a          = t.score_clean_a,
#                 ballsremaining_clean_a = t.ballsremaining_clean_a,
#                 wickets_clean_a        = t.wickets_clean_a,
#                 target_clean_a         = t.target_clean_a,
#                 ord_clean_a            = t.ord_clean_a,
#                 required_clean_a       = t.required_clean_a
#             FROM player_ratings.w_t20_bbb_clean t
#             WHERE a.id = t.id_clean_a
#                 AND a.id_clean_a IS NULL  -- skip already-updated rows
#         """))
#
# else:
#     sqlupload = allData.loc[:, ['id', 'ball', 'score', 'ballsremaining', 'wickets', 'target', 'ord', 'required']]
#     sqlupload.columns = ['id_clean_a', 'ball2_clean_a', 'score_clean_a', 'ballsremaining_clean_a', 'wickets_clean_a', 'target_clean_a', 'ord_clean_a', 'required_clean_a']
#
#     with connection.begin():
#         sqlupload.to_sql(
#             "w_t20_bbb_clean_temp",
#             con=connection,
#             schema="player_ratings",
#             if_exists='replace',
#             index=False
#         )
#
#         connection.execute(text("""
#             INSERT INTO player_ratings.w_t20_bbb_clean (id_clean_a, ball2_clean_a, score_clean_a, ballsremaining_clean_a, wickets_clean_a, target_clean_a, ord_clean_a, required_clean_a)
#             SELECT *
#             FROM player_ratings.w_t20_bbb_clean_temp t
#             WHERE NOT EXISTS (
#                 SELECT 1 FROM player_ratings.w_t20_bbb_clean c
#                 WHERE c.id_clean_a = t.id_clean_a
#             )
#         """))
#
#         connection.execute(text("""
#             UPDATE match_data.w_t20_bbb a
#             SET ballsremaining         = COALESCE(t.ballsremaining_clean_a, a.ballsremaining),
#                 score                  = COALESCE(t.score_clean_a, a.score),
#                 target                 = COALESCE(t.target_clean_a, a.target),
#                 ord                    = COALESCE(t.ord_clean_a, a.ord),
#                 id_clean_a             = t.id_clean_a,
#                 ball2_clean_a          = t.ball2_clean_a,
#                 score_clean_a          = t.score_clean_a,
#                 ballsremaining_clean_a = t.ballsremaining_clean_a,
#                 wickets_clean_a        = t.wickets_clean_a,
#                 target_clean_a         = t.target_clean_a,
#                 ord_clean_a            = t.ord_clean_a,
#                 required_clean_a       = t.required_clean_a
#             FROM player_ratings.w_t20_bbb_clean_temp t
#             WHERE a.id = t.id_clean_a
#               AND a.id_clean_a IS NULL
#         """))
#
#         connection.execute(text("DROP TABLE player_ratings.w_t20_bbb_clean_temp"))
#
#     allData = allData.sort_values(by='date', ascending=False)
#     date = allData.head(1)
#     date['date_of_run'] = pd.Timestamp(date.today())
#     date = date.loc[:,['date', 'date_of_run']]
#     date.to_csv(PROJECT_ROOT / 'Women/expBall&runsToCome/auxiliaries/latest_data_clean_w.csv', index=False)
#
#     subprocess.run(['git', 'add', str(PROJECT_ROOT / 'Women/expBall&runsToCome/auxiliaries/latest_data_clean_w.csv')])
#     subprocess.run(['git', 'commit', '-m', 'update csv files'])
#     subprocess.run(['git', 'push'])
#
connection.close()





