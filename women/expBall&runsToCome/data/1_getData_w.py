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
# get ball-by-ball data
# -------------------------
sql_query = '''
select matchid, id, tier, date, year, competition, venue, host, home, away, battingteam, innings, ball, delivery2, runs, score, t_runs, wicket, wickets, t_wickets,
       ballsremaining, target, reduced, max_balls, noball, wide, ord, byes, legbyes, innperiod, bowlerwicket, realexprbat, realexpwbat, realexprbowl, realexpwbowl,
       rating_sample_size, major_nation, batsmanballs, ovrexpr, ovrexpw, batsman, nonstriker, bowler, batterid, bowlerid, av_runs_bat, av_wkts_bat, style_new, power_surge
from match_data.w_t20_bbb
where tier < 3
and date > %s
and reduced is not true
order by matchid, innings, id
'''

allData = pd.read_sql_query(sql_query, con=connection, params=(format_date,))
allData = allData.sort_values(by=['matchid', 'innings', 'delivery2']).reset_index(drop=True)


# -------------------------
# Initial cleaning
# -------------------------
allData['target'] = np.where(allData['innings'] == 1, np.nan, allData['target'])
allData['reduced'] = allData['reduced'].fillna(False)
allData = allData.dropna(subset=['battingteam'])
allData['venue'] = allData['venue'].replace({'R.Premadasa Stadium': 'R Premadasa Stadium'})


# -------------------------
# re calculate second innings targets
# -------------------------
targets = pd.pivot_table(allData[allData['innings'] == 1], values='t_runs', index='matchid', aggfunc='mean').reset_index()
targets['target_new'] = targets['t_runs'] + 1
targets['innings'] = 2
targets = targets[['matchid', 'innings', 'target_new']]

allData = allData.merge(targets, how='left', on=['matchid', 'innings'])
allData['target'] = np.where(allData['innings'] == 2, allData['target_new'], allData['target'])
allData = allData.drop(columns=['target_new'])


# -------------------------
# Create over and extras variables
# -------------------------
allData['over_number'] = np.floor(allData['delivery2'])
allData['over'] = allData['over_number'] + 1
allData['extra'] = np.where((allData['wide'] + allData['noball']) > 0, 1, 0)


# -------------------------
# separate Hundred and T20 data
# -------------------------
hundredData = allData[allData['competition'] == 'The Hundred (Women\'s Comp)'].copy()
t20Data = allData[allData['competition'] != 'The Hundred (Women\'s Comp)'].copy()



# CLEAN T20 DATA
# -------------------------
# Identify reduced T20 games that aren't already flagged on SQL as reduced
# -------------------------
pivot = pd.pivot_table(
    t20Data,
    values=['t_runs', 't_wickets', 'max_balls', 'ball', 'noball', 'wide', 'target'],
    index=['matchid', 'innings', 'reduced'],
    aggfunc={'t_runs': 'max', 't_wickets': 'max', 'max_balls': 'min', 'ball': 'count', 'noball': 'sum', 'wide': 'sum', 'target': 'max'}
).reset_index()


# -------------------------
# identify 1st innings reduced games
# -------------------------
pivot_1 = pivot[pivot['innings'] == 1].copy()
pivot_1['innings_balls'] = pivot_1['ball'] - pivot_1['wide'] - pivot_1['noball']
pivot_1['remove'] = 0

# Known reduced innings
pivot_1['remove'] = np.where((pivot_1['max_balls'] < 120) & (pivot_1['max_balls'] > 0), 1, pivot_1['remove'])

# Max balls says 120 but innings was flagged as reduced and stopped early
pivot_1['remove'] = np.where((pivot_1['max_balls'] == 120) & pivot_1['reduced'] & (pivot_1['innings_balls'] < 115), 1, pivot_1['remove'])

# max_balls = 0 is unreliable, so use actual balls bowled
pivot_1['remove'] = np.where((pivot_1['max_balls'] == 0) & (pivot_1['innings_balls'] > 117), 0, pivot_1['remove'])
pivot_1['remove'] = np.where((pivot_1['max_balls'] == 0) & (pivot_1['innings_balls'] < 118) & (pivot_1['t_wickets'] < 10), 1, pivot_1['remove'])


# -------------------------
# identify 2nd innings reduced games
# -------------------------
pivot_2 = pivot[pivot['innings'] == 2].copy()
pivot_2['innings_balls'] = pivot_2['ball'] - pivot_2['wide'] - pivot_2['noball']
pivot_2['remove'] = 0

# Known reduced innings
pivot_2['remove'] = np.where(pivot_2['reduced'], 1, pivot_2['remove'])

# Reduced maximum innings length
pivot_2['remove'] = np.where((pivot_2['max_balls'] < 120) & (pivot_2['max_balls'] > 0), 1, pivot_2['remove'])

# Short innings which did not reach the target or lose 10 wickets
pivot_2['remove'] = np.where(~pivot_2['reduced'] & (pivot_2['max_balls'] == 120) & (pivot_2['innings_balls'] < 114) & (pivot_2['t_runs'] < pivot_2['target']) & (pivot_2['t_wickets'] < 10), 1, pivot_2['remove'])
pivot_2['remove'] = np.where(~pivot_2['reduced'] & (pivot_2['max_balls'] == 0) & (pivot_2['innings_balls'] < 114) & (pivot_2['t_runs'] < pivot_2['target']) & (pivot_2['t_wickets'] < 10), 1, pivot_2['remove'])


# -------------------------
# remove reduced games
# -------------------------
pivot = pd.concat([pivot_1, pivot_2], ignore_index=True)
t20Data = t20Data.merge(pivot[['matchid', 'innings', 'remove']], how='left', on=['matchid', 'innings'])
t20Data = t20Data[t20Data['remove'] == 0].copy()
t20Data = t20Data.drop(columns=['reduced', 'remove', 'max_balls'])


# -------------------------
# fix ball number, we need to count the number of illegal deliveries in the over so far to help (rollextra)
# -------------------------
rollextra = t20Data.groupby(['matchid', 'over_number', 'innings'], sort=False)['extra'].rolling(50, min_periods=1, closed='left').sum().reset_index().fillna(0)
rollextra = rollextra.sort_values(by=['matchid', 'level_3']).reset_index(drop=True)
rollextra['extra'] = rollextra['extra'] / 100
t20Data = t20Data.reset_index(drop=True)
t20Data['extra'] = rollextra['extra']
t20Data['ball'] = t20Data['delivery2'] - t20Data['extra']
t20Data['ballsremaining'] = np.round(120 - ((np.floor(t20Data['ball']) * 6) + ((t20Data['ball'] - np.floor(t20Data['ball'])) * 100) - 1), 0)


# -------------------------
# Fix score
# -------------------------
runs_comp = pd.pivot_table(t20Data, values=['runs', 't_runs'], index=['matchid', 'innings'], aggfunc={'runs': 'sum', 't_runs': 'mean'}).reset_index()
runs_comp['comp'] = runs_comp['runs'] - runs_comp['t_runs']
true_score = t20Data.groupby(['matchid', 'innings'], sort=False)['runs'].rolling(200, min_periods=1, closed='left').sum().reset_index().fillna(0)
t20Data = t20Data.reset_index(drop=True)
t20Data['true_score'] = true_score['runs']
t20Data = t20Data.merge(runs_comp[['matchid', 'innings', 'comp']], how='left', on=['matchid', 'innings'])
t20Data['score'] = np.where(t20Data['comp'] != 0, t20Data['score'], t20Data['true_score'])





# CLEAN HUNDRED DATA
# -------------------------
# fix ball number, we need to count the number of illegal deliveries in the over so far to help (rollextra)
# -------------------------
rollextra = hundredData.groupby(['matchid', 'over_number', 'innings'], sort=False)['extra'].rolling(50, min_periods=1, closed='left').sum().reset_index().fillna(0)
rollextra = rollextra.sort_values(by=['matchid', 'level_3']).reset_index(drop=True)
rollextra['extra'] = rollextra['extra'] / 100
hundredData = hundredData.reset_index(drop=True)
hundredData['extra'] = rollextra['extra']
hundredData['ball'] = round(hundredData['delivery2'] - hundredData['extra'], 2)
hundredData['ballsremaining'] = np.round(100 - ((np.floor(hundredData['ball']) * 5) + ((hundredData['ball'] - np.floor(hundredData['ball'])) * 100) - 1), 0)


# -------------------------
# Fix score
# -------------------------
runs_comp = pd.pivot_table(hundredData, values=['runs', 't_runs'], index=['matchid', 'innings'], aggfunc={'runs': 'sum', 't_runs': 'mean'}).reset_index()
runs_comp['comp'] = runs_comp['runs'] - runs_comp['t_runs']
true_score = hundredData.groupby(['matchid', 'innings'], sort=False)['runs'].rolling(200, min_periods=1, closed='left').sum().reset_index().fillna(0)
hundredData = hundredData.reset_index(drop=True)
hundredData['true_score'] = true_score['runs']
hundredData = hundredData.merge(runs_comp[['matchid', 'innings', 'comp']], how='left', on=['matchid', 'innings'])
hundredData['score'] = np.where(hundredData['comp'] != 0, hundredData['score'], hundredData['true_score'])





# -------------------------
# combine hundred and t20 data again now they've been cleaned seperately
# -------------------------
allData = pd.concat([t20Data, hundredData], ignore_index=True)








# -------------------------
# final bits
# -------------------------
allData['wickets'] = allData['wickets'] - allData['wicket']
allData['wickets'] = np.where(allData['wickets'] == -1, 0, allData['wickets'])
allData['required'] = allData['target'] - allData['score']
allData['runs_to_come'] = allData['t_runs'] - allData['score']
allData['result'] = np.where(allData['innings'] == 1, np.nan, np.where(allData['t_runs'] >= allData['target'], 1, 0))


# -------------------------
# Final cleaning
# -------------------------
allData = allData[allData['wickets'] < 10]
allData = allData[(allData['required'] > 0) | (allData['innings'] == 1)]
allData = allData[allData['ballsremaining'] > 0]
allData = allData[allData['score'] > -1]

wkt_value_sum = pd.read_csv(PROJECT_ROOT / 'Women/expBall&runsToCome/auxiliaries/wkt_sum_mean_w.csv')
allData = allData.merge(wkt_value_sum, how='left')

allData = allData.drop_duplicates(subset=['id']).reset_index(drop=True)




# -------------------------
# Export
# -------------------------
if run_type == 1:
    allData.to_csv(PROJECT_ROOT / 'women/expBall&runsToCome/data/Cleaned_t20bbb3_w.csv', index=False)

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
# connection.close()


