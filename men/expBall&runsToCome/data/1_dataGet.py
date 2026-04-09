import sqlalchemy
import pandas as pd
import numpy as np
from urllib.parse import quote
from datetime import datetime, timedelta

engine = sqlalchemy.create_engine('postgresql://jordan:%s@77.68.112.208:5432/postgres' % quote('cricket123'))
connection = engine.connect()

# format_date_new = last_date.strftime("%m/%d/%Y")
format_date = '12/31/2014'


#param = [(format_date,)] #here we convert the format_date to a tuple, because the method of inserting variables into the SQL query requires either a tuple or a dictionary. To make a dictionary we would need to put a placeholder name in the sql query but for some reason this doesn't work with postgres Sql (according to gpt)
sql_query ='''select  matchid, id, tier, date, year, competition, venue, host, home, away, battingteam, innings, ball, delivery2, runs, score, t_runs, wicket, wickets, t_wickets,
                                 ballsremaining, target, reduced, max_balls, noball, wide, ord, byes, legbyes, innperiod, bowlerwicket, realexprbat, realexpwbat, rating_sample_size, major_nation, batsmanballs, ovrexpr, ovrexpw, batsman, nonstriker
                                 from match_data.t20_bbb tb
                                 where tier = 1
                                 and date > %s
                                 and reduced is not true
                                 order by matchid, innings, id'''

sql_data = pd.read_sql_query(sql_query, con=connection, params=(format_date,)) #param is the format_date in this case. It is subbed in for %s within the query. You could have multiple %s in the query and multiple parameters in order


# export then import again, simple way to get date format correct
sql_data.to_csv('/Users/jordan/Documents/ArmadaCricket/Development/men/data/sqldataforclean2023.csv', index=False)
raw_data = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/Development/men/data/sqldataforclean2023.csv')

raw_data = raw_data.sort_values(by=['matchid', 'innings', 'delivery2'])

uniques = raw_data['date'].apply(type).unique()
# some cleaning, set target in 1st innings to nan, set reduced game to false when nan, change duplicate venue names
raw_data['target'] = np.where(raw_data['innings'] == 1, np.nan, raw_data['target'])
raw_data['reduced'] = raw_data['reduced'].fillna(False)
raw_data = raw_data.dropna(subset=['battingteam'], axis=0)
raw_data['venue'] = np.where(raw_data['venue'] == 'R.Premadasa Stadium', 'R Premadasa Stadium', raw_data['venue']) # could add to this (chinniswamy)

# make sure targets are 1 more than 1st innings total
targets = pd.pivot_table(raw_data[raw_data['innings'] == 1], values=['t_runs'], index=['matchid'], aggfunc=['mean']).reset_index()
targets['innings'] = 2
targets.columns = ['matchid', 'target_x', 'innings']
targets['target_x'] = targets['target_x'] + 1
raw_data = raw_data.merge(targets, how='left', on=['matchid', 'innings'])
raw_data['target'] = raw_data['target_x']
raw_data = raw_data.drop(labels=['target_x'], axis=1)


# take out big bash after 2019 season because of the power surge
raw_data = raw_data[(raw_data['competition'] != 'Big Bash League') | (raw_data['date'] < '2020-06-06')]
# take out the 100
raw_data = raw_data[(raw_data['competition'] != 'The Hundred Men\'s Competition')]
# for t20i only include the major nations
raw_data = raw_data[(raw_data['competition'] != 'T20I') |
                    (raw_data.home.isin(['England', 'India', 'Afghanistan', 'Australia', 'New Zealand', 'West Indies', 'Sri Lanka', 'Bangladesh', 'South Africa', 'Pakistan']) &
                     raw_data.away.isin(['England', 'India', 'Afghanistan', 'Australia', 'New Zealand', 'West Indies', 'Sri Lanka', 'Bangladesh', 'South Africa', 'Pakistan']))]

pivot = pd.pivot_table(raw_data, values=['t_runs', 't_wickets', 'max_balls', 'ball', 'noball', 'wide', 'target'], index=['matchid', 'innings', 'reduced'],
                       aggfunc={'t_runs': 'max', 't_wickets': 'max', 'max_balls': 'min', 'ball': 'count', 'noball': 'sum', 'wide': 'sum', 'target': 'max'}).reset_index()

# 1st innings reduced games
pivot_1 = pivot.copy()
pivot_1 = pivot_1[pivot_1['innings'] == 1]
pivot_1['innings_balls'] = pivot_1['ball'] - pivot_1['wide'] - pivot_1['noball']
# remove where max balls less than 120 but greater than 0 (because 0 max balls has errors, we'll deal with it separately)
pivot_1['remove'] = np.where((pivot_1['innings'] == 1) & (pivot_1['max_balls'] < 120) & (pivot_1['max_balls'] > 0), 1, 0)
# remove where max balls = 120, reduced = true and innings_balls < 120, sometimes max balls is 120 & only 2nd innings was reduced, here we can keep the first innings but only if there are 120 balls in the data
pivot_1['remove'] = np.where((pivot_1['innings'] == 1) & (pivot_1['max_balls'] == 120) & (pivot_1['reduced'] == True) & (pivot_1['innings_balls'] < 115), 1, pivot_1['remove'])
# where max balls is 0, look at innings_balls, if its 120 then not reduced. If it's less than 118 then remove unless there are 10 wickets
pivot_1['remove'] = np.where((pivot_1['innings'] == 1) & (pivot_1['max_balls'] == 0) & (pivot_1['innings_balls'] > 117), 0, pivot_1['remove'])
pivot_1['remove'] = np.where((pivot_1['innings'] == 1) & (pivot_1['max_balls'] == 0) & (pivot_1['innings_balls'] < 118) & (pivot_1['t_wickets'] < 10), 1, pivot_1['remove'])

# 2nd innings reduced games
pivot_2 = pivot.copy()
pivot_2 = pivot_2[pivot_2['innings'] == 2]
pivot_2['innings_balls'] = pivot_2['ball'] - pivot_2['wide'] - pivot_2['noball']
# if sql says reduced mark as reduced
pivot_2['remove'] = np.where((pivot_2['innings'] == 2) & (pivot_2['reduced'] == True), 1, 0)
# remove where max balls less than 120 but greater than 0 (because 0 max balls has errors, we'll deal with it separately)
pivot_2['remove'] = np.where((pivot_2['innings'] == 2) & (pivot_2['max_balls'] < 120) & (pivot_2['max_balls'] > 0), 1, pivot_2['remove'])
# look at non-reduced games with 120 max balls, check first if 120 bowled, if not then look if target reached or bowled out, mark as reduced accordingly
pivot_2['remove'] = np.where((pivot_2['innings'] == 2) & (pivot_2['reduced'] == False) & (pivot_2['max_balls'] == 120) & (pivot_2['innings_balls'] < 114) & (pivot_2['t_runs'] < pivot_2['target']) &
                             (pivot_2['t_wickets'] < 10), 1, pivot_2['remove'])
pivot_2['remove'] = np.where((pivot_2['innings'] == 2) & (pivot_2['reduced'] == False) & (pivot_2['max_balls'] == 0) & (pivot_2['innings_balls'] < 114) & (pivot_2['t_runs'] < pivot_2['target']) &
                             (pivot_2['t_wickets'] < 10), 1, pivot_2['remove'])

# merge the reduced into the raw data then remove reduced games
pivot = pd.concat([pivot_1, pivot_2], axis=0)
raw_data = raw_data.merge(pivot.loc[: , ['matchid', 'innings', 'remove']], how='left', on=['matchid', 'innings'])
raw_data = raw_data[raw_data['remove'] == 0]
raw_data = raw_data.drop(labels=['reduced', 'remove', 'max_balls'], axis=1) # same as dropping columns

# fix the ball
raw_data['extra'] = np.where(raw_data['wide'] + raw_data['noball'] > 0, 1, 0)
raw_data['over_number'] =  raw_data['delivery2'].apply(lambda x: np.floor(x))
rollextra = pd.DataFrame(raw_data.groupby(['matchid', 'over_number', 'innings'], sort=False)['extra'].rolling(50, min_periods=1, closed='left').sum()).reset_index().fillna(0)
rollextra = rollextra.sort_values(by=['matchid', 'level_3']).reset_index(drop=True)
rollextra['extra'] = rollextra['extra'] / 100
# now assign
raw_data = raw_data.reset_index(drop=True)
raw_data['extra'] = rollextra['extra']
raw_data['ball'] = raw_data['delivery2'] - raw_data['extra']
raw_data['ballsremaining'] = np.where(raw_data['competition'] == 'The Hundred Men\'s Competition', round((100-((np.floor(raw_data['ball'])*5)+((raw_data['ball']-np.floor(raw_data['ball']))*100)-1)), 0),
                                      round((120-((np.floor(raw_data['ball'])*6)+((raw_data['ball']-np.floor(raw_data['ball']))*100)-1)), 0))


# fix the score, first find games where total score != stated score, for these games stick with stated score, for others go to rolling score
runs_comp = pd.pivot_table(raw_data, values=['runs', 't_runs'], index=['matchid', 'innings'], aggfunc={'runs': 'sum', 't_runs': 'mean'}).reset_index()
runs_comp['comp'] = runs_comp['runs'] - runs_comp['t_runs']
true_score = pd.DataFrame(raw_data.groupby(['matchid', 'innings'], sort=False)['runs'].rolling(200, min_periods=1, closed='left').sum()).reset_index().fillna(0)
raw_data = raw_data.reset_index(drop=True)
raw_data['true_score'] = true_score['runs'] # does this join them effectively?
raw_data = raw_data.merge(runs_comp.loc[:, ['matchid', 'innings', 'comp']], on=['matchid', 'innings'], how='left')
raw_data['score'] = np.where(raw_data['comp'] != 0, raw_data['score'], raw_data['true_score'])

# clean
raw_data['wickets'] = raw_data['wickets'] - raw_data['wicket']
raw_data['wickets'] = np.where(raw_data['wickets'] == -1, 0, raw_data['wickets'])
raw_data['required'] = raw_data['target'] - raw_data['score']
raw_data['runs_to_come'] = raw_data['t_runs'] - raw_data['score']
raw_data['result'] = np.where(raw_data['innings'] == 1, np.nan, np.where(raw_data['t_runs'] >= raw_data['target'], 1, 0))

# final filter, remove obvious errors, wickets greater than 9, required must be more than 0 when innings is 2, balls remaining must be >0, score must be >-1
raw_data = raw_data[raw_data['wickets'] < 10]
raw_data = raw_data[(raw_data['required'] > 0) | (raw_data['innings'] == 1)]
raw_data = raw_data[raw_data['ballsremaining'] > 0]
raw_data = raw_data[raw_data['score'] > -1]
raw_data['over'] = raw_data['over_number'] + 1


# export data
raw_data.to_csv('/Users/jordan/Documents/ArmadaCricket/Development/men/data/cleanedT20BBB.csv', index=False)

