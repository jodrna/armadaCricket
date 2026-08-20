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

########need to run lineups_filler before running this!!!!!!!!!!!

connection = engine.connect()
raw_data_og = pd.read_csv(PROJECT_ROOT / 'men/expBall&runsToCome/data/Cleaned_t20bbb3.csv', parse_dates=['date'])
# raw_data_og_max_date = raw_data_og['date'].max()
raw_data_og = raw_data_og.drop_duplicates(subset=['ID'])

# # raw_data_og = raw_data_og[raw_data_og.matchID == 1233979]
bowler_data = pd.read_sql_query("""select bowler, playerid as bowlerid, competition, host, date, run_rating as run_rating_pr_bo, wkt_rating as wkt_rating_pr_bo from player_ratings.bowler_ratings_historic""", con=connection)
batter_data = pd.read_sql_query("""select batter as "batsmanName", playerid as batterid, competition, host, date, run_rating as run_rating_pr_ba, wkt_rating as wkt_rating_pr_ba, balls_faced as balls_faced_pr_ba from player_ratings.batter_ratings_historic""", con=connection)
ground_data = pd.read_sql_query("""select matchid as "matchID", venue, innperiod as "inningPhase", reverted_runs, reverted_wkts, runsratio_ground, wktsratio_ground from player_ratings.ground_table""", con=connection)
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


ground_data['reverted_runs'] = ground_data['reverted_runs']
ground_data['reverted_wkts'] = ground_data['reverted_wkts']


raw_data = raw_data_og.copy()
raw_data['date'] = raw_data['date'].dt.date
# raw_data = raw_data[raw_data.inningNumber == 2]
raw_data['ord'] = np.maximum(2, raw_data['ord'])
raw_data['count1'] = 1
raw_data.sort_values(by=['ID'], ascending=[True], inplace=True)
raw_data = raw_data.reset_index(drop=True)
raw_data = raw_data.loc[:, ['matchID', 'venue', 'inningNumber', 'date', 'host', 'competition', 'ID', 'nonstrikerName', 'battingTeam', 'overNumber', 'inningBallsRemaining', 'totalInningWickets', 'ord', 'batsmanName', 'bowler', 'batterid', 'bowlerid', 'inningPhase', 'realexprbat', 'realexpwbat', 'realexprbowl', 'realexpwbowl', 'ovrexpr', 'ovrexpw']]
# his_max_date0 = raw_data['date'].max()
##get recovery values as back up for the historic player ratings, in case historic values aren't available for specifc games
recoveries_bat = batter_data.groupby(['batterid'])[['run_rating_pr_ba', 'wkt_rating_pr_ba']].mean().reset_index()
recoveries_bat.rename(columns={'run_rating_pr_ba': 'oppo_bat_runs_old', 'wkt_rating_pr_ba': 'oppo_bat_wkts_old'}, inplace=True)
recoveries_bowl = bowler_data.groupby(['bowlerid'])[['run_rating_pr_bo', 'wkt_rating_pr_bo']].mean().reset_index()
recoveries_bowl.rename(columns={'run_rating_pr_bo': 'oppo_bowl_runs_old', 'wkt_rating_pr_bo': 'oppo_bowl_wkts_old'}, inplace=True)
recoveries_ground = ground_data.groupby(['venue', 'inningPhase'])[['runsratio_ground', 'wktsratio_ground']].mean().reset_index()
recoveries_ground.rename(columns={'runsratio_ground': 'ground_runs', 'wktsratio_ground': 'ground_wkts'}, inplace=True)

raw_data = raw_data.merge(recoveries_bat, on='batterid', how='left')
raw_data = raw_data.merge(recoveries_bowl, on='bowlerid', how='left')
raw_data = raw_data.merge(recoveries_ground, on=('venue', 'inningPhase'), how='left')

# his_max_date1 = raw_data['date'].max()
##merge in the bowler batter and ground ratings
bowler_data = bowler_data.drop_duplicates(subset=['date', 'bowlerid', 'competition', 'host'])
raw_data = raw_data.merge(bowler_data, on=('date', 'bowlerid', 'competition', 'host'), how='left')
raw_data['oppo_bowl_runs'] = raw_data['run_rating_pr_bo'].fillna(raw_data['oppo_bowl_runs_old'])
raw_data['oppo_bowl_wkts'] = raw_data['wkt_rating_pr_bo'].fillna(raw_data['oppo_bowl_wkts_old'])
# his_max_date2 = raw_data['date'].max()
#
batter_data = batter_data.drop_duplicates(subset=['batterid', 'date', 'competition', 'host'])
raw_data = raw_data.merge(batter_data.loc[:, ['batterid', 'date', 'competition', 'host', 'run_rating_pr_ba', 'wkt_rating_pr_ba', 'balls_faced_pr_ba']], on=('date', 'batterid', 'competition', 'host'), how='left')
raw_data['run_rating_pr_ba'] = raw_data['run_rating_pr_ba'].fillna(raw_data['oppo_bat_runs_old'])
raw_data['wkt_rating_pr_ba'] = raw_data['wkt_rating_pr_ba'].fillna(raw_data['oppo_bat_wkts_old'])
raw_data['oppo_bat_runs'] = np.where(raw_data['balls_faced_pr_ba'] >= 150, raw_data['run_rating_pr_ba'], raw_data['oppo_bat_runs_old'])
raw_data['oppo_bat_wkts'] = np.where(raw_data['balls_faced_pr_ba'] >= 150, raw_data['wkt_rating_pr_ba'], raw_data['oppo_bat_wkts_old'])


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

#merge in wkt_values and make the wkt_value * exp_wkts values so we have a value in runs for exp totalInningWickets
wktvalues = wktvalues.drop_duplicates(subset=['overNumber', 'totalInningWickets'])
raw_data = raw_data.merge(wktvalues, on=('overNumber', 'totalInningWickets'), how='left')
raw_data['bat_exp_wktvalue'] = raw_data['wktvalue'] * raw_data['realexpwbat']
raw_data['bowl_exp_wktvalue'] = raw_data['wktvalue'] * raw_data['realexpwbowl']
raw_data['ground_exp_wktvalue'] = raw_data['wktvalue'] * raw_data['ovrexpw']

###now we can work out the ra_bowl already for each ball, we'll sum them later
raw_data['rar_bowl'] = (raw_data['oppo_bowl_runs'] - 1) * raw_data['realexprbowl']
raw_data['raw_bowl'] = (1 - raw_data['oppo_bowl_wkts']) * raw_data['bowl_exp_wktvalue']

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

# raw_data.to_csv(fr'{user_name}\Documents\Tempdata\raw_data_mmrra.csv', index=False)
# raw_data = pd.read_csv(fr'{user_name}\Documents\Tempdata\raw_data_mmrra.csv')
#
#
#
# # ##################need from here to mkae his ########################################################
# raw_data = pd.read_csv(fr'{user_name}\Documents\Tempdata\raw_data_mmrra.csv')

###EXPECTED RUNS TO COME SMOOTHED, just for current batters
er_tc_avg_now = pd.pivot_table(raw_data, values=['er_tc', 'ew_tc'], index=['inningBallsRemaining', 'totalInningWickets', 'ord'], aggfunc={'er_tc': ['mean', 'count'], 'ew_tc': ['count', 'mean']}).reset_index()
er_tc_avg_now.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in er_tc_avg_now.columns.to_flat_index()]
er_tc_avg_now_ground = pd.pivot_table(raw_data, values=['er_tc_ground', 'ew_tc_ground'], index=['inningBallsRemaining', 'totalInningWickets', 'inningPhase'], aggfunc={'er_tc_ground': ['mean', 'count'], 'ew_tc_ground': ['count', 'mean']}).reset_index()
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
raw_data_inns1 = raw_data.copy()
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

test = raw_data_inns1.loc[:, ['matchID', 'inningNumber', 'totalInningWickets', 'ord', 'er_tc', 'ew_tc', 'inningBallsRemaining']]
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
        test4 = test4.groupby(['ord'])[['er_tc', 'ew_tc']].mean().reset_index()
        test4['inningBallsRemaining'] = BR
        test4['totalInningWickets'] = WL

        expr_pred = pd.concat([expr_pred, test4], ignore_index=True)
#
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

raw_data['rar_ground_sum'] = ((raw_data['ground_runs_1'] - 1) * raw_data['er_tc_1']) + ((raw_data['ground_runs_2'] - 1) * raw_data['er_tc_2']) + ((raw_data['ground_runs_3'] - 1) * raw_data['er_tc_3'])
raw_data['raw_ground_sum'] = -(((raw_data['ground_wkts_1'] - 1) * raw_data['er_tc_1']) + ((raw_data['ground_wkts_2'] - 1) * raw_data['ew_tc_2']) + ((raw_data['ground_wkts_3'] - 1) * raw_data['ew_tc_3']))

# raw_data.to_csv(fr'{user_name}\Documents\Tempdata\raw_data_mmrra2.csv', index=False)

lineups = pd.read_sql_query("""select matchid as "matchID", player as batter, playerid as batterid, team as "battingTeam", carded from player_ratings.t20_lineups_updated""", con=connection)
lineups = lineups.drop_duplicates(subset=['matchID', 'batterid'])
# lineups.to_csv(fr'{user_name}\Documents\Tempdata\lineups.csv', index=False)



# expr_pred = pd.read_csv(fr'{user_name}\OneDrive - Decimal Data Services Ltd\PythonData\MatchMarket\er_tc_smooth.csv')
# raw_data = pd.read_csv(fr'{user_name}\Documents\Tempdata\raw_data_mmrra2.csv')
# lineups = pd.read_csv(fr'{user_name}\Documents\Tempdata\lineups.csv')

####### applying to the data (both inningNumber):
his = raw_data.copy()
# his_max_date = his['date'].max()
# his = his[his.matchID == 80457]
his['date'] = pd.to_datetime(his['date'])
his['year'] = his['date'].dt.year
his = his.loc[:, ['competition', 'inningBallsRemaining', 'totalInningWickets', 'matchID', 'batsmanName', 'nonstrikerName', 'battingTeam', 'ID', 'er_tc_smooth_basic',  'ew_tc_smooth_basic', 'rar_ground_sum', 'raw_ground_sum', 'rar_bowl_sum', 'raw_bowl_sum', 'year']]
his = his.merge(lineups, on=('matchID', 'battingTeam')) #
his['in_now'] = np.where((his['batsmanName'] == his['batter']) | (his['nonstrikerName'] == his['batter']), 1, 0)
his['out'] = np.where((his['carded'] >= his['totalInningWickets'] + 2) | (his['in_now'] == 1), 1, 0)
his['wkts_till_bat'] = np.maximum((his['carded'] - his['totalInningWickets']) - 2, 0)
bat_ratings_game = raw_data.loc[:, ['matchID', 'batsmanName', 'oppo_bat_runs', 'oppo_bat_wkts']].drop_duplicates(subset=['matchID', 'batsmanName'])
bat_ratings_game.columns = ['matchID', 'batter', 'oppo_bat_runs', 'oppo_bat_wkts']
bat_ratings_comp = raw_data.loc[:, ['competition', 'batsmanName', 'oppo_bat_runs_old', 'oppo_bat_wkts_old']].drop_duplicates(subset=['competition', 'batsmanName'])
bat_ratings_comp.columns = ['competition', 'batter', 'oppo_bat_runs_comp', 'oppo_bat_wkts_comp']
bat_ratings_all = raw_data.loc[:, ['batsmanName', 'oppo_bat_runs_old', 'oppo_bat_wkts_old']].drop_duplicates(subset=['batsmanName'])
bat_ratings_all.columns = ['batter', 'oppo_bat_runs_old', 'oppo_bat_wkts_old']
his = his.merge(bat_ratings_game, how='left', on=('matchID', 'batter'))
his = his.merge(bat_ratings_comp, how='left', on=('competition', 'batter'))
his = his.merge(bat_ratings_all, how='left', on='batter')
his['oppo_bat_runs_comp'] = his['oppo_bat_runs_comp'].fillna(his['oppo_bat_runs_old'])
his['oppo_bat_wkts_comp'] = his['oppo_bat_wkts_comp'].fillna(his['oppo_bat_wkts_old'])
his['oppo_bat_runs'] = his['oppo_bat_runs'].fillna(his['oppo_bat_runs_comp'])
his['oppo_bat_wkts'] = his['oppo_bat_wkts'].fillna(his['oppo_bat_wkts_comp'])
his['oppo_bat_runs'] = his['oppo_bat_runs'].fillna(1) #just setting oppo_bat_runs to br_tc for now so they are same as above, trying to work out where difference comes from
his['oppo_bat_wkts'] = his['oppo_bat_wkts'].fillna(1)
his['extra_ord'] = ((1 / his['oppo_bat_wkts']) - 1) * his['out']
his.sort_values(by=['ID', 'carded'], ascending=[True, True], inplace=True)
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

his['rar_bat'] = (his['oppo_bat_runs'] - 1) * his['er_tc_smooth']
his['raw_bat'] = (1 - his['oppo_bat_wkts']) * his['ew_tc_smooth']
his = his.sort_values(by='year', ascending=False)

new_data = his.copy()
new_data = pd.pivot_table(new_data, values=['rar_bat', 'raw_bat',  'rar_ground_sum', 'raw_ground_sum', 'rar_bowl_sum', 'raw_bowl_sum'], index=['ID'], aggfunc={'rar_bat': 'sum', 'raw_bat': 'sum', 'rar_ground_sum': 'mean', 'raw_ground_sum': 'mean', 'rar_bowl_sum': 'mean', 'raw_bowl_sum': 'mean'}).reset_index()

new_data['RA_sum'] = new_data['rar_bat'] + new_data['raw_bat'] + new_data['rar_ground_sum'] + new_data['raw_ground_sum'] + new_data['rar_bowl_sum'] + new_data['raw_bowl_sum']

raw_data_og = raw_data_og.merge(new_data.loc[:, ['ID', 'rar_bat', 'raw_bat',  'rar_ground_sum', 'raw_ground_sum', 'rar_bowl_sum', 'raw_bowl_sum', 'RA_sum']], on='ID', how='left')

raw_data_og = raw_data_og.merge(raw_data.loc[:,['ID', 'oppo_bowl_runs', 'oppo_bowl_wkts', 'oppo_bat_runs', 'oppo_bat_wkts', 'ground_runs', 'ground_wkts']], on='ID', how='left')

# adjusted runs, used for the match market and year adjustment
raw_data_og['runsRequiredAdj'] = raw_data_og['runsRequired'] - raw_data_og['RA_sum']
raw_data_og['totalInningRunsToComeAdj'] = raw_data_og['totalInningRunsToCome'] - raw_data_og['RA_sum']

raw_data_og.to_csv(PROJECT_ROOT / 'men/expBall&runsToCome/Data/dataClean.csv', index=False)

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
