import pandas as pd
from db import engine
from paths import PROJECT_ROOT
connection = engine.connect()


# sql query to get data from database
allData =  pd.read_sql_query('''select id, matchid, date,
                                case when competition = 'WT20I' then
                                (case when major_nation = 2 then
                                (case when (battingteam  = 'Australia Women' or battingteam = 'England Women' or battingteam = 'India Women' or battingteam = 'New Zealand Women' or battingteam = 'South Africa Women') then 'WT20I' else 'tier_2' end)
                                else 'WT20I' end)
                                else (case when ("host" = 'South Africa' or "host" = 'New Zealand') then (case when competition = 'SA20' then 'SA20' else "host" end) else
                                case when (competition = 'Charlotte Edwards Cup' or competition = 'Women''s Cricket Super League') then 'Women''s Vitality Blast' else case when competition = 'Women''s T20 Challenge' then 'Women''s Premier League' else competition end end
                                end)
                                end as competition,
                                venue, host, innings, innperiod, home, away, battingteam, batterid, batsman, ord, batsmanballs balls_faced_innings, ball,
                                bowlerid, bowler, byes, legbyes, noball, wide, extras, runs, runs - noball - byes runs_raw, bowlerwicket wkt, realexprbat, realexpwbat, realexpwbowl, realexprbowl, ballsremaining, bowlerball
                                from match_data.w_t20_bbb
                                where year > 2014
                                and tier<3
                                and major_nation > 0
                                and reduced is not true
                                and realexprbat > -1
                                and realexpwbat > -1
                                order by date, matchid desc''', con=connection)



# bat data
batColumns = ['id','matchid','date','competition','venue','host','innings','innperiod','home','away','battingteam','batterid','batsman','ord','balls_faced_innings','bowlerid','bowler','byes','legbyes','noball','wide','extras','runs','wkt','realexprbat','realexpwbat','ballsremaining']
t20BatData = allData[batColumns].sort_values(['date', 'matchid'], ascending=[True, False]).reset_index(drop=True)
t20BatData = t20BatData[(t20BatData['realexprbat'] > -1) & (t20BatData['realexpwbat'] > -1)]
t20BatData['format'] = 't20'
# bowl data
bowlColumns = ['matchid','date','competition','host','ball','innings','innperiod','home','away','battingteam','batterid','batsman','ord','bowlerball','bowlerid','bowler','byes','legbyes','noball','wide','extras','runs','bowlerwicket','realexprbowl','realexpwbowl','ballsremaining']
t20BowlData = (
    allData.assign(competition=allData['competition'],
              runs=allData['runs'],
              bowlerwicket=lambda x: x['wkt'])
      [bowlColumns]
      .sort_values(['date','matchid'], ascending=[True, False])
      .reset_index(drop=True)
)
t20BowlData = t20BowlData[(t20BowlData['realexprbowl'] > -1) & (t20BowlData['realexpwbowl'] > -1)]
t20BowlData['format'] = 't20'







# odi download
odiRawData = pd.read_sql_query('''select *
                                from match_data.w_odi_bbb
                                order by date, matchid desc''', con=connection)
odiRawData = odiRawData.rename(columns={'batsmanballs': 'balls_faced_innings'})

# odi bat section
odiBatData = odiRawData.copy()
# odiBatData['innperiod'] = 0
# odiBatData['balls_faced_innings'] = 0
odiBatData['wkt'] = odiBatData['bowlerwicket']
odiBatData['runs'] = odiBatData['runs'] - odiBatData['noball'] - odiBatData['byes']
odiTeams = ['Australia Women', 'England Women', 'India Women', 'New Zealand Women', 'South Africa Women']
odiBatData = odiBatData[odiBatData['home'].isin(odiTeams)]
odiBatData = odiBatData[odiBatData['away'].isin(odiTeams)]
odiBatData['format'] = 'odi'
# select only the columns in odi which are also in t20, then combine the 2 dataframes
odiBatData = odiBatData[t20BatData.columns]
combinedBatData = pd.concat([odiBatData, t20BatData], axis=0)



# odi bowl section
odiBowlData = odiRawData.copy()
# odiBowlData['innperiod'] = 0
# odiBowlData['balls_bowled_innings'] = 0
odiTeams = ['Australia Women', 'England Women', 'India Women', 'New Zealand Women', 'South Africa Women']
odiBowlData = odiBowlData[odiBowlData['home'].isin(odiTeams)]
odiBowlData = odiBowlData[odiBowlData['away'].isin(odiTeams)]
odiBowlData['format'] = 'odi'
# select only the columns in odi which are also in t20, then combine the 2 dataframes
odiBowlData = odiBowlData[t20BowlData.columns]
combinedBowlData = pd.concat([odiBowlData, t20BowlData], axis=0)




# get player info like nationality etc
playerInfo = pd.read_sql_query("select name, newid as playerid, nationality, dob, batstyle, bowlstyle, playerid as cricinfo_id from players_teams.players where dob < '2024-01-01'",
                                con=connection)

# balls bowled per match on average, needed for replacement bowler value
balls_per_match = pd.read_sql_query("select * "
                                        "from player_ratings.w_t20_bowler_balls_per_match p ", con=connection)
balls_per_match.columns = ['playerid', 'ballspermatch']

# tier data
tier_data = pd.read_sql_query("select * "
                                        "from match_data.tier_lookup2_w", con=connection)
tier_data.columns = ['competition', 'avg_runs', 'avg_wkts']

# bat files
combinedBatData.to_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/data/batDataCombined_w.csv', index=False)
playerInfo.to_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/auxiliaries/playerInfo_w.csv', index=False)
tier_data.to_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/auxiliaries/batTierData_w.csv', index=False)

# bowl files
combinedBowlData.to_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/data/bowlDataCombined_w.csv', index=False)
balls_per_match.to_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/data/ballsPerMatch_w.csv', index=False)
playerInfo.to_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/auxiliaries/playerInfo_w.csv', index=False)
tier_data.to_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/auxiliaries/bowlTierData_w.csv', index=False)


