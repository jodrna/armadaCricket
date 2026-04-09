import sqlalchemy
import pandas as pd
from urllib.parse import quote
from pathlib import Path

# sql connections
# connection = sqlalchemy.create_engine('postgresql://x:%s@77.68.112.208:5432/postgres' % quote('x'))
connection = sqlalchemy.create_engine('postgresql://x:%s@77.68.112.208:5432/postgres' % quote('x'))

BASE_DIR = Path(__file__).resolve().parent


allData = pd.read_sql_query("""
SELECT id,
       matchid,
       date,
       CASE
         WHEN competition = 'T20I' THEN ( CASE
                                            WHEN major_nation = 2 THEN
                                            ( CASE
                                                WHEN ( battingteam = 'Australia'
                                                        OR battingteam =
                                                           'England'
                                                        OR battingteam = 'India'
                                                        OR battingteam =
                                                           'West Indies'
                                                        OR battingteam =
                                                           'Sri Lanka'
                                                        OR battingteam =
                                                           'Pakistan'
                                                        OR battingteam =
                                                           'New Zealand'
                                                        OR battingteam =
                                                           'South Africa'
                                                        OR battingteam =
                                                           'Afghanistan'
                                                        OR battingteam =
                                                           'Bangladesh' )
                                              THEN
                                                'T20I'
                                                ELSE 'tier_2'
                                              end )
                                            ELSE 'T20I'
                                          end )
         ELSE ( CASE
                  WHEN ( "host" = 'South Africa'
                         AND competition <> 'SA20' )
                        OR "host" = 'New Zealand'
                        OR ( "host" = 'England'
                             AND competition <> 'The Hundred (Men''s Comp)' )
                THEN (
                  CASE
                    WHEN
                "host" = 'England' THEN 'Vitality Blast'
                  ELSE
                "host"
                  end )
                  ELSE competition
                end )
       end                  AS competition,
       venue,
       host,
       innings,
       innperiod,
       home,
       away,
       battingteam,
       batterid,
       batsman,
       ord,
       batsmanballs         balls_faced_innings,
       bowlerid,
       bowler,
       byes,
       legbyes,
       noball,
       wide,
       extras,
       runs - noball - byes runs,
       bowlerwicket         wkt,
       realexprbat,
       realexpwbat,
       ballsremaining,
       -- bowl additions start
       ball,
       bowlerball,
       runs                 AS runs_raw,
       realexprbowl,
       realexpwbowl,
       CASE
         WHEN competition = 'T20I' THEN ( CASE
                                            WHEN major_nation = 2 THEN
                                            ( CASE
                                                WHEN ( bowlingteam = 'Australia'
                                                        OR bowlingteam =
                                                           'England'
                                                        OR bowlingteam = 'India'
                                                        OR bowlingteam =
                                                           'West Indies'
                                                        OR bowlingteam =
                                                           'Sri Lanka'
                                                        OR bowlingteam =
                                                           'Pakistan'
                                                        OR bowlingteam =
                                                           'New Zealand'
                                                        OR bowlingteam =
                                                           'South Africa'
                                                        OR bowlingteam =
                                                           'Afghanistan'
                                                        OR bowlingteam =
                                                           'Bangladesh' )
                                              THEN
                                                'T20I'
                                                ELSE 'tier_2'
                                              end )
                                            ELSE 'T20I'
                                          end )
         ELSE ( CASE
                  WHEN ( "host" = 'South Africa'
                         AND competition <> 'SA20' )
                        OR "host" = 'New Zealand'
                        OR ( "host" = 'England'
                             AND competition <> 'The Hundred (Men''s Comp)' )
                THEN (
                  CASE
                    WHEN
                "host" = 'England' THEN 'Vitality Blast'
                  ELSE
                "host"
                  end )
                  ELSE competition
                end )
       end                  AS competition_bowl
       -- bowl additions end
FROM   (SELECT *,
               CASE
                 WHEN battingteam = home THEN away
                 ELSE home
               end AS bowlingteam
        FROM   (SELECT *
                FROM   match_data.t20_bbb
                UNION ALL
                SELECT *
                FROM   match_data.t20_minor_bbb
                WHERE (major_nation = 2 or tier=1)) tb1) tb
WHERE  year > 2000
       AND ( tier = 1
              OR competition = 'CSA T20 Challenge'
              OR competition = 'CSA Provincial Pro20 Competition'
              OR competition = 'CSA Provincial T20 Challenge'
              OR competition = 'CSA Provincial T20 Cup'
              OR competition = 'MiWAY T20 Challenge'
              OR competition = 'Ram Slam T20 Challenge'
              OR competition = 'South Africa Domestic T20/Pro20'
              OR major_nation = 2 )
       AND major_nation > 0
       AND reduced IS NOT TRUE
--and rating_sample_size > 100
ORDER  BY date,
          matchid DESC
                             """, con=connection)



# bat data
batColumns = ['id','matchid','date','competition','venue','host','innings','innperiod','home','away','battingteam','batterid','batsman','ord','balls_faced_innings','bowlerid','bowler','byes','legbyes','noball','wide','extras','runs','wkt','realexprbat','realexpwbat','ballsremaining']
t20BatData = allData[batColumns].sort_values(['date', 'matchid'], ascending=[True, False]).reset_index(drop=True)
t20BatData = t20BatData[(t20BatData['realexprbat'] > -1) & (t20BatData['realexpwbat'] > -1)]
t20BatData['format'] = 't20'
# bowl data
bowlColumns = ['matchid','date','competition','host','ball','innings','innperiod','home','away','battingteam','batterid','batsman','ord','bowlerball','bowlerid','bowler','byes','legbyes','noball','wide','extras','runs','bowlerwicket','realexprbowl','realexpwbowl','ballsremaining']
t20BowlData = (
    allData.assign(competition=allData['competition_bowl'],
              runs=allData['runs_raw'],
              bowlerwicket=lambda x: x['wkt'])
      [bowlColumns]
      .sort_values(['date','matchid'], ascending=[True, False])
      .reset_index(drop=True)
)
t20BowlData = t20BowlData[(t20BowlData['realexprbowl'] > -1) & (t20BowlData['realexpwbowl'] > -1)]
t20BowlData['format'] = 't20'







# odi download
odiRawData = pd.read_sql_query('''select *
                                from match_data.odi_bbb
                                order by date, matchid desc''', con=connection)

# odi bat section
odiBatData = odiRawData.copy()
odiBatData['innperiod'] = 0
odiBatData['balls_faced_innings'] = 0
odiBatData['wkt'] = odiBatData['bowlerwicket']
odiBatData['runs'] = odiBatData['runs'] - odiBatData['noball'] - odiBatData['byes']
odiTeams = ['Australia', 'England', 'India', 'West Indies', 'Sri Lanka', 'Pakistan', 'New Zealand', 'South Africa', 'Afghanistan', 'Bangladesh']
odiBatData = odiBatData[odiBatData['home'].isin(odiTeams)]
odiBatData = odiBatData[odiBatData['away'].isin(odiTeams)]
odiBatData['format'] = 'odi'
# select only the columns in odi which are also in t20, then combine the 2 dataframes
odiBatData = odiBatData[t20BatData.columns]
combinedBatData = pd.concat([odiBatData, t20BatData], axis=0)



# odi bowl section
odiBowlData = odiRawData.copy()
odiBowlData['innperiod'] = 0
odiBowlData['balls_bowled_innings'] = 0
odiTeams = ['Australia', 'England', 'India', 'West Indies', 'Sri Lanka', 'Pakistan', 'New Zealand', 'South Africa', 'Afghanistan', 'Bangladesh']
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
                                        "from player_ratings.t20_bowler_balls_per_match p ", con=connection)
balls_per_match.columns = ['playerid', 'ballspermatch']


# bat files
combinedBatData.to_csv(BASE_DIR / 'bat_t20_mens' / 'all' / 'data' / 'combinedBatData.csv', index=False)
playerInfo.to_csv(BASE_DIR / 'bat_t20_mens' / 'all' / 'auxiliaries' / 'playerInfo.csv', index=False)

# bowl files
combinedBowlData.to_csv(BASE_DIR / 'bowl_t20_mens' / 'all' / 'data' / 'combinedBowlData.csv', index=False)
balls_per_match.to_csv(BASE_DIR / 'bowl_t20_mens' / 'all' / 'data' / 'ballsPerMatch.csv', index=False)
playerInfo.to_csv(BASE_DIR / 'bowl_t20_mens' / 'all' / 'auxiliaries' / 'playerInfo.csv', index=False)

