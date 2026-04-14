import pandas as pd
from db import engine
from paths import PROJECT_ROOT
connection = engine.connect()

combinedBatData = pd.read_sql_query('''select id, matchid, date,
                                case when competition = 'WT20I' then
                                (case when major_nation = 2 then
                                (case when (battingteam  = 'Australia Women' or battingteam = 'England Women' or battingteam = 'India Women' or battingteam = 'New Zealand Women' or battingteam = 'South Africa Women') then 'WT20I' else 'tier_2' end)
                                else 'WT20I' end)
                                else (case when ("host" = 'South Africa' or "host" = 'New Zealand') then (case when competition = 'SA20' then 'SA20' else "host" end) else
                                case when (competition = 'Charlotte Edwards Cup' or competition = 'Women''s Cricket Super League') then 'Women''s Vitality Blast' else case when competition = 'Women''s T20 Challenge' then 'Women''s Premier League' else competition end end
                                end)
                                end as competition,
                                venue, host, innings, innperiod, home, away, battingteam, batterid, batsman, ord, batsmanballs balls_faced_innings,
                                bowlerid, bowler, byes, legbyes, noball, wide, extras, runs - noball - byes runs, bowlerwicket wkt, realexprbat, realexpwbat, ballsremaining
                                from match_data.w_t20_bbb
                                where year > 2014
                                and tier<3
                                and major_nation > 0
                                and reduced is not true
                                and realexprbat > -1
                                and realexpwbat > -1
                                order by date, matchid desc''', con=connection)
combinedBatData['format'] = 't20'


# get player info like nationality etc
playerInfo = pd.read_sql_query("select name, newid as playerid, nationality, dob, batstyle, bowlstyle, playerid as cricinfo_id from players_teams.players where dob < '2024-01-01'",
                                con=connection)


# bat files
combinedBatData.to_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/data/combinedBatData.csv', index=False)
playerInfo.to_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/auxiliaries/playerInfo.csv', index=False)

# bowl files
# combinedBowlData.to_csv(PROJECT_ROOT / 'OneDrive - Decimal Data Services Ltd/player_ratings/bowl_t20_mens/all/data/combinedBowlData.csv', index=False)
# balls_per_match.to_csv(PROJECT_ROOT / 'OneDrive - Decimal Data Services Ltd/player_ratings/bowl_t20_mens/all/data/ballsPerMatch.csv', index=False)
# playerInfo.to_csv(PROJECT_ROOT / 'OneDrive - Decimal Data Services Ltd/player_ratings/bowl_t20_mens/all/auxiliaries/playerInfo.csv', index=False)


