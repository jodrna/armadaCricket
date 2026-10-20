# PROCESS: stage 1 - player ratings step 1: DB -> batting AND bowling data (data/*DataCombined, tier data, balls per
#   match).
# See PROCESS.txt at the top of prod_model for the whole run order.
# v2 (2026-10-12): starts as an exact copy of 1_dataGet_w.py; the original is kept in _v1/. The men's v2 changes (recency,
# reversion, age, bowler small samples) are to be re-tested on women's data before being applied here.
# project root = the first folder above this script holding paths.py, so paths / db / shared modules come from THIS
# project wherever it is (OneDrive, GitHub, a copy) - no PYTHONPATH needed (20 Oct 2026)
import sys as _sys
from pathlib import Path as _Path
_root = str(next(p for p in _Path(__file__).resolve().parents if (p / 'paths.py').exists()))
if _root in _sys.path: _sys.path.remove(_root)
_sys.path.insert(0, _root)
import pandas as pd
import numpy as np
from db import engine
from paths import PROJECT_ROOT
connection = engine.connect()


# -------------------------
# T20 Download
# -------------------------
t20_sql = '''
select id, matchid, date,
case when competition = 'WT20I' then 'WT20I'
else
(
    case when ("host" = 'South Africa' or "host" = 'New Zealand')
        then (case when competition = 'SA20' then 'SA20' else "host" end)
    else
        case when (competition = 'Charlotte Edwards Cup' or competition = 'Women''s Cricket Super League')
            then 'Women''s Vitality Blast'
        else
            case when competition = 'Women''s T20 Challenge'
                then 'Women''s Premier League'
            else competition
            end
        end
    end
)
end as competition,
venue, host, innings, innperiod, home, away, battingteam, batterid, batsman, ord, batsmanballs balls_faced_innings, ball,
bowlerid, bowler, byes, legbyes, noball, wide, extras, runs as runs_raw, runs - noball - byes runs, bowlerwicket wkt, realexprbat, realexpwbat, realexpwbowl, realexprbowl, ballsremaining, bowlerball
from match_data.w_t20_bbb
where year > 2014
and tier < 3
and major_nation > 0
and reduced is not true
order by date, matchid desc
'''

allData = pd.read_sql_query(t20_sql, con=connection)


# -------------------------
# set tiers
# -------------------------
allData['bowlingteam'] = np.where(allData['battingteam'] == allData['home'], allData['away'], allData['home'])
allData.loc[(allData['competition'] == 'WT20I') & (~allData['bowlingteam'].isin(['Australia Women', 'England Women', 'India Women', 'New Zealand Women', 'South Africa Women'])), 'competition'] = 'tier_2'
allData.loc[(allData['competition'] == 'WT20I') & (~allData['battingteam'].isin(['Australia Women', 'England Women', 'India Women', 'New Zealand Women', 'South Africa Women'])), 'competition'] = 'tier_2'


# -------------------------
# T20 Bat Data
# -------------------------
batColumns = ['id', 'matchid', 'date', 'competition', 'venue', 'host', 'innings', 'innperiod', 'home', 'away', 'battingteam', 'batterid', 'batsman', 'ord', 'balls_faced_innings', 'bowlerid', 'bowler', 'byes', 'legbyes', 'noball', 'wide', 'extras', 'runs', 'wkt', 'realexprbat', 'realexpwbat', 'ballsremaining']

t20BatData = allData[batColumns].sort_values(['date', 'matchid'], ascending=[True, False]).reset_index(drop=True)
t20BatData = t20BatData[(t20BatData['realexprbat'] > -1) & (t20BatData['realexpwbat'] > -1)]
t20BatData['format'] = 't20'

# -------------------------
# T20 Bowl Data
# -------------------------
bowlColumns = ['matchid', 'date', 'competition', 'host', 'ball', 'innings', 'innperiod', 'home', 'away', 'battingteam', 'batterid', 'batsman', 'ord', 'bowlerball', 'bowlerid', 'bowler', 'byes', 'legbyes', 'noball', 'wide', 'extras', 'runs', 'bowlerwicket', 'realexprbowl', 'realexpwbowl', 'ballsremaining']

t20BowlData = allData.assign(competition=allData['competition'], runs=allData['runs_raw'], bowlerwicket=lambda x: x['wkt'])[bowlColumns].sort_values(['date', 'matchid'], ascending=[True, False]).reset_index(drop=True)
t20BowlData = t20BowlData[(t20BowlData['realexprbowl'] > -1) & (t20BowlData['realexpwbowl'] > -1)]
t20BowlData['format'] = 't20'




# -------------------------
# ODI Download
# -------------------------
odi_sql = '''
select *
from match_data.w_odi_bbb
order by date, matchid desc
'''

odiRawData = pd.read_sql_query(odi_sql, con=connection)
odiRawData = odiRawData.rename(columns={'batsmanballs': 'balls_faced_innings'})

# -------------------------
# ODI Bat Data
# -------------------------
odiBatData = odiRawData.copy()
odiBatData['wkt'] = odiBatData['bowlerwicket']
odiBatData['runs'] = odiBatData['runs'] - odiBatData['noball'] - odiBatData['byes']

odiTeams = ['Australia Women', 'England Women', 'India Women', 'New Zealand Women', 'South Africa Women']

odiBatData = odiBatData[odiBatData['home'].isin(odiTeams)]
odiBatData = odiBatData[odiBatData['away'].isin(odiTeams)]
odiBatData['format'] = 'odi'
odiBatData = odiBatData[t20BatData.columns]

# -------------------------
# ODI Bowl Data
# -------------------------
odiBowlData = odiRawData.copy()

odiTeams = ['Australia Women', 'England Women', 'India Women', 'New Zealand Women', 'South Africa Women']

odiBowlData = odiBowlData[odiBowlData['home'].isin(odiTeams)]
odiBowlData = odiBowlData[odiBowlData['away'].isin(odiTeams)]
odiBowlData['format'] = 'odi'
odiBowlData = odiBowlData[t20BowlData.columns]



# -------------------------
# Combined data
# -------------------------
combinedBatData = pd.concat([odiBatData, t20BatData], axis=0)
combinedBowlData = pd.concat([odiBowlData, t20BowlData], axis=0)


# -------------------------
# Auxiliary Tables
# -------------------------
playerInfo = pd.read_sql_query("select name, newid as playerid, nationality, dob, batstyle, bowlstyle, playerid as cricinfo_id from players_teams.players where dob < '2024-01-01'", con=connection)

balls_per_match = pd.read_sql_query("select * from player_ratings.w_t20_bowler_balls_per_match p", con=connection)
balls_per_match.columns = ['playerid', 'ballspermatch']

tier_data = pd.read_sql_query("select * from match_data.tier_lookup2_w", con=connection)
tier_data.columns = ['competition', 'avg_runs', 'avg_wkts']


# -------------------------
# Exports
# -------------------------
combinedBatData.to_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/data/batDataCombined_w.csv', index=False)
playerInfo.to_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/auxiliaries/playerInfo_w.csv', index=False)
tier_data.to_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/auxiliaries/batTierData_w.csv', index=False)

combinedBowlData.to_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/data/bowlDataCombined_w.csv', index=False)
balls_per_match.to_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/data/ballsPerMatch_w.csv', index=False)
playerInfo.to_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/auxiliaries/playerInfo_w.csv', index=False)
tier_data.to_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/auxiliaries/bowlTierData_w.csv', index=False)