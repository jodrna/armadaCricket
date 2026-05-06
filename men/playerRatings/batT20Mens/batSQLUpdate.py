import pandas as pd
from datetime import date, timedelta
import sqlalchemy
from sqlalchemy import text
import runpy
from db import engine
from paths import PROJECT_ROOT
import subprocess
from pathlib import Path

connection = engine.connect()

run_type = 1
### 1 will only run if new matches are present in the db since last run
### 0 will push through batter ratings only, regardless of when the last run was

sql_test = pd.read_sql_query("""select max(last_date) as last_date from player_ratings.max_date_ratings""", con=connection)
last_date = str(sql_test['last_date'].iloc[0])[:10]
yesterday = (date.today() - timedelta(days=1)).isoformat()
if (last_date == yesterday) & (run_type == 1):
    print(f"Ratings including yesterday's ({yesterday})  games already, stopping.")
    exit()

# run data get
runpy.run_path('1_dataGet.py')

# check last date of downloaded data to check if need to run rest of the ratings:
last_date2 = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/data/batDataCombined.csv')
last_date2 = last_date2['date'].max()
if (last_date2 == last_date) & (run_type == 1):
    print(f"Max date ({last_date2}) is same as last run, stopping.")
    exit()

with engine.begin() as conn:
    conn.execute(text("UPDATE player_ratings.max_date_ratings SET last_date = :max_date"), {"max_date": last_date2})

# run the other outputs
runpy.run_path('2_batDataClean.py')
runpy.run_path('3_batModel.py')
runpy.run_path('4_batReplacement.py')
runpy.run_path('5_batReversion.py')


# Import
recencies = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/outputs/batRecencies.csv')
jungle = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/outputs/sqlUploadJungle.csv')
rasoi = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/outputs/sqlUploadRasoi.csv')
bat_sqldata_combo = jungle.merge(rasoi, on=('batter', 'playerid', 'host', 'external_rating', 'competition'), suffixes=('_jungle', '_rasoi'))
ratings = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/outputs/batRatingsJungle3.csv')
player_info = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/batT20Mens/auxiliaries/playerInfo.csv', parse_dates=['dob'])

# merge in cricinfo player id
bat_sqldata_combo = bat_sqldata_combo.merge(player_info.loc[:, ['playerid', 'cricinfo_id']], on='playerid', how='left')


# upload helper
def truncate_and_upload(df, table_name, dtype=None):
    with engine.begin() as conn:
        conn.execute(text(f'TRUNCATE TABLE player_ratings.{table_name}'))

    df.to_sql(
        table_name,
        con=engine,
        schema='player_ratings',
        if_exists='append',
        index=False,
        dtype=dtype
    )


# to SQL
truncate_and_upload(jungle, 'batter_ratings_jungle')
truncate_and_upload(rasoi, 'batter_ratings_rasoi')
truncate_and_upload(bat_sqldata_combo, 'batter_ratings_combo_odi')
truncate_and_upload(recencies, 'bat_recency_weightings_wh')

# Use a connection from the engine to execute GRANT statements
with engine.begin() as conn:
    conn.execute(text(
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.batter_ratings_jungle TO tableau;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.batter_ratings_jungle TO willhowie;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.batter_ratings_jungle TO jordan;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.batter_ratings_rasoi TO tableau;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.batter_ratings_rasoi TO willhowie;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.batter_ratings_rasoi TO jordan;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.batter_ratings_combo_odi TO tableau;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.batter_ratings_combo_odi TO willhowie;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.batter_ratings_combo_odi TO jordan;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.bat_recency_weightings_wh TO tableau;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.bat_recency_weightings_wh TO willhowie;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.bat_recency_weightings_wh TO jordan;'
    ))


# upload historic outputs
sql_upload_2 = ratings.copy()
sql_upload_2 = sql_upload_2[sql_upload_2.matchid > 0]
sql_upload_2 = sql_upload_2.loc[:, ['batsman', 'playerid', 'competition', 'host', 'run_rating_3', 'wkt_rating_3', 'balls_faced_r', 'date', 'matchid']]
sql_upload_2.columns = ['batter', 'playerid', 'competition', 'host', 'run_rating', 'wkt_rating', 'balls_faced', 'date', 'matchid']

truncate_and_upload(
    sql_upload_2,
    'batter_ratings_historic',
    dtype={'date': sqlalchemy.types.Date()}
)

# Use a connection from the engine to execute GRANT statements
with engine.begin() as conn:
    conn.execute(text(
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.batter_ratings_historic TO willhowie;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.batter_ratings_historic TO jakelingard;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.batter_ratings_historic TO jordan;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.batter_ratings_historic TO decimalwebsite;'
    ))

print(f"{last_date2} is now the latest game in the batter ratings. Beginning bowler ratings...")

if run_type == 1:
    script_path = Path(__file__).parent.parent / 'bowlT20Mens' / 'bowlSQLUpdate.py'
    subprocess.run(['python', str(script_path)])

    print(f"{last_date2} is now the latest game in the bowler ratings.")
