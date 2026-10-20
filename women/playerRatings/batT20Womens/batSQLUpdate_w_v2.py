# PROCESS: stage 1 - uploads the final batter AND bowler ratings (runs bowlSQLUpdate) ->
#   player_ratings.*_ratings_historic. Writes to the DB.
# See PROCESS.txt at the top of prod_model for the whole run order.
# v2 (2026-10-12): starts as an exact copy of batSQLUpdate_w.py; the original is kept in _v1/. The men's v2 changes (recency,
# reversion, age, bowler small samples) are to be re-tested on women's data before being applied here.

# -------------------------
# v2 safety guard: these scripts truncate and re-upload the LIVE player_ratings tables. v2 signed off 2026-10-14;
# set to False to block uploading again.
# -------------------------
ALLOW_LIVE_UPLOAD = True  # v2 signed off 2026-10-14
if not ALLOW_LIVE_UPLOAD:
    raise SystemExit('v2 upload blocked: set ALLOW_LIVE_UPLOAD = True in this script once v2 has been signed off.')
# project root = the first folder above this script holding paths.py, so paths / db / shared modules come from THIS
# project wherever it is (OneDrive, GitHub, a copy) - no PYTHONPATH needed (20 Oct 2026)
import sys as _sys
from pathlib import Path as _Path
_root = str(next(p for p in _Path(__file__).resolve().parents if (p / 'paths.py').exists()))
if _root in _sys.path: _sys.path.remove(_root)
_sys.path.insert(0, _root)
import pandas as pd
import sqlalchemy
from sqlalchemy import text
import runpy
from datetime import date, timedelta
from db import engine
from paths import PROJECT_ROOT
import subprocess
from pathlib import Path
connection = engine.connect()

### 1 will only run if new matches are present in the db since last run
### 0 will push through batter ratings only, regardless of when the last run was
run_type = 0


sql_test = pd.read_sql_query("""select max(last_date) as last_date from player_ratings.max_date_ratings_w""", con=connection)
last_date = str(sql_test['last_date'].iloc[0])[:10]
yesterday = (date.today() - timedelta(days=1)).isoformat()
if (last_date == yesterday) & (run_type == 1):
    print(f"Ratings_w including yesterday's ({yesterday})  games already, stopping.")
    exit()

# run data get
runpy.run_path('1_dataGet_w_v2.py')

# check last date of downloaded data to check if need to run rest of the ratings:
last_date2 = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/data/batDataCombined_w.csv')
last_date2 = last_date2['date'].max()
if (last_date2 == last_date) & (run_type == 1):
    print(f"Max date ({last_date2}) is same as last run, stopping.")
    exit()

# player_ratings.max_date_ratings_w is updated at the END (2026-10-15), only once every upload has succeeded: updated here,
# a failure in steps 2-5 or an upload would leave the date saying the ratings are current when they are not

# run the other files
runpy.run_path('2_batDataClean_w_v2.py')
runpy.run_path('3_batModel_w_v2.py')
runpy.run_path('4_batReplacement_w_v2.py')
runpy.run_path('5_batReversion_w_v2.py')

# Import
recencies = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/outputs/batRecencies_w.csv')
jungle = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/outputs/sqlUploadJungle_w.csv')
rasoi = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/outputs/sqlUploadRasoi_w.csv')
bat_sqldata_combo = jungle.merge(rasoi, on=('batter', 'playerid', 'host', 'external_rating', 'competition'), suffixes=('_jungle', '_rasoi'))
ratings = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/outputs/batRatingsJungle3_w.csv')
player_info = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/batT20Womens/auxiliaries/playerInfo_w.csv', parse_dates=['dob'])

# # merge in cricinfo player id
# # bat_sqldata_combo = bat_sqldata_combo.merge(player_info.loc[:, ['playerid', 'cricinfo_id', 'name']], on='playerid', how='left')
# bat_sqldata_combo['batter'] = bat_sqldata_combo['name']
# bat_sqldata_combo = bat_sqldata_combo.drop(columns=['name'])


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
truncate_and_upload(jungle, 'batter_ratings_jungle_w')
truncate_and_upload(rasoi, 'batter_ratings_rasoi_w')
truncate_and_upload(bat_sqldata_combo, 'batter_ratings_combo_odi_w')
truncate_and_upload(recencies, 'bat_recency_weightings_wh_w')

with engine.begin() as conn:
    conn.execute(text(
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.batter_ratings_jungle_w TO tableau;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.batter_ratings_jungle_w TO willhowie;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.batter_ratings_jungle_w TO jordan;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.batter_ratings_rasoi_w TO tableau;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.batter_ratings_rasoi_w TO willhowie;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.batter_ratings_rasoi_w TO jordan;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.batter_ratings_combo_odi_w TO tableau;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.batter_ratings_combo_odi_w TO willhowie;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.batter_ratings_combo_odi_w TO jordan;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.bat_recency_weightings_wh_w TO tableau;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.bat_recency_weightings_wh_w TO willhowie;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.bat_recency_weightings_wh_w TO jordan;'
    ))

# upload historic outputs
sql_upload_2 = ratings.copy()
sql_upload_2 = sql_upload_2[sql_upload_2.matchid > 0]
sql_upload_2 = sql_upload_2.loc[:, ['batsman', 'playerid', 'competition', 'host', 'run_rating_3', 'wkt_rating_3', 'balls_faced_r', 'date', 'matchid', 'ord_r']]
sql_upload_2.columns = ['batter', 'playerid', 'competition', 'host', 'run_rating', 'wkt_rating', 'balls_faced', 'date', 'matchid', 'ord']

truncate_and_upload(
    sql_upload_2,
    'batter_ratings_historic_w',
    dtype={'date': sqlalchemy.types.Date()}
)

with engine.begin() as conn:
    conn.execute(text(
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.batter_ratings_historic_w TO willhowie;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.batter_ratings_historic_w TO jakelingard;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.batter_ratings_historic_w TO jordan;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.batter_ratings_historic_w TO decimalwebsite;'
    ))

print(f"{last_date2} is now the latest game in the batter ratings_w.")

if run_type == 1:
    print("Beginning bowler ratings_w...")
    script_path = Path(__file__).parent.parent / 'bowlT20Womens' / 'bowlSQLUpdate_w_v2.py'
    subprocess.run(['python', str(script_path)], check=True)  # raises if the bowler update fails

    print(f"{last_date2} is now the latest game in the bowler ratings_w")

# all uploads done: record the latest game in the ratings
with engine.begin() as conn:
    conn.execute(text("UPDATE player_ratings.max_date_ratings_w SET last_date = :max_date"), {"max_date": last_date2})
print(f"player_ratings.max_date_ratings_w set to {last_date2}")
