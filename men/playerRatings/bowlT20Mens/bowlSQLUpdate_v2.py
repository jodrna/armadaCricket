# PROCESS: stage 1 - uploads the final bowler ratings (run from batSQLUpdate). Writes to the DB.
# See PROCESS.txt at the top of prod_model for the whole run order.
# v2 (2026-10-12): starts as an exact copy of bowlSQLUpdate.py; the original is kept in _v1/. To be reworked, starting with
# recency weighting and experience (ratings looked systematically out by experience in the ball outcome work).

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
from datetime import date
from db import engine
from paths import PROJECT_ROOT
from pathlib import Path
connection = engine.connect()

# run the files
script_dir = Path(__file__).parent
runpy.run_path(str(script_dir / '2_bowlDataClean_v2.py'))
runpy.run_path(str(script_dir / '3_bowlModel_v2.py'))
runpy.run_path(str(script_dir / '4_bowlReplacement_v2.py'))
runpy.run_path(str(script_dir / '5_bowlReversion_v2.py'))

# Import
recencies = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/bowlT20Mens/outputs/bowlRecencies.csv')
jungle = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/bowlT20Mens/outputs/sqlUploadJungle.csv')
rasoi = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/bowlT20Mens/outputs/sqlUploadRasoi.csv')
bowl_sqldata_combo = jungle.merge(rasoi, on=('bowler', 'playerid', 'host', 'external_rating', 'competition'), suffixes=('_jungle', '_rasoi'))
ratings = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/bowlT20Mens/outputs/bowlRatingsJungle3.csv')
player_info = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/bowlT20Mens/auxiliaries/playerInfo.csv', parse_dates=['dob'])

# merge in cricinfo player id
bowl_sqldata_combo = bowl_sqldata_combo.merge(player_info.loc[:, ['playerid', 'cricinfo_id', 'name']], on='playerid', how='left')
bowl_sqldata_combo['bowler'] = bowl_sqldata_combo['name']
bowl_sqldata_combo = bowl_sqldata_combo.drop(columns=['name'])


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
truncate_and_upload(jungle, 'bowler_ratings_jungle')
truncate_and_upload(rasoi, 'bowler_ratings_rasoi')
truncate_and_upload(bowl_sqldata_combo, 'bowler_ratings_combo_odi')
truncate_and_upload(recencies, 'bowl_recency_weightings_wh')

with engine.begin() as conn:
    conn.execute(text(
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowler_ratings_jungle TO tableau;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowler_ratings_jungle TO willhowie;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowler_ratings_jungle TO jordan;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowler_ratings_rasoi TO tableau;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowler_ratings_rasoi TO willhowie;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowler_ratings_rasoi TO jordan;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowler_ratings_combo_odi TO tableau;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowler_ratings_combo_odi TO willhowie;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowler_ratings_combo_odi TO jordan;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowl_recency_weightings_wh TO tableau;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowl_recency_weightings_wh TO willhowie;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowl_recency_weightings_wh TO jordan;'
    ))

# upload historic outputs
sql_upload_2 = ratings.copy()
sql_upload_2 = sql_upload_2[sql_upload_2.matchid > 0]
sql_upload_2 = sql_upload_2.loc[:, ['bowler', 'playerid', 'competition', 'host', 'run_rating_3', 'wkt_rating_3', 'balls_bowled_r', 'date', 'matchid']]
sql_upload_2.columns = ['bowler', 'playerid', 'competition', 'host', 'run_rating', 'wkt_rating', 'balls_faced', 'date', 'matchid']

truncate_and_upload(
    sql_upload_2,
    'bowler_ratings_historic',
    dtype={'date': sqlalchemy.types.Date()}
)

with engine.begin() as conn:
    conn.execute(text(
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowler_ratings_historic TO willhowie;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowler_ratings_historic TO jakelingard;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowler_ratings_historic TO jordan;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowler_ratings_historic TO decimalwebsite;'
    ))


