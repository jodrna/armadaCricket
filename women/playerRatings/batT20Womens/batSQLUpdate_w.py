import pandas as pd
import sqlalchemy
from sqlalchemy import text
import runpy
from datetime import date
from db import engine
from paths import PROJECT_ROOT
import subprocess
from pathlib import Path

lastRatingsUpdate = PROJECT_ROOT / 'women/playerRatings/batT20Womens/batLastRatingsUpdate_w.txt'
today = date.today().isoformat()

if lastRatingsUpdate.exists():
    last_run_date = lastRatingsUpdate.read_text().strip()
else:
    last_run_date = None

if last_run_date == today:
    print(f'Ratings already updated today: {today}')

else:
    connection = engine.connect()

    # run the files
    runpy.run_path('1_dataGet_w.py')
    runpy.run_path('2_batDataClean_w.py')
    runpy.run_path('3_batModel_w.py')
    runpy.run_path('4_batReplacement_w.py')
    runpy.run_path('5_batReversion_w.py')

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
    sql_upload_2 = sql_upload_2.loc[:, ['batsman', 'playerid', 'competition', 'host', 'run_rating_3', 'wkt_rating_3', 'balls_faced_r', 'date', 'matchid']]
    sql_upload_2.columns = ['batter', 'playerid', 'competition', 'host', 'run_rating', 'wkt_rating', 'balls_faced', 'date', 'matchid']

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

    lastRatingsUpdate.write_text(today)
    print(f'Bat ratings updated and last run date saved: {today}')

    script_path = Path(__file__).parent.parent / 'bowlT20Womens' / 'bowlSQLUpdate_w.py'
    subprocess.run(['python', str(script_path)])

    print(f"Bowler ratings updated.")




