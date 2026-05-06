import pandas as pd
import sqlalchemy
from sqlalchemy import text
import runpy
from datetime import date
from db import engine
from paths import PROJECT_ROOT

lastRatingsUpdate = PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/bowlLastRatingsUpdate_w.txt'
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
    runpy.run_path('2_bowlDataClean_w.py')
    runpy.run_path('3_bowlModel_w.py')
    runpy.run_path('4_bowlReplacement_w.py')
    runpy.run_path('5_bowlReversion_w.py')

    # Import
    recencies = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/outputs/bowlRecencies_w.csv')
    jungle = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/outputs/sqlUploadJungle_w.csv')
    rasoi = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/outputs/sqlUploadRasoi_w.csv')
    bowl_sqldata_combo = jungle.merge(rasoi, on=('bowler', 'playerid', 'host', 'external_rating', 'competition'), suffixes=('_jungle', '_rasoi'))
    ratings = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/outputs/bowlRatingsJungle3_w.csv')
    player_info = pd.read_csv(PROJECT_ROOT / 'women/playerRatings/bowlT20Womens/auxiliaries/playerInfo_w.csv', parse_dates=['dob'])

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
    truncate_and_upload(jungle, 'bowler_ratings_jungle_w')
    truncate_and_upload(rasoi, 'bowler_ratings_rasoi_w')
    truncate_and_upload(bowl_sqldata_combo, 'bowler_ratings_combo_odi_w')
    truncate_and_upload(recencies, 'bowl_recency_weightings_wh_w')

    with engine.begin() as conn:
        conn.execute(text(
            'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowler_ratings_jungle_w TO tableau;'
            'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowler_ratings_jungle_w TO willhowie;'
            'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowler_ratings_jungle_w TO jordan;'
            'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowler_ratings_rasoi_w TO tableau;'
            'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowler_ratings_rasoi_w TO willhowie;'
            'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowler_ratings_rasoi_w TO jordan;'
            'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowler_ratings_combo_odi_w TO tableau;'
            'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowler_ratings_combo_odi_w TO willhowie;'
            'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowler_ratings_combo_odi_w TO jordan;'
            'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowl_recency_weightings_wh_w TO tableau;'
            'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowl_recency_weightings_wh_w TO willhowie;'
            'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowl_recency_weightings_wh_w TO jordan;'
        ))

    # upload historic outputs
    sql_upload_2 = ratings.copy()
    sql_upload_2 = sql_upload_2[sql_upload_2.matchid > 0]
    sql_upload_2 = sql_upload_2.loc[:, ['bowler', 'playerid', 'competition', 'host', 'run_rating_3', 'wkt_rating_3', 'balls_bowled_r', 'date', 'matchid']]
    sql_upload_2.columns = ['bowler', 'playerid', 'competition', 'host', 'run_rating', 'wkt_rating', 'balls_bowled', 'date', 'matchid']

    truncate_and_upload(
        sql_upload_2,
        'bowler_ratings_historic_w',
        dtype={'date': sqlalchemy.types.Date()}
    )

    with engine.begin() as conn:
        conn.execute(text(
            'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowler_ratings_historic_w TO willhowie;'
            'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowler_ratings_historic_w TO jakelingard;'
            'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowler_ratings_historic_w TO jordan;'
            'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowler_ratings_historic_w TO decimalwebsite;'
        ))

    lastRatingsUpdate.write_text(today)
    print(f'Ratings updated and last run date saved: {today}')




