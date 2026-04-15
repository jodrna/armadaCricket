import pandas as pd
import sqlalchemy
from sqlalchemy import text
import runpy
from db import engine
from paths import PROJECT_ROOT
connection = engine.connect()

# # run the outputs
# runpy.run_path('1_dataGet.py')
# runpy.run_path('2_dataClean.py')
# runpy.run_path('3_batModel.py')
# runpy.run_path('4_batReplacement.py')
# runpy.run_path('5_batReversion.py')


# Import
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

# Use a connection from the engine to execute GRANT statements
with engine.connect() as conn:
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
    ))
    conn.commit()


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
with engine.connect() as conn:
    conn.execute(text(
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.batter_ratings_historic TO willhowie;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.batter_ratings_historic TO jakelingard;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.batter_ratings_historic TO jordan;'
    ))
    conn.commit()