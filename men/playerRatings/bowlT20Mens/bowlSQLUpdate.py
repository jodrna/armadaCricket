import pandas as pd
import sqlalchemy
from sqlalchemy import text
import runpy
from db import engine
from paths import PROJECT_ROOT
connection = engine.connect()

# # run the outputs
# runpy.run_path('2_dataClean.py')
# runpy.run_path('3_bowlModel.py')
# runpy.run_path('4_bowlReplacement.py')
# runpy.run_path('5_bowlReversion.py')


# Import
recencies = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/bowlT20Mens/outputs/recencies.csv')
jungle = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/bowlT20Mens/outputs/sqlUploadJungle.csv')
rasoi = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/bowlT20Mens/outputs/sqlUploadRasoi.csv')
bowl_sqldata_combo = jungle.merge(rasoi, on=('bowler', 'playerid', 'host', 'external_rating', 'competition'), suffixes=('_jungle', '_rasoi'))
ratings = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/bowlT20Mens/outputs/bowlRatingsJungle3.csv')
player_info = pd.read_csv(PROJECT_ROOT / 'men/playerRatings/bowlT20Mens/auxiliaries/playerInfo.csv', parse_dates=['dob'])       # date of birth, hand, nationality, bowler type etc

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

# Use a connection from the engine to execute GRANT statements
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

# Use a connection from the engine to execute GRANT statements
with engine.begin() as conn:
    conn.execute(text(
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowler_ratings_historic TO willhowie;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowler_ratings_historic TO jakelingard;'
        'GRANT ALL PRIVILEGES ON TABLE player_ratings.bowler_ratings_historic TO jordan;'
    ))