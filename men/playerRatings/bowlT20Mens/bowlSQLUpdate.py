import pandas as pd
import sqlalchemy
from sqlalchemy import text
from urllib.parse import quote
import runpy
from db import engine
from paths import PROJECT_ROOT

connection = engine.connect()

# run the outputs
runpy.run_path("2_dataClean.py")
runpy.run_path("3_bowlModel.py")
runpy.run_path("4_bowlReplacement.py")
runpy.run_path("5_bowlReversion.py")


# Import
jungle = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/OneDrive - Decimal Data Services Ltd/player_ratings/bowl_t20_mens/all/outputs/sqlUploadJungle.csv')
rasoi = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/OneDrive - Decimal Data Services Ltd/player_ratings/bowl_t20_mens/all/outputs/sqlUploadRasoi.csv')
bowl_sqldata_combo = jungle.merge(rasoi, on = ('bowler', 'playerid', 'host', 'external_rating', 'competition'), suffixes=('_jungle', '_rasoi'))
ratings = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/OneDrive - Decimal Data Services Ltd/player_ratings/bowl_t20_mens/all/outputs/bowlRatingsJungle3.csv')
player_info = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/OneDrive - Decimal Data Services Ltd/player_ratings/bowl_t20_mens/all/auxiliaries/playerInfo.csv', parse_dates=['dob'])       # date of birth, hand, nationality, bowler type etc

ratings = ratings[ratings['bowler'] == 'Usman Tariq']
# merge in cricinfo player id
bowl_sqldata_combo = bowl_sqldata_combo.merge(player_info.loc[:, ['playerid', 'cricinfo_id', 'name']], on='playerid', how='left')
bowl_sqldata_combo['bowler'] = bowl_sqldata_combo['name']
bowl_sqldata_combo = bowl_sqldata_combo.drop(columns=['name'])

# to SQl
jungle.to_sql("bowler_ratings_jungle", con=engine, schema="player_ratings", if_exists='replace', index=False)
rasoi.to_sql("bowler_ratings_rasoi", con=engine, schema="player_ratings", if_exists='replace', index=False)
bowl_sqldata_combo.to_sql("bowler_ratings_combo_odi", con=engine, schema="player_ratings", if_exists='replace', index=False)

# Use a connection from the engine to execute GRANT statements
with engine.connect() as conn:
    conn.execute(text("GRANT SELECT ON TABLE player_ratings.bowler_ratings_jungle TO tableau;"
                       "GRANT SELECT ON TABLE player_ratings.bowler_ratings_jungle TO willhowie;"
                      "GRANT SELECT ON TABLE player_ratings.bowler_ratings_rasoi TO tableau;"
                      "GRANT SELECT ON TABLE player_ratings.bowler_ratings_rasoi TO willhowie;"
                      "GRANT SELECT ON TABLE player_ratings.bowler_ratings_combo_odi TO tableau;"
                     "GRANT SELECT ON TABLE player_ratings.bowler_ratings_combo_odi TO willhowie;"
                      ))
    conn.commit()




# upload historic outputs
sql_upload_2 = ratings.copy()
sql_upload_2 = sql_upload_2[sql_upload_2.matchid > 0]
sql_upload_2 = sql_upload_2.loc[:, ['bowler', 'playerid', 'competition', 'host', 'run_rating_3', 'wkt_rating_3', 'balls_bowled_r', 'date', 'matchid']]
sql_upload_2.columns = ['bowler', 'playerid', 'competition', 'host', 'run_rating', 'wkt_rating', 'balls_faced', 'date', 'matchid']


sql_upload_2.to_sql("bowler_ratings_historic", con=engine, schema="player_ratings", if_exists='replace', index=False, dtype={'date': sqlalchemy.types.Date()})
# Use a connection from the engine to execute GRANT statements
with engine.connect() as conn:
    conn.execute(text(
                      "GRANT SELECT ON TABLE player_ratings.bowler_ratings_historic TO willhowie;"
                      "GRANT SELECT ON TABLE player_ratings.bowler_ratings_historic TO jakelingard;"

    ))
    conn.commit()




