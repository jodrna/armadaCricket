import pandas as pd
import sqlalchemy
from sqlalchemy import text
from urllib.parse import quote
import runpy

# run the outputs
runpy.run_path("1_dataGet.py")
runpy.run_path("2_dataClean.py")
runpy.run_path("3_batModel.py")
runpy.run_path("4_batReplacement.py")
runpy.run_path("5_batReversion.py")


# SQL connection (engine)
engine = sqlalchemy.create_engine('postgresql://jordan:%s@77.68.112.208:5432/postgres' % quote('cricket123'))

# Import
jungle = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/OneDrive - Decimal Data Services Ltd/player_ratings/bat_t20_mens/all/outputs/sqlUploadJungle.csv')
rasoi = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/OneDrive - Decimal Data Services Ltd/player_ratings/bat_t20_mens/all/outputs/sqlUploadRasoi.csv')
bat_sqldata_combo = jungle.merge(rasoi, on = ('batter', 'playerid', 'host', 'external_rating', 'competition'), suffixes=('_jungle', '_rasoi'))
ratings = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/OneDrive - Decimal Data Services Ltd/player_ratings/bat_t20_mens/all/outputs/batRatingsJungle3.csv')
player_info = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/OneDrive - Decimal Data Services Ltd/player_ratings/bat_t20_mens/all/auxiliaries/playerInfo.csv', parse_dates=['dob'])

# merge in cricinfo player id
bat_sqldata_combo = bat_sqldata_combo.merge(player_info.loc[:, ['playerid', 'cricinfo_id']], on='playerid', how='left')


# to SQl
jungle.to_sql("batter_ratings_jungle", con=engine, schema="player_ratings", if_exists='replace', index=False)
rasoi.to_sql("batter_ratings_rasoi", con=engine, schema="player_ratings", if_exists='replace', index=False)
bat_sqldata_combo.to_sql("batter_ratings_combo_odi", con=engine, schema="player_ratings", if_exists='replace', index=False)

# Use a connection from the engine to execute GRANT statements
with engine.connect() as conn:
    conn.execute(text("GRANT SELECT ON TABLE player_ratings.batter_ratings_jungle TO tableau;"
                       "GRANT SELECT ON TABLE player_ratings.batter_ratings_jungle TO willhowie;"
                      "GRANT SELECT ON TABLE player_ratings.batter_ratings_rasoi TO tableau;"
                      "GRANT SELECT ON TABLE player_ratings.batter_ratings_rasoi TO willhowie;"
                      "GRANT SELECT ON TABLE player_ratings.batter_ratings_combo_odi TO tableau;"
                     "GRANT SELECT ON TABLE player_ratings.batter_ratings_combo_odi TO willhowie;"
                      ))
    conn.commit()






# upload historic outputs
sql_upload_2 = ratings.copy()
sql_upload_2 = sql_upload_2[sql_upload_2.matchid > 0]
sql_upload_2 = sql_upload_2.loc[:, ['batsman', 'playerid', 'competition', 'host', 'run_rating_3', 'wkt_rating_3', 'balls_faced_r', 'date', 'matchid']]
sql_upload_2.columns = ['batter', 'playerid', 'competition', 'host', 'run_rating', 'wkt_rating', 'balls_faced', 'date', 'matchid']


sql_upload_2.to_sql("batter_ratings_historic", con=engine, schema="player_ratings", if_exists='replace', index=False, dtype={'date': sqlalchemy.types.Date()})
# Use a connection from the engine to execute GRANT statements
with engine.connect() as conn:
    conn.execute(text(
                      "GRANT SELECT ON TABLE player_ratings.batter_ratings_historic TO willhowie;"
                      "GRANT SELECT ON TABLE player_ratings.batter_ratings_historic TO jakelingard;"
    ))
    conn.commit()

