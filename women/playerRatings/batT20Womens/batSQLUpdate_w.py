import pandas as pd
import sqlalchemy
from sqlalchemy import text
from urllib.parse import quote
from db import engine
from paths import PROJECT_ROOT

connection = engine.connect()


# Import
player = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/OneDrive - Decimal Data Services Ltd/player_ratings/bat_t20_womens/all/outputs/sqlUploadPlayer.csv')
innings = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/OneDrive - Decimal Data Services Ltd/player_ratings/bat_t20_womens/all/outputs/sqlUploadInnings.csv')
bat_sqldata_combo = player.merge(innings, on = ('batter', 'playerid', 'host', 'external_rating', 'competition'), suffixes=('_jungle', '_rasoi'))
ratings = pd.read_csv('/Users/jordan/Documents/ArmadaCricket/OneDrive - Decimal Data Services Ltd/player_ratings/bat_t20_womens/all/outputs/batRatingsPlayer3.csv')

# to SQl
player.to_sql("batter_ratings_jungle_w", con=engine, schema="player_ratings", if_exists='replace', index=False)
innings.to_sql("batter_ratings_rasoi_w", con=engine, schema="player_ratings", if_exists='replace', index=False)
bat_sqldata_combo.to_sql("batter_ratings_combo_odi_w", con=engine, schema="player_ratings", if_exists='replace', index=False)

# Use a connection from the engine to execute GRANT statements
with engine.connect() as conn:
    conn.execute(text("GRANT SELECT ON TABLE player_ratings.batter_ratings_jungle_w TO tableau;"
                       "GRANT SELECT ON TABLE player_ratings.batter_ratings_jungle_w TO willhowie;"
                      "GRANT SELECT ON TABLE player_ratings.batter_ratings_rasoi_w TO tableau;"
                      "GRANT SELECT ON TABLE player_ratings.batter_ratings_rasoi_w TO willhowie;"
                      "GRANT SELECT ON TABLE player_ratings.batter_ratings_combo_odi_w TO tableau;"
                     "GRANT SELECT ON TABLE player_ratings.batter_ratings_combo_odi_w TO willhowie;"
                      ))
    conn.commit()




# upload historic outputs
sql_upload_2 = ratings.copy()
sql_upload_2 = sql_upload_2[sql_upload_2.matchid > 0]
sql_upload_2 = sql_upload_2.loc[:, ['batsman', 'playerid', 'competition', 'host', 'run_rating_3', 'wkt_rating_3', 'balls_faced_r', 'date', 'matchid']]
sql_upload_2.columns = ['batter', 'playerid', 'competition', 'host', 'run_rating', 'wkt_rating', 'balls_faced', 'date', 'matchid']


sql_upload_2.to_sql("batter_ratings_historic_w", con=engine, schema="player_ratings", if_exists='replace', index=False, dtype={'date': sqlalchemy.types.Date()})
# Use a connection from the engine to execute GRANT statements
with engine.connect() as conn:
    conn.execute(text(
                      "GRANT SELECT ON TABLE player_ratings.batter_ratings_historic_w TO willhowie;"
                      "GRANT SELECT ON TABLE player_ratings.batter_ratings_historic_w TO jakelingard;"
    ))
    conn.commit()


