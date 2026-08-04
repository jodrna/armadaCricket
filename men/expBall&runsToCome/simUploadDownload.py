import pandas as pd
from sqlalchemy import text
from db import engine
from paths import PROJECT_ROOT


# -------------------------
# Settings
# -------------------------

mode = 'download'          # 'upload' or 'download'

schema_name = 'player_ratings'
table_name = 'sim_class_adjusted_men'


# -------------------------
# Upload
# -------------------------

if mode == 'upload':

    simData = pd.read_csv(
        PROJECT_ROOT / 'men/expBall&runsToCome/outputs/ballSimsClassOrd.csv'
    )

    simData = simData.loc[:, [
        'simID',
        'inningBallNumber',
        'totalInningWickets',
        'totalInningRunsToCome',
        'sample'
    ]].copy()

    simData.to_sql(
        table_name,
        con=engine,
        schema=schema_name,
        if_exists='replace',
        index=False,
        chunksize=10000,
        method='multi'
    )

    with engine.begin() as conn:
        conn.execute(text(
            f'GRANT ALL PRIVILEGES ON TABLE {schema_name}.{table_name} TO tableau;'
            f'GRANT ALL PRIVILEGES ON TABLE {schema_name}.{table_name} TO willhowie;'
            f'GRANT ALL PRIVILEGES ON TABLE {schema_name}.{table_name} TO jordan;'
        ))

    print(f'Uploaded {len(simData):,} rows to {schema_name}.{table_name}.')


# -------------------------
# Download
# -------------------------

elif mode == 'download':

    simData = pd.read_sql_query(
        f'SELECT * FROM {schema_name}.{table_name}',
        con=engine
    )

    simData.to_csv(
        PROJECT_ROOT / 'men/expBall&runsToCome/outputs/simDataSmall.csv',
        index=False
    )

    print(
        f'Downloaded {len(simData):,} rows to '
        f'{PROJECT_ROOT / "men/expBall&runsToCome/outputs/simDataSmall.csv"}'
    )


else:
    raise ValueError("mode must be either 'upload' or 'download'")


