import pandas as pd
from sqlalchemy import text
from db import engine
from paths import PROJECT_ROOT


# -------------------------
# Settings
# -------------------------

mode = 'upload'          # 'upload' or 'download'

schema_name = 'player_ratings'
table_name = 'sim_class_adjusted_women'


# -------------------------
# Upload
# -------------------------

if mode == 'upload':

    simData = pd.read_csv(
        PROJECT_ROOT / 'women/expBall&runsToCome/outputs/ballSimsClassOrd_w.csv'
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
        PROJECT_ROOT / 'women/expBall&runsToCome/outputs/simDataSmall_w.csv',
        index=False
    )

    print(
        f'Downloaded {len(simData):,} rows to '
        f'{PROJECT_ROOT / "women/expBall&runsToCome/outputs/simDataSmall_w.csv"}'
    )


else:
    raise ValueError("mode must be either 'upload' or 'download'")


