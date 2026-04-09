import os
import sqlalchemy
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()

user = os.getenv("DB_USER")
password = quote(os.getenv("DB_PASSWORD"))
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
db = os.getenv("DB_NAME")

engine = sqlalchemy.create_engine(
    f'postgresql://{user}:{password}@{host}:{port}/{db}'
)
