# encoding=utf-8

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

database_url = "mysql+pymysql://root:941010@localhost:3306/modelapp"
# database_url = "mysql+pymysql://root:Caitao_1010@172.17.0.2:3306/modelapp"

engine = create_engine(database_url)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()