# encoding=utf-8
from sqlalchemy import Column, INT, VARCHAR, JSON, Float, Integer, DateTime
from database import Base
import datetime


class UserInput(Base):
    __tablename__ = 'user_input'

    ID = Column(INT, primary_key=True, autoincrement=True)
    TEXT1 = Column(VARCHAR)
    OUTTOP = Column(JSON)
    THRED = Column(Float)
    TOPN = Column(Integer)
    systemtime = Column(DateTime, default=datetime.datetime.now())


class Mostsim(Base):
    __tablename__ = 'mostsim'

    ID = Column(INT, primary_key=True, autoincrement=True)
    TEXT1 = Column(VARCHAR)
    MOSTSIM = Column(VARCHAR)
    systemtime = Column(DateTime, default=datetime.datetime.now())


class Greenindustry(Base):
    __tablename__ = 'green_industry'

    id = Column(INT, primary_key=True, autoincrement=True)
    theme = Column(VARCHAR)
    details = Column(VARCHAR)
    sim_pool = Column(VARCHAR)
