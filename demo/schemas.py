# encoding=utf-8
import datetime

from pydantic import BaseModel


class Uinput(BaseModel):
    text1: str
    outtop: str
    thred: float
    topn: int

    class Config:
        from_attributes = True


class Sim(BaseModel):
    text1: str
    mostsim: str

    class Config:
        from_attributes = True


class Greenindustry(BaseModel):
    theme: str
    details: str
    sim_pool: str

    class Config:
        from_attributes = True
