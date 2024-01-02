# enconding=utf-8

from sqlalchemy.orm import Session

import models
import schemas


def get_industry(db: Session):
    return db.query(models.Greenindustry).all()


def save_search(db: Session, inputs: schemas.Uinput):
    db_inputs = models.UserInput(TEXT1=inputs.text1, OUTTOP=inputs.outtop,
                                 THRED=inputs.thred, TOPN=inputs.topn)
    db.add(db_inputs)
    db.commit()
    db.refresh(db_inputs)
    return db_inputs


def save_mostsim(db: Session, inputs: schemas.Sim):
    db_inputs = models.Mostsim(TEXT1=inputs.text1, MOSTSIM=inputs.mostsim)
    db.add(db_inputs)
    db.commit()
    db.refresh(db_inputs)
    return db_inputs
