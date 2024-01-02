# encoding=utf-8
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from crud import save_search, save_mostsim
from database import SessionLocal
from schemas import Uinput, Sim
from loguru import logger

import sys

sys.path.append("..")

from pred import modelPipe


# 定义数据结构
class Simparam(BaseModel):
    text1: str
    thred: float
    num: int


class Humanlabel(BaseModel):
    text1: str
    mostsim: str


app = FastAPI()


@app.on_event("startup")
def startup():
    logger.add("logs/app.log")

    global db
    db = SessionLocal()

    global model
    basemodel = '../basemodel/hfl_chinese_bert_wwm_ext'  # 基础模型
    simcesmodel = '../trained_checkpoints/simcse/simcse_model_tao_pure_0.1_epoch_0'  # 对比学习训练结果
    trainedmodel = '../trained_checkpoints/simcse_bert/simcse2bert_pure_tanh_0'  # Bert cross encoder训练结果
    model = modelPipe(basemodel, simcesmodel, trainedmodel)

    # 将绿产目录加载至内存
    global industry
    industry = pd.read_excel('../sample/green_industry_sim.xlsx')


@app.post("/get_sim")
def get_sim(item: Simparam):
    item = item.model_dump()
    text1 = item['text1']
    # 模型推理
    result = pd.DataFrame()
    for i in range(industry.shape[0]):
        text2 = industry.iloc[i, 2]
        _theme = industry.iloc[i, 0]
        _detail = industry.iloc[i, 1]
        _score = model.pred1(text1, text2).item()

        _d = pd.DataFrame({
            'theme': [_theme],
            'detail': [_detail],
            'score': [_score]
        })
        result = pd.concat([result, _d])
    result.sort_values(by='score', ascending=False, inplace=True)
    result.reset_index(drop=True, inplace=True)
    # logger.info("入参：num:{}, thred{}".format(item['num'], item['thred']))
    out = result.iloc[:item['num'], :].copy()
    out1 = out[out['score'] > item['thred']].copy()
    out_response = json.dumps(out1.to_dict(orient='dict'), ensure_ascii=False)
    # 用户搜集记录保存
    try:
        save_search(db, Uinput(
            text1=item['text1'], outtop=out_response,
            thred=item['thred'], topn=item['num']
        ))
        logger.info("get_sim: 用户输入：{}，{}，{}，保存失败".format(item['text1'], item['thred'], item['num']))
    except:
        raise HTTPException(status_code=400, detail="搜索记录保存失败！")

    logger.info("get_sim: 用户输入：{}，{}，{}，保存成功".format(item['text1'], item['thred'], item['num']))
    return out_response


@app.post("/write_to_sql")
def write_to_sql(item: Humanlabel):
    item = item.model_dump()
    text1 = item['text1']
    mostsim = item['mostsim']
    # 最相似文本写入数据库
    try:
        save_mostsim(db, inputs=Sim(
            text1=text1, mostsim=mostsim
        ))
    except:
        logger.info("write_to_sql: 用户输入{},{}，保存失败".format(text1, mostsim))
        raise HTTPException(status_code=400, detail="人工标定最相似结果保存失败！")
    logger.info("write_to_sql: 用户输入{},{}，保存成功".format(text1, mostsim))
    return {'message': '保存成功'}
