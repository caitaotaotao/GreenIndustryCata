# encoding=utf-8

"""
Linux版本目前只能通过Pip安装1.12版stremlit
"""

import json
import time

import pandas as pd
import streamlit as st
import requests

st.markdown('''
# 用 途 绿 模 型 测 试
***
***
    模型实现：返回与输入文本最相关绿产目录条目，并按相似概率排序
    DEMO功能：模型推理结果显示，人工输入样本及标注结果保存（本地数据库）
    隐私提示：demo保存信息：1)输入文本；2)选择提交的最相近条目

### 关于绿产匹配任务的思考/说明
#### 1. 模型选择
    模型以Bert作为骨架，采用对比学习（SIMCSE）拉开文本向量空间，再采用Bert cross-encoder 进行二分类实现输入文本与绿产目录的匹配
    1) SIMCSE 对比学习
        无监督学习方式，削弱Bert在微调不足向量空间输出高度相似的问题（即，余弦距离无意义）
    2) 文本二分类
        绿产目录匹配任务有两种路径：利用三级条目做多分类；利用三级条目解释说明做文本相似度
        本次舍弃多分类的原因：211个三级目录，属于超多分类问题，面临类别过多且类别之间边界模糊，导致处理起来难度较大
        a) 文本相似度有两种实现方法：cross-encoder, bi-encoder
            cross-encoder (单塔)
                将两个待匹配文本拼接成一个，再进行分类。好处是可以学习到句子之间的关系，分类效果略好。
                缺点：在检索场景无法提前保存待匹配文本的向量，计算速度较慢
            bi-encoder (双塔)
                孪生神经网络结构，用一个共享网络分别计算两个文本，再通过余弦距离计算相似度
                优点：可提前保存待匹配文本向量，在检索场景推理速度快
                缺点：无法学习两个句子之间的关系，分类效果略差于cross-encoder
#### 2. 训练-验证集
        - 区别于之前的实现，本次实验仅以绿产目录原文作为语料库，未使用用途绿1.0API结果作为标注数据。
        1) 绿产目录原文清洗
        解释说明：
            去除各类技术标准描述（如：书名号内内容） 【降低文本噪音】
        2) SIMCSE
            训练集={三级条目名称, 清洗后的三级条目解释说明}
            温度超参数：tao=0.1
        3) Bert Cross-encoder
            遍历所有三级条目构建文本对：label={1,0}, 现成的交叉熵损失函数无法处理[0,1]以外的标签值域
                三级条目名称: 对应的清洗后三级条目解释说明 : 1
                三级条目名称: 其他的清洗后三级条目解释说明 : 0
                为平衡样本：
                生成的三级条目名称相似文本: 对应的清洗后三级条目解释说明 : 1
        - 验证集：
        人工标注的招投标项目名称数据
#### 3. 神经网络结构
        注：激活函数tanh，将向量全链接层结果拉开至[-1, 1]
        模型结构：bert - dropout(0.3) - Linear(768,1) - Tanh()
        后续可用人工标注好的数据，在此基础上继续训练
***
***
''')

text1 = st.text_input(
    label='**:blue[开始测试: 请分别输入待匹配文本、相似度阈值（默认0.8）、最大输出条目数量（默认5条）]**',
    placeholder='请输入...如：生活垃圾'
)

col1, col2 = st.columns(2, gap='medium')

with col1:
    thred = st.number_input(
        label=":red[请输入判断是否相似的阈值（0-1之间）]",
        min_value=0.0,
        max_value=1.0,
        step=0.1,
        value=0.8
    )

with col2:
    num = st.number_input(
        label=':red[请输入最多输出条目数量]',
        min_value=1,
        max_value=10,
        step=1,
        value=5
    )


def click_button(x):
    st.session_state.stage = x


def show_result(out):
    theme = []
    details = []
    score = []
    for k in out['theme'].keys():
        _theme = out['theme'][k]
        theme.append(_theme)
        _details = out['detail'][k]
        details.append(_details)
        _score = out['score'][k]
        score.append(_score)
    df = pd.DataFrame(data={
        'theme': theme, 'details': details, 'score': score
    })
    st.table(df)
    return df


@st.cache_data
def get_pred(text1, thred, num):
    out = requests.post(
        url=r'http://localhost:8000/get_sim',
        json={
            'text1': text1,
            'thred': thred,
            'num': num
        }
    ).json()

    out = json.loads(out)
    return out


if 'stage' not in st.session_state:
    st.session_state['stage'] = 0

st.session_state.out = pd.DataFrame()

if st.session_state.stage == 0:
    st.button('查询', on_click=click_button, args=[1])

if st.session_state.stage == 1:
    # 1.29.0版本功能
    with st.status("计算Top{}条最相似目录，模型推断中（GPU加速)...".format(num), expanded=True) as status:
        start = time.time()
        out = get_pred(text1, thred, num)
        end = time.time()
        if len(out['theme']) == 0:
            status.update(label='获取成功！总用时{}秒，无匹配结果请重新输入'.format(round(end - start, 2)),
                          state='complete', expanded=True)
            st.write("无匹配结果请重新输入")
            st.button("重新匹配", on_click=click_button, args=[0])
        else:
            status.update(label='获取成功！总用时{}秒'.format(round(end - start, 2)), state='complete', expanded=False)

            df = show_result(out)
            st.session_state.options = df['theme'].to_list() + ["均不符合"]
            st.session_state.stage = 3

if st.session_state.stage == 3:
    out = get_pred(text1, thred, num)
    if len(out['theme']) < num:
        st.write('相似结果不足{}条件，仅展示{}条返回结果'.format(num, len(out['theme'])))
    df = show_result(out)
    mostsim = st.radio(
        label='**请选择最符合的条目，并点击提交**',
        options=st.session_state.options
    )

    st.session_state.mostsim = mostsim

    st.button('提交', on_click=click_button, args=[4])

if st.session_state.stage == 4:
    a = requests.post(
        url=r'http://localhost:8000/write_to_sql',
        json={
            'text1': text1,
            'mostsim': st.session_state.mostsim
        }
    )
    st.write(a.json()['message'])
    st.button("重新匹配", on_click=click_button, args=[0])
