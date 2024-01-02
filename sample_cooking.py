import os.path
import torch
import math
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from random import sample
from itertools import product
from transformers import BertTokenizer, RoFormerConfig, RoFormerForCausalLM
from utilities import MyThread

global logger


class TextGenerate:
    def __init__(self, roformer_model_path):
        self.tokenizer = BertTokenizer.from_pretrained(roformer_model_path)
        self.roformer_model_path = roformer_model_path
        self.config = RoFormerConfig.from_pretrained(self.roformer_model_path)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = None

    def load_mode(self):
        self.config.is_decoder = True
        self.config.eos_token_id = self.tokenizer.sep_token_id
        self.config.pooler_activation = "linear"
        self.model = RoFormerForCausalLM.from_pretrained(self.roformer_model_path, config=self.config)
        self.model.to(self.device)
        self.model.eval()

    def gen_synonyms(self, text, n=100, k=20):
        """
            含义： 产生sent的n个相似句，然后返回最相似的k个。
            做法：用seq2seq生成，并用encoder算相似度并排序。
            返回：相似文本列表
        """

        # 寻找所有相似的句子
        r = []
        inputs1 = self.tokenizer(text, return_tensors="pt")

        def _raw(inputs, nn):
            rr = []
            for _ in range(nn):
                inputs.to(self.device)
                _output = self.tokenizer.batch_decode(self.model.generate(**inputs, top_p=0.95, do_sample=True,
                                                                          max_length=128),
                                                      skip_special_tokens=True)[0].replace(" ", "").replace(text, "")
                rr.append(_output)
            return rr

        # 开三个进程试试
        nn = math.floor(n / 3)
        a1 = MyThread(_raw, args=(inputs1, nn))
        a2 = MyThread(_raw, args=(inputs1, nn))
        a3 = MyThread(_raw, args=(inputs1, nn))

        a1.start()
        a2.start()
        a3.start()
        a1.join()
        a2.join()
        a3.join()

        r = a1.get_result() + a2.get_result() + a3.get_result()

        # 对相似的句子进行排序
        r = [i for i in set(r) if i != text and len(i) > 0]
        r = [text] + r
        inputs2 = self.tokenizer(r, padding=True, return_tensors="pt")  # 所有生成的句子一次性

        with torch.no_grad():
            inputs2.to(self.device)
            outputs = self.model(**inputs2)
            z = outputs.logits.cpu().numpy().mean(axis=1)
        z /= (z ** 2).sum(axis=1, keepdims=True) ** 0.5
        argsort = np.dot(z[1:], -z[0]).argsort()
        return [r[i + 1] for i in argsort[:k]]

def sample_cooking_pure(path):
    """
    不增加解释说明的分拆
    三级标题：解释说明
    :param path:
    :return:
    """
    roformer_model_path = r"./basemodel/junnyuroformer_chinese_sim_char_base"
    data_path = os.path.join(path, 'green_industry_sim.xlsx')
    data = pd.read_excel(data_path)

    out_txt = os.path.join(path, 'train_sample_1230_pure.txt')
    # 加载生成式模型
    model = TextGenerate(roformer_model_path)
    model.load_mode()
    with open(out_txt, 'w', encoding='utf-8') as f:
        sample_len = 0
        for i in tqdm(range(data.shape[0])):
            text1 = data.iloc[i, 0].split(" ")[1]
            text_positive = data.iloc[i, 2]
            f.write(text1 + '\t' + text_positive + '\t' + '1' + '\n')
            sample_len += 1
            text_negative = []
            for v in data['SIM_POOL_PURE'].to_list():
                if v != text_positive:
                    text_negative.append(v)
            for l in text_negative:
                f.write(text1 + '\t' + l + '\t' + '0' + '\n')
                sample_len += 1
            # 生成正向样本
            kk = 210
            nn = kk * 3
            text1_generate = model.gen_synonyms(text1, nn, kk)
            for x in text1_generate:
                f.write(str(x) + '\t' + text_positive + '\t' + '1' + '\n')
                sample_len += 1
    logger.info("样本长度{}".format(sample_len))


def simcse_sample(path, pure=False):
    """
    [三级目录, 清洗后的条目描述，条目描述拆分]
    :param pure:
    :param path:
    :return:
    """
    data_path = os.path.join(path, 'green_industry_sim.xlsx')
    data = pd.read_excel(data_path)
    result = []
    if pure:
        for i in tqdm(range(data.shape[0])):
            text1 = data.iloc[i, 0].split(" ")[1]
            result.append(text1)
            detail = data.iloc[i, 2]
            result.append(detail)

    else:
        for i in tqdm(range(data.shape[0])):
            text1 = data.iloc[i, 0].split(" ")[1]
            result.append(text1)
            detail = data.iloc[i, 2]
            result.append(detail)

            detail_break1 = data.iloc[i, 3]
            detail_break2 = data.iloc[i, 4]
            if detail_break1 is not np.nan:
                break1 = detail_break1.split(",")
                if detail_break2 is np.nan:
                    text1_break = break1
                    for x in text1_break:
                        result.append(x)
                else:
                    break2 = detail_break2.split(",")
                    _break = product(break1, break2)
                    text1_break = [j[0] + j[1] for j in _break]
                    for x in text1_break:
                        result.append(x)
            else:
                pass
    return result


if __name__ == '__main__':
    logger = logging.getLogger('samplecooking')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handle_txt = logging.FileHandler('./logging/samplecooking.txt')
    handle_txt.setLevel(logging.DEBUG)
    handle_txt.setFormatter(formatter)
    handle_con = logging.StreamHandler()
    handle_con.setLevel(logging.DEBUG)
    handle_con.setFormatter(formatter)

    logger.addHandler(handle_txt)
    logger.addHandler(handle_con)

    logger.info("测试日志，开始制作训练样本")

    sample_cooking_pure('./sample')
