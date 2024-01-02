import torch
import threading
import math
import pandas as pd
from transformers import BertTokenizer, BertModel
from torch.nn import Sequential, Linear, Sigmoid


class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.result = None
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None


class Bert2class(torch.nn.Module):
    def __init__(self, model_path, tokenizer):
        super(Bert2class, self).__init__()
        self.model = BertModel.from_pretrained(model_path)
        self.tokenizer = tokenizer

        self.classification = Sequential(
            torch.nn.Dropout(0.3),
            Linear(768, 1),
            Sigmoid()
        )

    def forward(self, input_ids, token_type_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         token_type_ids=token_type_ids)
        '''
        # 白化
        kernel, bias = compute_kernel_bias(out.pooler_output)
        kernel = kernel[:, :self.n_components]
        out1 = transform_and_normalize(out.pooler_output, kernel, bias)
        '''
        out1 = self.classification(out.pooler_output).squeeze(1)
        return out1


class Modelpredict:
    """
    用于加载训练模型，进行推理
    """

    def __init__(self, raw_model_path):
        self.tokenizer = BertTokenizer.from_pretrained(raw_model_path)
        self.model = None
        self.bert_model = raw_model_path
        self.device = torch.device('cuda')

    def load_model(self, finetunned_model_path):
        self.model = Bert2class(self.bert_model, self.tokenizer)
        model_load = torch.load(finetunned_model_path, map_location=torch.device('cuda'))
        self.model.load_state_dict(model_load['model_state_dict'])

        self.model.to(self.device)
        self.model.eval()

    def get_score(self, input_text):
        """
        单次推理
        :param input_text: (text1, text_pair)
        :return: 返回输入字符对是否相似
        """
        with torch.no_grad():
            inputs = self.tokenizer(text=input_text[0], text_pair=input_text[1], add_special_tokens=True,
                                    return_tensors='pt',
                                    return_token_type_ids=True)
            inputs.to(self.device)
            out = self.model(**inputs).item()
        return str(out)

    def predict_rank(self, input_text, df):
        """

        :param input_text:
        :param df: 绿产 or 绿债目录
        :return:
        """

        def _get_score(_input_text, _df):
            score = []
            for i in range(_df.shape[0]):
                _sim = _df.iloc[i, 2]
                with torch.no_grad():
                    _inputs = self.tokenizer(text=_input_text, text_pair=_sim, add_special_tokens=True,
                                             return_tensors='pt',
                                             return_token_type_ids=True)
                    _inputs.to(self.device)
                    _score = self.model(**_inputs).item()
                    score.append(_score)
            _df['SIM_SCORE'] = score
            return _df

        # 开20个线程处理
        theads = locals()
        _temp = math.ceil(df.shape[0] / 20)
        for t in range(20):
            df_part = df.iloc[_temp * t:_temp * (t + 1), :].copy()
            theads[str(t)] = MyThread(_get_score, args=(input_text, df_part))
        for t in range(20):
            theads[str(t)].start()
        for t in range(20):
            theads[str(t)].join()

        result = pd.DataFrame()
        for t in range(20):
            _d = theads[str(t)].get_result()
            result = pd.concat([result, _d])
        result.sort_values(by='SIM_SCORE', ascending=False, inplace=True)
        result.reset_index(drop=True, inplace=True)
        return result
