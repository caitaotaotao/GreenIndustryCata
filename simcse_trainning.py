import torch
import logging
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import logging as trans_logging
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sample_cooking import simcse_sample

global logger


class DataReader(Dataset):
    def __init__(self, tokenizer, sample, max_len):
        self.tokenizer = tokenizer
        self.sample = sample
        self.max_len = max_len
        self.dataList = self.datas_to_torachTensor()
        self.allLength = len(self.dataList)

    def convert_text2ids(self, text):
        text = text[0:self.max_len - 2]
        inputs = self.tokenizer(text)

        input_ids = inputs['input_ids']

        attention_mask = inputs['attention_mask']
        paddings = [0] * (self.max_len - len(input_ids))
        input_ids += paddings
        attention_mask += paddings

        token_type_id = [0] * self.max_len

        return input_ids, attention_mask, token_type_id

    def datas_to_torachTensor(self):
        res = []
        for line in tqdm(self.sample, desc='tokenization', ncols=50):
            temp = []
            input_ids, attention_mask, token_type_id = self.convert_text2ids(text=line)
            input_ids = torch.as_tensor(input_ids, dtype=torch.long)
            attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
            token_type_id = torch.as_tensor(token_type_id, dtype=torch.long)
            temp.append(input_ids)
            temp.append(attention_mask)
            temp.append(token_type_id)
            res.append(temp)
        return res

    def __getitem__(self, item):
        input_ids = self.dataList[item][0]
        attention_mask = self.dataList[item][1]
        token_type_id = self.dataList[item][2]
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_id}

    def __len__(self):
        return self.allLength


class CSECollator(object):
    def __init__(self,
                 tokenizer,
                 features=("input_ids", "attention_mask", "token_type_ids"),
                 max_len=100):
        self.tokenizer = tokenizer
        self.features = features
        self.max_len = max_len

    def collate(self, batch):
        new_batch = []
        for example in batch:
            for i in range(2):
                # 每个句子重复两次
                new_batch.append({fea: example[fea] for fea in self.features})
        new_batch = self.tokenizer.pad(
            new_batch,
            padding=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return new_batch


def load_data(tokenizer):
    sample = simcse_sample('./sample', pure=True)
    dataset = DataReader(tokenizer, sample, 128)
    collator = CSECollator(tokenizer, max_len=128)
    dl = DataLoader(dataset=dataset, collate_fn=collator.collate, batch_size=16, shuffle=False)  # 定义batch取数方法
    return dl


class SimCSE(torch.nn.Module):
    def __init__(self, modelpath, pool_type='pooler', dropout_rate=0.3):
        super().__init__()
        conf = BertConfig.from_pretrained(modelpath)
        conf.attention_probs_dropout_prob = dropout_rate
        conf.hidden_dropout_prob = dropout_rate
        self.bert = BertModel.from_pretrained(modelpath, config=conf)
        self.pool_type = pool_type

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids)
        if self.pool_type == "cls":
            output = output.last_hidden_state[:, 0]
        elif self.pool_type == "pooler":
            output = output.pooler_output
        return output


def compute_loss(y_pred, device, tao=0.5):
    idxs = torch.arange(0, y_pred.shape[0], device=device)
    y_true = idxs + 1 - idxs % 2 * 2
    similarities = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=2)
    similarities = similarities - torch.eye(y_pred.shape[0], device=device) * 1e12
    similarities = similarities / tao
    loss = F.cross_entropy(similarities, y_true)
    return torch.mean(loss)


def train(tao, modelpath, device, model_out, max_epoch, logger, pool_type='pooler', dropout_rate=0.3):
    tokenizer = BertTokenizer.from_pretrained(modelpath)
    dl = load_data(tokenizer)
    model = SimCSE(modelpath, pool_type, dropout_rate)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    model.train()
    batch_idx = 0
    for epoch_idx in range(max_epoch):
        logger.info("=========================epoch: {}========================================".format(epoch_idx))
        for data in tqdm(dl):
            batch_idx += 1
            pred = model(input_ids=data["input_ids"].to(device),
                         attention_mask=data["attention_mask"].to(device),
                         token_type_ids=data["token_type_ids"].to(device))
            loss = compute_loss(pred, device, tao)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()
            if batch_idx % 1000 == 0:
                logger.info(f"batch_idx: {batch_idx}, loss: {loss:>10f}")

        torch.save(model.state_dict(), model_out + '_model_tao_pure_{}_epoch_{}'.format(tao, epoch_idx))
    return


def main():
    logger = logging.getLogger('simcse_logger')
    logger.setLevel(logging.DEBUG)
    trans_logging.set_verbosity_error()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handle_txt = logging.FileHandler('./logging/simcse_1227.txt')
    handle_txt.setLevel(logging.DEBUG)
    handle_txt.setFormatter(formatter)
    handle_con = logging.StreamHandler()
    handle_con.setLevel(logging.DEBUG)
    handle_con.setFormatter(formatter)

    logger.addHandler(handle_txt)
    logger.addHandler(handle_con)

    logger.info("测试日志，对比学习开始")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tao = 0.1
    model_path = r"./basemodel/hfl_chinese_bert_wwm_ext"
    model_out = r"./trained_checkpoints/simcse"
    max_epoch = 1

    train(tao, model_path, device, model_out, max_epoch, logger)


if __name__ == '__main__':
    main()
