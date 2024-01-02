import torch
import os
import re
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertConfig, BertModel, AdamW
from transformers import logging as transformer_logging
from torch.utils.data import Dataset, DataLoader


def get_train_dev_dataset(path, train_size, random_seed):
    """
    将数据集分为训练集和验证集
    sklearn.train_test_split
    :return:
    """

    text = []
    label = []

    with open(path, 'r', encoding='utf-8') as sampel_file:
        for line in tqdm(sampel_file.readlines()):
            line = line.replace("\n", "")
            _a = line.split("\t")[0]  # 输入文本
            _b = line.split("\t")[1]  # 待匹配文本
            _label = float(line.split("\t")[2])  # 分类标签
            '''
            if _label == 0:
                _label = -1
            '''
            _c = (_a, _b)

            text.append(_c)
            label.append(_label)

    train_text, dev_text, train_label, dev_label = train_test_split(text, label, train_size=train_size,
                                                                    random_state=random_seed, stratify=label)
    return train_text, dev_text, train_label, dev_label


class CarbonDataset(Dataset):
    def __init__(self, input_text_pool, label, tokenizer, max_length=128):
        self.text_pool = input_text_pool
        self.labels = label
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        pair = self.text_pool[idx]
        text1 = ''.join(re.findall(r'[\u4e00-\u9fa5]', pair[0]))
        text2 = ''.join(re.findall(r'[\u4e00-\u9fa5]', pair[1]))
        inputs = self.tokenizer(text=text1, text_pair=text2, add_special_tokens=True, return_tensors='pt',
                                truncation=True, padding='max_length', max_length=self.max_length,
                                return_token_type_ids=True)
        ids = inputs['input_ids'][0]
        mask = inputs['attention_mask'][0]
        token_type_ids = inputs['token_type_ids'][0]
        label = self.labels[idx]

        return {
            'input_ids': ids.to(torch.long),
            'token_type_ids': token_type_ids.to(torch.long),
            'attention_mask': mask.to(torch.long),
            'labels': torch.tensor(label, dtype=torch.float)
        }

    def __len__(self):
        return len(self.text_pool)


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


class Simcse2BertClassification(torch.nn.Module):
    def __init__(self, pretrain_model, simcsemodel):
        super().__init__()
        self.model = SimCSE(pretrain_model)
        self.model.load_state_dict(torch.load(simcsemodel))
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_model)
        self.classification = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(768, 1),
            torch.nn.Tanh()
        )

    def forward(self, input_ids, token_type_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         token_type_ids=token_type_ids)
        out1 = self.classification(out).squeeze(1)
        return out1


def simcse2bert_train(model, max_epoch, train_loader, test_loader, optim, loss_function, device, savename, logger):
    """
    模型训练函数
    :return:
    """
    logger.info('------------------------------------ 开始训练 -----------------------------------------')
    logger.info('\n')
    step = 0
    for epoch in range(max_epoch):
        logger.info(
            '============================ epoch:{}============================================'.format(epoch))
        if hasattr(torch.cuda, 'empty_cache'):
            # 清空显存
            torch.cuda.empty_cache()

        for bacth in tqdm(train_loader):
            step += 1
            optim.zero_grad()
            # 传参
            input_ids = bacth['input_ids'].to(device)
            token_type_ids = bacth['token_type_ids'].to(device)
            attention_mask = bacth['attention_mask'].to(device)
            labels = bacth['labels'].to(device)

            _out = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)



            loss = loss_function(_out, labels)

            # 每1000步打印一次
            if step % 1000 == 0:
                logger.info('step: {}, loss: {}'.format(step, format(loss.item(), '.3f')))
            # 反向传播
            loss.backward()
            optim.step()

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        # 每epoch进行一次测试
        epoch_eval = carbon_eval(model, test_loader, device, thred=0)
        logger.info("eval--epoch{}: acc:{}, precision:{}, recall:{}".format(epoch, epoch_eval['acc'],
                                                                            epoch_eval['precision'],
                                                                            epoch_eval['recall']))

        # 保存check_point
        _checkPoint = os.path.join('./trained_checkpoints/simcse_bert', savename + str(epoch))
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict()
        },
            _checkPoint)


def carbon_eval(model, test_loader, device, thred):
    model.eval()
    total = len(test_loader)  # 总数
    logger.info("验证集长度{}".format(total))
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    with torch.no_grad():
        for text_data in test_loader:
            input_ids = text_data['input_ids'].to(device)
            token_type_ids = text_data['token_type_ids'].to(device)
            attention_mask = text_data['attention_mask'].to(device)
            labels = text_data['labels'].to(device)

            _out = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).item()
            if _out > thred:
                out_label = 1
            else:
                out_label = 0

            if out_label == labels:
                if out_label == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if out_label == 1:
                    fp += 1
                else:
                    fn += 1

    acc = (tp + tn) / total
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return {'acc': acc, 'precision': precision, 'recall': recall}


def main():
    global logger
    logger = logging.getLogger('bert2class')
    logger.setLevel(logging.DEBUG)
    transformer_logging.set_verbosity_error()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handle_txt = logging.FileHandler('./logging/bert2classification.txt')
    handle_txt.setLevel(logging.DEBUG)
    handle_txt.setFormatter(formatter)
    handle_con = logging.StreamHandler()
    handle_con.setLevel(logging.DEBUG)
    handle_con.setFormatter(formatter)

    logger.addHandler(handle_txt)
    logger.addHandler(handle_con)

    logger.info("测试日志，开始训练")
    # 加载样本数据
    train_text = []
    train_label = []
    with open(r"./sample/train_sample_1230_pure.txt", 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.replace("\n", "")
            train_text.append((line.split("\t")[0], line.split("\t")[1]))
            train_label.append(float(line.split("\t")[2]))

    # 模型参数
    modelpath = r"./basemodel/hfl_chinese_bert_wwm_ext"
    simcesmodel = r'./trained_checkpoints/simcse/simcse_model_tao_0.05_epoch_2'

    tokenizer = BertTokenizer.from_pretrained(modelpath)

    # 训练集
    train = CarbonDataset(train_text, train_label, tokenizer, 128)
    train_loader = DataLoader(train, batch_size=16, shuffle=True)
    # 验证集
    dev_text = []
    dev_label = []
    with open('./sample/eva_sup.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.replace("\n", "")
            dev_text.append((line.split("\t")[0], line.split("\t")[1]))
            dev_label.append(float(line.split("\t")[2]))

    deval = CarbonDataset(dev_text, dev_label, tokenizer, 128)
    deval_loader = DataLoader(deval, batch_size=1, shuffle=True)

    # 初始化模型
    model = Simcse2BertClassification(modelpath, simcesmodel)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    # 优化器
    optim = AdamW(model.parameters(), lr=1e-5)
    # loss_function = torch.nn.BCELoss()
    loss_function = torch.nn.BCEWithLogitsLoss()
    simcse2bert_train(model, 1, train_loader, deval_loader, optim, loss_function, device, 'simcse2bert_tanh_', logger)


if __name__ == '__main__':
    main()
