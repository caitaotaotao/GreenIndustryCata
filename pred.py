import torch
from bert_cross_encoder_trainning import Simcse2BertClassification
from transformers import BertTokenizer


class modelPipe:
    def __init__(self, basemodel, simcesmodel, trainedmodel):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = Simcse2BertClassification(basemodel, simcesmodel)
        self.tokenizer = BertTokenizer.from_pretrained(basemodel)
        self.model.load_state_dict(torch.load(trainedmodel)['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def pred1(self, text1, text2):
        with torch.no_grad():
            embedding = self.tokenizer(text1, text2, padding=True, max_length=128, return_tensors='pt')
            embedding.to(self.device)
            out = self.model(**embedding)
        return out
