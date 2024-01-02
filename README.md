【预训练模型】采用BERT模型，基于文本相似度，实现用户输入与绿色产业指导目录三级条目匹配。可根据场景内标注数据继续微调模型

***

### **模型结构**

以bert模型为骨架，模型结构：SimCSE -> Cross-Encoder -> Tanh

### &#x20;**训练样本**

&#x9;**1. SimCSE**

&#x9;	对比学习拉开文本向量空间

*   &#x9;	训练样本：x={'三级目录标题', '清洗后的三级目录解释说明'}

&#x9;**2. Bert Cross Encoder**

*   &#x9;	label = {0, 1}

<!---->

*   &#x9;	正向样本 (len=1)

&#x9;			x\_postive = ('三级目录标题', '清洗后的三级目录解释说明')

*   &#x9;	负向样本 (len=210)

&#x9;			包含所有其他三级目录 x\_negative = ('三级目录标题', '清洗后的三级目录解释说明')

*   &#x9;	平衡正样本 (len=209)

    &#x9;		x\_generate = ('生成相似三级目录标题'， '清洗后的三级目录解释说明')



***

### **模型推理**

```python
import pandas as pd

from pred import modelPipe
from tqdm import tqdm

basemodel = './basemodel/hfl_chinese_bert_wwm_ext'
simcesmodel = './trained_checkpoints/simcse/simcse_model_tao_pure_0.1_epoch_0'
trainedmodel = './trained_checkpoints/simcse_bert/simcse2bert_pure_tanh_0'

model = modelPipe(basemodel, simcesmodel, trainedmodel)

# 返回相似条目排序
raw = pd.read_excel('./sample/green_industry_sim.xlsx')

text1 = "新能源汽车电池拆解回收"
result = pd.DataFrame()

for i in tqdm(range(raw.shape[0])):
    text2 = raw.iloc[i, 2]
    _theme = raw.iloc[i, 0]
    _detail = raw.iloc[i, 1]
    _score = model.pred1(text1, text2).item()

    _d = pd.DataFrame({
        'theme': [_theme],
        'detail': [_detail],
        'score': [_score]
    })
    result = pd.concat([result, _d])

result.sort_values(by='score', ascending=False, inplace=True)
result.reset_index(inplace=True, drop=True)
result
```

***

### **参考**
SimCSE
https://github.com/KwangKa/SIMCSE_unsup

[junnyu/roformer\_chinese\_sim\_char\_base · Hugging Face](https://huggingface.co/junnyu/roformer_chinese_sim_char_base)

@misc{su2021roformer,
title={RoFormer: Enhanced Transformer with Rotary Position Embedding},
author={Jianlin Su and Yu Lu and Shengfeng Pan and Bo Wen and Yunfeng Liu},
year={2021},
eprint={2104.09864},
archivePrefix={arXiv},
primaryClass={cs.CL}
}

[hfl/chinese-bert-wwm-ext · Hugging Face](https://huggingface.co/hfl/chinese-bert-wwm-ext)

@article{chinese-bert-wwm,
title={Pre-Training with Whole Word Masking for Chinese BERT},
author={Cui, Yiming and Che, Wanxiang and Liu, Ting and Qin, Bing and Yang, Ziqing and Wang, Shijin and Hu, Guoping},
journal={arXiv preprint arXiv:1906.08101},
year={2019}
}
