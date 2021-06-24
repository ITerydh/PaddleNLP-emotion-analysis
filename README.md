# PaddleNLP-情感分析
PaddleNLP之七个数据集的情感分析

项目地址已经放在[基于[实践课5-情感分析baseline]优化的情感分析](https://aistudio.baidu.com/aistudio/projectdetail/2104045)

# 自然语言处理之情感分析实现
这里用飞桨的高层API快速搭建模型实现情感分析比赛的结果的提交。具体的原理和分析请参考[『NLP打卡营』实践课5：文本情感分析](https://aistudio.baidu.com/aistudio/projectdetail/1968542?channelType=0&channel=0)。以下将分三部分：句子级情感分析（NLPCC14-SC,ChnSentiCorp）；目标级情感分析（SE-ABSA16_PHNS,SE-ABSA16_CAME）；以及观点抽取（COTE-BD，COTE-DP，COTE-MFW）。

项目的使用非常简单，更改相应章节的data_name，并自己调整batch_size和epochs等以达到最佳的训练效果，并运行相应章节的所有代码即可得到对应数据集的预测结果。所有数据预测完成后，下载submission文件夹提交即可。

## 基于原baseline上的更改点
1. 更改了学习率
2. 更改了epoch
3. 添加了防止过拟合的正则化项
4. 添加了情感分析数据集（去除了原有的datasets，防止保存版本时忘记而丢失数据集，此点与成绩无关）

最终团队七个数据集的项目总分达到了0.81左右

## 建议
1. 跑之前，在左侧窗口**新建一个submission文件夹**，不然后续会报错，遇到就懂了 
2. 每跑一个数据集，重启一次项目，防止显存溢出。
3. 将每个数据集自己新建一个文件进行存储生成的模型，防止所有的数据集的模型在一块混乱。


```python
!pip install --upgrade paddlenlp -i https://pypi.org/simple 
```

    Collecting paddlenlp
    [?25l  Downloading https://files.pythonhosted.org/packages/63/7a/e6098c8794d7753470071f58b07843824c40ddbabe213eae458d321d2dbe/paddlenlp-2.0.3-py3-none-any.whl (451kB)
    [K     |████████████████████████████████| 460kB 26kB/s eta 0:00:012
    [?25hRequirement already satisfied, skipping upgrade: visualdl in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (2.1.1)
    Requirement already satisfied, skipping upgrade: more-itertools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from zipp>=0.5->importlib-metadata; python_version < "3.8"->pre-commit->visualdl->paddlenlp) (7.2.0)
    Installing collected packages: paddlenlp
      Found existing installation: paddlenlp 2.0.1
        Uninstalling paddlenlp-2.0.1:
          Successfully uninstalled paddlenlp-2.0.1
    Successfully installed paddlenlp-2.0.3


## 1. 句子级情感分析
句子级情感分析是针对输入的一段话，判断其感情倾向，一般为积极（1）或消极（0）。

### 1.0 载入模型和Tokenizer


```python
import paddlenlp
from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer
```

### 1.1 数据处理
虽然一些数据集在PaddleNLP已存在，但是为了数据处理上的一致性，这里统一从上传的datasets中处理。对于PaddleNLP已存在的数据集，强烈建议直接用API调用，非常方便。


```python
# 解压数据
!unzip -o data/data95319/ChnSentiCorp
!unzip -o data/data95319/NLPCC14-SC
```

    Archive:  data/data95319/ChnSentiCorp.zip
      inflating: ChnSentiCorp/License.pdf  
      inflating: ChnSentiCorp/dev.tsv    
      inflating: ChnSentiCorp/test.tsv   
      inflating: ChnSentiCorp/train.tsv  
    Archive:  data/data95319/NLPCC14-SC.zip
      inflating: NLPCC14-SC/License.pdf  
      inflating: NLPCC14-SC/test.tsv     
      inflating: NLPCC14-SC/train.tsv    


数据内部结构解析：

```
ChnSentiCorp:

train: 
label		text_a
0		房间太小。其他的都一般。。。。。。。。。
1		轻便，方便携带，性能也不错，能满足平时的工作需要，对出差人员来说非常不错

dev:
qid		label		text_a
0		1		這間酒店環境和服務態度亦算不錯,但房間空間太小~...

test:
qid		text_a
0		这个宾馆比较陈旧了，特价的房间也很一般。总体来说一般
...		...


NLPCC14-SC:

train:
label		text_a
1		请问这机不是有个遥控器的吗？
0		全是大道理啊

test:
qid		text_a
0		我终于找到同道中人啦～～～～从初中开始，我就...
...		...
```

从上可以看出两个数据集可以定义一致的读取方式，但是NLPCC14-SC没有dev数据集，因此不再定义dev数据


```python
# 得到数据集字典
def open_func(file_path):
    return [line.strip() for line in open(file_path, 'r', encoding='utf8').readlines()[1:] if len(line.strip().split('\t')) >= 2]

data_dict = {'chnsenticorp': {'test': open_func('ChnSentiCorp/test.tsv'),
                              'dev': open_func('ChnSentiCorp/dev.tsv'),
                              'train': open_func('ChnSentiCorp/train.tsv')},
             'nlpcc14sc': {'test': open_func('NLPCC14-SC/test.tsv'),
                           'train': open_func('NLPCC14-SC/train.tsv')}}
```

### 1.2 定义数据读取器


```python
# 定义数据集
from paddle.io import Dataset, DataLoader
from paddlenlp.data import Pad, Stack, Tuple
import numpy as np
label_list = [0, 1]

# 注意，由于token type在此项任务中并没有起作用，因此这里不再考虑，让模型自行填充。
class MyDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512, for_test=False):
        super().__init__()
        self._data = data
        self._tokenizer = tokenizer
        self._max_len = max_len
        self._for_test = for_test
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        samples = self._data[idx].split('\t')
        label = samples[-2]
        text = samples[-1]
        label = int(label)
        text = self._tokenizer.encode(text, max_seq_len=self._max_len)['input_ids']
        if self._for_test:
            return np.array(text, dtype='int64')
        else:
            return np.array(text, dtype='int64'), np.array(label, dtype='int64')

def batchify_fn(for_test=False):
    if for_test:
        return lambda samples, fn=Pad(axis=0, pad_val=tokenizer.pad_token_id): np.row_stack([data for data in fn(samples)])
    else:
        return lambda samples, fn=Tuple(Pad(axis=0, pad_val=tokenizer.pad_token_id),
                                        Stack()): [data for data in fn(samples)]


def get_data_loader(data, tokenizer, batch_size=32, max_len=512, for_test=False):
    dataset = MyDataset(data, tokenizer, max_len, for_test)
    shuffle = True if not for_test else False
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=batchify_fn(for_test), shuffle=shuffle)
    return data_loader
```

### 1.3 模型搭建并进行训练
模型非常简单，我们只需要调用对应的序列分类工具就行了。为了方便训练，直接用高层API Model完成训练。


```python
import paddle
from paddle.static import InputSpec

# 模型和分词
model = SkepForSequenceClassification.from_pretrained('skep_ernie_1.0_large_ch', num_classes=2)
tokenizer = SkepTokenizer.from_pretrained('skep_ernie_1.0_large_ch')

# 参数设置
data_name = 'nlpcc14sc'  # 更改此选项改变数据集

## 训练相关
epochs = 10
learning_rate = 5e-6
batch_size = 8
max_len = 512

## 数据相关
train_dataloader = get_data_loader(data_dict[data_name]['train'], tokenizer, batch_size, max_len, for_test=False)
if data_name == 'chnsenticorp':
    dev_dataloader = get_data_loader(data_dict[data_name]['dev'], tokenizer, batch_size, max_len, for_test=False)
else:
    dev_dataloader = None

input = InputSpec((-1, -1), dtype='int64', name='input')
label = InputSpec((-1, 2), dtype='int64', name='label')
model = paddle.Model(model, [input], [label])

# 模型准备
# 加入过拟合 正则化 
optimizer = paddle.optimizer.Adam(learning_rate=learning_rate,weight_decay=paddle.regularizer.L2Decay(coeff=1e-5),parameters=model.parameters())
model.prepare(optimizer, loss=paddle.nn.CrossEntropyLoss(), metrics=[paddle.metric.Accuracy()])
```

    [2021-06-22 12:30:47,799] [    INFO] - Downloading https://paddlenlp.bj.bcebos.com/models/transformers/skep/skep_ernie_1.0_large_ch.pdparams and saved to /home/aistudio/.paddlenlp/models/skep_ernie_1.0_large_ch
    [2021-06-22 12:30:47,859] [    INFO] - Downloading skep_ernie_1.0_large_ch.pdparams from https://paddlenlp.bj.bcebos.com/models/transformers/skep/skep_ernie_1.0_large_ch.pdparams
    100%|██████████| 1238309/1238309 [00:39<00:00, 31211.08it/s]
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1297: UserWarning: Skip loading for classifier.weight. classifier.weight is not found in the provided dict.
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1297: UserWarning: Skip loading for classifier.bias. classifier.bias is not found in the provided dict.
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
    [2021-06-22 12:31:39,146] [    INFO] - Downloading skep_ernie_1.0_large_ch.vocab.txt from https://paddlenlp.bj.bcebos.com/models/transformers/skep/skep_ernie_1.0_large_ch.vocab.txt
    100%|██████████| 55/55 [00:00<00:00, 2940.11it/s]



```python
# 开始训练
model.fit(train_dataloader, dev_dataloader, batch_size, epochs, eval_freq=5, save_freq=5, save_dir='./checkpoints', log_freq=200)
```

    The loss value printed in the log is the current step, and the metric is the average value of previous steps.
    Epoch 1/10


    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/layers/utils.py:77: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
      return (isinstance(seq, collections.Sequence) and


### 1.4 预测并保存


```python
# 导入预训练模型
checkpoint_path = './checkpoints/final'  # 填写预训练模型的保存路径

model = SkepForSequenceClassification.from_pretrained('skep_ernie_1.0_large_ch', num_classes=2)
input = InputSpec((-1, -1), dtype='int64', name='input')
model = paddle.Model(model, input)
model.load(checkpoint_path)

# 导入测试集
test_dataloader = get_data_loader(data_dict[data_name]['test'], tokenizer, batch_size, max_len, for_test=True)
# 预测保存

save_file = {'chnsenticorp': './submission/ChnSentiCorp.tsv', 'nlpcc14sc': './submission/NLPCC14-SC.tsv'}
predicts = []
for batch in test_dataloader:
    predict = model.predict_batch(batch)
    predicts += predict[0].argmax(axis=-1).tolist()

with open(save_file[data_name], 'w', encoding='utf8') as f:
    f.write("index\tprediction\n")
    for idx, sample in enumerate(data_dict[data_name]['test']):
        qid = sample.split('\t')[0]
        f.write(qid + '\t' + str(predicts[idx]) + '\n')
    f.close()
```

## 2. 目标级情感分析
目标级情感分析将对整句的情感倾向扩充为对多个特定属性的情感倾向，本质上仍然是序列分类，但是针对同一个序列需要进行多次分类，针对不同的属性。这里的思路是将针对的属性也作为输入的一部分传入模型，并预测情感倾向。

### 2.0 载入模型和Tokenizer


```python
import paddlenlp
from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer
```

### 2.1 数据处理


```python
# 解压数据
!unzip -o data/data95319/SE-ABSA16_CAME
!unzip -o data/data95319/SE-ABSA16_PHNS
```

数据内部结构解析（两个数据集的结构相同）：
```
train:
label		text_a		text_b
1		phone#design_features		今天有幸拿到了港版白色iPhone 5真机，试玩了一下，说说感受吧：1. 真机尺寸宽度与4/4s保持一致没有变化...
0		software#operation_performance		苹果iPhone5新机到手 对比4S使用感受1，外观。一开始看发布会和网上照片，我和大多数人观点一样：变化不大，有点小失望。...

test:
qid		text_a		text_b
0		software#usability		刚刚入手8600，体会。刚刚从淘宝购买，1635元（包邮）。1、全新，...
...		...		...


```python
# 得到数据集字典
# 得到数据集字典
def open_func(file_path):
    return [line.strip() for line in open(file_path, 'r', encoding='utf8').readlines()[1:] if len(line.strip().split('\t')) >= 2]

data_dict = {'seabsa16phns': {'test': open_func('SE-ABSA16_PHNS/test.tsv'),
                              'train': open_func('SE-ABSA16_PHNS/train.tsv')},
             'seabsa16came': {'test': open_func('SE-ABSA16_CAME/test.tsv'),
                              'train': open_func('SE-ABSA16_CAME/train.tsv')}}
```

### 2.2 定义数据读取器
方法与1.2中相似，基本是完全粘贴复制过来即可。这里注意需要两个text，并且要考虑token_type_id了。


```python
# 定义数据集
from paddle.io import Dataset, DataLoader
from paddlenlp.data import Pad, Stack, Tuple
import numpy as np
label_list = [0, 1]

# 考虑token_type_id
class MyDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512, for_test=False):
        super().__init__()
        self._data = data
        self._tokenizer = tokenizer
        self._max_len = max_len
        self._for_test = for_test
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        samples = self._data[idx].split('\t')
        label = samples[-3]
        text_b = samples[-1]
        text_a = samples[-2]
        label = int(label)
        encoder_out = self._tokenizer.encode(text_a, text_b, max_seq_len=self._max_len)
        text = encoder_out['input_ids']
        token_type = encoder_out['token_type_ids']
        if self._for_test:
            return np.array(text, dtype='int64'), np.array(token_type, dtype='int64')
        else:
            return np.array(text, dtype='int64'), np.array(token_type, dtype='int64'), np.array(label, dtype='int64')

def batchify_fn(for_test=False):
    if for_test:
        return lambda samples, fn=Tuple(Pad(axis=0, pad_val=tokenizer.pad_token_id),
                                        Pad(axis=0, pad_val=tokenizer.pad_token_type_id)): [data for data in fn(samples)]
    else:
        return lambda samples, fn=Tuple(Pad(axis=0, pad_val=tokenizer.pad_token_id),
                                        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
                                        Stack()): [data for data in fn(samples)]


def get_data_loader(data, tokenizer, batch_size=32, max_len=512, for_test=False):
    dataset = MyDataset(data, tokenizer, max_len, for_test)
    shuffle = True if not for_test else False
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=batchify_fn(for_test), shuffle=shuffle)
    return data_loader
```

### 2.3 模型搭建并进行训练
把1.3的复制粘贴过来，注意该数据集名称，并加上一个token_type_id的输入。


```python
import paddle
from paddle.static import InputSpec

# 模型和分词
model = SkepForSequenceClassification.from_pretrained('skep_ernie_1.0_large_ch', num_classes=2)
tokenizer = SkepTokenizer.from_pretrained('skep_ernie_1.0_large_ch')

# 参数设置
data_name = 'seabsa16phns'  # 更改此选项改变数据集

## 训练相关
epochs = 10
learning_rate = 5e-6
batch_size = 8
max_len = 512

## 数据相关
train_dataloader = get_data_loader(data_dict[data_name]['train'], tokenizer, batch_size, max_len, for_test=False)

input = InputSpec((-1, -1), dtype='int64', name='input')
token_type = InputSpec((-1, -1), dtype='int64', name='token_type')
label = InputSpec((-1, 2), dtype='int64', name='label')
model = paddle.Model(model, [input, token_type], [label])

# 模型准备
# 加入过拟合 正则化 
optimizer = paddle.optimizer.Adam(learning_rate=learning_rate,weight_decay=paddle.regularizer.L2Decay(coeff=1e-5),parameters=model.parameters())
model.prepare(optimizer, loss=paddle.nn.CrossEntropyLoss(), metrics=[paddle.metric.Accuracy()])
```


```python
# 开始训练
model.fit(train_dataloader, batch_size=batch_size, epochs=epochs, save_freq=5, save_dir='./checkpoints', log_freq=200)
```

### 2.4 预测并保存


```python
# 导入预训练模型
checkpoint_path = './checkpoints/final'  # 填写预训练模型的保存路径

model = SkepForSequenceClassification.from_pretrained('skep_ernie_1.0_large_ch', num_classes=2)
input = InputSpec((-1, -1), dtype='int64', name='input')
token_type = InputSpec((-1, -1), dtype='int64', name='token_type')
model = paddle.Model(model, [input, token_type])
model.load(checkpoint_path)

# 导入测试集
test_dataloader = get_data_loader(data_dict[data_name]['test'], tokenizer, batch_size, max_len, for_test=True)
# 预测保存

save_file = {'seabsa16phns': './submission/SE-ABSA16_PHNS.tsv', 'seabsa16came': './submission/SE-ABSA16_CAME.tsv'}
predicts = []
for batch in test_dataloader:
    predict = model.predict_batch(batch)
    predicts += predict[0].argmax(axis=-1).tolist()

with open(save_file[data_name], 'w', encoding='utf8') as f:
    f.write("index\tprediction\n")
    for idx, sample in enumerate(data_dict[data_name]['test']):
        qid = sample.split('\t')[0]
        f.write(qid + '\t' + str(predicts[idx]) + '\n')
    f.close()
```

## 3. 观点抽取
### 3.0 载入模型和Tokenizer


```python
import paddlenlp
from paddlenlp.transformers import SkepForTokenClassification, SkepTokenizer
```

### 3.1 数据处理


```python
# 解压数据
!unzip -o data/data95319/COTE-BD
!unzip -o data/data95319/COTE-DP
!unzip -o data/data95319/COTE-MFW
```

数据内部结构解析（三个数据集的结构相同）：
```
train:
label		text_a
鸟人		《鸟人》一书以鸟博士的遭遇作为主线，主要写了鸟博士从校园出来后的种种荒诞经历。
...		...
test:
qid		text_a
0		毕棚沟的风景早有所闻，尤其以秋季的风景最美，但是这次去晚了，红叶全掉完了，黄叶也看不到了，下了雪只...
...		...


```python
# 得到数据集字典
def open_func(file_path):
    return [line.strip() for line in open(file_path, 'r', encoding='utf8').readlines()[1:] if len(line.strip().split('\t')) >= 2]

data_dict = {'cotebd': {'test': open_func('COTE-BD/test.tsv'),
                        'train': open_func('COTE-BD/train.tsv')},
             'cotedp': {'test': open_func('COTE-DP/test.tsv'),
                        'train': open_func('COTE-DP/train.tsv')},
             'cotemfw': {'test': open_func('COTE-MFW/test.tsv'),
                        'train': open_func('COTE-MFW/train.tsv')}}
```

### 3.2 定义数据读取器
思路类似，需要注意的是这一次是Tokens级的分类。在数据读取器中，将label写成BIO的形式，每一个token都对应一个label。


```python
# 定义数据集
from paddle.io import Dataset, DataLoader
from paddlenlp.data import Pad, Stack, Tuple
import numpy as np
label_list = {'B': 0, 'I': 1, 'O': 2}
index2label = {0: 'B', 1: 'I', 2: 'O'}

# 考虑token_type_id
class MyDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=512, for_test=False):
        super().__init__()
        self._data = data
        self._tokenizer = tokenizer
        self._max_len = max_len
        self._for_test = for_test
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        samples = self._data[idx].split('\t')
        label = samples[-2]
        text = samples[-1]
        if self._for_test:
            origin_enc = self._tokenizer.encode(text, max_seq_len=self._max_len)['input_ids']
            return np.array(origin_enc, dtype='int64')
        else:
            
            # 由于并不是每个字都是一个token，这里采用一种简单的处理方法，先编码label，再编码text中除了label以外的词，最后合到一起
            texts = text.split(label)
            label_enc = self._tokenizer.encode(label)['input_ids']
            cls_enc = label_enc[0]
            sep_enc = label_enc[-1]
            label_enc = label_enc[1:-1]
            
            # 合并
            origin_enc = []
            label_ids = []
            for index, text in enumerate(texts):
                text_enc = self._tokenizer.encode(text)['input_ids']
                text_enc = text_enc[1:-1]
                origin_enc += text_enc
                label_ids += [label_list['O']] * len(text_enc)
                if index != len(texts) - 1:
                    origin_enc += label_enc
                    label_ids += [label_list['B']] + [label_list['I']] * (len(label_enc) - 1)

            origin_enc = [cls_enc] + origin_enc + [sep_enc]
            label_ids = [label_list['O']] + label_ids + [label_list['O']]
            
            # 截断
            if len(origin_enc) > self._max_len:
                origin_enc = origin_enc[:self._max_len-1] + origin_enc[-1:]
                label_ids = label_ids[:self._max_len-1] + label_ids[-1:]
            return np.array(origin_enc, dtype='int64'), np.array(label_ids, dtype='int64')


def batchify_fn(for_test=False):
    if for_test:
        return lambda samples, fn=Pad(axis=0, pad_val=tokenizer.pad_token_id): np.row_stack([data for data in fn(samples)])
    else:
        return lambda samples, fn=Tuple(Pad(axis=0, pad_val=tokenizer.pad_token_id),
                                        Pad(axis=0, pad_val=label_list['O'])): [data for data in fn(samples)]


def get_data_loader(data, tokenizer, batch_size=32, max_len=512, for_test=False):
    dataset = MyDataset(data, tokenizer, max_len, for_test)
    shuffle = True if not for_test else False
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=batchify_fn(for_test), shuffle=shuffle)
    return data_loader
```

### 3.3 模型搭建并进行训练
与之前不同的是模型换成了Token分类。由于Accuracy不再适用于Token分类，我们用Perplexity来大致衡量预测的准确度（接近1为最佳）。


```python
import paddle
from paddle.static import InputSpec
from paddlenlp.metrics import Perplexity

# 模型和分词
model = SkepForTokenClassification.from_pretrained('skep_ernie_1.0_large_ch', num_classes=3)
tokenizer = SkepTokenizer.from_pretrained('skep_ernie_1.0_large_ch')

# 参数设置
data_name = 'cotemfw'  # 更改此选项改变数据集

## 训练相关
epochs = 10
learning_rate = 5e-6
batch_size = 8
max_len = 512

## 数据相关
train_dataloader = get_data_loader(data_dict[data_name]['train'], tokenizer, batch_size, max_len, for_test=False)

input = InputSpec((-1, -1), dtype='int64', name='input')
label = InputSpec((-1, -1, 3), dtype='int64', name='label')
model = paddle.Model(model, [input], [label])

# 模型准备
# 加入过拟合 正则化 
optimizer = paddle.optimizer.Adam(learning_rate=learning_rate,weight_decay=paddle.regularizer.L2Decay(coeff=1e-5),parameters=model.parameters())
model.prepare(optimizer, loss=paddle.nn.CrossEntropyLoss(), metrics=[Perplexity()])
```

    [2021-06-21 15:24:20,369] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/skep_ernie_1.0_large_ch/skep_ernie_1.0_large_ch.pdparams
    [2021-06-21 15:24:25,086] [    INFO] - Found /home/aistudio/.paddlenlp/models/skep_ernie_1.0_large_ch/skep_ernie_1.0_large_ch.vocab.txt



```python
from visualdl import LogWriter
log_writer = LogWriter("./log")
# 安装VisualDL
!pip install --upgrade --pre visualdl
```


```python
# 开始训练
model.fit(train_dataloader,
        batch_size=batch_size,
        epochs=epochs,
        save_freq=2,
        save_dir='./checkpoints/cotemfw',
        log_freq=200)
```

### 3.4 预测并保存


```python
# 导入预训练模型
checkpoint_path = './checkpoints/cotemfw/final'  # 填写预训练模型的保存路径

model = SkepForTokenClassification.from_pretrained('skep_ernie_1.0_large_ch', num_classes=3)
input = InputSpec((-1, -1), dtype='int64', name='input')
model = paddle.Model(model, [input])
model.load(checkpoint_path)

# 导入测试集
test_dataloader = get_data_loader(data_dict[data_name]['test'], tokenizer, batch_size, max_len, for_test=True)
# 预测保存

save_file = {'cotebd': './submission/COTE_BD.tsv', 'cotedp': './submission/COTE_DP.tsv', 'cotemfw': './submission/COTE_MFW.tsv'}
predicts = []
input_ids = []
for batch in test_dataloader:
    predict = model.predict_batch(batch)
    predicts += predict[0].argmax(axis=-1).tolist()
    input_ids += batch.numpy().tolist()

# 先找到B所在的位置，即标号为0的位置，然后顺着该位置一直找到所有的I，即标号为1，即为所得。
def find_entity(prediction, input_ids):
    entity = []
    entity_ids = []
    for index, idx in enumerate(prediction):
        if idx == label_list['B']:
            entity_ids = [input_ids[index]]
        elif idx == label_list['I']:
            if entity_ids:
                entity_ids.append(input_ids[index])
        elif idx == label_list['O']:
            if entity_ids:
                entity.append(''.join(tokenizer.convert_ids_to_tokens(entity_ids)))
                entity_ids = []
    return entity

import re

with open(save_file[data_name], 'w', encoding='utf8') as f:
    f.write("index\tprediction\n")
    for idx, sample in enumerate(data_dict[data_name]['test']):
        qid = sample.split('\t')[0]
        entity = find_entity(predicts[idx], input_ids[idx])
        entity = list(set(entity))  # 去重
        entity = [re.sub('##', '', e) for e in entity]  # 去除英文编码时的特殊符号
        entity = [re.sub('[UNK]', '', e) for e in entity]  # 去除未知符号
        f.write(qid + '\t' + '\x01'.join(entity) + '\n')
    f.close()
```

    [2021-06-21 18:41:25,893] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/skep_ernie_1.0_large_ch/skep_ernie_1.0_large_ch.pdparams


# 总结

项目主要还是以[闫佬的项目------实践课5-情感分析Baseline](https://aistudio.baidu.com/aistudio/projectdetail/2085599)为主进行了参数的调整。

基于原baseline上的更改点
1. 更改了学习率
2. 更改了epoch
3. 添加了防止过拟合的正则化项

最终团队七个数据集的项目总分达到了0.81左右

建议
1. 跑之前，在左侧窗口**新建一个submission文件夹**，不然后续会报错，遇到就懂了 
2. 每跑一个数据集，重启一次项目，防止显存溢出。
3. 将每个数据集自己新建一个文件进行存储生成的模型，防止所有的数据集的模型在一块混乱。

我在AI Studio上获得钻石等级，点亮8个徽章，来互关呀~ 

https://aistudio.baidu.com/aistudio/personalcenter/thirdview/643467
