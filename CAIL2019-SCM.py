import gc
import torch
import time
import copy
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import warnings

from random import shuffle
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import json
from transformers import AutoModel, AutoTokenizer, BertModel

warnings.filterwarnings('ignore')

from torch.optim import lr_scheduler

gc.collect()
torch.cuda.empty_cache()


#########################
# config
#########################
class Config():#这个类用于储存各种参数
    # 路径配置
    train_path = '/Users/MacBook/PycharmProjects/BERT/CAIL-2019/CAIL-2019-SCM-data/train.json'
    eval_path = '/Users/MacBook/PycharmProjects/BERT/CAIL-2019/CAIL-2019-SCM-data/valid.json'

    # 通用配置
    plm_path = 'bert-base-chinese'
    tokenizer_path = 'bert-base-chinese'
    max_length = 100
    batch_size = 4
    epoch = 10
    learning_rate = 2e-5
    weight_decay = 2e-6#权重衰减系数，通过对权重施加惩罚来限制模型复杂度，和dropout作用相同，为了防止过拟合
    schedule = 'CosineAnnealingLR'#余弦退火学习度调度，让学习率按照余弦函数的形状逐渐减少，帮助模型更好收敛

    # # RCNN 相关配置
    # rnn_hidden = 128  # LSTM 隐藏层维度，隐藏层维度越大，模型的表达能力越强，可以捕获更复杂的上下文关系，但同时会增加计算开销
    # num_layers = 1  # LSTM 层数
    # dropout = 0.1  # dropout 概率
    embedding_dim = 128  # （全连接层输出），全连接层将输入转换成我们需要的维度的向量（128这个维度有利于相似性计算）


#########################
# data_loader
#########################

def read_json(path):
    data = []
    with open(path, encoding='utf-8') as f:
        for line in f:  # 逐行读取
            if not line.strip():
                continue
            data.append(json.loads(line))
    shuffle(data)#随机打乱数据的顺序
    return data


class MyDataset(Dataset):#将数据集变为getitem返回值的格式
    def __init__(self, data, Config):
        tokenizer = AutoTokenizer.from_pretrained(Config.tokenizer_path)
        self.inputs_A = tokenizer([x['A'] for x in data],
                                  truncation=True, max_length=Config.max_length,
                                  padding='max_length', return_tensors='pt')
        self.inputs_B = tokenizer([x['B'] for x in data],
                                  truncation=True, max_length=Config.max_length,
                                  padding='max_length', return_tensors='pt')
        self.inputs_C = tokenizer([x['C'] for x in data],
                                  truncation=True, max_length=Config.max_length,
                                  padding='max_length', return_tensors='pt')
        self.labels = torch.tensor([0 if x['label'] == 'B' else 1 for x in data], dtype=torch.long)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inputs_A = self.inputs_A['input_ids'][idx]
        attention_mask_A = self.inputs_A['attention_mask'][idx]
        inputs_B = self.inputs_B['input_ids'][idx]
        attention_mask_B = self.inputs_B['attention_mask'][idx]
        inputs_C = self.inputs_C['input_ids'][idx]
        attention_mask_C = self.inputs_C['attention_mask'][idx]
        label = self.labels[idx]
        return {'inputs_A': inputs_A, 'attention_mask_A': attention_mask_A,
                'inputs_B': inputs_B, 'attention_mask_B': attention_mask_B,
                'inputs_C': inputs_C, 'attention_mask_C': attention_mask_C,
                'label': label}
    # input_ids:输入的语言经过BERT分词器编码后的结果
    # attention_mask：注意力掩码，用来指示哪些是重要信息哪些是可以忽略的填充信息
    # label:真实的label（来自数据），预测值会与这个label进行比较


def get_train_eval_DataLoader(Config):#将数据变成数据加载器
    train = read_json(Config.train_path)
    eval = read_json(Config.eval_path)
    train_dl = DataLoader(MyDataset(train, Config), batch_size=Config.batch_size, shuffle=True, pin_memory=True)
    print('训练数据加载完成')
    eval_dl = DataLoader(MyDataset(eval, Config), batch_size=Config.batch_size, shuffle=True, pin_memory=True)
    print('开发验证数据加载完成')
    return train_dl, eval_dl


#########################
# strategy
#########################

# 选择学习率调整策略 1.（默认）余弦模拟退火 2.余弦模拟退火热重启
def fetch_scheduler(optimizer, schedule='CosineAnnealingLR'):
    if schedule == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6)
    elif schedule == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, eta_min=1e-6)
    elif schedule is None:
        return None
    return scheduler

#设置随机种子，确保在相同的代码、数据和硬件环境下，模型训练过程是可重复的
def set_seed(seed=42):#固定随机数生成器的初始状态，确保每次运行程序时生成的随机数序列一致
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#########################
# Bert
#########################

class MyModel(nn.Module):
    def __init__(self, Config):
        super(MyModel, self).__init__()
        self.bert = AutoModel.from_pretrained(Config.plm_path)
        self.fc = nn.Linear(768, Config.embedding_dim)#全连接层，将输出的向量转换成我们需要的维度

    def forward(self, inputs_A, attention_mask_A, inputs_B, attention_mask_B, inputs_C, attention_mask_C):
        emb_A = self._get_embedding(inputs_A, attention_mask_A)#获取ABC的向量表示
        emb_B = self._get_embedding(inputs_B, attention_mask_B)
        emb_C = self._get_embedding(inputs_C, attention_mask_C)

        sim_AB = torch.cosine_similarity(emb_A, emb_B, dim=1)#计算AB的余弦相似度
        sim_AC = torch.cosine_similarity(emb_A, emb_C, dim=1)
        return sim_AB, sim_AC

    def _get_embedding(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)#Bert将输入的自然语言转换成向量
        cls_embedding = output.last_hidden_state[:, 0, :]#提取 [CLS] 标记的嵌入表示，使用 [CLS] 嵌入作为句子的语义
        cls_embedding = self.fc(cls_embedding)#降到我们需要的维度
        return cls_embedding



#########################
# train & valid
#########################

def train_one_epoch(model, optimizer, scheduler, criterion, train_loader, device, epoch):
    model.train()
    num_examples = 0
    total_loss = 0.0
    total_correct = 0

    bar = tqdm(enumerate(train_loader), total=len(train_loader))
    for i, batch in bar:
        inputs_A = batch['inputs_A'].to(device)
        inputs_B = batch['inputs_B'].to(device)
        inputs_C = batch['inputs_C'].to(device)
        attention_mask_A = batch['attention_mask_A'].to(device)
        attention_mask_B = batch['attention_mask_B'].to(device)
        attention_mask_C = batch['attention_mask_C'].to(device)
        labels = batch['label'].to(device)

        sim_AB, sim_AC = model(inputs_A, attention_mask_A, inputs_B, attention_mask_B, inputs_C, attention_mask_C)
        logits = sim_AC - sim_AB
        loss = criterion(logits, labels.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = (logits > 0).long()  # 当sim_AC > sim_AB时预测为 1
        correct = (preds == labels).sum().item()
        total_correct += correct
        num_examples += labels.size(0)
        total_loss += loss.item() * labels.size(0)

        avg_loss = total_loss / num_examples
        accuracy = total_correct / num_examples
        bar.set_postfix(epoch=epoch, train_loss=avg_loss, train_accuracy=accuracy)#更新进度条信息

    if scheduler is not None:#学习率调整
        scheduler.step()
    return avg_loss, accuracy


@torch.no_grad()
def valid_one_epoch(model, criterion, valid_dl, device, epoch):
    model.eval()
    num_examples = 0
    total_correct = 0
    total_loss = 0.0

    bar = tqdm(enumerate(valid_dl), total=len(valid_dl))
    for i, batch in bar:
        inputs_A = batch['inputs_A'].to(device)
        inputs_B = batch['inputs_B'].to(device)
        inputs_C = batch['inputs_C'].to(device)
        attention_mask_A = batch['attention_mask_A'].to(device)
        attention_mask_B = batch['attention_mask_B'].to(device)
        attention_mask_C = batch['attention_mask_C'].to(device)
        labels = batch['label'].to(device)

        sim_AB, sim_AC = model(inputs_A, attention_mask_A, inputs_B, attention_mask_B, inputs_C, attention_mask_C)
        logits = sim_AC - sim_AB
        loss = criterion(logits, labels.float())

        preds = (logits > 0).long()
        correct = (preds == labels).sum().item()
        total_correct += correct
        num_examples += labels.size(0)
        total_loss += loss.item() * labels.size(0)

        avg_loss = total_loss / num_examples
        accuracy = total_correct / num_examples
        bar.set_postfix(epoch=epoch, valid_loss=avg_loss, valid_accuracy=accuracy)

    return avg_loss, accuracy


#########################
# 主程序
#########################

set_seed(2025)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = MyModel(Config)
optimizer = AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
criterion = nn.BCEWithLogitsLoss()
scheduler = fetch_scheduler(optimizer=optimizer, schedule=Config.schedule)
train_dl, valid_dl = get_train_eval_DataLoader(Config)

model.to(device)
best_model_state = copy.deepcopy(model.state_dict())#初始化最佳模型状态
best_valid_loss = np.inf
best_valid_accuracy = 0.0

start_time = time.time()

for epoch in range(1, Config.epoch + 1):
    train_loss, train_accuracy = train_one_epoch(model, optimizer, scheduler, criterion, train_dl, device, epoch)
    valid_loss, valid_accuracy = valid_one_epoch(model, criterion, valid_dl, device, epoch)
    if valid_loss <= best_valid_loss:
        print(f'best valid loss has improved ({best_valid_loss}---->{valid_loss})')
        best_valid_loss = valid_loss
        best_valid_accuracy = valid_accuracy
        best_model_state = copy.deepcopy(model.state_dict())
        torch.save(best_model_state, 'CAIL-2019')
        print('A new best model state has saved')

end_time = time.time()
print('Training Finish !!!!!!!!')
print(f'best valid loss == {best_valid_loss}, best valid accuracy == {best_valid_accuracy}')
time_cost = end_time - start_time
print(f'training cost time == {time_cost}s')
