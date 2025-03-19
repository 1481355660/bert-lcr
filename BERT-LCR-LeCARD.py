'''以下是一个基于 BERT 的实现，用于处理上述数据格式并完成 LCR（Legal Case Retrieval，法律案件检索）任务。
我们将使用 BERT 模型对查询案件和候选案件进行编码，并根据相关指数（0、1、2、3）进行训练。'''
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import json

json_path = "Datasets/output_case_texts.json"
case_texts = {}

# 示例数据
try:
    with open(json_path, "r", encoding="utf-8") as file:
        # 加载 JSON 文件为字典
        data = json.load(file)

        # 如果 data 是字符串，尝试将其解析为字典
        if isinstance(data, str):
            print("检测到 JSON 文件内容为字符串形式，尝试解析为字典...")
            data = json.loads(data)  # 将字符串形式的 JSON 解析为字典

        # 确保 data 是字典
        if not isinstance(data, dict):
            raise ValueError("JSON 文件内容不是字典！")

        # 将字典内容赋值给 case_texts
        case_texts.update(data)
        print("case_texts 已更新:")
        print(json.dumps(case_texts, ensure_ascii=False, indent=4))

except FileNotFoundError:
    print(f"文件 {json_path} 未找到，请检查路径是否正确！")
except json.JSONDecodeError as e:
    print(f"JSON 文件解析错误: {e}")
except ValueError as e:
    print(f"数据格式错误: {e}")

# 模拟案件文本（需要替换为实际案件文本）
# case_texts = {
#     "5156": "案件5156的详细描述",
#     "38633": "案件38633的详细描述",
#     "38632": "案件38632的详细描述",
#     "32518": "案件32518的详细描述",
#     "4891": "案件4891的详细描述",
#     "24048": "案件24048的详细描述",
#     "412": "案件412的详细描述",
#     "30682": "案件30682的详细描述",
#     "5187": "案件5187的详细描述",
#     "43487": "案件43487的详细描述",
#     "22069": "案件22069的详细描述",
#     "41975": "案件41975的详细描述",
# }

# 超参数
BATCH_SIZE = 8 #把庞大的数据集拆分成一个一个迷你数据集（batch），这里表示每个批次会从数据集中调8个样本
LEARNING_RATE = 2e-5#学习率用来控制模型在每次参数更新时的步伐（更新幅度），这个参数的大小决定了模型学习的速率
EPOCHS = 3
MAX_LEN = 512

# 加载 BERT 分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# 数据集类
class LegalCaseDataset(Dataset):#这个类是用来把数据集变成{inputs_ids,attention_mask,label}的形式
    def __init__(self, data, case_texts, tokenizer, max_len):
        self.data = data
        self.case_texts = case_texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pairs = []
        self.labels = []

        # 构造 (query, candidate, label) 数据对
        for query_id, candidates in data.items():
            query_text = case_texts[query_id]
            for candidate_id, label in candidates.items():
                candidate_text = case_texts[candidate_id]
                self.pairs.append((query_text, candidate_text))
                self.labels.append(label)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        query_text, candidate_text = self.pairs[index]
        label = self.labels[index]

        # 对查询和候选案件进行编码
        encoding = self.tokenizer(
            query_text,
            candidate_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )#使用 BERT 的分词器将查询案件文本和候选案件文本拼接起来，并将其转换为模型可接受的输入格式

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }
    #input_ids:输入的语言经过BERT分词器编码后的结果
    #attention_mask：注意力掩码，用来指示哪些是重要信息哪些是可以忽略的填充信息
    #label:真实的label（来自数据），预测值会与这个label进行比较


# 模型定义
class BertForLCR(nn.Module):
    def __init__(self):
        super(BertForLCR, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)#在训练时随机抽取30%神经元，将其输出置为0，防止过拟合
        self.classifier = nn.Linear(self.bert.config.hidden_size, 4)  # 4分类 (0, 1, 2, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  #  BERT 对输入序列开头的 [CLS] 标记进行编码后，生成的一个向量，这个向量代表了整句话的语义信息。pooler_output 就是这个向量。这样的行为提高了效率，下游任务不必再一个一个token进行读取，而是可以直接读取句子
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits#logits 是分类器输出的原始分数，每个值对应一个类别。简单来说就是已经分类了，但是不是以概率的形式分类的
                        #它们还没有转化为概率（未归一化），但可以通过 softmax 转化为概率，用于预测最终的类别


# 训练函数
def train_model(model, data_loader, optimizer, criterion, device):
    model.train()#开启训练模式
    total_loss = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        #将数据移动到指定设备
        optimizer.zero_grad()
        #清除模型参数中之前计算的梯度，如果不这么做，每一批次都会产生新的梯度加到这个梯度上，导致梯度计算错误，梯度可以理解为指路的方向
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        #前向传播，outputs是模型预测结果（logits或概率分布）
        loss = criterion(outputs, labels)
        #使用损失函数 criterion 计算模型预测值（outputs）与真实标签（labels）之间的误差
        loss.backward()
        #反向传播
        optimizer.step()
        #更新模型参数

        total_loss += loss.item()
        #计算累计损失
    return total_loss / len(data_loader)
        #返回平均损失

# 验证函数
def evaluate_model(model, data_loader, criterion, device):
    model.eval()#转为评估模式
    total_loss = 0#所有批次的损失值
    correct = 0#模型预测正确的样本数
    total = 0#总样本数

    with torch.no_grad():#禁用梯度计算，因为在评估阶段不需要进行反向传播，不进行反向传播是因为此阶段不需要再调整模型的参数，只需要评估性能
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)#对于每个样本（行），我们在所有类别（列）中找到分数最大的类别，以此找到每个样本的预测类别
            correct += (preds == labels).sum().item()#统计正确预测的样本数
            total += labels.size(0)#累计当前批次中的样本数量

    accuracy = correct / total
    return total_loss / len(data_loader), accuracy


# 主程序
def main():
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 data.json 文件
    data_path = "Datasets/LeCARD/label_top30_dict.json"  # 你的 data 文件路径
    try:
        with open(data_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            print("data 已加载:")
            print(json.dumps(data, ensure_ascii=False, indent=4))
    except FileNotFoundError:
        print(f"文件 {data_path} 未找到，请检查路径是否正确！")
        return
    except json.JSONDecodeError as e:
        print(f"JSON 文件解析错误: {e}")
        return

    # 数据集和数据加载器
    dataset = LegalCaseDataset(data, case_texts, tokenizer, MAX_LEN)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 模型、损失函数和优化器
    model = BertForLCR().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 训练和验证
    for epoch in range(EPOCHS):
        train_loss = train_model(model, data_loader, optimizer, criterion, device)
        val_loss, val_accuracy = evaluate_model(model, data_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "bert_lcr_model_LeCARD.pth")
    print("Model saved to bert_lcr_model.pth")


if __name__ == "__main__":
    main()
