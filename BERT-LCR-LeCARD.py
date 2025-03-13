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
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCHS = 3
MAX_LEN = 512

# 加载 BERT 分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# 数据集类
class LegalCaseDataset(Dataset):
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
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


# 模型定义
class BertForLCR(nn.Module):
    def __init__(self):
        super(BertForLCR, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 4)  # 4分类 (0, 1, 2, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token 的输出
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# 训练函数
def train_model(model, data_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)


# 验证函数
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

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
