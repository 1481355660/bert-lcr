import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. 数据集加载和预处理
class TextMatchDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text_a = item["A"]
        text_b = item["B"]
        text_c = item["C"]
        label = 0 if item["label"] == "B" else 1  # B -> 0, C -> 1

        # Tokenize A-B and A-C pairs
        inputs_ab = self.tokenizer(
            text_a, text_b, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )
        inputs_ac = self.tokenizer(
            text_a, text_c, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt"
        )

        return {
            "inputs_ab": {key: val.squeeze(0) for key, val in inputs_ab.items()},
            "inputs_ac": {key: val.squeeze(0) for key, val in inputs_ac.items()},
            "label": torch.tensor(label, dtype=torch.long),
        }

# 2. 数据加载函数
def load_data(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                # 逐行解析 JSON 对象
                value = json.loads(line.strip())
                data.append({
                    "A": value["A"],
                    "B": value["B"],
                    "C": value["C"],
                    "label": value.get("label", "B")  # 默认标签为 B
                })
            except json.JSONDecodeError as e:
                print(f"JSONDecodeError: {e} in line: {line}")
    return data


# 3. 保存模型函数
def save_model(model, tokenizer, save_path):
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"模型已保存到 {save_path}")

# 4. 模型训练和评估
def train_model(model, tokenizer, train_loader, val_loader, optimizer, device, epochs=3, save_path="saved_model"):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            # Forward pass for A-B and A-C pairs
            outputs_ab = model(**batch["inputs_ab"], labels=batch["label"])
            outputs_ac = model(**batch["inputs_ac"], labels=batch["label"])

            # Combine losses
            loss = outputs_ab.loss + outputs_ac.loss
            total_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    # 保存模型
    save_model(model, tokenizer, save_path)

    # Validation
    evaluate_model(model, val_loader, device)

def evaluate_model(model, val_loader, device):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            outputs_ab = model(**batch["inputs_ab"])
            outputs_ac = model(**batch["inputs_ac"])

            # Compare logits of A-B and A-C to decide prediction
            logits_ab = outputs_ab.logits[:, 1]  # Take positive class logits
            logits_ac = outputs_ac.logits[:, 1]
            preds = (logits_ac > logits_ab).long()  # If A-C similarity > A-B, predict C (1), else B (0)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(batch["label"].cpu().numpy())

    print(classification_report(true_labels, predictions, target_names=["B", "C"]))

# 5. 主函数
def main():
    # 文件路径
    file_path = "Datasets/CAIL2019-SCM/SCM_5k.json"  # 替换为你的数据集路径

    # 加载数据
    raw_data = load_data(file_path)

    # 划分训练集和验证集
    train_data, val_data = train_test_split(raw_data, test_size=0.2, random_state=42)

    # 加载 BERT 分词器和模型
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    # 创建数据集和数据加载器
    train_dataset = TextMatchDataset(train_data, tokenizer)
    val_dataset = TextMatchDataset(val_data, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # 设置设备和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # 训练模型
    train_model(model, tokenizer, train_loader, val_loader, optimizer, device, save_path="bcr_model——CAIL2019-SCM")

if __name__ == "__main__":
    main()

