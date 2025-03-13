import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, f1_score


# Step 1: Load and preprocess the data
class LegalCaseDataset(Dataset):
    def __init__(self, qrel_file, current_cases_dir, prior_cases_dir, tokenizer, max_length=512):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load documents
        self.current_cases = self.load_documents(current_cases_dir)
        self.prior_cases = self.load_documents(prior_cases_dir)

        # Validate and load qrel file
        self.validate_and_load_qrel(qrel_file)

    def load_documents(self, directory):
        """
        Load all .txt files in the given directory into a dictionary.
        The key is the file name (without extension), and the value is the file content.
        """
        documents = {}
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):  # Only process .txt files
                file_id = filename[:-4]  # Remove the .txt extension
                with open(os.path.join(directory, filename), 'r', encoding='latin-1') as file:
                    documents[file_id] = file.read()
        return documents

    def validate_and_load_qrel(self, qrel_file):
        """
        Validate that all query and prior case IDs in the qrel file exist in their respective folders.
        If a file is missing, raise an exception.
        """
        with open(qrel_file, 'r') as file:
            for line in file:
                query_id, _, prior_id, label = line.strip().split()
                # Check if query_id exists in current_cases
                if query_id not in self.current_cases:
                    raise FileNotFoundError(f"Query document '{query_id}.txt' not found in Current Cases folder.")
                # Check if prior_id exists in prior_cases
                if prior_id not in self.prior_cases:
                    raise FileNotFoundError(f"Prior document '{prior_id}.txt' not found in Prior Cases folder.")
                # If both files exist, add the data
                query_text = self.current_cases[query_id]
                prior_text = self.prior_cases[prior_id]
                label = int(label)
                self.data.append((query_text, prior_text, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        query_text, prior_text, label = self.data[idx]
        encoded = self.tokenizer.encode_plus(
            query_text,
            prior_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


# Step 2: Initialize tokenizer and dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_dataset = LegalCaseDataset(
    qrel_file='/Users/MacBook/PycharmProjects/BERT/Datasets/Fire/FIRE2017-IRLeD-track-data/FIRE2017-IRLeD-track-data/Task_2/irled-qrel.txt',  # Replace with your training qrel file
    current_cases_dir='/Users/MacBook/PycharmProjects/BERT/Datasets/Fire/FIRE2017-IRLeD-track-data/FIRE2017-IRLeD-track-data/Task_2/Current_Cases',
    prior_cases_dir='/Users/MacBook/PycharmProjects/BERT/Datasets/Fire/FIRE2017-IRLeD-track-data/FIRE2017-IRLeD-track-data/Task_2/Prior_Cases',
    tokenizer=tokenizer
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Step 3: Initialize the BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Step 4: Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=2e-5)

# Step 5: Training loop
model.train()
for epoch in range(3):  # Number of epochs
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

# Step 6: Save the model
model.save_pretrained('bert_lcr_model')
tokenizer.save_pretrained('bert_lcr_model')


# Step 7: Evaluation
def evaluate(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    return accuracy, f1


# Example evaluation
test_dataset = LegalCaseDataset(
    qrel_file='/Users/MacBook/PycharmProjects/BERT/Datasets/Fire/FIRE2017-IRLeD-track-data/FIRE2017-IRLeD-track-data/Task_2/irled-qrel.txt',  # Replace with your test qrel file
    current_cases_dir='/Users/MacBook/PycharmProjects/BERT/Datasets/Fire/FIRE2017-IRLeD-track-data/FIRE2017-IRLeD-track-data/Task_2/Current_Cases',
    prior_cases_dir='/Users/MacBook/PycharmProjects/BERT/Datasets/Fire/FIRE2017-IRLeD-track-data/FIRE2017-IRLeD-track-data/Task_2/Prior_Cases',
    tokenizer=tokenizer
)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
accuracy, f1 = evaluate(model, test_loader)
print(f"Accuracy: {accuracy}, F1 Score: {f1}")