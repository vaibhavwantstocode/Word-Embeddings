import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
from tqdm import tqdm
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
nltk.download('punkt')

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sns.set_theme(style="whitegrid")
plt.style.use('ggplot')
sns.set_palette("husl")
os.makedirs("plots", exist_ok=True)

# Load Brown corpus vocabulary
with open('brown_vocab.txt', 'r', encoding='utf-8') as f:
    brown_vocab = [line.strip() for line in f]

word_to_idx = {word: idx for idx, word in enumerate(brown_vocab)}
VOCAB_SIZE = len(brown_vocab)
PAD_IDX = word_to_idx['<PAD>']
UNK_IDX = word_to_idx['<UNK>']

# Dataset class
class NewsDataset(Dataset):
    def __init__(self, df, max_length=35):
        self.df = df
        self.max_length = max_length

    def __getitem__(self, index):
        text = self.df.iloc[index]['Description']
        label = self.df.iloc[index]['Class Index'] - 1  # 0-based
        tokens = word_tokenize(text.lower())[:self.max_length]
        indices = [word_to_idx.get(token, UNK_IDX) for token in tokens]
        indices += [PAD_IDX] * (self.max_length - len(indices))
        return torch.tensor(indices), torch.tensor(label)

    def __len__(self):
        return len(self.df)

# ELMo Model
class ELMo(nn.Module):
    def __init__(self, hidden_size=150):
        super(ELMo, self).__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, 300)
        self.layer_1 = nn.LSTM(300, hidden_size, bidirectional=True, batch_first=True)
        self.layer_2 = nn.LSTM(2 * hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(2 * hidden_size, VOCAB_SIZE)

    def forward(self, x):
        emb = self.embedding(x)
        out1, _ = self.layer_1(emb)
        out2, _ = self.layer_2(out1)
        return emb, out1, out2

# Load pretrained ELMo and freeze
elmo = ELMo().double().to(DEVICE)
elmo.load_state_dict(torch.load('bilstm.pt', map_location=DEVICE))
for param in elmo.parameters():
    param.requires_grad = False

# Downstream models
class LambdaModel(nn.Module):
    def __init__(self, trainable=True):
        super().__init__()
        self.lambdas = nn.Parameter(torch.ones(3), requires_grad=trainable)
        self.lstm = nn.LSTM(300, 128, batch_first=True)
        self.fc = nn.Linear(128, 4)

    def forward(self, x):
        with torch.no_grad():
            emb, l1, l2 = elmo(x)

        h0 = emb
        h1 = l1
        h2 = l2

        lam_sum = torch.sum(self.lambdas)
        combined = (self.lambdas[0]/lam_sum * h0 +
                    self.lambdas[1]/lam_sum * h1 +
                    self.lambdas[2]/lam_sum * h2)

        _, (hidden, _) = self.lstm(combined.float())
        return self.fc(hidden.squeeze(0))

class LearnableModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.combiner = nn.Sequential(
            nn.Linear(900, 300),
            nn.ReLU(),
            nn.Linear(300, 300)
        )
        self.lstm = nn.LSTM(300, 128, batch_first=True)
        self.fc = nn.Linear(128, 4)

    def forward(self, x):
        with torch.no_grad():
            emb, l1, l2 = elmo(x)

        combined = torch.cat([emb, l1, l2], dim=-1).float()
        transformed = self.combiner(combined)
        _, (hidden, _) = self.lstm(transformed)
        return self.fc(hidden.squeeze(0))

def plot_training_curve(train_losses, val_accs, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title(f'{model_name} Training Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.savefig(f'plots/{model_name}_training_curve.png', bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f'{model_name} Confusion Matrix')
    plt.savefig(f'plots/{model_name}_confusion_matrix.png', bbox_inches='tight')
    plt.close()

def train_model(model, train_loader, test_loader, model_name, epochs=10, lr=0.001):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    val_accs = []
    best_acc = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Evaluate on the test set (validation)
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1).cpu()
                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        val_accs.append(acc)

        # Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f'best_{model_name}.pt')
            plot_confusion_matrix(all_labels, all_preds, f"{model_name}_best")

        print(f'Epoch {epoch+1}: Loss: {avg_loss:.4f}, Acc: {acc:.4f}')

    # Plot the training curve
    plot_training_curve(train_losses, val_accs, model_name)

    # ---------------- NEW: Final Train-set evaluation ----------------
    train_preds = []
    train_labels_all = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu()
            train_preds.extend(preds.numpy())
            train_labels_all.extend(labels.numpy())

    print(f"\nTrain Classification Report for {model_name}:")
    print(classification_report(train_labels_all, train_preds, target_names=[
        'World', 'Sports', 'Business', 'Sci/Tech']))
    # -----------------------------------------------------------------

    # ---------------- Existing: Final Test-set evaluation ------------
    test_preds = []
    test_labels_all = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu()
            test_preds.extend(preds.numpy())
            test_labels_all.extend(labels.numpy())

    print(f"\nTest Classification Report for {model_name}:")
    print(classification_report(test_labels_all, test_preds, target_names=[
        'World', 'Sports', 'Business', 'Sci/Tech']))
    # -----------------------------------------------------------------

    return best_acc

if __name__ == "__main__":
    # Load data
    train_df = pd.read_csv('../data/train.csv')
    test_df = pd.read_csv('../data/test.csv')

    # Create datasets/loaders
    train_dataset = NewsDataset(train_df)
    test_dataset = NewsDataset(test_df)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Train and evaluate models
    models = {
        'trainable_lambda': LambdaModel(trainable=True),
        'frozen_lambda': LambdaModel(trainable=False),
        'learnable_func': LearnableModel()
    }

    results = {}
    for name, model in models.items():
        print(f'\n{"="*40}\nTraining {name.replace("_", " ").title()}\n{"="*40}')
        acc = train_model(model, train_loader, test_loader, name)
        results[name] = acc
        print(f'{name} Best Accuracy: {acc:.4f}')

    # Generate comparison plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(results.keys()), y=list(results.values()))
    plt.title('Model Comparison - Test Accuracy')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.savefig('plots/model_comparison.png')
    plt.close()

    print('\nFinal Results:')
    for name, acc in results.items():
        print(f'{name.replace("_", " ").title():<25} {acc:.4f}')