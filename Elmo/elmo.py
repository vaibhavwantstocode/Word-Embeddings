import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
import nltk
from tqdm import tqdm
import numpy as np
from collections import Counter

# Download NLTK data
nltk.download('brown')
nltk.download('punkt')

# Determine device (GPU if available, otherwise CPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# Preprocessing function for Brown Corpus
def preprocess_text(text):
    """Tokenize and clean a list of words from a sentence."""
    return [word.lower() for word in text if word.isalnum()]

# Build vocabulary manually
def build_vocab_manual(sentences, min_freq=5):
    """Build vocabulary from sentences with minimum frequency."""
    # Flatten sentences into a list of words
    total_words = [word for sent in sentences for word in preprocess_text(sent)]
    
    # Count word frequencies
    word_freq = Counter(total_words)
    
    # Filter words by minimum frequency and create vocab list
    vocab_list = ['<UNK>', '<PAD>'] + [word for word, freq in word_freq.items() if freq >= min_freq]
    
    # Create word-to-index mapping
    word_to_idx = {word: idx for idx, word in enumerate(vocab_list)}
    with open('brown_vocab.txt', 'w', encoding='utf-8') as f:
        for word in vocab_list:
            f.write(word + '\n')
    
    return word_to_idx, vocab_list

# Load GloVe embeddings manually
def load_glove_manual(glove_file, word_to_idx, embedding_dim=300):
    """Load GloVe embeddings into a matrix based on the vocabulary."""
    embeddings = np.zeros((len(word_to_idx), embedding_dim))
    
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading GloVe", total=400000):  # GloVe 6B has ~400k lines
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float64)
            if word in word_to_idx:
                embeddings[word_to_idx[word]] = vector
    
    return torch.tensor(embeddings, dtype=torch.float64)

# Dataset for Brown Corpus
class BrownDataset(Dataset):
    def __init__(self, word_to_idx, max_length=35):
        self.word_to_idx = word_to_idx
        self.sentences = [preprocess_text(sent) for sent in brown.sents()]
        self.max_length = max_length

    def __getitem__(self, index):
        tokens = self.sentences[index]
        # Convert tokens to indices, truncate or pad to max_length
        token_indices = [self.word_to_idx.get(token, self.word_to_idx['<UNK>']) 
                         for token in tokens[:self.max_length]]
        token_indices += [self.word_to_idx['<PAD>']] * (self.max_length - len(token_indices))
        return torch.tensor(token_indices, dtype=torch.long)

    def __len__(self):
        return len(self.sentences)

# ELMo Model
class ELMo(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, dropout, embeddings):
        super(ELMo, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)  # Freeze GloVe embeddings
        self.hidden_size = hidden_size

        # First Bi-LSTM layer
        self.layer_1 = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        # Second Bi-LSTM layer
        self.layer_2 = nn.LSTM(
            input_size=2 * hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

        # Linear layer for prediction (vocab_size output)
        self.linear = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        embeddings = self.embedding(x)  # [batch_size, seq_len, embedding_size]
        lstm1_output, _ = self.layer_1(embeddings)  # [batch_size, seq_len, hidden_size * 2]
        lstm2_output, _ = self.layer_2(lstm1_output)  # [batch_size, seq_len, hidden_size * 2]
        lstm2_output = self.dropout(lstm2_output)
        output = self.linear(lstm2_output)  # [batch_size, seq_len, vocab_size]
        output = torch.transpose(output, 1, 2)  # [batch_size, vocab_size, seq_len]
        return output

# Hyperparameters
BATCH_SIZE = 128
EMBEDDING_SIZE = 300
HIDDEN_SIZE = 150
DROPOUT = 0.1
LEARNING_RATE = 0.001
EPOCHS = 5 # Increase for better training (e.g., 5)
MAX_LENGTH = 50
GLOVE_FILE = '.vector_cache/glove.6B.300d.txt'  # Adjust path to your GloVe file

# Build vocabulary and dataset
print("Building vocabulary...")
sentences = brown.sents()
word_to_idx, vocab_list = build_vocab_manual(sentences, min_freq=5)
print(f"Vocabulary size: {len(word_to_idx)}")

dataset = BrownDataset(word_to_idx, max_length=MAX_LENGTH)
train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load GloVe embeddings
print("Loading GloVe embeddings...")
embed_matrix = load_glove_manual(GLOVE_FILE, word_to_idx, embedding_dim=EMBEDDING_SIZE)
embed_matrix = embed_matrix.to(DEVICE)

# Initialize model
elmo = ELMo(len(word_to_idx), EMBEDDING_SIZE, HIDDEN_SIZE, DROPOUT, embed_matrix).double().to(DEVICE)
criterion = nn.CrossEntropyLoss().to(DEVICE)
optimizer = optim.Adam(elmo.parameters(), lr=LEARNING_RATE)

# Training loop
print("Training start")
elmo.train()
for epoch in tqdm(range(EPOCHS), desc="Epochs"):
    total_train_loss = 0
    for batch in tqdm(train_dataloader, desc="Batch", leave=False):
        inp = batch[:, :-1].to(DEVICE)  # Input: all but last token
        targ = batch[:, 1:].to(DEVICE)  # Target: all but first token
        optimizer.zero_grad()
        output = elmo(inp)  # [batch_size, vocab_size, seq_len-1]
        loss = criterion(output, targ)  # Compare with shifted targets
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    
    avg_loss = total_train_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

# Save the model
output_path = 'bilstm.pt'
torch.save(elmo.state_dict(), output_path)
print(f"Model saved to {output_path}")