# cbow.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from nltk.corpus import brown
from collections import Counter
import nltk

# Download Brown corpus if not already present
nltk.download('brown')

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, pad_idx):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # Initialize embeddings
        self.embeddings.weight.data.uniform_(-0.5, 0.5)
        self.target_embeddings.weight.data.uniform_(-0.5, 0.5)

    def forward(self, context, target):
        # Ignore <PAD> tokens using masking
        mask = (context != 0).float().unsqueeze(-1)  # [batch_size, window_size, 1]
        emb_context = self.embeddings(context) * mask  # Zero out <PAD> embeddings
        emb_context = emb_context.mean(dim=1)  # [batch_size, emb_dim]

        # Target embedding (both positive and negative)
        emb_target = self.target_embeddings(target)  # [batch_size, num_samples, emb_dim]

        # Dot product between context and targets
        scores = torch.bmm(emb_target, emb_context.unsqueeze(2)).squeeze()  # [batch_size, num_samples]
        return scores


def preprocess_corpus(window_size=5, min_count=5):
    print("Preprocessing corpus...")
    
    # Tokenize by sentences, convert to lowercase, and remove non-alphabetic words
    sentences = [[word.lower() for word in sentence if word.isalpha()] for sentence in brown.sents()]

    # Flatten for word frequency count
    all_words = [word for sentence in sentences for word in sentence]
    print(f"Total words: {len(all_words)}")

    # Build vocabulary
    word_counts = Counter(all_words)
    vocab = ["<PAD>", "<UNK>"] + [word for word, count in word_counts.items() if count >= min_count]
    word2idx = {word: i for i, word in enumerate(vocab)}
    idx2word = {i: word for word, i in word2idx.items()}

    # Convert sentences to indices
    corpus_idx = [[word2idx[word] if word in word2idx else word2idx["<UNK>"] for word in sentence] for sentence in sentences]

    # Create unigram distribution for negative sampling
    word_freq = np.array([word_counts[word] if word in word2idx else 1 for word in vocab], dtype=np.float32)
    unigram_dist = word_freq ** 0.75
    unigram_dist /= unigram_dist.sum()

    print(f"Vocabulary size: {len(vocab)}")
    return corpus_idx, word2idx, idx2word, unigram_dist


def get_batches(corpus, batch_size=128, window_size=5, pad_idx=0):
    contexts, targets = [], []

    for sentence in corpus:  # Process each sentence separately
        if len(sentence) < 2 * window_size + 1:
            continue  # Skip short sentences

        for i in range(window_size, len(sentence) - window_size):
            context = sentence[i-window_size:i] + sentence[i+1:i+window_size+1]
            target = sentence[i]
            contexts.append(context)
            targets.append(target)

            if len(contexts) == batch_size:
                yield torch.tensor(contexts, dtype=torch.long), torch.tensor(targets, dtype=torch.long)
                contexts, targets = [], []

    # Yield remaining samples
    if len(contexts) > 0:
        yield torch.tensor(contexts, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def train_cbow(embedding_dim=300, num_epochs=10, neg_samples=5, batch_size=512):
    corpus_idx, word2idx, idx2word, unigram_dist = preprocess_corpus()
    vocab_size = len(word2idx)
    pad_idx = word2idx["<PAD>"]  # Get <PAD> index
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CBOW(vocab_size, embedding_dim, pad_idx).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()

    neg_dist = torch.tensor(unigram_dist, dtype=torch.float, device=device)

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()

        for contexts, targets in get_batches(corpus_idx, batch_size, pad_idx=pad_idx):
            contexts = contexts.to(device)
            targets = targets.to(device)

            # Generate negative samples
            negs = torch.multinomial(neg_dist, targets.size(0) * neg_samples, replacement=True).view(-1, neg_samples)

            # Combine positive and negative targets
            all_targets = torch.cat([targets.unsqueeze(1), negs], dim=1)

            # Forward pass
            optimizer.zero_grad()
            scores = model(contexts, all_targets)

            # Create labels
            labels = torch.cat([
                torch.ones(targets.size(0), 1, device=device),
                torch.zeros(targets.size(0), neg_samples, device=device)
            ], dim=1)

            # Calculate loss
            loss = loss_fn(scores, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(corpus_idx)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")

    embeddings = (model.embeddings.weight + model.target_embeddings.weight) / 2
    word_vectors = {idx2word[i]: embeddings[i].cpu().detach() for i in range(vocab_size)}
    torch.save(word_vectors, 'continous_bag.pt')
    print("Embeddings saved to continous_bag.pt")


if __name__ == '__main__':
    train_cbow(
        embedding_dim=400,
        num_epochs=15,
        neg_samples=5,
        batch_size=128
    )
