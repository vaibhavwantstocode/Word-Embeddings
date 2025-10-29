import torch
import nltk
import numpy as np
import matplotlib
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nltk.corpus import brown
from collections import defaultdict
matplotlib.use('Qt5Agg')
from scipy.stats import spearmanr
from scipy.spatial.distance import cosine

class SVDModel:
    def __init__(self, windowSize=5, embeddingDim=400, maxVocabSize=1000):
        self.windowSize = windowSize
        self.embeddingDim = embeddingDim
        self.maxVocabSize = maxVocabSize
        self.wordToIdx = {}
        self.idxToWord = {}
        self.cooccurMatrix = None
        self.vocabSize = 0

    def buildVocabulary(self, sentences):
        freqCounter = defaultdict(lambda: 0)
        for sent in sentences:
            for word in sent:
                freqCounter[word] += 1

        # Keep words appearing at least 5 times
        freqCounter = {key: value for key, value in freqCounter.items() if value >= 5}

        vocab = sorted(freqCounter.keys())  # Sort for consistency
        self.wordToIdx = {word: idx for idx, word in enumerate(vocab)}
        self.idxToWord = {idx: word for word, idx in self.wordToIdx.items()}
        self.vocabSize = len(vocab)

    def buildCooccurrenceMatrix(self, sentences):
        cooccurDict = defaultdict(float)
        for sent in sentences:
            sentLen = len(sent)
            for i, word in enumerate(sent):
                if word not in self.wordToIdx:
                    continue
                wordIndex = self.wordToIdx[word]
                start, end = max(0, i - self.windowSize), min(sentLen, i + self.windowSize + 1)
                for j in range(start, end):
                    if j != i and sent[j] in self.wordToIdx:
                        contextIndex = self.wordToIdx[sent[j]]
                        cooccurDict[(wordIndex, contextIndex)] += 1.0

        self.cooccurMatrix = torch.zeros(self.vocabSize, self.vocabSize)
        for (i, j), val in cooccurDict.items():
            self.cooccurMatrix[i, j] = val

    def fitSVD(self):
      """
      Performs Singular Value Decomposition (SVD) on the co-occurrence matrix
      and returns the resulting word embeddings.
      """
      # Check matrix dimensions before calling SVD
      if self.cooccurMatrix is None or self.cooccurMatrix.size(0) == 0 or self.cooccurMatrix.size(1) == 0:
          raise ValueError("Co-occurrence matrix is empty or has invalid dimensions.")

      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      print(f"Running SVD on{device}. Vocab size = {self.vocabSize}")
      cooccurMatrixTensor = torch.tensor(self.cooccurMatrix, dtype=torch.float32)
      cooccurMatrixCPU = cooccurMatrixTensor.to(device)

      print("Performing SVD (torch.linalg.svd)...")
      print(f"Co-occurrence matrix shape: {cooccurMatrixCPU.shape}")
      try:
          U, S, V = torch.linalg.svd(cooccurMatrixTensor)
      except RuntimeError as e:
          # Fallback to torch.svd if linalg.svd fails
          print("torch.linalg.svd() failed, trying torch.svd()...")
          U, S, V = torch.svd(cooccurMatrixTensor)

      U_k = U[:, :self.embeddingDim]
      S_k_sqrt = torch.diag(torch.sqrt(S[:self.embeddingDim]))

      embeddings = torch.mm(U_k, S_k_sqrt)
      return embeddings,U,S,cooccurMatrixCPU

def evaluate_wordsim353(embedding_matrix, idx2word, wordsim_path):
    """
    Embedding matrix shape: [vocab_size, current_dim]
    idx2word: dict mapping indices to words
    wordsim_path: path to WordSim-353 CSV (e.g. "wordsim353crowd.csv")
    """
    df = pd.read_csv("wordsim353crowd.csv")
    similarities = []
    human_scores = []
    
    # Build a word->index dict
    word2idx = {w: i for i, w in idx2word.items()}
    
    for _, row in df.iterrows():
        w1, w2 = row['Word 1'].lower(), row['Word 2'].lower()
        score = row['Human (Mean)']
        if w1 in word2idx and w2 in word2idx:
            idx1, idx2 = word2idx[w1], word2idx[w2]
            v1 = embedding_matrix[idx1]
            v2 = embedding_matrix[idx2]
            # Compute cosine similarity
            sim = np.dot(v1, v2) / (
                np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8
            )
            similarities.append(sim)
            human_scores.append(score)

    if not similarities:
        return 0.0  # No overlapping words found, return 0 or handle as needed
    
    corr, _ = spearmanr(similarities, human_scores)
    return corr

def plot_dimension_vs_correlation(U, S, cooccurMatrixGPU, idx2word, wordsim_path):
    """
    For each dimension in probe_dims, build embeddings and compute
    the correlation with WordSim-353 human similarity scores.
    """
    probe_dims = [50, 100,150, 200,250, 300,350, 400,450,500]
    max_dim = 400
    print("Performing a single SVD up to max_dim:", max_dim)

    # Use the precomputed U and S from your SVD, or recompute if needed.
    # If your SVD is already the full shape, just slice U, S to each dimension.

    # Take the largest slice for a reference. Then we’ll create sub-slices for each dimension below.
    U_max = U[:, :max_dim].cpu().numpy()      # shape: [vocab_size, max_dim]
    S_max_diag = np.diag(np.sqrt(S[:max_dim].cpu().numpy()))
    base_embedding = U_max @ S_max_diag      # shape: [vocab_size, max_dim]

    correlations = []

    for dim in probe_dims:
        # Slice to current dimensionality
        embedding_dim = base_embedding[:, :dim]  # shape: [vocab_size, dim]
        # Evaluate WordSim-353 correlation at 'dim'
        correlation = evaluate_wordsim353(embedding_dim, idx2word, wordsim_path)
        correlations.append(correlation)
        print(f"Dimension {dim} → correlation: {correlation:.3f}")

    # Plot the dimension vs correlation
    plt.figure(figsize=(8, 5))
    plt.plot(probe_dims, correlations, marker='o', linestyle='-', color='b')
    plt.xlabel("Embedding Dimension")
    plt.ylabel("Spearman Correlation (WordSim-353)")
    plt.title("Dimension vs. Correlation on WordSim-353")
    plt.grid(True)
    # If you’re on a headless environment, you typically do plt.savefig(...).
    plt.show()



def main():
    """
    Main execution: loads the Brown Corpus, filters & cleans data, builds vocabulary,
    creates a co-occurrence matrix, performs SVD, and saves embeddings to 'svd.pt'.
    """
    nltk.download('brown')


    # Extract the Brown corpus, remove stopwords, keep only alpha tokens
    sentences = []
    
    for sent in brown.sents():
        tokens = [word.lower() for word in sent if word.isalpha()]
        if tokens:
            sentences.append(tokens)
            # for t in tokens:
            #     s.add(t)
    #print(len(s))
    

    # Initialize the SVD model with a reduced vocabulary (top 5000)
    svdModel = SVDModel(windowSize=5, embeddingDim=500, maxVocabSize=1000)

    print("Building vocabulary...")
    svdModel.buildVocabulary(sentences)

    print("Building co-occurrence matrix...")
    svdModel.buildCooccurrenceMatrix(sentences)

    print("Fitting SVD...")
    embeddings,U,S,coocurmatrixcpu = svdModel.fitSVD()

    plot_dimension_vs_correlation(U,S,coocurmatrixcpu,svdModel.idxToWord,            # dictionary from indices to words
    "wordsim353crowd.csv")





    # word_vectors = {svdModel.idxToWord[i]: embeddings[i].cpu().detach() for i in range(svdModel.vocabSize)}

    
    # torch.save(word_vectors, 'svd.pt')
    # print("Saved SVD embeddings to 'svd.pt'.")

if __name__ == "__main__":
    main()