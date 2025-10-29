import torch
import nltk
from nltk.corpus import brown, stopwords
from collections import Counter, defaultdict

class SVDModel:
    """
    Builds a co-occurrence matrix from text and computes SVD-based embeddings.
    """
    def __init__(self, windowSize=5, embeddingDim=500, maxVocabSize=40097):
        """
        Initializes the SVD model.

        Args:
            windowSize (int): Number of words to consider on each side of a target word.
            embeddingDim (int): Dimensionality of the output embeddings.
            maxVocabSize (int): Maximum vocabulary size (top-K words by frequency).
        """
        self.windowSize = windowSize
        self.embeddingDim = embeddingDim
        self.maxVocabSize = maxVocabSize
        self.wordToIdx = {}
        self.idxToWord = {}
        self.cooccurMatrix = None
        self.vocabSize = 0

    def buildVocabulary(self, sentences):
        """
        Builds a truncated vocabulary from the input sentences, keeping
        only the top-K most frequent words.

        Args:
            sentences (list of list of str]): Tokenized sentences.
        """
        # Count frequencies of all words
        freqCounter = defaultdict(lambda:0)
        for sent in sentences:
            for word in sent:
                freqCounter[word] += 1
        # Remove Noise 

        freqCounter = {key: value for key, value in freqCounter.items() if value >= 5}

        # Sort words by frequency descending and pick the top-K.
        #mostCommon = freqCounter.most_common(self.maxVocabSize)
        # Extract just the words (we will then sort them for stable indexing).
        vocab = list(freqCounter.keys())
        # Sort the selected vocab for reproducibility
        vocab = sorted(vocab)

        self.wordToIdx = {word: idx for idx, word in enumerate(vocab)}
        self.idxToWord = {idx: word for word, idx in self.wordToIdx.items()}
        self.vocabSize = len(vocab)

    def buildCooccurrenceMatrix(self, sentences):
        """
        Creates a co-occurrence matrix using a symmetric context window
        but only for the truncated vocabulary.

        Args:
            sentences (list of list of str]): Tokenized sentences.
        """
        cooccurDict = defaultdict(float)
        for sent in sentences:
            sentLen = len(sent)
            for i, word in enumerate(sent):
                # Skip words not in the truncated vocab
                if word not in self.wordToIdx:
                     continue
                wordIndex = self.wordToIdx[word]
                start = max(0, i - self.windowSize)
                end = min(sentLen, i + self.windowSize + 1)
                for j in range(start, end):
                    if j != i:
                        contextWord = sent[j]
                        if contextWord in self.wordToIdx:
                            contextIndex = self.wordToIdx[contextWord]
                            cooccurDict[(wordIndex, contextIndex)] += 1.0

        # Build a PyTorch tensor for the co-occurrence matrix
        self.cooccurMatrix = torch.zeros(self.vocabSize, self.vocabSize)
        for (i, j), val in cooccurDict.items():
            self.cooccurMatrix[i, j] = val

    # def fitSVD(self):
    #     """
    #     Performs Singular Value Decomposition (SVD) on the co-occurrence matrix
    #     and returns the resulting word embeddings.

    #     Returns:
    #         embeddings (Tensor): Shape [vocabSize, embeddingDim].
    #     """
    #     # SVD on CPU to prevent GPU memory issues
    #     device = torch.device("cpu")
    #     print(f"Running SVD on CPU. Vocab size = {self.vocabSize}")
    #     cooccurMatrixCPU = self.cooccurMatrix.to(device)

    #     print("Performing SVD...")
    #     U, S, V = torch.svd(cooccurMatrixCPU)
    #     #U, S, V = torch.linalg.svd(cooccurMatrixCPU, full_matrices=False)

    #     # We can multiply U by sqrt(S) to incorporate singular values
    #     U_k = U[:, :self.embeddingDim]
    #     S_k_sqrt = torch.diag(torch.sqrt(S[:self.embeddingDim]))  # Take sqrt(S)
        
        # # Compute final word embeddings
        # embeddings = torch.mm(U_k, S_k_sqrt)
        # return embeddings
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
        return embeddings


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
    embeddings = svdModel.fitSVD()

    word_vectors = {svdModel.idxToWord[i]: embeddings[i].cpu().detach() for i in range(svdModel.vocabSize)}

    
    torch.save(word_vectors, 'svd.pt')
    print("Saved SVD embeddings to 'svd.pt'.")

if __name__ == "__main__":
    main()