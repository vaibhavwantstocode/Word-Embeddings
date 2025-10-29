import torch
import pandas as pd
from scipy.stats import spearmanr
import sys

def cosine_similarity(a, b):
    a = torch.tensor(a) if not isinstance(a, torch.Tensor) else a
    b = torch.tensor(b) if not isinstance(b, torch.Tensor) else b
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-8)

def main(embedding_path):
    # Load with map_location to handle device mismatch
    embeddings = torch.load(embedding_path, map_location='cpu')
    
    df = pd.read_csv('wordsim353crowd.csv')
    
    similarities = []
    human_scores = []
    
    for _, row in df.iterrows():
        w1 = row['Word 1'].lower()
        w2 = row['Word 2'].lower()
        score = row['Human (Mean)']
        
        if w1 in embeddings and w2 in embeddings:
            vec1 = embeddings[w1].float()
            vec2 = embeddings[w2].float()
            sim = cosine_similarity(vec1, vec2).item()
            similarities.append(sim)
            human_scores.append(score)
    
    print(f"Evaluated {len(similarities)}/{len(df)} pairs")
    corr, _ = spearmanr(similarities, human_scores)
    print(f'Spearman Correlation: {corr:.3f}')
    return corr

if __name__ == '__main__':
    embeddings_path = input("Enter the path to the embeddings file: ")
    main(embeddings_path)