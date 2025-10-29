import torch
import pandas as pd
from scipy.stats import spearmanr

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors, avoiding division by zero."""
    a = torch.tensor(a) if not isinstance(a, torch.Tensor) else a
    b = torch.tensor(b) if not isinstance(b, torch.Tensor) else b
    return torch.dot(a, b) / (torch.norm(a) * torch.norm(b) + 1e-8)

def evaluate_embeddings(embedding_path, df):
    """Evaluate embeddings using WordSim-353 and return cosine similarities."""
    embeddings = torch.load(embedding_path, map_location='cpu')
    similarities = []

    for _, row in df.iterrows():
        w1 = row['Word 1'].lower()
        w2 = row['Word 2'].lower()

        if w1 in embeddings and w2 in embeddings:
            vec1 = embeddings[w1].float()
            vec2 = embeddings[w2].float()
            sim = cosine_similarity(vec1, vec2).item()
        else:
            sim = None  # Word not found in vocab

        similarities.append(sim)

    return similarities

def main():
    # Load WordSim-353 dataset
    df = pd.read_csv('wordsim353crowd.csv')

    # Paths to saved models
    model_paths = {
        "CBOW": "continous_bag.pt",
        "SkipGram": "skip.pt",
        "SVD": "svd.pt"
    }

    # Compute similarities for each model
    results = {"Word 1": df["Word 1"], "Word 2": df["Word 2"], "Human Score": df["Human (Mean)"]}

    for model_name, path in model_paths.items():
        results[f"Similarity ({model_name})"] = evaluate_embeddings(path, df)

    # Convert to DataFrame and save as CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv("wordsim_results.csv", index=False)
    
    print("Saved WordSim-353 similarity results to 'wordsim_results.csv'")

    # Compute Spearman correlation for each model
    for model_name in model_paths.keys():
        valid_df = results_df.dropna(subset=[f"Similarity ({model_name})"])
        corr, _ = spearmanr(valid_df["Human Score"], valid_df[f"Similarity ({model_name})"])
        print(f'Spearman Correlation for {model_name}: {corr:.3f}')

if __name__ == '__main__':
    main()
