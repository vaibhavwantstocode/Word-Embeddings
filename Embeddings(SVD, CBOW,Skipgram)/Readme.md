# Word Embedding Analysis Report

## Introduction
This project explores three different word embedding techniques for **Natural Language Processing (NLP)**:
1. **SVD-based embeddings** (a frequency-based method),
2. **CBOW (Continuous Bag-of-Words)**, and
3. **Skip-gram** (a prediction-based Word2Vec variant).

These methods were trained on the **Brown Corpus** (~1M words) and evaluated on the **WordSim-353 dataset**, which consists of **353 word pairs** with human-assigned similarity scores. The models' performances were assessed using **cosine similarity** and ranked using **Spearman’s correlation**.

---

## Methodology

### **1. SVD-based Embeddings**
- **Approach:** Built a co-occurrence matrix with a **context window of 5** (without PPMI transformation).
- **SVD Computation:** Applied **truncated SVD** (rank `k = 100`).
- **Training Efficiency:** **Very fast** (single-pass computation).
- **Spearman’s Correlation:** **0.240**  
- **Vocabulary Size:** **13,366**
  
**Formula Used:**  
\[
\text{Embedding} = U \times \sqrt{S}
\]

---

### **2. CBOW Model (with Negative Sampling)**
- **Approach:** Predicts a target word from the **average of surrounding words**.
- **Model Details:**
  - Embedding size: **400**
  - Context window: **±5 words**
  - Optimizer: **Adam** (`lr = 0.001`)
  - Training: **14 epochs**
- **Spearman’s Correlation:** **0.297**
- **Strengths:** **Efficient training**, but **less effective on rare words**.

---

### **3. Skip-gram Model (with Negative Sampling)**
- **Approach:** Uses a **target word** to predict surrounding **context words**.
- **Model Details:**
  - Embedding size: **400**
  - Context window: **5**
  - Negative sampling: **5 negative samples per positive**
  - Optimizer: **Adam** (`lr = 0.001`)
  - Training: **15 epochs**
- **Spearman’s Correlation:** **0.306**
- **Strengths:** **Better at rare words**, **higher correlation with human similarity scores**.

---

## **Word Similarity Evaluation**
Each model was evaluated using **cosine similarity** on the **WordSim-353 dataset**.  
**Spearman’s correlation results:**
- **Skip-gram:** `0.306` (Best)
- **CBOW:** `0.297`
- **SVD:** `0.240`

**Interpretation:**
- **Higher correlation → Better alignment with human similarity judgments.**
- **Skip-gram performed the best**, likely due to its **predictive approach** and effectiveness in modeling rare words.

---

## **Discussion: Training Time vs. Embedding Quality**
| Model   | Training Time | Quality (Spearman’s Correlation) | Best For |
|---------|--------------|--------------------------------|----------|
| **SVD** | **Fastest** (single-pass computation) | **0.240** | Baseline embeddings |
| **CBOW** | **Moderate** (14 epochs) | **0.297** | General NLP tasks |
| **Skip-gram** | **Slowest** (15 epochs) | **0.306** (Best) | High-quality embeddings |

### **Observations:**
- **SVD** is extremely fast but lacks contextual nuance.
- **CBOW** is computationally efficient but **averages words**, which can lose fine-grained semantics.
- **Skip-gram** takes longer but **produces the most meaningful embeddings**.

---

## **Conclusion**
### **Key Takeaways:**
✔ **Skip-gram achieves the best correlation with human similarity judgments.**  
✔ **CBOW is faster but may lose semantic detail.**  
✔ **SVD provides a quick baseline but lacks contextual richness.**  

**Recommendation:**  
- **For computational efficiency → Use CBOW.**  
- **For better semantic representation → Use Skip-gram.**  
- **For fast, static embeddings → Use SVD.**  

Skip-gram is preferred when **higher precision in word relationships** is needed, but requires **more computational resources**.

---

## **Future Improvements**
- **Try larger datasets** (e.g., Wikipedia, Common Crawl).
- **Experiment with PPMI** (Positive Pointwise Mutual Information) in SVD.
- **Use subword information** (e.g., FastText) to capture **morphological** features.

[📄 Click here to view the full analysis report](https://drive.google.com/drive/folders/1AKBB1zY8PR3niiU5l3rbdUPr9IiNgkJp?usp=drive_link)
