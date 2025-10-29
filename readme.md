#  INLP Assignment : Word Embeddings & ELMo

## Overview

This repository contains two major NLP projects:
- **ELMo Implementation:** Contextual word embeddings using a 2-layer BiLSTM, applied to AG News classification.
- **Classical Word Embeddings:** SVD, CBOW, and Skip-gram models trained and evaluated on the Brown Corpus and WordSim-353 dataset.

It includes code, pretrained models, evaluation scripts, and analysis reports.

---

## Directory Structure

```
Word Embeddings/
├── Elmo/
│   ├── classification.py         # AG News classifier using ELMo embeddings
│   ├── elmo.py                   # ELMo model training
│   ├── inference.py              # Inference script for news classification
│   ├── readme.md                 # ELMo project README
│   ├── INLP_Assignment4.pdf      # Assignment instructions
│   ├── inlp_assignment_4_report.pdf # ELMo analysis report
│   ├── models/                   # Pretrained ELMo and classifier models
│   └── data/                     # AG News dataset
├── Embeddings(SVD, CBOW,Skipgram)/
│   ├── SingleValueDecomp.py      # SVD embedding training
│   ├── SVD.py                    # SVD embedding code
│   ├── ContinousBag.py           # CBOW embedding training
│   ├── skip.py                   # Skip-gram embedding training
│   ├── evaluate_wordsim.py       # Evaluation on WordSim-353
│   ├── wordsim.py                # Word similarity evaluation
│   ├── Readme.md                 # Classical embeddings README
│   ├── Analysis Report.pdf       # Classical embeddings analysis
│   ├── wordsim_results.csv       # Results on WordSim-353
│   ├── wordsim353crowd.csv       # WordSim-353 dataset
│   ├── svd.pt                    # SVD pretrained embeddings
│   ├── continous_bag.pt          # CBOW pretrained embeddings
│   ├── skip.pt                   # Skip-gram pretrained embeddings
└── Combined_README.md            # This combined README
```

---

## Installation

```sh
pip install torch nltk pandas scikit-learn matplotlib seaborn
python -m nltk.downloader punkt brown
```

---

## Usage

### ELMo Project

- **Train ELMo model:**  
  `python elmo.py`
- **Train AG News classifier:**  
  `python classification.py`
- **Run inference:**  
  `python inference.py models/classifier_learnable.pt "Your news text..."`

### Classical Embeddings Project

- **Train SVD embeddings:**  
  `python SingleValueDecomp.py`
- **Train CBOW embeddings:**  
  `python ContinousBag.py`
- **Train Skip-gram embeddings:**  
  `python skip.py`
- **Evaluate on WordSim-353:**  
  `python evaluate_wordsim.py`

---

## Results

### Classical Embeddings (WordSim-353)

| Model      | Spearman’s Correlation | Training Time | Best For                |
|------------|------------------------|--------------|-------------------------|
| SVD        | 0.240                  | Fastest      | Baseline embeddings     |
| CBOW       | 0.297                  | Moderate     | General NLP tasks       |
| Skip-gram  | 0.306 (Best)           | Slowest      | High-quality embeddings |

### ELMo (AG News Classification)

| Model               | Accuracy | F1-Score |
|---------------------|----------|----------|
| ELMo (Trainable λ)  | 88.21%   | 88       |
| ELMo (Frozen λ)     | 87.71%   | 87       |
| ELMo (Learnable λ)  | 87.39%   | 87       |

---

## Reports

- **Elmo/inlp_assignment_4_report.pdf:** ELMo analysis and results
- **Embeddings(SVD, CBOW,Skipgram)/Analysis Report.pdf:** Classical embeddings analysis
- **Elmo/INLP_Assignment4.pdf:** Assignment instructions

---

## References

- Peters et al. (2018) - ELMo paper
- Mikolov et al. (2013) - Word2Vec (CBOW & Skip-gram)
- NLTK Brown Corpus
- WordSim-353 Dataset
- PyTorch documentation

---

## License

For educational use as part of INLP coursework.

---

*For full details, see the included PDF reports in each folder.*