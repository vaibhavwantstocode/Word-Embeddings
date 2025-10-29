# Natural Language Processing - Assignment 4
**Roll Number:** 2024201044
**Name:** Vaibhav Gupta
**Course:** Introduction to Natural Language Processing  


## Table of Contents
- [Project Overview](#overview)
- [Directory Structure](#structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Report](#report)
- [References](#references)

<a name="overview"></a>
## 1. Project Overview
Implementation of ELMo with:
- 2-layer BiLSTM pretrained on Brown Corpus
- AG News classification downstream task
- Three embedding combination strategies:
  1. Trainable λ weights
  2. Frozen λ weights
  3. Learnable MLP function

<a name="structure"></a>
## 2. Directory Structure
```
2024201044_Assignment4/
├── elmo.py              # ELMo model training
├── classification.py    # Classifier training
├── inference.py         # Prediction script
├── data/
│   ├── train.csv        # AG News training data
│   └── test.csv         # AG News test data
├── models/              # Pretrained models
│   ├── blistm.pt       # ELMo model
│   ├── classifier_trainable.pt
│   ├── classifier_frozen.pt
│   └── classifier_learnable.pt
└── README.md
```

<a name="installation"></a>
## 3. Installation
```bash
# Install dependencies
pip install torch nltk pandas scikit-learn matplotlib seaborn

# Download NLTK data
python -m nltk.downloader punkt brown
```

<a name="usage"></a>
## 4. Usage

### Train ELMo
```bash
python elmo.py
```

### Train Classifiers
```bash
python classification.py
```

### Run Inference
```bash
python inference.py models/classifier_learnable.pt "Your news text..."
```

<a name="models"></a>
## 5. Pretrained Models
Download from [Google Drive](https://drive.google.com/drive/folders/1Lc16P_XTfeNZe6CKFXk0u4pV9LehPRuq?usp=sharing):
- `blistm.pt`: ELMo model
- `classifier_*.pt`: Downstream classifiers

<a name="results"></a>
## 6. Results
| Model               | Accuracy | F1-Score |
|---------------------|----------|----------|
| ELMo (Trainable λ)  | 88.21%    | 88    |
| ELMo (Frozen λ)     | 87.71%    | 87    |
| ELMo (Learnable λ)    | 87.39%    | 87    |

<a name="report"></a>
## 7. Report
Included `report.pdf` contains:
- Hyperparameter analysis
- Confusion matrices
- Comparison with SVD/Skip-gram/CBOW
- Technical justification of results

<a name="references"></a>
## 8. References
1. Peters et al. (2018) - ELMo paper
2. PyTorch documentation
3. NLTK Brown Corpus
```