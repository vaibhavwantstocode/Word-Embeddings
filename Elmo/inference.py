import torch
import argparse
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

# Device and vocab setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Brown corpus vocabulary
with open('brown_vocab.txt', 'r', encoding='utf-8') as f:
    brown_vocab = [line.strip() for line in f]
word_to_idx = {word: idx for idx, word in enumerate(brown_vocab)}
VOCAB_SIZE = len(brown_vocab)
MAX_LENGTH = 35

# ELMo Model (unchanged)
class ELMo(torch.nn.Module):
    def __init__(self, hidden_size=150):
        super().__init__()
        self.embedding = torch.nn.Embedding(VOCAB_SIZE, 300)
        self.layer_1 = torch.nn.LSTM(300, hidden_size, bidirectional=True, batch_first=True)
        self.layer_2 = torch.nn.LSTM(2*hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = torch.nn.Linear(2*hidden_size, VOCAB_SIZE)

    def forward(self, x):
        emb = self.embedding(x)
        out1, _ = self.layer_1(emb)
        out2, _ = self.layer_2(out1)
        return emb, out1, out2

# Classifier Models (all three types)
class TrainableLambdaClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lambdas = torch.nn.Parameter(torch.ones(3))
        self.lstm = torch.nn.LSTM(300, 128, batch_first=True)
        self.fc = torch.nn.Linear(128, 4)

    def forward(self, x, elmo_model):
        with torch.no_grad():
            emb, l1, l2 = elmo_model(x)
        h0 = emb
        h1 = l1
        h2 = l2
        lam_sum = torch.sum(self.lambdas)
        combined = (self.lambdas[0]/lam_sum * h0 +
                    self.lambdas[1]/lam_sum * h1 +
                    self.lambdas[2]/lam_sum * h2)
        _, (hidden, _) = self.lstm(combined.float())
        return self.fc(hidden.squeeze(0))

class FrozenLambdaClassifier(TrainableLambdaClassifier):
    def __init__(self):
        super().__init__()
        self.lambdas.requires_grad_(False)

class LearnableFunctionClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.combiner = torch.nn.Sequential(
            torch.nn.Linear(900, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 300)
        )
        self.lstm = torch.nn.LSTM(300, 128, batch_first=True)
        self.fc = torch.nn.Linear(128, 4)

    def forward(self, x, elmo_model):
        with torch.no_grad():
            emb, l1, l2 = elmo_model(x)
        combined = torch.cat([emb, l1, l2], dim=-1).float()
        transformed = self.combiner(combined)
        _, (hidden, _) = self.lstm(transformed)
        return self.fc(hidden.squeeze(0))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    indices = [word_to_idx.get(t, word_to_idx['<UNK>']) for t in tokens[:MAX_LENGTH]]
    indices += [word_to_idx['<PAD>']] * (MAX_LENGTH - len(indices))
    return torch.tensor(indices).unsqueeze(0).to(DEVICE)

def main(model_path, description):
    # Initialize models
    elmo = ELMo().double().to(DEVICE)
    elmo.load_state_dict(torch.load('bilstm.pt', map_location=DEVICE))
    elmo.eval()

    # Determine model type from filename
    if 'trainable' in model_path.lower():
        classifier = TrainableLambdaClassifier().to(DEVICE)
    elif 'frozen' in model_path.lower():
        classifier = FrozenLambdaClassifier().to(DEVICE)
    elif 'learnable' in model_path.lower():
        classifier = LearnableFunctionClassifier().to(DEVICE)
    else:
        raise ValueError("Model type not recognized from filename")

    classifier.load_state_dict(torch.load(model_path, map_location=DEVICE))
    classifier.eval()

    # Inference
    with torch.no_grad():
        inputs = preprocess(description)
        outputs = classifier(inputs, elmo)
        probs = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()

    for i, prob in enumerate(probs):
        print(f"class-{i+1} {prob:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('saved_model_path', help='Path to classifier model')
    parser.add_argument('description', help='News text to classify')
    args = parser.parse_args()
    main(args.saved_model_path, args.description)