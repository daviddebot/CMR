import torch
import lightning.pytorch as pl
from sklearn.metrics import accuracy_score, balanced_accuracy_score

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPS = 1e-18


class MNISTEncoder(torch.nn.Module):
    def __init__(self, emb_size, cp_output):
        super().__init__()
        self.embedding_size = emb_size
        self.concept_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 6, 5),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(6, 16, 5),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.ReLU(True),
            torch.nn.Flatten(),
            torch.nn.Linear(256, self.embedding_size),
        )
        self.concept_predictor = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, cp_output),
        )
        self.additional = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
        )

    def forward(self, batch_x):
        x = batch_x
        emb = self.concept_encoder(x)
        c_logits = self.concept_predictor(emb)
        c_probs1 = torch.softmax(c_logits[:, :10], dim=-1)
        c_probs2 = torch.sigmoid(c_logits[:, 10:])
        c_probs = torch.cat([c_probs1, c_probs2], dim=-1)
        return c_probs, self.additional(emb)
