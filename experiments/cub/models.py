import torch
from sklearn.metrics import classification_report, accuracy_score

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPS = 1e-18

CONCEPT_EMB_SIZE = 16


class CUBEncoder(torch.nn.Module):
    def __init__(self, input_size, emb_size, cp_output):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = emb_size

        self.concept_encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
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
        self.tuple_embedder = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
        )

    def forward(self, batch_x):
        embeddings = self.concept_encoder(batch_x)
        c_logits = self.concept_predictor(embeddings)
        c_probs = torch.sigmoid(c_logits)
        emb = self.tuple_embedder(embeddings)
        return c_probs, emb


def get_accuracy(model, data_loader):
    y_preds = []
    ys = []
    for x, c, y in data_loader:
        y_pred = model.predict((x, c, y)).float()
        y_preds.append(y_pred.argmax(dim=-1))
        ys.append(y.argmax(dim=-1))
    y_preds = torch.cat(y_preds, dim=0)
    ys = torch.cat(ys, dim=0)
    return accuracy_score(ys.detach(), y_preds.detach())
