from collections import defaultdict

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from torch.nn.functional import binary_cross_entropy
import matplotlib.pyplot as plt

from experiments.cebab.cbm_models import ConceptEmbedding, ConceptReasoningLayer

from experiments.celeba.models import CEMDeep, CBMDeep, CBMLinear, CBMCommon, StandardDCR

from utils.logic import ProductTNorm


class MNISTEncoderCopy(torch.nn.Module):
    def __init__(self, emb_size, cp_output, number_digits=2):
        super().__init__()
        self.embedding_size = emb_size
        self.number_digits = number_digits

        self.concept_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, 5),
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
        self.tuple_embedder = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size * self.number_digits, self.embedding_size * self.number_digits),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size * self.number_digits, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
        )

    def forward(self, batch_x):
        classes = []
        embeddings = []
        for i in range(batch_x.shape[1]):
            x = batch_x[:, i]
            emb = self.concept_encoder(x)
            c_logits = self.concept_predictor(emb)
            c_probs = torch.softmax(c_logits, dim=-1)
            classes.append(c_probs)
            embeddings.append(emb)
        # c_probs: list of (b, n) of length m to (b, m*n), same for emb
        c_probs = torch.cat(classes, dim=-1)
        emb = torch.cat(embeddings, dim=-1)
        emb = self.tuple_embedder(emb)
        return c_probs, emb


class DNN_CNN(pl.LightningModule):
    def __init__(self, input_size, emb_size, n_tasks, n_hidden_layers=4, lr=0.001):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.embedding_size = emb_size
        self.n_tasks = n_tasks
        self.input_size = input_size

        self.cnn = MNISTEncoderCopy(emb_size, 1)

        self.layers = torch.nn.ModuleList()
        for i in range(n_hidden_layers):
            self.layers.append(torch.nn.Linear(emb_size, emb_size))
            self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Linear(emb_size, self.n_tasks))
        self.layers.append(torch.nn.Sigmoid())

        self.info = defaultdict(list)
        self.skip_info = False
        self.val_info = defaultdict(list)

        self.seedd = None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def forward(self, x):
        batch_x, _, _ = x
        batch_x = self.cnn(batch_x)[1]
        for layer in self.layers:
            batch_x = layer(batch_x)
        return batch_x

    def predict(self, x):
        return self.forward(x) > 0.5

    def training_step(self, batch, batch_idx):
        _, _, batch_y = batch

        p_y = self.forward(batch)

        true_log_p_y_ch_cp_r = -binary_cross_entropy(p_y, batch_y, reduction='none')     # batch, task

        logprob_per_sample = torch.sum(true_log_p_y_ch_cp_r, dim=-1)         # batch
        loss = torch.mean(-logprob_per_sample)  # 1

        # === Information ===
        if self.skip_info:
            return loss

        y_accuracy = accuracy_score(p_y.detach().cpu() > 0.5, batch_y.detach().cpu())

        self.info["y_accuracy"].append(y_accuracy)
        self.info["loss"].append(loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        self.skip_info = True
        loss = self.training_step(batch, batch_idx)
        self.skip_info = False
        self.val_info["loss"].append(loss.item())
        y_pred = self.predict(batch)
        y_acc = accuracy_score(y_pred.detach().cpu() > 0.5, batch[2].detach().cpu())
        self.val_info["y_accuracy"].append(y_acc)

    def on_validation_epoch_start(self) -> None:
        self.val_info = defaultdict(list)

    def on_validation_epoch_end(self) -> None:
        print("val_y_acc: ", sum(self.val_info["y_accuracy"]) / len(self.val_info["y_accuracy"]))
        self.log("val_loss", sum(self.val_info["loss"]) / len(self.val_info["loss"]))

    def on_train_epoch_start(self) -> None:
        self.info = defaultdict(list)

    def on_train_epoch_end(self) -> None:
        y_acc = sum(self.info["y_accuracy"]) / len(self.info["y_accuracy"])
        loss = sum(self.info["loss"]) / len(self.info["loss"])
        print(" y: %f, loss: %f" % (y_acc, loss))

    def train_loop(self, train_loader, val_loader, callback_constructor, max_epochs=100):
        checkpoint_cb = ModelCheckpoint(dirpath="./results/celeba_base/DNN" + str(self.seedd) + "/",
                                        save_top_k=1,
                                        monitor="val_loss", mode='min')
        callback = callback_constructor()
        trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[callback, checkpoint_cb])
        trainer.fit(model=self, train_dataloaders=train_loader, val_dataloaders=val_loader)
        self.load_state_dict(callback.best_state_dict)
        print("Best epoch:", callback.best_epoch)
        self.train(False)


class CBMLinear_CNN(CBMLinear):
    def __init__(self, input_size, emb_size, n_tasks, n_concepts, concept_names, task_names, lr=0.001, mode='joint', hard=False, only_concept_prediction=False, task_weight=1.0):
        super().__init__(input_size, emb_size, n_tasks, n_concepts, concept_names, task_names, lr, mode, hard, only_concept_prediction, task_weight)
        self.save_hyperparameters()

        self.concept_predictor = MNISTEncoderCopy(emb_size, n_concepts // 2)

        self.task_predictors = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.n_concepts, 1)) for _ in range(self.n_tasks)])

    def predict_concepts(self, b):
        (batch_x, batch_c, _) = b
        if self.intervene:
            c_pred = batch_c
        else:
            c_pred, _ = self.concept_predictor(batch_x)
        c_pred = (c_pred > 0.5).float()
        return c_pred

    def forward(self, x):
        batch_x, batch_c, batch_y = x
        if self.intervene:
            c_pred = batch_c
        else:
            c_pred, _ = self.concept_predictor(batch_x)
        if self.mode == "joint":
            _c_pred = (c_pred.detach() > 0.5).float() if self.hard else c_pred.detach()
            y_pred = torch.cat([torch.sigmoid(self.task_predictors[i](_c_pred)) for i in range(self.n_tasks)], dim=-1)
        elif self.mode == "sequential":
            _c_pred = (c_pred.detach() > 0.5).float() if self.hard else c_pred.detach()
            y_pred = torch.cat([torch.sigmoid(self.task_predictors[i](_c_pred)) for i in range(self.n_tasks)], dim=-1)
        else:  # independent
            y_pred = torch.cat([torch.sigmoid(self.task_predictors[i](batch_c)) for i in range(self.n_tasks)], dim=-1)
        return c_pred, y_pred

    def predict(self, x):
        batch_x, batch_c, _ = x
        if self.training:
            return self.forward(x)[1] > 0.5
        else:
            if self.intervene:
                c_pred = batch_c
            else:
                c_pred, _ = self.concept_predictor(batch_x)

            if self.hard:
                c_pred = (c_pred > 0.5).float()
            y_pred = torch.cat([torch.sigmoid(self.task_predictors[i](c_pred)) for i in range(self.n_tasks)], dim=-1)
            return y_pred > 0.5


class CBMDeep_CNN(CBMDeep):
    def __init__(self, input_size, emb_size, n_tasks, n_concepts, concept_names, task_names, lr=0.001, mode="joint", hard=False, only_concept_prediction=False, task_weight=1.0):
        super().__init__(input_size, emb_size, n_tasks, n_concepts, concept_names, task_names, lr, mode, hard, only_concept_prediction, task_weight)
        self.save_hyperparameters()

        self.concept_predictor = MNISTEncoderCopy(emb_size, n_concepts // 2)

        self.task_predictor = torch.nn.Sequential(
            torch.nn.Linear(self.n_concepts, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, self.n_tasks),
            torch.nn.Sigmoid()
        )

    def predict_concepts(self, b):
        (batch_x, batch_c, _) = b
        if self.intervene:
            c_pred = batch_c
        else:
            c_pred, _ = self.concept_predictor(batch_x)
        c_pred = (c_pred > 0.5).float()
        return c_pred

    def forward(self, x):
        batch_x, batch_c, batch_y = x
        if self.intervene:
            c_pred = batch_c
        else:
            c_pred, _ = self.concept_predictor(batch_x)
        if self.mode == "joint":
            _c_pred = (c_pred.detach() > 0.5).float() if self.hard else c_pred.detach()
            y_pred = self.task_predictor(_c_pred)
        elif self.mode == "sequential":
            _c_pred = (c_pred.detach() > 0.5).float() if self.hard else c_pred.detach()
            y_pred = self.task_predictor(_c_pred.detach())
        else:  # independent
            y_pred = self.task_predictor(batch_c)
        return c_pred, y_pred

    def predict(self, x):
        batch_x, batch_c, _ = x
        if self.training:
            return self.forward(x)[1] > 0.5
        else:
            if self.intervene:
                c_pred = batch_c
            else:
                c_pred, _ = self.concept_predictor(batch_x)
            if self.hard:
                c_pred = (c_pred > 0.5).float()
            y_pred = self.task_predictor(c_pred)
            return y_pred > 0.5

class CEMDeep_CNN(CEMDeep):
    def __init__(self, input_size, emb_size, n_tasks, n_concepts, concept_names, task_names, lr=0.001, mode="joint", hard=False, only_concept_prediction=False, task_weight=1.0):
        super().__init__(input_size, emb_size, n_tasks, n_concepts, concept_names, task_names, lr, mode, hard, only_concept_prediction, task_weight)
        self.concept_predictor = MNISTEncoderCopy(emb_size, n_concepts // 2)

    def forward(self, x):
        batch_x, batch_c, batch_y = x
        _, emb = self.concept_predictor(batch_x)
        if self.mode in ["joint", "sequential"]:
            c_emb, c_pred = self.concept_embedder(emb, mode=self.mode, hard=self.hard)
            y_pred = self.task_predictor(c_emb.reshape(len(c_emb), -1)).squeeze()
        else:  # independent
            c_emb, c_pred = self.concept_embedder(emb, c=batch_c, mode='independent')
            y_pred = self.task_predictor(c_emb.reshape(len(c_emb), -1)).squeeze()
        return c_pred, y_pred

    def predict(self, x):
        batch_x, batch_c, _ = x
        if self.training:
            return self.forward(x)[1] > 0.5
        elif self.intervene:
            _, emb = self.concept_predictor(batch_x)
            c_emb, c_pred = self.concept_embedder(emb, c=batch_c, mode=self.mode,
                                                  hard=self.hard, train=False,
                                                  intervention_idxs=torch.arange(0, self.n_concepts))
            y_pred = self.task_predictor(c_emb.reshape(len(c_emb), -1)).squeeze()
            return y_pred > 0.5
        else:
            _, emb = self.concept_predictor(batch_x)
            c_emb, c_pred = self.concept_embedder(emb, mode=self.mode, hard=self.hard)
            y_pred = self.task_predictor(c_emb.reshape(len(c_emb), -1)).squeeze()
            return y_pred > 0.5

    def predict_concepts(self, b):
        (batch_x, batch_c, _) = b
        if self.intervene:
            return batch_c > 0.5
        else:
            _, emb = self.concept_predictor(batch_x)
            c_emb, c_pred = self.concept_embedder(emb, mode=self.mode, hard=self.hard)
            c_pred = (c_pred > 0.5).float()
            return c_pred


class StandardDCR_CNN(StandardDCR):
    def __init__(self, input_size, emb_size, n_tasks, n_concepts, concept_names, task_names, lr=0.001, mode="joint", hard=False, only_concept_prediction=False,
                 task_weight=1.0, temperature: float = 10, logic = ProductTNorm(), explanation_mode: str = 'local'):
        super().__init__(input_size, emb_size, n_tasks, n_concepts, concept_names, task_names, lr, mode, hard,
                         only_concept_prediction, task_weight, temperature, logic, explanation_mode)
        self.concept_embedder = MNISTEncoderCopy(emb_size, n_concepts // 2)

    def forward(self, x):
        batch_x, batch_c, batch_y = x
        _, emb = self.concept_embedder(batch_x)
        c_pred = self.concept_predictor(emb)
        if self.mode == "joint":
            _c_pred = (c_pred.detach() > 0.5).float() if self.hard else c_pred.detach()
            y_pred = self.task_predictor(emb, _c_pred)
        elif self.mode == "sequential":
            _c_pred = (c_pred.detach() > 0.5).float() if self.hard else c_pred.detach()
            y_pred = self.task_predictor(emb, _c_pred.detach())
        else:  # independent
            y_pred = self.task_predictor(emb, batch_c)
        return c_pred, y_pred

    def predict(self, x):
        batch_x, batch_c, _ = x
        if self.training:
            return self.forward(x)[1] > 0.5
        else:
            _, emb = self.concept_embedder(batch_x)
            if self.intervene:
                c_pred = batch_c
            else:
                c_pred = self.concept_predictor(emb)
            if self.hard:
                c_pred = (c_pred > 0.5).float()
            y_pred = self.task_predictor(emb, c_pred)
            return y_pred > 0.5

    def predict_concepts(self, b):
        (batch_x, batch_c, _) = b
        if self.intervene:
            return batch_c
        else:
            _, emb = self.concept_embedder(batch_x)
            c_pred = self.concept_predictor(emb)
            c_pred = (c_pred > 0.5).float()
            return c_pred






