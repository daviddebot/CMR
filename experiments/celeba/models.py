from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.types import STEP_OUTPUT
from sklearn.metrics import accuracy_score
from torch.nn.functional import binary_cross_entropy
import matplotlib.pyplot as plt

from experiments.cebab.cbm_models import ConceptEmbedding, ConceptReasoningLayer
from experiments.mnist.models_copy import SaveBestModelCallbackVal
from utils.logic import ProductTNorm

EPS = 1e-8

def sample_bernoulli(distribution):
    sig = torch.sigmoid(distribution.logits)
    return distribution.sample() - sig.detach() + sig


class CelebAEncoder(torch.nn.Module):
    def __init__(self, input_emb_size, emb_size, cp_output):
        super().__init__()
        self.embedding_size = emb_size

        self.concept_encoder = torch.nn.Sequential(
            torch.nn.Linear(input_emb_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
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

    def forward(self, batch_x):
        emb = self.concept_encoder(batch_x)
        c_logits = self.concept_predictor(emb)
        c_probs = torch.sigmoid(c_logits)
        return c_probs, emb


class DNN(pl.LightningModule):
    def __init__(self, input_size, emb_size, n_tasks, n_hidden_layers=4, lr=0.001):
        super().__init__()

        self.lr = lr
        self.embedding_size = emb_size
        self.n_tasks = n_tasks
        self.input_size = input_size

        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(input_size, emb_size))
        self.layers.append(torch.nn.ReLU())
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


class CBMCommon(pl.LightningModule):  # common functionality for CBMDeep and CBMLinear
    def __init__(self, input_size, emb_size, n_tasks, n_concepts, concept_names, task_names, lr=0.001, mode='joint', hard=False,
                 only_concept_prediction=False, task_weight=1.0, intervene=False):
        super().__init__()
        self.save_hyperparameters()

        self.input_size = input_size
        self.only_concept_prediction = only_concept_prediction
        self.freeze_cp = False
        self.mode = mode
        self.hard = hard
        self.lr = lr
        self.embedding_size = emb_size
        self.n_tasks = n_tasks
        self.n_concepts = n_concepts
        self.concept_names = concept_names
        self.task_names = task_names
        self.task_weight = task_weight
        self.intervene = intervene
        self.concept_predictor = torch.nn.Sequential(
            torch.nn.Linear(input_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, self.n_concepts),
            torch.nn.Sigmoid()
        )
        self.info = defaultdict(list)
        self.skip_info = False
        self.val_info = defaultdict(list)

        self.name = ""
        self.seedd = None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        batch_x, batch_c, batch_y = batch

        p_c, p_y = self.forward(batch)

        if self.freeze_cp:
            p_c = p_c.detach()

        true_log_p_c = -binary_cross_entropy(p_c, batch_c, reduction='none')     # batch, concept
        true_log_p_y_ch_cp_r = -binary_cross_entropy(p_y, batch_y, reduction='none')     # batch, task

        if not self.only_concept_prediction:
            logprob_per_sample = self.task_weight*torch.sum(true_log_p_y_ch_cp_r, dim=-1) + torch.sum(true_log_p_c, dim=-1)    # batch
        else:
            logprob_per_sample = torch.sum(true_log_p_c, dim=-1)  # batch
        loss = torch.mean(-logprob_per_sample)  # 1

        # === Information ===
        if self.skip_info:
            return loss

        y_accuracy = accuracy_score(p_y.detach().cpu() > 0.5, batch_y.detach().cpu())
        c_accuracy = accuracy_score(p_c.detach().cpu().flatten() > 0.5, batch_c.detach().cpu().flatten())

        self.info["y_accuracy"].append(y_accuracy)
        self.info["c_accuracy"].append(c_accuracy)
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
        c_acc = sum(self.info["c_accuracy"]) / len(self.info["c_accuracy"])
        loss = sum(self.info["loss"]) / len(self.info["loss"])
        print(" c: %f, y: %f, loss: %f" % (c_acc, y_acc, loss))

    def train_loop_base(self, train_loader, val_loader, callback_constructor, max_epochs=100):
        checkpoint_cb = ModelCheckpoint(dirpath="./results/celeba_base/" + str(self.name) + str(self.seedd) + "/", save_top_k=1,
                                        monitor="val_loss", mode='min')
        callback = callback_constructor()
        trainer = pl.Trainer(max_epochs=max_epochs, callbacks=[callback, checkpoint_cb])
        trainer.fit(model=self, train_dataloaders=train_loader, val_dataloaders=val_loader)
        self.load_state_dict(callback.best_state_dict)
        print("Best epoch:", callback.best_epoch)
        self.train(False)

    def train_loop(self, train_loader, val_loader, callback_constructor, max_epochs=100):
        if self.mode == "joint":
            self.train_loop_base(train_loader, val_loader, callback_constructor, max_epochs)
        elif self.mode == "sequential":
            self.only_concept_prediction = True
            self.freeze_cp = False
            self.train_loop_base(train_loader, val_loader, callback_constructor, max_epochs)
            self.only_concept_prediction = False
            self.freeze_cp = True
            self.train_loop_base(train_loader, val_loader, callback_constructor, max_epochs)
        else:
            self.only_concept_prediction = True
            self.freeze_cp = False
            self.train_loop_base(train_loader, val_loader, callback_constructor, max_epochs)
            self.only_concept_prediction = False
            self.freeze_cp = True
            self.train_loop_base(train_loader, val_loader, callback_constructor, max_epochs)

    def predict_concepts(self, b):
        (batch_x, batch_c, _) = b
        if self.intervene:
            c_pred = batch_c
        else:
            c_pred = self.concept_predictor(batch_x)
        c_pred = (c_pred > 0.5).float()
        return c_pred


class CBMLinear(CBMCommon):
    def __init__(self, input_size, emb_size, n_tasks, n_concepts, concept_names, task_names, lr=0.001, mode='joint', hard=False, only_concept_prediction=False, task_weight=1.0, intervene=False):
        super().__init__(input_size, emb_size, n_tasks, n_concepts, concept_names, task_names, lr, mode, hard, only_concept_prediction, task_weight, intervene)
        self.save_hyperparameters()

        self.task_predictors = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(self.n_concepts, 1)) for _ in range(self.n_tasks)])

    def forward(self, x):
        batch_x, batch_c, batch_y = x
        if self.intervene:
            c_pred = batch_c
        else:
            c_pred = self.concept_predictor(batch_x)

        if self.mode == "joint":
            # _c_pred = sample_bernoulli(torch.distributions.Bernoulli(c_pred)) if self.hard else c_pred
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
                c_pred = self.concept_predictor(batch_x)

            if self.hard:
                c_pred = (c_pred > 0.5).float()
            y_pred = torch.cat([torch.sigmoid(self.task_predictors[i](c_pred)) for i in range(self.n_tasks)], dim=-1)
            return y_pred > 0.5

    def plot_weights(self):
        # plot the weights of the linear layer
        for i in range(self.n_tasks):
            plt.figure(figsize=(20, 5))
            plt.title('Task ' + self.task_names[i])
            # plot each weight as an x y point
            plt.scatter(range(self.n_concepts), self.task_predictors[i][0].weight.detach().cpu().numpy().flatten())
            plt.axhline(0, color='black', lw=0.5)
            plt.xticks(range(self.n_concepts), self.concept_names[:self.n_concepts])
            plt.gcf().autofmt_xdate()
            plt.legend()
            plt.show()

class CBMDeep(CBMCommon):
    def __init__(self, input_size, emb_size, n_tasks, n_concepts, concept_names, task_names, lr=0.001, mode="joint", hard=False, only_concept_prediction=False, task_weight=1.0, intervene=False):
        super().__init__(input_size, emb_size, n_tasks, n_concepts, concept_names, task_names, lr, mode, hard, only_concept_prediction, task_weight, intervene)
        self.save_hyperparameters()

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

    def forward(self, x):
        batch_x, batch_c, batch_y = x
        if self.intervene:
            c_pred = batch_c
        else:
            c_pred = self.concept_predictor(batch_x)

        if self.mode == "joint":
            # _c_pred = sample_bernoulli(torch.distributions.Bernoulli(c_pred)) if self.hard else c_pred
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
                c_pred = self.concept_predictor(batch_x)

            if self.hard:
                c_pred = (c_pred > 0.5).float()
            y_pred = self.task_predictor(c_pred)
            return y_pred > 0.5

class CEMDeep(CBMCommon):
    def __init__(self, input_size, emb_size, n_tasks, n_concepts, concept_names, task_names, lr=0.001, mode="joint", hard=False, only_concept_prediction=False, task_weight=1.0, intervene=False):
        super().__init__(input_size, emb_size, n_tasks, n_concepts, concept_names, task_names, lr, mode, hard, only_concept_prediction, task_weight, intervene)
        self.concept_predictor = torch.nn.Sequential(
            torch.nn.Linear(input_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )
        self.concept_embedder = ConceptEmbedding(emb_size, n_concepts, 30)
        self.task_predictor = torch.nn.Sequential(
            torch.nn.Linear(self.n_concepts*30, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, self.n_tasks),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        batch_x, batch_c, batch_y = x
        emb = self.concept_predictor(batch_x)
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
            emb = self.concept_predictor(batch_x)
            c_emb, c_pred = self.concept_embedder(emb, c=batch_c, mode=self.mode,
                                                  hard=self.hard, train=False,
                                                  intervention_idxs=torch.arange(0, self.n_concepts))
            y_pred = self.task_predictor(c_emb.reshape(len(c_emb), -1)).squeeze()
            return y_pred > 0.5
        else:
            emb = self.concept_predictor(batch_x)
            c_emb, c_pred = self.concept_embedder(emb, mode=self.mode, hard=self.hard)
            y_pred = self.task_predictor(c_emb.reshape(len(c_emb), -1)).squeeze()
            return y_pred > 0.5

    def predict_concepts(self, b):
        (batch_x, batch_c, _) = b
        if self.intervene:
            return batch_c > 0.5
        else:
            emb = self.concept_predictor(batch_x)
            c_emb, c_pred = self.concept_embedder(emb, mode=self.mode, hard=self.hard)
            c_pred = (c_pred > 0.5).float()
            return c_pred

class StandardDCR(CBMCommon):
    def __init__(self, input_size, emb_size, n_tasks, n_concepts, concept_names, task_names, lr=0.001, mode="joint", hard=False, only_concept_prediction=False,
                 task_weight=1.0, temperature: float = 10, logic = ProductTNorm(), explanation_mode: str = 'local', intervene=False):
        super().__init__(input_size, emb_size, n_tasks, n_concepts, concept_names, task_names, lr, mode, hard, only_concept_prediction, task_weight, intervene)
        self.concept_embedder = torch.nn.Sequential(
            torch.nn.Linear(input_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )
        self.concept_predictor = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, n_concepts),
            torch.nn.Sigmoid()
        )
        self.temperature = temperature
        self.logic = logic
        self.explanation_mode = explanation_mode
        self.task_predictor = ConceptReasoningLayer(emb_size, n_concepts=n_concepts, logic=logic,
                                              n_classes=self.n_tasks, temperature=temperature)

    def forward(self, x):
        batch_x, batch_c, batch_y = x
        emb = self.concept_embedder(batch_x)
        c_pred = self.concept_predictor(emb)
        if self.mode == "joint":
            # _c_pred = sample_bernoulli(torch.distributions.Bernoulli(c_pred)) if self.hard else c_pred
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
            emb = self.concept_embedder(batch_x)
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
            emb = self.concept_embedder(batch_x)
            c_pred = self.concept_predictor(emb)
            c_pred = (c_pred > 0.5).float()
            return c_pred
