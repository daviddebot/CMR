from abc import abstractmethod
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
import pytorch_lightning as pl
from torch.nn import CrossEntropyLoss, BCELoss, ModuleList, BCEWithLogitsLoss
import torch
from collections import Counter

from utils.logic import ProductTNorm, GodelTNorm

def softselect(values, temperature, use_sigmoid = True):
    softmax_scores = torch.log_softmax(values, dim=1)
    softscores = softmax_scores - temperature * softmax_scores.mean(dim=1, keepdim=True)
    if use_sigmoid:
        softscores = torch.sigmoid(softscores)
    return softscores


class ConceptReasoningLayer(torch.nn.Module):
    def __init__(self, emb_size, n_concepts, n_classes, logic = GodelTNorm(), temperature: float = 1., set_level_rules: bool = False, use_polarity = True, output_sigmoid = True):
        super().__init__()
        self.emb_size = emb_size
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.logic = logic
        self.set_level_rules = set_level_rules
        self.emb_size_after_pool = emb_size
        if self.set_level_rules:
            self.emb_size_after_pool = emb_size * 4
        self.filter_nn = torch.nn.Sequential(
            torch.nn.Linear(self.emb_size_after_pool, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_concepts * n_classes),
        )
        self.sign_nn = torch.nn.Sequential(
            torch.nn.Linear(self.emb_size_after_pool, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_concepts * n_classes),
        )
        self.filter_nn_before_pool = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )
        self.sign_nn_before_pool = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )
        self.temperature = temperature
        self.identity = torch.nn.Identity()
        self.output_sigmoid = output_sigmoid
        self.use_polarity = use_polarity
        self.polarity_activation = self.identity if not output_sigmoid else torch.nn.Sigmoid()

    def forward(self, x, c, return_attn=False, sign_attn=None, filter_attn=None):
        values = c.unsqueeze(-1).repeat(1, 1, self.n_classes)

        sign_emb = filter_emb = x
        if sign_attn is None:
            if self.set_level_rules:
                sign_emb = self.sign_nn_before_pool(x)
                sign_emb = torch.concat([
                    torch.sum(sign_emb, dim=0, keepdim=True),
                    torch.mean(sign_emb, dim=0, keepdim=True),
                    torch.std(sign_emb, dim=0, keepdim=True),
                    torch.max(sign_emb, dim=0, keepdim=True)[0],
                ], dim=1)
                filter_emb = self.filter_nn_before_pool(x)
                filter_emb = torch.concat([
                    torch.sum(filter_emb, dim=0, keepdim=True),
                    torch.mean(filter_emb, dim=0, keepdim=True),
                    torch.std(filter_emb, dim=0, keepdim=True),
                    torch.max(filter_emb, dim=0, keepdim=True)[0],
                ], dim=1)

            # compute attention scores to build logic sentence
            # each attention score will represent whether the concept should be active or not in the logic sentence

            sign_attn = self.polarity_activation(self.sign_nn(sign_emb))
            if not self.use_polarity:
                if self.output_sigmoid:
                    sign_attn = torch.ones_like(sign_attn)
                else:
                    sign_attn = torch.inf * torch.ones_like(sign_attn)


            sign_attn = sign_attn.view(sign_attn.shape[0], self.n_concepts, self.n_classes)

            if self.set_level_rules:
                sign_attn = sign_attn.expand(len(values), -1, -1)

        # attention scores need to be aligned with predicted concept truth values (attn <-> values)
        # (not A or V) and (A or not V) <-> (A <-> V)
        # values = self.identity(values).detach() # TODO: remove this!!
        sign_terms = self.logic.iff_pair(sign_attn, values)

        if filter_attn is None:
            # compute attention scores to identify only relevant concepts for each class
            filtr = self.filter_nn(filter_emb)
            filtr = filtr.view(filtr.shape[0], self.n_concepts, self.n_classes)
            filter_attn = softselect(filtr, self.temperature, self.output_sigmoid)

            if self.set_level_rules:
                filter_attn = sign_attn.expand(len(sign_terms), -1, -1)

        # filter value
        # filtered implemented as "or(a, not b)", corresponding to "b -> a"
        filtered_values = self.logic.disj_pair(sign_terms, self.logic.neg(filter_attn))

        # generate minterm
        preds = self.logic.conj(filtered_values, dim=1).squeeze(1).float()

        # preds = torch.softmax(preds, dim=1)[:, 1:2]

        if return_attn:
            return preds, sign_attn, filter_attn
        else:
            return preds

    def explain(self, x, c, mode, concept_names=None, class_names=None, filter_attn=None, sign_attn=None, classification_threshold=0.5):
        assert mode in ['local', 'global', 'exact']

        if concept_names is None:
            concept_names = [f'c_{i}' for i in range(c.shape[1])]
        if class_names is None:
            class_names = [f'y_{i}' for i in range(self.n_classes)]

        # make a forward pass to get predictions and attention weights
        y_preds, sign_attn_mask, filter_attn_mask = self.forward(x, c, return_attn=True, filter_attn=filter_attn, sign_attn=sign_attn)

        explanations = []
        all_class_explanations = {cn: [] for cn in class_names}
        for sample_idx in range(len(x)):
            # if self.n_classes > 1:
            prediction = y_preds[sample_idx] > classification_threshold
            active_classes = torch.argwhere(prediction).ravel()
            # else:
            #     active_classes = []
            #     if y_preds[sample_idx] > 0.5:
            #         active_classes = [1]

            if len(active_classes) == 0:
                # if no class is active for this sample, then we cannot extract any explanation
                explanations.append({
                    'class': -1,
                    'explanation': '',
                    'attention': [],
                })
            else:
                # else we can extract an explanation for each active class!
                for target_class in active_classes:
                    attentions = []
                    minterm = []
                    for concept_idx in range(len(concept_names)):
                        c_pred = c[sample_idx, concept_idx]
                        sign_attn = sign_attn_mask[sample_idx, concept_idx, target_class]
                        filter_attn = filter_attn_mask[sample_idx, concept_idx, target_class]

                        # TODO: if Godel, then provide only max!

                        # we first check if the concept was relevant
                        # a concept is relevant <-> the filter attention score is lower than the concept probability
                        at_score = 0
                        sign_terms = self.logic.iff_pair(sign_attn, c_pred).item()
                        if self.logic.neg(filter_attn) < sign_terms:
                            if sign_attn >= 0.5:
                                # if the concept is relevant and the sign is positive we just take its attention score
                                at_score = filter_attn.item()
                                if mode == 'exact':
                                    minterm.append(f'{sign_terms:.3f} ({concept_names[concept_idx]})')
                                else:
                                    minterm.append(f'{concept_names[concept_idx]}')
                            else:
                                # if the concept is relevant and the sign is positive we take (-1) * its attention score
                                at_score = -filter_attn.item()
                                if mode == 'exact':
                                    minterm.append(f'{sign_terms:.3f} (~{concept_names[concept_idx]})')
                                else:
                                    minterm.append(f'~{concept_names[concept_idx]}')
                        attentions.append(at_score)

                    # add explanation to list
                    # if self.n_classes > 1:
                    target_class_name = class_names[target_class]
                    # else:
                    #     target_class_name = class_names[0]
                    minterm = ' & '.join(minterm)
                    all_class_explanations[target_class_name].append(minterm)
                    explanations.append({
                        'sample-id': sample_idx,
                        'class': target_class_name,
                        'explanation': minterm,
                        'attention': attentions,
                    })

        if mode == 'global':
            # count most frequent explanations for each class
            explanations = []
            for class_id, class_explanations in all_class_explanations.items():
                explanation_count = Counter(class_explanations)
                for explanation, count in explanation_count.items():
                    explanations.append({
                        'class': class_id,
                        'explanation': explanation,
                        'count': count,
                    })

        return explanations


class ConceptEmbedding(torch.nn.Module):
    def __init__(
            self,
            in_features,
            n_concepts,
            emb_size,
            active_intervention_values=None,
            inactive_intervention_values=None,
            intervention_idxs=None,
            training_intervention_prob=0.25,
    ):
        super().__init__()
        self.emb_size = emb_size
        self.intervention_idxs = intervention_idxs
        self.training_intervention_prob = training_intervention_prob
        if self.training_intervention_prob != 0:
            self.ones = torch.ones(n_concepts)

        self.concept_context_generators = torch.nn.ModuleList()
        for i in range(n_concepts):
            self.concept_context_generators.append(torch.nn.Sequential(
                torch.nn.Linear(in_features, 2 * emb_size),
                torch.nn.LeakyReLU(),
            ))
        self.concept_prob_predictor = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, 1),
            torch.nn.Sigmoid(),
        )

        # And default values for interventions here
        if active_intervention_values is not None:
            self.active_intervention_values = torch.tensor(
                active_intervention_values
            )
        else:
            self.active_intervention_values = torch.ones(n_concepts)
        if inactive_intervention_values is not None:
            self.inactive_intervention_values = torch.tensor(
                inactive_intervention_values
            )
        else:
            self.inactive_intervention_values = torch.zeros(n_concepts)

    def _after_interventions(
            self,
            prob,
            concept_idx,
            intervention_idxs=None,
            c_true=None,
            train=False,
    ):
        if train and (self.training_intervention_prob != 0) and (intervention_idxs is None):
            # Then we will probabilistically intervene in some concepts
            mask = torch.bernoulli(self.ones * self.training_intervention_prob)
            intervention_idxs = torch.nonzero(mask).reshape(-1)
        if (c_true is None) or (intervention_idxs is None):
            return prob
        if concept_idx not in intervention_idxs:
            return prob
        return (c_true[:, concept_idx:concept_idx + 1] * self.active_intervention_values[concept_idx]) + \
            ((c_true[:, concept_idx:concept_idx + 1] - 1) * -self.inactive_intervention_values[concept_idx])

    def forward(self, x, intervention_idxs=None, c=None, train=False, mode='joint', hard=True):
        c_emb_list, c_pred_list = [], []
        # We give precendence to inference time interventions arguments
        used_int_idxs = intervention_idxs
        if used_int_idxs is None:
            used_int_idxs = self.intervention_idxs
        for i, context_gen in enumerate(self.concept_context_generators):
            context = context_gen(x)
            c_pred = self.concept_prob_predictor(context)

            # Time to check for interventions
            if c is not None:
                c_pred = self._after_interventions(
                    prob=c_pred,
                    concept_idx=i,
                    intervention_idxs=used_int_idxs,
                    c_true=c,
                    train=train,
                )
            c_pred_list.append(c_pred)

            if mode == "joint":
                # c_pred = sample_bernoulli(torch.distributions.Bernoulli(c_pred)) if hard else c_pred.detach()
                c_pred = (c_pred.detach() > 0.5).float() if hard else c_pred.detach()
            elif mode == "sequential":
                c_pred = (c_pred.detach() > 0.5).float() if hard else c_pred.detach()

            context_pos = context[:, :self.emb_size]
            context_neg = context[:, self.emb_size:]
            c_emb = context_pos * c_pred + context_neg * (1 - c_pred)
            c_emb_list.append(c_emb.unsqueeze(1))

        return torch.cat(c_emb_list, axis=1), torch.cat(c_pred_list, axis=1)


class NeuralNet(pl.LightningModule):
    def __init__(self, input_features: int, n_classes: int, emb_size: int, learning_rate: float = 0.01):
        super().__init__()
        self.input_features = input_features
        self.n_classes = n_classes
        self.emb_size = emb_size
        self.learning_rate = learning_rate
        self.cross_entropy = CrossEntropyLoss(reduction="mean")
        self.bce = BCELoss(reduction="mean")
        self.bce_log = BCEWithLogitsLoss(reduction="mean")

    @abstractmethod
    def forward(self, X):
        raise NotImplementedError

    @abstractmethod
    def _unpack_input(self, I):
        raise NotImplementedError

    def training_step(self, I, batch_idx):
        X, _, y_true = self._unpack_input(I)

        c_preds, y_preds, explanation = self.forward(X)

        loss = self.bce(y_preds.squeeze(), y_true.float().squeeze())
        task_accuracy = accuracy_score(y_true.cpu().squeeze(), y_preds.cpu() > 0.5)
        if self.current_epoch % 10 == 0:
            print(f'{self.__class__} - Train Epoch {self.current_epoch}: task: {task_accuracy:.4f} {loss:.4f}')
        return loss

    def validation_step(self, I, batch_idx):
        X, _, y_true = self._unpack_input(I)
        c_preds, y_preds, explanation = self.forward(X)
        loss = self.bce(y_preds.squeeze(), y_true.float().squeeze())
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def score(self, X, c_true, y_true):
        c_preds, y_preds, explanation = self.forward(X)
        c_auc = 0
        if c_preds is not None:
            c_auc = accuracy_score(c_true.flatten().cpu(), c_preds.flatten().cpu()>0.5)

        y_auc = roc_auc_score(y_true.cpu().squeeze(), y_preds.detach().cpu())
        # y_auc = accuracy_score(y_true.argmax(dim=-1).cpu().squeeze(), y_preds.argmax(dim=-1).detach().cpu())
        return c_auc, y_auc


class StandardE2E(NeuralNet):
    def __init__(self, input_features: int, n_classes: int, emb_size: int, learning_rate: float = 0.01):
        super().__init__(input_features, n_classes, emb_size, learning_rate)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_features, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_classes),
            torch.nn.Sigmoid()
        )

    def _unpack_input(self, I):
        return I[0], I[1], I[2]

    def forward(self, X, explain=False):
        return None, self.model(X), None


class StandardCBM(StandardE2E):
    def __init__(self, input_features: int, n_concepts: int, n_classes: int, emb_size: int,
                 learning_rate: float = 0.01, concept_names: list = None, task_names: list = None,
                 task_weight: float = 0.1):
        super().__init__(input_features, n_classes, emb_size, learning_rate)
        self.n_concepts = n_concepts
        self.concept_names = concept_names
        self.task_names = task_names
        self.task_weight = task_weight

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_features, self.emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_size, self.emb_size),
            torch.nn.ReLU(),
        )
        self.relation_classifiers = torch.nn.Sequential(
            torch.nn.Linear(self.emb_size, self.emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_size, self.emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_size, self.emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_size, n_concepts),
            torch.nn.Sigmoid()
        )
        self.reasoner = torch.nn.Sequential(
            torch.nn.Linear(n_concepts, n_classes),
            torch.nn.Sigmoid()
        )

    def intervene(self, X, c_true, explain=False):
        return c_true, self.reasoner(c_true)

    def forward(self, X, explain=False):
        explanation = None
        embeddings = self.encoder(X)
        c_preds = self.relation_classifiers(embeddings)
        y_preds = self.reasoner((c_preds > 0.5).float())
        return c_preds, y_preds, explanation

    def training_step(self, I, batch_idx):
        X, c_true, y_true = self._unpack_input(I)

        c_preds, y_preds, _ = self.forward(X)

        concept_loss = self.bce(c_preds, c_true.float())
        task_loss = self.bce(y_preds, y_true.float())
        loss = concept_loss + self.task_weight*task_loss

        task_accuracy = f1_score(y_true.cpu(), y_preds.cpu()>0.5, average="micro")
        concept_accuracy = f1_score(c_true.cpu(), c_preds.cpu()>0.5, average="micro")
        if self.current_epoch % 10 == 0:
            print(f'{self.__class__} - Train Epoch {self.current_epoch}: task: {task_accuracy:.4f} {task_loss:.4f} '
                  f'concept: {concept_accuracy:.4f} {concept_loss:.4f}')
        return loss

    def validation_step(self, I, batch_idx):
        X, c_true, y_true = self._unpack_input(I)

        c_preds, y_preds, _ = self.forward(X)

        concept_loss = self.bce_log(c_preds, c_true.float())
        task_loss = self.bce_log(y_preds, y_true.float())
        loss = concept_loss + self.task_weight*task_loss
        self.log("val_loss", (concept_loss + task_loss / 2))
        task_accuracy = f1_score(y_true.cpu(), y_preds.cpu()>0.5, average="micro")
        concept_accuracy = f1_score(c_true.cpu(), c_preds.cpu()>0.5, average="micro")
        if self.current_epoch % 10 == 0:
            print(f'{self.__class__} - Valid Epoch {self.current_epoch}: task: {task_accuracy:.4f} {task_loss:.4f} '
                  f'concept: {concept_accuracy:.4f} {concept_loss:.4f}')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


class StandardCBMDeep(StandardCBM):
    def __init__(self, input_features: int, n_concepts: int, n_classes: int, emb_size: int,
                 learning_rate: float = 0.01, concept_names: list = None, task_names: list = None,
                 task_weight: float = 0.1):
        super().__init__(input_features, n_concepts, n_classes, emb_size, learning_rate, concept_names, task_names, task_weight)
        self.reasoner = torch.nn.Sequential(
            torch.nn.Linear(n_concepts, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_classes),
            torch.nn.Sigmoid()
        )


class StandardCEM(StandardCBM):
    def __init__(self, input_features: int, n_concepts: int, n_classes: int, emb_size: int,
                 learning_rate: float = 0.01, concept_names: list = None, task_names: list = None,
                 task_weight: float = 0.1):
        super().__init__(input_features, n_concepts, n_classes, emb_size, learning_rate, concept_names, task_names, task_weight)
        self.relation_classifiers = ConceptEmbedding(emb_size, n_concepts, emb_size)
        self.reasoner = torch.nn.Sequential(torch.nn.Linear(n_concepts*emb_size, n_classes), torch.nn.Sigmoid())

    def forward(self, X, explain=True):
        embeddings = self.encoder(X)
        c_emb, c_preds = self.relation_classifiers(embeddings)
        y_preds = self.reasoner(c_emb.reshape(len(c_emb), -1)).squeeze()
        return c_preds, y_preds, None


class StandardDCR(StandardCBM):
    def __init__(self, input_features: int, n_concepts: int, n_classes: int, emb_size: int,
                 learning_rate: float = 0.01, concept_names: list = None, task_names: list = None,
                 temperature: float = 10, logic = ProductTNorm(), explanation_mode: str = 'local',
                 task_weight: float = 0.1):
        super().__init__(input_features, n_concepts, n_classes, emb_size, learning_rate, concept_names, task_names, task_weight)
        self.temperature = temperature
        self.logic = logic
        self.explanation_mode = explanation_mode
        self.reasoner = ConceptReasoningLayer(emb_size, n_concepts=n_concepts, logic=logic,
                                              n_classes=n_classes, temperature=temperature)

    def intervene(self, X, c_true, explain=False):
        embeddings = self.encoder(X)
        return c_true, self.reasoner(embeddings, c_true)

    def forward(self, X, explain=True):
        embeddings = self.encoder(X)
        c_preds = self.relation_classifiers(embeddings)
        y_preds = self.reasoner(embeddings, (c_preds > 0.5).float())
        explanation = None
        if explain:
            explanation = self.reasoner.explain(embeddings, c_preds, self.explanation_mode,
                                                self.concept_names, self.task_names)
        return c_preds, y_preds, explanation
