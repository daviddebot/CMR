import copy
from functools import reduce

import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score, classification_report

from collections import defaultdict
from torch.nn.functional import binary_cross_entropy, cross_entropy
import math
import random
from utils.logic import ConceptEmbedding

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPS = 1e-18

CONCEPT_EMB_SIZE = 16


def reasoning(logic, concepts, polarity, relevance):
    pospolarity = polarity  # batch, task, rule, concept
    irrelevance = 1-relevance
    negpolarity = 1-pospolarity-irrelevance

    # avoid floating point errors resulting in > 1 probabilities
    pospolarity = 0.999 * pospolarity
    negpolarity = 0.999 * negpolarity
    irrelevance = 0.999 * irrelevance
    
    preds = irrelevance + (1-concepts)*negpolarity + concepts*pospolarity
    return torch.prod(preds, dim=-1)

def get_iteration_based_stop_criterion(max_it):
    def stop_criterion(it_nb, train_loader, val_loader, model):
        task_to_rules_yT, task_to_used_rules_idx_for_yT = model.aggregate_rules(train_loader, type='most_likely')
        task_to_rules_yF, task_to_used_rules_idx_for_yF = model.aggregate_rules(train_loader, type='most_likely', inv=True)
        for task in task_to_rules_yT:
            print('- Task', task, '= True, Most Probable Rules')
            for rule in task_to_rules_yT[task]:
                print(rule)
        for task in task_to_rules_yF:
            print('- Task', task, '= False, Most Probable Rules')
            for rule in task_to_rules_yF[task]:
                print(rule)
        return it_nb > max_it
    return stop_criterion

def get_accuracy(model, data_loader, print_report=False):
    y_preds = []
    ys = []
    for x, c, y in data_loader:
        y_pred = model.predict((x, c, y))
        y_preds.append(y_pred)
        ys.append(y)
    y_preds = torch.cat(y_preds, dim=0)
    ys = torch.cat(ys, dim=0)
    if print_report:
        print(classification_report(ys, y_preds.detach().cpu()))
    return accuracy_score(ys, y_preds)

def get_concept_accuracy(model, data_loader, print_report=False):
    c_preds = []
    cs = []
    cs_not_flat = []
    c_preds_not_flat = []
    for x, c, y in data_loader:
        c_pred = model.predict_concepts((x, c, y))
        c_preds.append(c_pred.flatten())
        cs.append(c.flatten())
        cs_not_flat.append(c)
        c_preds_not_flat.append(c_pred)
    c_preds = torch.cat(c_preds, dim=0)
    c_preds_not_flat = torch.cat(c_preds_not_flat, dim=0)
    cs = torch.cat(cs, dim=0)
    cs_not_flat = torch.cat(cs_not_flat, dim=0)
    if print_report:
        print(classification_report(cs_not_flat, c_preds_not_flat.cpu()))
    return accuracy_score(cs, c_preds.detach())


class InputTypes:
    """
        Possible inputs for the rule selector
    """
    concepts = 0
    embedding = 1
    concepts_ground_truth = 2


class SaveBestModelCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.best_loss = torch.tensor([float('inf')])
        self.best_state_dict = None
        self.best_epoch = 0

    def on_train_epoch_end(self, trainer, pl_module):
        if 'train_loss' in trainer.callback_metrics:
            train_loss = trainer.callback_metrics['train_loss']
            if train_loss.cpu() < self.best_loss.cpu():
                print("!!", train_loss.item(), "<", self.best_loss.item())
                self.best_loss = train_loss
                # Save the best model parameters
                self.best_state_dict = copy.deepcopy(trainer.model.state_dict())
                self.best_epoch = trainer.current_epoch


class SaveBestModelCallbackVal(pl.Callback):
    def __init__(self):
        super().__init__()
        self.best_loss = float('inf')
        self.best_state_dict = None
        self.best_epoch = 0

    def on_validation_end(self, trainer, pl_module):
        if 'val_loss' in trainer.callback_metrics:
            val_loss = trainer.callback_metrics['val_loss']
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                # Save the best model parameters
                self.best_state_dict = copy.deepcopy(trainer.model.state_dict())
                self.best_epoch = trainer.current_epoch


class SaveBestModelCallbackVal2(pl.Callback):
    def __init__(self):
        super().__init__()
        self.best_loss = float('inf')
        self.best_state_dict = None
        self.best_epoch = 0

    def on_validation_end(self, trainer, pl_module):
        if 'val_acc' in trainer.callback_metrics:
            val_loss = trainer.callback_metrics['val_acc']
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                # Save the best model parameters
                self.best_state_dict = copy.deepcopy(trainer.model.state_dict())
                self.best_epoch = trainer.current_epoch


class RuleModule(torch.nn.Module):
    def __init__(self, rule_emb_size, n_tasks, n_rules):
        """
            Abstract class for rule modules. A rule module stores rule embeddings, and provides (1) a way to decode them into
            polarities and relevances, (2) a way to predict the task given the symbolic rules, and (3) a way to compute the 'concept reconstruction'.
        """
        super().__init__()
        self.rules = torch.nn.Embedding(n_tasks * n_rules, rule_emb_size)
        self.n_rules = n_rules
        self.rule_emb_size = rule_emb_size

    def copy_embedding(self, task_idx, rule_idx_from, rule_idx_to):
        embedding_from = self.rules.weight[task_idx * self.n_rules + rule_idx_from]
        self.rules.weight.data[task_idx * self.n_rules + rule_idx_to] = torch.clone(embedding_from)

    def decode_rules(self, rule_embs):
        """
            Returns a tensor ending on dimensions (concepts, 3) where the last dimension is the probability for
            positive polarity, negative polarity and irrelevance for a specific concept.
        """
        raise NotImplementedError

    def calc_y(self, c, pospolarity, relevance):
        return reasoning(self.logic, c, pospolarity, relevance)

    def calc_c_rec(self, pospolarity, relevance):
        raise NotImplementedError


class ProbRDCat(RuleModule):
    def __init__(self, rule_emb_size, n_concepts, n_tasks, n_rules):
        """
            A rule module where rules are decoded into a categorical variable defining positve polarity, negative polarity,
            and irrelevance. Therefore, they are mutually exclusive.
        """
        super().__init__(rule_emb_size, n_tasks, n_rules)
        self.logic = None
        self.n_concepts = n_concepts
        self.rule_emb_size = rule_emb_size
        self.rule_decoder = torch.nn.Sequential(
            torch.nn.Linear(self.rule_emb_size, self.rule_emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.rule_emb_size, 3 * self.n_concepts),
        )

    def decode_rules(self, rule_embs):
        shape = rule_embs.shape[:-1]  # keep all dims except the embedding dim
        shape += (self.n_concepts, 3)
        logits = self.rule_decoder(rule_embs).view(shape)
        return torch.softmax(logits, dim=-1)

    def calc_c_rec(self, pospolarity, relevance):
        return 0.5 * (1-relevance) + pospolarity


class MNISTEncoder(torch.nn.Module):
    def __init__(self, emb_size,cp_output,number_digits=2):
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


class MNISTModel(pl.LightningModule):
    def __init__(self, encoder, emb_size, rule_emb_size, n_tasks, n_rules, n_concepts, concept_names, rule_module,
                 lr=0.001,
                 selector_input=InputTypes.concepts,
                 rel_concept_counts=None, weight_concepts=False,
                 w_c=1, w_y=1, w_yF=1,
                 c_pred_in_logic=False,
                 c_pred_in_rec=False,
                 orig_rule_sym_to_name=None,
                 reset_selector=True, reset_selector_every_n_epochs=30,
                 intervene=False,
                 mutex=False):
        """
        Args:
            encoder: Torch module with forward method that returns (c_probs, emb)
            emb_size: Embedding size
            rule_emb_size: Rule embedding size
            n_tasks: Number of tasks
            n_rules: Allowed number of rules per task
            n_concepts: Number of concepts
            concept_names: Names of the concepts
            rule_module: RuleModule instance
            lr: Learning rate
            selector_input: Type of selector input
            rel_concept_counts: Relative concept counts
            weight_concepts: Whether to weigh the concept reconstruction loss based on relative concept counts
            w_c: Weight for the concept reconstruction loss (w.r.t. the task loss)
            orig_rule_sym_to_name: None if printing rules should show pos polarity and irrelevance, otherwise prints pos and neg polarity
        """
        super().__init__()

        assert not mutex

        self.save_hyperparameters()

        self.reset_selector = reset_selector
        self.reset_selector_every_n_epochs = reset_selector_every_n_epochs

        self.lr = lr
        self.embedding_size = emb_size
        self.rule_emb_size = rule_emb_size
        self.n_tasks = n_tasks
        self.n_concepts = n_concepts
        self.n_rules = n_rules
        self.effective_n_rules = n_rules
        self.selector_input = selector_input
        self.rel_concept_counts = rel_concept_counts
        self.weight_concepts = weight_concepts
        self.rule_logger = None
        self.concept_names = concept_names
        self.w_c = w_c
        self.w_y = w_y
        self.w_yF = w_yF
        self.orig_rule_sym_to_name = orig_rule_sym_to_name
        self.skip_info = False
        self.c_pred_in_logic = c_pred_in_logic
        self.c_pred_in_rec = c_pred_in_rec
        self.freeze_rules = False
        self.intervene = intervene

        self.encoder = encoder

        self.concept_embedder = ConceptEmbedding(in_features=self.embedding_size, n_concepts=self.n_concepts, emb_size=CONCEPT_EMB_SIZE)
        self.c_emb_combiner = torch.nn.Sequential(
            torch.nn.Linear(CONCEPT_EMB_SIZE * self.n_concepts, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size)
        )

        self.rule_module = rule_module(self.rule_emb_size, self.n_concepts, self.n_tasks, self.n_rules)

        self.info = defaultdict(list)
        self.val_info = defaultdict(list)

        # === For RuleAdd Interventions ===
        self.added_rules = []  # list of lists of rules

        self.rule_mask = torch.ones(self.n_tasks, self.n_rules)

        self.initialized = False
        self.initialize_rule_selector(selector_input)

    def initialize_rule_selector(self, selector_input):
        if self.initialized:
            def initialize_weights(module):
                if isinstance(module, torch.nn.Linear):
                    module.reset_parameters()
            self.neural_rule_selector.apply(initialize_weights)
            return
        self.initialized = True

        if selector_input == InputTypes.concepts or selector_input == InputTypes.concepts_ground_truth:
            selector_input_size = self.n_concepts
        elif selector_input == InputTypes.embedding:
            selector_input_size = self.embedding_size
        else:
            raise NotImplementedError
        self.selector_input_size = selector_input_size
        self.selector_input = selector_input

        self.neural_rule_selector = torch.nn.Sequential(
            torch.nn.Linear(selector_input_size, self.embedding_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.embedding_size, self.n_tasks * self.effective_n_rules),
        ).to(self.device)

        if self.effective_n_rules > self.rule_mask.shape[1]:
            self.rule_mask = torch.cat([self.rule_mask, torch.ones(self.n_tasks, self.effective_n_rules - self.rule_mask.shape[1])], dim=-1)

        # === For Similarity Selector ===
        self.rule_emb_proj = torch.nn.Sequential(
            torch.nn.Linear(self.rule_emb_size, self.embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
        )
        self.input_emb_proj = torch.nn.Sequential(
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.embedding_size, self.embedding_size),
        )

    def decode_rules(self, rule_embs):
        decoded_rules = self.rule_module.decode_rules(rule_embs)  # tasks, rules, concepts, 3
        if not self.training:  # enforce crisp rules
            d_flat = decoded_rules.view(-1, 3)  # tasks*rules*concepts, 3
            max_indices_flat = torch.argmax(d_flat, dim=-1)
            temp = torch.zeros_like(d_flat)
            temp[torch.arange(d_flat.size(0)), max_indices_flat] = 1
            decoded_rules = temp.view(decoded_rules.shape)
        if self.freeze_rules:
            decoded_rules = decoded_rules.detach()
        return decoded_rules

    def add_rules(self, rules):
        added_rules = torch.tensor(rules)
        added_rules = torch.where(added_rules == 1, 1-1e-6, 1e-6)
        added_rules = added_rules.unsqueeze(-1).repeat(1, 1, 3)  # added_rules, concepts, 3 ; pos polarity
        added_rules[:, :, 1] = 1-added_rules[:, :, 0]  # neg polarity
        added_rules[:, :, 2] = 0  # irrelevance
        self.added_rules.append(added_rules)
        self.effective_n_rules = sum(map(len, self.added_rules)) + self.n_rules
        self.initialized = False
        self.initialize_rule_selector(self.selector_input)

    def add_rules_irr(self, rules):
        added_rules = rules
        added_rules = torch.where(added_rules == 1, 1-2e-7, added_rules)
        added_rules = torch.where(added_rules == 0, 1e-7, added_rules)
        self.added_rules.append(added_rules)
        self.effective_n_rules = sum(map(len, self.added_rules)) + self.n_rules
        self.initialized = False
        self.initialize_rule_selector(self.selector_input)

    def mask_rule(self, task_idx, rule_idx):
        self.rule_mask[task_idx, rule_idx] = 0

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def get_selector_input(self, batch_c, emb, _old1, c_pred, _old2):
        if self.intervene:
            return batch_c
        if self.selector_input == InputTypes.concepts:
            return c_pred.detach()
        elif self.selector_input == InputTypes.embedding:
            return emb
        elif self.selector_input == InputTypes.concepts_ground_truth:
            return batch_c
        else:
            raise NotImplementedError

    def get_added_rule_probs(self):
        """
        Returns the (pospol, negpol, irr) values for the manually added rules
        """
        pols = []  # times_added x (added_rules, concepts, 3)
        for added_rules in self.added_rules:
            pols.append(added_rules)
        return pols

    def combine_added_and_learned(self, r):
        if len(self.added_rules) == 0:
            return r
        added_rules = torch.cat(self.added_rules, dim=0)                        # total_added_rules, concepts, 3
        added_rules = added_rules.unsqueeze(0).repeat(self.n_tasks, 1, 1, 1)    # tasks, total_added_rules, concepts, 3
        r = torch.cat([added_rules.to(r.device), r], dim=1)                     # tasks, total_rules, concepts, 3
        return r

    def get_all_rule_vars(self):
        """
        Returns all rule (pos pol, neg pol, irr) values (of both learned and manually added rules)
        """
        r = self.rule_module.rules.weight
        r = r.view(self.n_tasks, self.n_rules, self.rule_emb_size)  # tasks, rules, emb
        rules = self.decode_rules(r)  # tasks, rules, concepts, 3
        # added rules
        if len(self.added_rules) != 0:
            added_rules = torch.cat(self.added_rules, dim=0)  # total_added_rules, concepts, 3
            added_rules = added_rules.unsqueeze(0).repeat(self.n_tasks, 1, 1, 1)  # tasks, total_added_rules, concepts, 3
            return torch.cat([added_rules.to(rules.device), rules], dim=1)  # tasks, total_rules, concepts, 3
        return rules

    def forward(self, x):
        batch_x, batch_c, batch_y = x
        batch_size = batch_c.shape[0]

        # === concept prediction ===
        c_embs = None
        c_probs, emb = self.encoder(batch_x)
        c_pred = c_probs

        # === decode rules ===
        r = self.get_all_rule_vars()

        # === select rules ===
        selector_input = self.get_selector_input(batch_c, emb, c_embs, c_pred, batch_y)
        logits_s = self.neural_rule_selector(selector_input).view(-1, self.n_tasks, self.effective_n_rules)
        log_p_s = torch.log_softmax(logits_s, dim=-1)  # batch, task, rules
        p_s = torch.softmax(logits_s, dim=-1)

        y_to_mask = torch.ones_like(batch_y).unsqueeze(2).repeat(1, 1, self.effective_n_rules)  # not used anymore

        entr = torch.sum(batch_y*torch.sum(-p_s * torch.log(p_s + EPS), dim=-1)) / batch_y.shape[0]  # for logging

        # === calculate task prediction and 'concept reconstruction' ===
        pospolarity = r[:, :, :, 0]  # task, rule, concept
        irrelevance = r[:, :, :, 2]  # task, rule, concept
        relevance = 1-irrelevance  # task, rule, concept

        c_intv = batch_c.clone()
        batch_c = batch_c.unsqueeze(1).unsqueeze(1).repeat(1, self.n_tasks, self.effective_n_rules, 1)
        _pospolarity = pospolarity.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        _relevance = relevance.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        p_c_rec = self.rule_module.calc_c_rec(_pospolarity, _relevance)
        if self.training and not self.c_pred_in_logic:
            y_per_rule = self.rule_module.calc_y(batch_c, _pospolarity, _relevance)  # batch, task, rule, concept
        elif self.intervene:
            y_per_rule = self.rule_module.calc_y(batch_c, _pospolarity, _relevance)  # batch, task, rule, concept
            c_pred = c_intv
        else:  # use thresholded c_pred for y_pred
            y_per_rule = self.rule_module.calc_y((c_pred.detach() > 0.5).float().unsqueeze(1).unsqueeze(1).repeat(1, self.n_tasks, self.effective_n_rules, 1), _pospolarity, _relevance)

        return log_p_s, p_c_rec, y_per_rule, c_pred, p_s, entr, y_to_mask

    def predict(self, x):
        _, _, y_per_rule, c_pred, p_s, _, _ = self.forward(x)
        y_pred = torch.einsum("btr,btr->bt", p_s, y_per_rule)
        y_pred = y_pred > 0.5
        return y_pred

    def predict_concepts(self, x):
        _, _, _, c_pred, _, _, _ = self.forward(x)
        return c_pred > 0.5

    def calc_avg_p_c_rec(self, dataloader):
        self.train(True)
        dens = []
        noms = []
        for x in dataloader:
            batch_x, batch_c, batch_y = x
            if self.c_pred_in_rec:
                batch_c = self.predict_concepts(x).float()
            (log_p_s, p_c_rec, _, _, _, _, _) = self.forward(x)
            b_y_btr = batch_y.unsqueeze(2).repeat(1, 1, self.effective_n_rules)  # batch,  task, _rule
            b_c_btrc = batch_c.unsqueeze(1).unsqueeze(2).repeat(1, self.n_tasks, self.effective_n_rules, 1)  # batch, _task, _rule, concept
            b_y_btrc = b_y_btr.unsqueeze(3).repeat(1, 1, 1, self.n_concepts)
            true_log_p_c_rec = -binary_cross_entropy(p_c_rec, b_c_btrc, reduction='none')  # batch, task, rule, concept
            sum1 = torch.sum(b_y_btrc * true_log_p_c_rec, dim=-1)  # batch, task, rule
            sum1 = b_y_btr*log_p_s + sum1
            noms.append(sum1)
            dens.append(b_y_btr*log_p_s)
        nom = torch.logsumexp(torch.cat(noms, dim=0), dim=0)  # task, rule
        den = torch.logsumexp(torch.cat(dens, dim=0), dim=0)
        return nom-den  # task, rule

    def training_step(self, batch, batch_idx):
        batch_x, batch_c, batch_y = batch

        (log_p_s,  # batch, task, rule
         p_c_rec,  # batch, task, rule, concept
         p_y,      # batch, task, rule
         p_c,      # batch, concept
         p_s,
         entr,
         y_to_mask,
         ) = self.forward(batch)

        b_y_btr = batch_y.unsqueeze(2).repeat(1, 1, self.effective_n_rules)                              # batch,  task, _rule
        b_c_btrc = batch_c.unsqueeze(1).unsqueeze(2).repeat(1, self.n_tasks, self.effective_n_rules, 1)  # batch, _task, _rule,  concept
        b_y_btrc = b_y_btr.unsqueeze(3).repeat(1, 1, 1, self.n_concepts)                                 # batch,  task, _rule, _concept

        # 'true' means the likelihood of the data
        true_log_p_y = -binary_cross_entropy(p_y, b_y_btr, reduction='none')           # batch, task, rule
        if not self.c_pred_in_rec:
            true_log_p_c_rec = -binary_cross_entropy(p_c_rec, b_c_btrc, reduction='none')  # batch, task, rule, concept
        else:
            c_pred_btrc = (p_c.detach() > 0.5).float().unsqueeze(1).unsqueeze(2).repeat(1, self.n_tasks, self.effective_n_rules, 1)
            true_log_p_c_rec = -binary_cross_entropy(p_c_rec, c_pred_btrc, reduction='none')  # batch, task, rule, concept
        true_log_p_c = -binary_cross_entropy(p_c, batch_c, reduction='none')           # batch, concept

        if not self.weight_concepts:
            sum1 = torch.sum(b_y_btrc * true_log_p_c_rec, dim=-1)  # batch, task, rule
        else:
            concept_w = 1 / self.rel_concept_counts.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_y.shape[0], self.n_tasks, self.effective_n_rules, 1).to(DEVICE)
            i_concept_w = 1 / (1 - self.rel_concept_counts.unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch_y.shape[0], self.n_tasks, self.effective_n_rules, 1)).to(DEVICE)
            w = torch.where(b_c_btrc == 1, concept_w, i_concept_w)
            sum1 = torch.sum(b_y_btrc * w * true_log_p_c_rec, dim=-1)

        w2 = torch.ones_like(batch_y.unsqueeze(-1).repeat(1, 1, self.effective_n_rules))
        w2[batch_y == 0] = self.w_yF
        temp = log_p_s + 0.1 * self.w_y * w2 * true_log_p_y + self.w_c * sum1  # batch, task, rule

        temp = torch.logsumexp(temp, dim=-1)                 # batch, task

        logprob_per_sample = torch.sum(temp, dim=-1)         # batch
        logprob_per_sample = logprob_per_sample + torch.sum(true_log_p_c, dim=-1)

        loss = torch.mean(-logprob_per_sample)  # 1

        # === information ===
        if self.skip_info:
            return loss

        p_s = torch.exp(log_p_s)  # batch, task, rule

        y_pred = torch.einsum("btr,btr->bt", p_s[y_to_mask[:, 0, 0] == 1], p_y[y_to_mask[:, 0, 0] == 1])  # batch, task
        avg_c_per_task = torch.einsum('br,brc->bc', p_s[torch.logical_and(batch_y == 1, y_to_mask[:, :, 0] == 1)],
                                      p_c_rec[torch.logical_and(batch_y == 1, y_to_mask[:, :, 0] == 1)])
        c_true = batch_c.unsqueeze(1).repeat(1, self.n_tasks, 1)
        c_true = c_true[torch.logical_and(batch_y == 1, y_to_mask[:, :, 0] == 1)]
        c_true = c_true.view(-1, self.n_concepts).detach().cpu().numpy()
        c_avg_pred = avg_c_per_task.detach().cpu().reshape(-1, self.n_concepts).numpy().round()
        c_true, c_avg_pred = c_true.flatten(), c_avg_pred.flatten()  # new
        c_avg_accuracy = accuracy_score(c_true, c_avg_pred)
        c_accuracy = accuracy_score(batch_c.cpu().numpy().flatten(), p_c.detach().cpu().numpy().round().flatten())
        y_pred = y_pred > 0.5
        y_accuracy_logic = accuracy_score(batch_y.detach().cpu()[y_to_mask[:, 0, 0].to('cpu') == 1], y_pred.detach().cpu())

        self.info["c_accuracy"].append(c_accuracy)
        self.info["c'_accuracy"].append(c_avg_accuracy)
        self.info["y_accuracy"].append(y_accuracy_logic)
        self.info["entropy"].append(entr.item())
        self.info["loss"].append(loss.item())
        self.info["p_s"].append(p_s.detach())

        return loss

    def validation_step(self, batch, batch_idx):
        self.skip_info = True
        val_loss = self.training_step(batch, batch_idx)
        self.val_info["loss"].append(val_loss.item())
        self.skip_info = False
        y_pred = self.predict(batch)
        y_acc = accuracy_score(y_pred.detach().cpu() > 0.5, batch[2].detach().cpu())
        self.val_info["y_accuracy"].append(y_acc)

    def on_validation_epoch_start(self) -> None:
        self.val_info = defaultdict(list)

    def on_validation_epoch_end(self) -> None:
        print("val_acc: %f" % (sum(self.val_info["y_accuracy"]) / len(self.val_info["y_accuracy"])))
        self.log("val_loss", sum(self.val_info["loss"]) / len(self.val_info["loss"]))
        self.log("val_acc", -sum(self.val_info["y_accuracy"]) / len(self.val_info["y_accuracy"]))

    def on_train_epoch_start(self) -> None:
        self.info = defaultdict(list)

        # reset selector if needed
        if self.reset_selector and self.current_epoch % self.reset_selector_every_n_epochs == 0 and self.current_epoch > 0:
            print("=== Re-initializing rule selector ===")
            self.initialize_rule_selector(self.selector_input)

        if self.rule_logger is not None:
            self.rule_logger.log(self.get_all_rule_vars())

    def on_train_epoch_end(self) -> None:
        if self.rule_logger is not None:
            self.rule_logger.log_probs(torch.cat(self.info["p_s"], dim=0))

        self.log("train_loss", sum(self.info["loss"]) / len(self.info["loss"]))

        c_acc = sum(self.info["c_accuracy"]) / len(self.info["c_accuracy"])
        c_prime_acc = sum(self.info["c'_accuracy"]) / len(self.info["c'_accuracy"])
        y_acc = sum(self.info["y_accuracy"]) / len(self.info["y_accuracy"])
        entr = sum(self.info["entropy"]) / len(self.info["entropy"])
        loss = sum(self.info["loss"]) / len(self.info["loss"])
        print(" c: %f, c': %f, y: %f, entr: %f, loss: %f" % (c_acc, c_prime_acc, y_acc, entr, loss))


    def get_rules_sym(self, rule_vars, rule_idx=None, task_idx=None):
        def to_rule_sym(r_idx, t_idx, rule_vars):
            c_type = torch.argmax(rule_vars[t_idx, r_idx, :, :].detach(), dim=-1)
            f = lambda argmax: 1 if argmax == 0 else 0 if argmax == 1 else 9
            r = [f(c_type[k]) for k in range(len(c_type))]
            if self.orig_rule_sym_to_name is None:  # show pospol and (irrel) only
                r = [f'({self.concept_names[k]})' if r[k] == 9 else self.concept_names[k] for k in range(len(r)) if r[k] in (1, 9)]
            else:  # show pospol and ~negpol only
                r = [f'~{self.concept_names[k]}' if r[k] == 0 else self.concept_names[k] for k in range(len(r)) if r[k] in (1, 0)]
            rule_sym = ' & '.join(r)  # only showing positive polarity atoms and not relevant atoms
            return rule_sym
        if rule_idx is not None and task_idx is not None:
            return to_rule_sym(rule_idx, task_idx, rule_vars)
        elif rule_idx is not None:
            return [to_rule_sym(rule_idx, t, rule_vars) for t in range(self.n_tasks)]
        elif task_idx is not None:
            return [to_rule_sym(r, task_idx, rule_vars) for r in range(self.effective_n_rules)]
        else:  # all of them
            return [[to_rule_sym(r, t, rule_vars) for r in range(self.effective_n_rules)] for t in range(self.n_tasks)]

    def aggregate_rules(self, dataloader, type='most_likely', inv=False):
        assert type in ('most_likely', 'mean_probability', 'concept_probs')
        rule_vars = self.get_all_rule_vars()
        rules_sym = self.get_rules_sym(rule_vars)
        if type == 'most_likely':
            task_to_rules = dict()
            task_to_rule_idx = dict()
            for task in range(self.n_tasks):
                task_to_rules[task] = dict()
                task_to_rule_idx[task] = set()
            for batch in dataloader:
                log_p_s_x, _, y_per_rule, _, _, _, _ = self(batch)
                rule_idxs = torch.argmax(log_p_s_x, dim=-1)
                for example_idx, y_true in enumerate(batch[2]):
                    for task in range(self.n_tasks):
                        if not y_true[task] and not inv:
                            continue
                        if y_true[task] and inv:
                            continue
                        rule_idx = rule_idxs[example_idx, task]
                        rule_sym = rules_sym[task][rule_idx]
                        if rule_sym not in task_to_rules[task]:
                            task_to_rules[task][rule_sym] = 0
                        task_to_rules[task][rule_sym] += 1
                        task_to_rule_idx[task].add(rule_idx.item())
            return task_to_rules, task_to_rule_idx
        elif type == 'mean_probability':
            rule_probs_per_task_T = [[[] for _ in range(self.effective_n_rules)] for _ in range(self.n_tasks)]  # n_tasks x n_rules x ..
            rule_probs_per_task_F = [[[] for _ in range(self.effective_n_rules)] for _ in range(self.n_tasks)]  # n_tasks x n_rules x ..
            for batch in dataloader:
                log_p_s_x, _, y_per_rule, _, _, _, _ = self(batch)
                for example_idx, y_true in enumerate(batch[2]):
                    for task in range(self.n_tasks):
                        arr = rule_probs_per_task_T if y_true[task] else rule_probs_per_task_F
                        for rule_idx, rule_prob in enumerate(torch.exp(log_p_s_x[example_idx, task]).detach()):
                            arr[task][rule_idx].append(rule_prob.item())
            rule_probs_per_task_T, rule_probs_per_task_F = (list(map(torch.tensor, rule_probs_per_task_T)), list(map(torch.tensor, rule_probs_per_task_F)))
            rule_probs_per_task_T_avg, rule_probs_per_task_F_avg = [x.mean(dim=-1) for x in rule_probs_per_task_T], [x.mean(dim=-1) for x in rule_probs_per_task_F]
            rule_probs_per_task_T_sdev, rule_probs_per_task_F_sdev = [x.std(dim=-1) for x in rule_probs_per_task_T], [x.std(dim=-1) for x in rule_probs_per_task_F]
            return rule_probs_per_task_T_avg, rule_probs_per_task_F_avg, rule_probs_per_task_T_sdev, rule_probs_per_task_F_sdev
        elif type == 'concept_probs':
            concept_probs = torch.zeros((self.n_tasks, self.effective_n_rules, self.n_concepts, 2))  # task x rule x concept x 2
            for batch in dataloader:
                b_x, b_c, b_y = batch
                b_c1 = b_c.unsqueeze(1).unsqueeze(2).repeat(1, self.n_tasks, self.effective_n_rules, 1)
                b_y1 = b_y.unsqueeze(2).unsqueeze(-1).repeat(1, 1, self.effective_n_rules, self.n_concepts)
                _, _, _, _, p_s, _, _ = self(batch)
                for b_idx in range(p_s.shape[0]):
                    probs = p_s[b_idx, :, :].unsqueeze(-1).repeat(1, 1, self.n_concepts)
                    concept_probs[:, :, :, 0] += probs * b_c1[b_idx, :, :, :].float() * b_y1[b_idx, :, :, :].float()
                    concept_probs[:, :, :, 1] += probs * (1-b_c1[b_idx, :, :, :]).float() * b_y1[b_idx, :, :, :].float()
            concept_probs /= len(dataloader.dataset)
            return concept_probs

    def check_polarity_crispness(self, used_idxsT, used_idxsF):
        rule_vars = self.get_all_rule_vars()  # task, rule, concept, 3
        temp = torch.all(rule_vars < 0.9, dim=-1).float()  # task, rule, concept
        for t_idx in range(temp.shape[0]):
            for r_idx in range(temp.shape[1]):
                if r_idx not in used_idxsT[t_idx] or r_idx not in used_idxsF[t_idx]:
                    continue
                rule_sym = self.get_rules_sym(rule_vars, r_idx, t_idx)
                if temp[t_idx, r_idx].sum().item() > 0.01:
                    print("Rule", rule_sym, "for task", t_idx, "does not have crisp polarity!")
                    for c_idx in range(temp.shape[1]):
                        if temp[t_idx, r_idx, c_idx].item() > 0.01:
                            print("-", c_idx, ":", rule_vars[t_idx, r_idx, c_idx])

class CMR(MNISTModel):
    def __init__(self, 
                 encoder, 
                 emb_size, rule_emb_size, 
                 n_tasks, n_rules, n_concepts, 
                 concept_names,
                 learning_rate=0.001,
                 selector_input=InputTypes.embedding,
                 reset_selector=True, reset_selector_every_n_epochs=30,
                 rel_concept_counts=None, weight_concepts=False,
                 w_c=1, w_y=1, w_yF=1,
                 c_pred_in_logic=True, c_pred_in_rec=False,
                 orig_rule_sym_to_name=None):
        super().__init__(encoder, emb_size, rule_emb_size, n_tasks, n_rules, n_concepts, concept_names, ProbRDCat,
                         learning_rate, selector_input, rel_concept_counts, weight_concepts, w_c, 10*w_y, w_yF, c_pred_in_logic, c_pred_in_rec,
                         orig_rule_sym_to_name, reset_selector, reset_selector_every_n_epochs, False, False)
