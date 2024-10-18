import unittest
from collections import defaultdict
import lightning.pytorch as pl
import pytorch_lightning as pl2
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix
from experiments.celeba.models import DNN, CBMLinear, CBMDeep, CEMDeep, StandardDCR
from experiments.cub.models import CUBEncoder
# from experiments.global_mnist_beta.models_copy import MNISTModel as MNISTModel_old, InputTypes, \
#     SaveBestModelCallback, ProbRDCat, get_concept_accuracy, SaveBestModelCallbackVal
from experiments.mnist.models_copy import InputTypes, \
    SaveBestModelCallback, get_concept_accuracy, SaveBestModelCallbackVal
from experiments.cub.CUB200.cub_loader import CLASS_NAMES, CONCEPT_SEMANTICS
from experiments.mnist.rule_logger import RuleLogger
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import os
import pandas as pd

from experiments.mnist.models_copy import MNISTModel as MNISTModel_old, ProbRDCat
from lightning.pytorch.loggers import TensorBoardLogger

def get_accuracy(model, data_loader):
    y_preds = []
    ys = []
    for x, c, y in data_loader:
        y_pred = model.predict((x, c, y)).float()
        y_preds.append(y_pred)
        ys.append(y)
    y_preds = torch.cat(y_preds, dim=0)
    ys = torch.cat(ys, dim=0)
    print(classification_report(ys.detach(), y_preds.detach().cpu()))

    return roc_auc_score(ys.detach(), y_preds.detach())


class CUBTest(unittest.TestCase):

    def cub_cmr(self):
        batch_size = 1280
        n_cv = 3
        results = []
        columns = ['seed', 'task_train_acc', 'task_test_acc', 'concepts_train_acc', 'concepts_test_acc', 'model', 'perc_correct_rules', 'task_to_rules']
        results_root_dir = f"./results/"
        os.makedirs(results_root_dir, exist_ok=True)
        results_dir = f"./results/cub_base/"
        results_file = "cmr_results.csv"
        os.makedirs(results_dir, exist_ok=True)
        if os.path.exists(os.path.join(results_dir, results_file)):
            print("Some results already existing, delete first")

        emb_dir = './embeddings/cub/'
        x_train, c_train, y_train = torch.load(os.path.join(emb_dir, 'train_embeddings.pt'))
        x_test, c_test, y_test = torch.load(os.path.join(emb_dir, 'test_embeddings.pt'))

        print("Number of tasks:", y_train.shape[1])
        print("Number of concepts:", c_train.shape[1])
        print("Number of examples:", y_train.shape[0])

        print("Task imbalance:", y_train.sum(dim=0) / y_train.shape[0])

        y_train = y_train[:, :10]
        y_test = y_test[:, :10]

        print(y_train.sum(dim=0))

        cs = [set() for _ in range(y_train.shape[1])]
        for i, c in enumerate(c_train):
            for j in range(len(cs)):
                if y_train[i][j] == 1:
                    cs[j].add(tuple(c.tolist()))
        print([len(x) for x in cs])

        emb_size = 100
        w_c = 1
        w_y = 30
        w_yF = 0.005
        n_rules = 3
        rule_emb_size = 500

        train_loader = DataLoader(TensorDataset(x_train, c_train.float(), y_train.float()), batch_size=batch_size)
        test_loader = DataLoader(TensorDataset(x_test, c_test.float(), y_test.float()), batch_size=batch_size)

        for seed in range(n_cv):
            pl.seed_everything(seed+1)
            cmr_model = MNISTModel_old(encoder=CUBEncoder(input_size=x_train.shape[1], emb_size=emb_size, cp_output=c_train.shape[1]),
                                        rule_module=ProbRDCat, n_concepts=c_train.shape[1], n_tasks=y_train.shape[1],
                                        n_rules=n_rules, rule_emb_size=rule_emb_size, emb_size=emb_size,
                                        selector_input=InputTypes.embedding,
                                        c_pred_in_logic=True,
                                        w_c=w_c,
                                        w_y=w_y,
                                        w_yF=w_yF,
                                        reset_selector_every_n_epochs=25,
                                        concept_names=CONCEPT_SEMANTICS,
                                        lr=0.001, mutex=False)
            cmr_model.rule_logger = RuleLogger(cmr_model.n_tasks, cmr_model.n_rules, None, 10)

            for (model, model_name) in [(cmr_model, 'CMR')]:
                checkpoint_cb = ModelCheckpoint(dirpath="./results/cub_base/CMR/" + str(seed) + "/", save_top_k=1,
                                                monitor="val_loss", mode='min')
                cb = SaveBestModelCallbackVal()
                trainer = pl.Trainer(max_epochs=300, callbacks=[cb, checkpoint_cb], accelerator="gpu")
                trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=train_loader)
                model.load_state_dict(cb.best_state_dict)
                print("Best epoch:", cb.best_epoch)
                print("Best model at path:", checkpoint_cb.best_model_path)
                model = MNISTModel_old.load_from_checkpoint(checkpoint_cb.best_model_path)
                model.to("cpu")
                model.train(False)

                rules = []
                task_to_rules, _ = model.aggregate_rules(train_loader, type='most_likely')
                for task in task_to_rules:
                    print('=== Task', CLASS_NAMES[task], "= True ", "===")
                    for rule in task_to_rules[task]:
                        print(rule, ":", task_to_rules[task][rule])
                        rule_latex = rule.replace('&', '\land').replace('|', '\lor').replace('~', '\\neg ')
                        rules.append([CLASS_NAMES[task], rule_latex, task_to_rules[task][rule] / len(x_train) * 100])
                pd.DataFrame(rules, columns=['task', 'rule', 'support']).to_csv(
                    os.path.join(results_dir, f'rules_{seed + 1}.csv'))

                train_acc = get_accuracy(model, train_loader)
                test_acc = get_accuracy(model, test_loader)
                train_c_acc = get_concept_accuracy(model, train_loader)
                test_c_acc = get_concept_accuracy(model, test_loader)
                print("Train task accuracy :", train_acc)
                print("Test task accuracy :", test_acc)
                print("Train concept accuracy :", train_c_acc)
                print("Test concept accuracy :", test_c_acc)

                results.append([seed + 1, train_acc, test_acc, train_c_acc, test_c_acc, model_name, None, None])
                pd.DataFrame(results, columns=columns).to_csv(os.path.join(results_dir, results_file))

                model.to('cpu')
                c_train_pred = model.predict_concepts((x_train, c_train, y_train))
                c_test_pred = model.predict_concepts((x_test, c_test, y_test))
                xgboost = XGBClassifier(random_state=seed+1, scale_pos_weight=200)
                xgboost.fit(c_train_pred.detach().cpu().numpy(), y_train.detach().cpu().numpy())
                train_acc = roc_auc_score(y_train, xgboost.predict(c_train_pred.detach().cpu().numpy()))
                test_acc = roc_auc_score(y_test, xgboost.predict(c_test_pred.detach().cpu().numpy()))
                results.append([seed + 1, train_acc, test_acc, train_c_acc, test_c_acc, 'XGBoost', None, None])
                pd.DataFrame(results, columns=columns).to_csv(os.path.join(results_dir, results_file))

                dt = DecisionTreeClassifier(random_state=seed+1)
                dt.fit(c_train_pred.detach().cpu().numpy(), y_train.detach().cpu().numpy())
                train_acc = roc_auc_score(y_train, dt.predict(c_train_pred.detach().cpu().numpy()))
                test_acc = roc_auc_score(y_test, dt.predict(c_test_pred.detach().cpu().numpy()))
                results.append([seed + 1, train_acc, test_acc, train_c_acc, test_c_acc, 'DT', None, None])
                pd.DataFrame(results, columns=columns).to_csv(os.path.join(results_dir, results_file))

    def cub_competitors(self):
        batch_size = 1280
        emb_size = 100
        n_cv = 3
        task_weight = 0.01
        results = []
        columns = ['seed', 'task_train_acc', 'task_test_acc', 'concepts_train_acc', 'concepts_test_acc', 'model', 'perc_correct_rules', 'task_to_rules']
        results_root_dir = f"./results/"
        os.makedirs(results_root_dir, exist_ok=True)
        results_dir = f"./results/cub_base/"
        results_file = "cbm_competitors.csv"
        os.makedirs(results_dir, exist_ok=True)
        if os.path.exists(os.path.join(results_dir, results_file)):
            print("Some results already existing, delete first")

        emb_dir = './embeddings/cub/'
        x_train, c_train, y_train = torch.load(os.path.join(emb_dir, 'train_embeddings.pt'))
        x_test, c_test, y_test = torch.load(os.path.join(emb_dir, 'test_embeddings.pt'))

        y_train = y_train[:, :10]
        y_test = y_test[:, :10]

        x_val, c_val, y_val = x_train.clone(), c_train.clone(), y_train.clone()
        train_loader = DataLoader(TensorDataset(x_train, c_train.float(), y_train.float()), batch_size=batch_size)
        val_loader = DataLoader(TensorDataset(x_val, c_val.float(), y_val.float()), batch_size=batch_size)
        test_loader = DataLoader(TensorDataset(x_test, c_test.float(), y_test.float()), batch_size=batch_size)

        for seed in range(n_cv):
            pl2.seed_everything(seed + 1)

            models = [
                ('DNN', DNN(input_size=x_train.shape[1], emb_size=emb_size, n_tasks=y_train.shape[1], n_hidden_layers=2, lr=0.001)),
                ('HardCBMLin', CBMLinear(input_size=x_train.shape[1], emb_size=emb_size, n_tasks=y_train.shape[1], n_concepts=c_train.shape[1], concept_names=CONCEPT_SEMANTICS, task_names=CLASS_NAMES, lr=0.001, mode='joint', hard=True, task_weight=task_weight)),
                ('SoftCBMLin', CBMLinear(input_size=x_train.shape[1], emb_size=emb_size, n_tasks=y_train.shape[1], n_concepts=c_train.shape[1], concept_names=CONCEPT_SEMANTICS, task_names=CLASS_NAMES, lr=0.001, mode='joint', hard=False, task_weight=task_weight)),
                ('HardCBMDeep', CBMDeep(input_size=x_train.shape[1], emb_size=emb_size, n_tasks=y_train.shape[1], n_concepts=c_train.shape[1], concept_names=CONCEPT_SEMANTICS, task_names=CLASS_NAMES, lr=0.001, mode='joint', hard=True, task_weight=task_weight)),
                ('SoftCBMDeep', CBMDeep(input_size=x_train.shape[1], emb_size=emb_size, n_tasks=y_train.shape[1], n_concepts=c_train.shape[1], concept_names=CONCEPT_SEMANTICS, task_names=CLASS_NAMES, lr=0.001, mode='joint', hard=False, task_weight=task_weight)),
                ('CEMDeep', CEMDeep(input_size=x_train.shape[1], emb_size=emb_size, n_tasks=y_train.shape[1], n_concepts=c_train.shape[1], concept_names=CONCEPT_SEMANTICS, task_names=CLASS_NAMES, lr=0.001, mode='joint', hard=True, task_weight=task_weight)),
                ('StandardDCR', StandardDCR(input_size=x_train.shape[1], emb_size=emb_size, n_tasks=y_train.shape[1], n_concepts=c_train.shape[1], concept_names=CONCEPT_SEMANTICS, task_names=CLASS_NAMES, lr=0.001, mode='joint', hard=True, task_weight=task_weight)),
            ]
            for model_name, model in models:
                task_to_rulesT, task_to_rulesF = None, None
                cb = SaveBestModelCallbackVal
                model.train_loop(train_loader, val_loader, cb, 500)
                model.train(False)

                train_acc = get_accuracy(model, train_loader)
                val_acc = get_accuracy(model, val_loader)
                test_acc = get_accuracy(model, test_loader)
                print("Train accuracy :", train_acc)
                print("Validation accuracy :", val_acc)
                print("Test accuracy :", test_acc)
                if isinstance(model, DNN):
                    train_c_acc, test_c_acc = 0, 0
                else:
                    train_c_acc = get_concept_accuracy(model, train_loader)
                    test_c_acc = get_concept_accuracy(model, test_loader)

                if isinstance(model, CBMLinear):
                    model.plot_weights()

                results.append([seed + 1, train_acc, test_acc, train_c_acc, test_c_acc, model_name, task_to_rulesT, task_to_rulesF])
                pd.DataFrame(results, columns=columns).to_csv(os.path.join(results_dir, results_file))
