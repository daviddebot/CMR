import unittest
from collections import defaultdict

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from xgboost import XGBClassifier

from experiments.celeba.dataset import load_celeba_data
from experiments.celeba.models import CelebAEncoder, DNN, CBMLinear, CBMDeep, CEMDeep, StandardDCR
from experiments.mnist.models_copy import MNISTModel as MNISTModel_old, InputTypes, \
    SaveBestModelCallback, ProbRDCat, get_accuracy, SaveBestModelCallbackVal, \
    get_iteration_based_stop_criterion, get_concept_accuracy
from torch.utils.data import TensorDataset, DataLoader
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt
import torch

import os
import pandas as pd
import numpy as np

pl.seed_everything(42)

N_CONCEPTS = [1, 12, 24, 37]


def get_data(tasks=None):
    if tasks is None:
        tasks = ['Black_Hair', 'Wavy_Hair', 'Male']
    x_train, c_train, y_train, concept_names, task_names = load_celeba_data('../data', split='train')
    val_split = 0.2
    train_set_size = int(len(x_train) * (1 - val_split))
    x_val, c_val, y_val = x_train[train_set_size:], c_train[train_set_size:], y_train[train_set_size:]
    x_train, c_train, y_train = x_train[:train_set_size], c_train[:train_set_size], y_train[:train_set_size]
    x_test, c_test, y_test, _, _ = load_celeba_data('../data', split='test')

    c_train = torch.cat([c_train, y_train], dim=-1)
    c_val = torch.cat([c_val, y_val], dim=-1)
    c_test = torch.cat([c_test, y_test], dim=-1)
    concept_names = concept_names + task_names

    assert all([task in concept_names for task in tasks])
    task_c_idxs = [concept_names.index(task) for task in tasks]
    task_c_idxs.sort()
    task_c_idxs.reverse()

    y_train, y_val, y_test = c_train[:, task_c_idxs], c_val[:, task_c_idxs], c_test[:, task_c_idxs]
    task_names = [concept_names[i] for i in task_c_idxs]

    for c_idx in task_c_idxs:
        c_train = torch.cat([c_train[:, :c_idx], c_train[:, c_idx + 1:]], dim=-1)
        c_val = torch.cat([c_val[:, :c_idx], c_val[:, c_idx + 1:]], dim=-1)
        c_test = torch.cat([c_test[:, :c_idx], c_test[:, c_idx + 1:]], dim=-1)
        concept_names = concept_names[:c_idx] + concept_names[c_idx + 1:]

    return x_train, c_train, y_train, x_val, c_val, y_val, x_test, c_test, y_test, concept_names, task_names


class CelebATest(unittest.TestCase):

    def celeba_cmr(self):
        n_cv = 3
        results = []
        columns = ['seed', 'n_concepts', 'task_train_acc', 'task_test_acc', 'concepts_train_acc', 'concepts_test_acc', 'model',
                   'rules_T', 'rules_F']
        results_root_dir = f"./results/"
        os.makedirs(results_root_dir, exist_ok=True)
        results_dir = f"./results/celeba_base/"
        current_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_file = "cmr_results_" + current_time + ".csv"
        os.makedirs(results_dir, exist_ok=True)

        x_train, c_train, y_train, x_val, c_val, y_val, x_test, c_test, y_test, concept_names, task_names = get_data()

        # for n_concepts in [1, 5, 10, 20, 30, 37]:
        for n_concepts in N_CONCEPTS:
            print("=== Running for n_concepts =", n_concepts, "===")
            _c_train = c_train[:, :n_concepts]
            _c_test = c_test[:, :n_concepts]
            _c_val = c_val[:, :n_concepts]
            train_loader = DataLoader(TensorDataset(x_train, _c_train, y_train), batch_size=1000, shuffle=False)
            test_loader = DataLoader(TensorDataset(x_test, _c_test, y_test), batch_size=1000, shuffle=False)
            val_loader = DataLoader(TensorDataset(x_val, _c_val, y_val), batch_size=1000, shuffle=False)

            for seed in range(n_cv):
                pl.seed_everything(seed + 1)

                models = []
                for c_pred_in_logic, c_pred_in_rec, w_y in [(True, False, 300)]:    
                    models.append(("CMR", MNISTModel_old(encoder=CelebAEncoder(input_emb_size=512, emb_size=100, cp_output=_c_train.shape[1]),
                                                 rule_module=ProbRDCat, n_concepts=_c_train.shape[1], n_tasks=y_train.shape[1],
                                                 n_rules=5, rule_emb_size=100, emb_size=100,
                                                 selector_input=InputTypes.embedding,
                                                 concept_names=concept_names,
                                                 w_c=1, w_y=w_y, w_yF=1, c_pred_in_logic=c_pred_in_logic, c_pred_in_rec=c_pred_in_rec,
                                                 orig_rule_sym_to_name='a',  # show rules in 'real' logic rule notation
                                                 reset_selector=True, reset_selector_every_n_epochs=35,
                                                 lr=0.001)))
                for model_name, model in models:
                    checkpoint_cb = ModelCheckpoint(dirpath="./results/celeba_base/CMR/" + str(seed) + "/", save_top_k=1, monitor="val_loss", mode='min')
                    cb = SaveBestModelCallbackVal()
                    trainer = pl.Trainer(max_epochs=100, callbacks=[cb, checkpoint_cb])
                    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
                    model.load_state_dict(cb.best_state_dict)
                    print("Best epoch:", cb.best_epoch)
                    print("Best model at path:", checkpoint_cb.best_model_path)
                    model = MNISTModel_old.load_from_checkpoint(checkpoint_cb.best_model_path)
                    model.to("cpu")
                    model.train(False)

                    train_acc = get_accuracy(model, train_loader)
                    val_acc = get_accuracy(model, val_loader)
                    test_acc = get_accuracy(model, test_loader)
                    print("Train accuracy :", train_acc)
                    print("Validation accuracy :", val_acc)
                    print("Test accuracy :", test_acc)
                    train_c_acc = get_concept_accuracy(model, train_loader)
                    test_c_acc = get_concept_accuracy(model, test_loader)

                    # check the rules
                    task_to_rulesT, _ = model.aggregate_rules(train_loader, type='most_likely')
                    task_to_rulesF, _ = model.aggregate_rules(train_loader, type='most_likely', inv=True)

                    if model.n_rules <= 30:
                        loader = DataLoader(TensorDataset(x_train, _c_train, y_train), batch_size=512)
                        (rule_probs_per_task_T_avg, rule_probs_per_task_F_avg,
                         rule_probs_per_task_T_sdev, rule_probs_per_task_F_sdev) = model.aggregate_rules(loader, type='mean_probability')
                        rules_vars = model.get_all_rule_vars()
                        for task in range(model.n_tasks):
                            plt.figure()
                            plt.title('Task %d' % task)
                            plt.errorbar(range(model.effective_n_rules), rule_probs_per_task_T_avg[task], yerr=rule_probs_per_task_T_sdev[task], fmt='o', label='True')
                            plt.errorbar([x+0.1 for x in range(model.effective_n_rules)], rule_probs_per_task_F_avg[task], yerr=rule_probs_per_task_F_sdev[task], fmt='o', label='False')
                            plt.xticks(range(model.effective_n_rules), model.get_rules_sym(rules_vars, task_idx=task))
                            plt.gcf().autofmt_xdate()
                            plt.legend()
                            plt.show()

                    results.append([seed + 1, n_concepts, train_acc, test_acc, train_c_acc, test_c_acc, model_name, task_to_rulesT, task_to_rulesF])
                    pd.DataFrame(results, columns=columns).to_csv(os.path.join(results_dir, results_file))

    def celeba_competitors(self):
        n_cv = 3
        n_epochs = 100
        results = []
        columns = ['seed', 'n_concepts', 'task_train_acc', 'task_test_acc', 'concepts_train_acc', 'concepts_test_acc', 'model',
                   'rules_T', 'rules_F']
        results_root_dir = f"./results/"
        os.makedirs(results_root_dir, exist_ok=True)
        results_dir = f"./results/celeba_base/"
        current_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_file = "comp_results_" + current_time + ".csv"
        os.makedirs(results_dir, exist_ok=True)
        if os.path.exists(os.path.join(results_dir, results_file)):
            print("Some results already existing, delete first")

        x_train, c_train, y_train, x_val, c_val, y_val, x_test, c_test, y_test, CN, TN = get_data()

        for n_concepts in N_CONCEPTS:
            _c_train = c_train[:, :n_concepts]
            _c_test = c_test[:, :n_concepts]
            _c_val = c_val[:, :n_concepts]
            train_loader = DataLoader(TensorDataset(x_train, _c_train, y_train), batch_size=1000, shuffle=False)
            test_loader = DataLoader(TensorDataset(x_test, _c_test, y_test), batch_size=1000, shuffle=False)
            val_loader = DataLoader(TensorDataset(x_val, _c_val, y_val), batch_size=1000, shuffle=False)

            for seed in range(n_cv):
                pl.seed_everything(seed + 1)

                models = [
                          ('DNN', DNN(input_size=512, emb_size=500, n_tasks=y_train.shape[1], n_hidden_layers=10, lr=0.001)),
                          ('HardCBMLin', CBMLinear(input_size=512, emb_size=100, n_tasks=y_train.shape[1], n_concepts=_c_train.shape[1], concept_names=CN, task_names=TN, lr=0.001, mode='joint', hard=True)),
                          ('HardCBMDeep', CBMDeep(input_size=512, emb_size=500, n_tasks=y_train.shape[1], n_concepts=_c_train.shape[1], concept_names=CN, task_names=TN, lr=0.001, mode='joint', hard=True)),
                          ('XGBoost2', None),
                          ('DT2', None),
                          ('CEMDeept1', CEMDeep(input_size=512, emb_size=500, n_tasks=y_train.shape[1], n_concepts=_c_train.shape[1],
                                   concept_names=CN, task_names=TN, lr=0.001, mode='joint', hard=True, task_weight=1)),
                          ('StandardDCRt1', StandardDCR(input_size=512, emb_size=500, n_tasks=y_train.shape[1],
                                                n_concepts=_c_train.shape[1], concept_names=CN, task_names=TN, lr=0.001,
                                                mode='joint', hard=True, task_weight=1))
                ]
                # For XGBoost2 and DT2
                cb = SaveBestModelCallbackVal
                cbm = CBMDeep(input_size=512, emb_size=500, n_tasks=y_train.shape[1], n_concepts=_c_train.shape[1], concept_names=CN,
                              task_names=TN, lr=0.001, mode='joint', hard=True)
                cbm.task_weight = 0.01
                cbm.train_loop(train_loader, val_loader, cb, n_epochs)
                cbm.train(False)

                for model_name, model in models:
                    task_to_rulesT, task_to_rulesF = None, None
                    if model_name in ('XGBoost2', 'DT2'):  # Thresholded c preds
                        c_train_preds = cbm.predict_concepts((x_train, _c_train, y_train))
                        c_val_preds = cbm.predict_concepts((x_val, _c_val, y_val))
                        c_test_preds = cbm.predict_concepts((x_test, _c_test, y_test))
                        train_c_acc = get_concept_accuracy(cbm, train_loader)
                        test_c_acc = get_concept_accuracy(cbm, test_loader)
                        if model_name == 'XGBoost2':
                            model = XGBClassifier(random_state=seed+1)
                            model.fit(c_train_preds, y_train)
                            train_acc = model.score(c_train_preds, y_train)
                            test_acc = model.score(c_test_preds, y_test)
                        else:
                            best_accs = (0, 0, 0)
                            best_md = 0
                            for md in range(1, 50, 2):
                                model = DecisionTreeClassifier(max_depth=md, random_state=seed+1)
                                model.fit(c_train_preds, y_train)
                                train_acc = model.score(c_train_preds, y_train)
                                valid_acc = model.score(c_val_preds, y_val)
                                test_acc = model.score(c_test_preds, y_test)
                                print("Train/Test accuracy for DT with max depth", md, ":", train_acc, test_acc)
                                if valid_acc > best_accs[1]:
                                    best_md = md
                                    best_accs = (train_acc, valid_acc, test_acc)
                            print("Best Train/Validation/Test accuracy for DT:", best_accs, "depth", best_md)
                            train_acc, _, test_acc = best_accs
                    else:
                        cb = SaveBestModelCallbackVal
                        model.train_loop(train_loader, val_loader, cb, n_epochs)
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

                    results.append([seed + 1, n_concepts, train_acc, test_acc, train_c_acc, test_c_acc, model_name, task_to_rulesT, task_to_rulesF])
                    pd.DataFrame(results, columns=columns).to_csv(os.path.join(results_dir, results_file))
