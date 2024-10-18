import unittest

import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import torch.nn.functional as F
import lightning.pytorch as pl
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

from experiments.colormnist.models_comps import CBMDeep_CNN, DNN_CNN, CBMLinear_CNN, CEMDeep_CNN, StandardDCR_CNN
from experiments.colormnist.dataset import CustomMNISTDataset
from experiments.colormnist.models import MNISTEncoder
from utils.logic import ProductTNorm
from experiments.mnist.models_copy import MNISTModel as Model, ProbRDCat, get_accuracy, \
    get_concept_accuracy, SaveBestModelCallback, SaveBestModelCallbackVal
from experiments.mnist.models_copy import InputTypes
import os
import pandas as pd


def c_idx_to_c_sym(c_idx, missing_digits=None):
    if missing_digits is None:
        missing_digits = []
    c_names = [*[str(i) for i in range(10) if i not in missing_digits], 'r', 'g']
    return c_names[c_idx]

def get_data():
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    custom_train_dataset = CustomMNISTDataset(mnist_train, random=True)
    custom_test_dataset = CustomMNISTDataset(mnist_test, random=True)
    lst = [(x, c, y) for x, c, y in custom_train_dataset]
    lst2 = [(x, c, y) for x, c, y in custom_test_dataset]
    train_x = torch.stack([x for x, c, y in lst], dim=0)
    train_c = torch.stack([c for x, c, y in lst], dim=0)
    train_y = torch.stack([y for x, c, y in lst], dim=0)
    test_x = torch.stack([x for x, c, y in lst2], dim=0)
    test_c = torch.stack([c for x, c, y in lst2], dim=0)
    test_y = torch.stack([y for x, c, y in lst2], dim=0)
    train_loader = DataLoader(TensorDataset(train_x, train_c, train_y), batch_size=512, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_x, test_c, test_y), batch_size=512, shuffle=False)
    return (train_x, train_c, train_y), (test_x, test_c, test_y), train_loader, test_loader


class ColorMNISTTest(unittest.TestCase):

    def colormnist_cmr(self):
        n_cv = 3

        results = []
        columns = ['seed', 'task_train_acc', 'task_test_acc', 'concepts_train_acc', 'concepts_test_acc', 'model',
                   'learned_and_selected_rules']
        results_root_dir = f"./results/"
        os.makedirs(results_root_dir, exist_ok=True)
        results_dir = f"./results/colormnist_base/"
        current_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_file = "results_" + current_time + ".csv"
        os.makedirs(results_dir, exist_ok=True)

        c2s = lambda c_idx: c_idx_to_c_sym(c_idx, [])

        for seed in range(n_cv):
            pl.seed_everything(seed + 1)

            _, _, train_loader, test_loader = get_data()

            c_train = torch.cat([c for x, c, y in train_loader], dim=0)
            rel_concept_counts = torch.sum(c_train, dim=0) / len(c_train)

            for (name, weight_concepts) in [('CMR_concept_weights', True),
                                            ]:
                model = Model(
                    encoder=MNISTEncoder(cp_output=12, emb_size=500),
                    rule_module=ProbRDCat, n_concepts=12, n_tasks=2,
                    n_rules=6, rule_emb_size=1000, emb_size=500,
                    selector_input=InputTypes.embedding,
                    w_c=1.0,
                    w_y=1.0 if name != 'CMR_concept_weights' else 10.0,
                    w_yF=1.0,
                    reset_selector_every_n_epochs=40,
                    rel_concept_counts=rel_concept_counts, weight_concepts=weight_concepts,
                    concept_names=[c2s(i) for i in range(12)],
                    lr=0.0001)

                cb = SaveBestModelCallbackVal()
                trainer = pl.Trainer(max_epochs=300, callbacks=[cb], deterministic=True, check_val_every_n_epoch=1)
                trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=train_loader)
                model.load_state_dict(cb.best_state_dict)
                print("Best epoch:", cb.best_epoch)
                model.train(False)

                task_to_rules, _ = model.aggregate_rules(train_loader, type='most_likely')
                all_rules = []
                for task in task_to_rules:
                    print('Task', task, 'most probable rules', task_to_rules[task])
                    for rule in task_to_rules[task]:
                        all_rules.append(rule)

                train_acc = get_accuracy(model, train_loader, print_report=True)
                test_acc = get_accuracy(model, test_loader)
                print("Train task accuracy :", train_acc)
                print("Test task accuracy :", test_acc)
                train_c_acc = get_concept_accuracy(model, train_loader)
                test_c_acc = get_concept_accuracy(model, test_loader)
                print("Train concept accuracy :", train_c_acc)
                print("Test concept accuracy :", test_c_acc)

                results.append([seed, train_acc, test_acc, train_c_acc, test_c_acc, name, str(all_rules)])
                pd.DataFrame(results, columns=columns).to_csv(os.path.join(results_dir, results_file))

    def colormnist_competitors(self):
        n_cv = 3
        n_epochs = 300

        results = []
        columns = ['seed', 'task_train_acc', 'task_test_acc', 'concepts_train_acc', 'concepts_test_acc', 'model',
                   'learned_and_selected_rules']
        results_root_dir = f"./results/"
        os.makedirs(results_root_dir, exist_ok=True)
        results_dir = f"./results/colormnist_base/"
        current_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_file = "results_" + current_time + ".csv"
        os.makedirs(results_dir, exist_ok=True)

        c2s = lambda c_idx: c_idx_to_c_sym(c_idx, [])

        CN = [c2s(i) for i in range(12)]
        concept_names = CN
        class_names = ['even', 'odd']

        for seed in range(n_cv):
            pl.seed_everything(seed + 1)

            (x_train, c_train, y_train), (x_test, c_test, y_test), train_loader, test_loader = get_data()

            models = [
                ('DNN', DNN_CNN(input_size=0, emb_size=500, n_tasks=y_train.shape[1], n_hidden_layers=10, lr=0.0001), DNN_CNN),
                ('HardCBMLin', CBMLinear_CNN(input_size=0, emb_size=100, n_tasks=y_train.shape[1], n_concepts=12,
                                         concept_names=concept_names, task_names=class_names, lr=0.0001, mode='joint', hard=True), CBMLinear_CNN),
                ('HardCBMDeep', CBMDeep_CNN(input_size=0, emb_size=500, n_tasks=y_train.shape[1], n_concepts=12,
                                        concept_names=concept_names, task_names=class_names, lr=0.0001, mode='joint', hard=True), CBMDeep_CNN),
                ('XGBoost2', None, None),
                ('DT2', None, None),
                ('CEMDeep', CEMDeep_CNN(input_size=0, emb_size=500, n_tasks=y_train.shape[1], n_concepts=12,
                                    concept_names=concept_names, task_names=class_names, lr=0.0001, mode='joint', hard=True, task_weight=0.01), CEMDeep_CNN),
                ('StandardDCR', StandardDCR_CNN(input_size=0, emb_size=500, n_tasks=y_train.shape[1], n_concepts=12,
                                            concept_names=concept_names, task_names=class_names, lr=0.0001,
                                            mode='joint', hard=True, task_weight=0.01), StandardDCR_CNN)
            ]
            # For XGBoost2 and DT2
            if "XGBoost2" in [m[0] for m in models] or "DT2" in [m[0] for m in models]:
                cb = SaveBestModelCallbackVal()
                cbm = CBMDeep_CNN(input_size=0, emb_size=500, n_tasks=y_train.shape[1], n_concepts=12,
                                  concept_names=CN,
                                  task_names=class_names, lr=0.0001, mode='joint', hard=True, task_weight=0.01)
                trainer = pl.Trainer(max_epochs=n_epochs, callbacks=[cb], accelerator='gpu', deterministic=True,
                                     check_val_every_n_epoch=1)
                trainer.fit(model=cbm, train_dataloaders=train_loader, val_dataloaders=train_loader)
                cbm.load_state_dict(cb.best_state_dict)
                print("Best epoch:", cb.best_epoch)

            for model_name, model, model_constructor in models:
                task_to_rulesT, task_to_rulesF = None, None
                if model_name in ('XGBoost2', 'DT2'):  # Thresholded c preds
                    c_train_preds = cbm.predict_concepts((x_train, c_train, y_train))
                    c_test_preds = cbm.predict_concepts((x_test, c_test, y_test))
                    train_c_acc = get_concept_accuracy(cbm, train_loader)
                    test_c_acc = get_concept_accuracy(cbm, test_loader)
                    if model_name == 'XGBoost2':
                        model = XGBClassifier(random_state=seed + 1)
                        model.fit(c_train_preds, y_train)
                        train_acc = model.score(c_train_preds, y_train)
                        test_acc = model.score(c_test_preds, y_test)
                    else:
                        best_accs = (0, 0)
                        best_md = 0
                        for md in range(1, 50, 2):
                            model = DecisionTreeClassifier(max_depth=md, random_state=seed + 1)
                            model.fit(c_train_preds, y_train)
                            train_acc = model.score(c_train_preds, y_train)
                            test_acc = model.score(c_test_preds, y_test)
                            print("Train/Test accuracy for DT with max depth", md, ":", train_acc, test_acc)
                            if train_acc > best_accs[1]:
                                best_md = md
                                best_accs = (train_acc, test_acc)
                        print("Best Train/Test accuracy for DT:", best_accs, "depth", best_md)
                        train_acc, test_acc = best_accs
                else:
                    checkpoint_cb = ModelCheckpoint(
                        dirpath="./results/mnist_base/comps/" + model_name + str(seed) + "/",
                        save_top_k=1,
                        monitor="val_loss", mode='min')
                    cb = SaveBestModelCallbackVal()
                    trainer = pl.Trainer(max_epochs=n_epochs, callbacks=[cb, checkpoint_cb], accelerator='gpu',
                                         deterministic=True, check_val_every_n_epoch=1)
                    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=train_loader)
                    model.load_state_dict(cb.best_state_dict)
                    print("Best epoch:", cb.best_epoch)
                    print("Best model at path:", checkpoint_cb.best_model_path)
                    model = model_constructor.load_from_checkpoint(checkpoint_cb.best_model_path)
                    model.to("cpu")
                    model.train(False)

                    train_acc = get_accuracy(model, train_loader)
                    test_acc = get_accuracy(model, test_loader)
                    print("Train accuracy :", train_acc)
                    print("Test accuracy :", test_acc)
                    if isinstance(model, DNN_CNN):
                        train_c_acc, test_c_acc = 0, 0
                    else:
                        train_c_acc = get_concept_accuracy(model, train_loader)
                        test_c_acc = get_concept_accuracy(model, test_loader)

                results.append([seed + 1, train_acc, test_acc, train_c_acc, test_c_acc, model_name, None])
                pd.DataFrame(results, columns=columns).to_csv(os.path.join(results_dir, results_file))
