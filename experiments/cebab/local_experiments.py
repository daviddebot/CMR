import unittest
from collections import defaultdict
import lightning.pytorch as pl
import pytorch_lightning as pl2
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from experiments.celeba.models import DNN, CBMLinear, CBMDeep, CEMDeep, StandardDCR
from experiments.cebab.models import KebabEncoder
from experiments.mnist.models_copy import MNISTModel as MNISTModel_old, InputTypes, \
    SaveBestModelCallback, ProbRDCat, SaveBestModelCallbackVal2, get_concept_accuracy, SaveBestModelCallbackVal, get_accuracy
from experiments.cebab.pretrain import CLASS_NAMES, CONCEPT_SEMANTICS
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pl2.seed_everything(42)

def get_rules(tree, feature_names, class_names):
    # Adapted from https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree
    feature_names = None
    class_names = None

    from sklearn.tree import _tree
    tree_ = tree.tree_
    if feature_names is not None:
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
    else:
        feature_name = [
            i if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

    paths = []
    path = []
    
    def recurse(node, path, paths):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"~{name}"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"{name}"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]
            
    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]
    
    rules = []
    for path in paths:
        rule = ""
        
        for p in path[:-1]:
            if rule != "":
                rule += ","
            rule += str(p)
        if class_names is None:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f",y{l}"
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f",{class_names[l]}"
        rules += [rule]
        
    return rules


class CebabTest(unittest.TestCase):

    def cebab_cmr_ablation1(self):       
        batch_size = 128
        n_cv = 3
        perc_intv = 1
        results = []
        interventions = []
        columns = ['seed', 'task_train_acc', 'task_test_acc', 'concepts_train_acc', 'concepts_test_acc', 'model', 'perc_correct_rules', 'n_rules']
        results_root_dir = f"./results/"
        os.makedirs(results_root_dir, exist_ok=True)
        results_dir = f"./results/"
        results_file = "cmr_results2.csv"
        os.makedirs(results_dir, exist_ok=True)
        if os.path.exists(os.path.join(results_dir, results_file)):
            print("Some results already existing, delete first")

        emb_dir = './embeddings/'
        x_train, c_train, y_train = torch.load(os.path.join(emb_dir, 'train_embeddings.pt'))
        x_test, c_test, y_test = torch.load(os.path.join(emb_dir, 'test_embeddings.pt'))

        y_train = F.one_hot(y_train.long())
        y_test = F.one_hot(y_test.long())

        y_train = y_train[:, 0:1]
        y_test = y_test[:, 0:1]

        n_val = 300
        x_val = x_train[:n_val].clone()
        c_val = c_train[:n_val].clone()
        y_val = y_train[:n_val].clone()
        x_train = x_train[n_val:]
        c_train = c_train[n_val:]
        y_train = y_train[n_val:]

        train_loader = DataLoader(TensorDataset(x_train, c_train.float(), y_train.float()), batch_size=batch_size)
        val_loader = DataLoader(TensorDataset(x_val, c_val.float(), y_val.float()), batch_size=batch_size)
        test_loader = DataLoader(TensorDataset(x_test, c_test.float(), y_test.float()), batch_size=batch_size)

        for seed in range(n_cv):
            for n_rules in [5, 10, 15, 20, 25, 30]:
                pl2.seed_everything(seed+1)
                cmr_model = MNISTModel_old(encoder=KebabEncoder(input_size=x_train.shape[1], emb_size=100, cp_output=c_train.shape[1]),
                                                    rule_module=ProbRDCat, n_concepts=c_train.shape[1], n_tasks=y_train.shape[1],
                                                    n_rules=n_rules, rule_emb_size=100, emb_size=100,  # 100
                                                    selector_input=InputTypes.embedding,
                                                    concept_names=CONCEPT_SEMANTICS,
                                                    w_c=1, w_y=40, w_yF=1, c_pred_in_logic=True, c_pred_in_rec=False,
                                                    orig_rule_sym_to_name='a',  # show rules in 'real' logic rule notation
                                                    reset_selector=False, reset_selector_every_n_epochs=40,
                                                    lr=0.001, mutex=False)

                for (model, model_name) in [(cmr_model, 'CMR')]:
                    checkpoint_cb = ModelCheckpoint(dirpath="./results/cebab_base/CMR/" + str(seed) + "/", save_top_k=1,
                                                    monitor="val_loss", mode='min')
                    cb = SaveBestModelCallbackVal2()
                    trainer = pl.Trainer(max_epochs=100, callbacks=[cb, checkpoint_cb], accelerator="gpu")
                    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
                    model.load_state_dict(cb.best_state_dict)
                    print("Best epoch:", cb.best_epoch)
                    print("Best model at path:", checkpoint_cb.best_model_path)
                    model.to("cpu")
                    model.train(False)

                    train_acc = get_accuracy(model, train_loader)
                    test_acc = get_accuracy(model, test_loader)
                    val_acc = get_accuracy(model, val_loader)
                    train_c_acc = get_concept_accuracy(model, train_loader)
                    test_c_acc = get_concept_accuracy(model, test_loader)
                    print("Train task accuracy :", train_acc)
                    print("Val task accuracy :", val_acc)
                    print("Test task accuracy :", test_acc)
                    print("Train concept accuracy :", train_c_acc)
                    print("Test concept accuracy :", test_c_acc)

                    results.append([seed+1, train_acc, test_acc, train_c_acc, test_c_acc, model_name, None, n_rules])
                    pd.DataFrame(results, columns=columns).to_csv(os.path.join(results_dir, results_file))

    def cebab_cmr_ablation2(self):
        N_C = None  # 12
        N_C = 6

        def DT(CONCEPT_SEMANTICS):
            n_epochs = 100
            batch_size = 128
            emb_size = 300
            n_cv = 1
            task_weight = 0.01

            emb_dir = './embeddings/'
            x_train, c_train, y_train = torch.load(os.path.join(emb_dir, 'train_embeddings.pt'))
            x_test, c_test, y_test = torch.load(os.path.join(emb_dir, 'test_embeddings.pt'))

            if N_C is not None:
                c_train = c_train[:, :N_C]
                c_test = c_test[:, :N_C]
                CONCEPT_SEMANTICS = CONCEPT_SEMANTICS[:N_C]

            y_train = F.one_hot(y_train.long())
            y_test = F.one_hot(y_test.long())

            y_train = y_train[:, 0:1]
            y_test = y_test[:, 0:1]

            n_val = 300
            x_val = x_train[:n_val].clone()
            c_val = c_train[:n_val].clone()
            y_val = y_train[:n_val].clone()
            x_train = x_train[n_val:]
            c_train = c_train[n_val:]
            y_train = y_train[n_val:]
            train_loader = DataLoader(TensorDataset(x_train, c_train.float(), y_train.float()), batch_size=batch_size)
            val_loader = DataLoader(TensorDataset(x_val, c_val.float(), y_val.float()), batch_size=batch_size)
            test_loader = DataLoader(TensorDataset(x_test, c_test.float(), y_test.float()), batch_size=batch_size)

            for seed in range(n_cv):

                models = [
                    ('DT', None)]
                if "XGBoost" in [m[0] for m in models] or "DT" in [m[0] for m in models]:
                    cb = SaveBestModelCallbackVal()
                    cbm = CBMDeep(input_size=x_train.shape[1], emb_size=500, n_tasks=y_train.shape[1], n_concepts=c_train.shape[1],
                                    concept_names=CONCEPT_SEMANTICS,
                                    task_names=CLASS_NAMES, lr=0.0001, mode='joint', hard=True, task_weight=0.01)
                    trainer = pl.Trainer(max_epochs=n_epochs, callbacks=[cb], accelerator='gpu', deterministic=True,
                                        check_val_every_n_epoch=1)
                    trainer.fit(model=cbm, train_dataloaders=train_loader, val_dataloaders=train_loader)
                    cbm.load_state_dict(cb.best_state_dict)
                    print("Best epoch:", cb.best_epoch)

                for model_name, model in models:
                    if model_name == 'DT':
                        c_train_preds = cbm.predict_concepts((x_train, c_train, y_train))
                        c_val_preds = cbm.predict_concepts((x_val, c_val, y_val))
                        c_test_preds = cbm.predict_concepts((x_test, c_test, y_test))
                        train_c_acc = get_concept_accuracy(cbm, train_loader)
                        test_c_acc = get_concept_accuracy(cbm, test_loader)
                        print(test_c_acc)
                        best_accs = (0, 0, 0)
                        best_md = 0
                        for md in range(1, 50, 2):
                            model = DecisionTreeClassifier(max_depth=md, random_state=seed + 1)
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

                    rules = get_rules(model, CONCEPT_SEMANTICS, CLASS_NAMES)
                    dt_rules = []
                    dt_rules_roles = []
                    CN = CONCEPT_SEMANTICS
                    for r in rules:
                        dt_rules.append(r)

                        lst = r.split(",")
                        r_task = lst[-1]
                        if r_task == "y0":
                            continue
                        r_concepts = lst[:-1]
                        assert len(r_concepts) == len(lst) - 1
                        r_pos_c = torch.tensor([1 if f'{i}' in r_concepts and f'~{i}' not in r_concepts else 0 for i in range(len(CN))])  # concepts
                        r_neg_c = torch.tensor([1 if f'~{i}' in r_concepts else 0 for i in range(len(CN))])  # concepts
                        r_irr_c = torch.tensor([1 if f'{i}' not in r_concepts and f'~{i}' not in r_concepts else 0 for i in range(len(CN))])  # concepts
                        r_roles = torch.stack([r_pos_c, r_neg_c, r_irr_c], dim=-1)  # concepts, 3
                        dt_rules_roles.append(r_roles)

                    print("Nb rules:", len(dt_rules))
                    print("Nb rules:", len(set(dt_rules)))
                    print("Nb rules (y=1):", len(dt_rules_roles))
                    dt_rules_roles = torch.stack(dt_rules_roles, dim=0)  # nb_added_rules, concepts, 3
                    print(dt_rules_roles.shape)
                    print("Best Train/Validation/Test accuracy for DT:", best_accs, "depth", best_md)
                    # input()

                    # === (For debugging): evaluate rules ===
                    _dt_rules_roles = dt_rules_roles.unsqueeze(0).repeat(c_test_preds.size(0), 1, 1, 1)
                    c_test_preds = c_test_preds.unsqueeze(1).repeat(1, dt_rules_roles.size(0), 1)
                    y_pred = _dt_rules_roles[:, :, :, 0] * c_test_preds + _dt_rules_roles[:, :, :, 1] * (1-c_test_preds) + _dt_rules_roles[:, :, :, 2]
                    y_pred = y_pred.prod(dim=-1)
                    y_pred = y_pred.sum(dim=-1)
                    print("acc", (y_pred == y_test.squeeze(-1)).sum() / len(y_pred), y_pred.shape, y_test.shape)

                    return dt_rules_roles, best_accs[-1]
                                    
        batch_size = 128
        n_cv = 3
        perc_intv = 1
        results = []
        interventions = []
        columns = ['seed', 'task_train_acc', 'task_test_acc', 'concepts_train_acc', 'concepts_test_acc', 'model', 'perc_correct_rules', 'n_rules', 'use_DT']
        results_root_dir = f"./results/"
        os.makedirs(results_root_dir, exist_ok=True)
        results_dir = f"./results/"
        results_file = f"cmr_results1_{N_C}.csv"
        os.makedirs(results_dir, exist_ok=True)
        if os.path.exists(os.path.join(results_dir, results_file)):
            print("Some results already existing, delete first")

        emb_dir = './embeddings/'
        x_train, c_train, y_train = torch.load(os.path.join(emb_dir, 'train_embeddings.pt'))
        x_test, c_test, y_test = torch.load(os.path.join(emb_dir, 'test_embeddings.pt'))

        if N_C is not None:
            global CONCEPT_SEMANTICS
            c_train = c_train[:, :N_C]
            c_test = c_test[:, :N_C]
            CONCEPT_SEMANTICS = CONCEPT_SEMANTICS[:N_C]

        y_train = F.one_hot(y_train.long())
        y_test = F.one_hot(y_test.long())

        y_train = y_train[:, 0:1]
        y_test = y_test[:, 0:1]

        n_val = 300
        x_val = x_train[:n_val].clone()
        c_val = c_train[:n_val].clone()
        y_val = y_train[:n_val].clone()
        x_train = x_train[n_val:]
        c_train = c_train[n_val:]
        y_train = y_train[n_val:]
        train_loader = DataLoader(TensorDataset(x_train, c_train.float(), y_train.float()), batch_size=batch_size)
        val_loader = DataLoader(TensorDataset(x_val, c_val.float(), y_val.float()), batch_size=batch_size)
        test_loader = DataLoader(TensorDataset(x_test, c_test.float(), y_test.float()), batch_size=batch_size)

        for seed in range(n_cv):
            for use_DT in [True, False]:
                for n_rules in [0, 15]:
                    if not use_DT and n_rules == 0:
                        continue
                    pl2.seed_everything(seed+1)
                    if use_DT:
                        new_rules, dt_acc = DT(CONCEPT_SEMANTICS)
                    cmr_model = MNISTModel_old(encoder=KebabEncoder(input_size=x_train.shape[1], emb_size=100, cp_output=c_train.shape[1]),
                                                        rule_module=ProbRDCat, n_concepts=c_train.shape[1], n_tasks=y_train.shape[1],
                                                        n_rules=n_rules, rule_emb_size=100, emb_size=100,  # 100
                                                        selector_input=InputTypes.embedding,
                                                        #  selector_input=InputTypes.concepts,
                                                        concept_names=CONCEPT_SEMANTICS,
                                                        w_c=1, w_y=40, w_yF=1, c_pred_in_logic=True, c_pred_in_rec=False,
                                                        orig_rule_sym_to_name='a',  # show rules in 'real' logic rule notation
                                                        reset_selector=False, reset_selector_every_n_epochs=40,
                                                        lr=0.001, mutex=False)

                    for (model, model_name) in [(cmr_model, 'CMR')]:

                        if use_DT:
                            new_rules = new_rules.to(model.device)
                            model.add_rules_irr(new_rules)

                        checkpoint_cb = ModelCheckpoint(dirpath="./results/cebab_base/CMR/" + str(seed) + "/", save_top_k=1,
                                                        monitor="val_loss", mode='min')
                        cb = SaveBestModelCallbackVal2()
                        trainer = pl.Trainer(max_epochs=100, callbacks=[cb, checkpoint_cb], accelerator="gpu")
                        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
                        model.load_state_dict(cb.best_state_dict)
                        print("Best epoch:", cb.best_epoch)
                        print("Best model at path:", checkpoint_cb.best_model_path)
                        model.to("cpu")
                        model.train(False)

                        train_acc = get_accuracy(model, train_loader)
                        test_acc = get_accuracy(model, test_loader)
                        val_acc = get_accuracy(model, val_loader)
                        train_c_acc = get_concept_accuracy(model, train_loader)
                        test_c_acc = get_concept_accuracy(model, test_loader)
                        print("Train task accuracy :", train_acc)
                        print("Val task accuracy :", val_acc)
                        print("Test task accuracy :", test_acc)
                        print("Train concept accuracy :", train_c_acc)
                        print("Test concept accuracy :", test_c_acc)

                        results.append([seed+1, train_acc, test_acc, train_c_acc, test_c_acc, model_name, None, n_rules, use_DT])
                        if use_DT:
                            results.append([seed+1, 0, dt_acc, 0, 0, "DT", None, 0, use_DT])
                        pd.DataFrame(results, columns=columns).to_csv(os.path.join(results_dir, results_file))

    def cebab_cmr(self):
        batch_size = 128
        n_cv = 3
        perc_intv = 1
        results = []
        interventions = []
        columns = ['seed', 'task_train_acc', 'task_test_acc', 'concepts_train_acc', 'concepts_test_acc', 'model', 'perc_correct_rules']
        columns_intv = ['seed', 'test_c_acc', 'test_c_intv', 'test_y_acc', 'test_y_intv', 'model', 'perc_intv']
        results_root_dir = f"./results/"
        os.makedirs(results_root_dir, exist_ok=True)
        results_dir = f"./results/cebab_base/"
        results_file = "cmr_results1.csv"
        intv_file = "cmr_interventions1.csv"
        os.makedirs(results_dir, exist_ok=True)
        if os.path.exists(os.path.join(results_dir, results_file)):
            print("Some results already existing, delete first")

        emb_dir = './embeddings/'
        x_train, c_train, y_train = torch.load(os.path.join(emb_dir, 'train_embeddings.pt'))
        x_test, c_test, y_test = torch.load(os.path.join(emb_dir, 'test_embeddings.pt'))

        y_train = F.one_hot(y_train.long())
        y_test = F.one_hot(y_test.long())

        n_val = 300
        x_val = x_train[:n_val].clone()
        c_val = c_train[:n_val].clone()
        y_val = y_train[:n_val].clone()
        x_train = x_train[n_val:]
        c_train = c_train[n_val:]
        y_train = y_train[n_val:]
        train_loader = DataLoader(TensorDataset(x_train, c_train.float(), y_train.float()), batch_size=batch_size)
        val_loader = DataLoader(TensorDataset(x_val, c_val.float(), y_val.float()), batch_size=batch_size)
        test_loader = DataLoader(TensorDataset(x_test, c_test.float(), y_test.float()), batch_size=batch_size)

        for seed in range(n_cv):
            pl2.seed_everything(seed+1)
            cmr_model = MNISTModel_old(encoder=KebabEncoder(input_size=x_train.shape[1], emb_size=100, cp_output=c_train.shape[1]),
                                                 rule_module=ProbRDCat, n_concepts=c_train.shape[1], n_tasks=y_train.shape[1],
                                                 n_rules=15, rule_emb_size=100, emb_size=100,
                                                 selector_input=InputTypes.embedding,
                                                 concept_names=CONCEPT_SEMANTICS,
                                                 w_c=1, w_y=40, w_yF=1, c_pred_in_logic=True, c_pred_in_rec=False,
                                                 orig_rule_sym_to_name='a',  # show rules in 'real' logic rule notation
                                                 reset_selector=True, reset_selector_every_n_epochs=10,
                                                 lr=0.001, mutex=False)

            for (model, model_name) in [(cmr_model, 'CMR')]:
                checkpoint_cb = ModelCheckpoint(dirpath="./results/cebab_base/CMR/" + str(seed) + "/", save_top_k=1,
                                                monitor="val_loss", mode='min')
                cb = SaveBestModelCallbackVal()
                trainer = pl.Trainer(max_epochs=100, callbacks=[cb, checkpoint_cb], 
                                    #  accelerator="gpu"
                                     )
                trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
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
                        rules.append([CLASS_NAMES[task], rule_latex, task_to_rules[task][rule]/len(x_train)*100])
                task_to_rules, _ = model.aggregate_rules(train_loader, type='most_likely', inv=True)
                for task in task_to_rules:
                    print('=== Task', CLASS_NAMES[task], "= False ", "===")
                    for rule in task_to_rules[task]:
                        print(rule, ":", task_to_rules[task][rule])
                        rule_latex = rule.replace('&', '\land').replace('|', '\lor').replace('~', '\\neg ')
                        rules.append([CLASS_NAMES[task], rule_latex, task_to_rules[task][rule] / len(x_train) * 100])
                pd.DataFrame(rules, columns=['task', 'rule', 'support']).to_csv(os.path.join(results_dir, f'rules1_{seed+1}.csv'))

                train_acc = get_accuracy(model, train_loader)
                test_acc = get_accuracy(model, test_loader)
                train_c_acc = get_concept_accuracy(model, train_loader)
                test_c_acc = get_concept_accuracy(model, test_loader)
                print("Train task accuracy :", train_acc)
                print("Test task accuracy :", test_acc)
                print("Train concept accuracy :", train_c_acc)
                print("Test concept accuracy :", test_c_acc)

                results.append([seed+1, train_acc, test_acc, train_c_acc, test_c_acc, model_name, None])
                pd.DataFrame(results, columns=columns).to_csv(os.path.join(results_dir, results_file))

    def cebab_competitors(self):
        n_epochs = 100
        batch_size = 128
        emb_size = 300
        n_cv = 3
        task_weight = 0.01
        perc_intv = 1
        results = []
        interventions = []
        columns = ['seed', 'task_train_acc', 'task_test_acc', 'concepts_train_acc', 'concepts_test_acc', 'model', 'perc_correct_rules']
        columns_intv = ['seed', 'test_c_acc', 'test_c_intv', 'test_y_acc', 'test_y_intv', 'model', 'perc_intv']
        results_root_dir = f"./results/"
        os.makedirs(results_root_dir, exist_ok=True)
        results_dir = f"./results/cebab_base/"
        results_file = "cbm_competitors1.csv"
        intv_file = "cbm_interventions1.csv"
        os.makedirs(results_dir, exist_ok=True)
        if os.path.exists(os.path.join(results_dir, results_file)):
            print("Some results already existing, delete first")

        emb_dir = './embeddings/'
        x_train, c_train, y_train = torch.load(os.path.join(emb_dir, 'train_embeddings.pt'))
        x_test, c_test, y_test = torch.load(os.path.join(emb_dir, 'test_embeddings.pt'))

        y_train = F.one_hot(y_train.long())
        y_test = F.one_hot(y_test.long())

        n_val = 300
        x_val = x_train[:n_val].clone()
        c_val = c_train[:n_val].clone()
        y_val = y_train[:n_val].clone()
        x_train = x_train[n_val:]
        c_train = c_train[n_val:]
        y_train = y_train[n_val:]
        train_loader = DataLoader(TensorDataset(x_train, c_train.float(), y_train.float()), batch_size=batch_size)
        val_loader = DataLoader(TensorDataset(x_val, c_val.float(), y_val.float()), batch_size=batch_size)
        test_loader = DataLoader(TensorDataset(x_test, c_test.float(), y_test.float()), batch_size=batch_size)

        for seed in range(n_cv):
            pl2.seed_everything(seed + 1)

            models = [
                ('DT', None),
                ('XGBoost', None),
                ('DNN', DNN(input_size=x_train.shape[1], emb_size=emb_size, n_tasks=y_train.shape[1], n_hidden_layers=10, lr=0.001)),
                ('HardCBMLin', CBMLinear(input_size=x_train.shape[1], emb_size=emb_size, n_tasks=y_train.shape[1], n_concepts=c_train.shape[1], concept_names=CONCEPT_SEMANTICS, task_names=CLASS_NAMES, lr=0.001, mode='joint', hard=True, task_weight=task_weight)),
                ('SoftCBMLin', CBMLinear(input_size=x_train.shape[1], emb_size=emb_size, n_tasks=y_train.shape[1], n_concepts=c_train.shape[1], concept_names=CONCEPT_SEMANTICS, task_names=CLASS_NAMES, lr=0.001, mode='joint', hard=False, task_weight=task_weight)),
                ('HardCBMDeep', CBMDeep(input_size=x_train.shape[1], emb_size=emb_size, n_tasks=y_train.shape[1], n_concepts=c_train.shape[1], concept_names=CONCEPT_SEMANTICS, task_names=CLASS_NAMES, lr=0.001, mode='joint', hard=True, task_weight=task_weight)),
                ('SoftCBMDeep', CBMDeep(input_size=x_train.shape[1], emb_size=emb_size, n_tasks=y_train.shape[1], n_concepts=c_train.shape[1], concept_names=CONCEPT_SEMANTICS, task_names=CLASS_NAMES, lr=0.001, mode='joint', hard=False, task_weight=task_weight)),
                ('CEMDeep', CEMDeep(input_size=x_train.shape[1], emb_size=emb_size, n_tasks=y_train.shape[1], n_concepts=c_train.shape[1], concept_names=CONCEPT_SEMANTICS, task_names=CLASS_NAMES, lr=0.001, mode='joint', hard=True, task_weight=task_weight)),
                ('StandardDCR', StandardDCR(input_size=x_train.shape[1], emb_size=emb_size, n_tasks=y_train.shape[1], n_concepts=c_train.shape[1], concept_names=CONCEPT_SEMANTICS, task_names=CLASS_NAMES, lr=0.001, mode='joint', hard=True, task_weight=task_weight)),
            ]

            if "XGBoost" in [m[0] for m in models] or "DT" in [m[0] for m in models]:
                cb = SaveBestModelCallbackVal()
                cbm = CBMDeep(input_size=x_train.shape[1], emb_size=500, n_tasks=y_train.shape[1], n_concepts=c_train.shape[1],
                                  concept_names=CONCEPT_SEMANTICS,
                                  task_names=CLASS_NAMES, lr=0.0001, mode='joint', hard=True, task_weight=0.01)
                trainer = pl.Trainer(max_epochs=n_epochs, callbacks=[cb], accelerator='gpu', deterministic=True,
                                     check_val_every_n_epoch=1)
                trainer.fit(model=cbm, train_dataloaders=train_loader, val_dataloaders=train_loader)
                cbm.load_state_dict(cb.best_state_dict)
                print("Best epoch:", cb.best_epoch)

            for model_name, model in models:
                task_to_rulesT, task_to_rulesF = None, None
                if model_name == 'DT':
                    c_train_preds = cbm.predict_concepts((x_train, c_train, y_train))
                    c_val_preds = cbm.predict_concepts((x_val, c_val, y_val))
                    c_test_preds = cbm.predict_concepts((x_test, c_test, y_test))
                    train_c_acc = get_concept_accuracy(cbm, train_loader)
                    test_c_acc = get_concept_accuracy(cbm, test_loader)
                    best_accs = (0, 0, 0)
                    best_md = 0
                    for md in range(1, 50, 2):
                        model = DecisionTreeClassifier(max_depth=md, random_state=seed + 1)
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
                elif model_name == 'XGBoost':
                    c_train_preds = cbm.predict_concepts((x_train, c_train, y_train))
                    c_val_preds = cbm.predict_concepts((x_val, c_val, y_val))
                    c_test_preds = cbm.predict_concepts((x_test, c_test, y_test))
                    train_c_acc = get_concept_accuracy(cbm, train_loader)
                    test_c_acc = get_concept_accuracy(cbm, test_loader)
                    model = XGBClassifier(random_state=seed + 1)
                    model.fit(c_train_preds, y_train)
                    train_acc = model.score(c_train_preds, y_train)
                    test_acc = model.score(c_test_preds, y_test)
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

                    model.intervene = True
                    intv_y_acc = get_accuracy(model, test_loader)
                    intv_c_acc = 0
                    if not isinstance(model, DNN):
                        intv_c_acc = get_concept_accuracy(model, test_loader)
                    print("Intervention task accuracy :", intv_y_acc)
                    print("Intervention concept accuracy :", intv_c_acc)

                    interventions.append([seed + 1, test_c_acc, intv_c_acc, test_acc, intv_y_acc, model_name, perc_intv])
                    pd.DataFrame(interventions, columns=columns_intv).to_csv(os.path.join(results_dir, intv_file))

                    if isinstance(model, CBMLinear):
                        model.plot_weights()

                results.append([seed + 1, train_acc, test_acc, train_c_acc, test_c_acc, model_name, task_to_rulesT])
                pd.DataFrame(results, columns=columns).to_csv(os.path.join(results_dir, results_file))
