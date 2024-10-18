import unittest
from collections import defaultdict
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import joblib

from experiments.celeba.models import DNN, CBMLinear, CBMDeep, CEMDeep, StandardDCR
# from experiments.global_mnist_beta.dataset import load_mnist_data
from experiments.mnist.models_comps import DNN_CNN, CBMLinear_CNN, CBMDeep_CNN, CEMDeep_CNN, StandardDCR_CNN
from experiments.mnist.models_copy import MNISTModel as MNISTModel_old, InputTypes, get_accuracy, get_concept_accuracy, SaveBestModelCallbackVal
from experiments.mnist.rule_logger import RuleLogger
from mnist_dataset import create_single_digit_addition, addition_dataset
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import os
import pandas as pd
from experiments.mnist.models_copy import MNISTModel as MNISTDistilled, MNISTEncoder, ProbRDCat
from lightning.pytorch.loggers import TensorBoardLogger


def c_idx_to_c_sym(c_idx, num_digits, digit_limit, missing_digits=None, digit_prefix=False):
    if missing_digits is None:
        missing_digits = []
    c_names = [*[str(i) for i in range(digit_limit) if i not in missing_digits]] * num_digits
    if digit_prefix:
        return 'd' + str(c_idx // 10) + '_' + c_names[c_idx]
    return c_names[c_idx]

def filter_concepts(c_train, REMOVE_DIGIT_CONCEPTS):
    new_c_train = []
    for example_idx in range(len(c_train)):
        new = []
        for c_idx in range(c_train[example_idx].shape[1]):
            if c_idx not in REMOVE_DIGIT_CONCEPTS:
                new.append(c_train[example_idx][:, c_idx].unsqueeze(-1))
        new_c_train.append(torch.cat(new, dim=-1))
    return new_c_train


class MNISTTest(unittest.TestCase):

    def mnist_cmr(self):
        n_cv = 3
        results = []
        columns = ['seed', 'task_train_acc', 'task_test_acc', 'concepts_train_acc',
                   'concepts_test_acc', 'model',
                   'interv_task_test_acc', 'interv_concepts_test_acc',
                   'perc_correct_rules', 'rules', 'path']
        results_root_dir = f"./results/"
        os.makedirs(results_root_dir, exist_ok=True)
        results_dir = f"./results/mnist_base/"
        current_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_file = "results_" + current_time + ".csv"
        os.makedirs(results_dir, exist_ok=True)

        concept_names, class_names, explanations = create_single_digit_addition(2, 10)
        true_explanations = defaultdict(set)
        nb_ground_truth_rules = 0
        for key in explanations:
            for conj in explanations[key]['explanation'].split(' | '):
                conj = conj.replace('(', '').replace(')', '')  # remove all ( and )
                true_explanations[key].add(conj)
                nb_ground_truth_rules += 1

        x_train, c_train, y_train = addition_dataset(True, 2, 10)
        x_test, c_test, y_test = addition_dataset(False, 2, 10)
        x_train = torch.stack(x_train, dim=1)
        c_train = torch.cat(c_train, dim=-1)
        y_train = F.one_hot(y_train.unsqueeze(-1).long().ravel()).float()

        val_split = 0.1
        train_set_size = int(len(x_train) * (1 - val_split))
        x_val, c_val, y_val = x_train[train_set_size:], c_train[train_set_size:], y_train[train_set_size:]
        x_train, c_train, y_train = x_train[:train_set_size], c_train[:train_set_size], y_train[:train_set_size]

        x_test = torch.stack(x_test, dim=1)
        c_test = torch.cat(c_test, dim=-1)
        y_test = F.one_hot(y_test.unsqueeze(-1).long().ravel()).float()
        train_loader = DataLoader(TensorDataset(x_train, c_train, y_train), batch_size=512, shuffle=False)
        test_loader = DataLoader(TensorDataset(x_test, c_test, y_test), batch_size=512)
        val_loader = DataLoader(TensorDataset(x_val, c_val, y_val), batch_size=512)

        for seed in range(n_cv):
            pl.seed_everything(seed + 1)

            c2s = lambda c_idx: c_idx_to_c_sym(c_idx, 2, 10, [])

            models = []
            cmr_model = MNISTModel_old(encoder=MNISTEncoder(emb_size=500, cp_output=10, number_digits=2),
                                        rule_module=ProbRDCat, n_concepts=c_train.shape[1], n_tasks=y_train.shape[1],
                                        n_rules=20, rule_emb_size=1000, emb_size=500,
                                        selector_input=InputTypes.concepts,
                                        # selector_input=InputTypes.embedding,
                                        concept_names=[c2s(i) for i in range(c_train.shape[1])],
                                        w_c=1,
                                        w_y=1,
                                        w_yF=1,
                                        reset_selector=True,
                                        reset_selector_every_n_epochs=40,
                                        lr=0.0001)
            models.append(("CMR", cmr_model))

            for model_name, model in models:
                checkpoint_cb = ModelCheckpoint(dirpath="./results/mnist_base/CMR/" + str(seed) + "/", save_top_k=1,
                                                monitor="val_loss", mode='min')
                cb = SaveBestModelCallbackVal()
                trainer = pl.Trainer(max_epochs=300, callbacks=[cb, checkpoint_cb], deterministic=True,
                                     )
                trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
                model.load_state_dict(cb.best_state_dict)
                print("Best epoch:", cb.best_epoch)
                print("Best model at path:", checkpoint_cb.best_model_path)
                model = MNISTModel_old.load_from_checkpoint(checkpoint_cb.best_model_path)
                model.to("cpu")
                model.train(False)

                # check the rules
                task_to_rules, task_to_rules_idx = model.aggregate_rules(train_loader, type='most_likely')
                total_nb_wrong_rules = 0
                for task in task_to_rules:
                    nb_wrong_rules = 0
                    for rule in task_to_rules[task]:
                        if not all([t.isdigit() for t in rule.split(' & ')]) or sum(map(int, rule.split(' & '))) != task:
                            nb_wrong_rules += 1  # then we have irrelevant atoms OR the sum is incorrect
                    print('=== Task', task, "===")
                    print('Most Probable Rules', task_to_rules[task])
                    if nb_wrong_rules > 0:
                        print('Nb Wrong Rules', nb_wrong_rules)
                    print()
                    total_nb_wrong_rules += nb_wrong_rules

                train_acc = get_accuracy(model, train_loader)
                test_acc = get_accuracy(model, test_loader)
                train_c_acc = get_concept_accuracy(model, train_loader)
                test_c_acc = get_concept_accuracy(model, test_loader)
                print("Train task accuracy :", train_acc)
                print("Test task accuracy :", test_acc)
                print("Train concept accuracy :", train_c_acc)
                print("Test concept accuracy :", test_c_acc)

                model.intervene = True
                intv_y_acc = get_accuracy(model, test_loader)
                intv_c_acc = get_concept_accuracy(model, test_loader)
                print("Intervention task accuracy :", intv_y_acc)
                print("Intervention concept accuracy :", intv_c_acc)

                results.append([seed + 1, train_acc, test_acc, train_c_acc, test_c_acc, model_name,
                                intv_y_acc, intv_c_acc,
                                1-total_nb_wrong_rules/nb_ground_truth_rules, task_to_rules, checkpoint_cb.best_model_path])
                pd.DataFrame(results, columns=columns).to_csv(os.path.join(results_dir, results_file))

    def mnist_competitors(self):
        use_embeddings = False
        n_cv = 3
        results = []
        columns = ['seed', 'task_train_acc', 'task_test_acc', 'concepts_train_acc', 'concepts_test_acc',
                   'interv_task_test_acc', 'interv_concepts_test_acc',
                   'model', 'perc_correct_rules', 'path']
        results_root_dir = f"./results/"
        os.makedirs(results_root_dir, exist_ok=True)
        results_dir = f"./results/mnist_base/"
        current_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_file = "comp_results_" + current_time + ".csv"
        os.makedirs(results_dir, exist_ok=True)

        concept_names, class_names, explanations = create_single_digit_addition(2, 10)
        if not use_embeddings:
            x_train, c_train, y_train = addition_dataset(True, 2, 10)
            x_test, c_test, y_test = addition_dataset(False, 2, 10)
            x_train = torch.stack(x_train, dim=1)
            c_train = torch.cat(c_train, dim=-1)
            y_train = F.one_hot(y_train.unsqueeze(-1).long().ravel()).float()
        else:
            raise NotImplementedError

        val_split = 0.1
        train_set_size = int(len(x_train) * (1 - val_split))
        x_val, c_val, y_val = x_train[train_set_size:], c_train[train_set_size:], y_train[train_set_size:]
        x_train, c_train, y_train = x_train[:train_set_size], c_train[:train_set_size], y_train[:train_set_size]
        if not use_embeddings:
            x_test = torch.stack(x_test, dim=1)
            c_test = torch.cat(c_test, dim=-1)
            y_test = F.one_hot(y_test.unsqueeze(-1).long().ravel()).float()
        train_loader = DataLoader(TensorDataset(x_train, c_train, y_train), batch_size=512, shuffle=False)
        test_loader = DataLoader(TensorDataset(x_test, c_test, y_test), batch_size=512)
        val_loader = DataLoader(TensorDataset(x_val, c_val, y_val), batch_size=512)
        n_epochs = 300

        for seed in range(n_cv):
            pl.seed_everything(seed + 1)

            models = [
                ('DNN', DNN_CNN(input_size=0, emb_size=500, n_tasks=y_train.shape[1], n_hidden_layers=10, lr=0.0001), DNN_CNN),
                ('HardCBMLin', CBMLinear_CNN(input_size=0, emb_size=100, n_tasks=y_train.shape[1], n_concepts=20,
                                         concept_names=concept_names, task_names=class_names, lr=0.0001, mode='joint', hard=True), CBMLinear_CNN),
                ('HardCBMDeep', CBMDeep_CNN(input_size=0, emb_size=500, n_tasks=y_train.shape[1], n_concepts=20,
                                        concept_names=concept_names, task_names=class_names, lr=0.0001, mode='joint', hard=True), CBMDeep_CNN),
                ('XGBoost2', None, None),
                ('DT2', None, None),
                ('CEMDeep', CEMDeep_CNN(input_size=0, emb_size=500, n_tasks=y_train.shape[1], n_concepts=20,
                                    concept_names=concept_names, task_names=class_names, lr=0.0001, mode='joint', hard=True, task_weight=1), CEMDeep_CNN),
                ('StandardDCR', StandardDCR_CNN(input_size=0, emb_size=500, n_tasks=y_train.shape[1], n_concepts=20,
                                            concept_names=concept_names, task_names=class_names, lr=0.0001,
                                            mode='joint', hard=True, task_weight=1), StandardDCR_CNN)
            ]
            # For XGBoost2 and DT2
            if "XGBoost2" in [m[0] for m in models] or "DT2" in [m[0] for m in models]:
                checkpoint_cb = ModelCheckpoint(dirpath="./results/mnist_base/comps/" + "cbm_DT" + str(seed) + "/",
                                                save_top_k=1,
                                                monitor="val_loss", mode='min')
                cb = SaveBestModelCallbackVal()
                cbm = CBMDeep_CNN(input_size=0, emb_size=500, n_tasks=y_train.shape[1], n_concepts=20, concept_names=concept_names,
                                  task_names=class_names, lr=0.0001, mode='joint', hard=True, task_weight=0.01)
                trainer = pl.Trainer(max_epochs=n_epochs, callbacks=[cb, checkpoint_cb], deterministic=True)
                trainer.fit(model=cbm, train_dataloaders=train_loader, val_dataloaders=val_loader)
                cbm.load_state_dict(cb.best_state_dict)
                print("Best epoch:", cb.best_epoch)
                print("Best model at path:", checkpoint_cb.best_model_path)
                cbm_DT_path = checkpoint_cb.best_model_path


            for model_name, model, model_constructor in models:
                task_to_rulesT, task_to_rulesF, path = None, None, None
                if model_name in ('XGBoost2', 'DT2'):  # Thresholded c preds
                    c_train_preds = cbm.predict_concepts((x_train, c_train, y_train))
                    c_val_preds = cbm.predict_concepts((x_val, c_val, y_val))
                    c_test_preds = cbm.predict_concepts((x_test, c_test, y_test))
                    train_c_acc = get_concept_accuracy(cbm, train_loader)
                    test_c_acc = get_concept_accuracy(cbm, test_loader)
                    if model_name == 'XGBoost2':
                        model = XGBClassifier(random_state=seed + 1)
                        model.fit(c_train_preds, y_train)
                        train_acc = model.score(c_train_preds, y_train)
                        test_acc = model.score(c_test_preds, y_test)
                        intv_y_acc = model.score(c_test, y_test)
                        print("Intervention task accuracy :", intv_y_acc)
                        path = cbm_DT_path
                        joblib.dump(model, f"./results/mnist_base/comps/{model_name}{seed}.joblib")
                    else:
                        best_accs = (0, 0, 0)
                        best_md = 0
                        for md in range(1, 50, 2):
                            model = DecisionTreeClassifier(max_depth=md, random_state=seed + 1)
                            model.fit(c_train_preds, y_train)
                            train_acc = model.score(c_train_preds, y_train)
                            val_acc = model.score(c_val_preds, y_val)
                            test_acc = model.score(c_test_preds, y_test)
                            print("Train/Test accuracy for DT with max depth", md, ":", train_acc, val_acc, test_acc)
                            if train_acc > best_accs[1]:
                                best_md = md
                                best_accs = (train_acc, val_acc, test_acc)
                        print("Best Train/Test accuracy for DT:", best_accs, "depth", best_md)
                        train_acc, val_acc, test_acc = best_accs
                        intv_y_acc = model.score(c_test, y_test)
                        print("Intervention task accuracy :", intv_y_acc)
                        path = cbm_DT_path
                    joblib.dump(model, f"./results/mnist_base/comps/{model_name}{seed}.joblib")
                else:
                    checkpoint_cb = ModelCheckpoint(dirpath="./results/mnist_base/comps/" + model_name + str(seed) + "/",
                                                    save_top_k=1,
                                                    monitor="val_loss", mode='min')
                    cb = SaveBestModelCallbackVal()
                    trainer = pl.Trainer(max_epochs=n_epochs, callbacks=[cb, checkpoint_cb],
                                         deterministic=True)
                    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
                    model.load_state_dict(cb.best_state_dict)
                    print("Best epoch:", cb.best_epoch)
                    print("Best model at path:", checkpoint_cb.best_model_path)
                    path = checkpoint_cb.best_model_path
                    model = model_constructor.load_from_checkpoint(checkpoint_cb.best_model_path)
                    model.to("cpu")
                    model.train(False)

                    train_acc = get_accuracy(model, train_loader)
                    test_acc = get_accuracy(model, test_loader)
                    print("Train accuracy :", train_acc)
                    print("Test accuracy :", test_acc)
                    if isinstance(model, DNN_CNN):
                        train_c_acc, test_c_acc = 0, 0
                        intv_y_acc = None
                    else:
                        train_c_acc = get_concept_accuracy(model, train_loader)
                        test_c_acc = get_concept_accuracy(model, test_loader)
                        model.intervene = True
                        intv_y_acc = get_accuracy(model, test_loader)
                        intv_c_acc = get_concept_accuracy(model, test_loader)
                        print("Intervention task accuracy :", intv_y_acc)
                        print("Intervention concept accuracy :", intv_c_acc)

                results.append([seed + 1, train_acc, test_acc, train_c_acc, test_c_acc, intv_y_acc, None, model_name, None, path])
                pd.DataFrame(results, columns=columns).to_csv(os.path.join(results_dir, results_file))

    def mnist_interventions(self):
        model_map = {
            'CMR': MNISTModel_old,
            'HardCBMLin': CBMLinear_CNN,
            'HardCBMDeep': CBMDeep_CNN,
            'XGBoost2': None,
            'DT2': None,
            'CEMDeep': CEMDeep_CNN,
            'StandardDCR': StandardDCR_CNN
        }
        csv_file_paths = [
            "./results/mnist_base/results_2024-10-25_15-50-54.csv",
            "./results/mnist_base/comp_results_2024-10-21_09-44-17.csv",
        ]

        results = []
        columns = ['seed', 'model', 'task_test_acc_before_int', 'task_test_acc_after_int']
        results_root_dir = f"./results/"
        os.makedirs(results_root_dir, exist_ok=True)
        results_dir = f"./results/mnist_base/"
        current_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_file = "int_results_" + current_time + ".csv"
        os.makedirs(results_dir, exist_ok=True)

        x_test, c_test, y_test = addition_dataset(False, 2, 10)
        x_test = torch.stack(x_test, dim=1)
        c_test = torch.cat(c_test, dim=-1)
        y_test = F.one_hot(y_test.unsqueeze(-1).long().ravel()).float()
        test_loader = DataLoader(TensorDataset(x_test, c_test, y_test), batch_size=512)

        df = pd.concat([pd.read_csv(f) for f in csv_file_paths])
        df = df[["seed", "task_test_acc", "model", "path"]]
        df = df[df["model"] != "DNN"]
        # load model for every row
        for idx, row in df.iterrows():
            test_acc_before_int = row["task_test_acc"]
            seed = row["seed"]
            path = row["path"]
            model_name = row["model"]
            model_constructor = model_map[model_name]
            pl.seed_everything(seed)

            if model_constructor is None:  # DT or XGBoost
                try:
                    file = open(path, "rb")
                    model = joblib.load(file)
                    intv_y_acc = model.score(c_test, y_test)
                    print("Intervention task accuracy :", intv_y_acc)
                except:
                    intv_y_acc = None
            else:
                model = model_constructor.load_from_checkpoint(path)
                model.to("cpu")
                model.train(False)
                model.intervene = True
                intv_y_acc = get_accuracy(model, test_loader)
                intv_c_acc = get_concept_accuracy(model, test_loader)
                print("Intervention task accuracy :", intv_y_acc)
                print("Intervention concept accuracy :", intv_c_acc)

            results.append(
                [seed + 1, model_name, test_acc_before_int, intv_y_acc])
            pd.DataFrame(results, columns=columns).to_csv(os.path.join(results_dir, results_file))

    def mnist_rule_interventions(self):
        n_cv = 3

        results = []
        columns = ['seed', 'task_train_acc', 'task_test_acc', 'concepts_train_acc', 'concepts_test_acc', 'model',
                   'perc_correct_rules', 'learned_and_selected_rules', 'added_rules']
        results_root_dir = f"./results/"
        os.makedirs(results_root_dir, exist_ok=True)
        results_dir = f"./results/mnist_adding_rules/"
        current_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_file = "results_" + current_time + ".csv"
        os.makedirs(results_dir, exist_ok=True)

        concept_names, _, explanations = create_single_digit_addition(2, 10)
        nb_ground_truth_rules = 0
        for key in explanations:
            for _ in explanations[key]['explanation'].split(' | '):
                nb_ground_truth_rules += 1

        x_train, c_train, y_train = addition_dataset(True, 2, 10)
        x_test, c_test, y_test = addition_dataset(False, 2, 10)
        x_train = torch.stack(x_train, dim=1)
        c_train = torch.cat(c_train, dim=-1)
        y_train = F.one_hot(y_train.unsqueeze(-1).long().ravel()).float()
        x_test = torch.stack(x_test, dim=1)
        c_test = torch.cat(c_test, dim=-1)
        y_test = F.one_hot(y_test.unsqueeze(-1).long().ravel()).float()
        train_loader = DataLoader(TensorDataset(x_train, c_train, y_train), batch_size=512)
        test_loader = DataLoader(TensorDataset(x_test, c_test, y_test), batch_size=512)
        c2s = lambda c_idx: c_idx_to_c_sym(c_idx, 2, 10, [], digit_prefix=True)

        for seed in range(n_cv):
            pl.seed_everything(seed + 1)

            model = MNISTModel_old(encoder=MNISTEncoder(emb_size=100, cp_output=10, number_digits=2),
                                   rule_module=ProbRDCat, n_concepts=c_train.shape[1], n_tasks=y_train.shape[1],
                                   n_rules=9, rule_emb_size=1000, emb_size=100,
                                   selector_input=InputTypes.embedding,
                                   reset_selector_every_n_epochs=40,
                                   concept_names=[c2s(i) for i in range(c_train.shape[1])],
                                   lr=0.0001)
            cb = SaveBestModelCallbackVal()
            trainer = pl.Trainer(max_epochs=300, callbacks=[cb], check_val_every_n_epoch=25, deterministic=True)
            trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=train_loader)
            model.load_state_dict(cb.best_state_dict)
            model.train(False)

            # check the rules
            wrong_rules_per_task = defaultdict(list)
            task_to_rules, _ = model.aggregate_rules(train_loader, type='most_likely')
            total_nb_wrong_rules = 0
            all_rules = []
            for task in task_to_rules:
                for rule in task_to_rules[task]:
                    all_rules.append('t%d <- %s' % (task, rule))
                    has_irrelevant_concepts = "(" in rule
                    correct_sum = sum(map(int, rule.replace('(', '').replace(')', '').replace('d0_', '').replace('d1_', '')
                                          .split(' & '))) == task
                    if has_irrelevant_concepts or not correct_sum:
                        print('Learned wrong rule for task', task, ':', rule)
                        wrong_rules_per_task[task].append(rule.replace('(', '').replace(')', '').split(' & '))
                        total_nb_wrong_rules += 1

            train_acc = get_accuracy(model, train_loader, print_report=True)
            test_acc = get_accuracy(model, test_loader)
            print("Train task accuracy :", train_acc)
            print("Test task accuracy :", test_acc)
            train_c_acc = get_concept_accuracy(model, train_loader)
            test_c_acc = get_concept_accuracy(model, test_loader)
            print("Train concept accuracy :", train_c_acc)
            print("Test concept accuracy :", test_c_acc)

            results.append([seed, train_acc, test_acc, train_c_acc, test_c_acc, 'CMRBefore',
                            1-total_nb_wrong_rules/nb_ground_truth_rules, str(all_rules), ''])

            rules_to_add = set()
            added_rules_str = []
            for task in wrong_rules_per_task:
                for wrong_rule_concepts in wrong_rules_per_task[task]:
                    allowed_nb_rules_to_add = len(wrong_rule_concepts) // 2 - 1  # such that 1 wrong rule remains
                    for concept in wrong_rule_concepts:
                        if allowed_nb_rules_to_add == 0:
                            break
                        if concept.startswith('d0_'):
                            digit = int(concept.replace('d0_', ''))
                            if 0 <= task - digit < 10:
                                rules_to_add.add((digit, task - digit + 10))
                                allowed_nb_rules_to_add -= 1
                                print('added rule', 'd0_%d & d1_%d' % (digit, task - digit))
                                added_rules_str.append('t%d <- d0_%d & d1_%d' % (task, digit, task - digit))

            new_rules = [[1 if i in concepts else 0 for i in range(20)] for concepts in rules_to_add]
            assert len(new_rules) != 0
            model.add_rules(new_rules)

            cb = SaveBestModelCallbackVal()
            trainer = pl.Trainer(max_epochs=100, callbacks=[cb], check_val_every_n_epoch=25, deterministic=True)
            trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=train_loader)
            model.load_state_dict(cb.best_state_dict)
            model.train(False)

            # check the rules
            nb_wrong_rules = 0
            task_to_rules, task_to_rules_idx = model.aggregate_rules(train_loader, type='most_likely')
            all_rules = []
            for task in task_to_rules:
                for rule in task_to_rules[task]:
                    all_rules.append('t%d <- %s' % (task, rule))
                    has_irrelevant_concepts = '(' in rule
                    correct_sum = sum(map(int, rule.replace('(', '').replace(')', '').replace('d0_', '').replace('d1_', '')
                                          .split(' & '))) == task
                    if has_irrelevant_concepts or not correct_sum:
                        nb_wrong_rules += 1
            print("Nb wrong rules after finetuning:", nb_wrong_rules)

            train_acc = get_accuracy(model, train_loader)
            test_acc = get_accuracy(model, test_loader)
            train_c_acc = get_concept_accuracy(model, train_loader)
            test_c_acc = get_concept_accuracy(model, test_loader)
            print("Train accuracy :", train_acc)
            print("Test accuracy :", test_acc)

            results.append([seed, train_acc, test_acc, train_c_acc, test_c_acc, 'CMRAfter',
                            1-nb_wrong_rules/nb_ground_truth_rules, str(all_rules), str(added_rules_str)])

            pd.DataFrame(results, columns=columns).to_csv(os.path.join(results_dir, results_file))

    def mnist_cmr_incomplete_concept_set(self):
        n_cv = 3
        results = []
        columns = ['seed', 'task_train_acc', 'task_test_acc', 'concepts_train_acc', 'concepts_test_acc', 'model', 'learned_and_selected_rules']
        results_root_dir = f"./results/"
        os.makedirs(results_root_dir, exist_ok=True)
        results_dir = f"./results/mnist_missing_digit_concepts/"
        current_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_file = "results_" + current_time + ".csv"
        os.makedirs(results_dir, exist_ok=True)

        REMOVE_DIGIT_CONCEPTS = [0, 1]  # remove these concepts for both digits

        concept_names, class_names, explanations = create_single_digit_addition(2, 10)
        true_explanations = defaultdict(set)
        nb_ground_truth_rules = 0
        for key in explanations:
            for conj in explanations[key]['explanation'].split(' | '):
                conj = conj.replace('(', '').replace(')', '').replace('x0', '').replace('x1', '')
                conj = [x for x in conj.split(' & ') if int(x) not in REMOVE_DIGIT_CONCEPTS]
                conj = ' & '.join(conj)
                true_explanations[int(key[1:])].add(conj)
                nb_ground_truth_rules += 1

        x_train, c_train, y_train = addition_dataset(True, 2, 10)
        x_test, c_test, y_test = addition_dataset(False, 2, 10)
        c_train = filter_concepts(c_train, REMOVE_DIGIT_CONCEPTS)
        c_test = filter_concepts(c_test, REMOVE_DIGIT_CONCEPTS)
        x_train = torch.stack(x_train, dim=1)
        c_train = torch.cat(c_train, dim=-1)
        y_train = F.one_hot(y_train.unsqueeze(-1).long().ravel()).float()

        val_split = 0.1
        train_set_size = int(len(x_train) * (1 - val_split))
        x_val, c_val, y_val = x_train[train_set_size:], c_train[train_set_size:], y_train[train_set_size:]
        x_train, c_train, y_train = x_train[:train_set_size], c_train[:train_set_size], y_train[:train_set_size]

        x_test = torch.stack(x_test, dim=1)
        c_test = torch.cat(c_test, dim=-1)
        y_test = F.one_hot(y_test.unsqueeze(-1).long().ravel()).float()
        train_loader = DataLoader(TensorDataset(x_train, c_train, y_train), batch_size=512)
        test_loader = DataLoader(TensorDataset(x_test, c_test, y_test), batch_size=512)
        val_loader = DataLoader(TensorDataset(x_val, c_val, y_val), batch_size=512)

        c2s = lambda c_idx: c_idx_to_c_sym(c_idx, 2, 10, REMOVE_DIGIT_CONCEPTS)

        for seed in range(n_cv):
            pl.seed_everything(seed + 1)

            model = MNISTModel_old(encoder=MNISTEncoder(emb_size=100, cp_output=10-len(REMOVE_DIGIT_CONCEPTS), number_digits=2),
                                   rule_module=ProbRDCat, n_concepts=c_train.shape[1], n_tasks=y_train.shape[1],
                                   n_rules=20, rule_emb_size=1000, emb_size=100,
                                   # selector_input=InputTypes.concepts,
                                   selector_input=InputTypes.embedding,
                                   w_y=1,
                                   reset_selector_every_n_epochs=40,
                                   concept_names=[c2s(i) for i in range(c_train.shape[1])],
                                   lr=0.0001)

            cb = SaveBestModelCallbackVal()
            trainer = pl.Trainer(max_epochs=300, callbacks=[cb], deterministic=True)
            trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
            model.load_state_dict(cb.best_state_dict)
            model.train(False)
            print("Best epoch", cb.best_epoch)

            # check the rules
            task_to_rules, task_to_rules_idx = model.aggregate_rules(train_loader, type='most_likely')
            all_rules = set()
            for task in task_to_rules:
                for rule in task_to_rules[task]:
                    all_rules.add('t%d <- %s' % (task, rule))

            task_to_rules, _ = model.aggregate_rules(train_loader, type='most_likely', inv=True)
            for task in task_to_rules:
                for rule in task_to_rules[task]:
                    all_rules.add('t%d <- %s' % (task, rule))

            train_acc = get_accuracy(model, train_loader)
            test_acc = get_accuracy(model, test_loader)
            train_c_acc = get_concept_accuracy(model, train_loader)
            test_c_acc = get_concept_accuracy(model, test_loader)
            print("Train accuracy :", train_acc)
            print("Test accuracy :", test_acc)

            results.append([seed, train_acc, test_acc, train_c_acc, test_c_acc, 'CMR', str(all_rules)])

            pd.DataFrame(results, columns=columns).to_csv(os.path.join(results_dir, results_file))

    def mnist_comps_incomplete_concept_set(self):
        n_cv = 3
        results = []
        columns = ['seed', 'task_train_acc', 'task_test_acc', 'concepts_train_acc', 'concepts_test_acc', 'model', 'learned_and_selected_rules']
        results_root_dir = f"./results/"
        os.makedirs(results_root_dir, exist_ok=True)
        results_dir = f"./results/comps_mnist_missing_digit_concepts/"
        current_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_file = "results_" + current_time + ".csv"
        os.makedirs(results_dir, exist_ok=True)

        REMOVE_DIGIT_CONCEPTS = [0, 1]  # remove these concepts for both digits

        concept_names, class_names, explanations = create_single_digit_addition(2, 10)
        true_explanations = defaultdict(set)
        nb_ground_truth_rules = 0
        for key in explanations:
            for conj in explanations[key]['explanation'].split(' | '):
                conj = conj.replace('(', '').replace(')', '').replace('x0', '').replace('x1', '')
                conj = [x for x in conj.split(' & ') if int(x) not in REMOVE_DIGIT_CONCEPTS]
                conj = ' & '.join(conj)
                true_explanations[int(key[1:])].add(conj)
                nb_ground_truth_rules += 1

        x_train, c_train, y_train = addition_dataset(True, 2, 10)
        x_test, c_test, y_test = addition_dataset(False, 2, 10)
        c_train = filter_concepts(c_train, REMOVE_DIGIT_CONCEPTS)
        c_test = filter_concepts(c_test, REMOVE_DIGIT_CONCEPTS)
        x_train = torch.stack(x_train, dim=1)
        c_train = torch.cat(c_train, dim=-1)
        y_train = F.one_hot(y_train.unsqueeze(-1).long().ravel()).float()

        val_split = 0.1
        train_set_size = int(len(x_train) * (1 - val_split))
        x_val, c_val, y_val = x_train[train_set_size:], c_train[train_set_size:], y_train[train_set_size:]
        x_train, c_train, y_train = x_train[:train_set_size], c_train[:train_set_size], y_train[:train_set_size]

        x_test = torch.stack(x_test, dim=1)
        c_test = torch.cat(c_test, dim=-1)
        y_test = F.one_hot(y_test.unsqueeze(-1).long().ravel()).float()
        train_loader = DataLoader(TensorDataset(x_train, c_train, y_train), batch_size=512)
        test_loader = DataLoader(TensorDataset(x_test, c_test, y_test), batch_size=512)
        val_loader = DataLoader(TensorDataset(x_val, c_val, y_val), batch_size=512)

        c2s = lambda c_idx: c_idx_to_c_sym(c_idx, 2, 10, REMOVE_DIGIT_CONCEPTS)

        for seed in range(n_cv):
            pl.seed_everything(seed + 1)

            models = [
                ('DNN', DNN_CNN(input_size=0, emb_size=500, n_tasks=y_train.shape[1], n_hidden_layers=10, lr=0.0001), DNN_CNN),
                ('HardCBMLin', CBMLinear_CNN(input_size=0, emb_size=100, n_tasks=y_train.shape[1], n_concepts=16,
                                         concept_names=concept_names, task_names=class_names, lr=0.0001, mode='joint', hard=True), CBMLinear_CNN),
                ('HardCBMDeep', CBMDeep_CNN(input_size=0, emb_size=500, n_tasks=y_train.shape[1], n_concepts=16,
                                        concept_names=concept_names, task_names=class_names, lr=0.0001, mode='joint', hard=True), CBMDeep_CNN),
                ('XGBoost2', None, None),
                ('DT2', None, None),
                ('CEMDeep', CEMDeep_CNN(input_size=0, emb_size=500, n_tasks=y_train.shape[1], n_concepts=16,
                                    concept_names=concept_names, task_names=class_names, lr=0.0001, mode='joint', hard=True, task_weight=1), CEMDeep_CNN),
                ('StandardDCR', StandardDCR_CNN(input_size=0, emb_size=500, n_tasks=y_train.shape[1], n_concepts=16,
                                            concept_names=concept_names, task_names=class_names, lr=0.0001,
                                            mode='joint', hard=True, task_weight=1), StandardDCR_CNN)
            ]
            # For XGBoost2 and DT2
            if "XGBoost2" in [m[0] for m in models] or "DT2" in [m[0] for m in models]:
                cb = SaveBestModelCallbackVal()
                cbm = CBMDeep_CNN(input_size=0, emb_size=500, n_tasks=y_train.shape[1], n_concepts=16,
                                  concept_names=concept_names,
                                  task_names=class_names, lr=0.0001, mode='joint', hard=True, task_weight=0.01)
                trainer = pl.Trainer(max_epochs=300, callbacks=[cb], deterministic=True)
                trainer.fit(model=cbm, train_dataloaders=train_loader, val_dataloaders=val_loader)
                cbm.load_state_dict(cb.best_state_dict)
                print("Best epoch:", cb.best_epoch)

            for model_name, model, model_constructor in models:
                task_to_rulesT, task_to_rulesF = None, None
                if model_name in ('XGBoost2', 'DT2'):  # Thresholded c preds
                    c_train_preds = cbm.predict_concepts((x_train, c_train, y_train))
                    c_val_preds = cbm.predict_concepts((x_val, c_val, y_val))
                    c_test_preds = cbm.predict_concepts((x_test, c_test, y_test))
                    train_c_acc = get_concept_accuracy(cbm, train_loader)
                    test_c_acc = get_concept_accuracy(cbm, test_loader)
                    if model_name == 'XGBoost2':
                        model = XGBClassifier(random_state=seed + 1)
                        model.fit(c_train_preds, y_train)
                        train_acc = model.score(c_train_preds, y_train)
                        test_acc = model.score(c_test_preds, y_test)
                    else:
                        best_accs = (0, 0, 0)
                        best_md = 0
                        for md in range(1, 50, 2):
                            model = DecisionTreeClassifier(max_depth=md, random_state=seed + 1)
                            model.fit(c_train_preds, y_train)
                            train_acc = model.score(c_train_preds, y_train)
                            val_acc = model.score(c_val_preds, y_val)
                            test_acc = model.score(c_test_preds, y_test)
                            print("Train/Test accuracy for DT with max depth", md, ":", train_acc, test_acc)
                            if val_acc > best_accs[1]:
                                best_md = md
                                best_accs = (train_acc, val_acc, test_acc)
                        print("Best Train/Test accuracy for DT:", best_accs, "depth", best_md)
                        train_acc, val_accs, test_acc = best_accs
                else:
                    checkpoint_cb = ModelCheckpoint(
                        dirpath="./results/mnist_base/comps/" + model_name + str(seed) + "/",
                        save_top_k=1,
                        monitor="val_loss", mode='min')
                    cb = SaveBestModelCallbackVal()
                    trainer = pl.Trainer(max_epochs=300, callbacks=[cb, checkpoint_cb], deterministic=True)
                    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
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

                    if isinstance(model, CBMLinear):
                        model.plot_weights()

                results.append([seed + 1, train_acc, test_acc, train_c_acc, test_c_acc, model_name, None])
                pd.DataFrame(results, columns=columns).to_csv(os.path.join(results_dir, results_file))

