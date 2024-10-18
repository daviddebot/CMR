import torch
import numpy as np
import matplotlib.pyplot as plt
import os


def to_rule_sym2(polarities, relevances, c2s):
    polarities, relevances = polarities.cpu().numpy().round(), relevances.cpu().numpy().round()
    f = lambda pol, rel: 1 if pol == 1 and rel == 1 else 0 if pol == 0 and rel == 1 else 9
    r = [f(polarities[k], relevances[k]) for k in range(len(polarities))]
    r = [("" if r[k] == 1 else "(") + c2s(k) + ("" if r[k] == 1 else ")") for k in range(len(r)) if r[k] in (1, 9)]
    rule_sym = ' & '.join(r)  # only showing positive polarity atoms and not relevant atoms
    return rule_sym


def is_correct(task_idx, rule_idx, pols, rels, c2s):
    current_rule_pols = pols[task_idx][rule_idx]
    current_rule_rels = rels[task_idx][rule_idx]
    r = to_rule_sym2(current_rule_pols, current_rule_rels, c2s).split(" & ")
    return not (not all([t.isdigit() for t in r]) or sum(map(int, r)) != task_idx)


class RuleLogger:
    def __init__(self, n_tasks, n_rules, c2s, log_freq=1):
        self.n_tasks = n_tasks
        self.n_rules = n_rules
        self.rules_per_task_per_epoch = []  # epoch x task x rule x concept x 3
        self.rule_probs_per_sample_per_task_per_rule_per_epoch = []

        self._counter = 0
        self.log_freq = log_freq

        self.c2s = c2s

    def reset(self):
        self.rules_per_task_per_epoch = []
        self.rule_probs_per_sample_per_task_per_rule_per_epoch = []
        self._counter = 0

    def log(self, rules):
        if self._counter % self.log_freq == 0:
            self.rules_per_task_per_epoch.append(torch.clone(rules.detach()))
        self._counter += 1

    def log_probs(self, probs):  # probs: (sample x task x rule)
        if self._counter % self.log_freq == 0:
            self.rule_probs_per_sample_per_task_per_rule_per_epoch.append(torch.clone(probs.detach()))

    def plot_probs(self):
        pols, rels = self.get_pols_rels()
        self.rule_probs_per_sample_per_task_per_rule_per_epoch = [x.cpu() for x in self.rule_probs_per_sample_per_task_per_rule_per_epoch]
        r_probs_per_epoch_sample_task_rule = torch.stack(self.rule_probs_per_sample_per_task_per_rule_per_epoch, dim=0)  # epoch x sample x task x rule
        avg_r_probs_per_epoch_task_rule = torch.mean(r_probs_per_epoch_sample_task_rule, dim=1)  # epoch x task x rule
        avg_r_probs_per_epoch_task_rule = avg_r_probs_per_epoch_task_rule.cpu().numpy()
        for task_idx in range(self.n_tasks):
            avg_r_probs_per_epoch_rule = avg_r_probs_per_epoch_task_rule[:, task_idx, :]
            avg_r_probs_per_epoch_rule = np.transpose(avg_r_probs_per_epoch_rule)  # rule x epoch
            plt.figure()
            plt.title("Task %d" % task_idx)
            plt.xlabel("Epoch")
            plt.ylabel("Avg Prob")
            for rule_idx in range(pols[0].shape[1]):
                clr = "green" if is_correct(task_idx, rule_idx, pols[-1], rels[-1], self.c2s) else "red"
                plt.plot(avg_r_probs_per_epoch_rule[rule_idx], label="Rule %d" % rule_idx, color=clr)
            # plt.legend()
            plt.show()
            plt.pause(0.5)

    def get_pols_rels(self):
        polarities = [tns[:, :, :, 0] for tns in self.rules_per_task_per_epoch]
        relevances = [1-tns[:, :, :, 2] for tns in self.rules_per_task_per_epoch]
        return polarities, relevances

    def interactive_rule_trace(self, model):
        print("Interactive rule trace")
        rule_vars = model.get_all_rule_vars()
        rule_vars_ = torch.stack(self.rules_per_task_per_epoch)
        while True:
            task_idx = input("Task index: ")
            if task_idx == '':
                break
            r_idx_to_r_sym = model.get_rules_sym(rule_vars, task_idx=int(task_idx))
            for r_idx in range(model.effective_n_rules):
                print(r_idx, ":", r_idx_to_r_sym[r_idx])
            rule_idx = input("Rule index: ")
            concept_idx = input("Concept index: ")
            self.get_rule_trace(rule_vars_, int(task_idx), int(rule_idx), int(concept_idx), plot=True)

    def save_full_rule_trace(self, model, path, only_task_rule_pairs=None):
        rule_vars_ = torch.stack(self.rules_per_task_per_epoch)
        for task_idx in range(model.n_tasks):
            for rule_idx in range(model.n_rules):
                for concept_idx in range(model.n_concepts):
                    if only_task_rule_pairs is not None and (task_idx, rule_idx) not in only_task_rule_pairs:
                        continue
                    if not os.path.exists(path + "/task_%d/rule_%d" % (task_idx, rule_idx)):
                        os.makedirs(path + "/task_%d/rule_%d" % (task_idx, rule_idx))
                    self.get_rule_trace(rule_vars_, task_idx, rule_idx, concept_idx, plot=False, save_loc=path + "/task_%d/rule_%d/concept_%d.png" % (task_idx, rule_idx, concept_idx))

    def get_rule_trace(self, rule_vars, task_idx, rule_idx, concept_idx, plot=True, save_loc=None):
        # plot trace of rule
        pospols = rule_vars[:, task_idx, rule_idx, concept_idx, 0].cpu().numpy()
        negpols = rule_vars[:, task_idx, rule_idx, concept_idx, 1].cpu().numpy()
        irrels = rule_vars[:, task_idx, rule_idx, concept_idx, 2].cpu().numpy()
        plt.figure()
        plt.title("Trace Task %d, Rule %d, Concept %d" % (task_idx, rule_idx, concept_idx))
        plt.plot(range(len(pospols)), pospols, label="Positive Polarity", marker='o')
        plt.plot(range(len(negpols)), negpols, label="Negative Polarity", marker='o')
        plt.plot(range(len(irrels)), irrels, label="Irrelevant", marker='o')
        plt.legend()
        if plot:
            plt.show()
        else:
            plt.savefig(save_loc)
            plt.close()


    def interactive_print_evolution(self):
        while True:
            task_idx = int(input("Task index: "))
            rule_idx = int(input("Rule index: "))
            self.print_evolution(only_correct_rules=False, only_task_idx=task_idx, only_rule_idx=rule_idx)

    def print_evolution(self, only_correct_rules=True, only_task_idx=None, only_rule_idx=None):
        pols, rels = self.get_pols_rels()
        for task_idx in range(self.n_tasks):
            if only_task_idx is not None and task_idx != only_task_idx:
                continue
            print("\n=== Rule Evolution Task %d ===" % task_idx)
            for rule_idx in range(len(pols[0][task_idx])):
                if only_rule_idx is not None and rule_idx != only_rule_idx:
                    continue
                previous_rule_pols = None
                previous_rule_rels = None
                previous_rule_sym = None
                if only_correct_rules and not is_correct(task_idx, rule_idx, pols[-1], rels[-1], self.c2s):
                    continue
                print("\n= Rule %d =" % rule_idx)
                for epoch in range(len(pols)):
                    current_rule_pols = pols[epoch][task_idx][rule_idx]
                    current_rule_rels = rels[epoch][task_idx][rule_idx]
                    rule_sym = to_rule_sym2(current_rule_pols, current_rule_rels, self.c2s)
                    if previous_rule_sym is None:
                        print("Epoch %d: %s" % (epoch, rule_sym))
                    elif rule_sym != previous_rule_sym:
                        s = ""
                        for c_idx in range(len(previous_rule_pols)):
                            c = c_idx % (len(previous_rule_pols) // 2)
                            if previous_rule_rels[c_idx].round().item() == 1 and current_rule_rels[c_idx].round().item() == 0:
                                s += " -%d" % c  # becomes irrelevant
                            elif previous_rule_rels[c_idx].round().item() == 0 and current_rule_rels[c_idx].round().item() == 1:  # becomes relevant
                                if current_rule_pols[c_idx].round().item() == 1:
                                    s += " +%d" % c  # and has positive polarity
                                else:
                                    s += " +~%d" % c  # and has negative polarity
                            elif current_rule_rels[c_idx].round().item() == 1:  # stays relevant
                                if previous_rule_pols[c_idx].round().item() == 1 and current_rule_pols[c_idx].round().item() == 0:
                                    s += " ~%d" % c # becomes negative polarity
                                elif previous_rule_pols[c_idx].round().item() == 0 and current_rule_pols[c_idx].round().item() == 1:
                                    s += " %d" % c  # becomes positive polarity
                        print("Epoch %d, diff: %s" % (epoch, s))
                        # print("Epoch %d, Rule %d: %s ==> %s" % (epoch, rule_idx, previous_rule_sym, rule_sym))
                    previous_rule_sym = rule_sym
                    previous_rule_pols = pols[epoch][task_idx][rule_idx]
                    previous_rule_rels = rels[epoch][task_idx][rule_idx]
                print("Final: %s" % previous_rule_sym)
        print()