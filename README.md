# Interpretable Concept-Based Memory Reasoning
### NeurIPS 2024

Concept-Based Memory Reasoner (CMR) is a neurosymbolic concept-based model introduced in the paper **"Interpretable Concept-Based Memory Reasoning"** (Debot et al.) presented at NeurIPS 2024. This repository contains the code and resources necessary to reproduce the experiments.

## Overview
The code in this repository supports running experiments on the different datasets to evaluate the proposed Concept-based Memory Reasoner (CMR) model and compare it with competitors. Detailed instructions for replicating experiments and aggregating results are provided in the README file located in `./experiments`.

## Model Summary

CMR integrates neural networks, symbolic reasoning and rule learning to create a powerful concept-based model. It operates by first predicting concepts and then selecting the most appropriate rule from a memory bank using a neural rule selector. CMR learns in an end-to-end fashion, has probabilistic semantics and scales linearly in the number of concepts and tasks.

CMR positions itself nicely within the accuracy-interpretability trade-off. One of the standout features of CMR is its interpretability; users can inspect the learned rules in the memory, providing insights into how predictions can be made at decision-time. This also allows for human interaction (e.g. incorporating expert knowledge) and verification of model properties. Additionally, the rule learning and selection makes CMR a universal binary classifier, which makes it possible to achieve near-black-box accuracy irrespective of which and how many concepts are employed in the architecture.

## Getting Started

### Prerequisites
A conda virtual environment can be set up by doing
```
conda env create -f environment.yml
```
**Note:** This might install a different version of Pytorch than was used for the paper. The version used for the paper was `pytorch=2.1.1`.

### Usage
You can easily use CMR by doing:

```python
import pytorch_lightning as pl
from ./experiments/mnist import CMR

# Define dataloaders
train_dl, val_dl = ...

# ...

# Instantiate the model
model = CMR(
    n_tasks=n_tasks,  # number of classes
    n_concepts=n_concepts,  # number of concepts
    concept_names=concept_names,
    encoder=my_encoder,  # user-defined concept predictor
    learning_rate=0.001,
    emb_size=100,
    rule_emb_size=200, 
    n_rules=20,  # number of rules to learn
    w_y=1  # relative importance of task loss vs prototype regularization
)

# Train the model
trainer = pl.Trainer(max_epochs=200, check_val_every_n_epoch=1)
trainer.fit(model, train_dl, val_dl)
```

## Paper

If you use our code, please consider citing our paper:
```bibtext
@inproceedings{debot2024interpretable,
  title={Interpretable Concept-Based Memory Reasoning},
  author={Debot, David and Barbiero, Pietro and Giannini, Francesco and Ciravegna, Gabriele and Diligenti, Michelangelo and Marra, Giuseppe},
  booktitle={Thirty-eight Conference on Neural Information Processing Systems},
  year={2024}
}
```
