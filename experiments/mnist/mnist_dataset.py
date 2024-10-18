
# Copied and adapted from https://github.com/pietrobarbiero/interpretable-relational-reasoning/blob/reasoning/experiments/distant_supervision/datasets.py

import random
import typing
from collections.abc import Sequence
from math import ceil
from pathlib import Path
from typing import List, Tuple, Generator, Union

import torch
import torchvision
from torchvision import transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset, DataLoader
from itertools import product
from collections import defaultdict

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
data_root = Path(__file__).parent / ".." / "data"

def get_mnist_data(train: bool) -> MNIST:
    return torchvision.datasets.MNIST(
        root=str(data_root / "raw/"), train=train, download=True, transform=transform
    )


def addition_dataset(train, num_digits, digit_limit=10):
    dataset = get_mnist_data(train)
    X, y = dataset.data, dataset.targets

    X = X[y < digit_limit]
    y = y[y < digit_limit]

    X = torch.unsqueeze(X, 1).float()
    size = len(X) // num_digits
    X, y = torch.split(X, size), torch.split(y, size)

    if len(X) % num_digits != 0:
        X = X[:-1]
        y = y[:-1]

    c = [torch.zeros((len(X[0]), digit_limit)).float() for _ in range(len(X))]
    for i, ys in enumerate(y):
        for j, yi in enumerate(ys):
            c[i][j, yi] = 1.0

    y = torch.sum(torch.stack(y, 0), 0)
    return X, c, y



def create_single_digit_addition(num_digits, digit_limit=10):
    concept_names = ["x%d%d" % (i, j) for i, j in product(range(num_digits), range(digit_limit))]


    sums = defaultdict(list)
    for d in product(*[range(digit_limit) for _ in range(num_digits)]):
        conj = []
        z = 0
        for i, n in enumerate(d):
            conj.append("x%d%d" % (i,n))
            z += n
        sums[z].append("(" + " & ".join(conj) + ")")


    explanations = {}
    class_names = ["z%d" % z for z in range(digit_limit*num_digits - num_digits + 1)]
    for z in range(digit_limit*num_digits - num_digits + 1):

        explanations["z%d" % z] = {"name": "%d" % z,
                                   "explanation": "(" + " | ".join(sums[z]) + ")"}

    return concept_names, class_names, explanations


if __name__ == '__main__':

    number_digits = 3

    print(create_single_digit_addition(number_digits))

    X, y = addition_dataset(True, 2*number_digits)

    dataset = TensorDataset(*X, y)
    loader = DataLoader(
        dataset,
        batch_size=100
    )

    for batch_idx, I in enumerate(loader):
        X = I[:-1]
        y = I[-1]
        print(batch_idx, y.shape, [x.shape for x in X])



