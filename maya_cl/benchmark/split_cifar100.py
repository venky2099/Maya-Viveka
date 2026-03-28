# split_cifar100.py — Split-CIFAR-100 benchmark for Maya-Viveka (Paper 5)
# 10 tasks, 10 classes per task. CIL protocol — no task oracle at inference.

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from maya_cl.utils.config import (
    BATCH_SIZE, DATA_DIR, NUM_TASKS, CLASSES_PER_TASK
)

TASK_CLASSES = [
    list(range(i * CLASSES_PER_TASK, (i + 1) * CLASSES_PER_TASK))
    for i in range(NUM_TASKS)
]


def _get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761)
        ),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761)
        ),
    ])
    return train_tf, test_tf


def _subset_by_classes(dataset, classes):
    targets = torch.tensor(dataset.targets)
    mask    = torch.zeros(len(targets), dtype=torch.bool)
    for c in classes:
        mask |= (targets == c)
    indices = mask.nonzero(as_tuple=True)[0].tolist()
    return Subset(dataset, indices)


def get_task_loaders(task_id: int):
    assert 0 <= task_id < NUM_TASKS
    train_tf, test_tf = _get_transforms()

    train_full = datasets.CIFAR100(DATA_DIR, train=True,  download=True, transform=train_tf)
    test_full  = datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=test_tf)

    classes   = TASK_CLASSES[task_id]
    train_sub = _subset_by_classes(train_full, classes)
    test_sub  = _subset_by_classes(test_full,  classes)

    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_sub,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)
    return train_loader, test_loader


def get_all_test_loaders():
    _, test_tf = _get_transforms()
    test_full  = datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=test_tf)
    loaders    = []
    for classes in TASK_CLASSES:
        sub = _subset_by_classes(test_full, classes)
        loaders.append(DataLoader(sub, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=2, pin_memory=True))
    return loaders
