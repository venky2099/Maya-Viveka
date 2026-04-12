import verify_provenance  # Maya Research Series -- Nexus Learning Labs, Bengaluru
verify_provenance.stamp()  # logs canary + ORCID on every run

# run_baseline.py — SGD fine-tuning baseline
# No plasticity, no metaplasticity, no Vairagya decay
# Expected result: ~19-20% AA — catastrophic forgetting floor
# This is the bottom anchor for all Paper 3 comparison tables

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from tqdm import tqdm

from maya_cl.utils.config import (
    EPOCHS_PER_TASK, NUM_TASKS, T_STEPS, RESULTS_DIR
)
from maya_cl.utils.seed import set_seed
from maya_cl.encoding.poisson import PoissonEncoder
from maya_cl.network.backbone import MayaCLNet
from maya_cl.benchmark.split_cifar10 import get_task_loaders, get_all_test_loaders
from maya_cl.eval.metrics import CLMetrics, evaluate_task
from maya_cl.eval.logger import RunLogger


def run_baseline(seed: int = 42):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Run: SGD Baseline | Seed: {seed}\n")

    model    = MayaCLNet().to(device)
    encoder  = PoissonEncoder(T_STEPS)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    metrics   = CLMetrics(NUM_TASKS)
    logger    = RunLogger("baseline")
    test_loaders = get_all_test_loaders()

    for task_id in range(NUM_TASKS):
        train_loader, _ = get_task_loaders(task_id)
        print(f"━━━ Task {task_id} ━━━")

        for epoch in range(EPOCHS_PER_TASK):
            model.train()
            epoch_loss = 0.0

            for batch_idx, (images, labels) in enumerate(tqdm(
                    train_loader, desc=f"  Epoch {epoch+1}/{EPOCHS_PER_TASK}")):

                images = images.to(device)
                labels = labels.to(device)

                spike_seq = encoder(images)    # [T, B, C, H, W]
                model.reset()
                logits = model(spike_seq)      # [B, 10]

                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                logger.log_batch(
                    task=task_id, epoch=epoch, batch=batch_idx,
                    loss=loss.item(), confidence=0.0, pain_fired=False,
                    lability_mean=1.0, vairagya_protection=0.0,
                    affective={"shraddha":0,"bhaya":0,"vairagya":0,"spanda":0}
                )

            print(f"    Loss: {epoch_loss/len(train_loader):.4f}")

        # ── Evaluate all tasks - TIL mode ────────────────────────
        print(f"  Evaluating all tasks after Task {task_id}...")
        acc_dict = {}
        for t in range(NUM_TASKS):
            from maya_cl.benchmark.split_cifar10 import TASK_CLASSES
            acc = evaluate_task(
                model, test_loaders[t], device, encoder, T_STEPS,
                task_classes=TASK_CLASSES[t]
            )
            metrics.update(trained_up_to=task_id, task_id=t, accuracy=acc)
            acc_dict[f"task_{t}"] = round(acc * 100, 2)
            print(f"    Task {t} accuracy: {acc * 100:.2f}%")

        logger.log_task_summary(task_id, acc_dict, metrics.summary())

    # ── Final report ──────────────────────────────────────────────
    metrics.print_matrix()
    final = metrics.summary()
    print(f"\n{'='*40}")
    print(f"  AA  : {final['AA']}%")
    print(f"  BWT : {final['BWT']}%")
    print(f"  FWT : {final['FWT']}%")
    print(f"{'='*40}")
    logger.log_final(final)
    logger.close()


if __name__ == "__main__":
    run_baseline()