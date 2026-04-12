import verify_provenance  # Maya Research Series -- Nexus Learning Labs, Bengaluru
verify_provenance.stamp()  # logs canary + ORCID on every run

# run_maya_cl.py — Maya-CL Paper 3 final experiment
# TIL evaluation on Split-CIFAR-10
#
# Plasticity mechanisms:
#   1. Nociceptive metaplasticity — pain-triggered lability elevation on fc1
#      Pain signal: loss spike ratio > 1.5 (oracle-free)
#   2. Vairagya gradient masking — protected synapses have gradients zeroed
#      fc_out: seen-classes only, current task always exempt
#      fc1: top 20% most task-connected neurons exempt (0.80 quantile)
#   3. Boundary Vairagya decay — 15% score decay at each task transition
#      Biological grounding: BCM sliding threshold — memories not recently
#      activated fade slightly, releasing capacity for new consolidation
#      This is the key addition for multi-step retention

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from tqdm import tqdm

from maya_cl.utils.config import (
    EPOCHS_PER_TASK, NUM_TASKS, T_STEPS, VAIRAGYA_PROTECTION_THRESHOLD
)
from maya_cl.utils.seed import set_seed
from maya_cl.encoding.poisson import PoissonEncoder
from maya_cl.network.backbone import MayaCLNet
from maya_cl.network.affective_state import AffectiveState
from maya_cl.benchmark.split_cifar10 import (
    get_task_loaders, get_all_test_loaders, TASK_CLASSES
)
from maya_cl.benchmark.task_sequence import TaskSequencer
from maya_cl.plasticity.lability import LabilityMatrix
from maya_cl.plasticity.vairagya_decay import VairagyadDecay
from maya_cl.eval.metrics import CLMetrics, evaluate_task
from maya_cl.eval.logger import RunLogger

# boundary decay rate — 15% score release at each task transition
# grounded in BCM theory: activity-dependent metaplastic threshold decay
BOUNDARY_DECAY = 0.85


def run_maya_cl(seed: int = 42):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Run: Maya-CL Full + Boundary Decay | Seed: {seed}\n")

    model     = MayaCLNet().to(device)
    encoder   = PoissonEncoder(T_STEPS)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    affect    = AffectiveState(device)
    sequencer = TaskSequencer()
    metrics   = CLMetrics(NUM_TASKS)
    logger    = RunLogger("maya_cl")
    test_loaders = get_all_test_loaders()

    # ── Plasticity matrix shapes ──────────────────────────────────
    fc1_shape  = (model.fc1.fc.weight.shape[0],  model.fc1.fc.weight.shape[1])
    fout_shape = (model.fc_out.weight.shape[0],   model.fc_out.weight.shape[1])

    print(f"fc1 weight shape:  {fc1_shape}   "
          f"(~{fc1_shape[0]*fc1_shape[1]/1e6:.1f}M synapses)")
    print(f"fout weight shape: {fout_shape}\n")

    lability_fc1  = LabilityMatrix(fc1_shape,  device)
    vairagya_fc1  = VairagyadDecay(fc1_shape,  device)
    vairagya_fout = VairagyadDecay(fout_shape, device)

    prev_loss = None

    for task_id in range(NUM_TASKS):
        train_loader, _ = get_task_loaders(task_id)
        sequencer.current_task = task_id
        current_classes = TASK_CLASSES[task_id]

        # seen classes including current task
        seen_classes = []
        for t in range(task_id + 1):
            seen_classes.extend(TASK_CLASSES[t])

        # fc_out seen-classes mask
        seen_mask = torch.zeros(fout_shape[0], dtype=torch.bool, device=device)
        for c in seen_classes:
            seen_mask[c] = True

        # ── Boundary Vairagya decay ────────────────────────────────
        # At every task transition (not Task 0), decay all Vairagya scores by 15%
        # This partially releases old protection, preventing saturation
        # and giving new tasks fair access to fc1 capacity
        # Biological basis: BCM sliding modification threshold —
        # synapses not recently activated gradually lose consolidation priority
        if task_id > 0:
            with torch.no_grad():
                vairagya_fc1.scores  *= BOUNDARY_DECAY
                vairagya_fout.scores *= BOUNDARY_DECAY
            print(f"  [Boundary decay applied: scores × {BOUNDARY_DECAY}]")
            print(f"  V-fc1 after decay: "
                  f"{vairagya_fc1.protection_fraction()*100:.1f}% | "
                  f"V-fout after decay: "
                  f"{vairagya_fout.protection_fraction()*100:.1f}%")

        print(f"━━━ Task {task_id} — classes {current_classes} "
              f"| seen: {seen_classes} ━━━")

        for epoch in range(EPOCHS_PER_TASK):
            model.train()
            epoch_loss = 0.0

            for batch_idx, (images, labels) in enumerate(tqdm(
                    train_loader, desc=f"  Epoch {epoch+1}/{EPOCHS_PER_TASK}")):

                images = images.to(device)
                labels = labels.to(device)

                # ── Pass 1: Forward ───────────────────────────────
                spike_seq = encoder(images)
                model.reset()
                logits = model(spike_seq)

                # read fc1 membrane potential BEFORE backward clears it
                with torch.no_grad():
                    v = model.fc1.lif.v
                    if v is not None and v.numel() > 0:
                        v_flat     = v.reshape(-1, fc1_shape[0])
                        post_mean  = v_flat.mean(dim=0)
                        active_fc1 = post_mean.unsqueeze(1).expand(
                            fc1_shape) > 0.05
                    else:
                        active_fc1 = torch.zeros(
                            fc1_shape, dtype=torch.bool, device=device)

                # ── Pass 1: Backward ──────────────────────────────
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()

                # ── Pass 2: Vairagya gradient masking ─────────────
                with torch.no_grad():

                    # fc_out: protect prior-task neurons, exempt current task
                    if model.fc_out.weight.grad is not None:
                        protected_fout = (
                            vairagya_fout.scores >= VAIRAGYA_PROTECTION_THRESHOLD
                        ).clone()
                        for c in current_classes:
                            protected_fout[c, :] = False
                        model.fc_out.weight.grad[protected_fout] = 0.0

                    # fc1: protect prior-task neurons
                    # exempt only top 20% most connected to current task
                    if model.fc1.fc.weight.grad is not None:
                        protected_fc1 = (
                            vairagya_fc1.scores >= VAIRAGYA_PROTECTION_THRESHOLD
                        ).clone()
                        class_weights = model.fc_out.weight[current_classes, :]
                        cw_mean       = class_weights.abs().mean(dim=0)
                        threshold_80  = torch.quantile(cw_mean, 0.80)
                        important_fc1 = cw_mean > threshold_80
                        protected_fc1[important_fc1, :] = False
                        model.fc1.fc.weight.grad[protected_fc1] = 0.0

                optimizer.step()
                epoch_loss += loss.item()

                # ── Pass 3: Affective + plasticity (no grad) ──────
                with torch.no_grad():
                    cur_loss = loss.item()

                    if prev_loss is not None:
                        pain = (cur_loss / (prev_loss + 1e-8)) > 1.5
                    else:
                        pain = False
                    prev_loss = cur_loss

                    conf       = sequencer.update_confidence(logits)
                    spike_rate = active_fc1.float().mean().item()
                    affect.update(conf, pain, spike_rate)

                    # fc1 plasticity
                    pain_fc1 = active_fc1 if pain else torch.zeros(
                        fc1_shape, dtype=torch.bool, device=device)
                    if pain:
                        lability_fc1.inject_pain(active_fc1)
                    lability_fc1.decay()
                    vairagya_fc1.accumulate(active_fc1, pain_fc1)
                    vairagya_fc1.apply_decay(model.fc1.fc.weight.data)

                    # fc_out plasticity — seen classes only
                    logit_mag   = logits.detach().abs().mean(dim=0)
                    active_fout = logit_mag.unsqueeze(1).expand(
                        fout_shape) > logit_mag.mean()
                    active_fout = active_fout & seen_mask.unsqueeze(1)
                    pain_fout   = active_fout if pain else torch.zeros(
                        fout_shape, dtype=torch.bool, device=device)
                    vairagya_fout.accumulate(active_fout, pain_fout)
                    vairagya_fout.apply_decay(model.fc_out.weight.data)

                # ── Log ───────────────────────────────────────────
                logger.log_batch(
                    task=task_id, epoch=epoch, batch=batch_idx,
                    loss=cur_loss, confidence=conf, pain_fired=pain,
                    lability_mean=lability_fc1.get().mean().item(),
                    vairagya_protection=vairagya_fc1.protection_fraction(),
                    affective=affect.as_dict()
                )

            print(f"    Loss: {epoch_loss/len(train_loader):.4f} | "
                  f"Bhaya: {affect.bhaya.item():.3f} | "
                  f"Vairagya: {affect.vairagya.item():.3f} | "
                  f"V-fc1: {vairagya_fc1.protection_fraction()*100:.1f}% | "
                  f"V-fout: {vairagya_fout.protection_fraction()*100:.1f}%")

        # ── Evaluate all tasks — TIL mode ─────────────────────────
        print(f"  Evaluating all tasks after Task {task_id}...")
        acc_dict = {}
        for t in range(NUM_TASKS):
            acc = evaluate_task(
                model, test_loaders[t], device, encoder, T_STEPS,
                task_classes=TASK_CLASSES[t]
            )
            metrics.update(trained_up_to=task_id, task_id=t, accuracy=acc)
            acc_dict[f"task_{t}"] = round(acc * 100, 2)
            print(f"    Task {t} accuracy: {acc*100:.2f}%")

        logger.log_task_summary(task_id, acc_dict, metrics.summary())

    # ── Final report ───────────────────────────────────────────────
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
    run_maya_cl()