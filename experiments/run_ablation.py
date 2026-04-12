import verify_provenance  # Maya Research Series -- Nexus Learning Labs, Bengaluru
verify_provenance.stamp()  # logs canary + ORCID on every run

# run_ablation.py — Maya-CL Paper 3 ablation study
# TIL evaluation on Split-CIFAR-10
#
# Three conditions:
#   A: SGD baseline — no plasticity (already run, reference only)
#   B: Lability only — pain-triggered metaplasticity, NO Vairagya gradient masking
#      Isolates: does nociceptive lability alone help retention?
#   C: Full Maya-CL — lability + Vairagya + boundary decay (main experiment)
#      Isolates: what does Vairagya add on top of lability?
#
# Difference B-A = lability contribution
# Difference C-B = Vairagya contribution
# Difference C-A = total Maya-CL contribution

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

BOUNDARY_DECAY = 0.85


def run_condition(condition: str, seed: int = 42):
    """
    condition: 'lability_only' or 'full'
    'lability_only' = pain metaplasticity active, Vairagya gradient masking OFF
    'full'          = both mechanisms active (same as run_maya_cl.py)
    """
    assert condition in ('lability_only', 'full')

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Run: Ablation — {condition.upper()} | Seed: {seed}\n")

    model     = MayaCLNet().to(device)
    encoder   = PoissonEncoder(T_STEPS)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    affect    = AffectiveState(device)
    sequencer = TaskSequencer()
    metrics   = CLMetrics(NUM_TASKS)
    logger    = RunLogger(f"ablation_{condition}")
    test_loaders = get_all_test_loaders()

    fc1_shape  = (model.fc1.fc.weight.shape[0],  model.fc1.fc.weight.shape[1])
    fout_shape = (model.fc_out.weight.shape[0],   model.fc_out.weight.shape[1])

    lability_fc1  = LabilityMatrix(fc1_shape,  device)
    vairagya_fc1  = VairagyadDecay(fc1_shape,  device)
    vairagya_fout = VairagyadDecay(fout_shape, device)

    prev_loss = None

    for task_id in range(NUM_TASKS):
        train_loader, _ = get_task_loaders(task_id)
        sequencer.current_task = task_id
        current_classes = TASK_CLASSES[task_id]

        seen_classes = []
        for t in range(task_id + 1):
            seen_classes.extend(TASK_CLASSES[t])

        seen_mask = torch.zeros(fout_shape[0], dtype=torch.bool, device=device)
        for c in seen_classes:
            seen_mask[c] = True

        # boundary decay only in full condition
        if task_id > 0 and condition == 'full':
            with torch.no_grad():
                vairagya_fc1.scores  *= BOUNDARY_DECAY
                vairagya_fout.scores *= BOUNDARY_DECAY

        print(f"━━━ Task {task_id} — classes {current_classes} ━━━")

        for epoch in range(EPOCHS_PER_TASK):
            model.train()
            epoch_loss = 0.0

            for batch_idx, (images, labels) in enumerate(tqdm(
                    train_loader, desc=f"  Epoch {epoch+1}/{EPOCHS_PER_TASK}")):

                images = images.to(device)
                labels = labels.to(device)

                spike_seq = encoder(images)
                model.reset()
                logits = model(spike_seq)

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

                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()

                # Vairagya gradient masking — ONLY in full condition
                if condition == 'full':
                    with torch.no_grad():
                        if model.fc_out.weight.grad is not None:
                            protected_fout = (
                                vairagya_fout.scores >= VAIRAGYA_PROTECTION_THRESHOLD
                            ).clone()
                            for c in current_classes:
                                protected_fout[c, :] = False
                            model.fc_out.weight.grad[protected_fout] = 0.0

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

                    # lability always active in both conditions
                    pain_fc1 = active_fc1 if pain else torch.zeros(
                        fc1_shape, dtype=torch.bool, device=device)
                    if pain:
                        lability_fc1.inject_pain(active_fc1)
                    lability_fc1.decay()
                    vairagya_fc1.accumulate(active_fc1, pain_fc1)
                    vairagya_fc1.apply_decay(model.fc1.fc.weight.data)

                    logit_mag   = logits.detach().abs().mean(dim=0)
                    active_fout = logit_mag.unsqueeze(1).expand(
                        fout_shape) > logit_mag.mean()
                    active_fout = active_fout & seen_mask.unsqueeze(1)
                    pain_fout   = active_fout if pain else torch.zeros(
                        fout_shape, dtype=torch.bool, device=device)
                    vairagya_fout.accumulate(active_fout, pain_fout)
                    vairagya_fout.apply_decay(model.fc_out.weight.data)

                logger.log_batch(
                    task=task_id, epoch=epoch, batch=batch_idx,
                    loss=cur_loss, confidence=conf, pain_fired=pain,
                    lability_mean=lability_fc1.get().mean().item(),
                    vairagya_protection=vairagya_fc1.protection_fraction(),
                    affective=affect.as_dict()
                )

            print(f"    Loss: {epoch_loss/len(train_loader):.4f} | "
                  f"Bhaya: {affect.bhaya.item():.3f} | "
                  f"V-fc1: {vairagya_fc1.protection_fraction()*100:.1f}%")

        print(f"  Evaluating after Task {task_id}...")
        acc_dict = {}
        for t in range(NUM_TASKS):
            acc = evaluate_task(
                model, test_loaders[t], device, encoder, T_STEPS,
                task_classes=TASK_CLASSES[t]
            )
            metrics.update(trained_up_to=task_id, task_id=t, accuracy=acc)
            acc_dict[f"task_{t}"] = round(acc * 100, 2)
            print(f"    Task {t}: {acc*100:.2f}%")

        logger.log_task_summary(task_id, acc_dict, metrics.summary())

    metrics.print_matrix()
    final = metrics.summary()
    print(f"\n{'='*40}")
    print(f"  Condition : {condition}")
    print(f"  AA  : {final['AA']}%")
    print(f"  BWT : {final['BWT']}%")
    print(f"  FWT : {final['FWT']}%")
    print(f"{'='*40}")
    logger.log_final(final)
    logger.close()
    return final


if __name__ == "__main__":
    print("\n" + "="*60)
    print("ABLATION B: Lability only (no Vairagya gradient masking)")
    print("="*60)
    results_b = run_condition('lability_only', seed=42)

    print("\n" + "="*60)
    print("ABLATION C: Full Maya-CL (lability + Vairagya + boundary decay)")
    print("="*60)
    results_c = run_condition('full', seed=42)

    print("\n" + "="*60)
    print("ABLATION SUMMARY")
    print("="*60)
    print(f"  SGD Baseline (A)  : AA=64.57% | BWT=-28.25% | FWT=+37.89%")
    print(f"  Lability only (B) : AA={results_b['AA']}% | "
          f"BWT={results_b['BWT']}% | FWT={results_b['FWT']}%")
    print(f"  Full Maya-CL (C)  : AA={results_c['AA']}% | "
          f"BWT={results_c['BWT']}% | FWT={results_c['FWT']}%")
    print(f"\n  Vairagya contribution (C-B):")
    print(f"    ΔAA  = {results_c['AA']  - results_b['AA']:.2f}%")
    print(f"    ΔBWT = {results_c['BWT'] - results_b['BWT']:.2f}%")
    print(f"    ΔFWT = {results_c['FWT'] - results_b['FWT']:.2f}%")
    print("="*60)