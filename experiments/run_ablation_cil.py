import verify_provenance  # Maya Research Series -- Nexus Learning Labs, Bengaluru
verify_provenance.stamp()  # logs canary + ORCID on every run

# run_ablation_cil.py — Maya-Smriti Paper 4 ablation study
# Class-Incremental Learning on Split-CIFAR-10
#
# Five conditions:
#   A: SGD Baseline        — no plasticity, no replay (AA ~18%)
#   B: Replay Only         — ring buffer, no Maya mechanisms
#   C: Maya Only           — lability + Vairagya, no replay (proves CIL gap)
#   D: Full Maya-Smriti    — all mechanisms + replay + gate (main result)
#   E: Maya-Smriti no Gate — all mechanisms + replay, gate OFF (gate ablation)
#
# Condition comparisons:
#   B - A = pure replay contribution
#   C - A = Maya mechanisms alone in CIL (expected ~0, proves replay is necessary)
#   D - B = what Maya adds on top of replay
#   D - E = what the Replay Exemption Gate specifically contributes
#   D - A = total Paper 4 contribution

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from tqdm import tqdm

from maya_cl.utils.config import (
    EPOCHS_PER_TASK, NUM_TASKS, T_STEPS,
    VAIRAGYA_PROTECTION_THRESHOLD,
    REPLAY_BUFFER_SIZE, REPLAY_RATIO,
    REPLAY_VAIRAGYA_PARTIAL_LIFT,
    CIL_BOUNDARY_DECAY,
    BATCH_SIZE,
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
from maya_cl.training.replay_buffer import ReplayBuffer

N_REPLAY = round(BATCH_SIZE * REPLAY_RATIO / (1.0 - REPLAY_RATIO))

CONDITIONS = {
    'baseline': {
        'use_lability':       False,
        'use_vairagya':       False,
        'use_boundary_decay': False,
        'use_replay':         False,
        'use_replay_gate':    False,
        'description': 'SGD Baseline — no plasticity, no replay',
    },
    'replay_only': {
        'use_lability':       False,
        'use_vairagya':       False,
        'use_boundary_decay': False,
        'use_replay':         True,
        'use_replay_gate':    False,
        'description': 'Replay Only — ring buffer, no Maya mechanisms',
    },
    'maya_no_replay': {
        'use_lability':       True,
        'use_vairagya':       True,
        'use_boundary_decay': True,
        'use_replay':         False,
        'use_replay_gate':    False,
        'description': 'Maya Only — Paper 3 mechanisms applied to CIL, no replay',
    },
    'maya_cil': {
        'use_lability':       True,
        'use_vairagya':       True,
        'use_boundary_decay': True,
        'use_replay':         True,
        'use_replay_gate':    True,
        'description': 'Full Maya-Smriti — Affective-Gated Episodic Replay',
    },
    'maya_cil_no_gate': {
        'use_lability':       True,
        'use_vairagya':       True,
        'use_boundary_decay': True,
        'use_replay':         True,
        'use_replay_gate':    False,
        'description': 'Maya-Smriti without Replay Gate — Vairagya/replay interference test',
    },
}


def run_condition(condition_name: str, seed: int = 42,
                  replay_size: int = REPLAY_BUFFER_SIZE) -> dict:
    assert condition_name in CONDITIONS, \
        f"Unknown condition '{condition_name}'. Options: {list(CONDITIONS.keys())}"

    cfg = CONDITIONS[condition_name]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  Condition: {condition_name}")
    print(f"  {cfg['description']}")
    print(f"  Seed: {seed} | Replay size: {replay_size}/class")
    print(f"{'='*60}")

    model     = MayaCLNet().to(device)
    encoder   = PoissonEncoder(T_STEPS)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    affect    = AffectiveState(device)
    sequencer = TaskSequencer()
    metrics   = CLMetrics(NUM_TASKS)
    logger    = RunLogger(f"ablation_cil_{condition_name}")
    test_loaders  = get_all_test_loaders()
    replay_buffer = ReplayBuffer(max_per_class=replay_size) \
        if cfg['use_replay'] else None

    fc1_shape  = (model.fc1.fc.weight.shape[0], model.fc1.fc.weight.shape[1])
    fout_shape = (model.fc_out.weight.shape[0], model.fc_out.weight.shape[1])

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

        if task_id > 0 and cfg['use_vairagya'] and cfg['use_boundary_decay']:
            with torch.no_grad():
                vairagya_fc1.scores  *= CIL_BOUNDARY_DECAY
                vairagya_fout.scores *= CIL_BOUNDARY_DECAY
            affect.reset_experience()
            print(f"  [Boundary decay ×{CIL_BOUNDARY_DECAY} | "
                  f"V-fc1={vairagya_fc1.protection_fraction()*100:.1f}% | "
                  f"Buddhi reset]")

        print(f"━━━ Task {task_id} — classes {current_classes} ━━━")

        for epoch in range(EPOCHS_PER_TASK):
            model.train()
            epoch_loss = 0.0

            for batch_idx, (images, labels) in enumerate(tqdm(
                    train_loader,
                    desc=f"  Epoch {epoch+1}/{EPOCHS_PER_TASK}",
                    leave=False)):

                images = images.to(device)
                labels = labels.to(device)

                # ── Replay injection ───────────────────────────────────────
                is_replay_batch = False
                if cfg['use_replay'] and replay_buffer is not None \
                        and replay_buffer.is_ready():
                    r_imgs, r_lbls = replay_buffer.sample(N_REPLAY, device)
                    if r_imgs is not None:
                        images = torch.cat([images, r_imgs], dim=0)
                        labels = torch.cat([labels, r_lbls], dim=0)
                        is_replay_batch = True

                # ── Forward ────────────────────────────────────────────────
                spike_seq = encoder(images)
                model.reset()
                logits = model(spike_seq)

                with torch.no_grad():
                    v = model.fc1.lif.v
                    if v is not None and v.numel() > 0:
                        v_flat    = v.reshape(-1, fc1_shape[0])
                        post_mean = v_flat.mean(dim=0)
                        active_fc1 = post_mean.unsqueeze(1).expand(fc1_shape) > 0.05
                    else:
                        active_fc1 = torch.zeros(fc1_shape, dtype=torch.bool, device=device)

                # ── Backward ───────────────────────────────────────────────
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()

                # ── Vairagya masking ───────────────────────────────────────
                if cfg['use_vairagya']:
                    with torch.no_grad():
                        if model.fc_out.weight.grad is not None:
                            protected_fout = (
                                vairagya_fout.scores >= VAIRAGYA_PROTECTION_THRESHOLD
                            ).clone()
                            for c in current_classes:
                                protected_fout[c, :] = False
                            if is_replay_batch and cfg['use_replay_gate']:
                                model.fc_out.weight.grad[protected_fout] *= (
                                    1.0 - REPLAY_VAIRAGYA_PARTIAL_LIFT
                                )
                            else:
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
                            if is_replay_batch and cfg['use_replay_gate']:
                                model.fc1.fc.weight.grad[protected_fc1] *= (
                                    1.0 - REPLAY_VAIRAGYA_PARTIAL_LIFT
                                )
                            else:
                                model.fc1.fc.weight.grad[protected_fc1] = 0.0

                optimizer.step()
                epoch_loss += loss.item()

                # ── Affective + plasticity ─────────────────────────────────
                with torch.no_grad():
                    cur_loss = loss.item()

                    # conf must be computed before check_pain_signal
                    conf = sequencer.update_confidence(logits)

                    # replay_conf for CIL-aware pain signal
                    replay_conf = None
                    if is_replay_batch:
                        n_current     = images.shape[0] - N_REPLAY
                        replay_logits = logits[n_current:]
                        replay_probs  = torch.softmax(replay_logits.detach(), dim=1)
                        replay_conf   = replay_probs.max(dim=1).values.mean().item()

                    gate_active = cfg['use_replay'] and cfg['use_replay_gate']
                    if gate_active and is_replay_batch:
                        pain = False
                    else:
                        if cfg['use_lability']:
                            pain = sequencer.check_pain_signal(
                                cur_loss, prev_loss, conf,
                                replay_conf=replay_conf
                            )
                        else:
                            pain = False
                        prev_loss = cur_loss

                    spike_rate = active_fc1.float().mean().item()
                    affect.update(conf, pain, spike_rate)

                    bhaya_val  = affect.bhaya.item()
                    buddhi_val = affect.buddhi.item()

                    if cfg['use_lability']:
                        pain_fc1 = active_fc1 if pain else torch.zeros(
                            fc1_shape, dtype=torch.bool, device=device)
                        if pain:
                            lability_fc1.inject_pain(active_fc1)
                        lability_fc1.decay()
                    else:
                        pain_fc1 = torch.zeros(fc1_shape, dtype=torch.bool, device=device)

                    if cfg['use_vairagya']:
                        vairagya_fc1.accumulate(active_fc1, pain_fc1,
                                                bhaya=bhaya_val, buddhi=buddhi_val)
                        vairagya_fc1.apply_decay(model.fc1.fc.weight.data)

                        logit_mag   = logits.detach().abs().mean(dim=0)
                        active_fout = logit_mag.unsqueeze(1).expand(fout_shape) > logit_mag.mean()
                        active_fout = active_fout & seen_mask.unsqueeze(1)
                        pain_fout   = active_fout if pain else torch.zeros(
                            fout_shape, dtype=torch.bool, device=device)
                        vairagya_fout.accumulate(active_fout, pain_fout,
                                                  bhaya=bhaya_val, buddhi=buddhi_val)
                        # fc_out weight decay disabled for CIL
                        # vairagya_fout.apply_decay(model.fc_out.weight.data)

                logger.log_batch(
                    task=task_id, epoch=epoch, batch=batch_idx,
                    loss=cur_loss, confidence=conf, pain_fired=pain,
                    lability_mean=lability_fc1.get().mean().item(),
                    vairagya_protection=vairagya_fc1.protection_fraction(),
                    affective=affect.as_dict()
                )

            if cfg['use_replay'] and replay_buffer is not None:
                with torch.no_grad():
                    for buf_imgs, buf_lbls in train_loader:
                        replay_buffer.update(buf_imgs, buf_lbls)
                        break

            v_pct  = vairagya_fc1.protection_fraction()*100 if cfg['use_vairagya'] else 0.0
            buf_sz = replay_buffer.size() if replay_buffer else 0
            print(f"    Loss: {epoch_loss/len(train_loader):.4f} | "
                  f"Bhaya: {affect.bhaya.item():.3f} | "
                  f"Buddhi: {affect.buddhi.item():.3f} | "
                  f"V-fc1: {v_pct:.1f}% | "
                  f"Buffer: {buf_sz}")

        print(f"  Evaluating after Task {task_id} [CIL — no oracle]...")
        acc_dict = {}
        for t in range(NUM_TASKS):
            acc = evaluate_task(
                model, test_loaders[t], device, encoder, T_STEPS,
                task_classes=None
            )
            metrics.update(trained_up_to=task_id, task_id=t, accuracy=acc)
            acc_dict[f"task_{t}"] = round(acc * 100, 2)
            print(f"    Task {t}: {acc*100:.2f}%")

        logger.log_task_summary(task_id, acc_dict, metrics.summary())

    metrics.print_matrix()
    final = metrics.summary()
    print(f"\n  Condition : {condition_name}")
    print(f"  AA  : {final['AA']}%")
    print(f"  BWT : {final['BWT']}%")
    print(f"  FWT : {final['FWT']}%")
    logger.log_final(final)
    logger.close()
    return final


def run_buffer_size_sweep(seed: int = 42) -> dict:
    buffer_sizes = [20, 50, 100, 200]
    results = {}
    print("\n" + "="*60)
    print("BUFFER SIZE SWEEP — Full Maya-Smriti")
    print("="*60)
    for m in buffer_sizes:
        print(f"\n→ M={m}/class ({m*10} total)")
        r = run_condition('maya_cil', seed=seed, replay_size=m)
        results[m] = r
        print(f"  M={m}: AA={r['AA']}% | BWT={r['BWT']}% | FWT={r['FWT']}%")
    print("\n" + "="*60)
    print(f"{'M/class':<12} {'Total':<12} {'AA':<10} {'BWT':<10} {'FWT'}")
    print("-"*55)
    for m, r in results.items():
        print(f"{m:<12} {m*10:<12} {r['AA']:<10} {r['BWT']:<10} {r['FWT']}")
    return results


if __name__ == "__main__":
    all_results = {}

    # A and B already completed — start from C
    # Once C, D, E are done, manually add A and B results below for the table
    for cond in ['maya_no_replay', 'maya_cil', 'maya_cil_no_gate']:
        all_results[cond] = run_condition(cond, seed=42)

    # Inject previously completed results
    all_results['baseline']    = {'AA': 17.98, 'BWT': -86.49, 'FWT': -10.0}
    all_results['replay_only'] = {'AA': 31.07, 'BWT': -69.38, 'FWT': -10.0}

    print("\n" + "="*70)
    print("MAYA-SMRITI PAPER 4 — ABLATION TABLE — Split-CIFAR-10 CIL")
    print("="*70)
    print(f"{'Condition':<25} {'Lability':<10} {'Vairagya':<10} "
          f"{'Replay':<10} {'Gate':<8} {'AA':<8} {'BWT':<8} {'FWT'}")
    print("-"*70)

    flags = {
        'baseline':         ('✗', '✗', '✗', '✗'),
        'replay_only':      ('✗', '✗', '✓', '✗'),
        'maya_no_replay':   ('✓', '✓', '✗', '✗'),
        'maya_cil':         ('✓', '✓', '✓', '✓'),
        'maya_cil_no_gate': ('✓', '✓', '✓', '✗'),
    }

    order = ['baseline', 'replay_only', 'maya_no_replay', 'maya_cil', 'maya_cil_no_gate']
    for cond in order:
        if cond not in all_results:
            continue
        r = all_results[cond]
        l, v, rp, g = flags[cond]
        print(f"{cond:<25} {l:<10} {v:<10} {rp:<10} {g:<8} "
              f"{r['AA']:<8} {r['BWT']:<8} {r['FWT']}")

    print("="*70)

    if 'maya_cil' in all_results and 'maya_cil_no_gate' in all_results:
        d = all_results['maya_cil']
        e = all_results['maya_cil_no_gate']
        print(f"\nReplay Gate contribution (D - E):")
        print(f"  ΔAA  = {d['AA']  - e['AA']:.2f}%")
        print(f"  ΔBWT = {d['BWT'] - e['BWT']:.2f}%")

    # Uncomment to run buffer size sweep:
    # run_buffer_size_sweep(seed=42)