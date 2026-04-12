import verify_provenance  # Maya Research Series -- Nexus Learning Labs, Bengaluru
verify_provenance.stamp()  # logs canary + ORCID on every run

# run_maya_cil.py — Maya-CL Paper 4 main experiment
# Class-Incremental Learning on Split-CIFAR-10
#
# Extends Paper 3 with:
#   - Class-wise ring buffer, interleaved replay
#   - Replay Exemption Gate (pain suppression on replay batches)
#   - Buddhi-gated Vairagya erosion (Viparita Buddhi at task boundaries)
#   - Multi-condition pain signal: loss spike + confidence collapse

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
    REPLAY_VAIRAGYA_PARTIAL_LIFT, REPLAY_PAIN_EXEMPT,
    CIL_BOUNDARY_DECAY, CIL_MAX_VFOUT_PROTECTION,
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


def run_maya_cil(seed: int = 42, replay_size: int = REPLAY_BUFFER_SIZE):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Maya-CIL | seed={seed} | buffer={replay_size}/class | "
          f"replay_ratio={REPLAY_RATIO} | boundary_decay={CIL_BOUNDARY_DECAY}\n")

    model         = MayaCLNet().to(device)
    encoder       = PoissonEncoder(T_STEPS)
    criterion     = nn.CrossEntropyLoss()
    optimizer     = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    affect        = AffectiveState(device)
    sequencer     = TaskSequencer()
    metrics       = CLMetrics(NUM_TASKS)
    logger        = RunLogger("maya_cil")
    test_loaders  = get_all_test_loaders()
    replay_buffer = ReplayBuffer(max_per_class=replay_size)

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

        if task_id > 0:
            with torch.no_grad():
                vairagya_fc1.scores  *= CIL_BOUNDARY_DECAY
                vairagya_fout.scores *= CIL_BOUNDARY_DECAY
                vairagya_fout.scores.clamp_(max=CIL_MAX_VFOUT_PROTECTION)

            affect.reset_experience()

            print(f"  [Task boundary | decay={CIL_BOUNDARY_DECAY} | "
                  f"V-fc1={vairagya_fc1.protection_fraction()*100:.1f}% | "
                  f"V-fout={vairagya_fout.protection_fraction()*100:.1f}% | "
                  f"Buddhi reset]")
            print(f"  {replay_buffer}")

        print(f"━━━ Task {task_id} — classes {current_classes} | seen {seen_classes} ━━━")

        for epoch in range(EPOCHS_PER_TASK):
            model.train()
            epoch_loss = 0.0
            n_replay   = 0

            for batch_idx, (images, labels) in enumerate(tqdm(
                    train_loader, desc=f"  Epoch {epoch+1}/{EPOCHS_PER_TASK}")):

                images = images.to(device)
                labels = labels.to(device)

                # ── Replay injection ───────────────────────────────────────
                is_replay_batch = False
                if replay_buffer.is_ready():
                    r_imgs, r_lbls = replay_buffer.sample(N_REPLAY, device)
                    if r_imgs is not None:
                        images = torch.cat([images, r_imgs], dim=0)
                        labels = torch.cat([labels, r_lbls], dim=0)
                        is_replay_batch = True
                        n_replay += 1

                # ── Forward ────────────────────────────────────────────────
                spike_seq = encoder(images)
                model.reset()
                logits = model(spike_seq)

                with torch.no_grad():
                    v = model.fc1.lif.v
                    if v is not None and v.numel() > 0:
                        v_flat     = v.reshape(-1, fc1_shape[0])
                        post_mean  = v_flat.mean(dim=0)
                        active_fc1 = post_mean.unsqueeze(1).expand(fc1_shape) > 0.05
                    else:
                        active_fc1 = torch.zeros(fc1_shape, dtype=torch.bool, device=device)

                # ── Backward ───────────────────────────────────────────────
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()

                # ── Vairagya masking + Replay Exemption Gate ───────────────
                with torch.no_grad():
                    if model.fc_out.weight.grad is not None:
                        protected_fout = (
                            vairagya_fout.scores >= VAIRAGYA_PROTECTION_THRESHOLD
                        ).clone()
                        for c in current_classes:
                            protected_fout[c, :] = False
                        if is_replay_batch:
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
                        if is_replay_batch:
                            model.fc1.fc.weight.grad[protected_fc1] *= (
                                1.0 - REPLAY_VAIRAGYA_PARTIAL_LIFT
                            )
                        else:
                            model.fc1.fc.weight.grad[protected_fc1] = 0.0

                optimizer.step()
                epoch_loss += loss.item()

                # ── Affective state + plasticity ───────────────────────────
                with torch.no_grad():
                    cur_loss = loss.item()

                    # conf must be computed before check_pain_signal
                    conf = sequencer.update_confidence(logits)

                    # Compute confidence specifically on replay portion if present
                    replay_conf = None
                    if is_replay_batch:
                        n_current = images.shape[0] - N_REPLAY
                        replay_logits = logits[n_current:]
                        replay_probs = torch.softmax(replay_logits.detach(), dim=1)
                        replay_conf = replay_probs.max(dim=1).values.mean().item()

                    if REPLAY_PAIN_EXEMPT and is_replay_batch:
                        pain = False
                    else:
                        pain = sequencer.check_pain_signal(
                            cur_loss, prev_loss, conf, replay_conf=replay_conf
                        )
                        prev_loss = cur_loss

                    spike_rate = active_fc1.float().mean().item()
                    affect.update(conf, pain, spike_rate)

                    bhaya_val  = affect.bhaya.item()
                    buddhi_val = affect.buddhi.item()

                    pain_fc1 = active_fc1 if pain else torch.zeros(
                        fc1_shape, dtype=torch.bool, device=device)
                    if pain:
                        lability_fc1.inject_pain(active_fc1)
                    lability_fc1.decay()

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
                    # fc_out weight decay disabled for CIL — replay gradients
                    # maintain output neuron activation. Vairagya governs fc1 only.
                    # vairagya_fout.apply_decay(model.fc_out.weight.data)

                logger.log_batch(
                    task=task_id, epoch=epoch, batch=batch_idx,
                    loss=cur_loss, confidence=conf, pain_fired=pain,
                    lability_mean=lability_fc1.get().mean().item(),
                    vairagya_protection=vairagya_fc1.protection_fraction(),
                    affective=affect.as_dict()
                )

            # Buffer update after each epoch
            with torch.no_grad():
                for buf_imgs, buf_lbls in train_loader:
                    replay_buffer.update(buf_imgs, buf_lbls)
                    break

            print(f"    Loss: {epoch_loss/len(train_loader):.4f} | "
                  f"Bhaya: {affect.bhaya.item():.3f} | "
                  f"Buddhi: {affect.buddhi.item():.3f} | "
                  f"V-fc1: {vairagya_fc1.protection_fraction()*100:.1f}% | "
                  f"V-fout: {vairagya_fout.protection_fraction()*100:.1f}% | "
                  f"Replay: {n_replay}/{len(train_loader)}")

        print(f"  Evaluating after Task {task_id} [CIL]...")
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
    print(f"\n{'='*50}")
    print(f"  Maya-CIL | buffer={replay_size}/class | seed={seed}")
    print(f"  AA  : {final['AA']}%")
    print(f"  BWT : {final['BWT']}%")
    print(f"  FWT : {final['FWT']}%")
    print(f"{'='*50}")
    logger.log_final(final)
    logger.close()

    return final


if __name__ == "__main__":
    run_maya_cil()