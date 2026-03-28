# run_viveka_cil.py — Maya-Viveka Paper 5 main experiment
# Split-CIFAR-100 CIL, 10 tasks, no task oracle at inference.

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
    BATCH_SIZE, REPLAY_PAIN_EXEMPT,
)
from maya_cl.utils.seed import set_seed
from maya_cl.encoding.poisson import PoissonEncoder
from maya_cl.network.backbone import MayaVivekaNet
from maya_cl.network.affective_state import AffectiveState
from maya_cl.benchmark.split_cifar100 import (
    get_task_loaders, get_all_test_loaders, TASK_CLASSES
)
from maya_cl.benchmark.task_sequence import TaskSequencer
from maya_cl.plasticity.lability import LabilityMatrix
from maya_cl.plasticity.vairagya_decay import VairagyadDecay
from maya_cl.plasticity.viveka import VivekaConsistency
from maya_cl.eval.metrics import CLMetrics, evaluate_task
from maya_cl.eval.logger import RunLogger
from maya_cl.training.replay_buffer import ReplayBuffer

N_REPLAY = round(BATCH_SIZE * REPLAY_RATIO / (1.0 - REPLAY_RATIO))


def run_viveka_cil(seed: int = 42, replay_size: int = REPLAY_BUFFER_SIZE) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*65}")
    print(f"  Maya-Viveka — Paper 5")
    print(f"  Split-CIFAR-100 CIL | 10 tasks | seed={seed}")
    print(f"  Viveka: ON | Orthogonal head: ON | Replay: {replay_size}/class")
    print(f"{'='*65}\n")

    model         = MayaVivekaNet(use_orthogonal_head=False).to(device)
    encoder       = PoissonEncoder(T_STEPS)
    criterion     = nn.CrossEntropyLoss()
    optimizer     = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    affect        = AffectiveState(device)
    sequencer     = TaskSequencer()
    metrics       = CLMetrics(NUM_TASKS)
    logger        = RunLogger("viveka_cil")
    test_loaders  = get_all_test_loaders()
    replay_buffer = ReplayBuffer(max_per_class=replay_size)

    fc1_shape = (model.fc1.fc.weight.shape[0], model.fc1.fc.weight.shape[1])
    print(f"fc1 shape: {fc1_shape} (~{fc1_shape[0]*fc1_shape[1]/1e6:.1f}M synapses)\n")

    lability_fc1 = LabilityMatrix(fc1_shape, device)
    vairagya_fc1 = VairagyadDecay(fc1_shape, device)
    viveka       = VivekaConsistency(fc1_shape, device)

    prev_loss = None

    for task_id in range(NUM_TASKS):
        train_loader, _ = get_task_loaders(task_id)
        sequencer.on_task_boundary(task_id)
        current_classes = TASK_CLASSES[task_id]

        if task_id > 0:
            with torch.no_grad():
                vairagya_fc1.scores *= CIL_BOUNDARY_DECAY
            affect.notify_task_boundary()
            viveka.on_task_boundary()
            print(f"  [Boundary | V-fc1={vairagya_fc1.protection_fraction()*100:.1f}% | "
                  f"Consist={viveka.mean_consistency():.3f} | {replay_buffer}]")

        print(f"\n{'─'*65}")
        print(f"  Task {task_id} — classes {current_classes}")
        print(f"{'─'*65}")

        for epoch in range(EPOCHS_PER_TASK):
            model.train()
            epoch_loss = 0.0
            n_replay   = 0

            for batch_idx, (images, labels) in enumerate(tqdm(
                    train_loader, desc=f"  Epoch {epoch+1}/{EPOCHS_PER_TASK}")):

                images = images.to(device)
                labels = labels.to(device)

                is_replay_batch = False
                if replay_buffer.is_ready():
                    r_imgs, r_lbls = replay_buffer.sample(N_REPLAY, device)
                    if r_imgs is not None:
                        images = torch.cat([images, r_imgs], dim=0)
                        labels = torch.cat([labels, r_lbls], dim=0)
                        is_replay_batch = True
                        n_replay += 1

                spike_seq = encoder(images)
                model.reset()
                logits = model(spike_seq)

                with torch.no_grad():
                    v = model.get_fc1_membrane()
                    if v is not None and v.numel() > 0:
                        v_flat     = v.reshape(-1, fc1_shape[0])
                        post_mean  = v_flat.mean(dim=0)
                        active_fc1 = post_mean.unsqueeze(1).expand(fc1_shape) > 0.05
                    else:
                        active_fc1 = torch.zeros(fc1_shape, dtype=torch.bool, device=device)

                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()

                with torch.no_grad():
                    if model.fc1.fc.weight.grad is not None:
                        protected_fc1 = (
                            vairagya_fc1.scores >= VAIRAGYA_PROTECTION_THRESHOLD
                        ).clone()
                        if is_replay_batch:
                            model.fc1.fc.weight.grad[protected_fc1] *= (
                                1.0 - REPLAY_VAIRAGYA_PARTIAL_LIFT
                            )
                        else:
                            model.fc1.fc.weight.grad[protected_fc1] = 0.0

                optimizer.step()
                epoch_loss += loss.item()

                with torch.no_grad():
                    cur_loss = loss.item()
                    conf     = sequencer.update_confidence(logits)

                    replay_conf = None
                    if is_replay_batch:
                        n_current     = images.shape[0] - N_REPLAY
                        replay_logits = logits[n_current:]
                        replay_probs  = torch.softmax(replay_logits.detach(), dim=1)
                        replay_conf   = replay_probs.max(dim=1).values.mean().item()

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
                    viveka_val = affect.viveka_signal()

                    viveka.update(active_fc1)
                    viveka_gain = viveka.compute_gain(
                        active_fc1, viveka_val, sequencer.tasks_seen
                    )

                    pain_fc1 = active_fc1 if pain else torch.zeros(
                        fc1_shape, dtype=torch.bool, device=device)
                    if pain:
                        lability_fc1.inject_pain(active_fc1)
                    lability_fc1.decay()

                    vairagya_fc1.accumulate(
                        active_fc1, pain_fc1,
                        bhaya=bhaya_val, buddhi=buddhi_val,
                        viveka_gain=viveka_gain
                    )
                    vairagya_fc1.apply_decay(model.fc1.fc.weight.data)

                logger.log_batch(
                    task=task_id, epoch=epoch, batch=batch_idx,
                    loss=cur_loss, confidence=conf, pain_fired=pain,
                    lability_mean=lability_fc1.get().mean().item(),
                    vairagya_protection=vairagya_fc1.protection_fraction(),
                    affective=affect.as_dict()
                )

            with torch.no_grad():
                for buf_imgs, buf_lbls in train_loader:
                    replay_buffer.update(buf_imgs, buf_lbls)
                    break

            print(f"    Loss: {epoch_loss/len(train_loader):.4f} | "
                  f"Bhaya: {bhaya_val:.3f} | Buddhi: {buddhi_val:.3f} | "
                  f"Viveka: {viveka_val:.3f} | "
                  f"V-fc1: {vairagya_fc1.protection_fraction()*100:.1f}% | "
                  f"Consist: {viveka.mean_consistency():.3f} | "
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
    print(f"\n{'='*65}")
    print(f"  Maya-Viveka | Split-CIFAR-100 CIL")
    print(f"  AA: {final['AA']}% | BWT: {final['BWT']}% | FWT: {final['FWT']}%")
    print(f"{'='*65}")
    logger.log_final(final)
    logger.close()
    return final


if __name__ == "__main__":
    run_viveka_cil()

