# task_sequence.py — task transition management for Maya-Viveka
# Updated for 10-task Split-CIFAR-100 CIL.

import torch
from maya_cl.utils.config import NUM_TASKS, PAIN_CONFIDENCE_THRESHOLD


class TaskSequencer:

    def __init__(self):
        self.current_task       = 0
        self.tasks_seen         = 0
        self._prev_loss         = None
        self._confidence_window = []
        self._window_size       = 10

    def update_confidence(self, logits: torch.Tensor) -> float:
        with torch.no_grad():
            probs = torch.softmax(logits.detach(), dim=1)
            conf  = probs.max(dim=1).values.mean().item()
            self._confidence_window.append(conf)
            if len(self._confidence_window) > self._window_size:
                self._confidence_window.pop(0)
        return conf

    def check_pain_signal(self,
                          cur_loss:    float,
                          prev_loss:   float,
                          confidence:  float,
                          replay_conf: float = None) -> bool:
        if prev_loss is None:
            return False

        loss_spike = (cur_loss / (prev_loss + 1e-8)) > 1.3

        conf_below_threshold = confidence < PAIN_CONFIDENCE_THRESHOLD
        if replay_conf is not None:
            conf_below_threshold = (
                conf_below_threshold and replay_conf < PAIN_CONFIDENCE_THRESHOLD
            )

        return loss_spike or conf_below_threshold

    def on_task_boundary(self, task_id: int) -> None:
        self.current_task = task_id
        self.tasks_seen   = task_id
        self._prev_loss   = None
        self._confidence_window.clear()
