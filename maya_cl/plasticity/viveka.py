# viveka.py â€” Viveka discriminative gain mechanism (Paper 5 core contribution)
#
# Viveka (à¤µà¤¿à¤µà¥‡à¤•): discriminative intellect â€” the faculty that distinguishes
# the permanent from the transient, the essential from the incidental.
#
# Computational role:
#   Tracks per-synapse cross-task consistency. A synapse that fires
#   consistently across multiple tasks encodes task-general features
#   (edges, textures, low-level structure). A synapse active only for
#   the current task encodes task-specific features.
#
#   Viveka amplifies Vairagya protection on task-general synapses
#   via a multiplicative gain signal. Task-specific synapses receive
#   no gain â€” Vairagya protects them at baseline rate only.
#
# Mechanism:
#   consistency_score[i,j] âˆˆ [0, 1] per synapse
#     rises  (RISE rate)  when synapse is active this batch
#     decays (DECAY rate) when synapse is inactive
#
#   gain[i,j] = 1.0 + VIVEKA_GAIN_MAX * consistency_score[i,j]
#
#   This gain multiplies the Vairagya accumulation rate.
#   Synapses with high consistency â†’ higher Vairagya â†’ stronger protection.
#   Synapses with low consistency â†’ gain â‰ˆ 1.0 â†’ baseline Vairagya rate.
#
# GANE connection:
#   consistency_score is the proxy for local glutamate concentration.
#   affect.viveka_signal() is the proxy for NE arousal level.
#   Their product determines hotspot gain â€” amplifying the strong,
#   leaving the weak at baseline. This is the GANE center-surround
#   translated into continual learning consolidation.

import torch
from maya_cl.utils.config import (
    VIVEKA_CONSISTENCY_RISE,
    VIVEKA_CONSISTENCY_DECAY,
    VIVEKA_GAIN_MAX,
    VIVEKA_MIN_TASKS,
)


class VivekaConsistency:
    """
    Per-synapse cross-task consistency tracker.

    shape: (out_features, in_features) â€” matches fc1 weight matrix.
    device: torch.device

    Usage:
        viveka = VivekaConsistency(fc1_shape, device)

        # each batch:
        gain = viveka.compute_gain(active_mask, viveka_signal, tasks_seen)
        # pass gain to VairagyadDecay.accumulate()

        # each task boundary:
        viveka.on_task_boundary(active_mask_this_task)
    """

    def __init__(self, shape: tuple, device: torch.device):
        # consistency score per synapse â€” starts at zero
        self.scores  = torch.zeros(shape, device=device)
        self.device  = device
        self.shape   = shape

        # running snapshot of which synapses were consistently active
        # in the most recently completed task â€” used at task boundaries
        self._task_activity_accumulator = torch.zeros(shape, device=device)
        self._batches_this_task         = 0

    def update(self, active_mask: torch.Tensor) -> None:
        """
        Update consistency scores from current batch activity.
        Called every batch during training.

        active_mask: bool tensor [out, in] â€” synapses active this batch
        """
        with torch.no_grad():
            self.scores[active_mask]  += VIVEKA_CONSISTENCY_RISE
            self.scores[~active_mask] -= VIVEKA_CONSISTENCY_DECAY
            self.scores.clamp_(0.0, 1.0)

            # accumulate activity for end-of-task snapshot
            self._task_activity_accumulator += active_mask.float()
            self._batches_this_task         += 1

    def compute_gain(self,
                     active_mask: torch.Tensor,
                     viveka_signal: float,
                     tasks_seen: int) -> torch.Tensor:
        """
        Compute per-synapse Vairagya gain multiplier.

        gain[i,j] = 1.0 + VIVEKA_GAIN_MAX * consistency[i,j] * viveka_signal

        Before VIVEKA_MIN_TASKS tasks have been seen, gain = 1.0 everywhere.
        This prevents Viveka from locking in patterns before it has seen
        enough tasks to distinguish general from specific.

        Returns:
            gain tensor [out, in], dtype float, values in [1.0, 1+VIVEKA_GAIN_MAX]
        """
        with torch.no_grad():
            if tasks_seen < VIVEKA_MIN_TASKS:
                return torch.ones(self.shape, device=self.device)

            # Viveka signal scales the gain â€” if affective Viveka is low
            # (e.g. at task boundary, high fear), gain collapses toward 1.0
            gain = 1.0 + VIVEKA_GAIN_MAX * self.scores * viveka_signal
            return gain.clamp(1.0, 1.0 + VIVEKA_GAIN_MAX)

    def on_task_boundary(self) -> None:
        """
        Called at each task transition.
        Resets the per-task activity accumulator.
        Consistency scores persist across tasks â€” they are the memory.
        """
        with torch.no_grad():
            self._task_activity_accumulator.zero_()
            self._batches_this_task = 0

    def mean_consistency(self) -> float:
        """Diagnostic: mean consistency score across all synapses."""
        return self.scores.mean().item()

    def high_consistency_fraction(self, threshold: float = 0.5) -> float:
        """Fraction of synapses above consistency threshold â€” tracks Viveka coverage."""
        return (self.scores >= threshold).float().mean().item()
