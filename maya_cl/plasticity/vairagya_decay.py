# vairagya_decay.py — affectively-gated heterosynaptic decay
# Paper 5: Viveka gain multiplier on accumulation.

import torch
from maya_cl.utils.config import (
    VAIRAGYA_DECAY_RATE,
    VAIRAGYA_PROTECTION_THRESHOLD,
    VAIRAGYA_ACCUMULATE_RATE,
    VAIRAGYA_PAIN_EROSION_RATE,
)


class VairagyadDecay:

    def __init__(self, shape: tuple, device: torch.device):
        self.scores = torch.zeros(shape, device=device)
        self.device = device

    def accumulate(self,
                   active_mask: torch.Tensor,
                   pain_mask: torch.Tensor,
                   bhaya: float = 0.0,
                   buddhi: float = 1.0,
                   viveka_gain: torch.Tensor = None) -> None:
        with torch.no_grad():
            base_rate = VAIRAGYA_ACCUMULATE_RATE * buddhi

            if viveka_gain is not None:
                effective_rate = base_rate * viveka_gain
                self.scores[active_mask] += effective_rate[active_mask]
            else:
                self.scores[active_mask] += base_rate

            pain_bonus = VAIRAGYA_ACCUMULATE_RATE * 5.0 * buddhi
            self.scores[pain_mask] += pain_bonus

            viparita = bhaya * (1.0 - buddhi)
            if viparita > 0.01:
                self.scores -= viparita * VAIRAGYA_PAIN_EROSION_RATE

            self.scores.clamp_(0.0, 1.0)

    def apply_decay(self, weight: torch.Tensor) -> None:
        with torch.no_grad():
            unprotected = self.scores < VAIRAGYA_PROTECTION_THRESHOLD
            weight[unprotected] *= (1.0 - VAIRAGYA_DECAY_RATE)

    def get_scores(self) -> torch.Tensor:
        return self.scores

    def protection_fraction(self) -> float:
        return (self.scores >= VAIRAGYA_PROTECTION_THRESHOLD).float().mean().item()
