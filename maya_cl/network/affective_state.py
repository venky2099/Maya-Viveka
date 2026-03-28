# affective_state.py — Maya-Viveka affective state machine
# Six dimensions: Bhaya, Vairagya, Shraddha, Spanda, Buddhi, Viveka

import torch
from maya_cl.utils.config import (
    TAU_BHAYA, TAU_SHRADDHA, TAU_VAIRAGYA, TAU_SPANDA, TAU_BUDDHI, TAU_VIVEKA
)


class AffectiveState:

    def __init__(self, device: torch.device):
        self.device = device
        self.bhaya    = torch.tensor(0.0, device=device)
        self.vairagya = torch.tensor(0.0, device=device)
        self.shraddha = torch.tensor(0.5, device=device)
        self.spanda   = torch.tensor(0.0, device=device)
        self.buddhi   = torch.tensor(0.0, device=device)
        self._buddhi_batches = 0
        self.viveka   = torch.tensor(0.0, device=device)
        self._task_boundary = False

    def update(self, confidence: float, pain: bool, spike_rate: float) -> None:
        dt = 1.0

        bhaya_target = 1.0 if pain else 0.0
        self.bhaya  += (bhaya_target - self.bhaya) * (dt / TAU_BHAYA)
        self.bhaya   = self.bhaya.clamp(0.0, 1.0)

        shraddha_target = confidence * (1.0 - self.bhaya.item())
        self.shraddha  += (shraddha_target - self.shraddha) * (dt / TAU_SHRADDHA)
        self.shraddha   = self.shraddha.clamp(0.0, 1.0)

        self.spanda += (spike_rate - self.spanda) * (dt / TAU_SPANDA)
        self.spanda  = self.spanda.clamp(0.0, 1.0)

        vairagya_target = self.shraddha.item()
        self.vairagya  += (vairagya_target - self.vairagya) * (dt / TAU_VAIRAGYA)
        self.vairagya   = self.vairagya.clamp(0.0, 1.0)

        self._buddhi_batches += 1
        buddhi_target    = min(1.0, self._buddhi_batches / TAU_BUDDHI)
        buddhi_fear_gate = 1.0 - self.bhaya.item()
        self.buddhi = torch.tensor(
            buddhi_target * buddhi_fear_gate, device=self.device
        ).clamp(0.0, 1.0)

        if self._task_boundary:
            self.viveka     *= 0.5
            self._task_boundary = False
        else:
            stability     = (1.0 - self.bhaya.item()) * confidence
            viveka_target = stability * self.buddhi.item()
            self.viveka  += (viveka_target - self.viveka) * (dt / TAU_VIVEKA)
            self.viveka   = self.viveka.clamp(0.0, 1.0)

    def notify_task_boundary(self) -> None:
        self._buddhi_batches = 0
        self.buddhi          = torch.tensor(0.0, device=self.device)
        self._task_boundary  = True

    def reset_experience(self) -> None:
        self.notify_task_boundary()

    def viveka_signal(self) -> float:
        return self.viveka.item()

    def as_dict(self) -> dict:
        return {
            'bhaya':    round(self.bhaya.item(),    4),
            'vairagya': round(self.vairagya.item(), 4),
            'shraddha': round(self.shraddha.item(), 4),
            'spanda':   round(self.spanda.item(),   4),
            'buddhi':   round(self.buddhi.item(),   4),
            'viveka':   round(self.viveka.item(),   4),
        }
