# backbone.py — Maya-Viveka network architecture
# Conv64 -> Conv128 -> Conv256 -> FC2048 -> OrthogonalPrototypeHead

import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, layer, functional

from maya_cl.utils.config import (
    CONV1_CHANNELS, CONV2_CHANNELS, CONV3_CHANNELS,
    FC1_SIZE, NUM_CLASSES, V_THRESHOLD, V_RESET, TAU_MEMBRANE,
    PROTOTYPE_DIM
)


class LIFLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc  = nn.Linear(in_features, out_features, bias=False)
        self.lif = neuron.LIFNode(
            tau=TAU_MEMBRANE, v_threshold=V_THRESHOLD,
            v_reset=V_RESET, detach_reset=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lif(self.fc(x))


class OrthogonalPrototypeHead(nn.Module):
    def __init__(self, num_classes: int, dim: int):
        super().__init__()
        raw = torch.randn(num_classes, dim)
        if num_classes <= dim:
            Q, _ = torch.linalg.qr(raw.T)
            prototypes = Q.T[:num_classes]
        else:
            prototypes = F.normalize(raw, dim=1)
        self.register_buffer('prototypes', prototypes)
        self.num_classes = num_classes
        self.dim         = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(x, dim=1)
        p_norm = F.normalize(self.prototypes, dim=1)
        return (x_norm @ p_norm.T) * 10.0


class MayaVivekaNet(nn.Module):
    def __init__(self, use_orthogonal_head: bool = True):
        super().__init__()

        self.conv1 = nn.Sequential(
            layer.Conv2d(3, CONV1_CHANNELS, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(CONV1_CHANNELS),
            neuron.LIFNode(tau=TAU_MEMBRANE, v_threshold=V_THRESHOLD,
                           v_reset=V_RESET, detach_reset=True)
        )
        self.conv2 = nn.Sequential(
            layer.Conv2d(CONV1_CHANNELS, CONV2_CHANNELS, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(CONV2_CHANNELS),
            neuron.LIFNode(tau=TAU_MEMBRANE, v_threshold=V_THRESHOLD,
                           v_reset=V_RESET, detach_reset=True),
            layer.MaxPool2d(2, 2)
        )
        self.conv3 = nn.Sequential(
            layer.Conv2d(CONV2_CHANNELS, CONV3_CHANNELS, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(CONV3_CHANNELS),
            neuron.LIFNode(tau=TAU_MEMBRANE, v_threshold=V_THRESHOLD,
                           v_reset=V_RESET, detach_reset=True),
            layer.MaxPool2d(2, 2)
        )

        self.flatten = layer.Flatten()

        conv_out_dim = CONV3_CHANNELS * 8 * 8
        self.fc1 = LIFLayer(conv_out_dim, FC1_SIZE)

        if use_orthogonal_head:
            self.fc_out = OrthogonalPrototypeHead(NUM_CLASSES, PROTOTYPE_DIM)
        else:
            self.fc_out = nn.Linear(FC1_SIZE, NUM_CLASSES, bias=False)

        functional.set_step_mode(self, step_mode='m')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = x.mean(dim=0)
        return self.fc_out(x)

    def reset(self):
        for m in self.modules():
            if m is not self and hasattr(m, 'reset'):
                m.reset()

    def get_fc1_membrane(self) -> torch.Tensor:
        return self.fc1.lif.v
