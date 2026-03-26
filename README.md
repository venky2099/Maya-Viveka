# Maya-Smriti

**Episodic Memory as a Biological Prior for Class-Incremental Learning in Affective Spiking Neural Networks**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19228975.svg)](https://doi.org/10.5281/zenodo.19228975)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1-orange.svg)](https://pytorch.org/)
[![SpikingJelly](https://img.shields.io/badge/SpikingJelly-0.0.0.0.14-green.svg)](https://github.com/fangwei123456/spikingjelly)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Paper 4 in the **Maya Research Series** — Venkatesh Swaminathan, Nexus Learning Labs, Bengaluru.

---

## Overview

Maya-Smriti extends the Maya affective SNN architecture to **Class-Incremental Learning (CIL)** on Split-CIFAR-10 through a minimal class-wise ring buffer with interleaved episodic replay. Three contributions beyond Paper 3:

- **Buddhi** — a fifth affective dimension (discriminative intellect) that gates Vairagya consolidation rate through an S-shaped experience curve, collapsing at task boundaries in a *Viparita Buddhi* state and recovering with experience
- **Ahamkara** — identification and resolution of the CIL failure mode where Vairagya's fc_out protection progressively locks out historical output neurons, rendering replay ineffective
- **Affective Quiescence** — the emergent suppression of Bhaya (fear) throughout replay-stabilised CIL training; a calm, memory-supported network does not experience nociceptive pain

---

## Key Results — Split-CIFAR-10 CIL, Seed 42

| Condition | AA (%) | BWT (%) | FWT (%) | ΔAA vs Baseline |
|-----------|--------|---------|---------|-----------------|
| A: SGD Baseline | 17.98 | −86.49 | −10.0 | — |
| B: Replay Only | 31.07 | −69.38 | −10.0 | +13.09 |
| C: Maya Only (no replay) | 17.77 | −86.61 | −10.0 | −0.21 |
| **D: Full Maya-Smriti** | **31.84** | **−68.36** | **−10.0** | **+13.86** |
| E: Maya-Smriti (no gate) | 31.82 | −68.29 | −10.0 | +13.84 |

**Key finding:** Condition C ≈ Condition A — Maya mechanisms alone cannot overcome CIL output-head interference (Ahamkara) without episodic memory. Full Maya-Smriti outperforms replay-only by +0.77% AA and +1.02% BWT.

---

## Architecture
```
PoissonEncoder(T=4)
→ Conv2d(3,32,3,pad=1) → LIF → MaxPool2d(2)
→ Conv2d(32,64,3,pad=1) → LIF → MaxPool2d(2)
→ FC(4096→2048) → LIF
→ FC(2048→10)
```

**Affective dimensions:** Bhaya (fear, τ=3), Shraddha (trust, τ=10), Vairagya (wisdom, τ=20), Spanda (aliveness, τ=5), **Buddhi (intellect, τ=200)** ← new in Paper 4

**Replay:** Class-wise ring buffer, M=50/class, 500 total. Interleaved at batch level (REPLAY_RATIO=0.3).

---

## Repository Structure
```
maya_cl/
  benchmark/       — Split-CIFAR-10 task sequencer
  encoding/        — Poisson spike encoder
  eval/            — Metrics (AA, BWT, FWT) and CSV logger
  network/         — Backbone, LIF layers, AffectiveState (Buddhi added)
  plasticity/      — Lability, Vairagya decay
  training/        — ReplayBuffer (new in Paper 4)
  utils/           — Config, seed
experiments/
  run_maya_cil.py       — Main CIL training run
  run_ablation_cil.py   — Five-condition ablation study
  run_maya_cl.py        — Paper 3 TIL run (carried forward)
  run_ablation.py       — Paper 3 TIL ablation (carried forward)
tests/
docs/
```

---

## Installation
```bash
git clone https://github.com/venky2099/Maya-Smriti.git
cd Maya-Smriti
pip install -r requirements.txt
```

CIFAR-10 downloads automatically on first run via torchvision.

---

## Running
```bash
# Main CIL result (Condition D)
python -m experiments.run_maya_cil

# Full five-condition ablation
python -m experiments.run_ablation_cil
```

---

## Vedantic Architecture — Antahkarana

| Dimension | Sanskrit | Role | Status |
|-----------|----------|------|--------|
| Bhaya | भय | Fear · pain trigger · τ=3 | Active P1–P4 |
| Vairagya | वैराग्य | Wisdom · heterosynaptic decay gating | Active P1–P4 |
| Shraddha | श्रद्धा | Trust · confidence integrator · τ=10 | Active P1–P4 |
| Spanda | स्पन्द | Aliveness · spike rate monitor · τ=5 | Active P1–P4 |
| Buddhi | बुद्धि | Intellect · consolidation rate gate · τ=200 | **New P4** |
| Viveka | विवेक | Discernment · dynamic feature discrimination | Planned P5 |

---

## Citation
```bibtex
@misc{swaminathan2026mayasmriti,
  title     = {Maya-Smriti: Episodic Memory as a Biological Prior for
               Class-Incremental Learning in Affective Spiking Neural Networks},
  author    = {Swaminathan, Venkatesh},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19228975},
  url       = {https://doi.org/10.5281/zenodo.19228975}
}
```

---

## Maya Research Series

| Paper | Title | Repo | DOI |
|-------|-------|------|-----|
| P1 | Nociceptive Metaplasticity and Graceful Decay in SNNs | [Maya-Nexus-Core](https://github.com/venky2099/Maya-Nexus-Core) | [10.5281/zenodo.19151562](https://doi.org/10.5281/zenodo.19151562) |
| P2 | Maya-OS: Affective SNN as Conversational OS Arbitration Layer | [Maya-OS](https://github.com/venky2099/Maya-OS) | [10.5281/zenodo.19160122](https://doi.org/10.5281/zenodo.19160122) |
| P3 | Maya-CL: Nociceptive Metaplasticity for Continual Learning | [Maya-CL](https://github.com/venky2099/Maya-CL) | [10.5281/zenodo.19201768](https://doi.org/10.5281/zenodo.19201768) |
| **P4** | **Maya-Smriti: Episodic Memory for CIL** | **This repo** | [10.5281/zenodo.19228975](https://doi.org/10.5281/zenodo.19228975) |

---

## Acknowledgements

Independent research conducted at Nexus Learning Labs as part of M.Sc. thesis work in Data Science and Artificial Intelligence, BITS Pilani. All experiments run on personal hardware (NVIDIA RTX 4060 8GB). No funding sources. No conflicts of interest.

---

## License

MIT License — © 2026 Venkatesh Swaminathan
