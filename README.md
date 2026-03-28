# Maya-Viveka

**Viveka-Gated Synaptic Discrimination for Class-Incremental Learning in Affective Spiking Neural Networks**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5.1](https://img.shields.io/badge/PyTorch-2.5.1-orange.svg)](https://pytorch.org/)
[![SpikingJelly](https://img.shields.io/badge/SpikingJelly-0.0.0.0.14-green.svg)](https://github.com/fangwei123456/spikingjelly)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Paper 5 in the **Maya Research Series** — Venkatesh Swaminathan, Nexus Learning Labs, Bengaluru.

---

## Overview

Maya-Viveka extends the Maya affective SNN architecture to **harder CIL** on Split-CIFAR-100 (10 tasks, 10 classes each) by introducing **Viveka** — a cross-task synaptic consistency tracker that gates Vairagya consolidation selectively based on representational stability across tasks. Two contributions beyond Paper 4:

- **Viveka** — a sixth affective dimension (discernment) that measures per-synapse cross-task consistency and modulates Vairagya protection accordingly; synapses that encode stable features across tasks are protected more strongly than those encoding task-specific noise
- **Orthogonal collapse finding** — empirical demonstration that orthogonal prototype enforcement at insufficient replay budget actively suppresses Vairagya (0.45→0.22), causing representational collapse equivalent to baseline; this failure mode motivates retrograde consolidation correction (P6)

---

## Key Results — Split-CIFAR-100 CIL, Seed 42

### Main Runs

| Config | Replay Budget | AA (%) | BWT (%) |
|--------|--------------|--------|---------|
| Run 1 | 20/class | 11.93 | −55.46 |
| **Run 2** | **50/class** | **16.03** | **−50.50** |

### Ablation Study (50/class replay)

| Condition | AA (%) | BWT (%) | ΔAA vs Baseline |
|-----------|--------|---------|-----------------|
| A: SGD Baseline | 6.82 | −62.01 | — |
| B: Replay Only | 15.32 | −53.62 | +8.50 |
| C: Maya-Smriti | 15.40 | −50.52 | +8.58 |
| D: Maya-Smriti + Ortho Head | 6.56 | −60.38 | −0.26 |
| **E: Viveka, learnable head** | **16.03** | **−50.50** | **+9.21** |
| F: Full Maya-Viveka (Viveka + Ortho) | 6.66 | −60.24 | −0.16 |

**Key findings:**
- Viveka gain alone (E vs C): **+0.63pp AA** — selective synaptic discrimination provides consistent improvement over replay + Smriti
- Orthogonal head penalty (D vs C): **−8.84pp AA** — prototype constraint suppresses Vairagya saturation from ~0.47 to ~0.24, causing collapse back to baseline territory
- Bhaya quiescence: fires only during Task 0 (rate=0.024), then exactly 0.000 for Tasks 1–9 across all replay conditions — replicable emergent property first confirmed in P4, now confirmed on harder benchmark
- Buddhi S-curve: architecturally stable across all conditions (0.10→0.30→0.50→0.70→0.90 at epochs 0–4), confirming design determinism
- Replay budget is the primary CIL bottleneck; Viveka provides additional selective consolidation that replay alone cannot achieve

---

## Interactive Dashboard

Full experimental results visualised as a self-contained interactive dashboard — ablation study, accuracy matrices, affective dynamics per condition, Viveka trajectory, Bhaya quiescence panel, Vairagya saturation comparison, and Vedantic architecture overview.

📊 **[maya_viveka_dashboard.html](maya_viveka_dashboard.html)** — download and open in any browser. No server required, no dependencies.

---

## Architecture

```
PoissonEncoder(T=4)
→ Conv2d(3,64,3,pad=1) → LIF → MaxPool2d(2)
→ Conv2d(64,64,3,pad=1) → LIF → MaxPool2d(2)
→ Conv2d(64,128,3,pad=1) → LIF → MaxPool2d(2)
→ FC(2048) → LIF
→ FC(100)
```

**Affective dimensions:** Bhaya (fear, τ=3), Shraddha (trust, τ=10), Vairagya (wisdom, τ=20), Spanda (aliveness, τ=5), Buddhi (intellect, τ=200), **Viveka (discernment, cross-task consistency)** ← new in Paper 5

**Replay:** Class-wise ring buffer, M=50/class (canonical run). Interleaved at batch level.

---

## Repository Structure

```
maya_cl/
  benchmark/       — Split-CIFAR-10 and Split-CIFAR-100 task sequencers
  encoding/        — Poisson spike encoder
  eval/            — Metrics (AA, BWT, FWT) and CSV logger
  network/         — Backbone, LIF layers, AffectiveState (Viveka added)
  plasticity/      — Lability, Vairagya decay, Viveka gain modulation
  training/        — ReplayBuffer
  utils/           — Config, seed
experiments/
  run_viveka_cil.py        — Main CIL training run (Split-CIFAR-100)
  run_ablation_viveka.py   — Six-condition ablation study
results/
  viveka_cil_*             — Main run CSVs (20/class and 50/class)
  ablation_*               — All six ablation condition CSVs and summaries
tests/
docs/
```

---

## Installation

```bash
git clone https://github.com/venky2099/Maya-Viveka.git
cd Maya-Viveka
pip install -r requirements.txt
```

CIFAR-100 downloads automatically on first run via torchvision.

---

## Running

```bash
# Main CIL result (50/class replay)
python -m experiments.run_viveka_cil

# Full six-condition ablation
python -m experiments.run_ablation_viveka
```

---

## Vedantic Architecture — Antahkarana

| Dimension | Sanskrit | Role | Status |
|-----------|----------|------|--------|
| Bhaya | भय | Fear · pain trigger · τ=3 | Active P1–P5 |
| Vairagya | वैराग्य | Wisdom · heterosynaptic decay gating | Active P1–P5 |
| Shraddha | श्रद्धा | Trust · confidence integrator · τ=10 | Active P1–P5 |
| Spanda | स्पन्द | Aliveness · spike rate monitor · τ=5 | Active P1–P5 |
| Buddhi | बुद्धि | Intellect · consolidation rate gate · τ=200 | Active P4–P5 |
| **Viveka** | **विवेक** | **Discernment · cross-task synaptic consistency** | **New P5** |
| Ahamkara | अहंकार | Ego · task-attachment as forgetting cause | Named P4, dissolved P5 |
| Samskara | संस्कार | Impression traces · cross-task memory | Planned P6 |
| Chitta | चित्त | Implicit synaptic memory | Planned P6 |
| Prana | प्राण | Metabolic plasticity budget | Planned P9 |

---

## Citation

> Zenodo DOI will be added upon preprint publication.

```bibtex
@misc{swaminathan2026mayaviveka,
  title     = {Maya-Viveka: Viveka-Gated Synaptic Discrimination for
               Class-Incremental Learning in Affective Spiking Neural Networks},
  author    = {Swaminathan, Venkatesh},
  year      = {2026},
  publisher = {Zenodo},
  note      = {Preprint. DOI to be assigned.}
}
```

---

## Maya Research Series

| Paper | Title | Repo | DOI |
|-------|-------|------|-----|
| P1 | Nociceptive Metaplasticity and Graceful Decay in SNNs | [Maya-Nexus-Core](https://github.com/venky2099/Maya-Nexus-Core) | [10.5281/zenodo.19151563](https://doi.org/10.5281/zenodo.19151563) |
| P2 | Maya-OS: Affective SNN as Conversational OS Arbitration Layer | [Maya-OS](https://github.com/venky2099/Maya-OS) | [10.5281/zenodo.19160123](https://doi.org/10.5281/zenodo.19160123) |
| P3 | Maya-CL: Nociceptive Metaplasticity for Continual Learning | [Maya-CL](https://github.com/venky2099/Maya-CL) | [10.5281/zenodo.19201769](https://doi.org/10.5281/zenodo.19201769) |
| P4 | Maya-Smriti: Episodic Memory for CIL | [Maya-Smriti](https://github.com/venky2099/Maya-Smriti) | [10.5281/zenodo.19228975](https://doi.org/10.5281/zenodo.19228975) |
| **P5** | **Maya-Viveka: Viveka-Gated Synaptic Discrimination** | **This repo** | Pending |

---

## Acknowledgements

Independent research conducted at Nexus Learning Labs as part of M.Sc. thesis work in Data Science and Artificial Intelligence, BITS Pilani. All experiments run on personal hardware (NVIDIA RTX 4060 8GB). No funding sources. No conflicts of interest.

---

## License

MIT License — © 2026 Venkatesh Swaminathan