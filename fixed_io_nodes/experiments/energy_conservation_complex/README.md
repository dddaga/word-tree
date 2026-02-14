# Energy-Conservation Complex Activation Experiment

## Overview

This experiment uses the **full complex number representation** for node activation instead of only the real part (cos θ). The activation is:

- **Complex signal**: \( e^{i\theta} \cdot e^{\gamma \sin m} = (\cos\theta + i\sin\theta) \cdot e^{\gamma\sin m} \)
- **Real part (conduction)**: \( \cos\theta \cdot e^{\gamma\sin m} \)
- **Imaginary part (radiation)**: \( \sin\theta \cdot e^{\gamma\sin m} \)

Previously, only the real part \( \cos\theta \cdot e^{\gamma\sin m} \) was used. Here we keep both real and imaginary components and use them for **conduction** (real) and **radiation** (imaginary), with **energy-conserving** propagation.

## Goals

1. **Conservation of activation**: Total activation in the system stays constant (no decay; decay = 0%).
2. **Proportion-based split**: When a source node sends activation to several targets, it splits its **real** part (conduction) and **imag** part (radiation) according to alignment with each target, so that the amount sent sums exactly to the source’s real/imag (no creation or loss).
3. **Forward pass only**: No backward path; we only observe how activation propagates and whether conservation and stability hold.

## Activation Strength and Beam Search

- **Activation strength** of a node = magnitude × exp(γ sin m):
  \[
  \text{strength} = \sqrt{\text{real}^2 + \text{imag}^2} \cdot e^{\gamma \sum_d \sin m_d}
  \]
  where `real` = \( \sum_d \cos(\theta_d) e^{\gamma\sin m_d} \), `imag` = \( \sum_d \sin(\theta_d) e^{\gamma\sin m_d} \) (sum over vector dimension \(d\)), and \( m \) is the node’s mag vector.

- **Beam search**: Only a fraction of nodes (e.g. top 10% by activation strength) are allowed to propagate. This mimics “only the strongest nodes transfer activation further.”

## Propagation and Conservation

1. **Effective phase at target**: When source (phase \( \theta_s \)) sends to a target with phase \( \theta_t \), we use effective phase \( \theta_{\text{eff}} = \theta_t + \theta_s \) (e.g. 15° + 10° = 25°) for computing alignment.

2. **Conduction alignment** (for splitting the **real** part of the source):
   \[
   \text{conduction}_j = \sum_d \cos(\theta_{\text{eff},d}) \cdot e^{\gamma\sin m_{t,d}}
   \]
   Sum over all targets from this source: \( S_{\text{con}} = \sum_j \text{conduction}_j \).  
   Proportion to target \(j\): \( p_{\text{con},j} = \text{conduction}_j / S_{\text{con}} \).  
   Source’s real part sent to \(j\) = \( \text{real}_s \cdot p_{\text{con},j} \). So \( \sum_j \text{real}_s \cdot p_{\text{con},j} = \text{real}_s \), i.e. conduction is conserved.

3. **Radiation alignment** (for splitting the **imag** part): Same idea with \( \sin(\theta_{\text{eff},d}) \), giving proportions \( p_{\text{rad},j} \). Source’s imag part sent to \(j\) = \( \text{imag}_s \cdot p_{\text{rad},j} \). So the imaginary part is also conserved across targets.

4. **Update at target**: Each target sums all received real and received imag from its sources. Then:
   - \( \theta = \operatorname{atan2}(\text{imag}, \text{real}) \)
   - Magnitude = \( \sqrt{\text{real}^2 + \text{imag}^2} \)
   - Node’s `phase_activation` and `mag_activation` are set from this \( \theta \) and magnitude so that the complex representation stays consistent.

## Decay and Forward-Only

- **Decay = 0%**: `temporal_decay=1.0`; no decay step is applied so that we can check conservation.
- **Forward path only**: No gradients or backward pass; the script only runs `one_step_forward` repeatedly and logs activations.

## What Is Logged

At each time step, for each active node, the logger records:

- `step`, `node_id`
- `real`, `imag`: real and imaginary parts of the complex activation (after sum over vector dim)
- `theta`: \( \operatorname{atan2}(\text{imag}, \text{real}) \)
- `magnitude`, `activation_strength`: magnitude = \( \sqrt{\text{real}^2 + \text{imag}^2} \); activation_strength = magnitude × \( e^{\gamma \sum_d \sin m_d} \)
- Optionally `phase_0..phase_{d-1}`, `mag_0..mag_{d-1}` (vector components)

These go to `runs/run_<timestamp>/activations.csv`.

## How to Run

From the repo root (e.g. `word-tree`):

```bash
python fixed_io_nodes/experiments/energy_conservation_complex/run_experiment.py
```

Requirements:

- Same PyTorch setup as the rest of `fixed_io_nodes`.
- Config uses the **PyTorch NodeStore** (in-memory graph); no Qdrant server needed (same as the default `NodeStore` in this codebase).

## Files in This Experiment

| File | Role |
|------|------|
| `utils.py` | Complex real/imag from phase & mag, activation strength, conduction/radiation alignments, θ from real/imag. |
| `gnn.py` | `EnergyConservationComplexGNN`: proportion-based conduction/radiation, beam search, 0% decay, forward-only. |
| `activation_logger.py` | CSV logger for real, imag, theta, magnitude (and optionally phase/mag vectors) per step. |
| `run_experiment.py` | Builds config and GNN, injects one Iris sample, runs N steps, logs to `runs/run_<timestamp>/activations.csv`. |
| `README.md` | This file. |

## Intuition

- **Energy** here is the total “activation” (real and imag) in the network.
- Input nodes get initial activation from the data; that activation then moves through the graph via **conduction** (real) and **radiation** (imag).
- The **proportion mechanism** ensures that when one node sends to many, the amount it sends in total equals its real (for conduction) and its imag (for radiation), so no activation is created or lost at the sender.
- By setting decay to 0% and only doing forward steps, we can check whether total activation remains constant over time and whether some nodes reach a stable state.

## Relation to Previous Model

- **Before**: Activation was \( \cos\theta \cdot e^{\gamma\sin m} \) (real part only).
- **Here**: Full complex \( e^{i\theta} \cdot e^{\gamma\sin m} \); real part used for conduction, imaginary for radiation; propagation is proportion-based and designed to conserve activation.

No changes are made to the existing core or other experiments; this lives entirely under `experiments/energy_conservation_complex/`.
