# Energy-Conservation Complex Activation Experiment

## Overview

This experiment uses the **full complex number representation** for node activation: \( s = e^{i\phi + \gamma\sin m} = r\,e^{i\phi} \) with **magnitude** \( r = e^{\gamma\sin m} \). So:

- **Complex signal**: \( s = r\,e^{i\phi} \), with \( r = e^{\gamma\sin m} \) (since \( \cos^2\phi + \sin^2\phi = 1 \), the modulus is \( r \)).
- **Real part (conduction)**: \( \text{real} = r\cos\phi = e^{\gamma\sin m}\cos\phi \)
- **Imaginary part (radiation)**: \( \text{imag} = r\sin\phi = e^{\gamma\sin m}\sin\phi \)
- **Phase** \( \phi = \operatorname{atan2}(\text{imag}, \text{real}) \); **magnitude** \( r = \sqrt{\text{real}^2 + \text{imag}^2} \).

Real and imaginary parts are used for **conduction** and **radiation**. Nodes **take and pass** activation: when a node sends via conduction it sends all its real and then its real becomes zero; when it sends via radiation it sends all its imag and then its imag becomes zero. This yields **wave-like** behavior (nodes gain then lose activation).

## Goals

1. **Take-and-pass (wave-like)**: When a node sends activation, it **loses** the part it sent: sending only via conduction zeros its real part; sending only via radiation zeros its imag part; sending via both zeros both. So nodes don’t store everything—they take (receive) and pass on (send), and can become inactive after sending.
2. **Proportion-based split**: When a source sends to several targets, it splits its **real** (conduction) and **imag** (radiation) according to alignment so the total sent equals the source’s real/imag.
3. **Forward pass only**: No backward path; we only observe propagation and wave-like activation.

## Activation Strength and Beam Search

- **Activation strength** of a node = **magnitude** \( r = \sqrt{\text{real}^2 + \text{imag}^2} \) (same as \( e^{\gamma\sin m} \) in the form \( s = r\,e^{i\phi} \)).

- **Beam search**: Only a fraction of nodes (e.g. top 10% by activation strength) are allowed to propagate.

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

4. **Update at target**: Each target starts with its current real/imag, then adds all received real and imag from sources. Then \( \phi = \operatorname{atan2}(\text{imag}, \text{real}) \), magnitude \( r = \sqrt{\text{real}^2 + \text{imag}^2} \), and the node’s state is set from \( \phi \) and \( r \).

5. **Take-and-pass (sender debit)**: After a source sends, it **loses** what it sent: if it had any conduction targets, its real is decremented by the full real it sent (so it goes to zero unless it also received real from others); if it had any radiation targets, its imag is decremented by the full imag it sent. So activation moves through the graph like a wave—nodes gain then lose activation.

## Decay and Forward-Only

- **Decay = 0%**: `temporal_decay=1.0`; no decay step is applied so that we can check conservation.
- **Forward path only**: No gradients or backward pass; the script only runs `one_step_forward` repeatedly and logs activations.

## What Is Logged

At each time step, for each active node, the logger records:

- `step`, `node_id`
- `real`, `imag`: real and imaginary parts of the complex activation (after sum over vector dim)
- `theta`: \( \operatorname{atan2}(\text{imag}, \text{real}) \)
- `magnitude`, `activation_strength`: both are \( r = \sqrt{\text{real}^2 + \text{imag}^2} \) (magnitude of \( s = r\,e^{i\phi} \)).
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

- Nodes **take** (receive) and **pass** (send) activation; they do not keep the part they send. So a node that sends only via conduction ends the step with real = 0 (and keeps any imag it had or received); one that sends only via radiation ends with imag = 0.
- This gives **wave-like** behavior: a node gains energy (incoming), becomes active, then loses energy (outgoing) and can become inactive (“dead”) until it receives again.
- The **proportion mechanism** splits each source’s real/imag among its targets by alignment; the sender then loses that real/imag (take-and-pass).

## Relation to Previous Model

- **Before**: Activation was \( \cos\theta \cdot e^{\gamma\sin m} \) (real part only).
- **Here**: Full complex \( s = r\,e^{i\phi} \) with \( r = e^{\gamma\sin m} \); real/imag for conduction/radiation; **take-and-pass** so senders lose what they send (wave-like propagation).

No changes are made to the existing core or other experiments; this lives entirely under `experiments/energy_conservation_complex/`.
