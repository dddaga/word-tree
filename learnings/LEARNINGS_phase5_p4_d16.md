# Phase 5: D=16 Era — New Mechanisms (2026-03-29+)

Active experiments and new mechanism designs.
Reference ceiling: 29.22% (step9A, D=16 Fourier N=512 dynamic_z_geo 150ep).

---

## Step 11: Routing Ablation WITH Geo at D=16 (RUNNING neuro_f)

Script: `train_step11_routing_geo_d16.py` — 120ep, D=16 Fourier, dynamic_z_geo

**Gap from step10a:** Step 10a used dynamic_z (no geo). Step 9 used full routing +
geo = 29.22%. Question: does geo change which routing mechanisms help?

Configs:
  A. theta only + geo     — reflect=0.0, turing=0.0
  B. theta+reflect + geo  — reflect=0.3, turing=0.0
  C. theta+turing + geo   — reflect=0.0, turing=0.3
  D. full routing + geo   — reflect=0.3, turing=0.3  (step9A equivalent at 120ep)

*Results pending.*

---

## Step 12: N Scale at D=16 (RUNNING neuro_g)

Script: `train_step12_n_scale_d16.py` — 120ep, D=16 Fourier, dynamic_z_geo

Tests polynomial width scaling: N=512 (ref) → N=1024 → N=2048.
At D=16 (S¹⁵), vastly more direction capacity — N scaling may beat D=4 (S³) where directions overcrowded.

*Results pending.*

---

## Step 13: Beam Size × K_iter Scaling (RUNNING neuro_h/i)

Script: `train_step13_beam_iter_scaling.py`

### Theoretical motivation
Deep learning scaling laws:
  - Width (N neurons): accuracy gains polynomial in parameter count
  - Depth (layers/iterations): accuracy gains EXPONENTIAL in depth

In SGNNET routing:
  - **beam_size** = "width" — neurons broadcasting per routing step
  - **K_iter** = "depth" — recursive propagation steps

At K_iter=k: each neuron integrates signal from k-hop neighborhood.
Small-world graph diameter ≈ log(512) ≈ 9 — K_iter≥9 reaches all neurons.

### Attribute accumulation
Each step: Z[k+1][h] = normalize(sum_{j in top-beam} score[h,j] * Z[k][j])
- Step 0: Z encodes input seed (VGG features + positional encoding)
- Step 1: 1-hop neighborhood average — local integration
- Step k: k-hop diffusion — information spreads like heat on graph
- Large K_iter risk: over-smoothing (all Z collapse to same vector)

### Beam sweep (neuro_h) — configs A-E: beam=8/16/32(ref)/64/128 at K_iter=3
### Depth sweep (neuro_i) — configs F-J: K_iter=1/2/5/8/12 at beam=32
### Interaction (pending) — configs K-L: (beam=8, K_iter=8) and (beam=128, K_iter=8)

*Results pending.*

---

## Step 14: Top-K Gated Conduction + Excitatory Radiation (RUNNING neuro_j)

Script: `train_step14_topk_cond_excrad.py` — 120ep, D=16 Fourier, dynamic_z_geo

### Architecture redesign rationale

**Current asymmetry:**
| Pathway | Topology | Direction | Selection |
|---|---|---|---|
| Conduction (conn_hh) | Fixed structural | Excitatory | NONE — all K_hh=6 contribute |
| Radiation (dynamic_z) | Rebuilt per step | **Inhibitory only** | Top-beam by magnitude |

**Two problems:**
1. Conduction has no competitive gating — weak/irrelevant structural neighbours dilute signal equally as strong relevant ones
2. Radiation only inhibits — no activation-based EXCITATORY dynamic pathway exists (W_phase tried but failed — learned static graph)

### Mechanism 1: Top-K gated conduction

For each neuron h, rank K_hh=6 structural neighbours by current cosine similarity. Only top-k_cond pass:

  k_cond=6 (all): current behaviour
  k_cond=4: 4 most resonant structural neighbours
  k_cond=2: strong competition within structure
  k_cond=1: winner-take-all within structural neighbourhood

**Key insight:** Structural graph provides LOCAL candidates; dynamic similarity adds selection within pool. Conduction + radiation become two-level hierarchy.

### Mechanism 2: Excitatory radiation (W_phase replacement)

For each neuron h, find top-K_exc most DIRECTIONALLY SIMILAR neurons by current normalised Z (not magnitude). Form transient EXCITATORY connections:

  Z_h += alpha_exc * weighted_avg(Z[similar neighbours])

`dynamic_z` for excitation (current dynamic_z inhibitory). Similar neurons reinforce each other → soft clustering.

**Key difference from W_phase:**
  - W_phase: learns static weight matrix → fixed graph baked in
  - Excitatory radiation: rebuilt every forward from current Z → truly dynamic

### Configs (RUNNING):
  A. k_cond=4  excrad=off
  B. k_cond=2  excrad=off
  C. k_cond=1  (winner-take-all conduction)
  F. k_cond=6  excrad=16  alpha=0.3  (excitatory radiation only)
  G. k_cond=6  excrad=16  alpha=0.1  (weak excitatory radiation)
  D. k_cond=4  excrad=8   alpha=0.3  (combined)
  E. k_cond=4  excrad=16  alpha=0.3  (combined, larger beam)

*Results pending.*

---

## Step 15: Beam-Gated Sparse Routing (PLANNED)

### User-proposed mechanism (2026-03-29)

Beam selects top-K neurons. ONLY beam neurons conduct AND radiate. Non-beam neurons have two options:

**Option A — Reset (zero state):** Non-beam neurons reset to zero after each step.
  - Blank slates, only receiving signals from beam neurons
  - Maximum sparsity: only 6.25% (32/512) active per step
  - Forces competitive "earn your way into beam" dynamics
  - Like spiking neurons: fire only if top-K

**Option B — Retain (accumulate):** Non-beam neurons keep Z state.
  - Receive new signals (from conducting beam neurons) but don't send
  - Can "charge up" over multiple steps until entering beam
  - Like integrate-and-fire: accumulate evidence until threshold
  - Two-speed dynamics: fast (beam, updates each step) + slow (non-beam, accumulates)

**Option C — Retain with exponential decay:**
  - alpha_retain * Z_old + (1-alpha_retain) * Z_new for non-beam neurons
  - Blends old representation with newly received signal

### Why interesting:
- Tests sparse routing (only top-K active) vs dense routing (all participate)
- Option A: fresh information each step — prevents stale representations
- Option B: history-dependent — allows temporal context building within routing
- Option C: exponential moving average — smooth memory

Script: `train_step15_beam_gated_routing.py` — implement after step 14 results.

---

## Other Inhibition Mechanisms Considered

### Divisive normalization (lateral shunting)
Each neuron's contribution divided by total neighbourhood activity:
```
Z_struct = Z_nb.sum(dim=2) / (1.0 + alpha_div * Z_nb.norm(dim=-1).mean(dim=2, keepdim=True))
```
Prevents one dominant neighbourhood from monopolising routing. Biologically: V1 normalization.

### Refractory inhibition
Neurons that contributed heavily in step k suppressed in step k+1:
```
Z_refractory = beta * Z_refractory + (1-beta) * activity
Z_available = Z * exp(-alpha_refract * Z_refractory)
```
Introduces temporal competition across routing steps. Forces information spread.

### Anti-Hebbian lateral inhibition
Neurons too similar in W_pos space inhibit each other's contribution — forces diverse representations (decorrelation). Creates Mexican-hat response profile.