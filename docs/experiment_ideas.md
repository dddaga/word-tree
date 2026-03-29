## Exp X: Reflective ReLU Routing

**Idea (from conversation 2026-03-26)**

In the routing step, split activation into propagating and reflected parts:

    Z_prop    = relu(Z)          # positive → travels to neighbours
    Z_reflect = min(Z, 0)        # negative → reflected back to self
    Z_new[h]  = Z_reflect[h] + Σ_k weight[h,k] * Z_prop[k]

**Expected properties:**
- Sparse activation propagation (only positive neurons transmit)
- Asymmetric wavefronts: active regions expand, quiet regions absorb
- Built-in inhibitory mechanism per neuron (sign determines role)
- Gradient sparsity (ReLU zeros gradient on negative activations)

**Key risk:** dying neuron cascade — once negative, a neuron stops propagating
and may never receive positive signal. Use leaky reflection α=0.1 as mitigation:
    Z_reflect = α * min(Z, 0)    # let negative drain slowly

**Phasor extension:** gate on real component (in-phase = propagate,
out-of-phase = reflect), preserve phase direction of imaginary component.

**Suggested experiment:** add `reflective=True` flag to ProximityWave._route(),
compare N=1024 with and without reflective routing at 100 epochs.
