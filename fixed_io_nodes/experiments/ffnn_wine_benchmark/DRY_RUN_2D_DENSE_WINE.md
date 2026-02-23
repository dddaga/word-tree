# Complete Step-by-Step Dry Run: 2D Dense FFNN on Wine Dataset

This document walks through **every calculation** for one forward pass and one backward pass of the 2D dense FFNN using the UCI Wine Quality (Red) setup: 11 features, 11 classes (quality 0–10), with positional encoding.

---

## 1. Problem Dimensions

| Symbol | Value | Meaning |
|--------|--------|--------|
| **F** | 11 | Number of features (wine: fixed acidity, volatile acidity, …) |
| **C** | 11 | Number of classes (quality 0–10) |
| **H** | 4 | Hidden size (chosen small for dry run; actual runs use 100–1000) |
| **B** | 2 | Batch size (two samples for this trace) |

---

## 2. Data Preparation

### 2.1 Raw features (after train/test split and StandardScaler)

Take the first 2 samples from the scaled training set. Each row is 11 features.

**X** (shape `B × F` = 2×11), example values (scaler output; your actual values will differ):

```
Row 0: [x₀₀, x₀₁, x₀₂, x₀₃, x₀₄, x₀₅, x₀₆, x₀₇, x₀₈, x₀₉, x₀₁₀]
Row 1: [x₁₀, x₁₁, x₁₂, x₁₃, x₁₄, x₁₅, x₁₆, x₁₇, x₁₈, x₁₉, x₁₁₀]
```

**y** (shape `B`, dtype long): class index per sample, in {0,…,10}. Red wine data use 3–8. Example: `y = [5, 6]`.

### 2.2 Sinusoidal positional encoding (feature_dim = 11)

Formula in code:

- `pos = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
- `div = exp(arange(0, 11, 2) * (-ln(10000)/11))`  
  So indices for even positions: 0,2,4,6,8,10 → 6 values:
  - `k = 0,1,2,3,4,5` → `div[k] = exp(k * (-ln(10000)/11)) = 10000^(-k/11)`
- `pe[0::2]` = sin(pos[0::2] * div[0:6]) → sin for indices 0,2,4,6,8,10
- `pe[1::2]` = cos(pos[1::2] * div[0:5]) → cos for indices 1,3,5,7,9

**Numerical (to 6 decimals):**

```
div = [1.0, 0.6810, 0.4638, 0.3162, 0.2154, 0.1468]   (length 6)
pos_even = [0, 2, 4, 6, 8, 10]
pos_odd  = [1, 3, 5, 7, 9]

pe[0]  = sin(0·1.0)     = 0.0
pe[2]  = sin(2·0.6810)  = sin(1.362) ≈ 0.978
pe[4]  = sin(4·0.4638)  = sin(1.855) ≈ 0.958
pe[6]  = sin(6·0.3162)  = sin(1.897) ≈ 0.947
pe[8]  = sin(8·0.2154)  = sin(1.723) ≈ 0.988
pe[10] = sin(10·0.1468) = sin(1.468) ≈ 0.994

pe[1]  = cos(1·0.6810)  = cos(0.681) ≈ 0.777
pe[3]  = cos(3·0.4638)  = cos(1.391) ≈ 0.179
pe[5]  = cos(5·0.3162)  = cos(1.581) ≈ -0.011
pe[7]  = cos(7·0.2154)  = cos(1.508) ≈ 0.063
pe[9]  = cos(9·0.1468)  = cos(1.321) ≈ 0.248
```

So **pe** is a vector of length 11 (one value per feature index).

### 2.3 Build 2D input

- `pe_broadcast`: shape (B, F) = (2, 11); each row is the same **pe**.
- **x_2d** = stack(X, pe_broadcast, axis=1) → shape **(B, 2, F) = (2, 2, 11)**.
  - `x_f = x_2d[:, 0, :]` → **(2, 11)**  (feature channel)
  - `x_p = x_2d[:, 1, :]` → **(2, 11)**  (position channel)

So we have two (2×11) matrices: one of scaled features, one of repeated positional encoding.

---

## 3. Model Parameters (Shapes and Init)

Two towers: **tower_f** (feature), **tower_p** (position). Each is Linear(F, H) → ReLU → Linear(H, C).

### 3.1 Tower f

- **W1_f** (H×F) = (4×11), **b1_f** (H,) = (4,)
- **W2_f** (C×H) = (11×4), **b2_f** (C,) = (11,)

### 3.2 Tower p

- **W1_p** (4×11), **b1_p** (4,)
- **W2_p** (11×4), **b2_p** (11,)

PyTorch `nn.Linear` stores weight as (out_features, in_features) and uses `y = x @ W.T + b`. So forward is linear in (W, b) as below.

---

## 4. Forward Pass (Every Step)

### 4.1 Feature tower (tower_f)

**Step 1 – First linear (hidden pre-activation)**

- **z_f** = x_f @ W1_f.T + b1_f  
- Shapes: (2×11) @ (11×4) + (4,) → **(2×4)**

For sample i and hidden unit j:

```
z_f[i,j] = Σ_{d=0}^{10} x_f[i,d]·W1_f[j,d] + b1_f[j]
```

Example (sample 0, hidden 0):

```
z_f[0,0] = x_f[0,0]·W1_f[0,0] + x_f[0,1]·W1_f[0,1] + … + x_f[0,10]·W1_f[0,10] + b1_f[0]
```

**Step 2 – ReLU**

- **h_f** = ReLU(z_f) = max(0, z_f), element-wise. Shape **(2×4)**.

```
h_f[i,j] = max(0, z_f[i,j])
```

**Step 3 – Second linear (logits from feature tower)**

- **out_f** = h_f @ W2_f.T + b2_f  
- Shapes: (2×4) @ (4×11) + (11,) → **(2×11)**

For sample i and class c:

```
out_f[i,c] = Σ_{j=0}^{3} h_f[i,j]·W2_f[c,j] + b2_f[c]
```

---

### 4.2 Position tower (tower_p)

Same three steps with (x_p, W1_p, b1_p, W2_p, b2_p):

1. **z_p** = x_p @ W1_p.T + b1_p   → (2×4)  
   `z_p[i,j] = Σ_d x_p[i,d]·W1_p[j,d] + b1_p[j]`
2. **h_p** = ReLU(z_p)             → (2×4)  
   `h_p[i,j] = max(0, z_p[i,j])`
3. **out_p** = h_p @ W2_p.T + b2_p → (2×11)  
   `out_p[i,c] = Σ_j h_p[i,j]·W2_p[c,j] + b2_p[c]`

---

### 4.3 Combine towers

- **logits** = out_f + out_p   (element-wise). Shape **(2×11)**.

```
logits[i,c] = out_f[i,c] + out_p[i,c]
```

---

## 5. Loss (CrossEntropy)

For each sample i, softmax over the 11 logits, then negative log of the probability of the true class y[i]:

- **p[i,c]** = exp(logits[i,c]) / Σ_{c'=0}^{10} exp(logits[i,c'])
- **loss_i** = -log(p[i, y[i]])
- **loss** = (1/B) Σ_i loss_i   (default reduction='mean')

**Concrete for sample 0 (true class y[0]=5):**

1. Compute logits[0,:] (11 numbers).
2. **max_log** = max(logits[0,:]) (for numerical stability).
3. **exp_logits** = exp(logits[0,:] - max_log).
4. **sum_exp** = Σ_c exp_logits[c].
5. **p[0,c]** = exp_logits[c] / sum_exp  for each c.
6. **loss_0** = -log(p[0, 5]).

Repeat for sample 1 with y[1]; then **loss** = (loss_0 + loss_1) / 2.

---

## 6. Backward Pass (Gradients, Step by Step)

Let **L** = loss. We compute ∂L/∂(every parameter and intermediate).

### 6.1 Gradient of L w.r.t. logits

For CrossEntropyLoss with softmax:

- **d_logits[i,c]** = p[i,c] - 1{c = y[i]}  
  So for the true class c = y[i], d_logits[i,y[i]] = p[i,y[i]] - 1; for c ≠ y[i], d_logits[i,c] = p[i,c].  
  Shape **(2×11)**.

(With mean reduction, this gradient is already averaged over B; in PyTorch the scaling by 1/B is inside the loss backward.)

### 6.2 Gradient at the sum (logits = out_f + out_p)

- ∂L/∂out_f = ∂L/∂logits  (because ∂logits/∂out_f = 1)
- ∂L/∂out_p = ∂L/∂logits  

So **d_out_f** = d_logits, **d_out_p** = d_logits. Both **(2×11)**.

### 6.3 Backprop through tower_f second linear (out_f = h_f @ W2_f.T + b2_f)

- **d_out_f** is (2×11). We need dL/dh_f, dL/dW2_f, dL/db2_f.

Standard linear backward (row vector form): if out_f = h_f @ W2_f.T + b2_f, then

- **d_h_f** = d_out_f @ W2_f   → (2×11) @ (11×4) = **(2×4)**  
  Formula: d_h_f[i,j] = Σ_c d_out_f[i,c]·W2_f[c,j]
- **d_W2_f** = d_out_f.T @ h_f  → (11×2) @ (2×4) = **(11×4)**  
  Formula: d_W2_f[c,j] = Σ_i d_out_f[i,c]·h_f[i,j]
- **d_b2_f** = Σ_i d_out_f[i,:]  → **(11,)**  
  Formula: d_b2_f[c] = Σ_i d_out_f[i,c]

### 6.4 Backprop through ReLU in tower_f (h_f = ReLU(z_f))

- **d_z_f** = d_h_f where z_f > 0, else 0. Shape **(2×4)**.

```
d_z_f[i,j] = d_h_f[i,j]  if z_f[i,j] > 0,  else 0
```

### 6.5 Backprop through tower_f first linear (z_f = x_f @ W1_f.T + b1_f)

- **d_x_f** = d_z_f @ W1_f   → (2×4) @ (4×11) = **(2×11)**  
  d_x_f[i,d] = Σ_j d_z_f[i,j]·W1_f[j,d]
- **d_W1_f** = d_z_f.T @ x_f  → (4×2) @ (2×11) = **(4×11)**  
  d_W1_f[j,d] = Σ_i d_z_f[i,j]·x_f[i,d]
- **d_b1_f** = Σ_i d_z_f[i,:]  → **(4,)**  
  d_b1_f[j] = Σ_i d_z_f[i,j]

### 6.6 Backprop through tower_p

Same structure as tower_f:

1. **d_out_p** = d_logits.
2. Second linear: **d_h_p** = d_out_p @ W2_p  (2×4); **d_W2_p** = d_out_p.T @ h_p  (11×4); **d_b2_p** = sum over batch (11,).
3. ReLU: **d_z_p** = d_h_p where z_p > 0, else 0  (2×4).
4. First linear: **d_x_p** = d_z_p @ W1_p  (2×11); **d_W1_p** = d_z_p.T @ x_p  (4×11); **d_b1_p** = sum over batch (4,).

### 6.7 Gradient for the 2D input x_2d

- **d_x_f** and **d_x_p** are already computed.
- **x_2d** has shape (2, 2, 11) with x_2d[:,0,:] = x_f, x_2d[:,1,:] = x_p.
- So **d_x_2d[:,0,:]** = d_x_f, **d_x_2d[:,1,:]** = d_x_p. Shape **(2, 2, 11)**.

---

## 7. Single-Sample Numeric Example (One Path)

Use one sample (B=1), F=11, H=2, C=11 to show numbers end-to-end.

### 7.1 Fake inputs (one sample)

- **x_f** = (1×11), e.g. [0.1, -0.2, 0.3, -0.1, 0.0, 0.2, -0.3, 0.1, 0.0, -0.2, 0.1]
- **x_p** = (1×11), one row of PE, e.g. [0.0, 0.777, 0.978, 0.179, 0.958, -0.011, 0.947, 0.063, 0.988, 0.248, 0.994]
- **y** = [5]  (true class 5)

### 7.2 Fake parameters (tower_f only for brevity; tower_p is analogous)

**W1_f** (2×11), **b1_f** (2,):

```
W1_f[0,:] = [0.1, -0.05, 0.0, 0.05, -0.1, 0.0, 0.05, 0.0, -0.05, 0.1, 0.0]
W1_f[1,:] = [0.0, 0.1, -0.1, 0.0, 0.05, 0.05, -0.05, 0.0, 0.0, -0.1, 0.1]
b1_f      = [0.0, 0.0]
```

**z_f[0,0]** (full sum over d = 0…10):

| d | x_f[0,d] | W1_f[0,d] | product |
|---|----------|------------|---------|
| 0 | 0.1      | 0.1        | 0.01    |
| 1 | -0.2     | -0.05      | 0.01    |
| 2 | 0.3      | 0.0        | 0.0     |
| 3 | -0.1     | 0.05       | -0.005  |
| 4 | 0.0      | -0.1       | 0.0     |
| 5 | 0.2      | 0.0        | 0.0     |
| 6 | -0.3     | 0.05       | -0.015  |
| 7 | 0.1      | 0.0        | 0.0     |
| 8 | 0.0      | -0.05      | 0.0     |
| 9 | -0.2     | 0.1        | -0.02   |
|10 | 0.1      | 0.0        | 0.0     |

Sum = 0.01 + 0.01 + 0 - 0.005 + 0 + 0 - 0.015 + 0 + 0 - 0.02 + 0 = **-0.02**. Then + b1_f[0] = 0 → **z_f[0,0] = -0.02**.

**z_f[0,1]** (full sum):

| d | x_f[0,d] | W1_f[1,d] | product |
|---|----------|------------|---------|
| 0 | 0.1      | 0.0        | 0.0     |
| 1 | -0.2     | 0.1        | -0.02   |
| 2 | 0.3      | -0.1       | -0.03   |
| 3 | -0.1     | 0.0        | 0.0     |
| 4 | 0.0      | 0.05       | 0.0     |
| 5 | 0.2      | 0.05       | 0.01    |
| 6 | -0.3     | -0.05      | 0.015   |
| 7 | 0.1      | 0.0        | 0.0     |
| 8 | 0.0      | 0.0        | 0.0     |
| 9 | -0.2     | -0.1       | 0.02    |
|10 | 0.1      | 0.1        | 0.01    |

Sum = 0 - 0.02 - 0.03 + 0 + 0 + 0.01 + 0.015 + 0 + 0 + 0.02 + 0.01 = **-0.025**. + b1_f[1] = 0 → **z_f[0,1] = -0.025**.

**ReLU:** h_f[0,0] = max(0, -0.02) = **0**; h_f[0,1] = max(0, -0.025) = **0**.

**W2_f** (11×2), **b2_f** (11,): take W2_f[c,0] = 0.1, W2_f[c,1] = -0.1 for all c; b2_f[c] = 0.

**out_f[0,c]** = h_f[0,0]·W2_f[c,0] + h_f[0,1]·W2_f[c,1] + b2_f[c] = 0·0.1 + 0·(-0.1) + 0 = **0** for every c = 0,…,10.

So **out_f[0,:]** = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].

Assume **tower_p** yields e.g. **out_p[0,:]** = [0.5, 0.2, 0.1, 0.0, -0.2, 0.8, 0.3, 0.1, -0.1, 0.0, 0.2] (example). Then **logits[0,:]** = out_f[0,:] + out_p[0,:] = [0.5, 0.2, 0.1, 0.0, -0.2, 0.8, 0.3, 0.1, -0.1, 0.0, 0.2].

**Softmax (sample 0):** max_log = 0.8. exp_logits = exp(logits[0,:] - 0.8):

| c | logits[0,c]-0.8 | exp(·)   |
|---|------------------|----------|
| 0 | -0.3             | 0.741    |
| 1 | -0.6             | 0.549    |
| 2 | -0.7             | 0.497    |
| 3 | -0.8             | 0.449    |
| 4 | -1.0             | 0.368    |
| 5 | 0.0              | 1.0      |
| 6 | -0.5             | 0.607    |
| 7 | -0.7             | 0.497    |
| 8 | -0.9             | 0.407    |
| 9 | -0.8             | 0.449    |
|10 | -0.6             | 0.549    |

sum_exp = 0.741 + 0.549 + 0.497 + 0.449 + 0.368 + 1.0 + 0.607 + 0.497 + 0.407 + 0.449 + 0.549 = **6.113**.  
p[0,c] = exp_logits[c] / 6.113. So p[0,5] = 1.0 / 6.113 ≈ **0.1636**.  
**loss_0** = -log(0.1636) ≈ **1.810**.

With two samples, loss = (loss_0 + loss_1)/2. Backward: d_logits from softmax gradient; d_out_f = d_out_p = d_logits; then through each linear and ReLU as in Section 6.

---

## 8. Summary of Shapes (2D Dense Forward)

| Variable | Shape | Formula |
|----------|--------|--------|
| X (raw) | (B, F) | (2, 11) |
| pe | (F,) | (11,) |
| x_2d | (B, 2, F) | (2, 2, 11) |
| x_f, x_p | (B, F) | (2, 11) |
| z_f, z_p | (B, H) | (2, 4) |
| h_f, h_p | (B, H) | (2, 4) |
| out_f, out_p | (B, C) | (2, 11) |
| logits | (B, C) | (2, 11) |
| loss | scalar | — |

Backward: every gradient has the same shape as the corresponding tensor (d_logits (2,11), d_out_f/d_out_p (2,11), d_h_f/d_h_p (2,4), d_z_f/d_z_p (2,4), d_x_f/d_x_p (2,11), d_W and d_b match W and b).

---

## 9. Run the Actual Dry Run in Code

To reproduce with the real wine pipeline and small H:

```python
from run_experiment import load_data, NUM_CLASSES
from ffnn_2d import sinusoidal_positional_encoding, to_2d_input, FeedForwardNet2D
import torch, torch.nn as nn

X_tr, _, _, y_tr, _, _ = load_data()
F = X_tr.shape[1]
pe = sinusoidal_positional_encoding(F)
x_2d = to_2d_input(X_tr[:2], pe)
x = torch.tensor(x_2d, dtype=torch.float32)
y = torch.tensor(y_tr[:2], dtype=torch.long)

model = FeedForwardNet2D(F, hidden_size=4, output_dim=NUM_CLASSES)
criterion = nn.CrossEntropyLoss()
logits = model(x)
loss = criterion(logits, y)
loss.backward()
# Inspect x.grad, model.tower_f[0].weight.grad, etc.
```

This matches the steps above with B=2, F=11, H=4, C=11; training and inference time are logged by the full experiment scripts.
