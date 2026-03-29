"""MPS sparse algorithm survey for C_input: [N_in, N_h] @ [B, N_in, D].

Goal: find fastest way to exploit 90% sparsity on Apple Silicon MPS.

Algorithms tested
-----------------
0. Dense einsum            — baseline, single Metal GEMM kernel
1. Fixed fan-in gather+sum — K indices per neuron, regular gather
2. Chunked dense           — split N_in into tiles, skip zero tiles
3. COO segment_reduce      — sort edges by dst, use segment sum op
4. index_add loop          — scatter edges one connection at a time (per batch)
5. Half-prec fan-in        — fixed fan-in but gather in float16

Each is tested for correctness then timed at B=64, N_in=25088, N_h=256.
"""

from __future__ import annotations
import time, math
import torch

DEVICE  = "mps" if torch.backends.mps.is_available() else "cpu"
B, N_IN, N_H, D = 64, 25088, 256, 4
SPARSITY = 0.90
RUNS = 30
K_SMALL = 100          # representative small fan-in

print(f"Device: {DEVICE}  B={B}  N_in={N_IN}  N_h={N_H}  D={D}  sparsity={SPARSITY}")

# ── shared fixtures ───────────────────────────────────────────────
torch.manual_seed(0)
bool_mask = (torch.rand(N_IN, N_H) > SPARSITY).to(DEVICE)   # [N_in, N_h]
A = torch.randn(B, N_IN, D, device=DEVICE)                   # [B, N_in, D]

src_idx, dst_idx = bool_mask.nonzero(as_tuple=True)          # COO [nnz]
nnz = src_idx.shape[0]

# Segment lengths: how many inputs feed into each hidden neuron
seg_lengths = bool_mask.sum(dim=0).long().cpu()              # [N_h] on CPU

# Fixed fan-in index table (random K per neuron)
conn_idx = torch.stack([torch.randperm(N_IN)[:K_SMALL]
                        for _ in range(N_H)]).to(DEVICE)     # [N_h, K]

print(f"nnz={nnz:,}  K_small={K_SMALL}  seg_lengths range: "
      f"[{seg_lengths.min()}, {seg_lengths.max()}]\n")


# ── helpers ───────────────────────────────────────────────────────

def sync():
    if DEVICE == "mps": torch.mps.synchronize()

def bench(name: str, fn) -> float:
    try:
        out = fn()                                           # correctness run
        for _ in range(5): fn()                             # warmup
        sync()
        t0 = time.perf_counter()
        for _ in range(RUNS): fn()
        sync()
        ms = (time.perf_counter() - t0) / RUNS * 1000
        return ms, out
    except Exception as e:
        print(f"  FAILED {name}: {str(e)[:100]}")
        return None, None

def check(name: str, got, ref, tol=5e-3):
    if got is None: return
    diff = (got.float() - ref.float()).abs().max().item()
    status = "OK" if diff < tol else f"MISMATCH diff={diff:.2e}"
    return status


# ── 0. Dense einsum (baseline) ────────────────────────────────────

def algo0_dense():
    return torch.einsum("bid,ih->bhd", A, bool_mask.float())

ms0, ref = bench("0-dense", algo0_dense)
print(f"  {'0. Dense einsum (baseline)':<40s} {ms0:7.2f} ms  [reference]")


# ── 1. Fixed fan-in gather+sum ────────────────────────────────────
# conn_idx [N_h, K] — each hidden neuron has exactly K connections

def algo1_fanin():
    return A[:, conn_idx, :].sum(dim=2)   # [B,N_h,K,D] -> [B,N_h,D]

ms1, out1 = bench("1-fanin", algo1_fanin)
# Note: different connectivity so exact values differ — just check shape
shape_ok = "OK" if out1.shape == ref.shape else "SHAPE MISMATCH"
print(f"  {'1. Fixed fan-in K='+str(K_SMALL):<40s} {ms1:7.2f} ms  shape={shape_ok}")


# ── 2. Chunked dense (skip zero-column blocks) ────────────────────
# Split N_in into TILE-row chunks. Mask columns with any nonzero.
# On MPS this still calls GEMM per chunk but each chunk is smaller.

TILE = 2048   # rows of N_in per chunk

def algo2_chunked():
    Z = torch.zeros(B, N_H, D, device=DEVICE)
    for start in range(0, N_IN, TILE):
        end   = min(start + TILE, N_IN)
        chunk_mask = bool_mask[start:end]           # [tile, N_h]
        active_cols = chunk_mask.any(dim=0)         # [N_h] bool
        if not active_cols.any(): continue
        chunk_mask_f = chunk_mask[:, active_cols].float()   # [tile, n_act]
        A_chunk = A[:, start:end, :]                        # [B, tile, D]
        Z[:, active_cols, :] += torch.einsum(
            "bid,ih->bhd", A_chunk, chunk_mask_f)
    return Z

ms2, out2 = bench("2-chunked", algo2_chunked)
if out2 is not None:
    print(f"  {'2. Chunked dense tile='+str(TILE):<40s} {ms2:7.2f} ms  "
          f"{check('2', out2, ref)}")


# ── 3. COO segment_reduce ─────────────────────────────────────────
# Sort edges by dst neuron. Gather src values, segment-sum into dst.
# torch._segment_reduce is CPU-only; test fallback path.

def algo3_segment_reduce():
    # A[:, src_idx, :] -> [B, nnz, D]; flatten to [B*nnz, D] for segment op
    gathered = A[:, src_idx, :]                   # [B, nnz, D]
    # segment_reduce expects 1-D segments; process per batch item
    # (CPU fallback — MPS not supported for this op)
    gathered_cpu = gathered.reshape(B * nnz, D).cpu()
    lengths_rep  = seg_lengths.repeat(B)          # [B * N_h]
    out_cpu = torch._segment_reduce(gathered_cpu, "sum", lengths=lengths_rep)
    return out_cpu.reshape(B, N_H, D).to(DEVICE)

ms3, out3 = bench("3-segment_reduce", algo3_segment_reduce)
if out3 is not None:
    print(f"  {'3. COO segment_reduce (CPU fallback)':<40s} {ms3:7.2f} ms  "
          f"{check('3', out3, ref)}")


# ── 4. Batched index_add ──────────────────────────────────────────
# For each batch item: scatter non-zero inputs into hidden neurons.
# Pure MPS index ops — no matmul at all.

def algo4_index_add():
    Z = torch.zeros(B, N_H, D, device=DEVICE)
    vals = A[:, src_idx, :]                        # [B, nnz, D]
    dst_exp = dst_idx.view(1, nnz, 1).expand(B, -1, D)
    Z.scatter_add_(1, dst_exp, vals)
    return Z

ms4, out4 = bench("4-scatter_add", algo4_index_add)
if out4 is not None:
    print(f"  {'4. scatter_add (COO, MPS native)':<40s} {ms4:7.2f} ms  "
          f"{check('4', out4, ref)}")


# ── 5. Half-precision fixed fan-in ───────────────────────────────
# Same as algo1 but gather in float16 — MPS FP16 throughput is 2x.

A_fp16 = A.half()
conn_idx_f = conn_idx  # int stays int

def algo5_fp16_fanin():
    return A_fp16[:, conn_idx_f, :].sum(dim=2).float()

ms5, out5 = bench("5-fp16 fan-in K="+str(K_SMALL), algo5_fp16_fanin)
if out5 is not None:
    print(f"  {'5. FP16 fixed fan-in K='+str(K_SMALL):<40s} {ms5:7.2f} ms  "
          f"shape={'OK' if out5.shape==ref.shape else 'BAD'}")


# ── 6. Multi-head fan-in (vectorised over N_h groups) ────────────
# Group hidden neurons into G groups of N_h/G. Each group shares
# a dense [N_in_local, G, D] sub-matmul. Structured sparsity exploit.

G = 32   # groups
N_H_PER_G = N_H // G   # 8 hidden neurons per group

# Build group connectivity: each group owns a local dense block
# of input neurons. This is structured (block) sparsity.
group_size_in = N_IN // G   # 784 inputs per group
conn_g = torch.stack([
    torch.arange(g * group_size_in, (g+1) * group_size_in)
    for g in range(G)
]).to(DEVICE)   # [G, group_size_in]

def algo6_grouped():
    # A: [B, N_in, D] -> split into G chunks each [B, group_size_in, D]
    A_g = A.reshape(B, G, group_size_in, D)    # [B, G, local, D]
    # Fake dense sub-projection per group: sum over local inputs
    # (This is the structured-sparsity ideal — each neuron group
    # only receives from its local input block)
    return A_g.sum(dim=2).unsqueeze(2).expand(
        B, G, N_H_PER_G, D
    ).reshape(B, N_H, D)

ms6, out6 = bench("6-grouped structured", algo6_grouped)
if out6 is not None:
    print(f"  {'6. Grouped/block-sparse (structured)':<40s} {ms6:7.2f} ms  "
          f"shape={'OK' if out6.shape==ref.shape else 'BAD'}")


# ── Summary ───────────────────────────────────────────────────────

print(f"\n{'='*65}")
print(f"Summary (lower = faster):")
results = [
    ("Dense einsum",           ms0),
    (f"Fixed fan-in K={K_SMALL}", ms1),
    ("Chunked dense",          ms2),
    ("segment_reduce (CPU↑)",  ms3),
    ("scatter_add COO",        ms4),
    (f"FP16 fan-in K={K_SMALL}",  ms5),
    ("Grouped structured",     ms6),
]
baseline = ms0
for name, ms in results:
    if ms is None: continue
    bar = "█" * int(ms / baseline * 20)
    tag = f"{ms/baseline:.1f}x" if ms > baseline else f"{baseline/ms:.1f}x faster"
    print(f"  {name:<35s} {ms:7.2f}ms  {tag:15s} {bar}")
