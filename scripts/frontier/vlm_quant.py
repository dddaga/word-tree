"""Weight-only quantization primitives + the shared verdict/bootstrap helpers for vlm_step010+.

LIBRARY ONLY -- no argparse, no I/O at import. Extracted at vlm_step010 because the step script hit
the repo's 200-line limit. `boot_dd` is a transcription of vlm_step009's, kept byte-identical in
behaviour so a step010 interaction number is directly comparable to a step009 one; vlm_step009 keeps
its inline copy deliberately, being a DONE experiment whose script must stay reproducible as run.

Quantization is symmetric and weight-only: quantize then dequantize back to fp32, so accuracy is
measured exactly as an int kernel would produce it while the byte saving stays analytic.
"""
from __future__ import annotations
import random

# arm name -> (bits, granularity, act_bits) or None for the untouched fp32 reference.
# granularity: True = one scale per row, False = one per tensor, int g = one per g columns.
# act_bits: None = WEIGHT-ONLY (activations stay fp32); an int = activations dynamically quantized
# to that width, per token, which is what an int8xint8 kernel actually requires.
#
# The third field exists because vlm_step015 exposed that the accuracy line and the kernel line were
# measuring different schemes: every arm below with act_bits=None is W*A16, while `torch._int_mm` and
# the Triton kernel both need TWO int8 operands. Making the activation policy an explicit field of
# every arm -- rather than an unstated property of all of them -- is the category fix, so a future
# arm cannot silently be read as having a runtime it has not been measured under.
ARMS = {"fp32": None, "int8_row": (8, True, None), "int4_row": (4, True, None),
        "int8_tensor": (8, False, None), "int4_g64": (4, 64, None), "int4_g192": (4, 192, None),
        "w8a8_row": (8, True, 8), "w4g64a8": (4, 64, 8)}


def fake_quant(w, bits, gran):
    """Symmetric quantize->dequantize. Returns the fp32 tensor an int kernel would see.

    Per-row means one scale per vocabulary token (dim 0). qmax = 2^(bits-1) - 1, so int8 -> 127 and
    int4 -> 7; the negative rail is left at -qmax rather than -2^(bits-1) so the grid stays symmetric
    about zero, which is what a symmetric kernel implements.

    Group-wise (`gran` an int) splits each row into consecutive blocks of `gran` columns with their
    own scale. This is the only untried 4-bit family: vlm_step010 KILLED int4 at ROW granularity,
    where one outlier column forces a coarse step over all 576 of them, and a group confines that
    damage to its own block. `gran` must divide the column count -- 576's divisors include 64 and
    192, but NOT the conventional 128, so the usual group size is silently wrong here and asserted
    against rather than rounded."""
    qmax = 2 ** (bits - 1) - 1
    if gran is True or gran is False:
        amax = w.abs().amax(dim=1, keepdim=True) if gran else w.abs().amax()
        scale = (amax / qmax).clamp(min=1e-12)
        return (w / scale).round().clamp(-qmax, qmax) * scale
    r, c = w.shape
    assert c % gran == 0, f"group {gran} does not divide {c} columns"
    g = w.view(r, c // gran, gran)
    scale = (g.abs().amax(dim=2, keepdim=True) / qmax).clamp(min=1e-12)
    return ((g / scale).round().clamp(-qmax, qmax) * scale).view(r, c)


def act_quant(x, bits):
    """Symmetric PER-TOKEN activation quantize->dequantize: one scale per row of the last dim.

    Per-token, not per-tensor, because that is what a dynamic-quantization kernel computes: the
    scale is derived from the activation actually being multiplied, with no calibration set. Rows
    are independent, so an outlier token cannot coarsen the grid of any other token."""
    qmax = 2 ** (bits - 1) - 1
    s = (x.abs().amax(dim=-1, keepdim=True) / qmax).clamp(min=1e-12)
    return (x / s).round().clamp(-qmax, qmax) * s


_ACT_HOOKS = []


def apply_arm(pristine, arm, act_mod=None):
    """Write arm `arm`'s weights into every table in place, and install/remove its activation hook.

    `pristine` is [(module, fp32_weight)]. `act_mod` is the single module whose INPUT gets quantized
    when the arm asks for it -- `lm_head` and nothing else: the other table is `embed_tokens`, a
    gather whose input is token indices, so "quantizing its activations" is not a defined operation
    and silently doing it to both would be wrong rather than conservative.

    Hooks from the previous arm are removed FIRST, unconditionally. Module-level hook state is ugly,
    but a leaked hook is the one failure that would corrupt the grid invisibly: it would quantize
    activations inside the `fp32` reference cell, and every paired delta in the run is measured
    against that cell."""
    spec = ARMS[arm]
    for h in _ACT_HOOKS:
        h.remove()
    _ACT_HOOKS.clear()
    for mod, w0 in pristine:
        mod.weight.data.copy_(w0 if spec is None else fake_quant(w0, spec[0], spec[1]))
    if spec is not None and spec[2] and act_mod is not None:
        bits = spec[2]
        _ACT_HOOKS.append(act_mod.register_forward_pre_hook(
            lambda mod, inp, b=bits: (act_quant(inp[0], b),) + tuple(inp[1:])))


def table_bytes(n_rows, n_cols, arm):
    """Exact stored bytes for ONE table under arm `arm`: packed weights + fp32 scales.

    Group-wise scales are NOT free and are counted here: g=64 over 576 columns is 9 fp32 scales per
    row, i.e. 4 + 32/64 = 4.5 effective bits per weight against int4_row's 4.06. A group scheme that
    ships must beat int4_row on ACCURACY while its byte cost is read from this function, never
    assumed to be 4 bits."""
    spec = ARMS[arm]
    if spec is None: return n_rows * n_cols * 4
    bits, gran = spec[0], spec[1]   # act_bits does not affect STORED bytes -- activations are transient
    packed = n_rows * n_cols * bits // 8
    if gran is True: return packed + n_rows * 4
    if gran is False: return packed + 4
    return packed + n_rows * (n_cols // gran) * 4


def hits(recs, name):
    return [int(r[name] == r["gold"]) for r in recs]


def boot_dd(recs, cells, n_boot, seed=0):
    """Percentile bootstrap on the double difference over four cell names, ordered
    (arm@depth, ref@depth, arm@ref_depth, ref@ref_depth). One shared resample index per replicate
    feeds all four, so the four-way pairing is preserved. Returns (point, lo, hi)."""
    cols = [hits(recs, c) for c in cells]
    n, rng, out = len(recs), random.Random(seed), []
    point = (sum(cols[0]) - sum(cols[1]) - sum(cols[2]) + sum(cols[3])) / n
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        s = [sum(c[i] for i in idx) for c in cols]
        out.append((s[0] - s[1] - s[2] + s[3]) / n)
    out.sort()
    return point, out[int(0.025 * n_boot)], out[int(0.975 * n_boot)]


def call_ship(lo, hi, tol=0.02):
    """Pre-registered ship rule: a setting ships iff its paired CI's lower bound clears -tol."""
    if lo >= -tol: return "SHIP"
    if hi < -tol: return "KILLED"
    return "INCONCLUSIVE at this n"


def call_ortho(lo, hi, tol=0.02, lever="the second lever"):
    """Pre-registered compounding rule, same tolerance, applied to the double difference.

    `lever` names whatever is crossed with depth, so the verdict string is accurate in every
    experiment that calls this. It defaulted to quantization when only vlm_step010 used the helper;
    vlm_step011 crosses depth with object SCALE, and a saved verdict that says "quantization" in a
    run with no quantization in it is a misleading record. Wording only -- the rule is unchanged."""
    if lo >= -tol and hi <= tol: return "ORTHOGONAL -- levers compound"
    if hi < -tol: return f"SUB-ADDITIVE -- {lever} costs the truncated tower more"
    if lo > tol: return "SUPER-ADDITIVE -- replicate before use"
    return "INCONCLUSIVE at this n"
