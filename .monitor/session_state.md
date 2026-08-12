## Session State — 2026-04-19 07:36 UTC

### Running Experiments
**All slots FREE.** Recently cleared: step877–890.

### Key Findings This Session

| Step | Config | Result | Verdict |
|------|--------|--------|---------|
| **step887** (canonical) | 34,976p, ref config | 96.38% ± 0.18pp (3-seed) | **PAPER CLAIM LOCKED** |
| **step889** | K_hh=1 + K_in=15 compound T2 | −1.43pp @ 0.57× FLOPs | **CONFIRMED VIABLE** (43% FLOPs cut) |
| **step885** | K_hh=1 T2 (no K_in change) | −0.74pp | Marginal; mention w/ caveat |
| **step886** | ΔW-proj ablation T1 | A_sign, B_no_ref load-bearing; C_no_theta neutral | **θ optional — simplify** |
| **step883** | ΔW-proj ablation T0 | D_rand_dir=−76.56pp → geometry essential | **Components all load-bearing at T1** |
| **step882** | CIFAR-10 T2 (canonical) | 80.69% vs Linear 86.24%, Δ=−5.55pp | **MARGINAL** (within paper range, 7.4× fewer params) |
| **step890** | CIFAR-10 T0 canonical | Δ=−10.70pp | T0 kills; T2 shows −5.55pp viable |

### Decisions Made
- **Paper baseline locked:** 96.38% ± 0.18pp (step887, 34,976 params). 
- **K_hh=1 efficiency route confirmed:** −1.43pp for 0.57× FLOPs; advancing as secondary finding.
- **ΔW-proj geometry essential:** A_sign + B_no_ref carry signal; θ removal viable (post-paper refactor).
- **CIFAR-10 gap honest:** −5.55pp on T2; within presentation margin. **Paper scope = Imagenette only** (text tasks negative per prior).
- **FLOPs audit closed:** Routing-only MACs = 0.98M; true per-sample ≈6.5M (19× vs VGG FC). Labels confirmed.

### Next Actions
- Paper assembly: canonical results + K_hh=1 efficiency trade-off + cross-dataset honest negative.
- If new direction: check memory/decision logs in `learnings/` for load-bearing findings before pivoting.
- All slots ready for final validation runs or pivot experiments.

---
