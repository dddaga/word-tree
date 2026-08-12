---
name: project_cifar10_scaling
description: CIFAR-10 N-scaling law results — monotonic confirmed through N=16384
metadata:
  type: project
---

step986 T2 DONE (2026-06-21): CIFAR-10 N=16384 T2 mean=84.66% ±0.06pp (3 seeds: 42=84.60%, 43=84.75%, 44=84.63%). 150ep, 100% data.

Full T2 scaling curve: N=2048=80.57%, N=4096=82.53%, N=8192=83.55%, N=16384=84.66%.
Gap to linear ceiling (86.24%): −1.58pp.

**Why:** Confirm whether N=16384 breaks the monotonic trend or is still scaling.
**How to apply:** Scaling curve is monotonic — N=16384 is NOT a ceiling. Paper scaling section should report this as an open trend. Next N=32768 would be the next test if needed.
