# Wave Architecture Comparison

## Aggregate Metrics

| Model | Params | % VGG FC | Top-1 | mAP | Key addition |
|-------|--------|----------|-------|-----|--------------|
| VGG16 FC (frozen) | 123,642,856 | 100% | 0.9954 | 0.9997 | Reference |
| SGNNET v1 (Phase 3) | 649,330 | 0.53% | -- | -- | Learned C, amplitude routing |
| Stage A (static) | 1,064 | 0.0% | 0.1027 | 0.1103 | Binary C, no dynamic |
| Exp 1 (spatial phase) | 1,064 | 0.0% | 0.1197 | 0.1160 | + proximity w/ path-length phase |
| Exp 2 (spatial + W_phase) | 2,128 | 0.0% | 0.1052 | 0.1047 | + learned phase operator |

## Contribution Analysis

- **Exp 1 vs Stage A:** top1 delta = +0.0171, mAP delta = +0.0057
  - Proximity routing with spatial phase improves over static wiring
- **Exp 2 vs Exp 1:** top1 delta = -0.0145, mAP delta = -0.0113
  - W_phase adds 1064 parameters
  - Learned phase operator does not improve over geometry-only phase

## Per-Class Comparison

| Class | VGG16 Acc | Stage A Acc | Exp1 Acc | Exp2 Acc | Exp1 AP | Exp2 AP |
|-------|-----------|-------------|----------|----------|---------|---------|
| tench | 1.0000 | 0.0000 | 0.5891 | 0.1266 | 0.0902 | 0.0831 |
| english_springer | 1.0000 | 0.0000 | 0.0380 | 0.7873 | 0.1194 | 0.0923 |
| cassette_player | 0.9944 | 0.0672 | 0.0308 | 0.0168 | 0.0809 | 0.0847 |
| chain_saw | 0.9845 | 0.9715 | 0.0000 | 0.0155 | 0.0957 | 0.0997 |
| church | 0.9951 | 0.0000 | 0.0293 | 0.0000 | 0.0979 | 0.0974 |
| french_horn | 0.9975 | 0.0102 | 0.1802 | 0.0635 | 0.1249 | 0.1497 |
| garbage_truck | 1.0000 | 0.0000 | 0.2365 | 0.0000 | 0.1756 | 0.1167 |
| gas_pump | 0.9928 | 0.0000 | 0.0024 | 0.0000 | 0.1134 | 0.1072 |
| golf_ball | 0.9925 | 0.0000 | 0.0927 | 0.0376 | 0.1574 | 0.1119 |
| parachute | 0.9974 | 0.0000 | 0.0077 | 0.0026 | 0.1049 | 0.1040 |

## Hyperparameters Used

**Stage A:** K=2, N_hidden=256, lr_Wpos=8.04e-03, lambda_safety=0.520, batch_size=256
**Exp 1:** K=2, N_hidden=256, lr_Wpos=2.36e-03, lambda_safety=0.691, batch_size=64
**Exp 2:** K=2, N_hidden=256, lr_Wpos=1.00e-02, lambda_safety=0.897, batch_size=64
