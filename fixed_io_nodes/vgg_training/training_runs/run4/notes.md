# run4 — Notes

**Config delta from run3:** +radiation_targets=32, scattering_prob=0.8, stochastic_radiation_duration=0.4, +lr_decay_factor=0.5, plateau_patience=5.
**Result:** val best=16.89% (ep2). Stopped at 3 epochs.
**Notes:** Collapsed immediately. Confounded — two changes at once (radiation + LR decay). Possible failure modes: (1) scattering_prob=0.8 too aggressive, destroying gradient signal; (2) LR scheduler interacting badly early in training.
**Finding: HYPOTHESIS — high scattering_prob is too aggressive. Not a clean test.**
