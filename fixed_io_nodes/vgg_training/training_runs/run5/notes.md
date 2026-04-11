# run5 — Notes

**Config delta from run4:** radiation_targets=16, scattering_prob=0.25, stochastic_radiation_duration=0.08, 40ep. LR decay kept.
**Result:** val best=71.46% (ep33), final=70.01% (ep39). Train acc=87.09%.
**Notes:** Recovered from run4 collapse but 10pp below run3 (81.38%) at 20ep. Val peaked ep33 then degraded — signs of overfitting without dropout. Radiation appears to hurt net.
**Finding: HYPOTHESIS — radiation (even at reduced settings) hurts vs no radiation. Confounded by LR decay addition.**
