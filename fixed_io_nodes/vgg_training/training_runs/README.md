## Details regarding all the runs

1) `run1`  - Original run, just to check if model was working and could learn. Ran WITHOUT layernorm, dropout, or radiation. Same with anything else. 

2) `run2` - Ran with a bigger network (config in the folder). It was found to run significantly slower, and found to be learning much slower than the network in `run1`, hence training was stopped in between. (No weights or graphs are available unfortunately)

3) `run3` - Same config as `run1`, with just added Layernorm. Training accuracy is lower, but Val accuracy goes higher than `run1`

4) `run4` - Included radiation while running with `run3` - but the radiation targets were 32, and stochastic radiation was 80%, along with linear decay, which was too high and the model had too much randomness. (This run cannot be replicated as the uniform decay is not replaced with exponential decay)

5) `run5` - Replaced uniform decay for stochastic radiation with exponential decay. Reduced radiation target from 32 to 16, and starts with 4 stochastic targets (25% of radiation targets). 

6) `run6` - Added dropout with 0.3 probability, along with layernorm. No radiation present. This was run on CPU since GPU was being used by (5)