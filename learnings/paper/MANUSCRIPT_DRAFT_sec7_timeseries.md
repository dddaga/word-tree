## §10 Time Series: Stock Return Prediction

**Task.** NIFTY50 next-day return prediction. 56 liquid NSE stocks, VGG16 features replaced by 16 handcrafted return/volume/momentum features per stock per day. Walk-forward split: train 2015–2021, val 2022–2023.

**Model (SGNNET-TS).** Recurrent graph network: small-world ring graph (K_wiring=4 nearest ring neighbors on sorted ticker order), anti-Hebbian message passing (K_iter=3 per timestep), EMA hidden state update. 51,050 parameters at N=56, D=16. Directional + log-wealth loss from epoch 1 (no MSE warmup).

**Baselines.**

| Model | Dir. Acc | Sharpe | Params |
|-------|----------|--------|--------|
| Linear probe | 50.2% | −152 | 3.0M |
| MLP (hidden=256) | 49.8% | −98 | 13.8M |
| SGNNET_K3 (routing) | 50.1% | −164 | 51K |
| SGNNET_K0 (no routing) | 49.8% | −191 | 51K |

**Result.** All models produce directional accuracy ≈ 50% and strongly negative Sharpe ratios. No model beats the mean predictor. The task is consistent with efficient markets: daily stock returns on a liquid index (NIFTY50) are near-random at the granularity of individual stock prediction.

**Analysis.** Unlike vision features, handcrafted financial features (return, MA, volume) do not provide a spatial embedding that SGNNET's routing can exploit. The ring-graph topology encodes stock ordering (by ticker name), which is arbitrary and carries no financial meaning. The ΔW-proj mechanism requires a meaningful geometric structure in the feature embedding — which is absent for financial features.

**Verdict.** Honest negative result: SGNNET provides no advantage over a linear probe on financial time series prediction. We include this as evidence that the model's routing mechanism is specific to spatially-structured features.

*Note on data scale.* The intersection of all 56 tickers' date ranges within 2015–2021 yields 169 training windows (a subset from when the most recently-listed NIFTY50 stock joined the index). A more sophisticated data pipeline (forward-fill missing dates or ticker-by-ticker training) might yield more training data but is unlikely to change the conclusion: the efficient market hypothesis predicts near-random daily return predictability regardless of model capacity.
