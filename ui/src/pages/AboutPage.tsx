export default function AboutPage() {
  return (
    <div className="max-w-2xl space-y-6">
      <h2 className="text-lg font-semibold text-slate-100">About</h2>

      <div className="bg-slate-800 rounded-lg p-6 space-y-4 text-sm text-slate-300 leading-relaxed">
        <p>
          <span className="text-slate-100 font-medium">Project:</span> SGNNET applied to financial
          time-series prediction. Comparison of CNN+LSTM, CNN+Transformer, MAE encoder, GAN
          encoder, and SGNNET with activation retention on 56 Indian stocks.
        </p>

        <div>
          <div className="text-slate-100 font-medium mb-2">Models evaluated</div>
          <ul className="space-y-2 list-none">
            {[
              ['CNN_LSTM', 'ts_step010', '2.1M params. Standard CNN feature extractor + LSTM temporal encoder. Trained 100 epochs with cosine LR warmup.'],
              ['CNN_Transformer', 'ts_step011', '791K params. CNN encoder + 2-layer Transformer (d=128, 4 heads). Trained 100 epochs.'],
              ['MAE_ContextEncoder', 'ts_step020', '52.5K params. Masked autoencoder pretrained on 25% masked patches, finetuned for direction prediction.'],
              ['GAN_ContextEncoder', 'ts_step021', '~279K total params (gen+disc+head). GAN pretrain then supervised finetune on direction.'],
              ['SGNNETRecurrent', 'ts_step030', '51K params. Sparse graph neural network with Z-memory (activation retention). Early 2-epoch run.'],
            ].map(([model, step, desc]) => (
              <li key={step} className="bg-slate-700/40 rounded p-3">
                <div className="flex items-baseline gap-2 mb-1">
                  <span className="text-slate-100 font-medium">{model}</span>
                  <span className="text-xs text-slate-500">{step}</span>
                </div>
                <p className="text-slate-400 text-xs">{desc}</p>
              </li>
            ))}
          </ul>
        </div>

        <div>
          <div className="text-slate-100 font-medium mb-2">Task</div>
          <p>
            Predict next-day return direction (up/down) for 56 NSE-listed stocks using a 60-day
            lookback window. Secondary targets: MAE on normalized return, portfolio Sharpe ratio
            via simulated trading.
          </p>
        </div>

        <div>
          <div className="text-slate-100 font-medium mb-2">Benchmark strategies (sim_step002)</div>
          <p>
            Four backtested strategies over ~493 trading days: NiftyIndex (passive NIFTY-50
            replication), BuyAndHold, MovingAverage crossover, RandomSignal baseline.
          </p>
        </div>

        <div>
          <div className="text-slate-100 font-medium mb-2">Thesis</div>
          <p>
            SGNNET's sparse O(N·K) message-passing on a spherical hypersphere encoding is
            hypothesized to improve parameter efficiency vs. dense architectures. The financial
            time-series track tests whether this extends from vision to market microstructure.
          </p>
        </div>
      </div>
    </div>
  );
}
