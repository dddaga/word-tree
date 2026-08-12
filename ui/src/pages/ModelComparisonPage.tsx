import { useEffect, useState } from 'react';
import type { ModelResult } from '../types';

interface ModelRow {
  model: string;
  step: string;
  n_params: number;
  dir_acc: number | null;
  mae: number | null;
  best_sharpe: number | null;
  notes: string;
}

const FILES = [
  '/data/ts_step010_cnn_lstm.json',
  '/data/ts_step011_cnn_transformer.json',
  '/data/ts_step020_mae_encoder.json',
  '/data/ts_step021_gan_encoder.json',
  '/data/ts_step030_sgnnet_ts.json',
];

function extractRow(d: ModelResult): ModelRow {
  // GAN has split param counts
  let n_params = d.n_params ?? 0;
  if (!d.n_params && d.n_params_generator != null) {
    n_params =
      (d.n_params_generator ?? 0) +
      (d.n_params_discriminator ?? 0) +
      (d.n_params_head ?? 0);
  }

  // MAE encoder uses different final metric keys
  const dir_acc =
    d.final_val_dir_acc ??
    d.val_directional_acc ??
    null;

  const mae = d.final_val_mae ?? null;

  const best_sharpe =
    d.best_val_sharpe ??
    d.val_sharpe ??
    null;

  const notes =
    d.model === 'SGNNETRecurrent'
      ? 'Sparse graph NN; 2 epochs only — early run'
      : d.model === 'MAE_ContextEncoder'
      ? 'Pretrain-only metrics; no finetune dir_acc per epoch'
      : d.model === 'GAN_ContextEncoder'
      ? 'Pretrain + finetune; large head (200K params)'
      : '';

  return {
    model: d.model,
    step: d.step,
    n_params,
    dir_acc,
    mae,
    best_sharpe,
    notes,
  };
}

function fmt(v: number | null, decimals = 4): string {
  if (v == null) return '—';
  return v.toFixed(decimals);
}

function fmtParams(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

export default function ModelComparisonPage() {
  const [rows, setRows] = useState<ModelRow[]>([]);
  const [errors, setErrors] = useState<string[]>([]);

  useEffect(() => {
    const loaded: ModelRow[] = [];
    const errs: string[] = [];

    Promise.all(
      FILES.map((file) =>
        fetch(file)
          .then((r) => r.json())
          .then((d: ModelResult) => loaded.push(extractRow(d)))
          .catch((e) => errs.push(`${file}: ${e}`)),
      ),
    ).then(() => {
      // Sort by params ascending
      loaded.sort((a, b) => a.n_params - b.n_params);
      setRows(loaded);
      setErrors(errs);
    });
  }, []);

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold text-slate-100">Model Comparison</h2>

      {errors.length > 0 && (
        <div className="bg-red-900/30 border border-red-700 rounded p-3 text-sm text-red-300">
          {errors.map((e, i) => <div key={i}>{e}</div>)}
        </div>
      )}

      <div className="bg-slate-800 rounded-lg p-4 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-slate-400 border-b border-slate-700">
              <th className="text-left py-2 pr-4">Model</th>
              <th className="text-right py-2 pr-4">Params</th>
              <th className="text-right py-2 pr-4">Dir Acc</th>
              <th className="text-right py-2 pr-4">MAE</th>
              <th className="text-right py-2 pr-4">Best Sharpe</th>
              <th className="text-left py-2">Notes</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => {
              const isSgnnet = row.model.toLowerCase().includes('sgnnet');
              return (
                <tr
                  key={row.step}
                  className={`border-b border-slate-700/50 ${i % 2 === 0 ? '' : 'bg-slate-700/20'} ${isSgnnet ? 'ring-1 ring-inset ring-blue-600/40' : ''}`}
                >
                  <td className="py-2 pr-4">
                    <div className="font-medium text-slate-200">{row.model}</div>
                    <div className="text-xs text-slate-500">{row.step}</div>
                  </td>
                  <td className="text-right py-2 pr-4 font-mono text-slate-300">
                    {fmtParams(row.n_params)}
                  </td>
                  <td className={`text-right py-2 pr-4 font-mono ${
                    row.dir_acc != null && row.dir_acc >= 0.52
                      ? 'text-green-400'
                      : row.dir_acc != null && row.dir_acc >= 0.505
                      ? 'text-yellow-400'
                      : 'text-slate-300'
                  }`}>
                    {fmt(row.dir_acc, 4)}
                  </td>
                  <td className="text-right py-2 pr-4 font-mono text-slate-300">
                    {fmt(row.mae, 5)}
                  </td>
                  <td className={`text-right py-2 pr-4 font-mono ${
                    row.best_sharpe != null && row.best_sharpe >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {fmt(row.best_sharpe, 2)}
                  </td>
                  <td className="py-2 text-xs text-slate-400 max-w-xs">{row.notes}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="bg-slate-800 rounded-lg p-4">
        <h3 className="text-sm font-medium text-slate-300 mb-2">Legend</h3>
        <div className="text-xs text-slate-400 space-y-1">
          <div><span className="text-green-400">Green dir_acc</span> — at or above 0.52 target</div>
          <div><span className="text-yellow-400">Yellow dir_acc</span> — between 0.505 and 0.52</div>
          <div><span className="text-blue-400 ring-1 ring-blue-600 px-1 rounded">SGNNET rows</span> — highlighted</div>
          <div>Best Sharpe — most negative Sharpe from validation (all models are pre-profitability)</div>
        </div>
      </div>
    </div>
  );
}
