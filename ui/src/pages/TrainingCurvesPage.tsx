import { useEffect, useState } from 'react';
import type { ValueType } from 'recharts/types/component/DefaultTooltipContent';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts';
import type { ModelResult, HistoryRow } from '../types';

interface PlotPoint {
  epoch: number;
  [modelKey: string]: number | undefined;
}

const MODEL_FILES = [
  { file: '/data/ts_step010_cnn_lstm.json', key: 'CNN_LSTM', color: '#60a5fa' },
  { file: '/data/ts_step011_cnn_transformer.json', key: 'CNN_Transformer', color: '#34d399' },
  { file: '/data/ts_step030_sgnnet_ts.json', key: 'SGNNET', color: '#f472b6' },
];

// MAE and GAN don't have val_dir_acc / val_mae per epoch in their history keys,
// so they are shown in a separate "pretrain loss" section only.
const PRETRAIN_FILES = [
  { file: '/data/ts_step020_mae_encoder.json', key: 'MAE_Encoder', color: '#fbbf24' },
];

function buildOverlayData(
  models: Array<{ key: string; history: HistoryRow[] }>,
  field: 'val_dir_acc' | 'val_mae',
): PlotPoint[] {
  // Collect all epochs across models
  const epochSet = new Set<number>();
  models.forEach(({ history }) => {
    history.forEach((h) => {
      if (h[field] != null) epochSet.add(h.epoch);
    });
  });
  const epochs = Array.from(epochSet).sort((a, b) => a - b);

  return epochs.map((epoch) => {
    const point: PlotPoint = { epoch };
    models.forEach(({ key, history }) => {
      const row = history.find((h) => h.epoch === epoch);
      if (row && row[field] != null) {
        point[key] = row[field] as number;
      }
    });
    return point;
  });
}

export default function TrainingCurvesPage() {
  const [models, setModels] = useState<Array<{ key: string; color: string; history: HistoryRow[] }>>([]);
  const [maeModel, setMaeModel] = useState<ModelResult | null>(null);
  const [errors, setErrors] = useState<string[]>([]);

  useEffect(() => {
    const loaded: typeof models = [];
    const errs: string[] = [];

    Promise.all(
      MODEL_FILES.map(({ file, key, color }) =>
        fetch(file)
          .then((r) => r.json())
          .then((d: ModelResult) => {
            const history = d.history ?? [];
            // Filter to rows that have the fields we need
            loaded.push({ key, color, history });
          })
          .catch((e) => errs.push(`${key}: ${e}`)),
      ),
    ).then(() => {
      setModels(loaded);
      setErrors((prev) => [...prev, ...errs]);
    });

    // Load MAE separately
    fetch(PRETRAIN_FILES[0].file)
      .then((r) => r.json())
      .then((d: ModelResult) => setMaeModel(d))
      .catch((e) => setErrors((prev) => [...prev, String(e)]));
  }, []);

  // Filter models that have val_dir_acc rows
  const modelsWithDirAcc = models.filter((m) =>
    m.history.some((h) => h.val_dir_acc != null),
  );

  // Filter models that have val_mae rows
  const modelsWithMae = models.filter((m) =>
    m.history.some((h) => h.val_mae != null),
  );

  const dirAccData = buildOverlayData(modelsWithDirAcc, 'val_dir_acc');
  const maeData = buildOverlayData(modelsWithMae, 'val_mae');

  // MAE encoder pretrain loss
  const maePretrainData =
    maeModel?.history?.map((h) => ({
      epoch: h.epoch,
      train_loss: h.train_loss,
      val_loss: h.val_loss,
    })) ?? [];

  const modelColorMap = Object.fromEntries(
    [...MODEL_FILES, ...PRETRAIN_FILES].map(({ key, color }) => [key, color]),
  );

  return (
    <div className="space-y-8">
      <h2 className="text-lg font-semibold text-slate-100">Training Curves</h2>

      {errors.length > 0 && (
        <div className="bg-red-900/30 border border-red-700 rounded p-3 text-sm text-red-300">
          {errors.map((e, i) => <div key={i}>{e}</div>)}
        </div>
      )}

      {/* Directional Accuracy */}
      <div className="bg-slate-800 rounded-lg p-4">
        <h3 className="text-sm font-medium text-slate-300 mb-1">
          Validation Directional Accuracy over Epochs
        </h3>
        <p className="text-xs text-slate-500 mb-4">
          Dashed lines: 0.52 target, 0.50 random baseline
        </p>
        {dirAccData.length === 0 ? (
          <div className="text-slate-400 text-sm">No directional accuracy data.</div>
        ) : (
          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={dirAccData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis
                dataKey="epoch"
                tick={{ fill: '#94a3b8', fontSize: 11 }}
                label={{ value: 'Epoch', position: 'insideBottom', offset: -2, fill: '#64748b', fontSize: 11 }}
              />
              <YAxis
                domain={[0.48, 0.56]}
                tick={{ fill: '#94a3b8', fontSize: 11 }}
                tickFormatter={(v) => v.toFixed(2)}
              />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', color: '#e2e8f0', fontSize: 12 }}
                formatter={(val: ValueType | undefined) => (typeof val === 'number' ? val.toFixed(4) : String(val ?? ''))}
              />
              <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
              <ReferenceLine y={0.52} stroke="#facc15" strokeDasharray="6 3" label={{ value: '0.52 target', fill: '#facc15', fontSize: 10 }} />
              <ReferenceLine y={0.50} stroke="#64748b" strokeDasharray="4 4" label={{ value: '0.50 random', fill: '#64748b', fontSize: 10 }} />
              {modelsWithDirAcc.map(({ key }) => (
                <Line
                  key={key}
                  type="monotone"
                  dataKey={key}
                  stroke={modelColorMap[key]}
                  dot={false}
                  strokeWidth={1.5}
                  connectNulls={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Validation MAE */}
      <div className="bg-slate-800 rounded-lg p-4">
        <h3 className="text-sm font-medium text-slate-300 mb-4">
          Validation MAE over Epochs
        </h3>
        {maeData.length === 0 ? (
          <div className="text-slate-400 text-sm">No MAE data.</div>
        ) : (
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={maeData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="epoch" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} tickFormatter={(v) => v.toFixed(3)} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', color: '#e2e8f0', fontSize: 12 }}
                formatter={(val: ValueType | undefined) => (typeof val === 'number' ? val.toFixed(5) : String(val ?? ''))}
              />
              <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
              {modelsWithMae.map(({ key }) => (
                <Line
                  key={key}
                  type="monotone"
                  dataKey={key}
                  stroke={modelColorMap[key]}
                  dot={false}
                  strokeWidth={1.5}
                  connectNulls={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* MAE Encoder pretrain loss */}
      {maePretrainData.length > 0 && (
        <div className="bg-slate-800 rounded-lg p-4">
          <h3 className="text-sm font-medium text-slate-300 mb-1">
            MAE Encoder — Pretrain Reconstruction Loss
          </h3>
          <p className="text-xs text-slate-500 mb-4">
            No per-epoch dir_acc available for MAE/GAN encoders; showing pretrain val_loss only.
          </p>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={maePretrainData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="epoch" tick={{ fill: '#94a3b8', fontSize: 11 }} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} tickFormatter={(v) => v.toFixed(4)} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', color: '#e2e8f0', fontSize: 12 }}
                formatter={(val: ValueType | undefined) => (typeof val === 'number' ? val.toFixed(6) : String(val ?? ''))}
              />
              <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
              <Line type="monotone" dataKey="train_loss" stroke="#fbbf24" dot={false} strokeWidth={1.5} name="Train Loss" />
              <Line type="monotone" dataKey="val_loss" stroke="#fb923c" dot={false} strokeWidth={1.5} name="Val Loss" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
