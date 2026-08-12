import { useEffect, useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

const MODELS = [
  { id: 'ts_step010', label: 'CNN+LSTM' },
  { id: 'ts_step011', label: 'CNN+Transformer' },
  { id: 'ts_step020', label: 'MAE Encoder' },
  { id: 'ts_step021', label: 'GAN Encoder' },
  { id: 'ts_step030', label: 'SGNNET-TS' },
];

const MODEL_LABELS: Record<string, string> = Object.fromEntries(
  MODELS.map((m) => [m.id, m.label])
);

interface Experiment {
  id: string;
  model: string;
  step?: string;
  status: string;
  epochs: number;
  val_sharpe?: number;
  dir_acc?: number;
  val_loss?: number;
  params?: number;
  created: string;
  [key: string]: unknown;
}

type SortKey = 'model' | 'step' | 'epochs' | 'val_sharpe' | 'dir_acc' | 'val_loss' | 'params' | 'created';

const SORT_KEYS: { key: SortKey; label: string }[] = [
  { key: 'model', label: 'Model' },
  { key: 'step', label: 'Step' },
  { key: 'epochs', label: 'Epochs' },
  { key: 'val_sharpe', label: 'Val Sharpe' },
  { key: 'dir_acc', label: 'Dir Acc' },
  { key: 'val_loss', label: 'Val Loss' },
  { key: 'params', label: 'Params' },
  { key: 'created', label: 'Created' },
];

const COMPARE_COLORS = ['#60a5fa', '#34d399', '#f472b6', '#fbbf24'];

export default function ResultsDBPage() {
  const [all, setAll] = useState<Experiment[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [modelFilter, setModelFilter] = useState<string[]>([]);
  const [statusFilter, setStatusFilter] = useState('');
  const [minSharpe, setMinSharpe] = useState(-10);

  // Sort
  const [sortKey, setSortKey] = useState<SortKey>('created');
  const [sortAsc, setSortAsc] = useState(false);

  // Expanded row
  const [expanded, setExpanded] = useState<string | null>(null);

  // Comparison selection (up to 4)
  const [compareIds, setCompareIds] = useState<string[]>([]);

  useEffect(() => {
    fetch('/api/experiments?limit=200')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setAll)
      .catch((e) => setError(String(e)));
  }, []);

  function toggleModelFilter(id: string) {
    setModelFilter((prev) =>
      prev.includes(id) ? prev.filter((m) => m !== id) : [...prev, id]
    );
  }

  function toggleCompare(id: string) {
    setCompareIds((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id);
      if (prev.length >= 4) return prev;
      return [...prev, id];
    });
  }

  function handleSortClick(key: SortKey) {
    if (sortKey === key) {
      setSortAsc((v) => !v);
    } else {
      setSortKey(key);
      setSortAsc(false);
    }
  }

  const filtered = all
    .filter((e) => modelFilter.length === 0 || modelFilter.includes(e.model))
    .filter((e) => !statusFilter || e.status === statusFilter)
    .filter((e) => (e.val_sharpe ?? -Infinity) >= minSharpe)
    .sort((a, b) => {
      const av = a[sortKey] ?? '';
      const bv = b[sortKey] ?? '';
      const cmp = av < bv ? -1 : av > bv ? 1 : 0;
      return sortAsc ? cmp : -cmp;
    });

  // Comparison chart data
  const compareExps = compareIds
    .map((id) => all.find((e) => e.id === id))
    .filter(Boolean) as Experiment[];

  const maxParams = Math.max(...compareExps.map((e) => e.params ?? 0), 1);

  const compareChartData = [
    {
      metric: 'Val Sharpe',
      ...Object.fromEntries(compareExps.map((e) => [MODEL_LABELS[e.model] ?? e.model, e.val_sharpe ?? 0])),
    },
    {
      metric: 'Dir Acc',
      ...Object.fromEntries(compareExps.map((e) => [MODEL_LABELS[e.model] ?? e.model, e.dir_acc != null ? e.dir_acc * 100 : 0])),
    },
    {
      metric: 'Params (norm)',
      ...Object.fromEntries(compareExps.map((e) => [MODEL_LABELS[e.model] ?? e.model, (e.params ?? 0) / maxParams])),
    },
  ];

  const compareLabels = compareExps.map((e) => MODEL_LABELS[e.model] ?? e.model);

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold text-slate-100">Results DB</h2>

      {/* Filter bar */}
      <div className="bg-slate-800 rounded-lg p-4 space-y-3">
        <h3 className="text-sm font-medium text-slate-300">Filters</h3>
        <div className="flex flex-wrap gap-4 items-end">
          {/* Model multi-select */}
          <div>
            <div className="text-xs text-slate-400 mb-1">Model</div>
            <div className="flex flex-wrap gap-1">
              {MODELS.map((m) => (
                <button
                  key={m.id}
                  onClick={() => toggleModelFilter(m.id)}
                  className={`px-2 py-0.5 rounded text-xs font-medium transition-colors ${
                    modelFilter.includes(m.id)
                      ? 'bg-blue-600 text-white'
                      : 'bg-slate-700 text-slate-300 hover:bg-slate-600'
                  }`}
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>

          {/* Status filter */}
          <div>
            <div className="text-xs text-slate-400 mb-1">Status</div>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="bg-slate-700 border border-slate-600 rounded px-2 py-1 text-sm text-slate-200 focus:outline-none focus:border-blue-500"
            >
              <option value="">All</option>
              <option value="running">running</option>
              <option value="done">done</option>
              <option value="failed">failed</option>
            </select>
          </div>

          {/* Min val sharpe slider */}
          <div>
            <div className="text-xs text-slate-400 mb-1">
              Min Val Sharpe: <span className="text-slate-300">{minSharpe.toFixed(1)}</span>
            </div>
            <input
              type="range"
              min={-10}
              max={5}
              step={0.1}
              value={minSharpe}
              onChange={(e) => setMinSharpe(Number(e.target.value))}
              className="w-40"
            />
          </div>
        </div>
      </div>

      {/* Results table */}
      <div className="bg-slate-800 rounded-lg p-4">
        <h3 className="text-sm font-medium text-slate-300 mb-3">
          Results
          <span className="ml-2 text-xs text-slate-500">({filtered.length} rows)</span>
        </h3>
        {error && <div className="text-red-400 text-sm">Error: {error}</div>}
        {!error && all.length === 0 && (
          <div className="text-slate-400 text-sm">Loading...</div>
        )}
        {filtered.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-slate-400 border-b border-slate-700">
                  <th className="text-left py-2 pr-2 w-6">
                    <span className="text-xs text-slate-500">Cmp</span>
                  </th>
                  {SORT_KEYS.map(({ key, label }) => (
                    <th
                      key={key}
                      className="text-left py-2 pr-4 cursor-pointer hover:text-slate-200 select-none whitespace-nowrap"
                      onClick={() => handleSortClick(key)}
                    >
                      {label}
                      {sortKey === key && (
                        <span className="ml-1 text-blue-400">{sortAsc ? '↑' : '↓'}</span>
                      )}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.map((exp, i) => (
                  <>
                    <tr
                      key={exp.id}
                      onClick={() => setExpanded(expanded === exp.id ? null : exp.id)}
                      className={`border-b border-slate-700/50 cursor-pointer transition-colors ${
                        i % 2 === 0 ? 'bg-slate-800' : 'bg-slate-750'
                      } hover:bg-slate-700/60`}
                    >
                      <td className="py-2 pr-2" onClick={(e) => e.stopPropagation()}>
                        <input
                          type="checkbox"
                          checked={compareIds.includes(exp.id)}
                          onChange={() => toggleCompare(exp.id)}
                          disabled={!compareIds.includes(exp.id) && compareIds.length >= 4}
                          className="accent-blue-500"
                        />
                      </td>
                      <td className="py-2 pr-4 text-slate-200 font-medium">
                        {MODEL_LABELS[exp.model] ?? exp.model}
                      </td>
                      <td className="py-2 pr-4 text-slate-400 text-xs">{exp.step ?? '—'}</td>
                      <td className="py-2 pr-4 text-slate-300">{exp.epochs}</td>
                      <td className={`py-2 pr-4 ${(exp.val_sharpe ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {exp.val_sharpe != null ? exp.val_sharpe.toFixed(3) : '—'}
                      </td>
                      <td className="py-2 pr-4 text-slate-300">
                        {exp.dir_acc != null ? `${(exp.dir_acc * 100).toFixed(1)}%` : '—'}
                      </td>
                      <td className="py-2 pr-4 text-slate-300">
                        {exp.val_loss != null ? exp.val_loss.toFixed(4) : '—'}
                      </td>
                      <td className="py-2 pr-4 text-slate-300">
                        {exp.params != null ? exp.params.toLocaleString() : '—'}
                      </td>
                      <td className="py-2 pr-4 text-slate-400 text-xs whitespace-nowrap">{exp.created}</td>
                    </tr>
                    {expanded === exp.id && (
                      <tr key={`${exp.id}-detail`} className="bg-slate-900/60">
                        <td colSpan={9} className="p-4">
                          <pre className="text-xs font-mono text-slate-300 whitespace-pre-wrap break-words overflow-x-auto max-h-64">
                            {JSON.stringify(exp, null, 2)}
                          </pre>
                        </td>
                      </tr>
                    )}
                  </>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Comparison chart */}
      {compareIds.length >= 2 && (
        <div className="bg-slate-800 rounded-lg p-4">
          <h3 className="text-sm font-medium text-slate-300 mb-4">
            Comparison ({compareIds.length} experiments)
          </h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={compareChartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', color: '#e2e8f0' }}
              />
              <Legend wrapperStyle={{ color: '#94a3b8' }} />
              {compareLabels.map((label, idx) => (
                <Bar key={label} dataKey={label} fill={COMPARE_COLORS[idx % COMPARE_COLORS.length]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
          <div className="mt-2 text-xs text-slate-500">
            Dir Acc shown as %. Params normalized to max in selection.
          </div>
        </div>
      )}

      {compareIds.length === 1 && (
        <div className="bg-slate-800 rounded-lg p-4 text-slate-400 text-sm">
          Select at least 2 experiments to compare
        </div>
      )}
    </div>
  );
}
