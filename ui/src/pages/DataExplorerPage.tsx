import { useEffect, useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

interface UniverseEntry {
  symbol: string;
  rows: number;
  start: string;
  end: string;
}

interface TickerRow {
  date: string;
  close: number;
  [key: string]: number | string;
}

const FEATURES = [
  'ret_1d', 'ma_ratio_5', 'ma_ratio_10', 'ma_ratio_20',
  'vol_5', 'vol_10', 'vol_20', 'rsi_14',
  'bb_pos', 'macd', 'macd_signal', 'obv_ratio',
  'ret_5d', 'ret_10d', 'ret_20d', 'atr_14',
];

function statsOf(values: number[]) {
  if (values.length === 0) return { mean: 0, std: 0, min: 0, max: 0 };
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const std = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length);
  return { mean, std, min: Math.min(...values), max: Math.max(...values) };
}

function oneYearAgo(): string {
  const d = new Date();
  d.setFullYear(d.getFullYear() - 1);
  return d.toISOString().slice(0, 10);
}

function today(): string {
  return new Date().toISOString().slice(0, 10);
}

export default function DataExplorerPage() {
  const [universe, setUniverse] = useState<UniverseEntry[]>([]);
  const [universeError, setUniverseError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  const [startDate, setStartDate] = useState(oneYearAgo());
  const [endDate, setEndDate] = useState(today());
  const [tickerData, setTickerData] = useState<TickerRow[]>([]);
  const [tickerLoading, setTickerLoading] = useState(false);
  const [tickerError, setTickerError] = useState<string | null>(null);
  const [feature, setFeature] = useState(FEATURES[0]);

  useEffect(() => {
    fetch('/api/universe')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setUniverse)
      .catch((e) => setUniverseError(String(e)));
  }, []);

  function loadTicker() {
    if (!selected) return;
    setTickerLoading(true);
    setTickerError(null);
    fetch(`/api/ticker/${selected}?start=${startDate}&end=${endDate}&features=true`)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data) => {
        setTickerData(data);
        setTickerLoading(false);
      })
      .catch((e) => {
        setTickerError(String(e));
        setTickerLoading(false);
      });
  }

  const featureValues = tickerData.map((row) => Number(row[feature])).filter((v) => !isNaN(v));
  const stats = statsOf(featureValues);

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold text-slate-100">Data Explorer</h2>

      <div className="flex gap-6">
        {/* Left panel — universe list */}
        <div className="w-80 flex-shrink-0">
          <div className="bg-slate-800 rounded-lg p-4">
            <h3 className="text-sm font-medium text-slate-300 mb-3">Universe</h3>
            {universeError && (
              <div className="text-red-400 text-sm">Error: {universeError}</div>
            )}
            {!universeError && universe.length === 0 && (
              <div className="text-slate-400 text-sm">Loading...</div>
            )}
            {universe.length > 0 && (
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="text-slate-400 border-b border-slate-700">
                      <th className="text-left py-1 pr-2">Symbol</th>
                      <th className="text-right py-1 pr-2">Rows</th>
                      <th className="text-right py-1 pr-2">Start</th>
                      <th className="text-right py-1">End</th>
                    </tr>
                  </thead>
                  <tbody>
                    {universe.map((u) => (
                      <tr
                        key={u.symbol}
                        onClick={() => setSelected(u.symbol)}
                        className={`border-b border-slate-700/50 cursor-pointer transition-colors ${
                          selected === u.symbol
                            ? 'bg-blue-900/40 text-blue-300'
                            : 'hover:bg-slate-700 text-slate-300'
                        }`}
                      >
                        <td className="py-1 pr-2 font-medium">{u.symbol}</td>
                        <td className="text-right py-1 pr-2">{u.rows.toLocaleString()}</td>
                        <td className="text-right py-1 pr-2">{u.start}</td>
                        <td className="text-right py-1">{u.end}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>

        {/* Right panel — ticker detail */}
        <div className="flex-1 space-y-4">
          {!selected ? (
            <div className="bg-slate-800 rounded-lg p-8 text-center text-slate-400 text-sm">
              Select a ticker from the universe list
            </div>
          ) : (
            <>
              {/* Controls */}
              <div className="bg-slate-800 rounded-lg p-4">
                <div className="flex items-end gap-4 flex-wrap">
                  <div>
                    <label className="block text-xs text-slate-400 mb-1">Symbol</label>
                    <div className="text-sm font-semibold text-blue-300">{selected}</div>
                  </div>
                  <div>
                    <label className="block text-xs text-slate-400 mb-1">Start</label>
                    <input
                      type="date"
                      value={startDate}
                      onChange={(e) => setStartDate(e.target.value)}
                      className="bg-slate-700 border border-slate-600 rounded px-2 py-1 text-sm text-slate-200 focus:outline-none focus:border-blue-500"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-slate-400 mb-1">End</label>
                    <input
                      type="date"
                      value={endDate}
                      onChange={(e) => setEndDate(e.target.value)}
                      className="bg-slate-700 border border-slate-600 rounded px-2 py-1 text-sm text-slate-200 focus:outline-none focus:border-blue-500"
                    />
                  </div>
                  <button
                    onClick={loadTicker}
                    disabled={tickerLoading}
                    className="px-4 py-1.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white text-sm rounded font-medium transition-colors"
                  >
                    {tickerLoading ? 'Loading...' : 'Load'}
                  </button>
                </div>
                {tickerError && (
                  <div className="mt-2 text-red-400 text-sm">Error: {tickerError}</div>
                )}
              </div>

              {tickerData.length > 0 && (
                <>
                  {/* Price chart */}
                  <div className="bg-slate-800 rounded-lg p-4">
                    <h3 className="text-sm font-medium text-slate-300 mb-3">
                      Close Price — {selected}
                    </h3>
                    <ResponsiveContainer width="100%" height={220}>
                      <LineChart data={tickerData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                        <XAxis
                          dataKey="date"
                          tick={{ fill: '#94a3b8', fontSize: 11 }}
                          interval="preserveStartEnd"
                        />
                        <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', color: '#e2e8f0' }}
                        />
                        <Line type="monotone" dataKey="close" stroke="#60a5fa" dot={false} strokeWidth={1.5} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Feature selector + chart */}
                  <div className="bg-slate-800 rounded-lg p-4">
                    <div className="flex items-center gap-4 mb-3">
                      <h3 className="text-sm font-medium text-slate-300">Feature</h3>
                      <select
                        value={feature}
                        onChange={(e) => setFeature(e.target.value)}
                        className="bg-slate-700 border border-slate-600 rounded px-2 py-1 text-sm text-slate-200 focus:outline-none focus:border-blue-500"
                      >
                        {FEATURES.map((f) => (
                          <option key={f} value={f}>{f}</option>
                        ))}
                      </select>
                    </div>
                    <ResponsiveContainer width="100%" height={200}>
                      <LineChart data={tickerData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                        <XAxis
                          dataKey="date"
                          tick={{ fill: '#94a3b8', fontSize: 11 }}
                          interval="preserveStartEnd"
                        />
                        <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
                        <Tooltip
                          contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', color: '#e2e8f0' }}
                        />
                        <Line type="monotone" dataKey={feature} stroke="#34d399" dot={false} strokeWidth={1.5} />
                      </LineChart>
                    </ResponsiveContainer>

                    {/* Stats card */}
                    <div className="mt-4 grid grid-cols-4 gap-3">
                      {[
                        { label: 'Mean', value: stats.mean.toFixed(4) },
                        { label: 'Std', value: stats.std.toFixed(4) },
                        { label: 'Min', value: stats.min.toFixed(4) },
                        { label: 'Max', value: stats.max.toFixed(4) },
                      ].map(({ label, value }) => (
                        <div key={label} className="bg-slate-700 rounded p-2 text-center">
                          <div className="text-xs text-slate-400">{label}</div>
                          <div className="text-sm font-mono text-slate-200 mt-0.5">{value}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
