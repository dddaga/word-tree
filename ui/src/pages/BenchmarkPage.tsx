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
import type { BenchmarkData } from '../types';

const COLORS = ['#60a5fa', '#34d399', '#f472b6', '#fbbf24'];

export default function BenchmarkPage() {
  const [data, setData] = useState<BenchmarkData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch('/data/sim_step002_benchmarks.json')
      .then((r) => r.json())
      .then(setData)
      .catch((e) => setError(String(e)));
  }, []);

  if (error) return <div className="text-red-400 p-4">Failed to load: {error}</div>;
  if (!data) return <div className="text-slate-400 p-4">Loading...</div>;

  const strategies = Object.keys(data);

  // Bar chart data — CAGR, Sharpe, Max DD
  const cagrData = strategies.map((s) => ({
    name: s.replace('Strategy', ''),
    CAGR: data[s].cagr_pct,
    Sharpe: data[s].sharpe,
    MaxDD: data[s].max_drawdown_pct,
  }));

  // Alpha vs benchmark
  const alphaData = strategies.map((s) => ({
    name: s.replace('Strategy', ''),
    alpha: data[s].vs_benchmark.alpha_cagr_pp,
  }));

  return (
    <div className="space-y-8">
      <h2 className="text-lg font-semibold text-slate-100">Strategy Benchmarks</h2>

      {/* CAGR / Sharpe / MaxDD bar chart */}
      <div className="bg-slate-800 rounded-lg p-4">
        <h3 className="text-sm font-medium text-slate-300 mb-4">
          CAGR%, Sharpe &amp; Max Drawdown% by Strategy
        </h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={cagrData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} />
            <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', color: '#e2e8f0' }}
            />
            <Legend wrapperStyle={{ color: '#94a3b8' }} />
            <Bar dataKey="CAGR" fill="#60a5fa" />
            <Bar dataKey="Sharpe" fill="#34d399" />
            <Bar dataKey="MaxDD" fill="#f87171" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Alpha vs benchmark */}
      <div className="bg-slate-800 rounded-lg p-4">
        <h3 className="text-sm font-medium text-slate-300 mb-4">
          Alpha vs Benchmark (CAGR pp)
        </h3>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={alphaData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
            <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 12 }} />
            <YAxis tick={{ fill: '#94a3b8', fontSize: 12 }} />
            <Tooltip
              contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', color: '#e2e8f0' }}
            />
            <Bar dataKey="alpha" fill="#a78bfa" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Full tearsheet table */}
      <div className="bg-slate-800 rounded-lg p-4">
        <h3 className="text-sm font-medium text-slate-300 mb-4">Full Tearsheet</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-slate-400 border-b border-slate-700">
                <th className="text-left py-2 pr-4">Strategy</th>
                <th className="text-right py-2 pr-4">CAGR%</th>
                <th className="text-right py-2 pr-4">Sharpe</th>
                <th className="text-right py-2 pr-4">Sortino</th>
                <th className="text-right py-2 pr-4">Max DD%</th>
                <th className="text-right py-2 pr-4">Win Rate%</th>
                <th className="text-right py-2 pr-4">Alpha pp</th>
                <th className="text-right py-2">Total Return%</th>
              </tr>
            </thead>
            <tbody>
              {strategies.map((s, i) => {
                const d = data[s];
                return (
                  <tr
                    key={s}
                    className={`border-b border-slate-700/50 ${i % 2 === 0 ? 'bg-slate-800' : 'bg-slate-750'}`}
                  >
                    <td className="py-2 pr-4 text-slate-200 font-medium">{s.replace('Strategy', '')}</td>
                    <td className={`text-right py-2 pr-4 ${d.cagr_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {d.cagr_pct.toFixed(2)}
                    </td>
                    <td className={`text-right py-2 pr-4 ${d.sharpe >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {d.sharpe.toFixed(3)}
                    </td>
                    <td className={`text-right py-2 pr-4 ${d.sortino >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {d.sortino.toFixed(3)}
                    </td>
                    <td className="text-right py-2 pr-4 text-red-400">
                      {d.max_drawdown_pct.toFixed(2)}
                    </td>
                    <td className="text-right py-2 pr-4 text-slate-300">
                      {d.win_rate_pct.toFixed(1)}
                    </td>
                    <td className={`text-right py-2 pr-4 ${d.vs_benchmark.alpha_cagr_pp >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {d.vs_benchmark.alpha_cagr_pp.toFixed(2)}
                    </td>
                    <td className={`text-right py-2 ${d.total_return_pct >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      {d.total_return_pct.toFixed(2)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Ending NAV summary */}
      <div className="bg-slate-800 rounded-lg p-4">
        <h3 className="text-sm font-medium text-slate-300 mb-4">
          Ending NAV (starting: ₹1,000,000)
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {strategies.map((s, i) => {
            const d = data[s];
            const gain = d.ending_nav - d.starting_capital;
            return (
              <div key={s} className="bg-slate-700 rounded p-3">
                <div className="text-xs text-slate-400 mb-1">{s.replace('Strategy', '')}</div>
                <div className="text-lg font-mono font-semibold" style={{ color: COLORS[i % COLORS.length] }}>
                  ₹{(d.ending_nav / 1000).toFixed(1)}K
                </div>
                <div className={`text-xs mt-1 ${gain >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {gain >= 0 ? '+' : ''}{(gain / 1000).toFixed(1)}K
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
