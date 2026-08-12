import { useEffect, useRef, useState } from 'react';

const MODELS = [
  { id: 'ts_step010', label: 'CNN+LSTM (ts_step010)' },
  { id: 'ts_step011', label: 'CNN+Transformer (ts_step011)' },
  { id: 'ts_step020', label: 'MAE Encoder (ts_step020)' },
  { id: 'ts_step021', label: 'GAN Encoder (ts_step021)' },
  { id: 'ts_step030', label: 'SGNNET-TS (ts_step030)' },
];

interface Experiment {
  id: string;
  model: string;
  status: string;
  epochs: number;
  val_sharpe?: number;
  dir_acc?: number;
  created: string;
  elapsed?: string;
}

interface LogModalState {
  id: string;
  lines: string[];
  error: string | null;
}

export default function ExperimentLabPage() {
  // Launch form
  const [model, setModel] = useState(MODELS[0].id);
  const [device, setDevice] = useState('cpu');
  const [epochs, setEpochs] = useState(50);
  const [batchSize, setBatchSize] = useState(64);
  const [lr, setLr] = useState(0.001);
  const [seed, setSeed] = useState(42);
  const [launchMsg, setLaunchMsg] = useState<string | null>(null);
  const [launchError, setLaunchError] = useState<string | null>(null);

  // Active runs
  const [running, setRunning] = useState<Experiment[]>([]);

  // Recent experiments
  const [recent, setRecent] = useState<Experiment[]>([]);
  const [recentError, setRecentError] = useState<string | null>(null);

  // Log modal
  const [logModal, setLogModal] = useState<LogModalState | null>(null);
  const logIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  function fetchRunning() {
    fetch('/api/experiments?status=running')
      .then((r) => r.json())
      .then(setRunning)
      .catch(() => {});
  }

  function fetchRecent() {
    fetch('/api/experiments?limit=10')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setRecent)
      .catch((e) => setRecentError(String(e)));
  }

  useEffect(() => {
    fetchRunning();
    fetchRecent();
    const interval = setInterval(() => {
      fetchRunning();
      fetchRecent();
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  // Log modal polling
  useEffect(() => {
    if (!logModal) {
      if (logIntervalRef.current) clearInterval(logIntervalRef.current);
      return;
    }
    function fetchLog() {
      if (!logModal) return;
      fetch(`/api/experiments/${logModal.id}/log`)
        .then((r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.json();
        })
        .then((data) => {
          const lines: string[] = Array.isArray(data) ? data : (data.lines ?? []);
          setLogModal((prev) => prev ? { ...prev, lines: lines.slice(-100), error: null } : null);
        })
        .catch((e) => {
          setLogModal((prev) => prev ? { ...prev, error: String(e) } : null);
        });
    }
    fetchLog();
    logIntervalRef.current = setInterval(fetchLog, 3000);
    return () => {
      if (logIntervalRef.current) clearInterval(logIntervalRef.current);
    };
  }, [logModal?.id]);

  function handleRun() {
    setLaunchMsg(null);
    setLaunchError(null);
    fetch('/api/experiments/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, device, epochs, batch_size: batchSize, lr, seed }),
    })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data) => {
        setLaunchMsg(`Launched experiment ${data.id}`);
        fetchRunning();
        fetchRecent();
      })
      .catch((e) => setLaunchError(String(e)));
  }

  function handleDelete(id: string) {
    fetch(`/api/experiments/${id}`, { method: 'DELETE' })
      .then(() => fetchRecent())
      .catch(() => {});
  }

  function openLog(id: string) {
    setLogModal({ id, lines: [], error: null });
  }

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold text-slate-100">Experiment Lab</h2>

      {/* Launch form */}
      <div className="bg-slate-800 rounded-lg p-4">
        <h3 className="text-sm font-medium text-slate-300 mb-4">Launch Experiment</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-xs text-slate-400 mb-1">Model</label>
            <select
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="w-full bg-slate-700 border border-slate-600 rounded px-2 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-blue-500"
            >
              {MODELS.map((m) => (
                <option key={m.id} value={m.id}>{m.label}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">Device</label>
            <select
              value={device}
              onChange={(e) => setDevice(e.target.value)}
              className="w-full bg-slate-700 border border-slate-600 rounded px-2 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-blue-500"
            >
              <option value="cpu">cpu</option>
              <option value="mps">mps</option>
              <option value="cuda">cuda</option>
            </select>
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">Epochs</label>
            <input
              type="number"
              value={epochs}
              onChange={(e) => setEpochs(Number(e.target.value))}
              min={1}
              className="w-full bg-slate-700 border border-slate-600 rounded px-2 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-blue-500"
            />
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">Batch Size</label>
            <input
              type="number"
              value={batchSize}
              onChange={(e) => setBatchSize(Number(e.target.value))}
              min={1}
              className="w-full bg-slate-700 border border-slate-600 rounded px-2 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-blue-500"
            />
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">Learning Rate</label>
            <input
              type="number"
              value={lr}
              onChange={(e) => setLr(Number(e.target.value))}
              step={0.0001}
              min={0}
              className="w-full bg-slate-700 border border-slate-600 rounded px-2 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-blue-500"
            />
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">Seed</label>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
              className="w-full bg-slate-700 border border-slate-600 rounded px-2 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-blue-500"
            />
          </div>
        </div>
        <div className="mt-4 flex items-center gap-4">
          <button
            onClick={handleRun}
            className="px-5 py-2 bg-blue-600 hover:bg-blue-500 text-white text-sm rounded font-medium transition-colors"
          >
            Run Experiment
          </button>
          {launchMsg && <span className="text-green-400 text-sm">{launchMsg}</span>}
          {launchError && <span className="text-red-400 text-sm">Error: {launchError}</span>}
        </div>
      </div>

      {/* Active runs */}
      <div className="bg-slate-800 rounded-lg p-4">
        <h3 className="text-sm font-medium text-slate-300 mb-3">
          Active Runs
          <span className="ml-2 text-xs text-slate-500">(refreshes every 5s)</span>
        </h3>
        {running.length === 0 ? (
          <div className="text-slate-400 text-sm">No running experiments</div>
        ) : (
          <div className="space-y-2">
            {running.map((exp) => (
              <div key={exp.id} className="flex items-center justify-between bg-slate-700 rounded px-3 py-2">
                <div>
                  <span className="text-sm font-medium text-slate-200">
                    {MODELS.find((m) => m.id === exp.model)?.label ?? exp.model}
                  </span>
                  {exp.elapsed && (
                    <span className="ml-3 text-xs text-slate-400">Elapsed: {exp.elapsed}</span>
                  )}
                </div>
                <button
                  onClick={() => openLog(exp.id)}
                  className="px-3 py-1 bg-slate-600 hover:bg-slate-500 text-sm text-slate-200 rounded transition-colors"
                >
                  View Log
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Recent experiments */}
      <div className="bg-slate-800 rounded-lg p-4">
        <h3 className="text-sm font-medium text-slate-300 mb-3">Recent Experiments</h3>
        {recentError && <div className="text-red-400 text-sm">Error: {recentError}</div>}
        {!recentError && recent.length === 0 && (
          <div className="text-slate-400 text-sm">No experiments yet</div>
        )}
        {recent.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-slate-400 border-b border-slate-700">
                  <th className="text-left py-2 pr-4">Model</th>
                  <th className="text-left py-2 pr-4">Status</th>
                  <th className="text-right py-2 pr-4">Epochs</th>
                  <th className="text-right py-2 pr-4">Val Sharpe</th>
                  <th className="text-right py-2 pr-4">Dir Acc</th>
                  <th className="text-left py-2 pr-4">Created</th>
                  <th className="text-right py-2">Actions</th>
                </tr>
              </thead>
              <tbody>
                {recent.map((exp, i) => (
                  <tr
                    key={exp.id}
                    className={`border-b border-slate-700/50 ${i % 2 === 0 ? 'bg-slate-800' : 'bg-slate-750'}`}
                  >
                    <td className="py-2 pr-4 text-slate-200 font-medium">
                      {MODELS.find((m) => m.id === exp.model)?.label ?? exp.model}
                    </td>
                    <td className="py-2 pr-4">
                      <span className={`text-xs px-1.5 py-0.5 rounded ${
                        exp.status === 'running'
                          ? 'bg-blue-900/60 text-blue-300'
                          : exp.status === 'done'
                          ? 'bg-green-900/60 text-green-300'
                          : 'bg-slate-700 text-slate-400'
                      }`}>
                        {exp.status}
                      </span>
                    </td>
                    <td className="text-right py-2 pr-4 text-slate-300">{exp.epochs}</td>
                    <td className="text-right py-2 pr-4 text-slate-300">
                      {exp.val_sharpe != null ? exp.val_sharpe.toFixed(3) : '—'}
                    </td>
                    <td className="text-right py-2 pr-4 text-slate-300">
                      {exp.dir_acc != null ? `${(exp.dir_acc * 100).toFixed(1)}%` : '—'}
                    </td>
                    <td className="py-2 pr-4 text-slate-400 text-xs">{exp.created}</td>
                    <td className="text-right py-2">
                      <div className="flex justify-end gap-2">
                        <button
                          onClick={() => openLog(exp.id)}
                          className="px-2 py-0.5 bg-slate-600 hover:bg-slate-500 text-xs text-slate-200 rounded transition-colors"
                        >
                          Log
                        </button>
                        <button
                          onClick={() => handleDelete(exp.id)}
                          className="px-2 py-0.5 bg-red-900/60 hover:bg-red-800 text-xs text-red-300 rounded transition-colors"
                        >
                          Delete
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Log modal */}
      {logModal && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50 p-4">
          <div className="bg-slate-800 border border-slate-600 rounded-lg w-full max-w-3xl max-h-[80vh] flex flex-col">
            <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700">
              <span className="text-sm font-medium text-slate-200">
                Log — {logModal.id}
                <span className="ml-2 text-xs text-slate-500">(refreshes every 3s)</span>
              </span>
              <button
                onClick={() => setLogModal(null)}
                className="text-slate-400 hover:text-slate-200 text-lg leading-none"
              >
                ✕
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-4">
              {logModal.error && (
                <div className="text-red-400 text-sm mb-2">Error: {logModal.error}</div>
              )}
              <pre className="text-xs font-mono text-slate-300 whitespace-pre-wrap break-words">
                {logModal.lines.length > 0
                  ? logModal.lines.join('\n')
                  : 'Loading log...'}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
