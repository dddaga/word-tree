import { useState } from 'react';
import BenchmarkPage from './pages/BenchmarkPage';
import TrainingCurvesPage from './pages/TrainingCurvesPage';
import ModelComparisonPage from './pages/ModelComparisonPage';
import AboutPage from './pages/AboutPage';
import DataExplorerPage from './pages/DataExplorerPage';
import ExperimentLabPage from './pages/ExperimentLabPage';
import ResultsDBPage from './pages/ResultsDBPage';
import './index.css';

const TABS = [
  { id: 'benchmarks', label: 'Benchmarks' },
  { id: 'training', label: 'Training Curves' },
  { id: 'comparison', label: 'Model Comparison' },
  { id: 'data', label: 'Data Explorer' },
  { id: 'lab', label: 'Experiment Lab' },
  { id: 'results', label: 'Results DB' },
  { id: 'about', label: 'About' },
] as const;

type TabId = typeof TABS[number]['id'];

export default function App() {
  const [activeTab, setActiveTab] = useState<TabId>('benchmarks');

  return (
    <div className="min-h-screen bg-slate-900 text-slate-200">
      <header className="border-b border-slate-700 bg-slate-800">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <h1 className="text-xl font-semibold text-slate-100 mb-4">
            Financial ML Backtesting Dashboard
          </h1>
          <nav className="flex gap-1 flex-wrap">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
                  activeTab === tab.id
                    ? 'bg-blue-600 text-white'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6">
        {activeTab === 'benchmarks' && <BenchmarkPage />}
        {activeTab === 'training' && <TrainingCurvesPage />}
        {activeTab === 'comparison' && <ModelComparisonPage />}
        {activeTab === 'data' && <DataExplorerPage />}
        {activeTab === 'lab' && <ExperimentLabPage />}
        {activeTab === 'results' && <ResultsDBPage />}
        {activeTab === 'about' && <AboutPage />}
      </main>
    </div>
  );
}
