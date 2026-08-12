// Benchmark data — sim_step002_benchmarks.json
// Top-level is an object keyed by strategy name
export interface BenchmarkEntry {
  total_return_pct: number;
  cagr_pct: number;
  sharpe: number;
  sortino: number;
  max_drawdown_pct: number;
  win_rate_pct: number;
  n_trading_days: number;
  starting_capital: number;
  ending_nav: number;
  vs_benchmark: {
    bench_cagr_pct: number;
    bench_sharpe: number;
    alpha_cagr_pp: number;
    information_ratio: number;
  };
  history?: Array<{ date: string; nav: number }>;
}

export type BenchmarkData = Record<string, BenchmarkEntry>;

// Training history row — fields are optional because different models have different schemas
export interface HistoryRow {
  epoch: number;
  train_mse?: number;
  train_loss?: number;
  val_mse?: number;
  val_mae?: number;
  val_dir_acc?: number;
  val_sharpe?: number;
  val_loss?: number;
  elapsed_s?: number;
}

// GAN pretrain history row
export interface GanHistoryRow {
  epoch: number;
  d_loss: number | null;
  g_loss: number;
  recon_loss: number;
}

// GAN finetune history row
export interface GanFinetuneRow {
  epoch: number;
  val_mae?: number;
  val_dir_acc?: number;
  val_sharpe?: number;
}

// Common model result shape (superset — optional fields)
export interface ModelResult {
  step: string;
  model: string;
  // param counts — may be split for GAN
  n_params?: number;
  n_params_generator?: number;
  n_params_discriminator?: number;
  n_params_head?: number;
  n_stocks: number;
  // final metrics
  best_val_sharpe?: number;
  final_val_dir_acc?: number;
  final_val_mae?: number;
  // MAE encoder uses different keys
  val_directional_acc?: number;
  val_sharpe?: number;
  // histories
  history?: HistoryRow[];
  pretrain_history?: GanHistoryRow[];
  finetune_history?: GanFinetuneRow[];
  // extra
  notes?: string;
}
