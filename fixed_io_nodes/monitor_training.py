#!/usr/bin/env python3
"""Monitor training progress and show convergence metrics."""

import csv
import time
import os
from collections import defaultdict
import statistics

def monitor_training(log_path='training_logs/autoregressive_log.csv', interval=10):
    """Monitor training progress from CSV log."""
    
    print("📊 Training Monitor Started")
    print(f"Monitoring: {log_path}")
    print(f"Update interval: {interval}s\n")
    print("="*80)
    
    last_size = 0
    
    while True:
        try:
            if not os.path.exists(log_path):
                print(f"Waiting for log file to be created...")
                time.sleep(interval)
                continue
            
            # Check if file has new data
            current_size = os.path.getsize(log_path)
            if current_size == last_size:
                time.sleep(interval)
                continue
            
            last_size = current_size
            
            # Read the CSV
            with open(log_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            if len(rows) == 0:
                print("No data yet...")
                time.sleep(interval)
                continue
            
            # Clear screen
            os.system('clear' if os.name != 'nt' else 'cls')
            
            print("="*80)
            print(f"📊 TRAINING PROGRESS MONITOR")
            print("="*80)
            
            # Parse data
            all_losses = [float(row['loss']) for row in rows]
            current_epoch = max(int(row['epoch']) for row in rows)
            
            # Overall stats
            print(f"\n📈 Overall Stats:")
            print(f"  Current Epoch: {current_epoch}")
            print(f"  Total Windows Processed: {len(rows)}")
            print(f"  Overall Avg Loss: {statistics.mean(all_losses):.4f}")
            print(f"  Overall Std Loss: {statistics.stdev(all_losses) if len(all_losses) > 1 else 0:.4f}")
            print(f"  Min Loss: {min(all_losses):.4f}")
            print(f"  Max Loss: {max(all_losses):.4f}")
            
            # Per-epoch stats
            print(f"\n📅 Per-Epoch Loss:")
            epoch_losses = defaultdict(list)
            for row in rows:
                epoch_losses[int(row['epoch'])].append(float(row['loss']))
            
            prev_avg = None
            for epoch in sorted(epoch_losses.keys()):
                losses = epoch_losses[epoch]
                avg_loss = statistics.mean(losses)
                std_loss = statistics.stdev(losses) if len(losses) > 1 else 0
                
                improvement = ""
                if prev_avg is not None:
                    delta = avg_loss - prev_avg
                    if delta < 0:
                        improvement = f"  (↓ {abs(delta):.4f} - IMPROVING ✅)"
                    else:
                        improvement = f"  (↑ {delta:.4f} - DEGRADING ⚠️)"
                
                print(f"  Epoch {epoch}: {avg_loss:.4f} ± {std_loss:.4f} ({len(losses)} windows){improvement}")
                prev_avg = avg_loss
            
            # Recent windows (last 20)
            print(f"\n📋 Recent Windows (last 20):")
            for row in rows[-20:]:
                print(f"  Epoch {int(row['epoch'])} | Seq {int(row['sequence_id'])} | Window {int(row['timestep'])}: loss={float(row['loss']):.4f}")
            
            print(f"\n{'='*80}")
            print(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}")
            
            time.sleep(interval)
            
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(interval)

if __name__ == "__main__":
    monitor_training()

