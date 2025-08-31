"""
Comprehensive Backward Pass Diagnostic Tools for NeuroGraph
Provides detailed monitoring and analysis of the discrete gradient computation system
"""

import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from datetime import datetime
import json
import os

class BackwardPassDiagnostics:
    """
    Comprehensive diagnostic system for NeuroGraph backward pass monitoring.
    
    Features:
    - Gradient flow tracking through all backward pass stages
    - Discrete update analysis and parameter change monitoring
    - Loss function decomposition and logit analysis
    - Performance timing and memory usage tracking
    - Training stability detection and alerting
    """
    
    def __init__(self, config: Dict[str, Any], device: str = 'cpu'):
        """
        Initialize backward pass diagnostics.
        
        Args:
            config: Configuration dictionary
            device: Device for tensor operations
        """
        self.config = config
        self.device = device
        
        # Diagnostic data storage
        self.gradient_stats = defaultdict(list)
        self.parameter_updates = defaultdict(list)
        self.loss_decomposition = defaultdict(list)
        self.timing_stats = defaultdict(list)
        self.convergence_metrics = defaultdict(list)
        
        # Real-time monitoring
        self.current_sample_idx = 0
        self.current_epoch = 0
        self.alert_thresholds = self._setup_alert_thresholds()
        
        # Performance tracking
        self.backward_pass_times = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        
        # Stability monitoring
        self.gradient_explosion_count = 0
        self.parameter_stagnation_count = 0
        self.unstable_updates_count = 0
        
        print(f"🔍 Backward Pass Diagnostics initialized")
        print(f"   📊 Device: {device}")
        print(f"   🚨 Alert system: {'enabled' if config.get('diagnostics.alerts_enabled', True) else 'disabled'}")
    
    def _setup_alert_thresholds(self) -> Dict[str, float]:
        """Setup alert thresholds for training stability monitoring."""
        return {
            'gradient_explosion_threshold': self.config.get('diagnostics.gradient_explosion_threshold', 10.0),
            'gradient_vanishing_threshold': self.config.get('diagnostics.gradient_vanishing_threshold', 1e-6),
            'parameter_stagnation_threshold': self.config.get('diagnostics.parameter_stagnation_threshold', 1e-8),
            'loss_spike_threshold': self.config.get('diagnostics.loss_spike_threshold', 2.0),
            'memory_usage_threshold': self.config.get('diagnostics.memory_usage_threshold', 1000.0)  # MB
        }
    
    def start_backward_pass_monitoring(self, sample_idx: int, epoch: int):
        """Start monitoring a backward pass."""
        self.current_sample_idx = sample_idx
        self.current_epoch = epoch
        self.backward_pass_start_time = time.perf_counter()
        
        # Clear temporary storage for this backward pass
        self.current_backward_pass_data = {
            'gradient_norms': {},
            'parameter_changes': {},
            'loss_components': {},
            'timing_breakdown': {},
            'alerts': []
        }
    
    def monitor_loss_computation(self, output_signals: Dict[int, torch.Tensor], 
                               target_label: int, loss: torch.Tensor, 
                               logits: torch.Tensor, class_encodings: Dict[int, Tuple[torch.Tensor, torch.Tensor]]):
        """
        Monitor loss computation stage of backward pass.
        
        Args:
            output_signals: Output node signals
            target_label: Ground truth label
            loss: Computed loss value
            logits: Class logits
            class_encodings: Class encoding vectors
        """
        loss_start_time = time.perf_counter()
        
        # Basic loss metrics
        loss_value = loss.item()
        self.loss_decomposition['total_loss'].append(loss_value)
        
        # Logit analysis
        logit_stats = self._analyze_logits(logits, target_label)
        for key, value in logit_stats.items():
            self.loss_decomposition[f'logit_{key}'].append(value)
        
        # Output signal analysis
        signal_stats = self._analyze_output_signals(output_signals, target_label, class_encodings)
        for key, value in signal_stats.items():
            self.loss_decomposition[f'signal_{key}'].append(value)
        
        # Confidence and prediction analysis
        confidence_stats = self._analyze_prediction_confidence(logits, target_label)
        for key, value in confidence_stats.items():
            self.loss_decomposition[f'confidence_{key}'].append(value)
        
        # Store timing
        loss_time = time.perf_counter() - loss_start_time
        self.current_backward_pass_data['timing_breakdown']['loss_computation'] = loss_time
        
        # Check for loss spikes
        if len(self.loss_decomposition['total_loss']) > 1:
            prev_loss = self.loss_decomposition['total_loss'][-2]
            if loss_value > prev_loss * self.alert_thresholds['loss_spike_threshold']:
                self._add_alert('loss_spike', f"Loss spiked from {prev_loss:.4f} to {loss_value:.4f}")
        
        print(f"      🔍 Loss Analysis: {loss_value:.4f} | Confidence: {confidence_stats['max_confidence']:.3f} | "
              f"Correct: {'✓' if confidence_stats['correct_prediction'] else '✗'}")
    
    def monitor_upstream_gradients(self, upstream_gradients: Dict[int, torch.Tensor]):
        """
        Monitor upstream gradient computation from loss function.
        
        Args:
            upstream_gradients: Gradients flowing back from loss
        """
        upstream_start_time = time.perf_counter()
        
        # Analyze upstream gradient statistics
        upstream_stats = self._analyze_gradient_dict(upstream_gradients, 'upstream')
        
        for key, value in upstream_stats.items():
            self.gradient_stats[f'upstream_{key}'].append(value)
        
        # Check for gradient explosion/vanishing
        max_grad_norm = upstream_stats.get('max_norm', 0.0)
        min_grad_norm = upstream_stats.get('min_norm', float('inf'))
        
        if max_grad_norm > self.alert_thresholds['gradient_explosion_threshold']:
            self.gradient_explosion_count += 1
            self._add_alert('gradient_explosion', f"Upstream gradient norm: {max_grad_norm:.4f}")
        
        if min_grad_norm < self.alert_thresholds['gradient_vanishing_threshold']:
            self._add_alert('gradient_vanishing', f"Upstream gradient norm: {min_grad_norm:.6f}")
        
        # Store timing
        upstream_time = time.perf_counter() - upstream_start_time
        self.current_backward_pass_data['timing_breakdown']['upstream_gradients'] = upstream_time
        
        print(f"      🔍 Upstream Gradients: {len(upstream_gradients)} nodes | "
              f"Max norm: {max_grad_norm:.4f} | Min norm: {min_grad_norm:.6f}")
    
    def monitor_discrete_gradient_computation(self, node_gradients: Dict[int, Tuple[torch.Tensor, torch.Tensor]]):
        """
        Monitor discrete gradient computation stage.
        
        Args:
            node_gradients: Computed discrete gradients (phase_grad, mag_grad)
        """
        discrete_start_time = time.perf_counter()
        
        # Separate phase and magnitude gradients
        phase_gradients = {node_id: grad_tuple[0] for node_id, grad_tuple in node_gradients.items()}
        mag_gradients = {node_id: grad_tuple[1] for node_id, grad_tuple in node_gradients.items()}
        
        # Analyze phase gradients
        phase_stats = self._analyze_gradient_dict(phase_gradients, 'discrete_phase')
        for key, value in phase_stats.items():
            self.gradient_stats[f'discrete_phase_{key}'].append(value)
        
        # Analyze magnitude gradients
        mag_stats = self._analyze_gradient_dict(mag_gradients, 'discrete_mag')
        for key, value in mag_stats.items():
            self.gradient_stats[f'discrete_mag_{key}'].append(value)
        
        # Analyze gradient correlation between phase and magnitude
        correlation_stats = self._analyze_phase_mag_correlation(phase_gradients, mag_gradients)
        for key, value in correlation_stats.items():
            self.gradient_stats[f'phase_mag_{key}'].append(value)
        
        # Store gradient norms for this backward pass
        self.current_backward_pass_data['gradient_norms']['discrete_phase'] = phase_stats.get('mean_norm', 0.0)
        self.current_backward_pass_data['gradient_norms']['discrete_mag'] = mag_stats.get('mean_norm', 0.0)
        
        # Store timing
        discrete_time = time.perf_counter() - discrete_start_time
        self.current_backward_pass_data['timing_breakdown']['discrete_gradients'] = discrete_time
        
        print(f"      🔍 Discrete Gradients: Phase norm: {phase_stats.get('mean_norm', 0.0):.4f} | "
              f"Mag norm: {mag_stats.get('mean_norm', 0.0):.4f} | "
              f"Correlation: {correlation_stats.get('mean_correlation', 0.0):.3f}")
    
    def monitor_parameter_updates(self, node_store, updated_nodes: List[int], 
                                learning_rate: float, before_params: Optional[Dict] = None):
        """
        Monitor actual parameter updates in node store.
        
        Args:
            node_store: NodeStore object
            updated_nodes: List of nodes that were updated
            learning_rate: Effective learning rate used
            before_params: Optional dict of parameters before update
        """
        update_start_time = time.perf_counter()
        
        # Track parameter changes
        parameter_changes = {}
        total_phase_change = 0.0
        total_mag_change = 0.0
        
        for node_id in updated_nodes:
            string_node_id = f"n{node_id}"
            
            if string_node_id in node_store.phase_table:
                current_phase = node_store.phase_table[string_node_id]
                current_mag = node_store.mag_table[string_node_id]
                
                if before_params and string_node_id in before_params:
                    prev_phase, prev_mag = before_params[string_node_id]
                    
                    # Calculate discrete parameter changes
                    # Get phase_bins and mag_bins from the lookup tables or config
                    phase_bins = getattr(node_store, 'phase_bins', 512)  # Default fallback updated
                    mag_bins = getattr(node_store, 'mag_bins', 1024)     # Default fallback updated
                    
                    phase_change = self._calculate_discrete_change(prev_phase, current_phase, phase_bins)
                    mag_change = self._calculate_discrete_change(prev_mag, current_mag, mag_bins)
                    
                    parameter_changes[node_id] = {
                        'phase_change': phase_change,
                        'mag_change': mag_change,
                        'total_change': phase_change + mag_change
                    }
                    
                    total_phase_change += abs(phase_change)
                    total_mag_change += abs(mag_change)
        
        # Store parameter update statistics
        if parameter_changes:
            avg_phase_change = total_phase_change / len(parameter_changes)
            avg_mag_change = total_mag_change / len(parameter_changes)
            
            self.parameter_updates['avg_phase_change'].append(avg_phase_change)
            self.parameter_updates['avg_mag_change'].append(avg_mag_change)
            self.parameter_updates['num_updated_nodes'].append(len(updated_nodes))
            self.parameter_updates['learning_rate'].append(learning_rate)
            
            # Check for parameter stagnation
            if avg_phase_change < self.alert_thresholds['parameter_stagnation_threshold'] and \
               avg_mag_change < self.alert_thresholds['parameter_stagnation_threshold']:
                self.parameter_stagnation_count += 1
                self._add_alert('parameter_stagnation', 
                              f"Very small parameter changes: phase={avg_phase_change:.8f}, mag={avg_mag_change:.8f}")
            
            # Store for current backward pass
            self.current_backward_pass_data['parameter_changes'] = {
                'avg_phase_change': avg_phase_change,
                'avg_mag_change': avg_mag_change,
                'num_updated': len(updated_nodes)
            }
        
        # Store timing
        update_time = time.perf_counter() - update_start_time
        self.current_backward_pass_data['timing_breakdown']['parameter_updates'] = update_time
        
        print(f"      🔍 Parameter Updates: {len(updated_nodes)} nodes | "
              f"Avg phase Δ: {avg_phase_change:.6f} | Avg mag Δ: {avg_mag_change:.6f}")
    
    def monitor_gradient_accumulation(self, gradient_accumulator, batch_controller):
        """
        Monitor gradient accumulation system performance.
        
        Args:
            gradient_accumulator: GradientAccumulator instance
            batch_controller: BatchController instance
        """
        if gradient_accumulator is None:
            return
        
        # Get accumulation statistics
        if hasattr(gradient_accumulator, 'get_statistics'):
            stats = gradient_accumulator.get_statistics()
            
            for key, value in stats.items():
                self.gradient_stats[f'accumulation_{key}'].append(value)
        
        # Monitor batch controller state
        if hasattr(batch_controller, 'get_state'):
            batch_state = batch_controller.get_state()
            
            for key, value in batch_state.items():
                self.gradient_stats[f'batch_{key}'].append(value)
        
        print(f"      🔍 Gradient Accumulation: Active")
    
    def finish_backward_pass_monitoring(self):
        """Finish monitoring current backward pass and compute summary statistics."""
        total_backward_time = time.perf_counter() - self.backward_pass_start_time
        self.backward_pass_times.append(total_backward_time)
        
        # Store total timing
        self.timing_stats['total_backward_pass'].append(total_backward_time)
        
        # Compute timing breakdown percentages
        timing_breakdown = self.current_backward_pass_data['timing_breakdown']
        if timing_breakdown:
            for component, time_spent in timing_breakdown.items():
                percentage = (time_spent / total_backward_time) * 100
                self.timing_stats[f'{component}_percentage'].append(percentage)
        
        # Update convergence metrics
        self._update_convergence_metrics()
        
        # Check memory usage
        self._monitor_memory_usage()
        
        # Print summary if alerts were generated
        if self.current_backward_pass_data['alerts']:
            print(f"      🚨 Alerts generated: {len(self.current_backward_pass_data['alerts'])}")
            for alert in self.current_backward_pass_data['alerts']:
                print(f"         - {alert['type']}: {alert['message']}")
        
        print(f"      ⏱️  Backward pass completed in {total_backward_time:.4f}s")
    
    def _analyze_logits(self, logits: torch.Tensor, target_label: int) -> Dict[str, float]:
        """Analyze logit distribution and quality."""
        with torch.no_grad():
            logits_np = logits.cpu().numpy()
            
            stats = {
                'max_logit': float(np.max(logits_np)),
                'min_logit': float(np.min(logits_np)),
                'mean_logit': float(np.mean(logits_np)),
                'std_logit': float(np.std(logits_np)),
                'target_logit': float(logits_np[target_label]),
                'logit_range': float(np.max(logits_np) - np.min(logits_np))
            }
            
            # Compute softmax probabilities
            probs = torch.softmax(logits, dim=0).cpu().numpy()
            stats['max_prob'] = float(np.max(probs))
            stats['target_prob'] = float(probs[target_label])
            stats['entropy'] = float(-np.sum(probs * np.log(probs + 1e-8)))
            
            return stats
    
    def _analyze_output_signals(self, output_signals: Dict[int, torch.Tensor], 
                              target_label: int, class_encodings: Dict[int, Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, float]:
        """Analyze output signal quality and alignment."""
        if not output_signals:
            return {'num_active_outputs': 0}
        
        stats = {'num_active_outputs': len(output_signals)}
        
        # Analyze signal strengths
        signal_norms = [torch.norm(signal).item() for signal in output_signals.values()]
        stats['mean_signal_strength'] = float(np.mean(signal_norms))
        stats['max_signal_strength'] = float(np.max(signal_norms))
        stats['min_signal_strength'] = float(np.min(signal_norms))
        
        # Analyze alignment with target class encoding if available
        if target_label in class_encodings:
            # This would require lookup_tables to compute target signal
            # For now, we'll skip this detailed analysis
            pass
        
        return stats
    
    def _analyze_prediction_confidence(self, logits: torch.Tensor, target_label: int) -> Dict[str, float]:
        """Analyze prediction confidence and correctness."""
        with torch.no_grad():
            probs = torch.softmax(logits, dim=0)
            predicted_class = torch.argmax(logits).item()
            max_confidence = torch.max(probs).item()
            target_confidence = probs[target_label].item()
            
            return {
                'predicted_class': float(predicted_class),
                'max_confidence': max_confidence,
                'target_confidence': target_confidence,
                'correct_prediction': float(predicted_class == target_label),
                'confidence_gap': max_confidence - target_confidence
            }
    
    def _analyze_gradient_dict(self, gradients: Dict[int, torch.Tensor], prefix: str) -> Dict[str, float]:
        """Analyze statistics of a dictionary of gradients."""
        if not gradients:
            return {f'{prefix}_count': 0}
        
        # Compute gradient norms
        grad_norms = []
        grad_means = []
        grad_stds = []
        
        for grad in gradients.values():
            if grad is not None:
                grad_norms.append(torch.norm(grad).item())
                grad_means.append(torch.mean(grad).item())
                grad_stds.append(torch.std(grad).item())
        
        if not grad_norms:
            return {f'{prefix}_count': 0}
        
        return {
            'count': len(grad_norms),
            'mean_norm': float(np.mean(grad_norms)),
            'max_norm': float(np.max(grad_norms)),
            'min_norm': float(np.min(grad_norms)),
            'std_norm': float(np.std(grad_norms)),
            'mean_mean': float(np.mean(grad_means)),
            'mean_std': float(np.mean(grad_stds))
        }
    
    def _analyze_phase_mag_correlation(self, phase_gradients: Dict[int, torch.Tensor], 
                                     mag_gradients: Dict[int, torch.Tensor]) -> Dict[str, float]:
        """Analyze correlation between phase and magnitude gradients."""
        correlations = []
        
        for node_id in phase_gradients:
            if node_id in mag_gradients:
                phase_grad = phase_gradients[node_id].flatten()
                mag_grad = mag_gradients[node_id].flatten()
                
                if len(phase_grad) > 1 and len(mag_grad) > 1:
                    # Compute correlation coefficient
                    correlation = torch.corrcoef(torch.stack([phase_grad, mag_grad]))[0, 1]
                    if not torch.isnan(correlation):
                        correlations.append(correlation.item())
        
        if correlations:
            return {
                'mean_correlation': float(np.mean(correlations)),
                'std_correlation': float(np.std(correlations)),
                'max_correlation': float(np.max(correlations)),
                'min_correlation': float(np.min(correlations))
            }
        else:
            return {'mean_correlation': 0.0}
    
    def _calculate_discrete_change(self, prev_indices: torch.Tensor, 
                                 current_indices: torch.Tensor, num_bins: int) -> float:
        """Calculate discrete parameter change with modular arithmetic."""
        # Handle modular arithmetic for discrete indices
        diff = current_indices - prev_indices
        
        # Handle wraparound
        diff = torch.where(diff > num_bins // 2, diff - num_bins, diff)
        diff = torch.where(diff < -num_bins // 2, diff + num_bins, diff)
        
        return torch.mean(torch.abs(diff.float())).item()
    
    def _update_convergence_metrics(self):
        """Update convergence monitoring metrics."""
        # Gradient norm trends
        if len(self.gradient_stats['discrete_phase_mean_norm']) > 1:
            recent_phase_norms = self.gradient_stats['discrete_phase_mean_norm'][-10:]
            phase_trend = np.polyfit(range(len(recent_phase_norms)), recent_phase_norms, 1)[0]
            self.convergence_metrics['phase_gradient_trend'].append(phase_trend)
        
        if len(self.gradient_stats['discrete_mag_mean_norm']) > 1:
            recent_mag_norms = self.gradient_stats['discrete_mag_mean_norm'][-10:]
            mag_trend = np.polyfit(range(len(recent_mag_norms)), recent_mag_norms, 1)[0]
            self.convergence_metrics['mag_gradient_trend'].append(mag_trend)
        
        # Loss trend
        if len(self.loss_decomposition['total_loss']) > 1:
            recent_losses = self.loss_decomposition['total_loss'][-10:]
            loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
            self.convergence_metrics['loss_trend'].append(loss_trend)
        
        # Parameter update stability
        if len(self.parameter_updates['avg_phase_change']) > 1:
            recent_changes = self.parameter_updates['avg_phase_change'][-10:]
            change_stability = np.std(recent_changes)
            self.convergence_metrics['parameter_stability'].append(change_stability)
    
    def _monitor_memory_usage(self):
        """Monitor memory usage during backward pass."""
        if torch.cuda.is_available() and self.device.startswith('cuda'):
            memory_mb = torch.cuda.memory_allocated(self.device) / (1024 * 1024)
            self.memory_usage.append(memory_mb)
            
            if memory_mb > self.alert_thresholds['memory_usage_threshold']:
                self._add_alert('high_memory_usage', f"Memory usage: {memory_mb:.1f} MB")
    
    def _add_alert(self, alert_type: str, message: str):
        """Add an alert to the current backward pass monitoring."""
        alert = {
            'type': alert_type,
            'message': message,
            'sample_idx': self.current_sample_idx,
            'epoch': self.current_epoch,
            'timestamp': datetime.now().isoformat()
        }
        
        self.current_backward_pass_data['alerts'].append(alert)
    
    def get_diagnostic_summary(self) -> Dict[str, Any]:
        """Get comprehensive diagnostic summary."""
        summary = {
            'gradient_statistics': self._summarize_gradient_stats(),
            'loss_analysis': self._summarize_loss_stats(),
            'parameter_updates': self._summarize_parameter_stats(),
            'timing_analysis': self._summarize_timing_stats(),
            'convergence_metrics': self._summarize_convergence_stats(),
            'stability_alerts': self._summarize_stability_alerts()
        }
        
        return summary
    
    def _summarize_gradient_stats(self) -> Dict[str, Any]:
        """Summarize gradient statistics."""
        summary = {}
        
        for key, values in self.gradient_stats.items():
            if values and len(values) > 0:
                try:
                    summary[key] = {
                        'count': len(values),
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'recent_trend': float(np.polyfit(range(len(values[-10:])), values[-10:], 1)[0]) if len(values) > 1 else 0.0
                    }
                except (ValueError, TypeError) as e:
                    # Handle empty arrays or invalid data
                    summary[key] = {
                        'count': len(values),
                        'mean': 0.0,
                        'std': 0.0,
                        'min': 0.0,
                        'max': 0.0,
                        'recent_trend': 0.0,
                        'error': str(e)
                    }
        
        return summary
    
    def _summarize_loss_stats(self) -> Dict[str, Any]:
        """Summarize loss analysis statistics."""
        summary = {}
        
        for key, values in self.loss_decomposition.items():
            if values:
                summary[key] = {
                    'count': len(values),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'latest': float(values[-1])
                }
        
        return summary
    
    def _summarize_parameter_stats(self) -> Dict[str, Any]:
        """Summarize parameter update statistics."""
        summary = {}
        
        for key, values in self.parameter_updates.items():
            if values:
                summary[key] = {
                    'count': len(values),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }
        
        return summary
    
    def _summarize_timing_stats(self) -> Dict[str, Any]:
        """Summarize timing statistics."""
        summary = {}
        
        for key, values in self.timing_stats.items():
            if values:
                summary[key] = {
                    'count': len(values),
                    'mean_ms': float(np.mean(values) * 1000),
                    'std_ms': float(np.std(values) * 1000),
                    'min_ms': float(np.min(values) * 1000),
                    'max_ms': float(np.max(values) * 1000),
                    'total_s': float(np.sum(values))
                }
        
        return summary
    
    def _summarize_convergence_stats(self) -> Dict[str, Any]:
        """Summarize convergence metrics."""
        summary = {}
        
        for key, values in self.convergence_metrics.items():
            if values:
                summary[key] = {
                    'count': len(values),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'latest': float(values[-1])
                }
        
        return summary
    
    def _summarize_stability_alerts(self) -> Dict[str, Any]:
        """Summarize stability alerts and warnings."""
        return {
            'gradient_explosion_count': self.gradient_explosion_count,
            'parameter_stagnation_count': self.parameter_stagnation_count,
            'unstable_updates_count': self.unstable_updates_count,
            'total_backward_passes': len(self.backward_pass_times),
            'average_backward_pass_time_ms': float(np.mean(self.backward_pass_times) * 1000) if self.backward_pass_times else 0.0
        }
    
    def print_diagnostic_report(self):
        """Print comprehensive diagnostic report."""
        print("\n" + "="*80)
        print("🔍 BACKWARD PASS DIAGNOSTIC REPORT")
        print("="*80)
        
        summary = self.get_diagnostic_summary()
        
        # Gradient Statistics
        print("\n📊 GRADIENT STATISTICS")
        print("-" * 40)
        grad_stats = summary['gradient_statistics']
        for key, stats in grad_stats.items():
            if 'mean_norm' in key:
                print(f"{key:30s}: {stats['mean']:.6f} ± {stats['std']:.6f} "
                      f"(range: {stats['min']:.6f} - {stats['max']:.6f})")
        
        # Loss Analysis
        print("\n📉 LOSS ANALYSIS")
        print("-" * 40)
        loss_stats = summary['loss_analysis']
        for key, stats in loss_stats.items():
            if key in ['total_loss', 'confidence_max_confidence', 'confidence_target_confidence']:
                print(f"{key:30s}: {stats['latest']:.4f} (avg: {stats['mean']:.4f})")
        
        # Parameter Updates
        print("\n🔧 PARAMETER UPDATES")
        print("-" * 40)
        param_stats = summary['parameter_updates']
        for key, stats in param_stats.items():
            print(f"{key:30s}: {stats['mean']:.6f} ± {stats['std']:.6f}")
        
        # Timing Analysis
        print("\n⏱️  TIMING ANALYSIS")
        print("-" * 40)
        timing_stats = summary['timing_analysis']
        for key, stats in timing_stats.items():
            if 'total' in key or 'percentage' not in key:
                print(f"{key:30s}: {stats['mean_ms']:.2f}ms ± {stats['std_ms']:.2f}ms")
        
        # Stability Alerts
        print("\n🚨 STABILITY ALERTS")
        print("-" * 40)
        stability = summary['stability_alerts']
        print(f"Gradient explosions: {stability['gradient_explosion_count']}")
        print(f"Parameter stagnations: {stability['parameter_stagnation_count']}")
        print(f"Unstable updates: {stability['unstable_updates_count']}")
        print(f"Total backward passes: {stability['total_backward_passes']}")
        print(f"Avg backward pass time: {stability['average_backward_pass_time_ms']:.2f}ms")
        
        print("\n" + "="*80)
    
    def save_diagnostic_data(self, filepath: str):
        """Save diagnostic data to file."""
        diagnostic_data = {
            'gradient_stats': dict(self.gradient_stats),
            'parameter_updates': dict(self.parameter_updates),
            'loss_decomposition': dict(self.loss_decomposition),
            'timing_stats': dict(self.timing_stats),
            'convergence_metrics': dict(self.convergence_metrics),
            'stability_counts': {
                'gradient_explosion_count': self.gradient_explosion_count,
                'parameter_stagnation_count': self.parameter_stagnation_count,
                'unstable_updates_count': self.unstable_updates_count
            },
            'config': self.config,
            'alert_thresholds': self.alert_thresholds,
            'timestamp': datetime.now().isoformat()
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(diagnostic_data, f, indent=2, default=str)
        
        print(f"💾 Diagnostic data saved to {filepath}")
    
    def load_diagnostic_data(self, filepath: str):
        """Load diagnostic data from file."""
        with open(filepath, 'r') as f:
            diagnostic_data = json.load(f)
        
        # Restore data structures
        self.gradient_stats = defaultdict(list, diagnostic_data['gradient_stats'])
        self.parameter_updates = defaultdict(list, diagnostic_data['parameter_updates'])
        self.loss_decomposition = defaultdict(list, diagnostic_data['loss_decomposition'])
        self.timing_stats = defaultdict(list, diagnostic_data['timing_stats'])
        self.convergence_metrics = defaultdict(list, diagnostic_data['convergence_metrics'])
        
        # Restore stability counts
        stability = diagnostic_data['stability_counts']
        self.gradient_explosion_count = stability['gradient_explosion_count']
        self.parameter_stagnation_count = stability['parameter_stagnation_count']
        self.unstable_updates_count = stability['unstable_updates_count']
        
        print(f"📂 Diagnostic data loaded from {filepath}")


class GradientFlowAnalyzer:
    """
    Specialized analyzer for gradient flow patterns in NeuroGraph.
    
    Focuses on understanding how gradients propagate through the discrete
    parameter space and identifying optimization bottlenecks.
    """
    
    def __init__(self, diagnostics: BackwardPassDiagnostics):
        """
        Initialize gradient flow analyzer.
        
        Args:
            diagnostics: BackwardPassDiagnostics instance
        """
        self.diagnostics = diagnostics
        self.flow_patterns = defaultdict(list)
        self.bottleneck_analysis = defaultdict(list)
        
    def analyze_gradient_flow_pattern(self, node_gradients: Dict[int, Tuple[torch.Tensor, torch.Tensor]]):
        """
        Analyze gradient flow patterns across nodes.
        
        Args:
            node_gradients: Node gradients (phase_grad, mag_grad)
        """
        if not node_gradients:
            return
        
        # Analyze gradient magnitude distribution
        phase_norms = []
        mag_norms = []
        
        for node_id, (phase_grad, mag_grad) in node_gradients.items():
            phase_norms.append(torch.norm(phase_grad).item())
            mag_norms.append(torch.norm(mag_grad).item())
        
        # Identify gradient flow patterns
        phase_pattern = self._classify_gradient_pattern(phase_norms)
        mag_pattern = self._classify_gradient_pattern(mag_norms)
        
        self.flow_patterns['phase_pattern'].append(phase_pattern)
        self.flow_patterns['mag_pattern'].append(mag_pattern)
        
        # Detect bottlenecks
        bottlenecks = self._detect_gradient_bottlenecks(node_gradients)
        self.bottleneck_analysis['bottleneck_nodes'].append(bottlenecks)
        
        print(f"      🔍 Gradient Flow: Phase={phase_pattern}, Mag={mag_pattern}, "
              f"Bottlenecks={len(bottlenecks)}")
    
    def _classify_gradient_pattern(self, gradient_norms: List[float]) -> str:
        """Classify gradient flow pattern based on norm distribution."""
        if not gradient_norms:
            return "empty"
        
        mean_norm = np.mean(gradient_norms)
        std_norm = np.std(gradient_norms)
        cv = std_norm / (mean_norm + 1e-8)  # Coefficient of variation
        
        if mean_norm < 1e-6:
            return "vanishing"
        elif mean_norm > 10.0:
            return "exploding"
        elif cv < 0.3:
            return "uniform"
        elif cv > 1.0:
            return "sparse"
        else:
            return "normal"
    
    def _detect_gradient_bottlenecks(self, node_gradients: Dict[int, Tuple[torch.Tensor, torch.Tensor]]) -> List[int]:
        """Detect nodes with unusually small gradients (potential bottlenecks)."""
        bottlenecks = []
        
        for node_id, (phase_grad, mag_grad) in node_gradients.items():
            phase_norm = torch.norm(phase_grad).item()
            mag_norm = torch.norm(mag_grad).item()
            
            # Consider a node a bottleneck if both gradients are very small
            if phase_norm < 1e-6 and mag_norm < 1e-6:
                bottlenecks.append(node_id)
        
        return bottlenecks
    
    def get_flow_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of gradient flow analysis."""
        summary = {}
        
        # Pattern frequency analysis
        for pattern_type, patterns in self.flow_patterns.items():
            if patterns:
                pattern_counts = {}
                for pattern in patterns:
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                
                summary[f'{pattern_type}_distribution'] = pattern_counts
                summary[f'{pattern_type}_most_common'] = max(pattern_counts, key=pattern_counts.get)
        
        # Bottleneck analysis
        if self.bottleneck_analysis['bottleneck_nodes']:
            all_bottlenecks = []
            for bottleneck_list in self.bottleneck_analysis['bottleneck_nodes']:
                all_bottlenecks.extend(bottleneck_list)
            
            if all_bottlenecks:
                bottleneck_counts = {}
                for node_id in all_bottlenecks:
                    bottleneck_counts[node_id] = bottleneck_counts.get(node_id, 0) + 1
                
                summary['frequent_bottlenecks'] = sorted(bottleneck_counts.items(), 
                                                       key=lambda x: x[1], reverse=True)[:5]
                summary['total_bottleneck_instances'] = len(all_bottlenecks)
        
        return summary


class DiscreteUpdateAnalyzer:
    """
    Analyzer for discrete parameter update effectiveness.
    
    Monitors how well continuous gradients translate to discrete parameter changes
    and identifies optimization inefficiencies.
    """
    
    def __init__(self, diagnostics: BackwardPassDiagnostics):
        """Initialize discrete update analyzer."""
        self.diagnostics = diagnostics
        self.update_effectiveness = defaultdict(list)
        self.quantization_analysis = defaultdict(list)
        
    def analyze_discrete_update_effectiveness(self, continuous_gradients: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
                                            discrete_updates: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
                                            learning_rate: float):
        """
        Analyze effectiveness of continuous-to-discrete gradient conversion.
        
        Args:
            continuous_gradients: Original continuous gradients
            discrete_updates: Applied discrete updates
            learning_rate: Learning rate used
        """
        if not continuous_gradients or not discrete_updates:
            return
        
        effectiveness_scores = []
        quantization_losses = []
        
        for node_id in continuous_gradients:
            if node_id in discrete_updates:
                cont_phase, cont_mag = continuous_gradients[node_id]
                disc_phase, disc_mag = discrete_updates[node_id]
                
                # Compute effectiveness score (how much of the gradient was preserved)
                phase_effectiveness = self._compute_update_effectiveness(cont_phase, disc_phase, learning_rate)
                mag_effectiveness = self._compute_update_effectiveness(cont_mag, disc_mag, learning_rate)
                
                effectiveness_scores.append((phase_effectiveness + mag_effectiveness) / 2)
                
                # Compute quantization loss
                phase_loss = self._compute_quantization_loss(cont_phase, disc_phase, learning_rate)
                mag_loss = self._compute_quantization_loss(cont_mag, disc_mag, learning_rate)
                
                quantization_losses.append((phase_loss + mag_loss) / 2)
        
        if effectiveness_scores:
            self.update_effectiveness['mean_effectiveness'].append(np.mean(effectiveness_scores))
            self.update_effectiveness['std_effectiveness'].append(np.std(effectiveness_scores))
            
        if quantization_losses:
            self.quantization_analysis['mean_quantization_loss'].append(np.mean(quantization_losses))
            self.quantization_analysis['std_quantization_loss'].append(np.std(quantization_losses))
        
        print(f"      🔍 Update Effectiveness: {np.mean(effectiveness_scores):.3f} ± {np.std(effectiveness_scores):.3f}")
    
    def _compute_update_effectiveness(self, continuous_grad: torch.Tensor, 
                                    discrete_update: torch.Tensor, learning_rate: float) -> float:
        """Compute how effectively continuous gradient was converted to discrete update."""
        # Calculate expected discrete changes based on gradient magnitude
        grad_norm = torch.norm(continuous_grad).item()
        expected_continuous_update = grad_norm * learning_rate
        
        # Calculate actual discrete changes (sum of absolute discrete updates)
        actual_discrete_changes = torch.sum(torch.abs(discrete_update)).item()
        
        # Effectiveness = ratio of actual discrete changes to expected magnitude
        # This measures how much of the gradient signal was captured in discrete updates
        if expected_continuous_update > 1e-8:
            # Scale by a reasonable factor to get percentage-like values
            # Assume typical discrete step size is ~0.01 (1% of parameter range)
            typical_step_size = 0.01
            expected_discrete_changes = expected_continuous_update / typical_step_size
            effectiveness = actual_discrete_changes / expected_discrete_changes
        else:
            effectiveness = 0.0
        
        # Return effectiveness as a ratio (can be > 1.0 if very effective)
        return max(0.0, effectiveness)
    
    def _compute_quantization_loss(self, continuous_grad: torch.Tensor, 
                                 discrete_update: torch.Tensor, learning_rate: float) -> float:
        """Compute information loss due to quantization."""
        expected_update = continuous_grad * learning_rate
        actual_update = discrete_update.float()
        
        # L2 loss between expected and actual updates
        loss = torch.nn.functional.mse_loss(expected_update, actual_update).item()
        
        return loss
    
    def get_update_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of discrete update analysis."""
        summary = {}
        
        # Effectiveness statistics
        for key, values in self.update_effectiveness.items():
            if values:
                summary[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'trend': float(np.polyfit(range(len(values)), values, 1)[0]) if len(values) > 1 else 0.0
                }
        
        # Quantization analysis
        for key, values in self.quantization_analysis.items():
            if values:
                summary[key] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'trend': float(np.polyfit(range(len(values)), values, 1)[0]) if len(values) > 1 else 0.0
                }
        
        return summary


def create_backward_pass_diagnostics(config: Dict[str, Any], device: str = 'cpu') -> BackwardPassDiagnostics:
    """
    Factory function to create backward pass diagnostics.
    
    Args:
        config: Configuration dictionary
        device: Device for tensor operations
        
    Returns:
        BackwardPassDiagnostics instance
    """
    return BackwardPassDiagnostics(config, device)


def create_gradient_flow_analyzer(diagnostics: BackwardPassDiagnostics) -> GradientFlowAnalyzer:
    """
    Factory function to create gradient flow analyzer.
    
    Args:
        diagnostics: BackwardPassDiagnostics instance
        
    Returns:
        GradientFlowAnalyzer instance
    """
    return GradientFlowAnalyzer(diagnostics)


def create_discrete_update_analyzer(diagnostics: BackwardPassDiagnostics) -> DiscreteUpdateAnalyzer:
    """
    Factory function to create discrete update analyzer.
    
    Args:
        diagnostics: BackwardPassDiagnostics instance
        
    Returns:
        DiscreteUpdateAnalyzer instance
    """
    return DiscreteUpdateAnalyzer(diagnostics)
