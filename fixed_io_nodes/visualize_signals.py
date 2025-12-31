"""
Visualize the synthetic signal dataset to verify generation is working correctly.
"""
import matplotlib.pyplot as plt
import numpy as np
from synthetic_data import SyntheticSignalDataset, TemporalSignalDataset
import yaml


def load_config(path='configs/config_temporal.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def visualize_signals(config):
    """Visualize sample signals from each class."""
    
    # Create dataset
    dataset = SyntheticSignalDataset(
        num_classes=config['data']['num_classes'],
        seq_length=config['data']['seq_length'],
        num_samples_per_class=config['data']['num_samples_per_class'],
        noise_level=config['data']['noise_level'],
    )
    
    # Get samples
    samples = dataset.get_sample_signals(num_per_class=3)
    
    # Create plot
    num_classes = config['data']['num_classes']
    fig, axes = plt.subplots(num_classes, 1, figsize=(12, 2.5*num_classes))
    
    if num_classes == 1:
        axes = [axes]
    
    for class_idx in range(num_classes):
        ax = axes[class_idx]
        class_samples = samples[f'class_{class_idx}']
        
        for i, signal in enumerate(class_samples):
            t = np.arange(len(signal))
            ax.plot(t, signal, alpha=0.7, label=f'Sample {i+1}')
        
        ax.set_title(f'Class {class_idx} Signal Patterns', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Amplitude')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_logs/signal_visualization.png', dpi=150, bbox_inches='tight')
    print("✅ Visualization saved to training_logs/signal_visualization.png")
    plt.show()


def visualize_temporal_windows(config):
    """Visualize temporal windowing."""
    
    # Create datasets
    base_dataset = SyntheticSignalDataset(
        num_classes=config['data']['num_classes'],
        seq_length=config['data']['seq_length'],
        num_samples_per_class=5,  # Just a few for visualization
        noise_level=config['data']['noise_level'],
    )
    
    temporal_dataset = TemporalSignalDataset(
        base_dataset,
        input_window=config['data']['input_window'],
        predict_ahead=config['data']['predict_ahead']
    )
    
    # Get first signal and show how it's windowed
    full_signal = base_dataset.data[0].numpy()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    # Plot full signal
    axes[0].plot(full_signal, 'b-', linewidth=2)
    axes[0].set_title('Full Signal', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    # Plot example windows
    input_window = config['data']['input_window']
    for i in range(0, len(full_signal) - input_window, input_window):
        window_start = i
        window_end = i + input_window
        if window_end <= len(full_signal):
            axes[0].axvspan(window_start, window_end, alpha=0.2, 
                          color=['red', 'green', 'blue', 'yellow', 'purple'][i // input_window % 5])
    
    # Plot several windows overlaid
    axes[1].set_title(f'First 5 Temporal Windows (length={input_window})', 
                     fontsize=12, fontweight='bold')
    for i in range(min(5, len(temporal_dataset))):
        window = temporal_dataset.temporal_data[i].numpy()
        axes[1].plot(window, alpha=0.6, label=f'Window {i}')
    
    axes[1].set_xlabel('Window Position')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_logs/temporal_windows.png', dpi=150, bbox_inches='tight')
    print("✅ Temporal window visualization saved to training_logs/temporal_windows.png")
    plt.show()


if __name__ == "__main__":
    print("Loading config...")
    config = load_config()
    
    print("\nGenerating signal visualizations...")
    visualize_signals(config)
    
    print("\nGenerating temporal window visualizations...")
    visualize_temporal_windows(config)
    
    print("\n✨ Done! Check training_logs/ for visualization images.")

