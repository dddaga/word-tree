#!/usr/bin/env python3
"""
Neurograph Training CLI

Command-line interface for training Neurograph networks with various
configurations and datasets.

Usage:
    # Run predefined config
    python train.py --config xor_basic
    
    # List available configs
    python train.py --list
    
    # Run all configs in a category
    python train.py --category pattern
    
    # Compare multiple configs
    python train.py --compare xor_basic xor_deep
    
    # Custom config file
    python train.py --custom my_config.yaml
    
    # Export results
    python train.py --config xor_basic --export results.json
"""

import argparse
import sys
import yaml
import json
import math
from pathlib import Path
from typing import List, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from training_framework import (
    NeurographTrainer,
    ExperimentConfig,
    DatasetConfig,
    TrainingResult,
    export_results,
    get_preset,
    list_presets,
    PRESET_EXPERIMENTS
)


def load_yaml_configs(path: str) -> Dict[str, Dict]:
    """Load configurations from YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def yaml_to_experiment(name: str, config: Dict) -> ExperimentConfig:
    """Convert YAML config to ExperimentConfig."""
    return ExperimentConfig(
        name=name,
        network=config.get('network', {}),
        training=config.get('training', {}),
        dataset=DatasetConfig.from_dict(config.get('dataset', {}))
    )


def list_yaml_configs(yaml_path: str):
    """List all configurations in YAML file."""
    configs = load_yaml_configs(yaml_path)
    
    print("\nAvailable configurations:")
    print("=" * 60)
    
    # Group by category (inferred from name prefix)
    categories = {}
    for name, config in configs.items():
        if name.startswith('#') or not isinstance(config, dict):
            continue
        
        desc = config.get('description', 'No description')
        category = name.split('_')[0] if '_' in name else 'other'
        
        if category not in categories:
            categories[category] = []
        categories[category].append((name, desc))
    
    for category, items in sorted(categories.items()):
        print(f"\n[{category}]")
        for name, desc in items:
            print(f"  {name:<25} - {desc}")
    
    print(f"\nTotal: {sum(len(items) for items in categories.values())} configurations")


def run_single_config(trainer: NeurographTrainer, 
                     config_name: str, 
                     yaml_path: str,
                     epochs: Optional[int] = None,
                     verbose: bool = True) -> TrainingResult:
    """Run training for a single configuration."""
    configs = load_yaml_configs(yaml_path)
    
    if config_name not in configs:
        # Try preset
        if config_name in PRESET_EXPERIMENTS:
            experiment = get_preset(config_name)
        else:
            available = list(configs.keys())
            print(f"Error: Unknown config '{config_name}'")
            print(f"Available: {available[:10]}..." if len(available) > 10 else f"Available: {available}")
            sys.exit(1)
    else:
        experiment = yaml_to_experiment(config_name, configs[config_name])
    
    return trainer.train(experiment, epochs=epochs, verbose=verbose)


def run_category(trainer: NeurographTrainer,
                category: str,
                yaml_path: str,
                epochs: Optional[int] = None,
                verbose: bool = True) -> List[TrainingResult]:
    """Run all configs matching a category prefix."""
    configs = load_yaml_configs(yaml_path)
    
    matching = [
        (name, config) 
        for name, config in configs.items()
        if name.startswith(category) and isinstance(config, dict) and 'network' in config
    ]
    
    if not matching:
        print(f"No configs found matching category '{category}'")
        sys.exit(1)
    
    print(f"\nRunning {len(matching)} configurations in category '{category}'")
    
    results = []
    for name, config in matching:
        experiment = yaml_to_experiment(name, config)
        result = trainer.train(experiment, epochs=epochs, verbose=verbose)
        results.append(result)
    
    return results


def compare_configs(trainer: NeurographTrainer,
                   config_names: List[str],
                   yaml_path: str,
                   epochs: int = 50,
                   verbose: bool = True) -> List[TrainingResult]:
    """Compare multiple configurations."""
    configs = load_yaml_configs(yaml_path)
    
    experiments = []
    for name in config_names:
        if name in configs:
            experiments.append(yaml_to_experiment(name, configs[name]))
        elif name in PRESET_EXPERIMENTS:
            experiments.append(get_preset(name))
        else:
            print(f"Warning: Unknown config '{name}', skipping")
    
    if not experiments:
        print("No valid configurations to compare")
        sys.exit(1)
    
    return trainer.compare_configs(experiments, epochs=epochs, verbose=verbose)


def run_all_presets(trainer: NeurographTrainer, epochs: int = 30):
    """Run all preset experiments."""
    print("\nRunning all preset experiments")
    print("=" * 60)
    
    results = []
    for name in list_presets():
        experiment = get_preset(name)
        result = trainer.train(experiment, epochs=epochs, verbose=True)
        results.append(result)
    
    return results


def print_summary(results: List[TrainingResult]):
    """Print summary table of results."""
    print("\n" + "=" * 70)
    print("Training Summary")
    print("=" * 70)
    print(f"{'Config':<25} {'Loss':<12} {'Epochs':<8} {'Time':<10} {'Converged':<10}")
    print("-" * 70)
    
    for r in results:
        converged = "Yes" if r.converged else "No"
        print(f"{r.config_name:<25} {r.final_loss:<12.6f} {r.total_epochs:<8} {r.training_time:>6.2f}s    {converged:<10}")
    
    # Stats
    avg_loss = sum(r.final_loss for r in results) / len(results)
    total_time = sum(r.training_time for r in results)
    converged_count = sum(1 for r in results if r.converged)
    
    print("-" * 70)
    print(f"{'Average':<25} {avg_loss:<12.6f} {'-':<8} {total_time:>6.2f}s    {converged_count}/{len(results)}")


def main():
    parser = argparse.ArgumentParser(
        description="Neurograph Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --list                      # List all configs
  python train.py --config xor_basic          # Run specific config
  python train.py --config xor_basic -e 200   # Run with 200 epochs
  python train.py --category temporal         # Run all temporal configs
  python train.py --compare xor_basic xor_deep
  python train.py --presets                   # Run all presets
  python train.py --config sine_simple --export results.json
        """
    )
    
    # Configuration selection
    parser.add_argument('--config', '-c', type=str, 
                       help='Configuration name to run')
    parser.add_argument('--category', type=str,
                       help='Run all configs in a category (prefix match)')
    parser.add_argument('--compare', nargs='+', 
                       help='Compare multiple configurations')
    parser.add_argument('--presets', action='store_true',
                       help='Run all preset experiments')
    parser.add_argument('--custom', type=str,
                       help='Path to custom YAML config file')
    
    # Options
    parser.add_argument('--epochs', '-e', type=int, default=None,
                       help='Override number of epochs')
    parser.add_argument('--list', '-l', action='store_true',
                       help='List available configurations')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Minimal output')
    
    # Output
    parser.add_argument('--export', type=str,
                       help='Export results to file (json or csv)')
    parser.add_argument('--format', choices=['json', 'csv'], default='json',
                       help='Export format (default: json)')
    
    # Server
    parser.add_argument('--url', type=str, default='http://localhost:8765',
                       help='Server URL (for HTTP mode)')
    parser.add_argument('--direct', action='store_true', default=True,
                       help='Use direct mode (no server, default)')
    parser.add_argument('--http', action='store_true',
                       help='Use HTTP mode (requires running server)')
    
    args = parser.parse_args()
    
    # Determine config file path
    yaml_path = args.custom or Path(__file__).parent / 'training_configs.yaml'
    
    # List configs
    if args.list:
        if args.custom:
            list_yaml_configs(args.custom)
        else:
            list_yaml_configs(yaml_path)
            print("\nPreset experiments (built-in):")
            for name in list_presets():
                exp = get_preset(name)
                print(f"  {name:<25} - {exp.name}")
        return 0
    
    # Create trainer
    use_direct = args.direct and not args.http
    trainer = NeurographTrainer(base_url=args.url, use_direct=use_direct)
    
    verbose = not args.quiet
    results = []
    
    # Run based on mode
    if args.presets:
        results = run_all_presets(trainer, epochs=args.epochs or 30)
    
    elif args.compare:
        results = compare_configs(
            trainer, args.compare, yaml_path,
            epochs=args.epochs or 50, verbose=verbose
        )
    
    elif args.category:
        results = run_category(
            trainer, args.category, yaml_path,
            epochs=args.epochs, verbose=verbose
        )
    
    elif args.config:
        result = run_single_config(
            trainer, args.config, yaml_path,
            epochs=args.epochs, verbose=verbose
        )
        results = [result]
    
    else:
        parser.print_help()
        print("\nError: Must specify --config, --category, --compare, or --presets")
        return 1
    
    # Print summary
    if len(results) > 1:
        print_summary(results)
    
    # Export results
    if args.export:
        export_results(results, args.export, format=args.format)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

