#!/usr/bin/env python3
"""
NeuroGraph Redundant Files Cleanup Script
Moves identified redundant files to organized archive structure
"""

import os
import shutil
from datetime import datetime
from pathlib import Path

def ensure_dir(path):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)

def move_file_safely(src, dst_dir, category):
    """Move file safely with logging."""
    if os.path.exists(src):
        ensure_dir(dst_dir)
        dst = os.path.join(dst_dir, os.path.basename(src))
        try:
            shutil.move(src, dst)
            print(f"✅ Moved {src} → {dst}")
            return True
        except Exception as e:
            print(f"❌ Failed to move {src}: {e}")
            return False
    else:
        print(f"⚠️  File not found: {src}")
        return False

def main():
    """Main cleanup function."""
    print("🧹 NeuroGraph Redundant Files Cleanup")
    print("=" * 50)
    
    base_archive = "archive/cleanup_2025_01_28"
    moved_count = 0
    failed_count = 0
    
    # 1. Legacy Training Contexts (HIGH REDUNDANCY - 5 files)
    print("\n📁 Moving Legacy Training Contexts...")
    training_files = [
        "train/single_sample_train_context.py",
        "train/specialized_train_context.py", 
        "train/enhanced_train_context.py",
        "train/train_context.py",
        "train/train_context_1000.py"
    ]
    
    for file in training_files:
        if move_file_safely(file, f"{base_archive}/legacy_training", "training"):
            moved_count += 1
        else:
            failed_count += 1
    
    # 2. Legacy Main Files (HIGH REDUNDANCY - 3 files)
    print("\n📁 Moving Legacy Main Files...")
    main_files = [
        "main_fixed.py",
        "main_specialized.py",
        "main_1000.py"
    ]
    
    for file in main_files:
        if move_file_safely(file, f"{base_archive}/legacy_main", "main"):
            moved_count += 1
        else:
            failed_count += 1
    
    # 3. Legacy Configuration Files (HIGH REDUNDANCY - 6 files)
    print("\n📁 Moving Legacy Configuration Files...")
    config_files = [
        "config/optimized.yaml",
        "config/large_1000_node.yaml",
        "config/default.yaml",
        "config/fast_test.yaml",
        "config/large_graph.yaml",
        "config/production_training.yaml"
    ]
    
    for file in config_files:
        if move_file_safely(file, f"{base_archive}/legacy_config", "config"):
            moved_count += 1
        else:
            failed_count += 1
    
    # 4. Legacy Module Files (HIGH REDUNDANCY - 3 files)
    print("\n📁 Moving Legacy Module Files...")
    module_files = [
        "modules/input_adapters_1000.py",
        "modules/output_adapters_1000.py",
        "modules/specialized_output_adapters.py"
    ]
    
    for file in module_files:
        if move_file_safely(file, f"{base_archive}/legacy_modules", "modules"):
            moved_count += 1
        else:
            failed_count += 1
    
    # 5. Legacy Core Components (MEDIUM REDUNDANCY - 2 files)
    print("\n📁 Moving Legacy Core Components...")
    core_files = [
        "core/enhanced_forward_engine.py",
        "core/high_res_tables.py"
    ]
    
    for file in core_files:
        if move_file_safely(file, f"{base_archive}/legacy_core", "core"):
            moved_count += 1
        else:
            failed_count += 1
    
    # 6. Test and Analysis Files (MEDIUM REDUNDANCY - 5 files)
    print("\n📁 Moving Test and Analysis Files...")
    test_files = [
        "test_training_fix.py",
        "analyze_class_encodings.py",
        "test_modular_cleanup.py",
        "test_activation_tracker.py",
        "test_activation_solutions.py"
    ]
    
    for file in test_files:
        if move_file_safely(file, f"{base_archive}/test_analysis", "test"):
            moved_count += 1
        else:
            failed_count += 1
    
    # 7. Utility Files (LOW REDUNDANCY - 1 file)
    print("\n📁 Moving Utility Files...")
    util_files = [
        "utils/activation_balancer.py"
    ]
    
    for file in util_files:
        if move_file_safely(file, f"{base_archive}/test_analysis", "utils"):
            moved_count += 1
        else:
            failed_count += 1
    
    # 8. Documentation Files (MEDIUM REDUNDANCY - 5 files)
    print("\n📁 Moving Documentation Files...")
    doc_files = [
        "ACCURACY_INVESTIGATION_SUMMARY.md",
        "MIGRATION_SUMMARY.md", 
        "NEUROGRAPH_TECHNICAL_DOCUMENTATION.md",
        "NEUROGRAPH_PERFORMANCE_OPTIMIZATIONS.md",
        "README_ACTIVATION_IMPROVEMENTS.md"
    ]
    
    for file in doc_files:
        if move_file_safely(file, f"{base_archive}/documentation", "docs"):
            moved_count += 1
        else:
            failed_count += 1
    
    # 9. Training Scripts (HIGH REDUNDANCY - 1 file)
    print("\n📁 Moving Training Scripts...")
    script_files = [
        "train_fast_continuous_gradients.py"
    ]
    
    for file in script_files:
        if move_file_safely(file, f"{base_archive}/training_scripts", "scripts"):
            moved_count += 1
        else:
            failed_count += 1
    
    # Create cleanup summary
    print("\n📝 Creating Cleanup Summary...")
    summary_content = f"""# Cleanup Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Files Moved: {moved_count}
## Files Failed: {failed_count}
## Total Processed: {moved_count + failed_count}

## Archive Structure Created:
- archive/cleanup_2025_01_28/legacy_training/ - Legacy training contexts
- archive/cleanup_2025_01_28/legacy_main/ - Legacy main files  
- archive/cleanup_2025_01_28/legacy_config/ - Legacy configuration files
- archive/cleanup_2025_01_28/legacy_modules/ - Legacy module files
- archive/cleanup_2025_01_28/legacy_core/ - Legacy core components
- archive/cleanup_2025_01_28/test_analysis/ - Test and analysis files
- archive/cleanup_2025_01_28/documentation/ - Legacy documentation
- archive/cleanup_2025_01_28/training_scripts/ - Legacy training scripts

## Production System Preserved:
- main.py (primary entry point)
- main_production.py (production training)
- config/production.yaml (production config)
- train/modular_train_context.py (production training context)
- All core/vectorized_* files (GPU optimization)
- All memory-bank/ files (project documentation)

## Next Steps:
1. Verify production system functionality
2. Update README.md to remove phantom train.py reference
3. Update memory bank with cleanup results
"""
    
    with open(f"{base_archive}/CLEANUP_SUMMARY.md", "w") as f:
        f.write(summary_content)
    
    print(f"\n✅ Cleanup Complete!")
    print(f"   📊 Files moved: {moved_count}")
    print(f"   ❌ Files failed: {failed_count}")
    print(f"   📁 Archive location: {base_archive}/")
    print(f"   📝 Summary: {base_archive}/CLEANUP_SUMMARY.md")
    
    return moved_count, failed_count

if __name__ == "__main__":
    moved, failed = main()
    exit(0 if failed == 0 else 1)
