# core/radiation.py

import torch
from typing import List, Set, Dict, Tuple
import time
from functools import lru_cache
from collections import defaultdict

# Enhanced caching system for radiation optimization
_static_neighbors_cache: Dict[str, Set[str]] = {}
_candidate_nodes_cache: Dict[str, List[str]] = {}
_radiation_pattern_cache: Dict[Tuple[str, int], List[str]] = {}

# Cache statistics for monitoring
_cache_stats = {
    'static_hits': 0,
    'static_misses': 0,
    'candidate_hits': 0,
    'candidate_misses': 0,
    'pattern_hits': 0,
    'pattern_misses': 0
}

def get_radiation_neighbors(
    current_node_id: str,
    ctx_phase_idx: torch.LongTensor,  # [D]
    node_store,
    graph_df,
    lookup_table,
    top_k: int,
    phase_bins: int,  # ADD: Phase bins parameter
    batch_size: int = 128  # Increased from 64 for better GPU utilization
) -> List[str]:
    """
    Selects Top-K radiation neighbors based on phase alignment.
    [OPTIMIZED] Enhanced with multi-level caching system.

    Args:
        current_node_id (str): ID of the active source node
        ctx_phase_idx (LongTensor): Phase index vector from activation table [D]
        node_store (NodeStore): Provides phase vectors
        graph_df (pd.DataFrame): Contains graph topology
        lookup_table (LookupTableModule): Provides cosine table
        top_k (int): Number of neighbors to select
        batch_size (int): Batch size for vectorized computation

    Returns:
        List of target node IDs (List[str])
    """
    global _cache_stats
    
    # Level 1: Check pattern cache (phase signature + top_k)
    phase_signature = _compute_phase_signature(ctx_phase_idx, phase_bins)
    pattern_key = (current_node_id, phase_signature, top_k)
    
    if pattern_key in _radiation_pattern_cache:
        _cache_stats['pattern_hits'] += 1
        return _radiation_pattern_cache[pattern_key].copy()
    
    _cache_stats['pattern_misses'] += 1
    
    # Level 2: Cache static neighbors to avoid repeated DataFrame lookups
    if current_node_id not in _static_neighbors_cache:
        static_neighbors = set(
            graph_df.loc[graph_df["node_id"] == current_node_id, "input_connections"].iloc[0]
        )
        _static_neighbors_cache[current_node_id] = static_neighbors
        _cache_stats['static_misses'] += 1
    else:
        static_neighbors = _static_neighbors_cache[current_node_id]
        _cache_stats['static_hits'] += 1
    
    # Level 3: Cache candidate nodes list
    if current_node_id not in _candidate_nodes_cache:
        phase_table = node_store.phase_table
        all_nodes = set(phase_table.keys())
        candidate_nodes = list(all_nodes - static_neighbors - {current_node_id})
        _candidate_nodes_cache[current_node_id] = candidate_nodes
        _cache_stats['candidate_misses'] += 1
    else:
        candidate_nodes = _candidate_nodes_cache[current_node_id]
        _cache_stats['candidate_hits'] += 1
    
    if not candidate_nodes:
        return []
    
    # Optimized vectorized computation
    selected_neighbors = _compute_radiation_neighbors_vectorized(
        current_node_id, ctx_phase_idx, candidate_nodes, node_store, 
        lookup_table, top_k, batch_size
    )
    
    # Cache the result for future use (with LRU eviction)
    _cache_radiation_pattern(pattern_key, selected_neighbors)
    
    return selected_neighbors

def _compute_phase_signature(ctx_phase_idx: torch.Tensor, phase_bins: int) -> int:
    """
    Compute a compact signature for phase indices to enable pattern caching.
    
    Args:
        ctx_phase_idx: Phase index tensor [D]
        phase_bins: Number of phase bins for adaptive quantization
        
    Returns:
        Integer signature for caching
    """
    # Use hash of phase indices as signature (quantized to reduce cache size)
    # Adaptive quantization based on actual phase_bins
    # For 512 bins: quantize by 4 gives 128 effective signatures
    # For 64 bins: quantize by 4 gives 16 effective signatures
    quantization_factor = max(1, phase_bins // 128)  # Adaptive quantization
    quantized_phases = (ctx_phase_idx // quantization_factor) * quantization_factor
    return hash(tuple(quantized_phases.cpu().numpy()))

def _compute_radiation_neighbors_vectorized(
    current_node_id: str,
    ctx_phase_idx: torch.Tensor,
    candidate_nodes: List[str],
    node_store,
    lookup_table,
    top_k: int,
    batch_size: int
) -> List[str]:
    """
    Vectorized computation of radiation neighbors with optimizations.
    
    Args:
        current_node_id: Source node ID
        ctx_phase_idx: Context phase indices [D]
        candidate_nodes: Pre-filtered candidate nodes
        node_store: Node parameter storage
        lookup_table: Lookup table for phase computation
        top_k: Number of neighbors to select
        batch_size: Batch size for processing
        
    Returns:
        List of selected neighbor node IDs
    """
    N = lookup_table.N
    ctx_phase = ctx_phase_idx.long()
    device = ctx_phase.device
    
    # Pre-allocate scores tensor for efficiency
    scores = torch.zeros(len(candidate_nodes), device=device)
    
    # Vectorized computation with memory optimization
    with torch.no_grad():
        # Process candidates in batches to manage memory
        effective_batch_size = min(batch_size, len(candidate_nodes))
        
        for i in range(0, len(candidate_nodes), effective_batch_size):
            batch_end = min(i + effective_batch_size, len(candidate_nodes))
            batch_candidates = candidate_nodes[i:batch_end]
            
            if batch_candidates:
                # Vectorized batch processing
                batch_scores = _compute_batch_alignment_scores(
                    ctx_phase, batch_candidates, node_store, lookup_table, N, device
                )
                
                # Store batch results
                scores[i:i + len(batch_candidates)] = batch_scores
    
    # Efficient top-k selection
    return _select_top_k_neighbors(scores, candidate_nodes, top_k)

def _compute_batch_alignment_scores(
    ctx_phase: torch.Tensor,
    batch_candidates: List[str],
    node_store,
    lookup_table,
    N: int,
    device: torch.device
) -> torch.Tensor:
    """
    Compute alignment scores for a batch of candidate nodes.
    
    Args:
        ctx_phase: Context phase tensor [D]
        batch_candidates: List of candidate node IDs
        node_store: Node parameter storage
        lookup_table: Lookup table for phase computation
        N: Number of phase bins
        device: Computation device
        
    Returns:
        Batch alignment scores [batch_size]
    """
    # Stack candidate phase vectors [batch_size, D]
    candidate_phases = torch.stack([
        node_store.get_phase(candidate_id).long().to(device) 
        for candidate_id in batch_candidates
    ])
    
    # Broadcast context phase [batch_size, D]
    ctx_phase_expanded = ctx_phase.unsqueeze(0).expand_as(candidate_phases)
    
    # Vectorized phase alignment computation [batch_size, D]
    phase_sums = (ctx_phase_expanded + candidate_phases) % N
    
    # Compute alignment scores [batch_size]
    batch_scores = lookup_table.lookup_phase(phase_sums).sum(dim=1)
    
    return batch_scores

def _select_top_k_neighbors(
    scores: torch.Tensor,
    candidate_nodes: List[str],
    top_k: int
) -> List[str]:
    """
    Select top-k neighbors based on alignment scores.
    
    Args:
        scores: Alignment scores for all candidates [N]
        candidate_nodes: List of candidate node IDs
        top_k: Number of neighbors to select
        
    Returns:
        List of selected neighbor node IDs
    """
    if len(scores) > top_k:
        # Use efficient top-k selection
        top_k_indices = torch.topk(scores, k=top_k, largest=True)[1]
        return [candidate_nodes[idx] for idx in top_k_indices.cpu().numpy()]
    else:
        # Sort all if fewer candidates than top_k
        indices = torch.argsort(scores, descending=True)
        return [candidate_nodes[idx] for idx in indices.cpu().numpy()]

def _cache_radiation_pattern(pattern_key: Tuple, neighbors: List[str]):
    """
    Cache radiation pattern with LRU eviction policy.
    
    Args:
        pattern_key: Cache key (node_id, phase_signature, top_k)
        neighbors: Selected neighbors to cache
    """
    global _radiation_pattern_cache
    
    # Simple LRU eviction: remove oldest entries if cache is too large
    MAX_PATTERN_CACHE_SIZE = 1000
    
    if len(_radiation_pattern_cache) >= MAX_PATTERN_CACHE_SIZE:
        # Remove 20% of oldest entries
        keys_to_remove = list(_radiation_pattern_cache.keys())[:MAX_PATTERN_CACHE_SIZE // 5]
        for key in keys_to_remove:
            del _radiation_pattern_cache[key]
    
    # Cache the new pattern
    _radiation_pattern_cache[pattern_key] = neighbors.copy()

def clear_radiation_cache():
    """Clear all radiation caches. Useful for testing or graph changes."""
    global _static_neighbors_cache, _candidate_nodes_cache, _radiation_pattern_cache, _cache_stats
    
    _static_neighbors_cache.clear()
    _candidate_nodes_cache.clear()
    _radiation_pattern_cache.clear()
    
    # Reset cache statistics
    _cache_stats = {
        'static_hits': 0,
        'static_misses': 0,
        'candidate_hits': 0,
        'candidate_misses': 0,
        'pattern_hits': 0,
        'pattern_misses': 0
    }

def get_cache_stats() -> Dict:
    """
    Get comprehensive statistics about the radiation cache system.
    
    Returns:
        Dictionary with detailed cache statistics
    """
    global _cache_stats
    
    # Calculate cache sizes
    static_cache_size = len(_static_neighbors_cache)
    candidate_cache_size = len(_candidate_nodes_cache)
    pattern_cache_size = len(_radiation_pattern_cache)
    
    # Calculate memory usage estimates
    static_memory = sum(len(neighbors) * 8 for neighbors in _static_neighbors_cache.values())
    candidate_memory = sum(len(candidates) * 8 for candidates in _candidate_nodes_cache.values())
    pattern_memory = sum(len(pattern) * 8 for pattern in _radiation_pattern_cache.values())
    total_memory = static_memory + candidate_memory + pattern_memory
    
    # Calculate hit rates
    total_static = _cache_stats['static_hits'] + _cache_stats['static_misses']
    total_candidate = _cache_stats['candidate_hits'] + _cache_stats['candidate_misses']
    total_pattern = _cache_stats['pattern_hits'] + _cache_stats['pattern_misses']
    
    static_hit_rate = _cache_stats['static_hits'] / max(total_static, 1)
    candidate_hit_rate = _cache_stats['candidate_hits'] / max(total_candidate, 1)
    pattern_hit_rate = _cache_stats['pattern_hits'] / max(total_pattern, 1)
    
    return {
        # Cache sizes
        'static_cache_entries': static_cache_size,
        'candidate_cache_entries': candidate_cache_size,
        'pattern_cache_entries': pattern_cache_size,
        'total_cache_entries': static_cache_size + candidate_cache_size + pattern_cache_size,
        
        # Memory usage (bytes)
        'static_memory_bytes': static_memory,
        'candidate_memory_bytes': candidate_memory,
        'pattern_memory_bytes': pattern_memory,
        'total_memory_bytes': total_memory,
        'total_memory_mb': total_memory / (1024 * 1024),
        
        # Hit/miss statistics
        'static_hits': _cache_stats['static_hits'],
        'static_misses': _cache_stats['static_misses'],
        'static_hit_rate': static_hit_rate,
        
        'candidate_hits': _cache_stats['candidate_hits'],
        'candidate_misses': _cache_stats['candidate_misses'],
        'candidate_hit_rate': candidate_hit_rate,
        
        'pattern_hits': _cache_stats['pattern_hits'],
        'pattern_misses': _cache_stats['pattern_misses'],
        'pattern_hit_rate': pattern_hit_rate,
        
        # Overall performance
        'total_requests': total_static + total_candidate + total_pattern,
        'overall_hit_rate': (
            (_cache_stats['static_hits'] + _cache_stats['candidate_hits'] + _cache_stats['pattern_hits']) /
            max(total_static + total_candidate + total_pattern, 1)
        )
    }

def print_cache_performance():
    """Print a formatted cache performance report."""
    stats = get_cache_stats()
    
    print("\n🚀 Radiation Cache Performance Report")
    print("=" * 50)
    print(f"📊 Cache Entries:")
    print(f"   Static neighbors: {stats['static_cache_entries']:,}")
    print(f"   Candidate nodes: {stats['candidate_cache_entries']:,}")
    print(f"   Radiation patterns: {stats['pattern_cache_entries']:,}")
    print(f"   Total entries: {stats['total_cache_entries']:,}")
    
    print(f"\n💾 Memory Usage:")
    print(f"   Total memory: {stats['total_memory_mb']:.2f} MB")
    print(f"   Static cache: {stats['static_memory_bytes'] / 1024:.1f} KB")
    print(f"   Candidate cache: {stats['candidate_memory_bytes'] / 1024:.1f} KB")
    print(f"   Pattern cache: {stats['pattern_memory_bytes'] / 1024:.1f} KB")
    
    print(f"\n📈 Hit Rates:")
    print(f"   Static neighbors: {stats['static_hit_rate']:.1%} ({stats['static_hits']}/{stats['static_hits'] + stats['static_misses']})")
    print(f"   Candidate nodes: {stats['candidate_hit_rate']:.1%} ({stats['candidate_hits']}/{stats['candidate_hits'] + stats['candidate_misses']})")
    print(f"   Radiation patterns: {stats['pattern_hit_rate']:.1%} ({stats['pattern_hits']}/{stats['pattern_hits'] + stats['pattern_misses']})")
    print(f"   Overall: {stats['overall_hit_rate']:.1%}")
    
    print(f"\n⚡ Performance Impact:")
    if stats['overall_hit_rate'] > 0.7:
        print("   🟢 Excellent cache performance (>70% hit rate)")
    elif stats['overall_hit_rate'] > 0.5:
        print("   🟡 Good cache performance (>50% hit rate)")
    else:
        print("   🔴 Poor cache performance (<50% hit rate)")
    
    estimated_speedup = 1 + (stats['overall_hit_rate'] * 2)  # Rough estimate
    print(f"   Estimated speedup: {estimated_speedup:.1f}x")

def optimize_cache_settings():
    """
    Analyze cache performance and suggest optimizations.
    
    Returns:
        Dictionary with optimization suggestions
    """
    stats = get_cache_stats()
    suggestions = []
    
    # Check pattern cache effectiveness
    if stats['pattern_hit_rate'] < 0.3:
        suggestions.append("Pattern cache hit rate is low. Consider increasing quantization in _compute_phase_signature()")
    
    # Check memory usage
    if stats['total_memory_mb'] > 100:
        suggestions.append("Cache memory usage is high. Consider reducing MAX_PATTERN_CACHE_SIZE")
    
    # Check cache balance
    if stats['pattern_cache_entries'] > 800:
        suggestions.append("Pattern cache is near capacity. LRU eviction may be too frequent")
    
    return {
        'suggestions': suggestions,
        'current_performance': 'excellent' if stats['overall_hit_rate'] > 0.7 else 
                             'good' if stats['overall_hit_rate'] > 0.5 else 'poor',
        'estimated_speedup': 1 + (stats['overall_hit_rate'] * 2)
    }
