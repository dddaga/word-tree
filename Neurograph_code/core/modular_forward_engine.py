"""
Vectorized Forward Engine for GPU-First NeuroGraph
Complete GPU-optimized forward pass with parallel processing
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
import time
from collections import deque

from core.activation_table import VectorizedActivationTable
from core.vectorized_propagation import VectorizedPropagationEngine
from core.node_store import NodeStore
from core.modular_cell import ModularPhaseCell


def bfs_to_outputs(adjacency, start_node, output_nodes):
    """
    BFS to find shortest path to ANY output and track which outputs are reachable.
    Returns (min_hops, set_of_reachable_outputs)
    """
    if start_node in output_nodes:
        return 0, {start_node}
    
    queue = deque([(start_node, 0)])
    visited = {start_node}
    reachable_outputs = set()
    min_hops = float('inf')
    
    while queue:
        current_node, distance = queue.popleft()
        
        # Check all neighbors
        for neighbor in adjacency.get(current_node, []):
            if neighbor in output_nodes:
                reachable_outputs.add(neighbor)
                min_hops = min(min_hops, distance + 1)
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    
    return min_hops, reachable_outputs


def validate_input_output_paths(adjacency, input_nodes, output_nodes):
    """
    Enhanced connectivity validation with average hops and detailed diagnostics.
    """
    print(f"🔍 Analyzing graph connectivity:")
    print(f"   📊 Input nodes: {len(input_nodes)}")
    print(f"   📊 Output nodes: {len(output_nodes)}")
    
    connected_inputs = 0
    reachable_outputs = set()
    min_hops = float('inf')
    all_hop_counts = []
    connectivity_details = []
    direct_connections = []
    
    # Test each input node
    for input_node in input_nodes:
        hops, reached_outputs = bfs_to_outputs(adjacency, input_node, output_nodes)
        
        if hops < float('inf'):
            connected_inputs += 1
            reachable_outputs.update(reached_outputs)
            min_hops = min(min_hops, hops)
            all_hop_counts.append(hops)
            
            # Track direct connections (suspicious)
            if hops == 1:
                for output in reached_outputs:
                    direct_connections.append((input_node, output))
            
            connectivity_details.append({
                'input': input_node,
                'min_hops': hops,
                'reachable_outputs': len(reached_outputs)
            })
    
    # Calculate statistics
    input_connectivity_rate = connected_inputs / len(input_nodes) if input_nodes else 0
    output_reachability_rate = len(reachable_outputs) / len(output_nodes) if output_nodes else 0
    
    # Calculate average hops
    avg_hops = sum(all_hop_counts) / len(all_hop_counts) if all_hop_counts else 8.0
    
    print(f"   ✅ Connected inputs: {connected_inputs}/{len(input_nodes)} ({input_connectivity_rate:.1%})")
    print(f"   ✅ Reachable outputs: {len(reachable_outputs)}/{len(output_nodes)} ({output_reachability_rate:.1%})")
    
    # Path analysis
    if all_hop_counts:
        from collections import Counter
        hop_distribution = Counter(all_hop_counts)
        print(f"   📊 Path Analysis:")
        print(f"      • Total valid paths: {len(all_hop_counts)}")
        print(f"      • Average hops: {avg_hops:.2f}")
        print(f"      • Min hops: {min_hops}")
        print(f"      • Max hops: {max(all_hop_counts)}")
        print(f"      • Hop distribution: {dict(hop_distribution)}")
    
    # Check for suspicious direct connections
    if direct_connections:
        print(f"   🔗 Direct input→output connections: {len(direct_connections)}")
        print(f"      Examples: {direct_connections[:5]}")
        if len(direct_connections) > 50:
            print(f"   ⚠️  WARNING: Unusually high number of direct connections!")
    
    # Warnings for connectivity issues
    if input_connectivity_rate < 0.8:
        print(f"   ⚠️  WARNING: Only {input_connectivity_rate:.1%} of input nodes can reach outputs")
        print(f"   💡 This may indicate graph connectivity issues")
    
    if output_reachability_rate < 0.8:
        print(f"   ⚠️  WARNING: Only {output_reachability_rate:.1%} of output nodes are reachable")
        print(f"   💡 Some outputs may never activate through static paths")
    
    if min_hops == float('inf'):
        print(f"   🚨 CRITICAL: No paths found from inputs to outputs!")
        print(f"   💡 Using fallback minimum timesteps (10)")
        min_hops = 8  # fallback - 2 = 10 total timesteps
        avg_hops = 8.0
    else:
        print(f"   🔗 Minimum input→output hops: {min_hops}")
        print(f"   📈 Average input→output hops: {avg_hops:.2f}")
    
    return {
        'min_hops': min_hops,
        'avg_hops': avg_hops,
        'connected_inputs': connected_inputs,
        'input_connectivity_rate': input_connectivity_rate,
        'reachable_outputs': len(reachable_outputs),
        'output_reachability_rate': output_reachability_rate,
        'connectivity_details': connectivity_details,
        'direct_connections': direct_connections,
        'hop_distribution': dict(Counter(all_hop_counts)) if all_hop_counts else {}
    }


def analyze_graph_connectivity(graph_df):
    """
    Comprehensive graph analysis with validation and diagnostics.
    Returns connectivity metrics and minimum timesteps.
    """
    # Build graph structures
    adjacency = {}  # forward adjacency: node -> [targets]
    reverse_adjacency = {}  # reverse: node -> [sources]
    input_nodes = set()
    output_nodes = set()
    
    for _, row in graph_df.iterrows():
        node_id = row['node_id']
        connections = row['input_connections']
        
        if row.get('is_input', False):
            input_nodes.add(node_id)
        if row.get('is_output', False):
            output_nodes.add(node_id)
        
        # Build adjacencies
        for source in connections:
            # Forward: source -> node_id
            if source not in adjacency:
                adjacency[source] = []
            adjacency[source].append(node_id)
            
            # Reverse: node_id <- source
            if node_id not in reverse_adjacency:
                reverse_adjacency[node_id] = []
            reverse_adjacency[node_id].append(source)
    
    # Connectivity validation
    connectivity_results = validate_input_output_paths(
        adjacency, input_nodes, output_nodes
    )
    
    return {
        'adjacency': adjacency,
        'input_nodes': input_nodes,
        'output_nodes': output_nodes,
        'connectivity': connectivity_results
    }


class VectorizedForwardEngine:
    """
    GPU-first forward engine with complete vectorization and graph connectivity analysis.
    
    Key optimizations:
    - Vectorized activation table with GPU tensors
    - Batch propagation processing
    - Parallel timestep computation
    - Graph topology-aware termination logic
    - Output node exclusion from radiation
    - Elimination of Python loops and CPU bottlenecks
    - Direct GPU memory operations
    """
    
    def __init__(
        self,
        graph_df: pd.DataFrame,
        node_store: NodeStore,
        phase_cell,  # PhaseCell or ModularPhaseCell
        lookup_table,
        max_nodes: int ,
        vector_dim: int ,
        phase_bins: int ,
        mag_bins: int ,
        max_timesteps: int ,
        decay_factor: float ,
        min_strength: float ,
        top_k_neighbors: int ,
        use_radiation: bool = True,
        radiation_batch_size: int ,
        min_output_activation_timesteps: int ,
        device: str = 'auto',
        verbose: bool = False
    ):
        """
        Initialize vectorized forward engine.
        
        Args:
            graph_df: Static topology DataFrame
            node_store: Node parameter storage
            phase_cell: Phase computation cell
            lookup_table: Lookup table module
            max_nodes: Maximum number of nodes
            vector_dim: Dimensionality of phase/mag vectors
            phase_bins: Number of discrete phase values
            mag_bins: Number of discrete magnitude values
            max_timesteps: Maximum propagation timesteps
            decay_factor: Activation decay per timestep
            min_strength: Minimum activation strength
            top_k_neighbors: Top-K neighbors for radiation
            use_radiation: Enable dynamic radiation
            radiation_batch_size: Batch size for radiation computation
            min_output_activation_timesteps: Min timesteps before output check
            device: Computation device
            verbose: Enable verbose logging
        """
        self.graph_df = graph_df
        self.node_store = node_store
        self.phase_cell = phase_cell
        self.lookup_table = lookup_table
        self.max_nodes = max_nodes
        self.vector_dim = vector_dim
        self.phase_bins = phase_bins
        self.mag_bins = mag_bins
        self.max_timesteps = max_timesteps
        self.decay_factor = decay_factor
        self.min_strength = min_strength
        self.top_k_neighbors = top_k_neighbors
        self.use_radiation = use_radiation
        self.radiation_batch_size = radiation_batch_size
        self.min_output_activation_timesteps = min_output_activation_timesteps
        self.verbose = verbose
        
        # Device setup
        if device == 'auto':
            from utils.device_manager import get_device_manager
            self.device = get_device_manager().device
        else:
            self.device = torch.device(device)
        
        # Extract output nodes
        self.output_nodes = set(node_store.output_nodes) if hasattr(node_store, 'output_nodes') else set()
        
        # Comprehensive graph analysis
        print(f"🔧 Analyzing graph topology...")
        self.graph_analysis = analyze_graph_connectivity(graph_df)
        
        # Calculate minimum timesteps based on graph topology using average hops
        min_hops = self.graph_analysis['connectivity']['min_hops']
        avg_hops = self.graph_analysis['connectivity']['avg_hops']
        self.min_required_timesteps = max(int(avg_hops) + 2, 5)  # At least 5 timesteps
        
        print(f"📊 Forward engine configuration:")
        print(f"   🔗 Minimum hops: {min_hops}")
        print(f"   📈 Average hops: {avg_hops:.2f}")
        print(f"   ⏱️ Minimum timesteps: {self.min_required_timesteps} (based on avg hops)")
        
        # Pre-create output node indices for propagation engine
        self._init_output_node_indices()
        
        # Initialize vectorized components
        self._init_vectorized_components()
        
        # Performance statistics
        self.stats = {
            'total_forward_passes': 0,
            'total_timesteps': 0,
            'vectorized_operations': 0,
            'gpu_memory_peak': 0,
            'forward_pass_times': [],
            'timestep_times': [],
            'activation_counts': [],
            'early_terminations': 0
        }
        
        print(f"🚀 Vectorized Forward Engine initialized:")
        print(f"   📊 Max nodes: {max_nodes}")
        print(f"   💾 Device: {self.device}")
        print(f"   🎯 Output nodes: {len(self.output_nodes)}")
        print(f"   ⚡ GPU memory allocated: {self._estimate_total_memory():.2f} MB")
    
    def _init_output_node_indices(self): #to be cleaned up , is redundant now with the updated FP termination condition.
        """Initialize output node indices tensor for propagation engine."""
        # Create a temporary propagation engine to get node mappings
        temp_propagation_engine = VectorizedPropagationEngine(
            graph_df=self.graph_df,
            node_store=self.node_store,
            phase_cell=self.phase_cell,
            max_nodes=self.max_nodes,
            device=self.device
        )
        
        # Create output node indices tensor
        self.output_node_indices = torch.tensor([
            temp_propagation_engine.node_to_index.get(node_id, -1)
            for node_id in self.output_nodes
            if node_id in temp_propagation_engine.node_to_index
        ], dtype=torch.long, device=self.device)
        
        # Filter out invalid indices
        valid_mask = self.output_node_indices >= 0
        self.output_node_indices = self.output_node_indices[valid_mask]
        
        print(f"   🚫 Outputs excluded from radiation: {len(self.output_node_indices)}")###
    
    def _init_vectorized_components(self): ### TBR , Not compatible with change of databases
        """Initialize all vectorized components."""
        # Vectorized activation table
        self.activation_table = VectorizedActivationTable(
            max_nodes=self.max_nodes,
            vector_dim=self.vector_dim,
            phase_bins=self.phase_bins,
            mag_bins=self.mag_bins,
            decay_factor=self.decay_factor,
            min_strength=self.min_strength,
            device=self.device
        )
        
        # Vectorized propagation engine (output exclusion removed)
        self.propagation_engine = VectorizedPropagationEngine(
            graph_df=self.graph_df,
            node_store=self.node_store,
            phase_cell=self.phase_cell,
            max_nodes=self.max_nodes,
            device=self.device
        )
        
        # Pre-allocate tensors for batch operations
        self._init_batch_tensors()
    
    def _init_batch_tensors(self): ### TBR, '''
        """Initialize pre-allocated tensors for batch operations."""
        # Batch processing tensors
        self.batch_target_indices = torch.zeros(
            self.max_nodes * 10,  # Allow for multiple targets per node
            dtype=torch.long,
            device=self.device
        )
        self.batch_new_phases = torch.zeros(
            (self.max_nodes * 10, self.vector_dim),
            dtype=torch.long,
            device=self.device
        )
        self.batch_new_mags = torch.zeros(
            (self.max_nodes * 10, self.vector_dim),
            dtype=torch.long,
            device=self.device
        )
        self.batch_strengths = torch.zeros(
            self.max_nodes * 10,
            dtype=torch.float32,
            device=self.device
        )
        
        # Output detection tensors
        self.output_node_indices = torch.tensor([
            self.propagation_engine.node_to_index.get(node_id, -1)
            for node_id in self.output_nodes
            if node_id in self.propagation_engine.node_to_index
        ], dtype=torch.long, device=self.device)
        
        # Filter out invalid indices
        valid_mask = self.output_node_indices >= 0
        self.output_node_indices = self.output_node_indices[valid_mask]
    
    def _estimate_total_memory(self) -> float: ### TBC ,Useful for diagnostics
        """Estimate total GPU memory usage in MB."""
        activation_memory = self.activation_table._estimate_gpu_memory()
        
        # Propagation engine memory
        prop_memory = (
            self.max_nodes * self.vector_dim * 8 * 4 +  # batch tensors
            self.max_nodes * self.max_nodes * 4 * 2     # radiation tensors
        ) / (1024 * 1024)
        
        # Batch operation tensors
        batch_memory = (
            self.max_nodes * 10 * (8 + self.vector_dim * 8 * 2 + 4)
        ) / (1024 * 1024)
        
        return activation_memory + prop_memory + batch_memory
    
    def forward_pass_vectorized(
        self, 
        input_context: Dict[str, Tuple[torch.Tensor, torch.Tensor]] ###Seperate app. functions (input context, etc)
    ) -> VectorizedActivationTable:
        """
        Perform vectorized forward pass with GPU optimization.
        
        Args:
            input_context: Input node activations {node_id: (phase_indices, mag_indices)}
            
        Returns:
            Final vectorized activation table
        """
        start_time = time.time()
        
        # Clear activation table
        self.activation_table.clear() ###Activation table to be reworked,keeping track of current and old act. nodes (history)
        
        # Inject initial input context using batch injection
        self._inject_input_context_batch(input_context)  ### TBC
        
        if self.verbose:
            print(f"🚀 Starting vectorized forward propagation")
            print(f"   📊 Input nodes: {list(input_context.keys())}")
            print(f"   🎯 Output nodes: {list(self.output_nodes)}")
            print(f"   💾 Device: {self.device}")
        ### Many parts below to be extracted out and modularized to multiple functions
        # Vectorized propagation loop
        timestep = 0
        timestep_times = []
        
        while timestep < self.max_timesteps:
            timestep_start = time.time()
            
            # Get vectorized active context
            active_indices, active_phases, active_mags = self.activation_table.get_active_context_vectorized()
            
            if len(active_indices) == 0:
                if self.verbose:
                    print(f"💀 Network died at timestep {timestep} - no active nodes")
                break
            
            # Check for output activation using vectorized operations
            output_active = self._check_output_activation_vectorized(active_indices)
            ###Diagnostic to be handled asynchronously("Out" of this function,otherwise this would become a bottleneck)
            if self.verbose:
                print(f"\n⏱️ Timestep {timestep}")
                print(f"   🔹 Active nodes: {len(active_indices)}")
                
                # ENHANCED DIAGNOSTICS
                output_diagnostics = self._detailed_output_diagnostics(active_indices)
                print(f"   🎯 Output diagnostics: {output_diagnostics}")
                
                # Check signal strengths reaching outputs
                output_signals = self._get_output_signal_strengths()
                print(f"   💪 Output signal strengths: {output_signals}")
                
                # Check if termination condition would trigger
                termination_ready = (timestep >= self.min_required_timesteps)
                print(f"   ⏰ Termination ready: {termination_ready} (min: {self.min_required_timesteps})")
                print(f"   🔍 Output active mask: {output_active.tolist()}")
                
                if output_active.any():
                    active_output_indices = self.output_node_indices[output_active]
                    active_output_nodes = [
                        self.propagation_engine.index_to_node.get(idx.item(), f"idx_{idx}")
                        for idx in active_output_indices
                    ]
                    print(f"   ✅ Active outputs detected: {active_output_nodes}")
            
            # Enhanced termination with graph topology awareness  ###TBR , have two seperate functions (for one-shot and continuous input)
            if (timestep >= self.min_required_timesteps and output_active.any()):
                if self.verbose:
                    print(f"✅ Output activation detected at timestep {timestep}")
                    print(f"   📊 Required minimum: {self.min_required_timesteps} (graph depth + 2)")
                    active_output_nodes = [
                        self.propagation_engine.index_to_node.get(idx.item(), f"idx_{idx}")
                        for idx in self.output_node_indices[output_active]
                    ]
                    print(f"   🎯 Active outputs: {active_output_nodes}")
                self.stats['early_terminations'] += 1
                break
            
            # Additional safety check for very long runs
            if timestep >= self.max_timesteps - 1:
                if self.verbose:
                    print(f"⏰ Maximum timesteps reached ({self.max_timesteps})")
                    if not output_active.any():
                        print(f"   ⚠️  No output activation achieved")
                break
            
            # Vectorized propagation step
            target_indices, new_phases, new_mags, strengths = self.propagation_engine.propagate_vectorized(
                active_indices=active_indices,
                active_phases=active_phases,
                active_mags=active_mags,
                use_radiation=self.use_radiation,
                top_k_neighbors=self.top_k_neighbors,
                radiation_batch_size=self.radiation_batch_size
            )
            
            # CRITICAL FIX: Remove the destructive clear() call
            # self.activation_table.clear()  # REMOVED - was destroying activations!
            
            # Batch inject new activations (accumulative)
            if len(target_indices) > 0:
                self._inject_propagation_results_batch(target_indices, new_phases, new_mags, strengths)
            
            # Enhanced vectorized decay and prune with output protection
            self.activation_table.decay_and_prune_vectorized(output_nodes=self.output_nodes)
            
            timestep_end = time.time()
            timestep_times.append(timestep_end - timestep_start)
            timestep += 1
        
        # Final summary
        final_indices, final_phases, final_mags = self.activation_table.get_active_context_vectorized()
        final_output_active = self._check_output_activation_vectorized(final_indices)
        
        if self.verbose:
            print(f"\n🏁 Vectorized forward pass completed after {timestep} timesteps")
            
            # DETAILED FINAL STATE ANALYSIS
            print(f"   📊 Final timestep analysis:")
            print(f"      • Active nodes: {len(final_indices)}")
            if len(final_indices) > 0:
                final_node_ids = [
                    self.activation_table.index_to_node_id.get(idx.item(), f'idx_{idx}') 
                    for idx in final_indices[:10]
                ]
                print(f"      • Active node IDs (first 10): {final_node_ids}")
            
            # Check each output node individually
            print(f"   🎯 Individual output node analysis:")
            for output_node_id in self.output_nodes:
                is_active = self.activation_table.is_active(output_node_id)
                strength = 0.0
                if is_active:
                    act_idx = self.activation_table._get_node_index(output_node_id)
                    strength = self.activation_table.strength_storage[act_idx].item()
                print(f"      • Output {output_node_id}: active={is_active}, strength={strength:.4f}")
            
            # Final output detection summary
            if final_output_active.any():
                final_active_outputs = self.output_node_indices[final_output_active]
                final_output_nodes = [
                    self.propagation_engine.index_to_node.get(idx.item(), f"idx_{idx}")
                    for idx in final_active_outputs
                ]
                print(f"   ✅ Final active output nodes: {final_output_nodes}")
            else:
                print(f"   ❌ No output nodes detected as active in final timestep")
            
            print(f"   ⚡ Average timestep time: {sum(timestep_times)/max(len(timestep_times), 1):.4f}s")
        
        # Update statistics
        end_time = time.time()
        self.stats['total_forward_passes'] += 1
        self.stats['total_timesteps'] += timestep
        self.stats['vectorized_operations'] += timestep
        self.stats['forward_pass_times'].append(end_time - start_time)
        self.stats['timestep_times'].extend(timestep_times)
        self.stats['activation_counts'].append(len(final_indices))
        
        # Update GPU memory peak
        if hasattr(torch.cuda, 'max_memory_allocated'):
            current_memory = torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
            self.stats['gpu_memory_peak'] = max(self.stats['gpu_memory_peak'], current_memory)
        
        return self.activation_table
    
    def _inject_input_context_batch(
        self, 
        input_context: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ):
        """
        Inject input context using batch operations for maximum performance.
        
        Args:
            input_context: Input node activations {node_id: (phase_indices, mag_indices)}
        """
        if not input_context:
            return
        
        # Convert to batch format
        node_ids = list(input_context.keys())
        batch_size = len(node_ids)
        
        # Get node indices
        node_indices = torch.tensor([
            self.activation_table._get_node_index(node_id) for node_id in node_ids
        ], dtype=torch.long, device=self.device)
        
        # Stack phase and mag data
        phase_data = torch.stack([
            input_context[node_id][0].to(self.device) for node_id in node_ids
        ])
        mag_data = torch.stack([
            input_context[node_id][1].to(self.device) for node_id in node_ids
        ])
        
        # Batch strengths (all inputs start with strength 1.0)
        strengths = torch.ones(batch_size, dtype=torch.float32, device=self.device)
        
        # Batch injection
        self.activation_table.inject_batch(node_indices, phase_data, mag_data, strengths)
    
    def _inject_propagation_results_batch(
        self,
        target_indices: torch.Tensor,
        new_phases: torch.Tensor,
        new_mags: torch.Tensor,
        strengths: torch.Tensor
    ):
        """
        Inject propagation results using batch operations.
        
        Args:
            target_indices: Target node indices [num_targets] - already filtered for excitatory signals
            new_phases: New phase indices [num_targets, vector_dim] - already filtered
            new_mags: New magnitude indices [num_targets, vector_dim] - already filtered
            strengths: Activation strengths [num_targets] - already filtered (all > 0)
        """
        if len(target_indices) == 0:
            return
        
        # Ensure all tensors have consistent batch size
        batch_size = target_indices.size(0)
        assert new_phases.size(0) == batch_size, f"Phase batch size {new_phases.size(0)} != target batch size {batch_size}"
        assert new_mags.size(0) == batch_size, f"Mag batch size {new_mags.size(0)} != target batch size {batch_size}"
        assert strengths.size(0) == batch_size, f"Strength batch size {strengths.size(0)} != target batch size {batch_size}"
        
        # Convert target indices to activation table indices
        activation_indices = torch.zeros_like(target_indices)
        valid_mask = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        for i, target_idx in enumerate(target_indices):
            if target_idx.item() in self.propagation_engine.index_to_node:
                node_id = self.propagation_engine.index_to_node[target_idx.item()]
                activation_indices[i] = self.activation_table._get_node_index(node_id)
                valid_mask[i] = True
        
        # Only inject valid targets
        if valid_mask.any():
            valid_activation_indices = activation_indices[valid_mask]
            valid_new_phases = new_phases[valid_mask]
            valid_new_mags = new_mags[valid_mask]
            valid_strengths = strengths[valid_mask]
            
            # Batch injection with validated tensors
            self.activation_table.inject_batch(
                valid_activation_indices, 
                valid_new_phases, 
                valid_new_mags, 
                valid_strengths
            )
    
    def _check_output_activation_vectorized(self, active_indices: torch.Tensor) -> torch.Tensor:
        """
        ENHANCED: Check for output activation with robust index mapping.
        
        Args:
            active_indices: Currently active node indices [num_active]
            
        Returns:
            Boolean mask indicating which output nodes are active [num_output_nodes]
        """
        if len(self.output_node_indices) == 0 or len(active_indices) == 0:
            return torch.zeros(len(self.output_node_indices), dtype=torch.bool, device=self.device)
        
        # ENHANCED: Direct check using activation table (more reliable)
        output_active = torch.zeros(len(self.output_node_indices), dtype=torch.bool, device=self.device)
        
        for i, output_idx in enumerate(self.output_node_indices):
            output_node_id = self.propagation_engine.index_to_node.get(output_idx.item())
            if output_node_id:
                # Direct check in activation table (bypasses index conversion issues)
                is_active = self.activation_table.is_active(output_node_id)
                output_active[i] = is_active
        
        return output_active
    
    def _detailed_output_diagnostics(self, active_indices: torch.Tensor) -> Dict:
        """Detailed diagnostics for output node detection."""
        diagnostics = {}
        
        for i, output_idx in enumerate(self.output_node_indices):
            output_node_id = self.propagation_engine.index_to_node.get(output_idx.item())
            
            # Check if output is in active indices
            is_active_in_table = any(
                self.activation_table.index_to_node_id.get(act_idx.item()) == output_node_id
                for act_idx in active_indices
            )
            
            # Check activation table directly
            is_active_direct = self.activation_table.is_active(output_node_id) if output_node_id else False
            
            # Get signal strength if active
            strength = None
            if is_active_direct and output_node_id:
                act_idx = self.activation_table._get_node_index(output_node_id)
                if act_idx < len(self.activation_table.strength_storage):
                    strength = self.activation_table.strength_storage[act_idx].item()
            
            diagnostics[output_node_id or f"idx_{output_idx}"] = {
                'active_in_table': is_active_in_table,
                'active_direct': is_active_direct,
                'strength': strength
            }
        
        return diagnostics
    
    def _get_output_signal_strengths(self) -> Dict:
        """Get signal strengths for all output nodes."""
        strengths = {}
        
        for output_node_id in self.output_nodes:
            if self.activation_table.is_active(output_node_id):
                act_idx = self.activation_table._get_node_index(output_node_id)
                strength = self.activation_table.strength_storage[act_idx].item()
                strengths[output_node_id] = strength
            else:
                strengths[output_node_id] = 0.0
        
        return strengths
    
    def get_active_output_nodes(self) -> List[str]:
        """
        Get list of currently active output node IDs.
        
        Returns:
            List of active output node IDs
        """
        active_indices, _, _ = self.activation_table.get_active_context_vectorized()
        output_active = self._check_output_activation_vectorized(active_indices)
        
        active_output_nodes = []
        for i, is_active in enumerate(output_active):
            if is_active and i < len(self.output_node_indices):
                prop_idx = self.output_node_indices[i].item()
                if prop_idx in self.propagation_engine.index_to_node:
                    node_id = self.propagation_engine.index_to_node[prop_idx]
                    active_output_nodes.append(node_id)
        
        return active_output_nodes
    
    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics."""
        avg_forward_time = (
            sum(self.stats['forward_pass_times']) / 
            max(len(self.stats['forward_pass_times']), 1)
        )
        avg_timestep_time = (
            sum(self.stats['timestep_times']) / 
            max(len(self.stats['timestep_times']), 1)
        )
        avg_activations = (
            sum(self.stats['activation_counts']) / 
            max(len(self.stats['activation_counts']), 1)
        )
        
        # Calculate samples per second
        samples_per_second = 1.0 / max(avg_forward_time, 0.001)
        
        return {
            'total_forward_passes': self.stats['total_forward_passes'],
            'total_timesteps': self.stats['total_timesteps'],
            'vectorized_operations': self.stats['vectorized_operations'],
            'early_terminations': self.stats['early_terminations'],
            'avg_forward_time': avg_forward_time,
            'avg_timestep_time': avg_timestep_time,
            'avg_activations': avg_activations,
            'samples_per_second': samples_per_second,
            'gpu_memory_peak_mb': self.stats['gpu_memory_peak'],
            'estimated_memory_mb': self._estimate_total_memory(),
            'device': str(self.device),
            'vectorization_ratio': (
                self.stats['vectorized_operations'] / 
                max(self.stats['total_timesteps'], 1)
            )
        }
    
    def print_performance_report(self):
        """Print comprehensive performance report."""
        stats = self.get_performance_stats()
        activation_stats = self.activation_table.get_performance_stats()
        propagation_stats = self.propagation_engine.get_performance_stats()
        
        print(f"\n🚀 Vectorized Forward Engine Performance Report")
        print(f"=" * 60)
        
        print(f"Forward Pass Performance:")
        print(f"  Total passes: {stats['total_forward_passes']:,}")
        print(f"  Average time: {stats['avg_forward_time']:.4f}s")
        print(f"  Samples/second: {stats['samples_per_second']:.2f}")
        print(f"  Early terminations: {stats['early_terminations']:,}")
        
        print(f"\nTimestep Performance:")
        print(f"  Total timesteps: {stats['total_timesteps']:,}")
        print(f"  Average time: {stats['avg_timestep_time']:.4f}s")
        print(f"  Average activations: {stats['avg_activations']:.1f}")
        
        print(f"\nGPU Memory:")
        print(f"  Estimated usage: {stats['estimated_memory_mb']:.2f} MB")
        print(f"  Peak usage: {stats['gpu_memory_peak_mb']:.2f} MB")
        print(f"  Device: {stats['device']}")
        
        print(f"\nVectorization:")
        print(f"  Vectorized operations: {stats['vectorized_operations']:,}")
        print(f"  Vectorization ratio: {stats['vectorization_ratio']:.1%}")
        
        # Component performance
        print(f"\nComponent Performance:")
        print(f"  Activation table vectorization: {activation_stats['vectorization_ratio']:.1%}")
        print(f"  Propagation vectorization: {propagation_stats['vectorization_ratio']:.1%}")
        
        # Performance assessment
        print(f"\nOverall Assessment:")
        if stats['samples_per_second'] > 10:
            print("  🟢 Excellent performance (>10 samples/sec)")
        elif stats['samples_per_second'] > 2:
            print("  🟡 Good performance (>2 samples/sec)")
        else:
            print("  🔴 Poor performance (<2 samples/sec)")
        
        speedup_vs_baseline = stats['samples_per_second'] / 0.5  # vs 0.5 samples/sec baseline
        print(f"  Speedup vs baseline: {speedup_vs_baseline:.1f}x")


# Factory function for easy creation
def create_vectorized_forward_engine(
    graph_df: pd.DataFrame,
    node_store: NodeStore,
    phase_cell,
    lookup_table,
    config: Dict,
    device: str = 'auto'
) -> VectorizedForwardEngine:
    """
    Factory function to create vectorized forward engine.
    
    Args:
        graph_df: Static topology DataFrame
        node_store: Node parameter storage
        phase_cell: Phase computation cell
        lookup_table: Lookup table module
        config: Configuration dictionary
        device: Computation device
        
    Returns:
        Configured vectorized forward engine
    """
    return VectorizedForwardEngine(
        graph_df=graph_df,
        node_store=node_store,
        phase_cell=phase_cell,
        lookup_table=lookup_table,
        max_nodes=config.get('architecture', {}).get('total_nodes', 1000) ,
        vector_dim=config.get('architecture', {}).get('vector_dim', 5),
        phase_bins=config.get('resolution', {}).get('phase_bins', 512),
        mag_bins=config.get('resolution', {}).get('mag_bins', 1024),
        max_timesteps=config.get('forward_pass', {}).get('max_timesteps', 35),
        decay_factor=config.get('forward_pass', {}).get('decay_factor', 0.95),
        min_strength=config.get('forward_pass', {}).get('min_activation_strength', 0.001),
        top_k_neighbors=config.get('graph_structure', {}).get('top_k_neighbors', 4),
        use_radiation=config.get('graph_structure', {}).get('use_radiation', True),
        radiation_batch_size=config.get('radiation', {}).get('batch_size', 128),
        min_output_activation_timesteps=config.get('forward_pass', {}).get('min_output_activation_timesteps', 2),
        device=device,
        verbose=config.get('forward_pass', {}).get('verbose', False)
    )
