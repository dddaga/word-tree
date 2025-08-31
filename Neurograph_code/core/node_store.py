# core/node_store.py

import torch
import torch.nn as nn


class NodeStore(nn.Module):
    def __init__(self, graph_df, vector_dim, phase_bins, mag_bins):
        """
        Stores discrete phase and magnitude index vectors per node with activation tracking.
        """
        super().__init__()

        self.node_ids = graph_df["node_id"].tolist()
        self.node_index = {nid: idx for idx, nid in enumerate(self.node_ids)}
        self.input_nodes = set(graph_df[graph_df["is_input"]]["node_id"])
        self.output_nodes = set(graph_df[graph_df["is_output"]]["node_id"])
        self.connections = {
            nid: graph_df.loc[graph_df["node_id"] == nid, "input_connections"].values[0]
            for nid in self.node_ids
        }

        # Store discrete indices directly as parameters (for manual updates)
        # ALL nodes get random phase/mag values (preserves radiation diversity)
        self.phase_table = nn.ParameterDict({
            nid: nn.Parameter(
                torch.randint(low=0, high=phase_bins, size=(vector_dim,), dtype=torch.long),
                requires_grad=False  # you'll manage gradients manually
            )
            for nid in self.node_ids
        })

        self.mag_table = nn.ParameterDict({
            nid: nn.Parameter(
                torch.randint(low=0, high=mag_bins, size=(vector_dim,), dtype=torch.long),
                requires_grad=False
            )
            for nid in self.node_ids
        })
        
        # NEW: Activity flags - Only input nodes start active
        self.node_active_flags = nn.ParameterDict({
            nid: nn.Parameter(
                torch.tensor(nid in self.input_nodes, dtype=torch.bool),
                requires_grad=False
            )
            for nid in self.node_ids
        })
        
        print(f"🔧 NodeStore initialized:")
        print(f"   📊 Total nodes: {len(self.node_ids)}")
        print(f"   🎯 Input nodes: {len(self.input_nodes)} (initially active)")
        print(f"   🔄 Other nodes: {len(self.node_ids) - len(self.input_nodes)} (initially inactive)")

    def get_phase(self, node_id):
        return self.phase_table[node_id]  # Returns LongTensor [D]

    def get_mag(self, node_id):
        return self.mag_table[node_id]    # Returns LongTensor [D]

    def get_inputs(self, node_id):
        return self.connections.get(node_id, [])

    def is_input(self, node_id):
        return node_id in self.input_nodes

    def is_output(self, node_id):
        return node_id in self.output_nodes
    
    def is_node_active(self, node_id):
        """Check if node is currently active."""
        return self.node_active_flags[node_id].item()
    
    def activate_node(self, node_id):
        """Activate a node (mark as active)."""
        self.node_active_flags[node_id].data = torch.tensor(True)
    
    def deactivate_node(self, node_id):
        """Deactivate a node (mark as inactive)."""
        self.node_active_flags[node_id].data = torch.tensor(False)
    
    def get_active_nodes(self):
        """Get list of currently active node IDs."""
        return [nid for nid in self.node_ids if self.is_node_active(nid)]
    
    def get_active_input_nodes(self):
        """Get list of currently active input node IDs."""
        return [nid for nid in self.input_nodes if self.is_node_active(nid)]
    
    def get_active_output_nodes(self):
        """Get list of currently active output node IDs."""
        return [nid for nid in self.output_nodes if self.is_node_active(nid)]
    
    def reset_all_activations(self):
        """Reset all nodes to inactive except input nodes."""
        for nid in self.node_ids:
            if nid in self.input_nodes:
                self.activate_node(nid)
            else:
                self.deactivate_node(nid)
    
    def get_state(self):
        """Get the current state for checkpointing."""
        return {
            'phase_table': {k: v.clone() for k, v in self.phase_table.items()},
            'mag_table': {k: v.clone() for k, v in self.mag_table.items()},
            'node_ids': self.node_ids,
            'input_nodes': self.input_nodes,
            'output_nodes': self.output_nodes,
            'connections': self.connections
        }
    
    def load_state(self, state):
        """Load state from checkpoint."""
        for k, v in state['phase_table'].items():
            if k in self.phase_table:
                self.phase_table[k].data = v
        for k, v in state['mag_table'].items():
            if k in self.mag_table:
                self.mag_table[k].data = v
