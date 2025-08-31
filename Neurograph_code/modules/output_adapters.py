# modules/output_adapters.py

import torch
from torch.nn.functional import cosine_similarity

def predict_label_from_output(activation_table, output_nodes, class_encodings, lookup_table):
    """
    Predicts the digit class by comparing the output vector to fixed class encodings.

    Args:
        activation_table: ActivationTable after forward pass
        output_nodes: set of output node IDs
        class_encodings: dict[label → (phase_idx, mag_idx)]
        lookup_table: ExtendedLookupTableModule

    Returns:
        int: predicted label (0–9), or -1 if no active nodes
    """
    vectors = []
    for node_id in output_nodes:
        if not activation_table.is_active(node_id):
            continue

        p_idx, m_idx = activation_table.table[node_id][:2]
        phase_vec = lookup_table.lookup_phase(p_idx)
        mag_vec   = lookup_table.lookup_magnitude(m_idx)
        signal    = phase_vec * mag_vec
        vectors.append(signal)

    if not vectors:
        return -1  # No active output nodes

    output_vector = torch.stack(vectors, dim=0).mean(dim=0)  # Average across output nodes

    scores = []
    for digit in range(10):
        tgt_phase, tgt_mag = class_encodings[digit]
        tgt_vec = lookup_table.lookup_phase(tgt_phase) * lookup_table.lookup_magnitude(tgt_mag)
        score = cosine_similarity(output_vector, tgt_vec, dim=0)
        scores.append(score.item())

    return int(torch.argmax(torch.tensor(scores)))
