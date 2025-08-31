# modules/specialized_output_adapters.py

import torch
from torch.nn.functional import cosine_similarity


def predict_label_from_specialized_output(activation_table, node_specializations, class_encodings, lookup_table):
    """
    Predicts the digit class using specialized output nodes.
    Each node is trained to recognize specific digits, so we use the most confident
    specialized node to make the prediction.

    Args:
        activation_table: ActivationTable after forward pass
        node_specializations: dict[node_id → list of assigned digits]
        class_encodings: dict[label → (phase_idx, mag_idx)]
        lookup_table: ExtendedLookupTableModule

    Returns:
        int: predicted label (0–9), or -1 if no active nodes
    """
    
    # Get predictions from each specialized node
    node_predictions = {}
    node_confidences = {}
    
    for node_id, assigned_digits in node_specializations.items():
        if not activation_table.is_active(node_id):
            continue

        # Get node's output vector
        p_idx, m_idx = activation_table.table[node_id][:2]
        phase_vec = lookup_table.lookup_phase(p_idx)
        mag_vec = lookup_table.lookup_magnitude(m_idx)
        node_output = phase_vec * mag_vec

        # Find best match among this node's assigned digits
        best_digit = -1
        best_confidence = -1.0
        
        for digit in assigned_digits:
            tgt_phase, tgt_mag = class_encodings[digit]
            tgt_vec = lookup_table.lookup_phase(tgt_phase) * lookup_table.lookup_magnitude(tgt_mag)
            confidence = cosine_similarity(node_output, tgt_vec, dim=0).item()
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_digit = digit
        
        if best_digit != -1:
            node_predictions[node_id] = best_digit
            node_confidences[node_id] = best_confidence

    if not node_predictions:
        return -1  # No active specialized nodes

    # Return prediction from most confident node
    most_confident_node = max(node_confidences.keys(), key=lambda k: node_confidences[k])
    return node_predictions[most_confident_node]


def predict_label_with_voting(activation_table, node_specializations, class_encodings, lookup_table):
    """
    Alternative prediction method using voting across specialized nodes.
    Each node votes for its best digit, weighted by confidence.

    Args:
        activation_table: ActivationTable after forward pass
        node_specializations: dict[node_id → list of assigned digits]
        class_encodings: dict[label → (phase_idx, mag_idx)]
        lookup_table: ExtendedLookupTableModule

    Returns:
        int: predicted label (0–9), or -1 if no active nodes
    """
    
    # Collect votes from each specialized node
    digit_votes = {i: 0.0 for i in range(10)}
    total_votes = 0.0
    
    for node_id, assigned_digits in node_specializations.items():
        if not activation_table.is_active(node_id):
            continue

        # Get node's output vector
        p_idx, m_idx = activation_table.table[node_id][:2]
        phase_vec = lookup_table.lookup_phase(p_idx)
        mag_vec = lookup_table.lookup_magnitude(m_idx)
        node_output = phase_vec * mag_vec

        # Vote for best digit among assigned digits, weighted by confidence
        best_digit = -1
        best_confidence = -1.0
        
        for digit in assigned_digits:
            tgt_phase, tgt_mag = class_encodings[digit]
            tgt_vec = lookup_table.lookup_phase(tgt_phase) * lookup_table.lookup_magnitude(tgt_mag)
            confidence = cosine_similarity(node_output, tgt_vec, dim=0).item()
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_digit = digit
        
        if best_digit != -1 and best_confidence > 0:  # Only positive confidence votes
            digit_votes[best_digit] += best_confidence
            total_votes += best_confidence

    if total_votes == 0:
        return -1  # No positive votes

    # Return digit with highest weighted vote
    return max(digit_votes.keys(), key=lambda k: digit_votes[k])


def analyze_specialized_predictions(activation_table, node_specializations, class_encodings, lookup_table):
    """
    Analyze predictions from all specialized nodes for debugging.
    
    Returns:
        dict: Analysis results including per-node predictions and confidences
    """
    
    analysis = {
        'active_nodes': [],
        'node_predictions': {},
        'node_confidences': {},
        'digit_votes': {i: 0.0 for i in range(10)},
        'final_prediction_confident': -1,
        'final_prediction_voting': -1
    }
    
    for node_id, assigned_digits in node_specializations.items():
        if not activation_table.is_active(node_id):
            continue
            
        analysis['active_nodes'].append(node_id)

        # Get node's output vector
        p_idx, m_idx = activation_table.table[node_id][:2]
        phase_vec = lookup_table.lookup_phase(p_idx)
        mag_vec = lookup_table.lookup_magnitude(m_idx)
        node_output = phase_vec * mag_vec

        # Analyze predictions for all assigned digits
        node_digit_confidences = {}
        best_digit = -1
        best_confidence = -1.0
        
        for digit in assigned_digits:
            tgt_phase, tgt_mag = class_encodings[digit]
            tgt_vec = lookup_table.lookup_phase(tgt_phase) * lookup_table.lookup_magnitude(tgt_mag)
            confidence = cosine_similarity(node_output, tgt_vec, dim=0).item()
            
            node_digit_confidences[digit] = confidence
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_digit = digit
        
        analysis['node_predictions'][node_id] = {
            'assigned_digits': assigned_digits,
            'digit_confidences': node_digit_confidences,
            'best_digit': best_digit,
            'best_confidence': best_confidence
        }
        
        # Add to voting
        if best_digit != -1 and best_confidence > 0:
            analysis['digit_votes'][best_digit] += best_confidence
    
    # Get final predictions
    if analysis['node_predictions']:
        # Most confident node method
        best_node = max(analysis['node_predictions'].keys(), 
                       key=lambda k: analysis['node_predictions'][k]['best_confidence'])
        analysis['final_prediction_confident'] = analysis['node_predictions'][best_node]['best_digit']
        
        # Voting method
        if sum(analysis['digit_votes'].values()) > 0:
            analysis['final_prediction_voting'] = max(analysis['digit_votes'].keys(), 
                                                    key=lambda k: analysis['digit_votes'][k])
    
    return analysis
