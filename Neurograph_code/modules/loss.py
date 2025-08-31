# modules/loss.py

import torch

def signal_loss_from_lookup(pred_phase_idx, pred_mag_idx, tgt_phase_idx, tgt_mag_idx, lookup):
    """
    Computes real-valued vector loss using cosine + magnitude embeddings.

    Args:
        pred_phase_idx, pred_mag_idx: tensors of shape [D]
        tgt_phase_idx, tgt_mag_idx: tensors of shape [D]
        lookup: ExtendedLookupTableModule

    Returns:
        Scalar MSE loss between decoded signal vectors
    """
    pred_signal = lookup.get_signal_vector(pred_phase_idx, pred_mag_idx)  # [D]
    tgt_signal  = lookup.get_signal_vector(tgt_phase_idx, tgt_mag_idx)   # [D]
    return torch.mean((pred_signal - tgt_signal) ** 2)  # MSE
