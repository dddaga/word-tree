"""SGNNET Triton kernels — fused routing hot path."""
try:
    from .routing_kernel import routing_step_proj, TRITON_AVAILABLE
    from .sgnnet_proj_triton import SGNNET_DeltaProjTriton
    __all__ = ["routing_step_proj", "TRITON_AVAILABLE", "SGNNET_DeltaProjTriton"]
except Exception:
    TRITON_AVAILABLE = False
    __all__ = ["TRITON_AVAILABLE"]
