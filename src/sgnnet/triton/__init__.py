"""SGNNET Triton kernels — fused routing hot path."""
try:
    from .routing_kernel import routing_step_proj, TRITON_AVAILABLE
    from .sgnnet_proj_triton import SGNNET_DeltaProjTriton
    __all__ = ["routing_step_proj", "TRITON_AVAILABLE", "SGNNET_DeltaProjTriton"]
except Exception:
    TRITON_AVAILABLE = False
    __all__ = ["TRITON_AVAILABLE"]

try:
    from .gather_norm import route_k_iter as triton_route_k_iter, is_supported as triton_route_supported
    __all__ += ["triton_route_k_iter", "triton_route_supported"]
except Exception:
    def triton_route_k_iter(Z, conn, k_iter, **kw):
        raise ImportError("gather_norm Triton kernel unavailable")
    def triton_route_supported(D):
        return False
