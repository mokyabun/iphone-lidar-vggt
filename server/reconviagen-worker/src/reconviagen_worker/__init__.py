from __future__ import annotations

__all__ = [
    "ReconViaGenResult",
    "_alignment_rmse",
    "_best_metric_transform",
    "_export_print_stl",
    "_nearest_distances",
    "_object_asset_normalization",
    "_prepare_print_mesh",
    "_refine_metric_transform_icp",
    "prepare_multiview_input",
    "run_reconviagen",
]


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(name)
    from . import client

    return getattr(client, name)
