"""Small, genuine PhysicsNeMo models behind one batch-size-one interface.

Batch tensors are already normalized and on the model's device: ``xyz`` [N,3]
in [0,1], ``thickness`` [N,1], and training-normalized ``globals`` [3]. MGN
additionally takes actual, directed FEM ``edge_index`` [2,E]; no edges are invented.
Outputs are [1,N,out_dim], with the caller defining the time/channel ordering.
Importing this file does not import torch or PhysicsNeMo or construct a model.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import os
from pathlib import Path


ARCHITECTURES = (
    "transolver", "flare", "geotransolver", "geoflare", "meshgraphnet", "figconvunet"
)
_ALIASES = {"geo_flare": "geoflare", "geo_transolver": "geotransolver",
            "mgn": "meshgraphnet", "figconv": "figconvunet"}
_CLASSES = {
    "transolver": ("physicsnemo.models.transolver", "Transolver"),
    "flare": ("physicsnemo.models.flare", "FLARE"),
    "geotransolver": ("physicsnemo.models.geotransolver", "GeoTransolver"),
    "geoflare": ("physicsnemo.models.geotransolver", "GeoTransolver"),
    "meshgraphnet": ("physicsnemo.models.meshgraphnet", "MeshGraphNet"),
    "figconvunet": ("physicsnemo.models.figconvnet", "FIGConvUNet"),
}


def _canonical(architecture: str) -> str:
    name = _ALIASES.get(architecture.lower(), architecture.lower())
    if name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {architecture!r}; choose {ARCHITECTURES}")
    return name


def model_config(architecture: str, out_dim: int, global_dim: int = 3) -> dict:
    """Return JSON-serializable constructor kwargs without importing a framework."""
    name = _canonical(architecture)
    if isinstance(out_dim, bool) or not isinstance(out_dim, int) or out_dim <= 0:
        raise ValueError("out_dim must be a positive integer (time_count * channel_count)")
    if not isinstance(global_dim, int) or global_dim < 1:
        raise ValueError("global_dim must be a positive integer")
    small = dict(n_layers=2, n_hidden=48, n_head=4, mlp_ratio=2,
                 slice_num=8, dropout=0.0, use_te=False, time_input=False)
    if name in ("transolver", "flare"):
        config = dict(small, functional_dim=1+global_dim, embedding_dim=3, out_dim=out_dim,
                      unified_pos=False, structured_shape=None)
        if name == "transolver":
            config["plus"] = False
        return config
    if name in ("geotransolver", "geoflare"):
        return dict(small, functional_dim=4, out_dim=out_dim,
                    geometry_dim=3, global_dim=global_dim, include_local_features=False,
                    structured_shape=None, plus=False, concrete_dropout=False,
                    state_mixing_mode="weighted",
                    attention_type="GALE" if name == "geotransolver" else "GALE_FA")
    if name == "meshgraphnet":
        return dict(
            input_dim_nodes=4+global_dim, input_dim_edges=4, output_dim=out_dim,
            processor_size=3, hidden_dim_processor=48,
            hidden_dim_node_encoder=48, hidden_dim_edge_encoder=48,
            hidden_dim_node_decoder=48, num_layers_node_encoder=2,
            num_layers_edge_encoder=2, num_layers_node_decoder=2,
            num_layers_node_processor=2, num_layers_edge_processor=2,
            mlp_activation_fn="relu", aggregation="sum", do_concat_trick=False,
            num_processor_checkpoint_segments=0, checkpoint_offloading=False,
            recompute_activation=False, norm_type="LayerNorm",
        )
    return dict(
        in_channels=4+global_dim, out_channels=out_dim, kernel_size=3,
        hidden_channels=[16, 24, 32], num_levels=2,
        num_down_blocks=1, num_up_blocks=1,
        aabb_min=(0.0, 0.0, 0.0), aabb_max=(1.0, 1.0, 1.0),
        voxel_size=None,
        resolution_memory_format_pairs=[
            ("b_xc_y_z", (2, 16, 16)),
            ("b_yc_x_z", (16, 2, 16)),
            ("b_zc_x_y", (16, 16, 2)),
        ],
        use_rel_pos=True, use_rel_pos_embed=True, pos_encode_dim=8,
        communication_types=["sum"], to_point_sample_method="graphconv",
        neighbor_search_type="knn", knn_k=8, reductions=["mean"],
        use_scalar_output=False, has_input_features=True,
    )


def build_model(architecture: str, out_dim: int, config: dict | None = None, global_dim: int = 3):
    """Return ``(model, constructor_config, source_metadata)``; never substitute.

    Missing dependencies or unsupported installed APIs deliberately raise. The
    GeoFLARE label is GeoTransolver's installed GALE_FA backend, not a class name.
    The caller owns device placement, initialization seed, and optimizer. Saved
    constructor kwargs may override defaults for exact checkpoint reconstruction.
    """
    name = _canonical(architecture)
    kwargs = model_config(name, out_dim, global_dim=global_dim)
    if config is not None:
        kwargs.update(config)
    output_key = {"meshgraphnet": "output_dim", "figconvunet": "out_channels"}.get(name, "out_dim")
    if kwargs[output_key] != out_dim:
        raise ValueError("Saved config output dimension does not match out_dim")
    if kwargs.get("use_te", False) or kwargs.get("norm_type", "LayerNorm") != "LayerNorm":
        raise ValueError("These benchmark adapters require the PyTorch backend without TE")
    module_name, class_name = _CLASSES[name]
    # MGN's LayerNorm provider reads this flag while constructing its layers.
    previous_te = os.environ.get("PHYSICSNEMO_FORCE_TE")
    os.environ["PHYSICSNEMO_FORCE_TE"] = "False"
    try:
        model_class = getattr(importlib.import_module(module_name), class_name)
        model = model_class(**kwargs)
    finally:
        if previous_te is None:
            os.environ.pop("PHYSICSNEMO_FORCE_TE", None)
        else:
            os.environ["PHYSICSNEMO_FORCE_TE"] = previous_te
    source_file = Path(inspect.getfile(model_class)).resolve()
    try:
        version = importlib.metadata.version("nvidia-physicsnemo")
    except importlib.metadata.PackageNotFoundError:
        version = None
    metadata = dict(
        architecture=name, module=module_name, class_name=class_name,
        implementation_module=model_class.__module__,
        physicsnemo_distribution_version=version,
        source_file=str(source_file),
        source_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
        transformer_engine=False,
        output_layout="batch,node,(caller-defined time*channel)",
        input_information=["initial_xyz", "nodal_thickness", f"{global_dim}_global_parameters"],
    )
    if name == "meshgraphnet":
        metadata.update(
            extra_structure="actual directed FEM mesh edges supplied by caller",
            edge_features="xyz[source]-xyz[target], Euclidean length",
            dependencies=["torch_geometric", "torch_scatter"],
        )
    elif name == "figconvunet":
        metadata.update(
            dependencies=["warp-lang (import)", "torch_scatter (CSR reduction)"],
            neighbor_backend="installed torch.cdist/topk KNN; no radius-Warp execution",
        )
    elif name == "geoflare":
        metadata["variant"] = "GeoTransolver attention_type=GALE_FA (FLARE + geometry/global context)"
    return model, kwargs, metadata


def validate_batch(batch: dict, *, require_edges: bool = False,
                   check_values: bool = True) -> int:
    """Validate one batch; use the value checks once before timed training."""
    import torch

    xyz, thickness, global_values = (batch[key] for key in ("xyz", "thickness", "globals"))
    for key, value in (("xyz", xyz), ("thickness", thickness), ("globals", global_values)):
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise TypeError(f"{key} must be a floating-point torch.Tensor")
        if value.device != xyz.device or value.dtype != xyz.dtype:
            raise ValueError("xyz, thickness and globals must share device and dtype")
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] == 0:
        raise ValueError("xyz must have shape [N,3] with N > 0")
    n = xyz.shape[0]
    if tuple(thickness.shape) != (n, 1) or global_values.ndim != 1 or global_values.numel() < 1:
        raise ValueError("thickness must be [N,1] and globals must be nonempty [G]")
    if check_values:
        if not all(bool(torch.isfinite(v).all()) for v in (xyz, thickness, global_values)):
            raise ValueError("All inputs must be finite")
        if bool((xyz < 0).any()) or bool((xyz > 1).any()):
            raise ValueError("xyz must be normalized into [0,1]")
    if require_edges:
        edges = batch["edge_index"]
        if not isinstance(edges, torch.Tensor) or edges.dtype != torch.long:
            raise TypeError("edge_index must be a torch.long tensor")
        if edges.ndim != 2 or edges.shape[0] != 2 or edges.shape[1] == 0:
            raise ValueError("edge_index must have shape [2,E] with E > 0")
        if edges.device != xyz.device:
            raise ValueError("edge_index and xyz must share a device")
        if check_values and (int(edges.min()) < 0 or int(edges.max()) >= n):
            raise ValueError("edge_index contains a node outside [0,N)")
    return n


def forward_model(architecture: str, model, batch: dict):
    """Dispatch a validated, unbatched sample and return [1,N,out_dim].

    Call ``validate_batch(..., check_values=True)`` before timed work. Shape,
    dtype and device checks remain here without GPU-to-CPU value synchronization.
    """
    import torch

    name = _canonical(architecture)
    n = validate_batch(batch, require_edges=name == "meshgraphnet", check_values=False)
    xyz, thickness, global_values = (batch[key] for key in ("xyz", "thickness", "globals"))
    globals_per_node = global_values.reshape(1, -1).expand(n, -1)
    functional = torch.cat((thickness, globals_per_node), dim=-1)
    if name in ("transolver", "flare"):
        output = model(functional.unsqueeze(0), embedding=xyz.unsqueeze(0))
    elif name in ("geotransolver", "geoflare"):
        local = torch.cat((xyz, thickness), dim=-1)
        output = model(local.unsqueeze(0), geometry=xyz.unsqueeze(0),
                       global_embedding=global_values.reshape(1, 1, -1))
    elif name == "meshgraphnet":
        from torch_geometric.data import Data

        edges = batch["edge_index"]
        nodes = torch.cat((xyz, functional), dim=-1)
        delta = xyz[edges[0]] - xyz[edges[1]]
        edge_features = torch.cat((delta, torch.linalg.vector_norm(delta, dim=-1, keepdim=True)), dim=-1)
        graph = Data(edge_index=edges, num_nodes=n)
        output = model(nodes, edge_features, graph).unsqueeze(0)
    else:
        if n <= 8:
            raise ValueError("FIGConvUNet knn_k=8 requires more than 8 input points")
        features = torch.cat((xyz, functional), dim=-1)
        output, _ = model(xyz.unsqueeze(0), features=features.unsqueeze(0))
    if output.ndim != 3 or tuple(output.shape[:2]) != (1, n):
        raise ValueError(f"Model returned {tuple(output.shape)}; expected [1,{n},out_dim]")
    return output
