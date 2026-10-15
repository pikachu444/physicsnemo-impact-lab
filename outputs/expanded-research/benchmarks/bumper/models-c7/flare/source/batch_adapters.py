"""Native batch layouts only; framework imports occur after explicit GPU allocation."""


def validate(batch, *, require_edges=False):
    import torch

    xyz, scalar, globals_ = (batch[k] for k in ("xyz", "scalar", "globals"))
    assert xyz.ndim == 3 and xyz.shape[-1] == 3
    b, n, _ = xyz.shape
    assert b > 0 and n > 0 and scalar.shape == (b, n, 1)
    assert globals_.ndim == 2 and globals_.shape[0] == b and globals_.shape[1] > 0
    assert all(v.dtype == torch.float32 and v.device == xyz.device for v in (xyz, scalar, globals_))
    if require_edges:
        edges = batch["edge_index"]
        assert edges.ndim == 2 and edges.shape[0] == 2 and edges.shape[1] > 0
        assert edges.dtype == torch.long and edges.device == xyz.device
    return b, n


def single(batch, index):
    """The original main adapter's B1 input contract, without copying topology."""
    return {"xyz": batch["xyz"][index], "thickness": batch["scalar"][index],
            "globals": batch["globals"][index], "edge_index": batch["edge_index"]}


def subset(batch, indices):
    return {k: v if k == "edge_index" else v[indices].contiguous()
            for k, v in batch.items()}


def forward_batched(architecture, model, batch):
    import torch

    b, n = validate(batch, require_edges=architecture == "meshgraphnet")
    xyz, scalar, globals_ = (batch[k] for k in ("xyz", "scalar", "globals"))
    per_node = globals_[:, None, :].expand(-1, n, -1)
    functional = torch.cat((scalar, per_node), dim=-1)
    if architecture in ("transolver", "flare"):
        output = model(functional, embedding=xyz)
    elif architecture in ("geotransolver", "geoflare"):
        local = torch.cat((xyz, scalar), dim=-1)
        output = model(local, geometry=xyz, global_embedding=globals_[:, None, :])
    elif architecture == "meshgraphnet":
        from torch_geometric.data import Data

        edges = batch["edge_index"]
        # Preserve every directed FEM edge; disconnected graph blocks cannot mix cases.
        offsets = torch.arange(b, device=xyz.device)[:, None, None] * n
        batched_edges = (edges[None] + offsets).permute(1, 0, 2).reshape(2, -1).contiguous()
        delta = xyz[:, edges[0], :] - xyz[:, edges[1], :]
        edge_features = torch.cat((delta, torch.linalg.vector_norm(delta, dim=-1, keepdim=True)), -1)
        nodes = torch.cat((xyz, functional), -1).reshape(b * n, -1)
        graph = Data(edge_index=batched_edges, num_nodes=b * n)
        output = model(nodes, edge_features.reshape(-1, 4), graph).reshape(b, n, -1)
    elif architecture == "figconvunet":
        assert n > 8, "Native knn_k=8 requires at least nine points"
        # Native batched_neighbor_knn_search already offsets each reference by b*N.
        features = torch.cat((xyz, functional), -1).contiguous()
        output, _ = model(xyz.contiguous(), features=features)
    else:
        raise ValueError(f"Unsupported original architecture: {architecture}")
    assert output.ndim == 3 and output.shape[:2] == (b, n), tuple(output.shape)
    return output
