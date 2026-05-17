"""
aggregation.py — Token aggregation strategy and feature extraction
               (student-implemented).

Converts per-token, per-layer hidden states from the extraction loop in
``solution.py`` into flat feature vectors for the probe classifier.

Two stages can be customised independently:

  1. ``aggregate`` — select layers and token positions, pool into a vector.
  2. ``extract_geometric_features`` — optional hand-crafted features
     (enabled by setting ``USE_GEOMETRIC = True`` in ``solution.py``).

Both stages are combined by ``aggregation_and_feature_extraction``, the
single entry point called from the notebook.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _selected_layer_indices(n_layers: int) -> list[int]:
    """Choose stable middle-to-late transformer layers, excluding embeddings."""
    last_layer = n_layers - 1
    fractions = (0.33, 0.50, 0.67, 0.83, 1.00)
    indices = [max(1, min(last_layer, round(last_layer * f))) for f in fractions]
    return sorted(set(indices))


def _summary_statistics(
    layer_features: torch.Tensor,
    attention_mask: torch.Tensor,
    context_vectors: torch.Tensor,
    answer_vectors: torch.Tensor,
    local_mean_vectors: torch.Tensor,
    broad_mean_vectors: torch.Tensor,
) -> torch.Tensor:
    """Small geometric descriptors that complement the high-dimensional states."""
    valid_positions = attention_mask.nonzero(as_tuple=False).squeeze(-1)
    n_real = int(valid_positions.numel())
    seq_len = int(attention_mask.numel())

    context_norms = context_vectors.norm(dim=1)
    answer_norms = answer_vectors.norm(dim=1)
    local_norms = local_mean_vectors.norm(dim=1)
    broad_norms = broad_mean_vectors.norm(dim=1)

    context_drift = 1.0 - F.cosine_similarity(
        context_vectors[:-1], context_vectors[1:], dim=1
    )
    answer_drift = 1.0 - F.cosine_similarity(
        answer_vectors[:-1], answer_vectors[1:], dim=1
    )
    local_drift = 1.0 - F.cosine_similarity(
        local_mean_vectors[:-1], local_mean_vectors[1:], dim=1
    )
    broad_drift = 1.0 - F.cosine_similarity(
        broad_mean_vectors[:-1], broad_mean_vectors[1:], dim=1
    )

    context_updates = (context_vectors[1:] - context_vectors[:-1]).norm(dim=1)
    answer_updates = (answer_vectors[1:] - answer_vectors[:-1]).norm(dim=1)
    local_updates = (local_mean_vectors[1:] - local_mean_vectors[:-1]).norm(dim=1)
    broad_updates = (broad_mean_vectors[1:] - broad_mean_vectors[:-1]).norm(dim=1)
    answer_norm_deltas = answer_norms[1:] - answer_norms[:-1]

    layer_variance = layer_features.var(dim=1, unbiased=False).mean(dim=1)
    variance_deltas = layer_variance[1:] - layer_variance[:-1]

    token_updates = (layer_features[1:] - layer_features[:-1]).norm(dim=2)
    token_instability = token_updates.mean(dim=0)
    instability_stats = torch.stack(
        [
            token_instability.mean(),
            token_instability.std(unbiased=False),
            token_instability.max(),
            torch.quantile(token_instability.float(), 0.90).to(layer_features.dtype),
        ]
    )

    stats = [
        torch.tensor(
            [
                n_real / max(seq_len, 1),
                min(n_real, 512) / 512.0,
            ],
            dtype=layer_features.dtype,
            device=layer_features.device,
        ),
        context_norms,
        answer_norms,
        local_norms,
        broad_norms,
        context_drift,
        answer_drift,
        local_drift,
        broad_drift,
        context_updates,
        answer_updates,
        local_updates,
        broad_updates,
        answer_norm_deltas,
        layer_variance,
        variance_deltas,
        instability_stats,
    ]
    return torch.cat(stats, dim=0)


def _tail_window(
    positions: torch.Tensor,
    selected: torch.Tensor,
    size: int,
) -> torch.Tensor:
    """Return selected-layer states for the final ``size`` valid positions."""
    window_size = min(size, int(positions.numel()))
    window_positions = positions[-window_size:].to(selected.device)
    return selected.index_select(dim=1, index=window_positions)


def aggregate(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert per-token hidden states into a single feature vector.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
                        Layer index 0 is the token embedding; index -1 is the
                        final transformer layer.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D feature tensor of shape ``(hidden_dim,)`` or
        ``(k * hidden_dim,)`` if multiple layers are concatenated.

    Student task:
        Replace or extend the skeleton below with alternative layer selection,
        token pooling (mean, max, weighted), or multi-layer fusion strategies.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the aggregation below.
    # ------------------------------------------------------------------

    real_positions = attention_mask.nonzero(as_tuple=False).squeeze(-1)
    last_pos = int(real_positions[-1].item())

    # The final token is usually the explicit EOS marker in this benchmark.
    # Keep it as a context summary, but pool answer-local windows before it so
    # localized factuality signals are not diluted by the prompt or EOS token.
    answer_positions = real_positions[:-1] if real_positions.numel() > 1 else real_positions
    answer_last_pos = int(answer_positions[-1].item())

    layer_indices = _selected_layer_indices(hidden_states.shape[0])
    selected = hidden_states[layer_indices]  # (k, seq_len, hidden_dim)
    local_window = _tail_window(answer_positions, selected, 8)
    mid_window = _tail_window(answer_positions, selected, 32)
    broad_window = _tail_window(answer_positions, selected, 96)

    context_vectors = selected[:, last_pos, :]          # (k, hidden_dim)
    answer_vectors = selected[:, answer_last_pos, :]    # (k, hidden_dim)
    local_mean_vectors = local_window.mean(dim=1)       # (k, hidden_dim)
    mid_mean_vectors = mid_window.mean(dim=1)           # (k, hidden_dim)
    broad_mean_vectors = broad_window.mean(dim=1)       # (k, hidden_dim)
    stats = _summary_statistics(
        broad_window,
        attention_mask,
        context_vectors,
        answer_vectors,
        local_mean_vectors,
        broad_mean_vectors,
    )

    return torch.cat(
        [
            context_vectors.flatten(),
            answer_vectors.flatten(),
            local_mean_vectors.flatten(),
            mid_mean_vectors.flatten(),
            broad_mean_vectors.flatten(),
            stats,
        ],
        dim=0,
    )
    # ------------------------------------------------------------------


def extract_geometric_features(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Extract hand-crafted geometric / statistical features from hidden states.

    Called only when ``USE_GEOMETRIC = True`` in ``solution.ipynb``.  The
    returned tensor is concatenated with the output of ``aggregate``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.

    Returns:
        A 1-D float tensor of shape ``(n_geometric_features,)``.  The length
        must be the same for every sample.

    Student task:
        Replace the stub below.  Possible features: layer-wise activation
        norms, inter-layer cosine similarity (representation drift), or
        sequence length.
    """
    # ------------------------------------------------------------------
    # STUDENT: Replace or extend the geometric feature extraction below.
    # ------------------------------------------------------------------

    real_positions = attention_mask.nonzero(as_tuple=False).squeeze(-1)
    last_pos = int(real_positions[-1].item())
    answer_positions = real_positions[:-1] if real_positions.numel() > 1 else real_positions
    answer_last_pos = int(answer_positions[-1].item())
    layer_indices = _selected_layer_indices(hidden_states.shape[0])
    selected = hidden_states[layer_indices]
    local_window = _tail_window(answer_positions, selected, 8)
    broad_window = _tail_window(answer_positions, selected, 96)
    return _summary_statistics(
        broad_window,
        attention_mask,
        selected[:, last_pos, :],
        selected[:, answer_last_pos, :],
        local_window.mean(dim=1),
        broad_window.mean(dim=1),
    )


def aggregation_and_feature_extraction(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    use_geometric: bool = False,
) -> torch.Tensor:
    """Aggregate hidden states and optionally append geometric features.

    Main entry point called from ``solution.ipynb`` for each sample.
    Concatenates the output of ``aggregate`` with that of
    ``extract_geometric_features`` when ``use_geometric=True``.

    Args:
        hidden_states:  Tensor of shape ``(n_layers, seq_len, hidden_dim)``
                        for a single sample.
        attention_mask: 1-D tensor of shape ``(seq_len,)`` with 1 for real
                        tokens and 0 for padding.
        use_geometric:  Whether to append geometric features.  Controlled by
                        the ``USE_GEOMETRIC`` flag in ``solution.ipynb``.

    Returns:
        A 1-D float tensor of shape ``(feature_dim,)`` where
        ``feature_dim = hidden_dim`` (or larger for multi-layer or geometric
        concatenations).
    """
    agg_features = aggregate(hidden_states, attention_mask)  # (feature_dim,)

    if use_geometric:
        geo_features = extract_geometric_features(hidden_states, attention_mask)
        return torch.cat([agg_features, geo_features], dim=0)

    return agg_features
