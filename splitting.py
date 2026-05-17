"""
splitting.py — Train / validation / test split utilities (student-implementable).

``split_data`` receives the label array ``y`` and, optionally, the full
DataFrame ``df`` (for group-aware splits).  It must return a list of
``(idx_train, idx_val, idx_test)`` tuples of integer index arrays.

Contract
--------
* ``idx_train``, ``idx_val``, ``idx_test`` are 1-D NumPy arrays of integer
  indices into the full dataset.
* ``idx_val`` may be ``None`` if no separate validation fold is needed.
* All indices must be non-overlapping; together they must cover every sample.
* Return a **list** — one element for a single split, K elements for k-fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Split dataset indices into train, validation, and test subsets.

    The default strategy performs stratified k-fold evaluation.  Each fold has
    a held-out test fold plus a small validation split carved from the
    remaining samples for threshold tuning.  This is more stable than a single
    random split on a 689-example dataset, and ``solution.py`` will train the
    final predictor on all labelled examples because every row appears in a
    train/validation partition in at least one fold.

    Args:
        y:            Label array of shape ``(N,)`` with values in ``{0, 1}``.
                      Used for stratification.
        df:           Optional full DataFrame (same row order as ``y``).
                      Required for group-aware splits.
        test_size:    Fraction of samples reserved for the held-out test set.
        val_size:     Fraction of samples reserved for validation.
        random_state: Random seed for reproducible splits.
        n_splits:     Number of stratified folds to evaluate.

    Returns:
        A list of ``(idx_train, idx_val, idx_test)`` tuples of integer index
        arrays.  ``idx_val`` may be ``None``.

    Student task:
        Replace or extend the skeleton below.  The only contract is that the
        function returns the list described above.
    """

    idx = np.arange(len(y))
    min_class = int(np.bincount(y.astype(int), minlength=2).min())
    n_splits = max(2, min(n_splits, min_class))

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []
    for fold, (idx_train_val, idx_test) in enumerate(splitter.split(idx, y)):
        # Keep the validation fraction comparable to the single-split baseline:
        # val_size is interpreted as a fraction of the full dataset.
        relative_val = min(0.5, val_size / (1.0 - (1.0 / n_splits)))
        idx_train, idx_val = train_test_split(
            idx_train_val,
            test_size=relative_val,
            random_state=random_state + fold,
            stratify=y[idx_train_val],
        )
        splits.append((idx_train, idx_val, idx_test))

    return splits
