"""
probe.py — Hallucination probe classifier (student-implemented).

Implements ``HallucinationProbe``, a regularized binary classifier that
classifies feature vectors as truthful (0) or hallucinated (1).  Called from
``solution.py`` via ``evaluate.run_evaluation``.  All four public methods
(``fit``, ``fit_hyperparameters``, ``predict``, ``predict_proba``) must be
implemented and their signatures must not change.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


_QWEN_HIDDEN_DIM = 896


class HallucinationProbe(nn.Module):
    """Binary classifier that detects hallucinations from hidden-state features.

    Extends ``torch.nn.Module`` for compatibility with the template, but uses
    scikit-learn's regularized logistic regression.  This is intentionally
    lower variance than a neural probe for a 689-example dataset.
    """

    def __init__(self) -> None:
        super().__init__()
        self._scaler = StandardScaler()
        self._pca: PCA | None = None
        self._clf: LogisticRegressionCV | None = None
        self._passthrough_dim = 0
        self._clip_value = 3.0
        self._threshold: float = 0.5  # tuned by fit_hyperparameters()

    @staticmethod
    def _best_accuracy_threshold(probs: np.ndarray, y: np.ndarray) -> float:
        """Choose the threshold with best accuracy, using F1 only as a tie-break."""
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))
        best_threshold = 0.5
        best_accuracy = -1.0
        best_f1 = -1.0
        for t in candidates:
            y_pred_t = (probs >= t).astype(int)
            accuracy = float(np.mean(y_pred_t == y))
            f1 = f1_score(y, y_pred_t, zero_division=0)
            if (accuracy > best_accuracy) or (
                accuracy == best_accuracy and f1 > best_f1
            ):
                best_accuracy = accuracy
                best_f1 = f1
                best_threshold = float(t)
        return best_threshold

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe on labelled feature vectors.

        Scales features, optionally compresses them with PCA, and fits a
        cross-validated L2-regularised logistic regression.  The dataset is
        small, so a linear probe is usually a better bias than a flexible MLP:
        it tests whether hallucination is linearly exposed in the representation
        while reducing the chance of memorising fold-specific noise.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.
            y: Integer label vector of shape ``(n_samples,)``; 0 = truthful,
               1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        X_scaled = self._scaler.fit_transform(X).astype(np.float32, copy=False)
        # A light 3-sigma clip reduces fold-specific extreme activations before
        # PCA, while preserving the linear probe's interpretability.
        X_scaled = np.clip(X_scaled, -self._clip_value, self._clip_value)

        # Aggregation returns several hidden-state blocks followed by compact
        # geometric statistics. Keep the statistics explicit so PCA only
        # compresses the large representation blocks.
        self._passthrough_dim = (
            X_scaled.shape[1] % _QWEN_HIDDEN_DIM
            if X_scaled.shape[1] > _QWEN_HIDDEN_DIM
            else 0
        )
        if self._passthrough_dim:
            X_for_pca = X_scaled[:, :-self._passthrough_dim]
            X_passthrough = X_scaled[:, -self._passthrough_dim:]
        else:
            X_for_pca = X_scaled
            X_passthrough = None

        # PCA is fit inside each training fold, so validation/test information
        # never leaks into the representation used by the classifier.
        n_components = min(128, X_for_pca.shape[0] - 1, X_for_pca.shape[1])
        if n_components >= 2 and n_components < X_for_pca.shape[1]:
            self._pca = PCA(
                n_components=n_components,
                whiten=True,
                svd_solver="randomized",
                random_state=42,
            )
            X_model = self._pca.fit_transform(X_for_pca)
        else:
            self._pca = None
            X_model = X_for_pca
        if X_passthrough is not None:
            X_model = np.hstack([X_model, X_passthrough])

        min_class = int(np.bincount(y.astype(int), minlength=2).min())
        cv_folds = max(2, min(5, min_class))
        self._clf = LogisticRegressionCV(
            Cs=np.logspace(-3, 2, 10),
            cv=cv_folds,
            scoring="accuracy",
            class_weight=None,
            penalty="l2",
            solver="liblinear",
            max_iter=5000,
            random_state=42,
            refit=True,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self._clf.fit(X_model, y.astype(int))
        self._threshold = self._best_accuracy_threshold(
            self._clf.predict_proba(X_model)[:, 1],
            y.astype(int),
        )
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on validation accuracy.

        The chosen threshold is stored in ``self._threshold`` and used by
        subsequent ``predict`` calls.  Call this after ``fit`` and before
        ``predict``.

        Args:
            X_val: Validation feature matrix of shape
                   ``(n_val_samples, feature_dim)``.
            y_val: Integer label vector of shape ``(n_val_samples,)``;
                   0 = truthful, 1 = hallucinated.

        Returns:
            ``self`` (for method chaining).
        """
        probs = self.predict_proba(X_val)[:, 1]
        self._threshold = self._best_accuracy_threshold(probs, y_val)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels for feature vectors.

        Uses the decision threshold in ``self._threshold`` (default ``0.5``;
        updated by ``fit_hyperparameters``).

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Integer array of shape ``(n_samples,)`` with values in ``{0, 1}``.
        """
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates.

        Args:
            X: Feature matrix of shape ``(n_samples, feature_dim)``.

        Returns:
            Array of shape ``(n_samples, 2)`` where column 1 contains the
            estimated probability of the hallucinated class (label 1).
            Used to compute AUROC.
        """
        if self._clf is None:
            raise RuntimeError("Classifier has not been fit yet. Call fit() first.")
        X_scaled = self._scaler.transform(X).astype(np.float32, copy=False)
        X_scaled = np.clip(X_scaled, -self._clip_value, self._clip_value)
        if self._passthrough_dim:
            X_for_pca = X_scaled[:, :-self._passthrough_dim]
            X_passthrough = X_scaled[:, -self._passthrough_dim:]
        else:
            X_for_pca = X_scaled
            X_passthrough = None
        X_model = self._pca.transform(X_for_pca) if self._pca is not None else X_for_pca
        if X_passthrough is not None:
            X_model = np.hstack([X_model, X_passthrough])
        return self._clf.predict_proba(X_model)
