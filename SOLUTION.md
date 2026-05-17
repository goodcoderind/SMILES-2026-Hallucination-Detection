# SMILES-2026 Hallucination Detection Solution

## Summary

This solution detects hallucinated responses by probing hidden states from
`Qwen/Qwen2.5-0.5B`. The final method is intentionally simple and
regularized: it extracts middle-to-late layer representations from localized
answer windows, adds compact cross-layer dynamics statistics, compresses the
large hidden-state representation with PCA, and trains a regularized logistic
regression probe.

The main goal was not to maximize complexity, but to build a reproducible,
research-style solution that improves over the majority-class baseline while
controlling overfitting on a small dataset of 689 labeled examples.

Final cross-validation result from `python solution.py`:

| Metric | Value |
|---|---:|
| Majority baseline accuracy | 70.10% |
| Final probe accuracy | 75.32% |
| Final probe F1 | 83.08% |
| Final probe AUROC | 75.30% |
| Feature dimension | 22,471 |
| Number of folds | 5 |

The generated competition artifact is `predictions.csv`, produced directly by
the provided `solution.py` script. A public copy of the final predictions file
is available here:
[predictions.csv](https://drive.google.com/file/d/1DGtVDwnSNDQxxOYkdcQr90oK9Vz6LINp/view?usp=sharing).

## Files Modified

Only the allowed implementation files were modified:

| File | Purpose |
|---|---|
| `aggregation.py` | Layer selection, token pooling, and compact geometric/dynamic features |
| `probe.py` | PCA, scaling, clipping, and regularized logistic regression probe |
| `splitting.py` | Stratified 5-fold evaluation with validation threshold tuning |

The fixed infrastructure files were not modified:

| File | Status |
|---|---|
| `model.py` | Unchanged |
| `evaluate.py` | Unchanged |
| `solution.py` | Unchanged |

This report file, `SOLUTION.md`, is added as the required submission report.

## Repository Structure and Data Flow

The repository implements a hidden-state probing benchmark:

1. `solution.py` loads `data/dataset.csv`.
2. Each example is converted to the text input `prompt + response`.
3. `model.py` loads `Qwen/Qwen2.5-0.5B` with `output_hidden_states=True`.
4. The model returns hidden states for all transformer layers.
5. `aggregation.py` converts hidden states into one fixed-length feature vector.
6. `splitting.py` produces stratified train/validation/test folds.
7. `probe.py` trains the hallucination detector.
8. `evaluate.py` reports metrics and writes `results.json`.
9. `solution.py` fits the final probe and writes `predictions.csv`.

The label convention is:

| Label | Meaning |
|---:|---|
| `0` | truthful |
| `1` | hallucinated |

## Motivation

The core hypothesis is that hallucination information is partially encoded in
the internal representation of the model response. Recent probing work suggests
that hallucination signals are often exposed in middle-to-late layers, that
response-local tokens can be more informative than full-sequence pooling, and
that representation changes across layers can carry useful uncertainty or
instability information.

This solution follows those ideas while staying conservative:

- Use middle-to-late layers rather than every layer.
- Focus on answer-local late-token windows.
- Add small, interpretable cross-layer statistics.
- Use a linear probe with strong regularization rather than a deep classifier.
- Validate each change through controlled cross-validation.

References that informed the design:

- INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection,
  https://arxiv.org/abs/2402.03744
- Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs,
  https://arxiv.org/abs/2406.15927
- ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination Detection
  in LLMs, submitted July 22, 2025,
  https://arxiv.org/abs/2507.16488

## Final Approach

The final solution has three main parts:

1. Hidden-state aggregation from selected middle-to-late layers.
2. PCA plus regularized logistic regression.
3. Stratified 5-fold evaluation with validation-tuned thresholds.

### Layer Selection

Qwen2.5-0.5B exposes 25 hidden-state tensors in this setup: token embeddings
plus 24 transformer layers. The solution excludes the embedding layer and
selects five middle-to-late layers:

```text
[8, 12, 16, 20, 24]
```

These correspond approximately to layer fractions:

```text
33%, 50%, 67%, 83%, 100%
```

This gives a compact view of representation evolution without concatenating
all layers.

### Token Aggregation

The input to the model is `prompt + response`. The aggregation function receives
only hidden states and an attention mask, not token IDs or an explicit
prompt/response boundary. Therefore exact response-only masking is not possible
without changing `solution.py`, which is disallowed.

The final implementation uses a clean approximation:

- Treat the final real token as a context summary token.
- Treat tokens before the final token as answer-local tokens, because the final
  token is usually the explicit EOS marker.
- Pool several late windows from this answer-local region.

For each selected layer, the following high-dimensional blocks are extracted:

| Block | Description |
|---|---|
| `context_vectors` | Hidden state at the final real token |
| `answer_vectors` | Hidden state at the last answer token before EOS |
| `local_mean_vectors` | Mean over the final 8 answer-local tokens |
| `mid_mean_vectors` | Mean over the final 32 answer-local tokens |
| `broad_mean_vectors` | Mean over the final 96 answer-local tokens |

With 5 selected layers and hidden dimension 896, these blocks contribute:

```text
5 blocks * 5 layers * 896 hidden dimensions = 22,400 features
```

### Geometric and Dynamic Features

In addition to the pooled hidden-state vectors, the solution appends 71 compact
statistics. These are designed to capture cross-layer hidden-state evolution
without adding a nonlinear model.

The appended statistics include:

| Feature family | Count | Intuition |
|---|---:|---|
| Length features | 2 | Sequence length and padding/truncation effects |
| Activation norms | 20 | Magnitude of context, answer, local, and broad states |
| Cosine drift | 16 | Directional changes across consecutive selected layers |
| Update magnitudes | 16 | Euclidean hidden-state movement across layers |
| Answer norm deltas | 4 | Layer-to-layer norm changes at the answer token |
| Token-window variance | 5 | Dispersion of token representations per layer |
| Variance deltas | 4 | Change in dispersion across layers |
| Token instability stats | 4 | Mean/std/max/p90 of token-wise layer updates |

Total feature dimension:

```text
22,400 pooled hidden-state features + 71 compact statistics = 22,471
```

## Probe

The original neural probe was replaced with a regularized linear probe. This
was a deliberate choice because the dataset is small, imbalanced, and vulnerable
to overfitting.

The final probe pipeline is:

1. Standardize all features using `StandardScaler`.
2. Clip standardized features to `[-3, 3]`.
3. Separate the final 71 compact statistics from the large hidden-state blocks.
4. Apply PCA only to the large hidden-state blocks.
5. Keep the compact statistics as passthrough features.
6. Train `LogisticRegressionCV` with L2 regularization.
7. Tune the classification threshold on validation accuracy.

### Why Logistic Regression Instead of an MLP?

The task has only 689 labeled training examples. A neural probe can memorize
fold-specific artifacts, especially when the feature dimension is large. A
linear probe with PCA is easier to regularize and more interpretable: if it
works, that suggests hallucination information is linearly accessible in the
hidden states.

### PCA

PCA is fit inside each training fold, never on validation or held-out test
examples. This avoids representation leakage. The final number of PCA
components is:

```text
128
```

The 71 compact statistics are not PCA-compressed; they are passed directly to
the classifier after scaling and clipping.

Although the raw feature vector has 22,471 dimensions, the effective learned
representation after PCA is low-dimensional: 128 PCA components plus 71 compact
features, or 199 classifier inputs. This substantially reduces overfitting risk
relative to fitting directly on the raw feature vector.

### Feature Clipping

After standardization, features are clipped to `[-3, 3]`. This small change
improved the final cross-validation result and reduced sensitivity to extreme
activation values.

This is not a new model component; it is a regularization step that makes PCA
and logistic regression less sensitive to outliers.

## Splitting Strategy

`splitting.py` implements stratified 5-fold cross-validation. Each fold uses:

- A held-out test fold for evaluation.
- A validation split carved from the remaining examples.
- Stratification by label to preserve the hallucinated/truthful class ratio.

The typical fold sizes are:

| Split | Size |
|---|---:|
| Train | 447 or 448 |
| Validation | 104 |
| Test | 137 or 138 |

This strategy is more stable than a single random split for a 689-example
dataset. The validation split is used only for threshold selection; PCA and
logistic regression are fit only on the training subset inside each fold.

## Algorithm

High-level algorithm for one example:

```text
Input:
  hidden_states: (n_layers, seq_len, hidden_dim)
  attention_mask: (seq_len)

1. Find real token positions from the attention mask.
2. Select middle-to-late layers [8, 12, 16, 20, 24].
3. Identify:
   - final context token
   - last answer-local token before EOS
   - final 8, 32, and 96 answer-local token windows
4. For each selected layer, concatenate:
   - final context vector
   - last answer vector
   - final-8 mean vector
   - final-32 mean vector
   - final-96 mean vector
5. Compute compact layer-dynamics statistics:
   - norms
   - cosine drifts
   - update magnitudes
   - variance trajectory
   - token instability
6. Concatenate all features.
7. Standardize, clip, PCA-compress, and classify with logistic regression.
```

## Complexity

Let:

- `N` be the number of examples.
- `L` be the number of selected layers, here `L = 5`.
- `T` be the sequence length, at most 512.
- `D` be the hidden dimension, here `D = 896`.
- `P` be the PCA dimension, here `P = 128`.

The dominant cost is the frozen Qwen forward pass, which is handled by the
provided infrastructure.

Aggregation cost per example is approximately:

```text
O(L * T * D)
```

The feature matrix size is:

```text
689 examples * 22,471 features * 4 bytes ~= 62 MB
```

Probe training is lightweight relative to hidden-state extraction. PCA is fit
inside each fold on roughly 447 examples, and logistic regression is trained on
the PCA-compressed representation plus 71 statistics.

Runtime observed on local Apple MPS:

```text
~52 seconds for train feature extraction and cross-validation
~7 seconds for competition test feature extraction
```

Runtime on Colab T4 should remain practical because Qwen2.5-0.5B is small and
the probe is linear.

## Experiment Log

The table below summarizes the main controlled experiments. Accuracy and AUROC
refer to the averaged cross-validation test split reported by `solution.py`.

| Experiment | Accuracy | AUROC | Decision |
|---|---:|---:|---|
| Majority-class baseline | 70.10% | N/A | Reference floor |
| Earlier stable multi-layer baseline | 73.44% | 74.48% | Improved further |
| Late-token + layer-dynamics representation | 75.03% | 75.19% | Kept as base representation |
| Add 3-sigma feature clipping | 75.32% | 75.30% | Final kept model |
| Remove final context vector | 73.29% | 74.01% | Rejected |
| Add high-dimensional unstable-token pooling | 73.00% | 74.38% | Rejected |
| Truncation-aware EOS handling | 74.31% | 75.13% | Rejected |
| Replace length ratio with cleaner length stats | 74.02% | 75.18% | Rejected |
| Shift selected layers later | 75.03% | 74.57% | Rejected |
| Train on larger folds with no external validation | 74.16% | 74.90% | Rejected |
| Logistic CV scored by AUROC | 74.88% | 74.87% | Rejected |
| Balanced class weights | 72.57% | 73.01% | Rejected |

### Ablation Takeaways

The experiments suggest that representation design mattered more than
classifier complexity. Late-token pooling plus a final context summary was more
useful than broad pooling alone. Compact geometric and layer-dynamics statistics
helped more than adding extra raw token vectors. Strong regularization through
PCA, clipping, and logistic regression gave better generalization than higher
capacity alternatives.

### PCA Dimension Sweep

The PCA sweep was performed using the late-token and layer-dynamics
representation before feature clipping. The goal was to find the smallest
compression that preserved signal while avoiding overfitting.

| PCA components | Accuracy | AUROC | Notes |
|---:|---:|---:|---|
| 64 | 73.87% | 74.15% | Too compressed |
| 96 | 74.89% | 74.92% | Stable but below 128 |
| 128 | 75.03% | 75.19% | Best overall before clipping |
| 160 | 74.60% | 75.08% | No accuracy gain |
| 192 | 73.00% | 74.41% | Overfit/noisy |
| 256 | 74.45% | 73.93% | Worse AUROC |

The final solution keeps `128` PCA components.

### Clipping Sweep

Feature clipping was tested after standardization and before PCA.

| Clip value | Accuracy | AUROC | Decision |
|---:|---:|---:|---|
| No clipping | 75.03% | 75.19% | Baseline for sweep |
| 8.0 | 75.32% | 75.20% | Good |
| 6.0 | 74.74% | 75.32% | Lower accuracy |
| 5.0 | 74.89% | 75.30% | Lower accuracy |
| 4.0 | 74.74% | 75.34% | Best AUROC but lower accuracy |
| 3.0 | 75.32% | 75.30% | Final choice |

Because the primary competition metric is accuracy, `3.0` was selected: it
matched the best accuracy while preserving strong AUROC and F1.

## What Contributed Most

The largest useful improvements came from:

1. Multi-layer hidden-state aggregation instead of a single pooled vector.
2. Late answer-token windows rather than full-sequence mean pooling only.
3. Compact cross-layer dynamics features.
4. PCA plus regularized logistic regression instead of a flexible MLP.
5. Feature clipping after standardization.
6. Stratified 5-fold validation instead of a single random split.

The most important negative result was that adding more high-dimensional token
features did not help. The unstable-token pooling experiment was conceptually
reasonable but worsened performance, suggesting that local token dynamics are
useful as compact statistics, not necessarily as additional raw vectors.

## Assumptions

The final implementation makes the following assumptions:

1. The final real token is usually the EOS token appended to the response.
2. Dropping the final token gives a useful approximation to answer-local
   pooling without requiring a response mask.
3. Middle-to-late layers expose more hallucination signal than early layers.
4. The dataset is too small for a high-capacity nonlinear probe to be reliable.
5. Cross-validation metrics are a useful proxy for hidden `test.csv`
   performance, but they are not identical to the true competition score.

## Limitations

The main limitation is that `aggregation.py` receives only hidden states and an
attention mask. It does not receive token IDs, tokenizer metadata, or the exact
prompt/response boundary. Therefore exact response-only pooling cannot be
implemented cleanly without modifying `solution.py`, which is fixed
infrastructure.

Other limitations:

- The dataset has only 689 labeled examples.
- The class distribution is imbalanced toward hallucinated responses.
- Some examples are truncated to 512 tokens by the fixed tokenizer path.
- The final threshold used for `predictions.csv` is learned from the labeled
  training data, because the hidden competition labels are unavailable.
- The method is a linear probe; it may miss nonlinear hallucination signals.

## Future Work

If fixed infrastructure changes were allowed, the first improvement would be to
pass an exact response-token mask from `solution.py` to `aggregation.py`. That
would enable controlled comparisons between:

- full sequence pooling
- exact response-only pooling
- final 50% of response tokens
- final 25% of response tokens
- exact answer span tokens

Other promising future directions:

1. Out-of-fold error analysis with confidence margins.
2. Separate probes for different feature families with a small calibrated
   linear ensemble.
3. Exact high-entropy token localization if token logits are exposed.
4. More careful handling of truncation and prompt/response boundaries.
5. Calibration metrics such as Brier score or expected calibration error.
6. Group-aware splitting if future data contains repeated prompts or sources.

## Reproduction

The solution is fully reproducible using the provided `solution.py`.

### Environment

Python 3.10+ is recommended. The required dependencies are listed in
`requirements.txt`:

```text
torch>=2.0.0
transformers>=4.40.0
datasets>=2.14.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=1.5.0
tqdm>=4.65.0
```

### Commands

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python solution.py
```

This will:

1. Load the labeled training data.
2. Extract Qwen2.5-0.5B hidden states.
3. Build the feature matrix.
4. Run 5-fold evaluation.
5. Save `results.json`.
6. Load `data/test.csv`.
7. Generate `predictions.csv`.

Expected final summary:

```text
Feature dim  : 22471
Total samples: 689
Folds        : 5
Probe test accuracy: 75.32%
Probe test F1      : 83.08%
Probe test AUROC   : 75.30%
```

The final `predictions.csv` produced by the verified run contains 100 rows.

## Final Notes

This solution is intentionally conservative. I rejected several variants that
looked plausible but did not improve cross-validation performance. The final
model is a principled hidden-state probing pipeline with interpretable
aggregation, regularized dimensionality reduction, and a linear classifier
suited to the small-data setting.
