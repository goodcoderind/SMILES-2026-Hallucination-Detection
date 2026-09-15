# Hallucination Detection with Qwen Hidden States

**SMILES 2026 entrance technical challenge · Abhra Kanti Dubey**

A lightweight classifier that predicts whether a language model's response is truthful or hallucinated from its internal representations. My solution combines features from a frozen Qwen2.5-0.5B model with dimensionality reduction and regularized logistic regression.

**Recorded result: 75.32% mean five-fold cross-validation accuracy on 689 labeled examples**, compared with a 70.10% majority-class baseline.

[Solution report](SOLUTION.md) · [Saved results](results.json) · [Feature extraction](aggregation.py) · [Classifier](probe.py)

## My contribution

I implemented the three solution components allowed by the challenge:

| Component | Implementation |
|---|---|
| [aggregation.py](aggregation.py) | Multi-layer token pooling and compact statistics describing changes in hidden states |
| [probe.py](probe.py) | Feature scaling, clipping, PCA, L2-regularized logistic regression, and threshold selection |
| [splitting.py](splitting.py) | Stratified five-fold evaluation with a separate validation subset in each fold |

The challenge provides the model loader, evaluation code, datasets, and execution script. The [solution report](SOLUTION.md) records the design choices, ablations, and unsuccessful experiments.

## Results

The committed [results.json](results.json) contains these averages over held-out folds of the labeled dataset:

| Metric | Majority baseline | My probe |
|---|---:|---:|
| Accuracy | 70.10% | **75.32%** |
| F1, hallucinated class | 82.42% | 83.08% |
| AUROC | — | 75.30% |

The accuracy improvement is **5.22 percentage points**. These are recorded cross-validation results; the hidden competition-test labels are unavailable.

## How it works

1. Feed each `prompt + response` to the frozen **Qwen/Qwen2.5-0.5B** model.
2. Read hidden states from layers **8, 12, 16, 20, and 24**. Extract the final two token states separately and mean-pool windows of 8, 32, and 96 tokens preceding the final token.
3. Add **71 statistics** covering activation norms, cross-layer drift, update magnitudes, token variance, instability, and sequence length.
4. Standardize and clip features to `[-3, 3]`. Compress the 22,400 pooled hidden-state features into **128 PCA components**, retaining the 71 statistics: **22,471 extracted features → 199 classifier inputs**.
5. Fit L2-regularized logistic regression and tune its decision threshold on the validation subset.

Scaling and PCA are fitted on the training portion of each evaluation fold. The final submission model is fitted on all 689 labeled examples.

## Run locally

Use **Python 3.10+**. The first run downloads Qwen model weights from Hugging Face; the script selects CUDA, Apple MPS, or CPU automatically. The loader uses `bfloat16`, so the selected PyTorch backend must support it.

```bash
git clone https://github.com/goodcoderind/SMILES-2026-Hallucination-Detection.git
cd SMILES-2026-Hallucination-Detection
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python solution.py
```

On Windows, activate the environment with `.venv\Scripts\activate.bat` in Command Prompt.

The run reads [data/dataset.csv](data/dataset.csv) and [data/test.csv](data/test.csv), evaluates the probe, and writes:

- `results.json` — per-fold metrics and averages; overwrites the saved result file.
- `predictions.csv` — predictions for the 100 unlabeled competition examples, with `id` and `label` columns.

Labels: **`0` = truthful**, **`1` = hallucinated**. Dependencies in [requirements.txt](requirements.txt) specify minimum versions rather than a locked environment; exact results may vary across software and hardware setups.

## Limitations

- **Small, imbalanced dataset:** results cover 689 examples from this benchmark; broader generalization has not been established.
- **Approximate response pooling:** aggregation receives no prompt/response boundary. Tail windows may include prompt tokens, and the final token is assumed to be an end-of-sequence marker.
- **Truncation:** the provided tokenizer path limits inputs to 512 tokens, which can remove response content.
- **Model selection:** variants were compared using cross-validation; a separate untouched test set would provide a stronger final estimate.
- **Final threshold:** the submission model chooses its threshold from its training predictions, while evaluation folds use separate validation data.

## Challenge and attribution

Built on [ahdr3w/SMILES-HALLUCINATION-DETECTION](https://github.com/ahdr3w/SMILES-HALLUCINATION-DETECTION). The original task instructions are preserved in [docs/CHALLENGE.md](docs/CHALLENGE.md).

See the [MIT license](LICENSE), which retains the upstream copyright notice, and [SOLUTION.md](SOLUTION.md) for the research references that informed the method.
