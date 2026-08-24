# Minimal Jacobian Lens on Qwen

Project ID: `jacobian-lens-qwen-minimal`
Status: runnable initial workspace

This workspace reproduces the **Jacobian-lens computation path**, at deliberately
small scale, on the open-weight
[`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
model. It fits an average downstream Jacobian from a local calibration corpus,
then compares the J-lens and logit lens layer by layer on one fixed two-hop case.

本项目只声称复现方法与数据管线，不以一个小模型、六条校准文本和一个评测样例证明
“global workspace” 的存在。若要复现论文的结构性结论，需要更大的模型、约千条序列、
更多随机投影、跨任务评测与因果干预。

## Exact experiment contract

| Choice | Pinned value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Hugging Face revision | `c89bee90d9f811437d9735454613c35b4a3c4dc8` |
| Model license | Apache-2.0 |
| Weight dtype | `float32` |
| Attention implementation | Hugging Face `eager` (portable reference path) |
| Calibration data | 6 local, synthetic English snippets in `data/calibration.jsonl` |
| Evaluation data | 1 local, synthetic two-hop case in `data/evaluation.jsonl` |
| Seed | `20260824` |
| Jacobian estimator | 4 Rademacher/VJP samples per calibration prompt |
| Tokenization | raw text, `add_special_tokens=false`, `max_length=64` |
| Analysis position | final prompt token |
| Output | JSON, Markdown, PyTorch lens artifact, SHA-256 manifest |

All scientific choices live in [`configs/minimal.json`](configs/minimal.json).
The model revision and every input file hash are also copied into the run output.

## Setup and run

Requirements: macOS/Linux, Python 3.9–3.12, about 2.5 GB free disk space, and
network access for the first model download. CUDA, Apple Silicon MPS, and CPU are
selected in that order. CPU is valid but slower.

```bash
./scripts/setup.sh
./scripts/run_minimal.sh
```

The first command creates `.venv` and installs exact dependency versions. The
second downloads the pinned model revision if necessary, fits the lens, and writes:

```text
outputs/minimal/
  jlens.pt              # low-rank Hutchinson/VJP lens artifact
  fit_summary.json      # fit samples and provenance
  evaluation.json       # complete machine-readable layer results
  evaluation.md         # compact human-readable result
  manifest.sha256       # portable relative-path hashes of code, inputs, and results
```

Run the local tests after setup:

```bash
.venv/bin/python -m pytest
```

To force CPU, or use a separate output directory without changing the scientific
configuration:

```bash
JLENS_DEVICE=cpu JLENS_OUTPUT_DIR=outputs/cpu ./scripts/run_minimal.sh
```

## What is computed

For layer `l`, source position `t`, and later target position `t'`, the paper's
average transport is

```text
J_l = E[prompt,t,t' >= t] [ d h_final[t'] / d h_l[t] ]
J-lens(h_l) = W_U norm(J_l h_l)
```

This project estimates each matrix with Rademacher vectors `v` and vector-Jacobian
products `g = J_l^T v`:

```text
J_l h ~= mean_s v_s (g_s dot h)
```

Saving `v_s` and `g_s` is mathematically the same estimator as saving the summed
dense outer products `mean_s(v_s g_s^T)`, while avoiding large dense matrices.
Source/target positions are seeded causal samples, and their exact indices are
recorded in `fit_summary.json`.

## Initial evaluation case

The sole case is:

```text
A spider has eight legs. Therefore, two spiders have
```

The expected next-token concept is `" sixteen"`; the two tracked intermediate concepts
are `" spider"` and `" legs"`. The runner first verifies that each surface form is
exactly one Qwen vocabulary token. It then reports, for every layer:

- J-lens and logit-lens top-10 vocabulary tokens;
- exact vocabulary rank for each tracked concept;
- best rank in the predefined middle third of layers;
- final-layer rank of the expected answer token.

The only automatic sanity criterion is `final_answer_rank <= 10`. The intermediate
rank comparison is exploratory: with this tiny calibration set, either lens may
win. A failed sanity criterion is recorded in the report but does not erase the
artifacts.

## Interpretation limits

- A J-lens readout is a first-order, corpus-averaged disposition to affect later
  verbal output; it is not a transcript of hidden thought.
- Decodability alone is not causality for this prompt. This minimal workspace does
  not perform concept swaps or ablations.
- The paper averages a much larger and more diverse corpus. Four projections per
  prompt are useful for a smoke-sized reproduction, not a stable scientific lens.
- Hardware kernels can produce small floating-point differences even with fixed
  seeds. The report records device and library versions to make them auditable.

## Workspace map

- `configs/`: all model, estimator, and evaluation choices.
- `data/`: committed synthetic calibration/evaluation inputs; no remote dataset.
- `src/jlens_qwen/`: fitting, transport, reporting, and CLI implementation.
- `scripts/`: executable setup and end-to-end run paths.
- `tests/`: offline unit tests for the estimator and causal-pair sampler.
- `results/RESULTS_TEMPLATE.md`: experiment log template for subsequent runs.
- `Progress.md`: current state and tightly scoped next steps.

## References

- Gurnee et al., *Verbalizable Representations Form a Global Workspace in Language
  Models* (2026), [arXiv:2607.15495](https://arxiv.org/abs/2607.15495).
- Qwen model card and pinned files:
  [Qwen2.5-0.5B-Instruct at the selected revision](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/c89bee90d9f811437d9735454613c35b4a3c4dc8).
