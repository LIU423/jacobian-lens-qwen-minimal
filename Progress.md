# Progress

## 2026-08-24 — Initial workspace

- [x] Pin the open-weight model and its immutable Hugging Face revision.
- [x] Commit all calibration and evaluation text locally.
- [x] Implement low-rank Hutchinson/VJP fitting for the average Jacobian lens.
- [x] Add layerwise J-lens versus logit-lens evaluation.
- [x] Add executable setup/run scripts, offline tests, provenance, and result logs.
- [x] Execute the full model-backed run on the target machine (CPU, 2026-08-24).
- [x] Review `outputs/minimal/evaluation.md`: answer rank 10/threshold 10; the
  24-sample J-lens remains noisy, as expected for this scope.

## Expansion gate

Do not add a larger dataset or model until the initial run passes the answer-rank
sanity check and the fit provenance contains the expected 24 calibration samples.
The next scientific step should be estimator stability across at least three seeds,
not additional claims about a global workspace.
