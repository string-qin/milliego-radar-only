# milliEgo radar-only spectrum adaptation

This repository maintains our PyTorch adaptation of milliEgo for radar-only experiments, including ADC range–azimuth power spectra on ColoRadar. It is an adaptation, not an exact reproduction of the paper's fused model or original dataset. IMU is not used in our radar-only experiments. Radar self-attention is retained.

## Storage

Code stays under `/home/metaiot_guest/Workspace/qinzx/milliEgo`. Large data, checkpoints, caches, virtual environments and generated results reside under `/storage/ssd/metaiot_guest/RadarProbMap/baselines/milliego`, with symlinks at their previous project paths. The seven artifact directories were migrated and checksum-verified on 2026-11-05. See [STORAGE_LAYOUT.md](STORAGE_LAYOUT.md) for the migration and checkout setup commands.

The `.gitignore` excludes these directories and binary datasets/checkpoints. The public raw dataset is an external read-only dependency. Do not commit model weights or cached spectra.

## Maintained experiments

- `milliego/`: radar CNN, radar attention, Keras-style LSTM and pose heads.
- `scripts/`: data preparation, evaluation, diagnostics and storage tools.
- `experiments/legacy_runs/`: versioned copies of historical experiment scripts that previously lived below the generated `runs/` directory.
- `experiments/author_frozen/`: contiguous-sampling experiment with frozen author CNN weights.
- `experiments/author_scratch/`: contiguous sampling with all model parameters initialized and trained from scratch.
- `experiments/turn_sampling/`: four sampling comparisons starting from the same yaw-warmup checkpoint; these continue joint pose training.
- `experiments/angle_comparison/`: matched-timestamp single-step rotation comparison from saved trajectories, without calibration.
- `experiment_records/`: small configurations and evaluation summaries, not checkpoint binaries.
- `configs/coloradar_split.json`: 40 training, 4 validation and 8 test routes.

The later two author-style experiments trained for 200 epochs (64,200 updates). The frozen experiment selected epoch 136 and the scratch experiment selected epoch 164 by validation loss. The scratch model's z-angle predictions were close to zero on many strong-turn samples. Results remain exploratory and do not establish paper-level reproduction. Different training settings and single checkpoints prevent causal claims about yaw warmup alone. These held-out routes have already been inspected during development.

The archived scripts retain server-specific paths and dependencies on existing caches. They are records of completed experiments, not a fresh-data turnkey training interface. Restore the storage layout and runtime scripts first. No preprocessing, loss, network or model weights are changed by the storage migration.

## Upstream provenance

Original sources are preserved in `upstream/` and `reference_odombeyondvision/`, excluding their nested Git metadata. Exact upstream URLs and revisions are recorded in [UPSTREAM_SOURCES.json](UPSTREAM_SOURCES.json); upstream notices and available licenses are retained. `SOURCE_EXPORT_MANIFEST.json` records hashes of the initial source export before repository documentation and storage tooling were added.

## Earlier environment notes

The uv-managed PyTorch CUDA environment is installed and validated.
A PyTorch radar-only single-sequence training/evaluation pilot is implemented and executed.
See SINGLE_SEQUENCE_REPORT_20261102.md. Pipeline checks pass; temporal-test accuracy remains poor.
Official TensorFlow/Keras sources are preserved in upstream/.

See VALIDATION_20261102.md for actual GPU tests, pretrained conversion checks,
all-sequence dataset metadata checks, sampled payload coverage, and limitations.

```bash
cd /home/metaiot_guest/Workspace/qinzx/milliEgo
source env.sh
uv sync --locked --python /usr/bin/python3.10
CUDA_VISIBLE_DEVICES=3 uv run --locked python scripts/check_gpu.py
```

Outputs, caches, environments, and temporary paths are accessed through the project directory and are intended to resolve to the data disk after migration.
Raw data under /storage/data/metaiot_data/public_dataset are read-only inputs.
The following historical notes describe early diagnostics; later spectrum experiments are listed above.


Current baseline constraint and preprocessing status: see PREPROCESS_CONTRACT_AUDIT_20261102.md and BASELINE_CONTRACT.json. Prior pilot scores are exploratory, not formal baseline results.

Expanded training (5593 pairs) and matched-update controls completed. Results are mixed: local pose errors improve but trajectory ATE worsens. See runs/expanded_training_v1/REPORT.md; this is not a certified original-paper baseline.
