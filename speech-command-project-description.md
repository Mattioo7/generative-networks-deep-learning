# Project Description: `transformers-speech-commands-clean`

## Overview

A speech keyword recognition system that classifies short audio clips into one of 12 classes (10 target commands + "unknown" + "silence") using the Google Speech Commands dataset. The project's primary goal is systematic comparison of LSTM and Transformer architectures under varied data and training configurations, driven by a grid-search experiment framework.

---

## Features

### Audio Processing Pipeline
- Reads 16-bit PCM WAV files, resamples to 16 kHz, and pads/trims clips to exactly 1 second (16,000 samples)
- Computes log mel-spectrograms (64 bins, FFT window 512, hop 160) via `torchaudio`
- Applies z-score normalization per feature

### Dataset Management
- Source data lives in a `.7z` archive; the codebase handles listing, selective extraction, and caching without unpacking everything
- Silence examples are synthesized on-the-fly from background noise files in the archive
- Supports two sampling strategies: **natural** (preserve original class distribution) and **balanced** (WeightedRandomSampler for uniform class frequency during training)
- All unknown-class words (everything not in the 10 target labels) are remapped to a single "unknown" bucket
- Train/validation/test splits are controlled by fraction parameters; validation and test file lists from the original dataset are respected

### Model Architectures

**LSTMBaseline**
- Input: mel-spectrogram frames `[batch, time, n_mels]`
- Bidirectional LSTM (configurable hidden size and number of layers)
- Mean-pool over time, dropout, linear classifier

**TransformerBaseline**
- Linear projection from `n_mels` → `d_model`
- Sinusoidal positional encoding
- Standard `TransformerEncoder` (multi-head attention + feedforward)
- Mean-pool over time, dropout, linear classifier

Both models output 12 logits; loss is CrossEntropyLoss, optimizer is AdamW.

### Training Infrastructure
- Early stopping with configurable patience and minimum improvement delta; best weights are restored before final evaluation
- Automatic CUDA/CPU device selection
- Per-epoch tracking of loss and accuracy for both train and validation sets
- Learning rate is logged each epoch

### Experiment Framework
- All parameters are encoded in typed dataclasses split into **fixed** (constant across runs) and **grid** (lists of values to sweep)
- `expand_experiment_grid()` computes the Cartesian product across grid axes, producing one config per run
- Results are written to a structured output directory per run: JSON config, per-epoch CSV history, summary CSV, confusion matrix CSV, loss/accuracy curves PNG, confusion matrix heatmap PNG

---

## Code Structure

```
transformers-speech-commands-clean/
├── pyproject.toml            # uv-managed deps: torch 2.11+cu130, torchaudio, numpy, pandas, matplotlib, jupyterlab
├── scripts/
│   ├── __init__.py           # Public re-exports
│   ├── config.py             # All dataclasses: DataFixedParams, FeatureFixedParams, FitFixedParams,
│   │                         #   DataGridParams, ModelGridParams, FitGridParams
│   ├── data.py               # build_experiment_manifest, add_silence_examples, sample_split,
│   │                         #   extract_experiment_audio_files, SpeechCommandsDataset, get_dataloaders
│   ├── models.py             # LSTMBaseline, TransformerBaseline (both nn.Module subclasses)
│   ├── training.py           # fit_model, train_one_epoch, evaluate (returns metrics dict)
│   ├── runner.py             # run_experiment: wires config → data → model → training → outputs
│   ├── outputs.py            # run_name (slug), output_paths, save_history_plot, save_confusion_matrix_plot, save_json
│   ├── archive.py            # archive_files, archive_lines, extract_archive_files (7z/tar abstraction)
│   ├── progress.py           # progress_bar (tqdm wrapper: terminal / notebook / auto), stage()
│   └── check_setup.py        # Validates Python ≥ 3.12, 7z/tar, data archive, CUDA
└── notebooks/
    ├── 00_testing.ipynb              # Smoke-test baseline
    ├── 01_dataset_analysis.ipynb     # Class distribution, audio stats
    ├── 02_baseline_models.ipynb      # First LSTM/Transformer runs
    ├── 03_hyperparameter_experiments.ipynb
    ├── 04_split_unknown_known_experiment.ipynb
    └── 05_sampling_experiment.ipynb
```

### Data Flow (end-to-end)

```
7z archive
  └─ archive.py ──► manifest (label → file list)
                ──► add_silence_examples (background noise → silence class)
                ──► sample_split (subsample by fraction per label)
                ──► extract_experiment_audio_files (cache WAVs to disk)
                        │
                        ▼
              SpeechCommandsDataset
                (read WAV → resample → pad/trim → mel-spec → normalize)
                        │
                        ▼
              DataLoader [+ WeightedRandomSampler if balanced]
                        │
                        ▼
              fit_model → train_one_epoch / evaluate → early stopping
                        │
                        ▼
              outputs.py → CSV + PNG reports
```

### Configuration Schema

| Group | Key params |
|---|---|
| `DataFixedParams` | data dir, archive name, cache dir, output dir, target labels list, sample rate, clip length |
| `FeatureFixedParams` | `n_mels`, `n_fft`, `hop_length`, `normalize` |
| `FitFixedParams` | device, `num_workers`, early-stop patience/delta, logging verbosity |
| `DataGridParams` | `train_fraction`, `validation_fraction`, `test_fraction`, `unknown_fraction`, `silence_samples`, `sampling_strategy`, `seed` |
| `ModelGridParams` | `model_type` (`lstm`/`transformer`), `dropout`; LSTM: `hidden_size`, `num_layers`, `bidirectional`; Transformer: `d_model`, `nhead`, `num_layers`, `dim_feedforward` |
| `FitGridParams` | `epochs`, `batch_size`, `learning_rate`, `weight_decay` |

---

## Design Principles

- **Separation of concerns**: config, data, model, training, and I/O are in distinct modules with clean interfaces
- **Functional data pipeline**: each step returns a new object/dataframe rather than mutating state
- **Deferred extraction**: audio files are extracted from the archive only for the specific subset needed per experiment run, keeping disk usage low
- **Reproducibility**: every run saves its full config JSON alongside results; seeds are explicit parameters
- **Notebook-friendly**: `progress_bar()` transparently switches between `tqdm.tqdm` and `tqdm.notebook.tqdm`
