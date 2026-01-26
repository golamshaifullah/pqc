# PQC

PQC is a lightweight quality-control toolkit for pulsar timing array (PTA)
residuals. It parses tempo2 timfiles, loads TOA/residual arrays via libstempo,
merges timing arrays with timfile metadata, normalizes backend keys, and runs
two QC passes: bad-measurement detection and transient exponential recovery
detection. The intended workflow is a single-call pipeline that returns a
fully-annotated pandas DataFrame or a small CLI for batch runs.

## Features

- Parse tempo2 timfiles (including INCLUDE recursion and flags)
- Load TOA/residual arrays via libstempo
- Merge timing arrays with timfile metadata
- Normalize backend keys (sys/group) for per-backend analysis
- Detect bad measurements and transient exponential recoveries

## Installation

Editable install (for development):

```bash
pip install -e .
```

## Usage

CLI (basic run):

```bash
pqc --par /path/to/pulsar.par --out out.csv
```

CLI (custom thresholds):

```bash
pqc --par /path/to/pulsar.par --out out.csv \
  --backend-col group \
  --tau-corr-min 45 \
  --fdr-q 0.02 \
  --tau-rec-days 10 \
  --delta-chi2 30
```

Python (default pipeline):

```python
from pqc.pipeline import run_pipeline

df = run_pipeline("/path/to/pulsar.par")
```

Python (configured pipeline):

```python
from pqc.pipeline import run_pipeline
from pqc.config import BadMeasConfig, TransientConfig, MergeConfig

df = run_pipeline(
    "/path/to/pulsar.par",
    backend_col="group",
    bad_cfg=BadMeasConfig(tau_corr_days=0.03, fdr_q=0.02),
    tr_cfg=TransientConfig(tau_rec_days=10.0, delta_chi2_thresh=30.0),
    merge_cfg=MergeConfig(tol_days=3.0 / 86400.0),
)
```

## Documentation

- Overview and examples: this README
- CLI help: run `pqc --help`

## API Details

- Pipeline entry point: `src/pqc/pipeline.py`
- Configuration objects: `src/pqc/config.py`
- CLI: `src/pqc/cli.py`
- Detection modules: `src/pqc/detect/`
- I/O and parsing: `src/pqc/io/`

## License

MIT License. See `LICENSE`.
