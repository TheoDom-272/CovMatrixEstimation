# QuantPortfolioEngine

A Python framework for evaluating covariance matrix estimators.

## Research question

Which covariance estimator best minimizes portfolio tracking error across realistic market conditions?

Five estimators are compared: Rolling Sample, EWMA (RiskMetrics), Ledoit-Wolf linear shrinkage (LW2004), Analytical Non-Linear Shrinkage (ANLS2020), and Quadratic-Inverse Shrinkage (QIS2022). Test universes are MSCI EMU (~240 European equities) and MSCI US (~600 US equities), with data loaded from local Bloomberg files.

## Key findings

- **QIS and LW consistently dominate** Rolling and EWMA across all tested configurations.
- **The concentration ratio c = p/n is the principal source of estimation uncertainty:** shrinkage estimators, particularly QIS, are substantially less sensitive to c than Rolling Sample, making them more robust across varying universe sizes and sample lengths.
- **Exclusion rate is the strongest driver of absolute tracking error levels**, rebalancing frequency is secondary, with quarterly offering the best TE/turnover trade-off.
- **Daily data outperforms weekly** for large-dimension equity universes: reducing the p/n ratio outweighs microstructure noise reduction from weekly aggregation.
- EWMA exhibits structural unsuitability for this application.

## Architecture overview

```
QuantPortfolioEngine/
│
├── Cov_Modelisation_App.py          # Tkinter application entry point
│
├── Modules/
│   ├── App/tkinter_app/
│   │   ├── widget.py                # Reusable Tkinter components
│   │   └── model_builder.py         # Model configuration panel
│   │
│   ├── Financial_engineering/
│   │   ├── statistics/
│   │   │   ├── multivariate_vol_estimation.py   # Base classes (MultiVolModel, CovariancePath)
│   │   │   ├── ledoit_wolf.py                   # LW2004, ANLS2020, QIS2022, OAS
│   │   │   ├── EWMACov.py                       # RiskMetrics EWMA
│   │   │   └── Engle/ewma_qmv_numba.py          # Numba-accelerated lambda tuning
│   │   │
│   │   └── optimization/
│   │       ├── base.py              # Optimization contracts (Problem, Constraints, Bounds)
│   │       ├── optimizers.py        # SLSQP, Clarabel, make_optimizer()
│   │       ├── tracking_error.py    # TE-min objective functions
│   │       ├── moments.py           # Return/risk moment helpers
│   │       ├── cost.py              # Transaction cost models
│   │       └── metrics.py          # Portfolio metrics (Sharpe, TE, turnover…)
│   │
│   ├── portfolio_management/backtesting/
│   │   ├── engine.py                # Backtest loop (BacktestEngine)
│   │   ├── engine_types.py          # Dataclasses & Protocols (BacktestConfig, BacktestResult…)
│   │   ├── engine_execution.py      # Position execution (shares mode)
│   │   ├── engine_logger.py         # Console logger
│   │   ├── covariance_provider.py   # CovarianceProvider (path / rebal modes)
│   │   ├── core_replication.py      # TE-min allocator (CoreTEMinAllocator)
│   │   ├── core_satellite_allocation.py
│   │   ├── rebalancing.py           # RebalanceSchedule
│   │   ├── rebalance_rules.py       # Rebalancing rules
│   │   └── port_inventory_exporter.py
│   │
│   ├── data/
│   │   ├── local_files.py           # LocalIndexFolderDataSource (Bloomberg local files)
│   │   ├── base.py                  # PriceAPI / BatchPriceAPI protocols
│   │   └── models.py                # PriceRequest, CorporateActionRequest
│   │
│   └── study/covariance_study/
│       ├── pipeline.py              # ModelEvaluator (entry point for all evaluations)
│       ├── stat_study.py            # Statistical evaluation (DGP simulations, matrix losses)
│       ├── eco_study.py             # Economic evaluation (real-data TE-min backtest)
│       └── __init__.py
│
├── Reports/
│   └── outputs/
│       └── montecarlo/              # Excel checkpoint files (auto-generated, gitignored)
│
└── Données/                         # Bloomberg data (gitignored — not included in repo)
    └── IndicesLocaux/
        └── <IndexName>/
            ├── 2020.xlsx … 2024.xlsx
            ├── Compo.xlsx
            ├── Mapping.xlsx
            └── Exclusions.xlsx      (optional)
```

## Installation

**Requirements:** Python 3.10+, Windows recommended (memmap path handling).

```bash
git clone https://github.com/<your-username>/QuantPortfolioEngine.git
cd QuantPortfolioEngine
pip install -r requirements.txt
```

### Dependencies

```
numpy
pandas
scipy
scikit-learn
clarabel
numba
tqdm
openpyxl
matplotlib
reportlab
joblib
tkinter          # bundled with Python on Windows
```

> **Note:** `clarabel` requires Rust to compile from source on some systems. Use `pip install clarabel`.

## Data setup

Bloomberg data is **not included** in this repository due to licensing restrictions. To use the engine with your own data, place files in `Données/IndicesLocaux/<IndexName>/` following the expected format described in `local_files.py`.

The application auto-detects index subfolders when launched.

## Usage

### GUI application

```bash
python Cov_Modelisation_App.py
```

The application exposes two tabs:

**Estimation tab** - Single backtest on real data. Configure the index, date range, covariance models, rebalancing frequency, and optimizer. Generates a multi-page PDF report with ex-post TE, NAV, portfolio diagnostics, and model comparisons.

**Monte Carlo tab** - Two parallel branches:
- *Economic MC*: sweeps a grid of (rolling window × exclusion fraction × rebalancing frequency) over multiple random universe draws, exports an Excel checkpoint file with per-scenario and aggregated metrics.
- *Statistical MC*: DGP-based simulation (Static Oracle or Factor Shock) computing Frobenius / spectral / Stein / precision matrix losses for each estimator.


## Covariance estimation modes

| Model | Class | `compute_mode` |
|---|---|---|
| Rolling Sample | `RollingSampleCov` | `rebal` |
| EWMA (RiskMetrics) | `EWMACov` | `path` (mandatory) |
| Ledoit-Wolf 2004 | `LedoitWolfLinearShrinkage` | `rebal` |
| ANLS 2020 | `LedoitWolfANLS` | `rebal` |
| QIS 2022 | `LedoitWolfQIS` | `rebal` |
| OAS | `LedoitWolfOAS` | `rebal` |
| DCC-GARCH | `DCCModel` | `path` |

**`compute_mode="path"`** pre-computes the full covariance series and stores it as a memory-mapped array. Required for EWMA (recursive formulation).

**`compute_mode="rebal"`** computes covariance on-demand at each rebalancing date. Lower memory footprint, suitable for all non-recursive estimators.

## Monte Carlo design

- Seeding convention: `random_state + i` for scenario `i` (base seed 42 by default), ensures full reproducibility.
- Benchmark index tickers are always excluded from the investable universe before random exclusion sampling is applied.
- Checkpoint/resume via Excel `run_log` sheet: already-computed scenarios are skipped on restart.
- Common start date enforced across all models: `common_start = universe_returns.index[max_rolling]` to ensure fair comparison.
- Parallelization via `joblib.Parallel(backend="loky", return_as="generator")`.

## Optimizer

Two solvers are supported and produce different economic trade-offs:

- **Clarabel**: finds the true global QP minimum. Produces the lowest ex-ante TE.
- **SLSQP**: acts as an implicit proximal regularizer via warm start, producing higher ex-ante TE but lower ex-post TE due to reduced turnover.

## Repository structure notes

The following are excluded from the repository via `.gitignore`:

- `Données/` — Bloomberg data (licensing restrictions)
- `Reports/outputs/montecarlo/*.xlsx` — large simulation output files
- `*.npy`, `*.dat` — memory-mapped covariance arrays
- `__pycache__/`, `.pytest_cache/`


---

*Research conducted as part of a personnal quantitative finance thesis on covariance matrix estimation for index replication.*
