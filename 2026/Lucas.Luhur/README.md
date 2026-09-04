# Statistical Traffic Analysis of Mix Networks for Anonymous Broadcasting in Privacy-Preserving Proof-of-Stake Blockchains

MSc Financial Technology dissertation summer project, University College London (UCL)

Partner: Institute of Free Technology (IFT), Logos blockchain team

Author: Lucas Luhur (Student no. 25052649, lucas.luhur.25@ucl.ac.uk)

Academic Supervisor: Dr Java Xu

Industrial Supervisor: Dr Alexander Mozeika

## Abstract

In a proof-of-stake blockchain, a lottery weighted by stake selects the
validator that publishes the next block. Privacy-preserving protocols
conceal the winner cryptographically, yet it must still broadcast its
block. An observer of the network can then identify the sender from the
timing of its traffic and estimate its stake from the frequency. This dissertation
measures that leakage and the protection that an anonymous communication
system provides. The complete system is simulated and calibrated to deployed
networks. The traffic is protected by mix networks, in which relays
delay and reorder messages, and by dummy cover traffic. Two attacks are
then run on it, the Bayesian attribution of each block to its sender and
the inference of every validator's stake from its sending frequency. By the
anonymity trilemma, strong anonymity cannot be obtained together with
low latency and low bandwidth, and the question is on which of the two
the blockchain can afford to pay. Against attribution, the random delay of the mixes reduces the
identification of the sender to a residual just above a random guess, at
a small fraction of the delay the protocol tolerates. Stake privacy
depends instead on the concentration of stake, a property of the chain
rather than of the defence. At the concentration observed in practice
the largest stakeholder is identified within weeks. A limit on the stake
per validator, supplemented by cover traffic at a quantified bandwidth
cost, postpones this to years rather than preventing it. The framework allows a designer to evaluate a defence
and to set its parameters under different constraints.

## Repository structure

```
.
├── README.md
├── comp0176.yml         # conda environment spec (Python 3.11 scientific stack)
├── src/                 # validated model components — pure, rng-threaded functions
│   ├── consensus/       # PoS leader lottery -> broadcast events (who won, when)
│   ├── network/         # random-regular graph + quenched link-latency laws
│   ├── anonymity/       # the defence under evaluation: pluggable layers -> a Trace
│   ├── adversary/       # observer (GPA) + de-anonymisation attacks -> a guess
│   ├── metrics/         # measures that grade a guess -> a privacy number
│   ├── theory/          # closed forms drawn against the simulations
│   ├── pipeline_contract.py  # the attack<->measure guess-type contract
│   └── plotstyle.py     # project-wide figure style
├── tests/               # per-component validation scripts (simulation vs. closed forms)
├── experiments/         # the simulation framework: compose components into swept runs
│   ├── config.py        #   Config dataclass + YAML loader
│   ├── pipeline.py      #   run_once (compose) + sweep (grid over config axes)
│   ├── run.py           #   CLI entry point -> results/tables + figures
│   ├── stake_epochs.py  #   extend a stake experiment to many epochs
│   ├── size_bracket.py  #   compare results across the network-size ladder
│   └── configs/         #   one declarative YAML per experiment (see the two template_*.yaml)
├── notebooks/
│   ├── pareto_k_analysis.ipynb      # choosing the Pareto stake-shape k (Cardano calibration)
│   └── fetch_cardano_pool_stakes.py # the Koios download behind the notebook
├── data/                # WonderNetwork ping data (link-latency calibration)
├── results/
│   ├── figures/         # figures (via plotstyle)
│   └── tables/          # result CSVs
└── thesis/              # the dissertation PDF
```

## Setup and installation

The project runs on Python 3.11 with the scientific Python stack (NumPy, SciPy, pandas, matplotlib, NetworkX, SymPy, PyYAML, tqdm, psutil; JupyterLab for the notebook). All dependencies are listed in [comp0176.yml](comp0176.yml).

Create and activate the conda environment:

```bash
conda env create -f comp0176.yml
conda activate comp0176
```

## Running the simulation

Every experiment is one YAML file under `experiments/configs/`. The two templates,
[template_stake_inference.yaml](experiments/configs/template_stake_inference.yaml) and
[template_bayesian_inference.yaml](experiments/configs/template_bayesian_inference.yaml),
document every parameter, its default and the sweep/plot syntax; copy one to start a new
experiment. Run from the repository root:

```bash
python experiments/run.py --list                            # available configs
python experiments/run.py recommendation_stake              # full run -> results/tables + results/figures
python experiments/run.py recommendation_stake --quick      # smoke run (short epoch, one rep)
python experiments/run.py recommendation_stake --plot-only  # redraw from the saved table
python experiments/stake_epochs.py recommendation_stake     # extend a stake run to many epochs
python experiments/size_bracket.py                          # compare across N = 100 / 1000 / 3000
python tests/leader-election-validation.py                  # any validation script runs standalone
```

Results are written to `results/tables/<name>.csv` and `results/figures/<name>/`.

### Running from VS Code (the Run / play button)

The Run button executes the open file with no command-line arguments, so what runs is
set by the module-level variables at the top of the script instead of by flags:

1. Select the `comp0176` interpreter in VS Code (`Python: Select Interpreter`).
2. Open [experiments/run.py](experiments/run.py). Near the top is a menu of
   commented-out `# CONFIG = "..."` lines, one per experiment in
   `experiments/configs/`, grouped into stake, single-path and mix-net experiments.
   Uncomment the one you want to run (and comment out the current one) — exactly one
   `CONFIG = "..."` line must be active, e.g.

   ```python
   CONFIG = "recommendation_attribution"
   ```

3. Set the four defaults just below the menu — `QUICK`, `PLOT_ONLY`, `JOBS`,
   `MEM_FRACTION` (meanings in the table below). For a first look set
   `QUICK = True`; for a redraw of existing results set `PLOT_ONLY = True`.
4. Press Run. The console prints the resolved config name with `(from CONFIG)`, the run
   header, the sweep progress bar and the paths of the table and figures written.

Every script in `tests/` takes no arguments and runs as it is.

### Run modes and parallel processing

`experiments/run.py` has four module-level defaults, used when the file is run without
flags (the Run button); each has a command-line override:

| Default in `run.py` | Flag | Meaning |
|---|---|---|
| `QUICK = False` | `--quick` | Smoke run: the epoch is shortened to `QUICK_T = 20 000` slots (instead of 388 800) and `reps` is forced to 1. Outputs get a `_quick` suffix so they never overwrite a full run. |
| `PLOT_ONLY = False` | `--plot-only` | Skip the sweep and redraw the figures from the existing `results/tables/<name>.csv` (and its cached theory curves). |
| `JOBS = -1` | `--jobs N` | Worker processes per sweep cell. `1` runs serially; `<= 0` picks automatically (CPU cores − 2, capped at 16). |
| `MEM_FRACTION = 0.70` | `--mem-fraction F` | Fraction of usable RAM (the smaller of total and currently free) the worker pool may occupy. Lower it if the machine starts swapping; ~0.8 is safe on 16 GB. |

The sweep (`experiments/pipeline.py`, `sweep`) processes grid cells one at a time and
runs each cell's `reps` realisations in a process pool of size
`min(JOBS, MEM_FRACTION × RAM / estimated peak memory per run, reps)`. Memory-heavy
cells (large `N`, wide or deep mix-nets) therefore get fewer workers; count-based stake
configs (`fast_counts: true`) get the full CPU cap. Each `(cell, rep)` pair draws its
own seed from the config's master `seed`, so the worker count affects only wall-clock
time, never results. A `--quick` run has `reps = 1` and runs serially. The run header
shows the resolved `jobs`, memory budget and estimated memory per run.

The closed-form theory curves drawn on the figures are cached in
`results/tables/<name>_theory.csv`; pass `--refresh-theory` after changing anything in
`src/theory/` to recompute them.
