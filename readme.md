# Predictive signal control

Does forecasting arrivals a few seconds ahead make a traffic-signal controller better than reacting to current queues? This project compares LSTM, GRU and hybrid forecasters with simple baselines, then compares controllers that use them with fixed-time, reactive-rule and reactive-RL (DQN) control, at a four-way intersection in SUMO. Evaluation is multi-seed, with paired comparisons on held-out episodes.

The camera-side work that motivated this (vehicle detection and counting on UA-DETRAC) lives in [traffic-vision-yolov8-bytetrack](https://github.com/riyanshigambhir/traffic-vision-yolov8-bytetrack).

## Status

| Phase | Work | Status |
| --- | --- | --- |
| 1 | SUMO four-way intersection, Weibull demand, fixed-time baseline, logging | Code written; first run pending |
| 2 | Control baselines: fixed-time, longest-queue-first | Not started |
| 3 | Forecasting baselines (persistence, moving average, Holt, ARIMA, ridge, gradient boosting) plugged into a predictive controller | Not started |
| 4 | Reactive RL controller after Vidali et al. (DQN) | Not started |
| 5 | LSTM and GRU forecasters plugged into the controller | Not started |
| 6 | Hybrids: Holt/ARIMA + GRU residual; CNN-GRU on cell occupancy; forecast + queue-balance controller | Not started |
| 7 | Final comparison, ablations, oracle test, paper | Not started |

## Design at a glance

| Item | Choice |
| --- | --- |
| Environment | Four arms, 4 lanes each, 750 m; phases NSA, NSLA, EWA, EWLA; 10 s green, 4 s yellow (after Vidali et al., 2019) |
| Demand | Weibull arrival times (shape 2); scenarios low, high, NS, EW; demand scale 0.8, 1.0, 1.2; 5,400 s episodes |
| Forecast target | Arrivals per arm in 5 s bins, 10, 20 and 30 s ahead, from a 60 s look-back |
| Control baselines | Fixed-time (30/15/30/15 s), longest-queue-first, Vidali-style DQN |
| Splits | By episode: training seeds 0-59, validation 60-71, test 100-123 (`src/simulation/splits.py`) |
| Metrics | Average wait (including time waiting to enter), mean queue, trip time, time loss, average speed, completed trips; paired differences with 95% intervals |

The full protocol, decisions and paper notes are in `docs/research-log.md`.

## Quick start

```bash
conda create -n signal-ai python=3.10 -y
conda activate signal-ai
pip install -r requirements.txt
python src/simulation/splits.py            # prints the episode splits
python src/simulation/run_baseline.py 3    # fixed-time baseline: 4 scenarios x 3 seeds (about 2 minutes)
```

Outputs: `outputs/sumo_fixed_baseline.csv` (one row per run), `outputs/sumo_fixed_baseline_meta.json` (versions and settings), and per-second logs in `data/sumo/logs/` (git-ignored).

## Reproducibility

- A seed fixes an episode's arrival times, origins and turns; `splits.py` fixes the scenario and demand scale from the seed.
- Every results file has a `_meta.json` with the SUMO and Python versions, the settings and the git commit. Commit before running an experiment so the recorded commit is the code that ran.
- The generated network files in `data/sumo/` are committed so every run uses the same geometry; routes, logs and trip records are regenerated and git-ignored.
- Every run is listed in the experiment register in `docs/research-log.md`.

## Repository layout

```
src/
  simulation/    sumo_env.py, run_baseline.py, splits.py          (Phase 1)
  control/       longest-queue-first, predictive rule, DQN        (Phases 2 and 4)
  forecasting/   baselines, LSTM, GRU, hybrids                    (Phases 3, 5, 6)
  analysis/      tables, paired tests, figures                    (Phase 7)
configs/         experiment and model settings
data/sumo/       network files (committed); logs, routes, trip records (ignored)
docs/            research-log.md (exported from the running research document)
outputs/         result tables and run metadata
```

## Related work this builds on

- Vidali, Crociani, Vizzari, Bandini (2019), *A Deep Reinforcement Learning Approach to Adaptive Traffic Lights Management*, WOA 2019: the environment, scenarios and reactive RL baseline.
- Gao, Shen, Liu, Ito, Shiratori (2017), *Adaptive Traffic Signal Control*, arXiv:1705.02755: longest-queue-first and fixed-time baselines.
- Genders and Razavi (2016), *Using a Deep Reinforcement Learning Agent for Traffic Signal Control*, arXiv:1611.01142: the cell-based state.

## Licence and citation

Not yet chosen; all rights reserved until a licence file is added. Citation details will be added with the paper.