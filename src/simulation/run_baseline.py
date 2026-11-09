"""
Run the fixed-timing baseline on every scenario and seed, and save the results.

What this does: for each scenario (low, high, ns, ew) and each seed, it runs one 5,400 s simulation in which
the traffic lights follow the fixed plan, then records how long vehicles waited, how long the queues were,
and so on. These numbers are the reference every smarter controller is compared with later.

Usage (from the project root):
    python src/simulation/run_baseline.py            # 4 scenarios x 3 seeds (the default)
    python src/simulation/run_baseline.py 5          # 4 scenarios x 5 seeds

Writes:
    outputs/sumo_fixed_baseline.csv              one row per run (summary metrics)
    outputs/sumo_fixed_baseline_meta.json        versions and settings (for the paper's methods section)
    data/sumo/logs/fixed_<scenario>_<seed>.csv   per-second counts, queues, arrivals per arm
    data/sumo/logs/fixed_<scenario>_<seed>_occ.npz   vehicle positions in 10 m cells (last 300 m of each arm)
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from sumo_env import FixedTime, SCENARIOS, env_info, run_episode  # noqa: E402

# ----------------------------------- SETTINGS YOU CAN CHANGE -----------------------------------
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3   # how many random seeds per scenario (seeds 0, 1, 2, ...)
SCALE = 1.0                                              # demand scale: 1.0 = the vehicle counts in SCENARIOS
GREENS = (30, 15, 30, 15)                                # fixed plan: NSA, NSLA, EWA, EWLA green seconds
OUT_DIR = Path("outputs")                                # where the result table and metadata go
# -------------------------------------------------------------------------------------------------


def main():
    rows = []
    for scenario in SCENARIOS:
        for seed in range(N_SEEDS):
            summary, _ = run_episode(FixedTime(GREENS), scenario, seed, scale=SCALE,
                                     log_path=f"data/sumo/logs/fixed_{scenario}_{seed}.csv")
            rows.append(summary)
            print(f"{scenario:5s} seed {seed}: avg wait {summary['avg_wait_s']:7.1f} s | "
                  f"mean queue {summary['mean_queue']:6.1f} | arrived {summary['arrived']}/{summary['scheduled']} | "
                  f"{summary['runtime_s']} s", flush=True)
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(exist_ok=True)
    df.to_csv(OUT_DIR / "sumo_fixed_baseline.csv", index=False)
    meta = {"created": datetime.now().isoformat(timespec="seconds"), "seeds": list(range(N_SEEDS)),
            "scale": SCALE, "controller": f"fixed (NSA/NSLA/EWA/EWLA greens {GREENS}, yellow between phases)",
            **env_info()}
    (OUT_DIR / "sumo_fixed_baseline_meta.json").write_text(json.dumps(meta, indent=2))
    cols = ["avg_wait_s", "mean_queue", "max_queue", "arrived", "avg_trip_s", "avg_timeloss_s", "avg_speed_mps"]
    print("\nMean over seeds (fixed timing):")
    pd.set_option("display.width", 200)
    print(df.groupby("scenario")[cols].mean().round(1).loc[list(SCENARIOS)])


if __name__ == "__main__":
    main()