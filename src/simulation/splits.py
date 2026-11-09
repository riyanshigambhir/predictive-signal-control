"""
Episode splits for the experiments (protocol v1).

Every episode is one simulation run, identified by its seed. The seed also fixes the scenario and the
demand scale, so each set contains all 12 scenario x scale combinations in equal numbers:
    combination index = seed % 12
    scenario = SCENARIOS[index // 3],  scale = SCALES[index % 3]

    training    seeds 0-59     (60 episodes, each combination 5 times)
    validation  seeds 60-71    (12 episodes, each combination once)
    test        seeds 100-123  (24 episodes, each combination twice) - never used for tuning
"""
SCENARIOS = ["low", "high", "ns", "ew"]
SCALES = [0.8, 1.0, 1.2]
SPLITS = {"train": range(0, 60), "validation": range(60, 72), "test": range(100, 124)}


def episode(seed):
    """(scenario, demand scale) for a seed."""
    i = seed % 12
    return SCENARIOS[i // 3], SCALES[i % 3]


def episodes(split):
    """List of (seed, scenario, scale) for 'train', 'validation' or 'test'."""
    return [(seed, *episode(seed)) for seed in SPLITS[split]]


if __name__ == "__main__":
    from collections import Counter
    for name, seeds in SPLITS.items():
        counts = Counter(episode(s) for s in seeds)
        print(f"{name:10s} {len(seeds):3d} episodes | {len(counts)} combinations | each appears {sorted(set(counts.values()))} times")
    print("first test episodes:", episodes("test")[:4])