"""
SUMO four-way intersection environment (design follows Vidali et al., 2019).

Provides:
  build_network()    writes the intersection (4 arms x 4 lanes, 750 m long) and runs netconvert
  generate_routes()  Weibull-shaped demand for four scenarios: low, high, ns, ew (with a demand scale)
  run_episode()      runs one simulation under a controller; returns summary metrics and a per-second log
  FixedTime          the fixed-timing baseline controller
  env_info()         versions and settings to save beside every result (for the paper's methods section)

One simulation step = exactly 1 simulated second.
Lane layout per arm (SUMO lane 0 is the rightmost):
  lane 0: right turn or straight, lanes 1-2: straight, lane 3: left turn only.
Signal phases (as in Vidali et al.): NSA, NSLA, EWA, EWLA (advance = straight + right; LA = left).

'Camera view' = the last ZONE_LEN metres of each arm. Everything the controller and the forecaster see
(counts, queues, arrivals) is measured inside that zone. Cell occupancy over the last OCC_LEN metres is
also saved, for models that use the position of vehicles along the approach.
"""
import csv
import platform
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import sumolib
import traci
import traci.constants as tc

# =====================================================================================================
# SETTINGS YOU CAN CHANGE
# Settings marked [paper] come from Vidali et al. (2019): change them only if you accept that results are
# then no longer comparable with that paper. After changing EDGE_LEN or SPEED_LIMIT, delete
# data/sumo/intersection.net.xml so the road network is rebuilt.
# =====================================================================================================
SUMO_DIR = Path("data/sumo")          # where the network, routes, logs and trip records go

EDGE_LEN = 750.0       # [paper] metres, each arm, node to node (the usable lane is a little shorter)
SPEED_LIMIT = 13.89    # m/s (50 km/h). Vidali gives none; Gao uses 19.44 m/s (70 km/h)
EPISODE_S = 5400       # [paper] seconds per episode (1.5 hours)
YELLOW_S = 4           # [paper] seconds of yellow whenever the phase changes

ZONE_LEN = 150.0       # 'camera view': metres before the stop line where counts and queues are measured
OCC_LEN = 300.0        # metres before the stop line covered by the saved cell occupancy
CELL_M = 10.0          # length of one occupancy cell, metres

# scenario: (number of vehicles in the episode, share of vehicles starting on the N or S arms)
SCENARIOS = {          # [paper] all four
    "low": (600, 0.5),
    "high": (4000, 0.5),
    "ns": (2000, 0.9),
    "ew": (2000, 0.1),
}
P_STRAIGHT, P_RIGHT, P_LEFT = 0.75, 0.125, 0.125   # [paper: 75% straight, 25% turning]; the left/right split is ours
WEIBULL_SHAPE = 2      # [paper] shape of the arrival-time distribution (2 = demand rises, then falls)
CLEAR_OUT_S = 200      # seconds at the end of an episode with no new arrivals, so queues can clear

# One vehicle type for all vehicles. Length and gap follow Gao et al.; the rest are our choices.
VEH_LENGTH, VEH_MIN_GAP = 5.0, 2.5   # metres
VEH_ACCEL, VEH_DECEL = 1.0, 4.5      # m/s^2
VEH_SIGMA = 0.5                      # driver imperfection, 0 (perfect) to 1
DEPART_SPEED = 10                    # m/s at which vehicles enter the network

# =====================================================================================================
# NOT SETTINGS: tied to the code (lane layout, phase names, signal-link logic). Do not edit.
# =====================================================================================================
NET_FILE = SUMO_DIR / "intersection.net.xml"
ARMS = ["N", "S", "E", "W"]
TLS = "C"
N_CELLS = int(OCC_LEN / CELL_M)
LEFT_LANE = 3
PHASES = ["NSA", "NSLA", "EWA", "EWLA"]

LINK_KINDS = ["right", "straight0", "straight1", "straight2", "left"]
STRAIGHT = {"N": "S", "S": "N", "E": "W", "W": "E"}
RIGHT = {"N": "W", "S": "E", "E": "N", "W": "S"}   # driving on the right
LEFT = {"N": "E", "S": "W", "E": "S", "W": "N"}
N_LINKS = len(ARMS) * len(LINK_KINDS)


def env_info():
    """Settings and versions to store beside results."""
    try:
        sumo_v = subprocess.run([sumolib.checkBinary("sumo"), "--version"], capture_output=True,
                                text=True).stdout.splitlines()[0]
    except Exception:
        sumo_v = "unknown"
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True).stdout.strip()
    except Exception:
        commit = ""
    return {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
            "sumo": sumo_v, "git_commit": commit, "edge_len_m": EDGE_LEN, "zone_len_m": ZONE_LEN,
            "occupancy_len_m": OCC_LEN, "cell_m": CELL_M, "episode_s": EPISODE_S, "yellow_s": YELLOW_S,
            "scenarios": SCENARIOS, "turn_split": [P_STRAIGHT, P_RIGHT, P_LEFT], "speed_limit_mps": SPEED_LIMIT,
            "weibull_shape": WEIBULL_SHAPE, "clear_out_s": CLEAR_OUT_S, "depart_speed_mps": DEPART_SPEED,
            "vehicle": {"length_m": VEH_LENGTH, "min_gap_m": VEH_MIN_GAP, "accel": VEH_ACCEL,
                        "decel": VEH_DECEL, "sigma": VEH_SIGMA}}


def movement(in_lane, out_edge):
    """Classify a signal link as 'right' / 'straight' / 'left' from its incoming lane and outgoing edge."""
    arm = in_lane[0]
    dest = out_edge[-1]
    if dest == RIGHT[arm]:
        return arm, "right"
    if dest == LEFT[arm]:
        return arm, "left"
    return arm, "straight"


def build_phase_states(left_signal="G"):
    """Ask SUMO which link index is which, then build the green/yellow state string of each phase.
    (netconvert orders the links by geometry, so the order must be read, not assumed.)"""
    links = traci.trafficlight.getControlledLinks(TLS)
    states = {}
    for phase in PHASES:
        arms = ("N", "S") if phase in ("NSA", "NSLA") else ("E", "W")
        want_left = phase in ("NSLA", "EWLA")
        green, yellow = ["r"] * len(links), ["r"] * len(links)
        for i, group in enumerate(links):
            in_lane, out_lane, _ = group[0]
            arm, kind = movement(in_lane, out_lane.rsplit("_", 1)[0])
            if arm in arms and (kind == "left") == want_left:
                green[i] = left_signal if kind == "left" else "G"
                yellow[i] = "y"
        states[phase] = ("".join(green), "".join(yellow))
    return states


def build_network(force=False):
    SUMO_DIR.mkdir(parents=True, exist_ok=True)
    if NET_FILE.exists() and not force:
        return NET_FILE
    pos = {"C": (0, 0), "N": (0, EDGE_LEN), "S": (0, -EDGE_LEN), "E": (EDGE_LEN, 0), "W": (-EDGE_LEN, 0)}
    nodes = ['<nodes>', '  <node id="C" x="0" y="0" type="traffic_light"/>']
    nodes += [f'  <node id="{a}" x="{pos[a][0]}" y="{pos[a][1]}" type="priority"/>' for a in ARMS]
    nodes.append('</nodes>')
    edges = ['<edges>']
    for a in ARMS:
        edges.append(f'  <edge id="{a}_C" from="{a}" to="C" numLanes="4" speed="{SPEED_LIMIT}"/>')
        edges.append(f'  <edge id="C_{a}" from="C" to="{a}" numLanes="4" speed="{SPEED_LIMIT}"/>')
    edges.append('</edges>')
    cons = ['<connections>']
    for a in ARMS:
        moves = [("right", 0, RIGHT[a], 0), ("straight0", 0, STRAIGHT[a], 0),
                 ("straight1", 1, STRAIGHT[a], 1), ("straight2", 2, STRAIGHT[a], 2),
                 ("left", 3, LEFT[a], 3)]
        for kind, fl, dest, tl_ in moves:
            cons.append(f'  <connection from="{a}_C" to="C_{dest}" fromLane="{fl}" toLane="{tl_}" '
                        f'tl="{TLS}"/>')
    cons.append('</connections>')
    tll = ['<tlLogics version="1.0">',
           f'  <tlLogic id="{TLS}" type="static" programID="0" offset="0">',
           f'    <phase duration="99999" state="{"r" * N_LINKS}"/>',
           '  </tlLogic>', '</tlLogics>']
    for name, lines in [("nodes.nod.xml", nodes), ("edges.edg.xml", edges),
                        ("connections.con.xml", cons), ("tll.tll.xml", tll)]:
        (SUMO_DIR / name).write_text("\n".join(lines) + "\n")
    subprocess.run([sumolib.checkBinary("netconvert"),
                    "--node-files", str(SUMO_DIR / "nodes.nod.xml"),
                    "--edge-files", str(SUMO_DIR / "edges.edg.xml"),
                    "--connection-files", str(SUMO_DIR / "connections.con.xml"),
                    "--tllogic-files", str(SUMO_DIR / "tll.tll.xml"),
                    "--no-turnarounds", "true", "--no-warnings", "true",
                    "-o", str(NET_FILE)], check=True)
    return NET_FILE


def generate_routes(scenario, seed, out_path, T=EPISODE_S, scale=1.0):
    """Write a route file; returns the number of scheduled vehicles. scale multiplies the vehicle count."""
    n, p_ns = SCENARIOS[scenario]
    n = int(round(n * scale))
    rng = np.random.default_rng(seed)
    x = np.sort(rng.weibull(WEIBULL_SHAPE, n))                       # Weibull, shape 2: a rising then falling demand
    t = np.rint((x - x[0]) / (x[-1] - x[0]) * (T - CLEAR_OUT_S)).astype(int)   # leave 200 s to clear the queues
    lines = ['<routes>',
             f'  <vType id="car" length="{VEH_LENGTH}" minGap="{VEH_MIN_GAP}" accel="{VEH_ACCEL}" '
             f'decel="{VEH_DECEL}" sigma="{VEH_SIGMA}" maxSpeed="{SPEED_LIMIT}"/>']
    for o in ARMS:
        for kind, dest in (("S", STRAIGHT[o]), ("R", RIGHT[o]), ("L", LEFT[o])):
            lines.append(f'  <route id="{o}_{kind}" edges="{o}_C C_{dest}"/>')
    for i, depart in enumerate(t):
        if rng.random() < p_ns:
            o = "N" if rng.random() < 0.5 else "S"
        else:
            o = "E" if rng.random() < 0.5 else "W"
        u = rng.random()
        kind = "S" if u < P_STRAIGHT else ("R" if u < P_STRAIGHT + P_RIGHT else "L")
        lines.append(f'  <vehicle id="v{i}" type="car" route="{o}_{kind}" depart="{depart}" '
                     f'departLane="best" departSpeed="{DEPART_SPEED}"/>')
    lines.append('</routes>')
    Path(out_path).write_text("\n".join(lines) + "\n")
    return n


class FixedTime:
    """Fixed-timing baseline: NSA 30 s, NSLA 15 s, EWA 30 s, EWLA 15 s, 4 s yellow between phases."""
    name = "fixed"

    def __init__(self, greens=(30, 15, 30, 15)):
        self.greens, self.i = greens, 0

    def reset(self):
        self.i = 0

    def next(self, obs):
        idx = self.i % len(PHASES)
        self.i += 1
        return PHASES[idx], self.greens[idx]


def read_tripinfo(path):
    """Average trip duration, time loss and speed of the vehicles that finished their trip."""
    dur, loss, speed, wait = [], [], [], []
    if Path(path).exists():
        for _, el in ET.iterparse(path):
            if el.tag == "tripinfo":
                d = float(el.get("duration"))
                dur.append(d)
                loss.append(float(el.get("timeLoss")))
                wait.append(float(el.get("waitingTime")))
                speed.append(float(el.get("routeLength")) / d if d > 0 else 0.0)
                el.clear()
    m = lambda v: round(float(np.mean(v)), 2) if v else 0.0
    return {"avg_trip_s": m(dur), "avg_timeloss_s": m(loss), "avg_speed_mps": m(speed),
            "avg_wait_finished_s": m(wait)}


def run_episode(controller, scenario="high", seed=0, T=EPISODE_S, scale=1.0, gui=False,
                log_path=None, left_signal="G"):
    """Run one episode. Returns (summary dict, per-second log as a list of dicts).
    If log_path is given, the log is saved as CSV and the cell occupancy as <log_path>_occ.npz."""
    net = build_network()
    tag = f"{scenario}_{seed}_{scale:g}"
    routes = SUMO_DIR / f"routes_{tag}.rou.xml"
    n_sched = generate_routes(scenario, seed, routes, T, scale)
    tripinfo = SUMO_DIR / "tripinfo" / f"{controller.name}_{tag}.xml"
    tripinfo.parent.mkdir(parents=True, exist_ok=True)
    controller.reset()
    cmd = [sumolib.checkBinary("sumo-gui" if gui else "sumo"), "-n", str(net), "-r", str(routes),
           "--step-length", "1", "--no-step-log", "true", "--no-warnings", "true",
           "--time-to-teleport", "-1", "--seed", str(seed), "--tripinfo-output", str(tripinfo)]
    traci.start(cmd)
    t0 = time.time()
    pstates = build_phase_states(left_signal)
    lane_len = traci.lane.getLength("N_C_0")            # the usable lane is a little shorter than 750 m
    zone_start, occ_start = lane_len - ZONE_LEN, lane_len - OCC_LEN

    state = {"t": 0}
    hist = {a: [] for a in ARMS}                       # vehicles in the zone, one value per second
    wait, depart_t, travel = {}, {}, []
    pending_wait = {}                                   # seconds spent waiting OUTSIDE the network to enter
    log, queue_series, pending_series, occ_series = [], [], [], []
    arrived = 0
    prev_ids = {a: set() for a in ARMS}
    last = {"zone": {a: 0 for a in ARMS}, "halt": {a: 0 for a in ARMS},
            "zoneL": {a: 0 for a in ARMS}, "haltL": {a: 0 for a in ARMS}}
    cur = {"phase": None, "start": 0}

    def one_second(signal_state, label):
        nonlocal arrived, prev_ids
        traci.trafficlight.setRedYellowGreenState(TLS, signal_state)
        traci.simulationStep()
        state["t"] += 1
        t = state["t"]
        for vid in traci.simulation.getDepartedIDList():
            traci.vehicle.subscribe(vid, [tc.VAR_ROAD_ID, tc.VAR_SPEED, tc.VAR_LANEPOSITION, tc.VAR_LANE_INDEX])
            depart_t[vid] = t
            wait[vid] = pending_wait.pop(vid, 0)     # insertion delay counts as waiting
        pend = traci.simulation.getPendingVehicles()
        for vid in pend:
            pending_wait[vid] = pending_wait.get(vid, 0) + 1
        for vid in traci.simulation.getArrivedIDList():
            arrived += 1
            travel.append(t - depart_t[vid])
        ids = {a: set() for a in ARMS}
        zone = {a: 0 for a in ARMS}
        halt = {a: 0 for a in ARMS}
        zoneL = {a: 0 for a in ARMS}
        haltL = {a: 0 for a in ARMS}
        occ = np.zeros((len(ARMS), N_CELLS), dtype=np.uint8)
        for vid, res in traci.vehicle.getAllSubscriptionResults().items():
            speed = res[tc.VAR_SPEED]
            stopped = speed < 0.1
            if stopped:
                wait[vid] += 1
            road = res[tc.VAR_ROAD_ID]
            if road.endswith("_C") and road[0] in zone:
                a, pos = road[0], res[tc.VAR_LANEPOSITION]
                if pos >= occ_start:
                    occ[ARMS.index(a), min(int((pos - occ_start) // CELL_M), N_CELLS - 1)] += 1
                if pos >= zone_start:
                    ids[a].add(vid)
                    zone[a] += 1
                    halt[a] += stopped
                    if res[tc.VAR_LANE_INDEX] == LEFT_LANE:
                        zoneL[a] += 1
                        haltL[a] += stopped
        arr = {a: len(ids[a] - prev_ids[a]) for a in ARMS}    # vehicles that entered the zone this second
        dep = {a: len(prev_ids[a] - ids[a]) for a in ARMS}    # vehicles that crossed the stop line
        prev_ids = ids
        for a in ARMS:
            hist[a].append(zone[a])
        last.update(zone=zone, halt=halt, zoneL=zoneL, haltL=haltL)
        queue_series.append(sum(halt.values()))
        pending_series.append(len(pend))
        occ_series.append(occ)
        row = {"t": t, "phase": label, "since_phase": (t - cur["start"]) if label != "Y" else 0}
        for name, d in (("zone", zone), ("zoneL", zoneL), ("halt", halt), ("haltL", haltL),
                        ("arr", arr), ("dep", dep)):
            row.update({f"{name}_{a}": d[a] for a in ARMS})
        row.update({"pending": len(pend), "departed": len(depart_t), "arrived": arrived})
        log.append(row)

    while state["t"] < T:
        obs = {"t": state["t"], "zone": dict(last["zone"]), "halt": dict(last["halt"]),
               "zoneL": dict(last["zoneL"]), "haltL": dict(last["haltL"]),
               "pending": pending_series[-1] if pending_series else 0,
               "history": hist, "phase": cur["phase"], "since_phase": state["t"] - cur["start"]}
        phase, dur = controller.next(obs)
        if cur["phase"] is not None and phase != cur["phase"]:
            for _ in range(YELLOW_S):
                if state["t"] < T:
                    one_second(pstates[cur["phase"]][1], "Y")
        cur["start"] = state["t"]
        for _ in range(dur):
            if state["t"] < T:
                one_second(pstates[phase][0], phase)
        cur["phase"] = phase
    traci.close()

    waits = list(wait.values()) + list(pending_wait.values())   # includes vehicles never inserted
    summary = {
        "controller": controller.name, "scenario": scenario, "seed": seed, "scale": scale,
        "scheduled": n_sched, "departed": len(depart_t), "arrived": arrived,
        "unfinished": len(depart_t) - arrived,
        "avg_wait_s": round(float(np.mean(waits)), 2) if waits else 0.0,
        "total_wait_s": int(sum(waits)),
        "mean_queue": round(float(np.mean(queue_series)), 2),
        "max_queue": int(max(queue_series)),
        "mean_pending": round(float(np.mean(pending_series)), 1),
        "waits_counted": len(waits),
        **read_tripinfo(tripinfo),
        "runtime_s": round(time.time() - t0, 1),
    }
    if log_path:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
            w.writeheader()
            w.writerows(log)
        np.savez_compressed(log_path.with_name(log_path.stem + "_occ.npz"), occ=np.stack(occ_series))
    return summary, log


if __name__ == "__main__":
    s, _ = run_episode(FixedTime(), "low", 0)
    print(s)