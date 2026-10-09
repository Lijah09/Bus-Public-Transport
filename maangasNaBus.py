"""
==============================================================================
EDSA CAROUSEL (SOUTHBOUND) - FIXED-FLEET BUS SIMULATION  (24-HOUR VERSION)
==============================================================================

THE QUESTION THIS CODE ANSWERS
    We have a FIXED number of buses. When should a bus leave a station?
    Two rules decide it (these are the two "knobs" we test):
        1. MIN_LOAD   -> minimum number of people on the bus before it may leave
        2. MAX_DWELL  -> maximum minutes a bus may stay at a station

    Why it's a trade-off:
        - Wait longer / wait for more people  -> fuller buses, BUT buses behind
          it get delayed, so everyone else waits longer.
        - Leave quickly                       -> short delays, BUT buses run
          emptier and people may be left behind.

HOW THE SIMULATION WORKS (in plain words)
    * Passengers randomly arrive at each of the 23 stations and line up in ONE
      FIRST-COME-FIRST-SERVED (FCFS) line: whoever arrived first boards first.
    * A fixed number of buses drive the route again and again:
        Station 1 -> Station 23 -> drive back to Station 1 -> repeat.
    * At each station a bus: enters the bay -> lets people off -> lets people
      on -> leaves when the departure rule says so.
    * Each station has only ONE bay, so a bus that arrives while another is
      still loading must wait (this is how bus bunching/delays appear).
    * We measure passenger waiting time, queue length, bus fullness, etc.
    * The measured period now covers a FULL 24 HOURS, with a different demand
      level for every hour (rush hours, midday dip, quiet overnight hours).

HOW TO USE
    1. Edit the settings in the "EXPERIMENT SETTINGS" and "Config" sections.
    2. Run:  maangasNaBus.py
    3. Read the printed table, then open the saved CSV and PNG files.

TIME UNIT: everything is in MINUTES.
==============================================================================
"""
import csv
import heapq
import math
import random
import statistics as stats
from dataclasses import dataclass, field, replace

import matplotlib.pyplot as plt
import numpy as np
import simpy
from matplotlib.ticker import PercentFormatter

# ==============================================================================
# SECTION 1 - EXPERIMENT SETTINGS (what combinations to test)
# ==============================================================================
# The program tries EVERY combination of MIN_LOAD_VALUES x MAX_DWELL_VALUES.
# Example: 5 x 5 = 25 combinations, each repeated REPLICATIONS times.
MIN_LOAD_VALUES = [45,50,52,55]         # "leave only if at least this many people are on board"
MAX_DWELL_VALUES = [1.5, 2.0, 3.0, 4.0, 5.0]  # "never stay longer than this many minutes"
FLEET_SIZES = [30]                            # number of buses in operation (fixed). Add more, e.g. [24, 30, 36], to compare
REPLICATIONS = 10                             # how many times each combination is repeated with different random numbers
                                              # (randomness means one run can be lucky/unlucky; more runs = more reliable. Use 30 for the paper)
                                              # TIP: a 24-hour run is 3x longer than the old 8-hour run. While testing,
                                              # set this to 2-3 so the program finishes quickly.
BASE_SEED = 1000                              # starting random seed (same seed -> same results every time you run)
COMPARE_QUEUE_DISCIPLINES = False             # True = also compare "pure first-come-first-served" vs "priority first" at the best setting
BUNCH_DECIMALS = 3                            # decimal places shown for "bunched departures %" (heatmap + console).
                                              # Bunching is tiny (<1%), so 1 decimal showed 0.0%. Raise this if it still shows 0.000%.

# How the program decides which combination is "best":
#   1) throw away combinations that break these two rules...
MAX_ALLOWED_QUEUE = 200        #    ...peak queue at any station must be <= this (platform safety limit)
MIN_LOAD_FACTOR = 0.40         #    ...buses must be at least this full on average (0.40 = 40% of seats/spaces used)
                               #    NOTE: now that overnight hours are included, the 24-hour average load factor
                               #    will be LOWER than before. If no combination passes, lower this value (e.g. 0.30).
#   2) among the rest, pick the one with the LOWEST average passenger wait.


# ==============================================================================
# SECTION 2 - MODEL SETTINGS (all the adjustable numbers live here)
# ==============================================================================
@dataclass
class Config:
    # --- The two main research variables (these get overwritten by the sweep) ---
    num_buses: int = 30           # fixed fleet size
    min_load: int = 45            # bus may leave once it has at least this many people on board (and nobody is waiting)
    max_dwell: float = 2.0        # bus must leave after this many minutes at the station, even if not full
                                  # (counted from entering the bay until it pulls out, including docking + door closing)

    # --- Simulation time ---
    warmup: float = 120.0         # first 120 min are "getting started" time (buses still spreading out, stations empty).
                                  # These minutes are NOT counted in the results.
    sim_time: float = 1440.0      # minutes that ARE measured (1440 = 24 hours)
    start_hour: int = 0           # clock hour at which the measured period begins (0 = 12:00 AM midnight).
                                  # Only used to label the charts/tables; hourly_profile[0] is this hour.

    # --- The route / network ---
    num_stations: int = 23        # southbound stations
    bays_per_station: int = 1     # how many buses can load at once at one station
    bus_capacities: tuple = (45, 50, 52, 55)   # each bus is randomly given one of these capacities
    travel_time: tuple = (1.5, 3.0)            # driving time between two stations: random between these (min, max)
    deadhead_time: tuple = (20.0, 30.0)        # driving time from the LAST station back to Station 1 (empty bus)
    dispatch_stagger: float = 3.0              # at the start, bus #0 leaves at 0 min, bus #1 at 3 min, bus #2 at 6 min...
                                               # (so all buses do not start at the same moment)

    # --- What happens at a station (time costs) ---
    dock_time: float = 0.5        # time to pull into the bay
    door_time: float = 0.5        # time to close doors and pull out
    board_time: float = 0.05      # time for ONE person to get on (0.05 min = 3 sec)
    alight_time: float = 0.03     # time for ONE person to get off (0.03 min = ~2 sec)
    poll_step: float = 0.1        # while a bus is waiting for more people, it re-checks the line every 0.1 min

    # --- Passengers ---
    priority_share: float = 0.20  # 20% of passengers are "priority" (senior, PWD, pregnant)
    queue_discipline: str = "fcfs"
        # "fcfs"     -> PURE FIRST-COME-FIRST-SERVED: everybody is in one line and boards strictly in the
        #               order they arrived (the priority tag is kept only so we can still measure how long
        #               those passengers waited - it does NOT change who boards first)
        # "priority" -> priority passengers board first; inside each group it is first-come-first-served
        #               (non-preemptive: nobody is kicked off a bus)

    # Base arrival rate = average number of NEW passengers per minute, per station.
    # List order = Station 1, 2, 3 ... 23.
    # Station 1 is the terminal (very busy). A few stations are busier than others. The last station is 0
    # because nobody boards there (it is the end of the line).
    # NOTE: these are PLACEHOLDER numbers - replace them with values from your data sources.
    station_rates: list = field(default_factory=lambda: (
        [4.0]                       # Station 1  (terminal)
        + [0.6] * 3                 # Stations 2-4
        + [1.2]                     # Station 5  (busy)
        + [0.6] * 3                 # Stations 6-8
        + [1.2]                     # Station 9  (busy)
        + [0.6] * 3                 # Stations 10-12
        + [1.2]                     # Station 13 (busy)
        + [0.6] * 8                 # Stations 14-21
        + [0.6, 0.0]                # Station 22, Station 23 (last = nobody boards)
    ))

    # Demand pattern for the WHOLE DAY: the base rate above is multiplied by this number, hour by hour.
    # 1.0 = normal, 1.5 = 50% more people, 0.3 = 70% fewer people.
    # EXACTLY 24 values, one per measured hour. With start_hour = 0 the list reads:
    #   index  0 -> 12:00 AM,  1 -> 1:00 AM, ... 6 -> 6:00 AM, ... 12 -> 12:00 PM, ... 23 -> 11:00 PM
    # The warm-up uses the first value.
    # NOTE: PLACEHOLDER shape (AM peak ~6-9 AM, midday dip, PM peak ~4-8 PM, quiet overnight).
    #       Replace with values derived from your data sources.
    hourly_profile: tuple = (
        0.3, 0.2, 0.2, 0.3,         # 12 AM - 3 AM  (overnight, very quiet)
        0.6, 1.0,                   #  4 AM - 5 AM  (service ramps up)
        1.5, 1.6, 1.4,              #  6 AM - 8 AM  (AM peak)
        1.0, 0.8, 0.8,              #  9 AM - 11 AM (easing off)
        0.8, 0.8, 0.8,              # 12 PM - 2 PM  (midday off-peak)
        1.0, 1.3,                   #  3 PM - 4 PM  (building up)
        1.6, 1.5, 1.2,              #  5 PM - 7 PM  (PM peak)
        0.9, 0.7, 0.5, 0.4,         #  8 PM - 11 PM (evening wind-down)
    )


# ==============================================================================
# SECTION 3 - THE MODEL (stations, buses, passengers)
# ==============================================================================
class Station:
    """One station: holds the waiting line and keeps the statistics for that station."""

    def __init__(self, env, sid, rate, cfg):
        self.id = sid                  # station number (1 to 23)
        self.rate = rate               # base passengers per minute arriving here
        # The bay = the loading spot. Only `bays_per_station` buses can use it at a time;
        # other buses wait their turn (first come, first served).
        self.bay = simpy.Resource(env, capacity=cfg.bays_per_station)

        # ONE queue per station (stored as a "heap"). Each passenger is stored as
        #     (rank, arrival_time, tie_breaker, is_priority)
        # and the heap always hands out the SMALLEST tuple first, which means:
        #     1) a lower rank comes out first (rank 0 only exists in "priority" mode; in "fcfs" mode
        #        everybody has rank 1, so rank never changes the order)
        #     2) inside the same rank, the EARLIEST arrival comes out first (= first-come-first-served)
        #     3) tie_breaker only matters if two people arrive at the exact same instant
        self.queue = []
        self.seq = 0                   # counter used for the tie_breaker

        # Per-hour statistics (one slot for each measured hour)
        n = int(cfg.sim_time // 60)
        self.h_wsum = [0.0] * n        # total waiting minutes of people who boarded in that hour
        self.h_wn = [0] * n            # number of people who boarded in that hour
        self.h_maxq = [0] * n          # longest line seen in that hour
        self.h_dsum = [0.0] * n        # total bus dwell minutes in that hour
        self.h_dn = [0] * n            # number of bus visits in that hour

        self.waits_pri = []            # waiting time of every priority passenger who boarded
        self.waits_reg = []            # waiting time of every regular passenger who boarded
        self.departures = []           # times when buses left this station (used to compute gaps between buses)

    def qlen(self):
        """People currently waiting here."""
        return len(self.queue)


class Bus:
    """One bus: its size and who is on board."""

    def __init__(self, bid, capacity, num_stations):
        self.id = bid
        self.cap = capacity                         # max people allowed on board
        self.load = 0                               # people on board right now
        # onboard[d] = how many people on this bus are going to get off at station d.
        # (Each passenger picks a destination when boarding, so they get off where they are headed.)
        self.onboard = [0] * (num_stations + 2)


class Metrics:
    """System-wide records collected during the run (only AFTER the warm-up)."""

    def __init__(self):
        self.platform_waits = []   # how long buses waited for a free bay
        self.dwells = []           # time a bus spent at a station (bay entry -> leaving)
        self.load_factors = []     # how full the bus was when it left (load / capacity)
        self.trip_times = []       # time for one full run Station 1 -> Station 23
        self.left_behind = []      # people still waiting right after a bus left
        self.boarded = 0           # total passengers who boarded


def hour_index(cfg, now):
    """Which measured hour is it? (0 = first hour after warm-up). Warm-up counts as hour 0."""
    return min(max(int((now - cfg.warmup) // 60), 0), len(cfg.hourly_profile) - 1)


def hour_label(cfg, h):
    """Clock-time label for measured hour h, e.g. '06:00' (uses cfg.start_hour)."""
    return f"{(cfg.start_hour + h) % 24:02d}:00"


def rate_multiplier(cfg, now):
    """Demand multiplier for the current time of day."""
    return cfg.hourly_profile[hour_index(cfg, now)]


def passenger_generator(env, st, cfg, rng):
    """Creates passengers at one station, forever.

    Arrivals are random (Poisson process): some minutes are busy, some quiet,
    but the long-run average matches the station rate x the hourly multiplier.
    The rate changes by hour, so we use a standard trick ("thinning"):
      1) generate arrivals at the HIGHEST possible rate,
      2) randomly skip some of them so the real rate matches the current hour.
    """
    if st.rate <= 0:
        return                                           # e.g. last station: nobody arrives
    max_rate = st.rate * max(cfg.hourly_profile)         # the highest rate used in step 1
    while True:
        # Wait a random time until the next potential arrival
        yield env.timeout(rng.expovariate(max_rate))

        # Step 2: skip this arrival with the right probability (so rate matches the hour)
        if rng.random() > st.rate * rate_multiplier(cfg, env.now) / max_rate:
            continue

        # Decide if this passenger is a priority passenger (random draw happens in BOTH
        # modes, so "priority" and "fcfs" runs use the exact same random passengers)
        is_pri = rng.random() < cfg.priority_share

        # Give the passenger a rank: 0 = served first, 1 = served after.
        #   "fcfs" mode    : EVERYONE gets rank 1, so only arrival time decides the order  (current setting)
        #   "priority" mode: priority passengers get rank 0, everyone else rank 1
        rank = 0 if (is_pri and cfg.queue_discipline == "priority") else 1
        heapq.heappush(st.queue, (rank, env.now, st.seq, is_pri))   # join the queue
        st.seq += 1

        # Remember the longest line seen this hour (only after warm-up)
        if env.now >= cfg.warmup:
            h = hour_index(cfg, env.now)
            q = st.qlen()
            if q > st.h_maxq[h]:
                st.h_maxq[h] = q


def bus_process(env, bus, stations, cfg, rng, m):
    """The life of ONE bus: drive the whole route, go back to the start, repeat forever.
    (Because every bus repeats forever, the number of buses never changes = FIXED fleet.)"""
    N = cfg.num_stations

    # Stagger the start so buses don't all leave Station 1 together
    yield env.timeout(bus.id * cfg.dispatch_stagger)

    while True:
        trip_start = env.now

        for st in stations:
            is_last = st.id == N

            # ---- 1. DRIVE to this station (no driving needed for Station 1) ----
            if st.id > 1:
                yield env.timeout(rng.uniform(*cfg.travel_time))

            # ---- 2. WAIT for the bay if another bus is still loading ----
            req_time = env.now
            with st.bay.request() as req:
                yield req                              # (continues once the bay is free)
                bay_start = env.now                    # dwell time is counted from this moment
                plat_wait = bay_start - req_time       # how long this bus was stuck waiting for the bay

                # ---- 3. PULL IN to the bay ----
                yield env.timeout(cfg.dock_time)

                # ---- 4. PASSENGERS GET OFF (only those whose destination is this station) ----
                n_off = bus.onboard[st.id]
                if n_off:
                    bus.load -= n_off
                    bus.onboard[st.id] = 0
                    yield env.timeout(n_off * cfg.alight_time)

                # ---- 5. PASSENGERS GET ON, following the DEPARTURE RULE ----
                if not is_last:                        # nobody boards at the last station
                    # The dwell cap includes door-closing time, so stop boarding early enough
                    # that doors can still close within max_dwell.
                    board_deadline = cfg.max_dwell - cfg.door_time
                    while bus.load < bus.cap:
                        # Rule A: max dwell time reached -> stop boarding and leave
                        if env.now - bay_start >= board_deadline:
                            break

                        if st.queue:
                            # Someone is waiting -> take the next person from the queue.
                            # heappop gives the lowest (rank, arrival_time). In "fcfs" mode every rank is
                            # the same, so this is simply the person who arrived earliest.
                            _, arr, _, pri = heapq.heappop(st.queue)
                            wait = env.now - arr       # how long this person waited in line
                            bus.load += 1
                            # Passenger picks a random destination further down the route
                            bus.onboard[rng.randint(st.id + 1, N)] += 1
                            # Save the waiting time (only for people who arrived after warm-up)
                            if arr >= cfg.warmup:
                                (st.waits_pri if pri else st.waits_reg).append(wait)
                                h = hour_index(cfg, env.now)
                                st.h_wsum[h] += wait
                                st.h_wn[h] += 1
                                m.boarded += 1
                            yield env.timeout(cfg.board_time)   # boarding takes a few seconds

                        elif bus.load >= cfg.min_load:
                            # Rule B: nobody waiting AND the bus has enough people -> leave
                            break
                        else:
                            # Rule C: nobody waiting but the bus is below MIN_LOAD
                            # -> hold the bus and check again shortly (until Rule A kicks in)
                            yield env.timeout(cfg.poll_step)
                    # Loop also ends if the bus is full (bus.load == bus.cap)

                # ---- 6. CLOSE DOORS and pull out ----
                yield env.timeout(cfg.door_time)
                dwell = env.now - bay_start            # total time the bus occupied the bay
            # (leaving the "with" block frees the bay for the next bus)

            # ---- 7. RECORD statistics (skip the warm-up period) ----
            if env.now >= cfg.warmup:
                h = hour_index(cfg, env.now)
                st.h_dsum[h] += dwell
                st.h_dn[h] += 1
                m.dwells.append(dwell)
                m.platform_waits.append(plat_wait)
                if not is_last:
                    m.load_factors.append(bus.load / bus.cap)   # how full the bus is when leaving
                    m.left_behind.append(st.qlen())             # people it left standing at the station
                    st.departures.append(env.now)               # used later to compute gaps between buses

        # Finished the whole route -> record trip time, then drive back to Station 1
        if env.now >= cfg.warmup:
            m.trip_times.append(env.now - trip_start)
        yield env.timeout(rng.uniform(*cfg.deadhead_time))


# ==============================================================================
# SECTION 4 - RUNNING THE SIMULATION AND COMPUTING RESULTS
# ==============================================================================
def pct(data, p):
    """p-th percentile (e.g. p=95 -> 95% of values are below this number)."""
    if not data:
        return 0.0
    return float(np.percentile(data, p))


def run_simulation(cfg, seed, keep_details=False):
    """Run ONE complete simulation and return a dictionary of result numbers.

    seed: the random-number starting point. Same seed + same settings = same result.
    keep_details: if True, also return the stations (needed for per-station/per-hour reports).
    """
    rng = random.Random(seed)
    env = simpy.Environment()
    stations = [Station(env, i + 1, cfg.station_rates[i], cfg) for i in range(cfg.num_stations)]
    m = Metrics()

    # Start the passenger generators (one per station) and the buses
    for st in stations:
        env.process(passenger_generator(env, st, cfg, rng))
    for b in range(cfg.num_buses):
        bus = Bus(b, rng.choice(cfg.bus_capacities), cfg.num_stations)
        env.process(bus_process(env, bus, stations, cfg, rng, m))

    # Run: warm-up + measured time
    end = cfg.warmup + cfg.sim_time
    env.run(until=end)

    # ---- Gather waiting times ----
    waits = []                                   # waits of people who actually boarded
    for st in stations:
        waits += st.waits_pri + st.waits_reg
    # People STILL in line when the simulation ended never got a bus. Ignoring them would make bad
    # scenarios look better than they are, so we count how long they have waited so far.
    stranded_waits = [end - a for st in stations for (_, a, _, _) in st.queue
                      if a >= cfg.warmup]
    all_waits = waits + stranded_waits

    # ---- Headway = time gap between two consecutive buses leaving the same station ----
    headways = []
    for st in stations:
        d = st.departures
        headways += [d[i + 1] - d[i] for i in range(len(d) - 1)]

    pri_w = [w for st in stations for w in st.waits_pri]
    reg_w = [w for st in stations for w in st.waits_reg]

    summary = {
        # ---- Passenger waiting time ----
        "avg_wait": float(np.mean(waits)) if waits else 0.0,                    # average wait (boarded people only)
        "p95_wait": pct(waits, 95),                                             # 95% of people waited less than this
        "avg_wait_incl_stranded": float(np.mean(all_waits)) if all_waits else 0.0,  # MAIN SCORE: includes people never served
        "avg_wait_priority": float(np.mean(pri_w)) if pri_w else 0.0,           # priority passengers only
        "avg_wait_regular": float(np.mean(reg_w)) if reg_w else 0.0,            # regular passengers only
        # ---- Queues ----
        "max_queue": max((max(st.h_maxq) for st in stations), default=0),       # longest line at any station
        "avg_left_behind": float(np.mean(m.left_behind)) if m.left_behind else 0.0,  # avg people left when a bus departs
        "stranded_at_end": len(stranded_waits),                                 # people still waiting when the sim ended
        # ---- Bus / station operations ----
        "avg_dwell": float(np.mean(m.dwells)) if m.dwells else 0.0,             # avg minutes a bus stays at a station
        "avg_platform_wait": float(np.mean(m.platform_waits)) if m.platform_waits else 0.0,  # avg minutes a bus waits for a free bay
        "avg_headway": float(np.mean(headways)) if headways else 0.0,           # avg minutes between buses
        "headway_cv": (float(np.std(headways) / np.mean(headways)) if headways else 0.0),  # how uneven the gaps are (0 = perfectly even)
        "bunching_pct": (100.0 * sum(h < 1.0 for h in headways) / len(headways)) if headways else 0.0,  # % of buses leaving <1 min after the previous one
        "load_factor": float(np.mean(m.load_factors)) if m.load_factors else 0.0,  # how full buses are (1.0 = completely full)
        "trip_time": float(np.mean(m.trip_times)) if m.trip_times else 0.0,     # minutes to run Station 1 -> 23
        "throughput_per_hr": m.boarded / (cfg.sim_time / 60.0),                 # passengers carried per hour
    }
    if keep_details:
        return summary, stations
    return summary


# --- Confidence interval helpers -------------------------------------------------
# A confidence interval shows how much to trust an average. "3.3 ± 0.8" means the true
# average is probably between 2.5 and 4.1. A smaller "±" means a more reliable result.
# T95 = standard statistics table values (the more runs, the smaller the number).
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
       9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042}


def t_crit(df):
    """Look up the table value for (number of runs - 1)."""
    if df < 1:
        return 0.0
    keys = [k for k in T95 if k <= df]
    return T95[max(keys)] if df < 31 else 1.96


def mean_ci(values):
    """Return (average, ± margin) for a list of numbers, at 95% confidence."""
    n = len(values)
    mu = sum(values) / n
    if n < 2:
        return mu, 0.0
    sd = stats.stdev(values)
    return mu, t_crit(n - 1) * sd / math.sqrt(n)


def run_experiment(base_cfg, min_loads, max_dwells, reps, seed0):
    """Test EVERY (min_load, max_dwell) combination, `reps` times each, and average the results.

    Fair comparison trick: run number r uses the SAME random seed for every combination, so the
    combinations face the same random passenger crowds. Differences then come from the rules, not luck.
    """
    results = {}
    total = len(min_loads) * len(max_dwells)
    k = 0
    for ml in min_loads:
        for md in max_dwells:
            k += 1
            cfg = replace(base_cfg, min_load=ml, max_dwell=md)
            runs = [run_simulation(cfg, seed0 + r) for r in range(reps)]
            # Average every metric over the repeated runs (and store its ± margin as key + "_ci")
            agg = {}
            for key in runs[0]:
                mu, ci = mean_ci([r[key] for r in runs])
                agg[key] = mu
                agg[key + "_ci"] = ci
            results[(ml, md)] = agg
            print(f"  [{k:>2}/{total}] min_load={ml:<3} max_dwell={md:<4} "
                  f"wait={agg['avg_wait_incl_stranded']:.2f}±{agg['avg_wait_incl_stranded_ci']:.2f} min | "
                  f"maxQ={agg['max_queue']:.0f} | LF={agg['load_factor']:.2f} | "
                  f"dwell={agg['avg_dwell']:.2f} | headway={agg['avg_headway']:.2f} | "
                  f"bunched={agg['bunching_pct']:.{BUNCH_DECIMALS}f}%")
    return results


def pick_best(results):
    """Choose the best combination:
       1) keep only combinations within the queue limit and minimum bus fullness,
       2) from those, take the one with the lowest average passenger wait.
       If NOTHING passes the limits, fall back to the lowest wait overall (and say so)."""
    feasible = {k: v for k, v in results.items()
                if v["max_queue"] <= MAX_ALLOWED_QUEUE and v["load_factor"] >= MIN_LOAD_FACTOR}
    pool = feasible if feasible else results
    best = min(pool, key=lambda k: pool[k]["avg_wait_incl_stranded"])
    return best, bool(feasible)


# ==============================================================================
# SECTION 5 - OUTPUT (CSV file, charts, tables)
# ==============================================================================
def save_csv(results, path):
    """Save every combination's results to a spreadsheet-friendly CSV file."""
    keys = list(next(iter(results.values())).keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["min_load", "max_dwell"] + keys)
        for (ml, md), v in results.items():
            w.writerow([ml, md] + [round(v[k], 4) for k in keys])
    print(f"Saved {path}")


def heatmaps(results, min_loads, max_dwells, best, fleet, path):
    """Draw colored grids: x-axis = MAX_DWELL, y-axis = MIN_LOAD, color/number = the result.
    The cell with the thick black box is the best combination.
    The bunched-departures panel is shown in PERCENT (e.g. 12.5%) instead of a plain number."""
    # (result key, panel title, color map, how to print the number in each cell)
    panels = [("avg_wait_incl_stranded", "Avg passenger wait (min)", "YlOrRd", "{:.1f}"),
              ("max_queue", "Peak platform queue (people)", "Reds", "{:.1f}"),
              ("load_factor", "Bus load factor", "YlGn", "{:.1f}"),
              ("avg_dwell", "Avg dwell time (min)", "Blues", "{:.1f}"),
              ("avg_headway", "Avg headway (min)", "Purples", "{:.1f}"),
              ("bunching_pct", "Bunched departures (% < 1 min)", "Oranges", "{:." + str(BUNCH_DECIMALS) + "f}%")]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, (key, title, cmap, fmt) in zip(axes.flat, panels):
        grid = np.array([[results[(ml, md)][key] for md in max_dwells] for ml in min_loads])
        im = ax.imshow(grid, cmap=cmap, aspect="auto", origin="lower")
        ax.set_xticks(range(len(max_dwells)))
        ax.set_xticklabels(max_dwells)
        ax.set_yticks(range(len(min_loads)))
        ax.set_yticklabels(min_loads)
        ax.set_xlabel("MAX_DWELL (min)")
        ax.set_ylabel("MIN_LOAD (people)")
        ax.set_title(title, fontsize=11, fontweight="bold")
        for i in range(len(min_loads)):
            for j in range(len(max_dwells)):
                ax.text(j, i, fmt.format(grid[i, j]), ha="center", va="center", fontsize=8)
        # Draw the black box around the best combination
        bi, bj = min_loads.index(best[0]), max_dwells.index(best[1])
        ax.add_patch(plt.Rectangle((bj - 0.5, bi - 0.5), 1, 1, fill=False, ec="black", lw=3))
        cbar = fig.colorbar(im, ax=ax)
        if key == "bunching_pct":
            # values are already in percent (0-100), so xmax=100 just adds the % sign to the ticks
            cbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=BUNCH_DECIMALS))
    fig.suptitle(f"Fleet = {fleet} buses  (black box = best configuration)", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    print(f"Saved {path}")


def detailed_report(cfg, seed, path, print_table=True):
    """Run ONE setting and show the details: a per-station, per-hour table (optional)
    and a 4-chart dashboard (waiting time and queue by hour, and by station)."""
    summary, stations = run_simulation(cfg, seed, keep_details=True)
    H = int(cfg.sim_time // 60)

    if print_table:
        print("\n" + "=" * 85)
        print(f"{'PER-STATION, PER-HOUR BREAKDOWN':^85}")
        print(f"{f'Fleet={cfg.num_buses} | Min Load={cfg.min_load} | Max Dwell={cfg.max_dwell}m':^85}")
        print("=" * 85)
        for st in stations:
            print(f"\n--- STATION {st.id} ---")
            print(f"{'Hour':<8} | {'Boarded':<8} | {'Avg Wait':<9} | {'Max Queue':<10} | {'Avg Dwell':<9}")
            for h in range(H):
                aw = st.h_wsum[h] / st.h_wn[h] if st.h_wn[h] else 0
                ad = st.h_dsum[h] / st.h_dn[h] if st.h_dn[h] else 0
                print(f"{hour_label(cfg, h):<8} | {st.h_wn[h]:<8} | {aw:<9.2f} | {st.h_maxq[h]:<10} | {ad:<9.2f}")

    # Combine all stations to get system-wide numbers for each hour (labelled with clock time)
    hours = [hour_label(cfg, h) for h in range(H)]
    sys_wait = []
    sys_q = []
    for h in range(H):
        n = sum(st.h_wn[h] for st in stations)
        sys_wait.append(sum(st.h_wsum[h] for st in stations) / n if n else 0)
        sys_q.append(max(st.h_maxq[h] for st in stations))
    # Whole-period numbers for each station
    labels = [f"S{st.id}" for st in stations]
    st_wait = []
    for st in stations:
        w = st.waits_pri + st.waits_reg
        st_wait.append(sum(w) / len(w) if w else 0)
    st_q = [max(st.h_maxq) for st in stations]

    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(16, 10))
    a1.plot(hours, sys_wait, marker="o", color="darkorange", lw=2.5)       # chart 1: wait by hour
    a1.set_title("System-wide hourly avg waiting time", fontweight="bold")
    a1.set_ylabel("Minutes")
    a1.set_xlabel("Hour of day")
    a1.tick_params(axis="x", rotation=60)
    a1.grid(True, ls="--", alpha=.7)
    a2.bar(hours, sys_q, color="crimson", alpha=.8, ec="black")           # chart 2: longest queue by hour
    a2.set_title("System-wide peak queue per hour", fontweight="bold")
    a2.set_ylabel("Passengers")
    a2.set_xlabel("Hour of day")
    a2.tick_params(axis="x", rotation=60)
    a2.grid(axis="y", ls="--", alpha=.7)
    a3.bar(labels, st_wait, color="royalblue", alpha=.8, ec="black")      # chart 3: wait by station
    a3.set_title("Avg wait per station (measured period)", fontweight="bold")
    a3.set_ylabel("Minutes")
    a3.tick_params(axis="x", rotation=45)
    a3.grid(axis="y", ls="--", alpha=.7)
    a4.bar(labels, st_q, color="forestgreen", alpha=.8, ec="black")       # chart 4: longest queue by station
    a4.set_title("Peak queue per station", fontweight="bold")
    a4.set_ylabel("Passengers")
    a4.tick_params(axis="x", rotation=45)
    a4.grid(axis="y", ls="--", alpha=.7)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    print(f"Saved {path}")
    return summary


def compare_disciplines(base_cfg, min_load, max_dwell, reps, seed0, path):
    """Same bus rules, same random passengers, but two different line rules:
         - Pure FCFS      : everyone boards strictly in order of arrival
         - Priority-first : priority passengers board before regular ones
    Shows how much the priority rule helps priority passengers and what it costs everyone else."""
    keys = ["avg_wait", "p95_wait", "avg_wait_priority", "avg_wait_regular",
            "avg_wait_incl_stranded", "max_queue", "load_factor", "avg_dwell"]
    out = {}
    for disc in ("fcfs", "priority"):
        cfg = replace(base_cfg, min_load=min_load, max_dwell=max_dwell, queue_discipline=disc)
        runs = [run_simulation(cfg, seed0 + r) for r in range(reps)]
        out[disc] = {k: mean_ci([r[k] for r in runs]) for k in keys}   # each value = (average, ± margin)

    # Print side-by-side table
    print("\n" + "=" * 78)
    print(f"QUEUE DISCIPLINE COMPARISON  (fleet={base_cfg.num_buses}, min_load={min_load}, "
          f"max_dwell={max_dwell}, {reps} reps)")
    print("=" * 78)
    print(f"{'Metric':<26} | {'Pure FCFS':^20} | {'Priority-first':^20}")
    print("-" * 78)
    for k in keys:
        a, b = out["fcfs"][k], out["priority"][k]
        print(f"{k:<26} | {a[0]:>9.2f} ± {a[1]:<7.2f} | {b[0]:>9.2f} ± {b[1]:<7.2f}")

    # Bar chart: waiting time of each passenger group under both rules
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = ["Priority passengers", "Regular passengers", "All passengers"]
    mkeys = ["avg_wait_priority", "avg_wait_regular", "avg_wait"]
    x = np.arange(len(groups))
    for i, (disc, color, label) in enumerate([("fcfs", "slategray", "Pure FCFS"),
                                              ("priority", "royalblue", "Priority-first")]):
        vals = [out[disc][k][0] for k in mkeys]
        errs = [out[disc][k][1] for k in mkeys]
        ax.bar(x + (i - 0.5) * 0.35, vals, 0.35, yerr=errs, capsize=4, label=label,
               color=color, alpha=.85, ec="black")
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("Avg wait (min)")
    ax.set_title("Effect of queue discipline on waiting time (95% CI)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", ls="--", alpha=.7)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    print(f"Saved {path}")
    return out


# ==============================================================================
# SECTION 6 - MAIN PROGRAM (runs when you execute the file)
# ==============================================================================
if __name__ == "__main__":
    overall = []
    for fleet in FLEET_SIZES:
        print(f"\n{'=' * 80}\nFLEET SIZE = {fleet} buses | {REPLICATIONS} replications per config | 24-hour day\n{'=' * 80}")
        base = Config(num_buses=fleet)

        # Step 1: test every MIN_LOAD x MAX_DWELL combination
        res = run_experiment(base, MIN_LOAD_VALUES, MAX_DWELL_VALUES, REPLICATIONS, BASE_SEED)

        # Step 2: pick and report the best combination
        best, ok = pick_best(res)
        b = res[best]
        print(f"\n>>> BEST for fleet {fleet}: MIN_LOAD={best[0]}, MAX_DWELL={best[1]} min"
              + ("" if ok else "   (no config met the constraints; showing lowest wait overall)"))
        print(f"    avg wait (incl. unserved) = {b['avg_wait_incl_stranded']:.2f} ± {b['avg_wait_incl_stranded_ci']:.2f} min"
              f" | p95 wait = {b['p95_wait']:.1f} | peak queue = {b['max_queue']:.0f}"
              f" | load factor = {b['load_factor']:.2f} | dwell = {b['avg_dwell']:.2f}"
              f" | headway = {b['avg_headway']:.2f} | bunching = {b['bunching_pct']:.{BUNCH_DECIMALS}f}%")

        # Step 3: save results (CSV + heatmap charts)
        save_csv(res, f"results_fleet{fleet}.csv")
        heatmaps(res, MIN_LOAD_VALUES, MAX_DWELL_VALUES, best, fleet, f"heatmaps_fleet{fleet}.png")
        overall.append((fleet, best, b))

        # Step 4 (optional): compare pure FCFS vs priority-first at the best setting
        if COMPARE_QUEUE_DISCIPLINES:
            compare_disciplines(base, best[0], best[1], REPLICATIONS, BASE_SEED,
                                f"discipline_comparison_fleet{fleet}.png")

        # Step 5: per-station / per-hour dashboard for the best setting
        detailed_report(replace(base, min_load=best[0], max_dwell=best[1]),
                        BASE_SEED, f"dashboard_fleet{fleet}.png", print_table=False)
        # (set print_table=True above to also print the long table for all 23 stations)

    # If several fleet sizes were tested, summarize how the best setting changes with fleet size
    if len(overall) > 1:
        print("\nFLEET SENSITIVITY SUMMARY")
        for fleet, best, b in overall:
            print(f"  fleet={fleet:<3} best=(min_load={best[0]}, max_dwell={best[1]}) "
                  f"wait={b['avg_wait_incl_stranded']:.2f} min, peakQ={b['max_queue']:.0f}, LF={b['load_factor']:.2f}")
    plt.show()