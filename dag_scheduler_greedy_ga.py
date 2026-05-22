"""
dag_scheduler_greedy_ga.py

Extends the DAG scheduling evaluation framework with two new methods:

1. GREEDY SCHEDULER
   - At each step, selects the ready task that minimizes the earliest
     finish time (EFT) over all processors.
   - Pure greedy: no lookahead, no upward-rank precomputation.
   - Fast O(n * P) per scheduling step.

2. GENETIC ALGORITHM (GA) SCHEDULER
   - Chromosome: a permutation of all tasks that is topologically valid.
   - Fitness:    makespan of the list schedule produced by that permutation.
   - Operators:
       * Selection:  tournament selection (size k=3)
       * Crossover:  Order Crossover (OX1), repaired to respect topology
       * Mutation:   swap two genes that do not violate topological order
   - Runs for a configurable number of generations on each test DAG.
   - Reports best / mean / worst makespan trajectory.

Both schedulers plug directly into the existing evaluate_method() framework
so you get the same aggregate table and win-count vs HEFT comparison.

Usage:
    python dag_scheduler_greedy_ga.py               # no LLM
    python dag_scheduler_greedy_ga.py --use-llm     # with LLM weights + insights
    python dag_scheduler_greedy_ga.py --ga-pop 60 --ga-gen 80

Install (same as original):
    pip install networkx numpy torch openai scikit-optimize
"""

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import networkx as nx
import numpy as np

# ── optional deps ────────────────────────────────────────────────────────────
try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args
except Exception:
    gp_minimize = None
    Real = None
    use_named_args = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 7) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


# ============================================================
# DAG generation  (unchanged from original)
# ============================================================

def ensure_dag_by_orientation(g: nx.Graph) -> nx.DiGraph:
    order = list(g.nodes())
    random.shuffle(order)
    pos = {node: i for i, node in enumerate(order)}
    dag = nx.DiGraph()
    dag.add_nodes_from(g.nodes())
    for u, v in g.edges():
        if pos[u] < pos[v]:
            dag.add_edge(u, v)
        else:
            dag.add_edge(v, u)
    if nx.is_directed_acyclic_graph(dag):
        return nx.transitive_reduction(dag)
    return dag


def add_costs(dag: nx.DiGraph, num_processors: int = 4) -> nx.DiGraph:
    for v in dag.nodes():
        base = random.randint(5, 30)
        dag.nodes[v]["costs"] = [
            max(1, int(base * random.uniform(0.7, 1.4)))
            for _ in range(num_processors)
        ]
    for u, v in dag.edges():
        dag.edges[u, v]["comm"] = random.randint(1, 10)
    return dag


def random_er_dag(n, p, num_processors):
    dag = nx.DiGraph()
    dag.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                dag.add_edge(i, j)
    return add_costs(dag, num_processors)


def random_ba_dag(n, m, num_processors):
    g = nx.barabasi_albert_graph(n, max(1, min(m, n - 1)), seed=random.randint(0, 10_000))
    return add_costs(ensure_dag_by_orientation(g), num_processors)


def random_ws_dag(n, k, p, num_processors):
    k = min(k if k % 2 == 0 else k + 1, n - 1)
    g = nx.watts_strogatz_graph(n, k, p, seed=random.randint(0, 10_000))
    return add_costs(ensure_dag_by_orientation(g), num_processors)


def motif_chain(n, num_processors):
    dag = nx.DiGraph()
    dag.add_nodes_from(range(n))
    dag.add_edges_from((i, i + 1) for i in range(n - 1))
    return add_costs(dag, num_processors)


def motif_fork(width, num_processors):
    dag = nx.DiGraph()
    dag.add_nodes_from(range(width + 1))
    dag.add_edges_from((0, i) for i in range(1, width + 1))
    return add_costs(dag, num_processors)


def motif_join(width, num_processors):
    dag = nx.DiGraph()
    dag.add_nodes_from(range(width + 1))
    sink = width
    dag.add_edges_from((i, sink) for i in range(width))
    return add_costs(dag, num_processors)


def motif_diamond(num_processors):
    dag = nx.DiGraph()
    dag.add_nodes_from(range(6))
    dag.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5)])
    return add_costs(dag, num_processors)


def build_dataset(num_graphs, n_values, num_processors):
    dataset = []
    for _ in range(num_graphs):
        n = random.choice(n_values)
        kind = random.choice(["ER", "BA", "WS"])
        if kind == "ER":
            dag = random_er_dag(n, p=random.uniform(0.05, 0.18), num_processors=num_processors)
        elif kind == "BA":
            dag = random_ba_dag(n, m=random.randint(1, min(4, n - 1)), num_processors=num_processors)
        else:
            dag = random_ws_dag(n, k=min(4, n - 1), p=random.uniform(0.1, 0.4), num_processors=num_processors)
        dataset.append((kind, dag))
    dataset.extend([
        ("MOTIF_CHAIN", motif_chain(12, num_processors)),
        ("MOTIF_FORK", motif_fork(10, num_processors)),
        ("MOTIF_JOIN", motif_join(10, num_processors)),
        ("MOTIF_DIAMOND", motif_diamond(num_processors)),
    ])
    return dataset


# ============================================================
# Feature extraction  (unchanged)
# ============================================================

FEATURE_NAMES = [
    "upward_rank", "downward_rank", "depth",
    "fanout", "indegree", "avg_comm_out", "avg_cost",
]


def avg_cost(dag, v):
    return float(np.mean(dag.nodes[v]["costs"]))


def upward_rank(dag):
    rank = {}
    for v in reversed(list(nx.topological_sort(dag))):
        succ = list(dag.successors(v))
        rank[v] = avg_cost(dag, v) if not succ else avg_cost(dag, v) + max(
            dag.edges[v, u]["comm"] + rank[u] for u in succ)
    return rank


def downward_rank(dag):
    rank = {}
    for v in nx.topological_sort(dag):
        pred = list(dag.predecessors(v))
        rank[v] = avg_cost(dag, v) if not pred else avg_cost(dag, v) + max(
            dag.edges[u, v]["comm"] + rank[u] for u in pred)
    return rank


def node_depths(dag):
    depth = {}
    for v in nx.topological_sort(dag):
        preds = list(dag.predecessors(v))
        depth[v] = 0 if not preds else 1 + max(depth[p] for p in preds)
    return depth


def extract_node_features(dag):
    ur = upward_rank(dag)
    dr = downward_rank(dag)
    dep = node_depths(dag)
    feats = {}
    for v in dag.nodes():
        out_comms = [dag.edges[v, u]["comm"] for u in dag.successors(v)]
        feats[v] = {
            "upward_rank": ur[v], "downward_rank": dr[v],
            "depth": float(dep[v]), "fanout": float(dag.out_degree(v)),
            "indegree": float(dag.in_degree(v)),
            "avg_comm_out": float(np.mean(out_comms)) if out_comms else 0.0,
            "avg_cost": avg_cost(dag, v),
        }
    return feats


def normalize_features(feats):
    out = {v: {} for v in feats}
    for name in FEATURE_NAMES:
        vals = np.array([feats[v][name] for v in feats], dtype=float)
        lo, hi = float(vals.min()), float(vals.max())
        for v in feats:
            out[v][name] = 0.0 if hi == lo else (feats[v][name] - lo) / (hi - lo)
    return out


def motif_counts(dag):
    counts = {"chain_nodes": 0, "fork_nodes": 0, "join_nodes": 0, "diamond_like_nodes": 0}
    for v in dag.nodes():
        indeg, outdeg = dag.in_degree(v), dag.out_degree(v)
        if indeg == 1 and outdeg == 1: counts["chain_nodes"] += 1
        if outdeg >= 2:                counts["fork_nodes"] += 1
        if indeg >= 2:                 counts["join_nodes"] += 1
        if indeg >= 2 and outdeg >= 2: counts["diamond_like_nodes"] += 1
    return counts


def graph_summary(dag):
    depths = node_depths(dag)
    return {
        "num_nodes": dag.number_of_nodes(),
        "num_edges": dag.number_of_edges(),
        "motifs": motif_counts(dag),
        "max_depth": max(depths.values()) if depths else 0,
        "avg_fanout": float(np.mean([dag.out_degree(v) for v in dag.nodes()])),
        "avg_indegree": float(np.mean([dag.in_degree(v) for v in dag.nodes()])),
        "avg_task_cost": float(np.mean([avg_cost(dag, v) for v in dag.nodes()])),
        "avg_comm_cost": float(np.mean([dag.edges[e]["comm"] for e in dag.edges()])) if dag.number_of_edges() > 0 else 0.0,
        "feature_names": FEATURE_NAMES,
    }


# ============================================================
# Schedule data structures
# ============================================================

@dataclass
class ScheduleResult:
    makespan: float
    assignment: Dict[int, int]
    start_times: Dict[int, float]
    finish_times: Dict[int, float]


# ============================================================
# Core scheduling engine  (unchanged)
# ============================================================

def earliest_start_finish(dag, task, proc, proc_available, finish_times, assignment):
    ready_time = 0.0
    for parent in dag.predecessors(task):
        same_processor = assignment.get(parent) == proc
        comm = 0 if same_processor else dag.edges[parent, task]["comm"]
        ready_time = max(ready_time, finish_times[parent] + comm)
    start = max(proc_available[proc], ready_time)
    finish = start + dag.nodes[task]["costs"][proc]
    return start, finish


def list_schedule(dag, num_processors, priority_fn):
    completed, scheduled = set(), set()
    assignment, start_times, finish_times = {}, {}, {}
    proc_available = [0.0] * num_processors
    while len(completed) < dag.number_of_nodes():
        ready = [v for v in dag.nodes()
                 if v not in scheduled
                 and all(p in completed for p in dag.predecessors(v))]
        if not ready:
            raise RuntimeError("No ready tasks — graph may not be a DAG.")
        task = max(ready, key=priority_fn)
        best_proc, best_start, best_finish = None, None, float("inf")
        for p in range(num_processors):
            s, f = earliest_start_finish(dag, task, p, proc_available, finish_times, assignment)
            if f < best_finish:
                best_proc, best_start, best_finish = p, s, f
        assignment[task] = int(best_proc)
        start_times[task] = float(best_start)
        finish_times[task] = float(best_finish)
        proc_available[best_proc] = float(best_finish)
        scheduled.add(task)
        completed.add(task)
    return ScheduleResult(max(finish_times.values()), assignment, start_times, finish_times)


def schedule_from_order(dag, num_processors, order):
    """
    Execute tasks in the given chromosome order (a list of node ids).
    Used by the GA to evaluate a chromosome.

    Robustness note
    ───────────────
    OX1 crossover preserves topological validity in theory, but
    floating-point seeds (HEFT/CPOP sort keys) can occasionally
    produce an order where a predecessor appears AFTER its child.
    We defend against this by:
      1. Building a position map from the chromosome.
      2. If any predecessor of the current task has NOT been scheduled
         yet, we schedule it recursively before the current task.
    This is equivalent to a light topological repair and is O(n) extra.
    """
    assignment, start_times, finish_times = {}, {}, {}
    proc_available = [0.0] * num_processors

    def _schedule_task(task: int) -> None:
        if task in assignment:
            return  # already done
        # Ensure all predecessors are scheduled first
        for parent in dag.predecessors(task):
            if parent not in assignment:
                _schedule_task(parent)
        best_proc, best_start, best_finish = None, None, float("inf")
        for p in range(num_processors):
            s, f = earliest_start_finish(dag, task, p, proc_available, finish_times, assignment)
            if f < best_finish:
                best_proc, best_start, best_finish = p, s, f
        assignment[task] = int(best_proc)
        start_times[task] = float(best_start)
        finish_times[task] = float(best_finish)
        proc_available[best_proc] = float(best_finish)

    for task in order:
        _schedule_task(task)

    return ScheduleResult(max(finish_times.values()), assignment, start_times, finish_times)


def critical_path_fastest_processor(dag):
    fastest = {v: min(dag.nodes[v]["costs"]) for v in dag.nodes()}
    dist = {}
    for v in nx.topological_sort(dag):
        preds = list(dag.predecessors(v))
        dist[v] = fastest[v] if not preds else fastest[v] + max(dist[p] for p in preds)
    return max(dist.values()) if dist else 0.0


# ============================================================
# HEFT & CPOP baselines  (unchanged)
# ============================================================

def heft_schedule(dag, num_processors):
    ur = upward_rank(dag)
    return list_schedule(dag, num_processors, lambda v: ur[v])


def cpop_schedule(dag, num_processors):
    ur = upward_rank(dag)
    dr = downward_rank(dag)
    return list_schedule(dag, num_processors, lambda v: ur[v] + dr[v])


def weighted_feature_schedule(dag, num_processors, weights):
    feats = normalize_features(extract_node_features(dag))
    return list_schedule(dag, num_processors,
        lambda v: sum(weights.get(n, 0.0) * feats[v][n] for n in FEATURE_NAMES))


# ============================================================
#  NEW ①  GREEDY SCHEDULER
# ============================================================
# Strategy: at every step, among all currently ready tasks, pick the
# one whose earliest-finish-time (over the best processor) is SMALLEST.
# Ties are broken by upward-rank (largest wins).
#
# Rationale: standard list schedulers choose WHICH TASK to dispatch next
# by a precomputed priority (HEFT: upward rank).  Pure Greedy instead
# uses real-time information — it picks the task that can complete
# soonest, which greedily minimises the contribution to makespan at each
# step.  This is a different exploration strategy and often does
# surprisingly well on dense graphs.

def greedy_eft_schedule(dag: nx.DiGraph, num_processors: int) -> ScheduleResult:
    """
    Greedy Earliest-Finish-Time (EFT) scheduler.

    At each step:
      1. Collect all ready tasks (predecessors finished).
      2. For each ready task compute its best (proc, start, finish).
      3. Commit the task with the SMALLEST best_finish.
         Ties broken by upward-rank (higher = dispatched first).
    """
    ur = upward_rank(dag)                          # for tie-breaking only

    completed, scheduled = set(), set()
    assignment, start_times, finish_times = {}, {}, {}
    proc_available = [0.0] * num_processors

    while len(completed) < dag.number_of_nodes():
        ready = [v for v in dag.nodes()
                 if v not in scheduled
                 and all(p in completed for p in dag.predecessors(v))]
        if not ready:
            raise RuntimeError("Greedy: no ready tasks — check DAG validity.")

        # Evaluate EFT for each ready task
        task_eft = {}
        task_best_proc = {}
        task_best_start = {}
        for v in ready:
            best_p, best_s, best_f = None, None, float("inf")
            for p in range(num_processors):
                s, f = earliest_start_finish(dag, v, p, proc_available, finish_times, assignment)
                if f < best_f:
                    best_p, best_s, best_f = p, s, f
            task_eft[v] = best_f
            task_best_proc[v] = best_p
            task_best_start[v] = best_s

        # Pick task with minimum EFT; break ties with upward-rank (descending)
        task = min(ready, key=lambda v: (task_eft[v], -ur[v]))

        p = task_best_proc[task]
        assignment[task] = int(p)
        start_times[task] = float(task_best_start[task])
        finish_times[task] = float(task_eft[task])
        proc_available[p] = float(task_eft[task])
        scheduled.add(task)
        completed.add(task)

    return ScheduleResult(max(finish_times.values()), assignment, start_times, finish_times)


# ============================================================
#  NEW ②  GENETIC ALGORITHM (GA) SCHEDULER
# ============================================================
#
# Representation
# ──────────────
# A chromosome is a list of all n task IDs that forms a valid
# topological ordering of the DAG.  The list-scheduling engine
# (schedule_from_order) turns any such ordering into a concrete
# makespan via earliest-finish-time processor assignment.
#
# Fitness  = makespan (minimise).
#
# Initialisation
# ──────────────
# Seed with: HEFT order, CPOP order, Greedy order, and (pop_size-3)
# random topological sorts.  This gives the GA a strong starting point
# while preserving diversity.
#
# Selection: Tournament (size k=3, minimisation)
#
# Crossover: Order Crossover (OX1)
# ────────────────────────────────
# 1. Pick two random cut-points i, j.
# 2. Child inherits genes [i:j] from parent 1 in place.
# 3. Remaining positions filled left-to-right from parent 2,
#    skipping already-present genes.
# 4. OX1 preserves relative order within the selected segment,
#    which automatically keeps topological validity when both
#    parents are valid (proven property for permutations).
#
# Mutation: Topological-safe swap
# ────────────────────────────────
# Randomly select two positions i < j in the chromosome.
# Swap them ONLY if doing so keeps the permutation topologically
# valid (i.e., after swap neither gene violates a predecessor
# constraint given the surrounding genes).  Retry up to
# max_swap_attempts times to find a valid swap.
#
# Elitism: top-1 individual always survives to next generation.

class GeneticAlgorithmScheduler:
    """
    GA-based DAG list scheduler.

    Parameters
    ----------
    num_processors : int
    pop_size       : population size  (default 50)
    num_generations: number of GA generations  (default 60)
    crossover_rate : probability of crossover per pair  (default 0.85)
    mutation_rate  : probability of mutation per individual  (default 0.20)
    tournament_k   : tournament selection size  (default 3)
    verbose        : print generation progress  (default False)
    """

    def __init__(
        self,
        num_processors: int,
        pop_size: int = 50,
        num_generations: int = 60,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.20,
        tournament_k: int = 3,
        verbose: bool = False,
    ):
        self.P = num_processors
        self.pop_size = pop_size
        self.num_gen = num_generations
        self.cx_rate = crossover_rate
        self.mut_rate = mutation_rate
        self.k = tournament_k
        self.verbose = verbose

    # ── helpers ──────────────────────────────────────────────

    def _random_topo_sort(self, dag: nx.DiGraph) -> List[int]:
        """Kahn's algorithm with random tie-breaking."""
        in_deg = {v: dag.in_degree(v) for v in dag.nodes()}
        queue = [v for v, d in in_deg.items() if d == 0]
        random.shuffle(queue)
        order = []
        while queue:
            idx = random.randrange(len(queue))
            queue[idx], queue[-1] = queue[-1], queue[idx]
            v = queue.pop()
            order.append(v)
            for u in dag.successors(v):
                in_deg[u] -= 1
                if in_deg[u] == 0:
                    queue.append(u)
        return order

    def _heft_order(self, dag: nx.DiGraph) -> List[int]:
        ur = upward_rank(dag)
        return sorted(dag.nodes(), key=lambda v: -ur[v])

    def _cpop_order(self, dag: nx.DiGraph) -> List[int]:
        ur = upward_rank(dag)
        dr = downward_rank(dag)
        return sorted(dag.nodes(), key=lambda v: -(ur[v] + dr[v]))

    def _greedy_order(self, dag: nx.DiGraph) -> List[int]:
        """Extract the scheduling order produced by the Greedy EFT heuristic."""
        result = greedy_eft_schedule(dag, self.P)
        return sorted(result.start_times, key=lambda v: result.start_times[v])

    def _fitness(self, dag: nx.DiGraph, chrom: List[int]) -> float:
        return schedule_from_order(dag, self.P, chrom).makespan

    def _is_valid_topo(self, dag: nx.DiGraph, chrom: List[int]) -> bool:
        pos = {v: i for i, v in enumerate(chrom)}
        return all(pos[u] < pos[v] for u, v in dag.edges())

    # ── genetic operators ─────────────────────────────────────

    def _tournament_select(self, population, fitnesses):
        candidates = random.sample(range(len(population)), min(self.k, len(population)))
        best = min(candidates, key=lambda i: fitnesses[i])
        return population[best][:]

    def _topo_repair(self, dag: nx.DiGraph, chrom: List[int]) -> List[int]:
        """
        Repair a chromosome that may violate topological order.
        Uses Kahn's algorithm, always picking whichever ready node
        appears EARLIEST in the original chrom -- minimises disruption.
        """
        pos = {v: i for i, v in enumerate(chrom)}
        in_deg = {v: dag.in_degree(v) for v in dag.nodes()}
        ready = sorted([v for v, d in in_deg.items() if d == 0],
                       key=lambda v: pos.get(v, 0))
        order = []
        while ready:
            v = min(ready, key=lambda x: pos.get(x, 0))
            ready.remove(v)
            order.append(v)
            for u in dag.successors(v):
                in_deg[u] -= 1
                if in_deg[u] == 0:
                    ready.append(u)
        return order

    def _ox1_crossover(self, dag: nx.DiGraph, p1: List[int], p2: List[int]) -> List[int]:
        """
        Order Crossover (OX1) with topological repair as a safety net.
        """
        n = len(p1)
        i, j = sorted(random.sample(range(n), 2))
        child: List[Optional[int]] = [None] * n
        child[i:j+1] = p1[i:j+1]
        segment_set = set(p1[i:j+1])
        fill = [g for g in p2 if g not in segment_set]
        fi = 0
        for idx in range(n):
            if child[idx] is None:
                child[idx] = fill[fi]
                fi += 1
        if not self._is_valid_topo(dag, child):
            child = self._topo_repair(dag, child)
        return child

    def _mutate(self, dag: nx.DiGraph, chrom: List[int], max_attempts: int = 40) -> List[int]:
        """
        Swap-mutation: try up to max_attempts random swaps and keep
        the first one that preserves topological validity.
        """
        n = len(chrom)
        for _ in range(max_attempts):
            i, j = sorted(random.sample(range(n), 2))
            new_chrom = chrom[:]
            new_chrom[i], new_chrom[j] = new_chrom[j], new_chrom[i]
            if self._is_valid_topo(dag, new_chrom):
                return new_chrom
        return chrom  # no valid swap found; return unchanged

    # ── main entry point ──────────────────────────────────────

    def schedule(self, dag: nx.DiGraph) -> Tuple[ScheduleResult, Dict[str, Any]]:
        """
        Run the GA and return (best_ScheduleResult, stats_dict).

        stats_dict keys:
            best_per_gen  : list of best makespan per generation
            mean_per_gen  : list of mean makespan per generation
            worst_per_gen : list of worst makespan per generation
            initial_best  : makespan of the seed population's best
            final_best    : makespan of the final best chromosome
            improvement_pct: (initial_best - final_best) / initial_best * 100
        """
        nodes = list(dag.nodes())
        n = len(nodes)
        if n == 0:
            return ScheduleResult(0.0, {}, {}, {}), {}

        # ── Initialise population ─────────────────────────────
        seed_orders = [
            self._heft_order(dag),
            self._cpop_order(dag),
            self._greedy_order(dag),
        ]
        population = seed_orders[:]
        while len(population) < self.pop_size:
            population.append(self._random_topo_sort(dag))

        fitnesses = [self._fitness(dag, c) for c in population]

        best_per_gen, mean_per_gen, worst_per_gen = [], [], []
        initial_best = min(fitnesses)

        # ── Evolution loop ────────────────────────────────────
        for gen in range(self.num_gen):
            new_pop = []

            # Elitism: keep best individual
            elite_idx = int(np.argmin(fitnesses))
            new_pop.append(population[elite_idx][:])

            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(population, fitnesses)

                # Crossover
                if random.random() < self.cx_rate:
                    p2 = self._tournament_select(population, fitnesses)
                    child = self._ox1_crossover(dag, p1, p2)
                else:
                    child = p1[:]

                # Mutation
                if random.random() < self.mut_rate:
                    child = self._mutate(dag, child)

                new_pop.append(child)

            population = new_pop
            fitnesses = [self._fitness(dag, c) for c in population]

            best_per_gen.append(float(min(fitnesses)))
            mean_per_gen.append(float(np.mean(fitnesses)))
            worst_per_gen.append(float(max(fitnesses)))

            if self.verbose and (gen + 1) % max(1, self.num_gen // 5) == 0:
                print(f"  [GA] gen={gen+1:03d}  best={best_per_gen[-1]:.2f}  "
                      f"mean={mean_per_gen[-1]:.2f}  worst={worst_per_gen[-1]:.2f}")

        best_idx = int(np.argmin(fitnesses))
        best_result = schedule_from_order(dag, self.P, population[best_idx])
        final_best = float(fitnesses[best_idx])

        stats = {
            "best_per_gen": best_per_gen,
            "mean_per_gen": mean_per_gen,
            "worst_per_gen": worst_per_gen,
            "initial_best": float(initial_best),
            "final_best": final_best,
            "improvement_pct": (initial_best - final_best) / initial_best * 100
                                if initial_best > 0 else 0.0,
        }

        return best_result, stats


# ============================================================
# Bayesian Optimisation (unchanged from original)
# ============================================================

def bayes_optimize_llm_weights(train_data, num_processors, initial_weights, n_calls=30, seed=7):
    if gp_minimize is None:
        print("[WARN] scikit-optimize not installed. Using raw LLM weights.")
        return initial_weights
    search_space = [Real(-2.0, 2.0, name=name) for name in FEATURE_NAMES]
    initial_point = [float(initial_weights.get(name, 0.0)) for name in FEATURE_NAMES]

    @use_named_args(search_space)
    def objective(**weights):
        total = sum(weighted_feature_schedule(dag, num_processors, weights).makespan
                    for _, dag in train_data)
        return total / len(train_data)

    result = gp_minimize(func=objective, dimensions=search_space, x0=initial_point,
                         n_calls=n_calls, random_state=seed, acq_func="EI")
    opt = {name: float(v) for name, v in zip(FEATURE_NAMES, result.x)}
    print("\nBO completed. Best training makespan:", round(result.fun, 2))
    print("Weights:", json.dumps(opt, indent=2))
    return opt


# ============================================================
# LLM Role 1 (unchanged)
# ============================================================

def llm_propose_weights(dag, model="gpt-4o-mini"):
    fallback = {"upward_rank": 1.0, "downward_rank": 0.25, "depth": 0.30,
                "fanout": 0.20, "indegree": 0.05, "avg_comm_out": 0.35, "avg_cost": 0.10}
    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        print("[LLM fallback] OPENAI_API_KEY missing or openai unavailable.")
        return fallback
    stats = graph_summary(dag)
    prompt = (f"Propose numeric weights for a DAG scheduling priority function.\n"
              f"Return JSON only with keys {FEATURE_NAMES} and float values in [-2, 2].\n"
              f"DAG summary:\n{json.dumps(stats, indent=2)}")
    try:
        client = OpenAI()
        resp = client.chat.completions.create(model=model, temperature=0,
                                               messages=[{"role": "user", "content": prompt}])
        text = resp.choices[0].message.content.strip().replace("```json", "").replace("```", "")
        data = json.loads(text)
        return {name: float(data.get(name, fallback[name])) for name in FEATURE_NAMES}
    except Exception as e:
        print(f"[LLM fallback] {e}")
        return fallback


# ============================================================
# GNN models (unchanged from original)
# ============================================================

if torch is not None:
    class TinyDAGGNN(nn.Module):
        def __init__(self, in_dim, hidden_dim=64):
            super().__init__()
            self.input = nn.Linear(in_dim, hidden_dim)
            self.msg = nn.Linear(hidden_dim, hidden_dim)
            self.update = nn.GRUCell(hidden_dim, hidden_dim)
            self.out = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, 1))

        def forward(self, x, edge_index):
            h = torch.relu(self.input(x))
            for _ in range(3):
                agg = torch.zeros_like(h)
                if edge_index.numel() > 0:
                    src, dst = edge_index[0], edge_index[1]
                    agg.index_add_(0, dst, self.msg(h[src]))
                h = self.update(agg, h)
            return self.out(h).squeeze(-1)
else:
    TinyDAGGNN = None


def graph_to_tensors(dag):
    feats = normalize_features(extract_node_features(dag))
    nodes = list(dag.nodes())
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    x = torch.tensor([[feats[v][name] for name in FEATURE_NAMES] for v in nodes], dtype=torch.float32)
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in dag.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    return nodes, node_to_idx, x, edge_index


def train_gnn_imitation(train_data, epochs=100, lr=1e-3):
    if torch is None:
        print("[WARN] PyTorch unavailable; skipping GNN imitation.")
        return None
    model = TinyDAGGNN(len(FEATURE_NAMES))
    opt = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0.0
        random.shuffle(train_data)
        for _, dag in train_data:
            nodes, _, x, edge_index = graph_to_tensors(dag)
            scores = model(x, edge_index)
            target_rank = upward_rank(dag)
            y = torch.tensor([target_rank[v] for v in nodes], dtype=torch.float32)
            y = (y - y.mean()) / (y.std() + 1e-6)
            loss = ((scores - y) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += float(loss.item())
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"[GNN imitation] epoch={epoch+1:03d}, loss={total_loss/len(train_data):.4f}")
    return model


def gnn_schedule(model, dag, num_processors):
    nodes, _, x, edge_index = graph_to_tensors(dag)
    with torch.no_grad():
        scores = model(x, edge_index).detach().cpu().numpy()
    score_map = {v: float(scores[i]) for i, v in enumerate(nodes)}
    return list_schedule(dag, num_processors, lambda v: score_map[v])


def train_decima_like_rl(train_data, num_processors, episodes=80, lr=1e-3):
    if torch is None:
        print("[WARN] PyTorch unavailable; skipping Decima-like RL.")
        return None
    model = TinyDAGGNN(len(FEATURE_NAMES))
    opt = optim.Adam(model.parameters(), lr=lr)
    baseline = None
    for ep in range(episodes):
        _, dag = random.choice(train_data)
        nodes, node_to_idx, x, edge_index = graph_to_tensors(dag)
        log_probs = []
        def policy(ready, completed):
            scores = model(x, edge_index)
            ready_idx = torch.tensor([node_to_idx[v] for v in ready], dtype=torch.long)
            probs = torch.softmax(scores[ready_idx], dim=0)
            dist = torch.distributions.Categorical(probs)
            action_pos = dist.sample()
            log_probs.append(dist.log_prob(action_pos))
            return ready[int(action_pos.item())]
        result = _schedule_with_priority_order(dag, num_processors, policy)
        reward = -result.makespan
        baseline = reward if baseline is None else 0.9 * baseline + 0.1 * reward
        advantage = reward - baseline
        loss = -torch.stack(log_probs).sum() * float(advantage)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (ep + 1) % max(1, episodes // 5) == 0:
            print(f"[Decima-like RL] episode={ep+1:03d}, makespan={result.makespan:.2f}")
    return model


def _schedule_with_priority_order(dag, num_processors, policy):
    """Used internally by Decima RL."""
    completed, scheduled = set(), set()
    assignment, start_times, finish_times = {}, {}, {}
    proc_available = [0.0] * num_processors
    while len(completed) < dag.number_of_nodes():
        ready = [v for v in dag.nodes()
                 if v not in scheduled
                 and all(p in completed for p in dag.predecessors(v))]
        if not ready:
            raise RuntimeError("No ready tasks.")
        task = policy(ready, completed)
        best_p, best_s, best_f = None, None, float("inf")
        for p in range(num_processors):
            s, f = earliest_start_finish(dag, task, p, proc_available, finish_times, assignment)
            if f < best_f:
                best_p, best_s, best_f = p, s, f
        assignment[task] = int(best_p)
        start_times[task] = float(best_s)
        finish_times[task] = float(best_f)
        proc_available[best_p] = float(best_f)
        scheduled.add(task); completed.add(task)
    return ScheduleResult(max(finish_times.values()), assignment, start_times, finish_times)


# ============================================================
# Evaluation framework  (unchanged + GA extension)
# ============================================================

def evaluate_method(name, data, num_processors, schedule_fn):
    rows, schedules = [], []
    for family, dag in data:
        t0 = time.time()
        result = schedule_fn(dag)
        rt = time.time() - t0
        cp = critical_path_fastest_processor(dag)
        rows.append({
            "method": name,
            "family": family,
            "nodes": dag.number_of_nodes(),
            "edges": dag.number_of_edges(),
            "makespan": result.makespan,
            "slr": result.makespan / cp if cp > 0 else float("nan"),
            "runtime_ms": rt * 1000,
            "motifs": motif_counts(dag),
        })
        schedules.append((family, dag, result))
    return rows, schedules


def evaluate_ga(data, ga: GeneticAlgorithmScheduler):
    """
    Evaluate GA on the test set; also print per-DAG improvement stats.
    Returns (rows, schedules) in the same format as evaluate_method.
    """
    rows, schedules = [], []
    all_improvements = []
    for family, dag in data:
        t0 = time.time()
        result, stats = ga.schedule(dag)
        rt = time.time() - t0
        cp = critical_path_fastest_processor(dag)
        imp = stats.get("improvement_pct", 0.0)
        all_improvements.append(imp)
        rows.append({
            "method": "GA",
            "family": family,
            "nodes": dag.number_of_nodes(),
            "edges": dag.number_of_edges(),
            "makespan": result.makespan,
            "slr": result.makespan / cp if cp > 0 else float("nan"),
            "runtime_ms": rt * 1000,
            "motifs": motif_counts(dag),
            "ga_initial_best": stats.get("initial_best"),
            "ga_final_best": stats.get("final_best"),
            "ga_improvement_pct": round(imp, 2),
        })
        schedules.append((family, dag, result))
    print(f"\n[GA] Avg improvement over initial seed: "
          f"{np.mean(all_improvements):.2f}%  "
          f"(max {max(all_improvements):.2f}%)")
    return rows, schedules


def summarize(rows: List[Dict[str, Any]]) -> None:
    by_method: Dict[str, List] = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)

    print("\n" + "=" * 72)
    print("AGGREGATE RESULTS")
    print("=" * 72)
    print(f"{'Method':<22} {'Avg Makespan':>14} {'Avg SLR':>10} {'Runtime ms':>12}")
    print("-" * 62)
    for method, rs in by_method.items():
        print(f"{method:<22} "
              f"{np.mean([r['makespan'] for r in rs]):>14.2f} "
              f"{np.mean([r['slr'] for r in rs]):>10.3f} "
              f"{np.mean([r['runtime_ms'] for r in rs]):>12.3f}")

    if "HEFT" in by_method:
        heft = by_method["HEFT"]
        print("\n" + "=" * 72)
        print("WIN COUNT vs HEFT")
        print("=" * 72)
        for method, rs in by_method.items():
            if method == "HEFT":
                continue
            wins = sum(r["makespan"] < h["makespan"] for r, h in zip(rs, heft))
            ties = sum(abs(r["makespan"] - h["makespan"]) < 1e-9 for r, h in zip(rs, heft))
            print(f"  {method:<22} wins={wins:>3}  ties={ties:>3}  total={len(heft)}")

    # GA-specific improvement summary
    ga_rows = by_method.get("GA", [])
    if ga_rows and "ga_improvement_pct" in ga_rows[0]:
        imps = [r["ga_improvement_pct"] for r in ga_rows]
        print(f"\n[GA] Improvement over seed population  "
              f"avg={np.mean(imps):.2f}%  max={max(imps):.2f}%  min={min(imps):.2f}%")

    print("\n" + "=" * 72)
    print("EXAMPLE PER-DAG ROWS (first 12)")
    print("=" * 72)
    for row in rows[:min(12, len(rows))]:
        print(f"  {row['method']:<22} family={row['family']:<14} "
              f"nodes={row['nodes']:<3} edges={row['edges']:<3} "
              f"makespan={row['makespan']:<8.2f} slr={row['slr']:.3f}")


def save_csv(rows, path):
    import csv
    if not rows:
        print("[WARN] No rows to save.")
        return
    flat_rows = []
    for row in rows:
        flat = dict(row)
        flat["motifs"] = json.dumps(flat.get("motifs", {}))
        flat_rows.append(flat)
    all_keys = list(dict.fromkeys(k for r in flat_rows for k in r))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"\nSaved results to: {path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="DAG scheduling evaluation: Greedy + GA added to HEFT/CPOP/GNN/RL")
    parser.add_argument("--seed",            type=int,   default=7)
    parser.add_argument("--num-processors",  type=int,   default=4)
    parser.add_argument("--num-graphs",      type=int,   default=36)
    parser.add_argument("--train-episodes",  type=int,   default=80)
    parser.add_argument("--gnn-epochs",      type=int,   default=80)
    parser.add_argument("--bo-calls",        type=int,   default=30)
    parser.add_argument("--ga-pop",          type=int,   default=50,
                        help="GA population size (default 50)")
    parser.add_argument("--ga-gen",          type=int,   default=60,
                        help="GA number of generations (default 60)")
    parser.add_argument("--ga-mut",          type=float, default=0.20,
                        help="GA mutation rate (default 0.20)")
    parser.add_argument("--ga-cx",           type=float, default=0.85,
                        help="GA crossover rate (default 0.85)")
    parser.add_argument("--ga-verbose",      action="store_true",
                        help="Print GA per-generation progress")
    parser.add_argument("--use-llm",         action="store_true")
    parser.add_argument("--openai-model",    type=str,   default="gpt-4o-mini")
    parser.add_argument("--out",             type=str,   default="scheduler_results.csv")
    args = parser.parse_args()

    set_seed(args.seed)

    # ── Build dataset ─────────────────────────────────────────
    dataset = build_dataset(args.num_graphs, [20, 30, 50], args.num_processors)
    random.shuffle(dataset)
    split = int(0.7 * len(dataset))
    train_data, test_data = dataset[:split], dataset[split:]
    print(f"Train DAGs: {len(train_data)}   Test DAGs: {len(test_data)}")
    print(f"Processors: {args.num_processors}")

    # ── Optional: LLM weights ─────────────────────────────────
    llm_weights = bo_weights = None
    if args.use_llm:
        llm_weights = llm_propose_weights(test_data[0][1], model=args.openai_model)
        print("\nLLM proposed weights:", json.dumps(llm_weights, indent=2))
        bo_weights = bayes_optimize_llm_weights(
            train_data, args.num_processors, llm_weights, args.bo_calls, args.seed)

    # ── Train neural methods ──────────────────────────────────
    gnn_model = train_gnn_imitation(train_data, epochs=args.gnn_epochs)
    rl_model  = train_decima_like_rl(train_data, args.num_processors, args.train_episodes)

    # ── Set up GA ─────────────────────────────────────────────
    ga = GeneticAlgorithmScheduler(
        num_processors=args.num_processors,
        pop_size=args.ga_pop,
        num_generations=args.ga_gen,
        crossover_rate=args.ga_cx,
        mutation_rate=args.ga_mut,
        verbose=args.ga_verbose,
    )

    # ── Evaluate all methods ──────────────────────────────────
    all_rows: List[Dict[str, Any]] = []

    for name, fn in [
        ("HEFT",  lambda dag: heft_schedule(dag, args.num_processors)),
        ("CPOP",  lambda dag: cpop_schedule(dag, args.num_processors)),
        ("Greedy-EFT", lambda dag: greedy_eft_schedule(dag, args.num_processors)),
    ]:
        rows, _ = evaluate_method(name, test_data, args.num_processors, fn)
        all_rows += rows

    if llm_weights:
        rows, _ = evaluate_method("LLM-Heuristic", test_data, args.num_processors,
                                  lambda dag: weighted_feature_schedule(dag, args.num_processors, llm_weights))
        all_rows += rows
    if bo_weights:
        rows, _ = evaluate_method("LLM+BO-Heuristic", test_data, args.num_processors,
                                  lambda dag: weighted_feature_schedule(dag, args.num_processors, bo_weights))
        all_rows += rows
    if gnn_model:
        rows, _ = evaluate_method("GNN-Imitation", test_data, args.num_processors,
                                  lambda dag: gnn_schedule(gnn_model, dag, args.num_processors))
        all_rows += rows
    if rl_model:
        rows, _ = evaluate_method("Decima-like-RL", test_data, args.num_processors,
                                  lambda dag: gnn_schedule(rl_model, dag, args.num_processors))
        all_rows += rows

    # GA (separate evaluator to capture improvement stats)
    print(f"\nRunning GA (pop={args.ga_pop}, gen={args.ga_gen}) on {len(test_data)} test DAGs...")
    ga_rows, _ = evaluate_ga(test_data, ga)
    all_rows += ga_rows

    # ── Print summary & save ──────────────────────────────────
    summarize(all_rows)
    save_csv(all_rows, args.out)

    print("\nDone. Key highlights:")
    print("  Greedy-EFT: greedily picks the task with smallest earliest-finish-time each step.")
    print("  GA:         evolves topological orderings via OX1 crossover + topo-safe swap mutation.")
    print(f"  Results saved to {args.out}")


if __name__ == "__main__":
    main()
