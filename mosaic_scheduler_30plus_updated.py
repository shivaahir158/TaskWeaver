"""
mosaic_scheduler_30plus.py

Aggressive modular DAG scheduling framework for MoSAIC-style experiments.

Goal
----
This version is written to make >30% improvement over HEFT more achievable by
combining:
  1. Classical baselines: HEFT, CPOP, Greedy-EFT
  2. LLM weight generation: one optional OpenAI call
  3. Bayesian Optimization: global interpretable feature-weight search
  4. Multi-start weighted heuristics
  5. Per-DAG Genetic Algorithm search over priority weights
  6. Optional local search / simulated annealing refinement
  7. Oracle-style HYBRID-BEST reporting: best valid schedule among all enabled methods

Important honesty note
----------------------
No scheduler can guarantee >30% improvement over HEFT on every DAG. If HEFT is
already close to the critical-path lower bound, a 30% gain is mathematically
impossible. This code therefore reports both:
  - improvement over HEFT, and
  - distance to the critical-path lower bound.

Install
-------
pip install networkx numpy scipy scikit-optimize openai

Optional:
pip install torch

Run strong setting
------------------
python mosaic_scheduler_30plus.py --use-llm --bo-calls 80 --ga-generations 120 --ga-pop-size 96 --sa-steps 800

Run without LLM
---------------
python mosaic_scheduler_30plus.py --bo-calls 80 --ga-generations 120 --ga-pop-size 96 --sa-steps 800
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args
except Exception:  # pragma: no cover
    gp_minimize = None
    Real = None
    use_named_args = None

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover
    wilcoxon = None

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:  # pragma: no cover
    torch = None
    nn = None
    optim = None


# ============================================================
# Configuration
# ============================================================

FEATURE_NAMES = [
    "upward_rank",
    "downward_rank",
    "depth",
    "fanout",
    "indegree",
    "avg_comm_out",
    "avg_cost",
    "min_cost",
    "max_cost",
    "cost_spread",
    "succ_count_weighted",
    "pred_count_weighted",
    "comm_pressure",
    "mobility_proxy",
]

BASE_FEATURE_NAMES = [
    "upward_rank",
    "downward_rank",
    "depth",
    "fanout",
    "indegree",
    "avg_comm_out",
    "avg_cost",
]


@dataclass
class ScheduleResult:
    makespan: float
    assignment: Dict[int, int]
    start_times: Dict[int, float]
    finish_times: Dict[int, float]
    method: str = ""
    weights: Optional[Dict[str, float]] = None


@dataclass
class BOResult:
    weights: Dict[str, float]
    best_train_makespan: float
    convergence_curve: List[float] = field(default_factory=list)


@dataclass
class MethodSummary:
    method: str
    avg_makespan: float
    avg_slr: float
    avg_runtime_ms: float
    avg_improvement_vs_heft: Optional[float]
    win_rate_vs_heft: Optional[float]


# ============================================================
# Reproducibility and safe utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def mean(xs):
    xs = np.asarray(xs)

    if xs.size == 0:
        return float("nan")

    return float(np.mean(xs))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_json_loads(text: str) -> Optional[dict]:
    try:
        text = text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception:
        return None


# ============================================================
# DAG generation
# ============================================================

def ensure_dag_by_orientation(g: nx.Graph) -> nx.DiGraph:
    order = list(g.nodes())
    random.shuffle(order)
    pos = {node: i for i, node in enumerate(order)}
    dag = nx.DiGraph()
    dag.add_nodes_from(g.nodes())
    for u, v in g.edges():
        dag.add_edge(u, v) if pos[u] < pos[v] else dag.add_edge(v, u)
    return nx.transitive_reduction(dag) if nx.is_directed_acyclic_graph(dag) else dag


def add_costs(dag: nx.DiGraph, num_processors: int = 4) -> nx.DiGraph:
    for v in dag.nodes():
        base = random.randint(5, 30)

        dag.nodes[v]["costs"] = [
            max(1, int(base * random.uniform(0.3, 2.5)))
            for _ in range(num_processors)
        ]

    for u, v in dag.edges():
        dag.edges[u, v]["comm"] = random.randint(20, 80)

    return dag


def random_er_dag(n: int, p: float, num_processors: int) -> nx.DiGraph:
    dag = nx.DiGraph()
    dag.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                dag.add_edge(i, j)
    return add_costs(dag, num_processors)


def random_ba_dag(n: int, m: int, num_processors: int) -> nx.DiGraph:
    g = nx.barabasi_albert_graph(n, max(1, min(m, n - 1)), seed=random.randint(0, 1_000_000))
    return add_costs(ensure_dag_by_orientation(g), num_processors)


def random_ws_dag(n: int, k: int, p: float, num_processors: int) -> nx.DiGraph:
    k = min(k if k % 2 == 0 else k + 1, n - 1)
    g = nx.watts_strogatz_graph(n, k, p, seed=random.randint(0, 1_000_000))
    return add_costs(ensure_dag_by_orientation(g), num_processors)


def motif_chain(n: int, num_processors: int) -> nx.DiGraph:
    dag = nx.DiGraph()
    dag.add_nodes_from(range(n))
    dag.add_edges_from((i, i + 1) for i in range(n - 1))
    return add_costs(dag, num_processors)


def motif_fork(width: int, num_processors: int) -> nx.DiGraph:
    dag = nx.DiGraph()
    dag.add_nodes_from(range(width + 1))
    dag.add_edges_from((0, i) for i in range(1, width + 1))
    return add_costs(dag, num_processors)


def motif_join(width: int, num_processors: int) -> nx.DiGraph:
    dag = nx.DiGraph()
    dag.add_nodes_from(range(width + 1))
    sink = width
    dag.add_edges_from((i, sink) for i in range(width))
    return add_costs(dag, num_processors)


def motif_diamond(num_processors: int) -> nx.DiGraph:
    dag = nx.DiGraph()
    dag.add_nodes_from(range(6))
    dag.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (3, 5)])
    return add_costs(dag, num_processors)


def build_dataset(num_graphs: int, n_values: List[int], num_processors: int) -> List[Tuple[str, nx.DiGraph]]:
    dataset: List[Tuple[str, nx.DiGraph]] = []
    for _ in range(num_graphs):
        n = random.choice(n_values)
        family = random.choice(["ER", "BA", "WS"])
        if family == "ER":
            dag = random_er_dag(n, random.uniform(0.04, 0.20), num_processors)
        elif family == "BA":
            dag = random_ba_dag(n, random.randint(1, min(5, n - 1)), num_processors)
        else:
            dag = random_ws_dag(n, min(6, n - 1), random.uniform(0.1, 0.45), num_processors)
        dataset.append((family, dag))

    dataset.extend([
        ("MOTIF_CHAIN", motif_chain(16, num_processors)),
        ("MOTIF_FORK", motif_fork(14, num_processors)),
        ("MOTIF_JOIN", motif_join(14, num_processors)),
        ("MOTIF_DIAMOND", motif_diamond(num_processors)),
    ])
    return dataset


# ============================================================
# Feature extraction
# ============================================================

def avg_cost(dag: nx.DiGraph, v: int) -> float:
    return float(np.mean(dag.nodes[v]["costs"]))


def min_cost(dag: nx.DiGraph, v: int) -> float:
    return float(min(dag.nodes[v]["costs"]))


def max_cost(dag: nx.DiGraph, v: int) -> float:
    return float(max(dag.nodes[v]["costs"]))


def upward_rank(dag: nx.DiGraph) -> Dict[int, float]:
    rank: Dict[int, float] = {}
    for v in reversed(list(nx.topological_sort(dag))):
        succ = list(dag.successors(v))
        rank[v] = avg_cost(dag, v) if not succ else avg_cost(dag, v) + max(dag.edges[v, u]["comm"] + rank[u] for u in succ)
    return rank


def downward_rank(dag: nx.DiGraph) -> Dict[int, float]:
    rank: Dict[int, float] = {}
    for v in nx.topological_sort(dag):
        pred = list(dag.predecessors(v))
        rank[v] = avg_cost(dag, v) if not pred else avg_cost(dag, v) + max(dag.edges[u, v]["comm"] + rank[u] for u in pred)
    return rank


def node_depths(dag: nx.DiGraph) -> Dict[int, int]:
    depth: Dict[int, int] = {}
    for v in nx.topological_sort(dag):
        preds = list(dag.predecessors(v))
        depth[v] = 0 if not preds else 1 + max(depth[p] for p in preds)
    return depth


def extract_node_features(dag: nx.DiGraph) -> Dict[int, Dict[str, float]]:
    ur = upward_rank(dag)
    dr = downward_rank(dag)
    depth = node_depths(dag)
    max_depth = max(depth.values()) if depth else 1

    feats: Dict[int, Dict[str, float]] = {}

    for v in dag.nodes():
        out_comms = [dag.edges[v, u]["comm"] for u in dag.successors(v)]
        ac = avg_cost(dag, v)
        mic = min_cost(dag, v)
        mac = max_cost(dag, v)

        avg_comm = float(np.mean(out_comms)) if out_comms else 0.0
        fanout = float(dag.out_degree(v))
        indegree = float(dag.in_degree(v))

        feats[v] = {
            "upward_rank": ur[v],
            "downward_rank": dr[v],
            "depth": float(depth[v]),
            "fanout": fanout,
            "indegree": indegree,
            "avg_comm_out": avg_comm,
            "avg_cost": ac,
            "min_cost": mic,
            "max_cost": mac,
            "cost_spread": mac - mic,
            "succ_count_weighted": float(
                sum(1.0 + dag.edges[v, u]["comm"] for u in dag.successors(v))
            ),
            "pred_count_weighted": float(
                sum(1.0 + dag.edges[u, v]["comm"] for u in dag.predecessors(v))
            ),
            "mobility_proxy": float(
                (max_depth - depth[v]) + fanout - 0.5 * indegree
            ),
            "comm_pressure": float(fanout * avg_comm),
        }

    return feats


def normalize_features(feats: Dict[int, Dict[str, float]]) -> Dict[int, Dict[str, float]]:
    out = {v: {} for v in feats}
    for name in FEATURE_NAMES:
        values = np.array([feats[v][name] for v in feats], dtype=float)
        lo, hi = float(values.min()), float(values.max())
        for v in feats:
            out[v][name] = 0.0 if hi == lo else (feats[v][name] - lo) / (hi - lo)
    return out


def motif_counts(dag: nx.DiGraph) -> Dict[str, int]:
    counts = {"chain_nodes": 0, "fork_nodes": 0, "join_nodes": 0, "diamond_like_nodes": 0}
    for v in dag.nodes():
        indeg, outdeg = dag.in_degree(v), dag.out_degree(v)
        if indeg == 1 and outdeg == 1:
            counts["chain_nodes"] += 1
        if outdeg >= 2:
            counts["fork_nodes"] += 1
        if indeg >= 2:
            counts["join_nodes"] += 1
        if indeg >= 2 and outdeg >= 2:
            counts["diamond_like_nodes"] += 1
    return counts


def graph_summary(dag: nx.DiGraph) -> Dict[str, Any]:
    depths = node_depths(dag)
    return {
        "num_nodes": dag.number_of_nodes(),
        "num_edges": dag.number_of_edges(),
        "max_depth": max(depths.values()) if depths else 0,
        "avg_fanout": float(np.mean([dag.out_degree(v) for v in dag.nodes()])),
        "avg_indegree": float(np.mean([dag.in_degree(v) for v in dag.nodes()])),
        "avg_task_cost": float(np.mean([avg_cost(dag, v) for v in dag.nodes()])),
        "avg_comm_cost": float(np.mean([dag.edges[e]["comm"] for e in dag.edges()])) if dag.number_of_edges() else 0.0,
        "motifs": motif_counts(dag),
        "features": FEATURE_NAMES,
    }


# ============================================================
# Lower bound and schedule construction
# ============================================================

def critical_path_fastest_processor(dag: nx.DiGraph) -> float:
    fastest = {v: min_cost(dag, v) for v in dag.nodes()}
    dist: Dict[int, float] = {}
    for v in nx.topological_sort(dag):
        preds = list(dag.predecessors(v))
        dist[v] = fastest[v] if not preds else fastest[v] + max(dist[p] for p in preds)
    return max(dist.values()) if dist else 0.0


def workload_lower_bound(dag: nx.DiGraph, num_processors: int) -> float:
    return sum(min_cost(dag, v) for v in dag.nodes()) / max(1, num_processors)


def combined_lower_bound(dag: nx.DiGraph, num_processors: int) -> float:
    return max(critical_path_fastest_processor(dag), workload_lower_bound(dag, num_processors))


def earliest_start_finish(
    dag: nx.DiGraph,
    task: int,
    proc: int,
    proc_available: List[float],
    finish_times: Dict[int, float],
    assignment: Dict[int, int],
) -> Tuple[float, float]:
    ready_time = 0.0
    for parent in dag.predecessors(task):
        comm = 0.0 if assignment.get(parent) == proc else float(dag.edges[parent, task]["comm"])
        ready_time = max(ready_time, finish_times[parent] + comm)
    start = max(proc_available[proc], ready_time)
    finish = start + dag.nodes[task]["costs"][proc]
    return start, finish


def choose_best_processor(
    dag: nx.DiGraph,
    task: int,
    proc_available: List[float],
    finish_times: Dict[int, float],
    assignment: Dict[int, int],
    tie_break: str = "eft",
) -> Tuple[int, float, float]:
    candidates = []
    for p in range(len(proc_available)):
        st, ft = earliest_start_finish(dag, task, p, proc_available, finish_times, assignment)
        candidates.append((ft, st, proc_available[p], dag.nodes[task]["costs"][p], p))
    # Lower finish first; then lower start; then less busy proc; then lower proc id.
    best = min(candidates, key=lambda x: (x[0], x[1], x[2], x[4]))
    return int(best[4]), float(best[1]), float(best[0])


def list_schedule(dag: nx.DiGraph, num_processors: int, priority_fn: Callable[[int], float], method: str = "list") -> ScheduleResult:
    completed: set[int] = set()
    scheduled: set[int] = set()
    assignment: Dict[int, int] = {}
    start_times: Dict[int, float] = {}
    finish_times: Dict[int, float] = {}
    proc_available = [0.0] * num_processors

    while len(completed) < dag.number_of_nodes():
        ready = [v for v in dag.nodes() if v not in scheduled and all(p in completed for p in dag.predecessors(v))]
        if not ready:
            raise RuntimeError("No ready tasks found; graph may not be a DAG.")
        task = max(ready, key=lambda v: (priority_fn(v), -v))
        proc, st, ft = choose_best_processor(dag, task, proc_available, finish_times, assignment)
        assignment[task] = proc
        start_times[task] = st
        finish_times[task] = ft
        proc_available[proc] = ft
        scheduled.add(task)
        completed.add(task)

    return ScheduleResult(max(finish_times.values()) if finish_times else 0.0, assignment, start_times, finish_times, method=method)


def schedule_from_static_order(dag: nx.DiGraph, num_processors: int, order_scores: Dict[int, float], method: str) -> ScheduleResult:
    return list_schedule(dag, num_processors, lambda v: order_scores[v], method=method)


def schedule_with_ready_policy(
    dag: nx.DiGraph,
    num_processors: int,
    policy_fn: Callable[[List[int], set[int]], int],
    method: str,
) -> ScheduleResult:
    completed: set[int] = set()
    scheduled: set[int] = set()
    assignment: Dict[int, int] = {}
    start_times: Dict[int, float] = {}
    finish_times: Dict[int, float] = {}
    proc_available = [0.0] * num_processors

    while len(completed) < dag.number_of_nodes():
        ready = [
            v for v in dag.nodes()
            if v not in scheduled and all(p in completed for p in dag.predecessors(v))
        ]
        if not ready:
            raise RuntimeError("No ready tasks found; graph may not be a DAG.")

        task = policy_fn(ready, completed)
        if task not in ready:
            task = ready[0]

        proc, st, ft = choose_best_processor(
            dag, task, proc_available, finish_times, assignment
        )
        assignment[task] = proc
        start_times[task] = st
        finish_times[task] = ft
        proc_available[proc] = ft
        scheduled.add(task)
        completed.add(task)

    return ScheduleResult(
        max(finish_times.values()) if finish_times else 0.0,
        assignment,
        start_times,
        finish_times,
        method=method,
    )


# ============================================================
# Baselines and weighted heuristics
# ============================================================

def heft_schedule(dag: nx.DiGraph, num_processors: int) -> ScheduleResult:
    ur = upward_rank(dag)
    return list_schedule(dag, num_processors, lambda v: ur[v], method="HEFT")


def cpop_schedule(dag: nx.DiGraph, num_processors: int) -> ScheduleResult:
    ur, dr = upward_rank(dag), downward_rank(dag)
    return list_schedule(dag, num_processors, lambda v: ur[v] + dr[v], method="CPOP")


def greedy_eft_schedule(dag: nx.DiGraph, num_processors: int) -> ScheduleResult:
    # Schedule the ready task whose best possible EFT is smallest. Useful baseline, not usually best.
    completed: set[int] = set()
    scheduled: set[int] = set()
    assignment: Dict[int, int] = {}
    start_times: Dict[int, float] = {}
    finish_times: Dict[int, float] = {}
    proc_available = [0.0] * num_processors

    while len(completed) < dag.number_of_nodes():
        ready = [v for v in dag.nodes() if v not in scheduled and all(p in completed for p in dag.predecessors(v))]
        def best_eft(v: int) -> float:
            return min(earliest_start_finish(dag, v, p, proc_available, finish_times, assignment)[1] for p in range(num_processors))
        task = min(ready, key=lambda v: (best_eft(v), v))
        proc, st, ft = choose_best_processor(dag, task, proc_available, finish_times, assignment)
        assignment[task], start_times[task], finish_times[task] = proc, st, ft
        proc_available[proc] = ft
        scheduled.add(task)
        completed.add(task)
    return ScheduleResult(max(finish_times.values()), assignment, start_times, finish_times, method="Greedy-EFT")


def weighted_feature_schedule(dag: nx.DiGraph, num_processors: int, weights: Dict[str, float], method: str = "Weighted") -> ScheduleResult:
    feats = normalize_features(extract_node_features(dag))
    result = list_schedule(
        dag,
        num_processors,
        lambda v: sum(weights.get(name, 0.0) * feats[v][name] for name in FEATURE_NAMES),
        method=method,
    )
    result.weights = dict(weights)
    return result


def default_llm_fallback_weights() -> Dict[str, float]:
    # Stronger than the original fallback: critical path + communication + processor sensitivity.
    return {
        "upward_rank": 1.75,
        "downward_rank": 0.35,
        "depth": 0.25,
        "fanout": 0.45,
        "indegree": -0.10,
        "avg_comm_out": 0.70,
        "avg_cost": 0.25,
        "min_cost": 0.10,
        "max_cost": 0.15,
        "comm_pressure": 3.0,
        "cost_spread": 0.30,
        "succ_count_weighted": 0.55,
        "pred_count_weighted": -0.15,
        "mobility_proxy": 0.30,
    }


# ============================================================
# LLM weight generation
# ============================================================

def llm_propose_weights(dag: nx.DiGraph, model: str = "gpt-4o-mini") -> Dict[str, float]:
    fallback = default_llm_fallback_weights()
    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        print("[LLM fallback] Missing OPENAI_API_KEY or openai package. Using engineered fallback weights.")
        return fallback

    prompt = f"""
You are designing an interpretable priority function for DAG list scheduling.
Return JSON only. No prose.

Priority: H(v)=sum_i theta_i * normalized_feature_i(v). Higher H(v) schedules earlier.

Feature names:
{FEATURE_NAMES}

Rules:
- Every key must be present.
- Values must be floats in [-3, 3].
- Favor critical path, communication-heavy successors, and tasks with processor-cost sensitivity.
- Penalize features only if scheduling them early may create synchronization/resource contention.
- The goal is to beat HEFT by a large margin when possible.

DAG summary:
{json.dumps(graph_summary(dag), indent=2)}
"""
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        parsed = safe_json_loads(response.choices[0].message.content or "")
        if not parsed:
            return fallback
        return {name: clamp(float(parsed.get(name, fallback[name])), -3.0, 3.0) for name in FEATURE_NAMES}
    except Exception as exc:
        print(f"[LLM fallback] OpenAI call failed: {exc}. Using engineered fallback weights.")
        return fallback


# ============================================================
# Bayesian Optimization
# ============================================================

def evaluate_weights_on_dataset(data: List[Tuple[str, nx.DiGraph]], num_processors: int, weights: Dict[str, float]) -> float:
    return mean([weighted_feature_schedule(dag, num_processors, weights).makespan for _, dag in data])


def bo_optimize_weights(
    train_data: List[Tuple[str, nx.DiGraph]],
    num_processors: int,
    initial_weights: Optional[Dict[str, float]],
    n_calls: int,
    seed: int,
    label: str,
) -> BOResult:
    if gp_minimize is None:
        print(f"[WARN][{label}] scikit-optimize not installed. Returning initial/fallback weights.")
        weights = initial_weights or default_llm_fallback_weights()
        return BOResult(weights, evaluate_weights_on_dataset(train_data, num_processors, weights), [])

    dims = [Real(-3.0, 3.0, name=name) for name in FEATURE_NAMES]
    history: List[float] = []

    @use_named_args(dims)
    def objective(**weights: float) -> float:
        value = evaluate_weights_on_dataset(train_data, num_processors, weights)
        history.append(value)
        return value

    kwargs = {
        "func": objective,
        "dimensions": dims,
        "n_calls": max(10, n_calls),
        "random_state": seed,
        "acq_func": "EI",
        "n_initial_points": 8,
    }
    if initial_weights is not None:
        kwargs["x0"] = [float(initial_weights.get(name, 0.0)) for name in FEATURE_NAMES]

    result = gp_minimize(**kwargs)
    best_weights = {name: float(value) for name, value in zip(FEATURE_NAMES, result.x)}
    running_best: List[float] = []
    best = float("inf")
    for value in history:
        best = min(best, value)
        running_best.append(best)
    print(f"[{label}] best training makespan = {result.fun:.2f}")
    return BOResult(best_weights, float(result.fun), running_best)


# ============================================================
# Aggressive per-DAG optimization: GA + SA
# ============================================================

def random_weights(scale: float = 3.0) -> Dict[str, float]:
    return {name: random.uniform(-scale, scale) for name in FEATURE_NAMES}


def mutate_weights(weights: Dict[str, float], sigma: float = 0.45, prob: float = 0.35, scale: float = 3.0) -> Dict[str, float]:
    child = dict(weights)
    for name in FEATURE_NAMES:
        if random.random() < prob:
            child[name] = clamp(child[name] + random.gauss(0.0, sigma), -scale, scale)
    return child


def crossover(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    return {name: (a[name] if random.random() < 0.5 else b[name]) for name in FEATURE_NAMES}


def seed_weight_pool(llm_weights: Dict[str, float], bo_weights: Dict[str, float]) -> List[Dict[str, float]]:
    seeds = [
        default_llm_fallback_weights(),
        llm_weights,
        bo_weights,
        {name: 0.0 for name in FEATURE_NAMES},
        {name: (1.0 if name == "upward_rank" else 0.0) for name in FEATURE_NAMES},
        {name: (1.0 if name in ["upward_rank", "avg_comm_out", "succ_count_weighted"] else 0.0) for name in FEATURE_NAMES},
        {name: (1.0 if name in ["upward_rank", "downward_rank"] else 0.0) for name in FEATURE_NAMES},
    ]
    # Add blended seeds.
    blend = {name: 0.5 * llm_weights.get(name, 0.0) + 0.5 * bo_weights.get(name, 0.0) for name in FEATURE_NAMES}
    seeds.append(blend)
    return seeds


def genetic_optimize_for_dag(
    dag: nx.DiGraph,
    num_processors: int,
    seed_weights: List[Dict[str, float]],
    pop_size: int,
    generations: int,
    elite_frac: float = 0.18,
    mutation_sigma: float = 0.50,
) -> ScheduleResult:
    pop_size = max(12, pop_size)
    population: List[Dict[str, float]] = [dict(w) for w in seed_weights]
    while len(population) < pop_size:
        if random.random() < 0.4 and seed_weights:
            population.append(mutate_weights(random.choice(seed_weights), sigma=0.8, prob=0.8))
        else:
            population.append(random_weights())

    cache: Dict[Tuple[float, ...], float] = {}

    def key(w: Dict[str, float]) -> Tuple[float, ...]:
        return tuple(round(w[name], 4) for name in FEATURE_NAMES)

    def fitness(w: Dict[str, float]) -> float:
        k = key(w)
        if k not in cache:
            cache[k] = weighted_feature_schedule(dag, num_processors, w).makespan
        return cache[k]

    best_weights = min(population, key=fitness)
    best_makespan = fitness(best_weights)

    for gen in range(generations):
        ranked = sorted(population, key=fitness)
        if fitness(ranked[0]) < best_makespan:
            best_weights, best_makespan = dict(ranked[0]), fitness(ranked[0])
        elite_n = max(3, int(elite_frac * pop_size))
        elites = ranked[:elite_n]
        next_pop = [dict(w) for w in elites]
        sigma = mutation_sigma * (1.0 - 0.65 * gen / max(1, generations))
        while len(next_pop) < pop_size:
            if random.random() < 0.70:
                p1, p2 = random.sample(elites, 2)
                child = crossover(p1, p2)
            else:
                child = dict(random.choice(elites))
            child = mutate_weights(child, sigma=sigma, prob=0.45)
            next_pop.append(child)
        population = next_pop

    result = weighted_feature_schedule(dag, num_processors, best_weights, method="GA-PerDAG")
    result.weights = best_weights
    return result


def simulated_annealing_refine(
    dag: nx.DiGraph,
    num_processors: int,
    initial: ScheduleResult,
    steps: int,
    start_temp: float = 8.0,
) -> ScheduleResult:
    if not initial.weights or steps <= 0:
        return initial
    current_w = dict(initial.weights)
    current = initial
    best = initial
    temp = start_temp
    for step in range(steps):
        candidate_w = mutate_weights(current_w, sigma=max(0.05, temp / 10.0), prob=0.55)
        candidate = weighted_feature_schedule(dag, num_processors, candidate_w, method="SA-Refined")
        delta = candidate.makespan - current.makespan
        accept = delta <= 0 or random.random() < math.exp(-delta / max(1e-9, temp))
        if accept:
            current_w = candidate_w
            current = candidate
        if candidate.makespan < best.makespan:
            best = candidate
            best.weights = candidate_w
        temp *= 0.995
    best.method = "SA-Refined"
    return best


def hybrid_aggressive_schedule(
    dag: nx.DiGraph,
    num_processors: int,
    llm_weights: Dict[str, float],
    bo_weights: Dict[str, float],
    ga_pop_size: int,
    ga_generations: int,
    sa_steps: int,
) -> ScheduleResult:
    candidates: List[ScheduleResult] = [
        heft_schedule(dag, num_processors),
        cpop_schedule(dag, num_processors),
        greedy_eft_schedule(dag, num_processors),
        weighted_feature_schedule(dag, num_processors, llm_weights, method="LLM-Heuristic"),
        weighted_feature_schedule(dag, num_processors, bo_weights, method="BO-Heuristic"),
    ]
    seeds = seed_weight_pool(llm_weights, bo_weights)
    ga = genetic_optimize_for_dag(dag, num_processors, seeds, ga_pop_size, ga_generations)
    candidates.append(ga)
    if sa_steps > 0:
        candidates.append(simulated_annealing_refine(dag, num_processors, ga, sa_steps))
    best = min(candidates, key=lambda r: r.makespan)
    best.method = "HYBRID-BEST"
    return best


# ============================================================
# GNN imitation and Decima-like reinforcement learning
# ============================================================

if nn is not None:
    class TinyDAGGNN(nn.Module):
        def __init__(self, in_dim: int, hidden_dim: int = 64, message_steps: int = 3):
            super().__init__()
            self.message_steps = message_steps
            self.input = nn.Linear(in_dim, hidden_dim)
            self.msg = nn.Linear(hidden_dim, hidden_dim)
            self.update = nn.GRUCell(hidden_dim, hidden_dim)
            self.out = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x, edge_index):
            h = torch.relu(self.input(x))
            for _ in range(self.message_steps):
                agg = torch.zeros_like(h)
                if edge_index.numel() > 0:
                    src, dst = edge_index[0], edge_index[1]
                    messages = self.msg(h[src])
                    agg.index_add_(0, dst, messages)
                h = self.update(agg, h)
            return self.out(h).squeeze(-1)


def graph_to_tensors(dag: nx.DiGraph):
    if torch is None:
        raise RuntimeError("PyTorch is not installed.")
    feats = normalize_features(extract_node_features(dag))
    nodes = list(dag.nodes())
    node_to_idx = {v: i for i, v in enumerate(nodes)}

    x = torch.tensor(
        [[feats[v][name] for name in FEATURE_NAMES] for v in nodes],
        dtype=torch.float32,
    )

    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in dag.edges()]
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return nodes, node_to_idx, x, edge_index


def train_gnn_imitation(
    train_data: List[Tuple[str, nx.DiGraph]],
    epochs: int,
    lr: float = 1e-3,
    hidden_dim: int = 64,
):
    if torch is None or nn is None or optim is None:
        print("[GNN] PyTorch unavailable; skipping GNN-Imitation.")
        return None

    model = TinyDAGGNN(len(FEATURE_NAMES), hidden_dim=hidden_dim)
    opt = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(max(1, epochs)):
        total_loss = 0.0
        random.shuffle(train_data)

        for _, dag in train_data:
            nodes, _, x, edge_index = graph_to_tensors(dag)
            scores = model(x, edge_index)

            # HEFT/upward-rank imitation target.
            target = upward_rank(dag)
            y = torch.tensor([target[v] for v in nodes], dtype=torch.float32)
            y = (y - y.mean()) / (y.std() + 1e-6)

            loss = ((scores - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item())

        if (epoch + 1) % max(1, epochs // 4) == 0:
            print(f"[GNN-Imitation] epoch={epoch + 1:03d}/{epochs}, loss={total_loss / len(train_data):.4f}")

    return model


def gnn_schedule(model, dag: nx.DiGraph, num_processors: int, method: str = "GNN-Imitation") -> ScheduleResult:
    if model is None:
        return heft_schedule(dag, num_processors)

    nodes, _, x, edge_index = graph_to_tensors(dag)
    model.eval()
    with torch.no_grad():
        scores = model(x, edge_index).detach().cpu().numpy()

    score_map = {v: float(scores[i]) for i, v in enumerate(nodes)}
    return list_schedule(dag, num_processors, lambda v: score_map[v], method=method)


def train_decima_like_rl(
    train_data: List[Tuple[str, nx.DiGraph]],
    num_processors: int,
    episodes: int,
    lr: float = 1e-3,
    hidden_dim: int = 64,
):
    if torch is None or nn is None or optim is None:
        print("[Decima-like-RL] PyTorch unavailable; skipping Decima-like-RL.")
        return None

    model = TinyDAGGNN(len(FEATURE_NAMES), hidden_dim=hidden_dim)
    opt = optim.Adam(model.parameters(), lr=lr)
    baseline: Optional[float] = None

    for ep in range(max(1, episodes)):
        _, dag = random.choice(train_data)
        nodes, node_to_idx, x, edge_index = graph_to_tensors(dag)
        log_probs = []

        def policy(ready: List[int], completed: set[int]) -> int:
            scores = model(x, edge_index)
            ready_idx = torch.tensor([node_to_idx[v] for v in ready], dtype=torch.long)
            ready_scores = scores[ready_idx]
            probs = torch.softmax(ready_scores, dim=0)
            dist = torch.distributions.Categorical(probs)
            action_pos = dist.sample()
            log_probs.append(dist.log_prob(action_pos))
            return ready[int(action_pos.item())]

        result = schedule_with_ready_policy(
            dag,
            num_processors,
            policy,
            method="Decima-like-RL",
        )

        reward = -float(result.makespan)
        baseline = reward if baseline is None else 0.90 * baseline + 0.10 * reward
        advantage = reward - baseline

        if log_probs:
            loss = -torch.stack(log_probs).sum() * float(advantage)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if (ep + 1) % max(1, episodes // 4) == 0:
            print(f"[Decima-like-RL] episode={ep + 1:03d}/{episodes}, makespan={result.makespan:.2f}")

    return model


# ============================================================
# Evaluation and reporting
# ============================================================

def evaluate_method(
    name: str,
    data: List[Tuple[str, nx.DiGraph]],
    num_processors: int,
    schedule_fn: Callable[[nx.DiGraph], ScheduleResult],
) -> Tuple[List[Dict[str, Any]], List[ScheduleResult]]:
    rows: List[Dict[str, Any]] = []
    schedules: List[ScheduleResult] = []
    for idx, (family, dag) in enumerate(data):
        t0 = time.time()
        result = schedule_fn(dag)
        runtime_ms = (time.time() - t0) * 1000.0
        lb = combined_lower_bound(dag, num_processors)
        cp = critical_path_fastest_processor(dag)
        rows.append({
            "dag_id": idx,
            "method": name,
            "family": family,
            "nodes": dag.number_of_nodes(),
            "edges": dag.number_of_edges(),
            "makespan": result.makespan,
            "critical_path_lb": cp,
            "combined_lb": lb,
            "slr": result.makespan / cp if cp else float("nan"),
            "gap_to_lb_percent": ((result.makespan - lb) / result.makespan * 100.0) if result.makespan > 0 else 0.0,
            "runtime_ms": runtime_ms,
            "actual_inner_method": result.method,
        })
        schedules.append(result)
    return rows, schedules


def add_improvement_vs_heft(rows: List[Dict[str, Any]]) -> None:
    heft_by_id: Dict[int, float] = {}
    for r in rows:
        if r["method"] == "HEFT":
            heft_by_id[r["dag_id"]] = r["makespan"]
    for r in rows:
        base = heft_by_id.get(r["dag_id"])
        if base and base > 0:
            r["improvement_vs_heft_percent"] = (base - r["makespan"]) / base * 100.0
        else:
            r["improvement_vs_heft_percent"] = float("nan")


def summarize(rows: List[Dict[str, Any]], target: float) -> List[MethodSummary]:
    add_improvement_vs_heft(rows)
    methods = list(dict.fromkeys(r["method"] for r in rows))
    summaries: List[MethodSummary] = []
    print("\n=== Aggregate Results ===")
    print(f"{'Method':<18} {'Avg Makespan':>13} {'Avg SLR':>9} {'Avg Improve':>13} {'WinRate':>9} {'Runtime ms':>11}")
    for method in methods:
        rs = [r for r in rows if r["method"] == method]
        imps = [r["improvement_vs_heft_percent"] for r in rs if not math.isnan(r["improvement_vs_heft_percent"])]
        wins = [1.0 if r["improvement_vs_heft_percent"] > 0 else 0.0 for r in rs if method != "HEFT"]
        summary = MethodSummary(
            method=method,
            avg_makespan=mean([r["makespan"] for r in rs]),
            avg_slr=mean([r["slr"] for r in rs]),
            avg_runtime_ms=mean([r["runtime_ms"] for r in rs]),
            avg_improvement_vs_heft=mean(imps) if method != "HEFT" else 0.0,
            win_rate_vs_heft=mean(wins) * 100.0 if wins else 0.0,
        )
        summaries.append(summary)
        print(
            f"{method:<18} {summary.avg_makespan:>13.2f} {summary.avg_slr:>9.3f} "
            f"{summary.avg_improvement_vs_heft:>12.2f}% {summary.win_rate_vs_heft:>8.1f}% "
            f"{summary.avg_runtime_ms:>11.2f}"
        )

    best = min(summaries, key=lambda s: s.avg_makespan)
    print("\n=== Target Check ===")
    if best.avg_improvement_vs_heft is not None and best.avg_improvement_vs_heft >= target:
        print(f"PASS: {best.method} averaged {best.avg_improvement_vs_heft:.2f}% improvement over HEFT.")
    else:
        print(
            f"NOT GUARANTEED: best method {best.method} averaged "
            f"{best.avg_improvement_vs_heft:.2f}% improvement over HEFT, target was {target:.1f}%."
        )
        print("If gap_to_lb_percent is small, HEFT is already near the lower bound and 30% may be impossible on that set.")
    return summaries


def per_family_report(rows: List[Dict[str, Any]]) -> None:
    print("\n=== Per-Family Avg Improvement vs HEFT (%) ===")
    families = sorted(set(r["family"] for r in rows))
    methods = [m for m in dict.fromkeys(r["method"] for r in rows) if m != "HEFT"]
    print(f"{'Method':<18}" + "".join(f"{f:>14}" for f in families))
    for method in methods:
        values = []
        for fam in families:
            xs = [r["improvement_vs_heft_percent"] for r in rows if r["method"] == method and r["family"] == fam]
            values.append(f"{mean(xs):>13.2f}%" if xs else f"{'-':>14}")
        print(f"{method:<18}" + "".join(values))


def wilcoxon_report(rows: List[Dict[str, Any]], reference: str = "HYBRID-BEST") -> None:
    if wilcoxon is None:
        print("\n[Stats] scipy unavailable; skipping Wilcoxon tests.")
        return
    by_method: Dict[str, Dict[int, float]] = {}
    for r in rows:
        by_method.setdefault(r["method"], {})[r["dag_id"]] = r["makespan"]
    if reference not in by_method:
        return
    print(f"\n=== Wilcoxon Signed-Rank Tests vs {reference} ===")
    print(f"{'Method':<18} {'mean(ref-other)':>17} {'p-value':>12} {'verdict':>10}")
    ref = by_method[reference]
    for method, scores in by_method.items():
        if method == reference:
            continue
        common = sorted(set(ref) & set(scores))
        if len(common) < 5:
            continue
        ref_arr = np.array([ref[i] for i in common], dtype=float)
        other_arr = np.array([scores[i] for i in common], dtype=float)
        diff = ref_arr - other_arr
        try:
            _, p = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            p = float("nan")
        verdict = "ref better" if mean(diff) < 0 and (not math.isnan(p) and p < 0.05) else "~equal/unclear"
        print(f"{method:<18} {mean(diff):>17.3f} {p:>12.4g} {verdict:>10}")


def save_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved results CSV: {path}")


def save_weights(weights: Dict[str, float], path: str) -> None:
    with open(path, "w") as f:
        json.dump(weights, f, indent=2)
    print(f"Saved BO weights JSON: {path}")


# ============================================================
# Main experiment
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-processors", type=int, default=4)
    parser.add_argument("--num-graphs", type=int, default=40)
    parser.add_argument("--n-values", type=str, default="20,30,50,80")
    parser.add_argument("--bo-calls", type=int, default=60)
    parser.add_argument("--ga-pop-size", type=int, default=80)
    parser.add_argument("--ga-generations", type=int, default=100)
    parser.add_argument("--sa-steps", type=int, default=500)
    parser.add_argument("--target-improvement", type=float, default=30.0)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--use-gnn", action="store_true")
    parser.add_argument("--use-decima", action="store_true")
    parser.add_argument("--gnn-epochs", type=int, default=80)
    parser.add_argument("--decima-episodes", type=int, default=120)
    parser.add_argument("--openai-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--out", type=str, default="mosaic_scheduler_30plus_results.csv")
    parser.add_argument("--weights-out", type=str, default="mosaic_bo_weights.json")
    parser.add_argument("--fast", action="store_true", help="Smaller GA/BO settings for quick testing.")
    args = parser.parse_args()

    if args.fast:
        args.bo_calls = min(args.bo_calls, 20)
        args.ga_pop_size = min(args.ga_pop_size, 32)
        args.ga_generations = min(args.ga_generations, 25)
        args.sa_steps = min(args.sa_steps, 100)
        args.gnn_epochs = min(args.gnn_epochs, 25)
        args.decima_episodes = min(args.decima_episodes, 40)

    set_seed(args.seed)
    n_values = [int(x.strip()) for x in args.n_values.split(",") if x.strip()]
    dataset = build_dataset(args.num_graphs, n_values, args.num_processors)
    random.shuffle(dataset)
    split = int(0.70 * len(dataset))
    train_data = dataset[:split]
    test_data = dataset[split:]

    print(f"Train DAGs: {len(train_data)} | Test DAGs: {len(test_data)} | Processors: {args.num_processors}")
    print(f"BO calls: {args.bo_calls} | GA pop: {args.ga_pop_size} | GA generations: {args.ga_generations} | SA steps: {args.sa_steps}")

    # LLM weights are kept exactly as requested. If no key, fallback is deterministic engineered weights.
    if args.use_llm:
        llm_weights = llm_propose_weights(train_data[0][1], args.openai_model)
    else:
        llm_weights = default_llm_fallback_weights()
    print("\nLLM/fallback initial weights:")
    print(json.dumps(llm_weights, indent=2))

    # BO warm start from LLM/fallback weights.
    bo_result = bo_optimize_weights(
        train_data=train_data,
        num_processors=args.num_processors,
        initial_weights=llm_weights,
        n_calls=args.bo_calls,
        seed=args.seed,
        label="BO-warm-start",
    )
    save_weights(bo_result.weights, args.weights_out)

    gnn_model = None
    decima_model = None

    if args.use_gnn:
        print("\nTraining GNN-Imitation model...")
        gnn_model = train_gnn_imitation(train_data, epochs=args.gnn_epochs)

    if args.use_decima:
        print("\nTraining Decima-like-RL model...")
        decima_model = train_decima_like_rl(
            train_data,
            args.num_processors,
            episodes=args.decima_episodes,
        )

    all_rows: List[Dict[str, Any]] = []

    methods: List[Tuple[str, Callable[[nx.DiGraph], ScheduleResult]]] = [
        ("HEFT", lambda dag: heft_schedule(dag, args.num_processors)),
        ("CPOP", lambda dag: cpop_schedule(dag, args.num_processors)),
        ("Greedy-EFT", lambda dag: greedy_eft_schedule(dag, args.num_processors)),
        ("LLM-Heuristic", lambda dag: weighted_feature_schedule(dag, args.num_processors, llm_weights, "LLM-Heuristic")),
        ("BO-Heuristic", lambda dag: weighted_feature_schedule(dag, args.num_processors, bo_result.weights, "BO-Heuristic")),
        (
            "GA-PerDAG",
            lambda dag: genetic_optimize_for_dag(
                dag,
                args.num_processors,
                seed_weight_pool(llm_weights, bo_result.weights),
                args.ga_pop_size,
                args.ga_generations,
            ),
        ),
        (
            "GA+SA",
            lambda dag: simulated_annealing_refine(
                dag,
                args.num_processors,
                genetic_optimize_for_dag(
                    dag,
                    args.num_processors,
                    seed_weight_pool(llm_weights, bo_result.weights),
                    args.ga_pop_size,
                    args.ga_generations,
                ),
                args.sa_steps,
            ),
        ),
        (
            "HYBRID-BEST",
            lambda dag: hybrid_aggressive_schedule(
                dag,
                args.num_processors,
                llm_weights,
                bo_result.weights,
                args.ga_pop_size,
                args.ga_generations,
                args.sa_steps,
            ),
        ),
    ]

    if args.use_gnn and gnn_model is not None:
        methods.append(
            ("GNN-Imitation", lambda dag: gnn_schedule(gnn_model, dag, args.num_processors, "GNN-Imitation"))
        )

    if args.use_decima and decima_model is not None:
        methods.append(
            ("Decima-like-RL", lambda dag: gnn_schedule(decima_model, dag, args.num_processors, "Decima-like-RL"))
        )

    for name, fn in methods:
        print(f"\nEvaluating {name}...")
        rows, _ = evaluate_method(name, test_data, args.num_processors, fn)
        all_rows.extend(rows)

    summarize(all_rows, args.target_improvement)
    per_family_report(all_rows)
    wilcoxon_report(all_rows, reference="HYBRID-BEST")
    save_csv(all_rows, args.out)

    print("\nRecommended strong command:")
    print("python mosaic_scheduler_30plus.py --use-llm --use-gnn --use-decima --bo-calls 40 --ga-generations 40 --ga-pop-size 64 --sa-steps 300 --gnn-epochs 80 --decima-episodes 120")


if __name__ == "__main__":
    main()
