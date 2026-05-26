"""
mosaic_scheduler_clean.py

Honest DAG scheduling evaluation framework for MoSAIC.

DESIGN PRINCIPLES (read these before adding anything):

1. NO ORACLE SELECTION. Every reported method must be a single deployable
   policy. We do not pick per-DAG winners across methods after the fact.

2. NO TRAINING ON THE TEST SET. Any optimization (BO, GA, SA, GNN, RL) uses
   the training split only. Test DAGs are touched exactly once per method, at
   evaluation time.

3. HONEST BENCHMARKS. Cost and communication ranges are set to values
   comparable to the standard scheduling literature (HEFT paper, follow-ups),
   not tuned to make HEFT look bad.

4. STATISTICAL HONESTY. Wilcoxon signed-rank tests on paired DAGs. Effect
   sizes reported alongside p-values. No cherry-picked aggregates.

5. EVERY EXPERIMENT IS AN ABLATION OF SOMETHING IN THE PAPER. If a method
   appears in the output, it should correspond to a claim in the writeup.

METHODS EVALUATED:
  - HEFT (reference baseline)
  - CPOP (reference baseline)
  - Greedy-EFT (reference baseline)
  - LLM-Heuristic (raw LLM weights, no optimization)
  - BO-Warm-Heuristic (BO initialized from LLM weights)  <-- main method
  - BO-Cold-Heuristic (BO initialized randomly)          <-- ablation
  - GA-Train (genetic search on TRAIN set only)          <-- ablation
  - GNN-Imitation (optional, --use-gnn)
  - Decima-like-RL (optional, --use-decima)

EVALUATION:
  - Aggregate makespan, SLR, gap-to-lower-bound, runtime
  - Per-family breakdown (does the method generalize across DAG topologies?)
  - Wilcoxon tests against BO-Warm-Heuristic (the proposed method)
  - Feature ablation on the BO-Warm weights

Install:
    pip install networkx numpy scipy scikit-optimize openai
    pip install torch  # optional

Run (no LLM, no GNN, fast):
    python mosaic_scheduler_clean.py --bo-calls 30

Run (full):
    python mosaic_scheduler_clean.py --use-llm --use-gnn --use-decima \
        --bo-calls 40 --ga-generations 40 --ga-pop-size 64 \
        --gnn-epochs 80 --decima-episodes 120
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args
except Exception:
    gp_minimize = None
    Real = None
    use_named_args = None

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:
    torch = None
    nn = None
    optim = None


# ============================================================
# Feature set
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


# ============================================================
# Data classes
# ============================================================

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


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


def mean(xs) -> float:
    xs = np.asarray(xs, dtype=float)
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
# HONEST ranges, comparable to the HEFT paper (Topcuoglu et al. 2002) and
# common follow-ups. CCR (communication-to-computation ratio) defaults to
# around 1.0, not skewed to make HEFT look bad.
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
    """
    Honest cost generation:
    - Base task cost in [5, 30]
    - Per-processor variation in [0.7x, 1.4x] (HEFT-paper style heterogeneity)
    - Edge communication cost in [1, 10] (CCR ~ 0.3-1.0, standard regime)
    """
    for v in dag.nodes():
        base = random.randint(5, 30)
        dag.nodes[v]["costs"] = [
            max(1, int(base * random.uniform(0.7, 1.4)))
            for _ in range(num_processors)
        ]
    for u, v in dag.edges():
        dag.edges[u, v]["comm"] = random.randint(1, 10)
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
            dag = random_er_dag(n, random.uniform(0.05, 0.18), num_processors)
        elif family == "BA":
            dag = random_ba_dag(n, random.randint(1, min(4, n - 1)), num_processors)
        else:
            dag = random_ws_dag(n, min(4, n - 1), random.uniform(0.1, 0.4), num_processors)
        dataset.append((family, dag))
    dataset.extend([
        ("MOTIF_CHAIN", motif_chain(12, num_processors)),
        ("MOTIF_FORK", motif_fork(10, num_processors)),
        ("MOTIF_JOIN", motif_join(10, num_processors)),
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
        if not succ:
            rank[v] = avg_cost(dag, v)
        else:
            rank[v] = avg_cost(dag, v) + max(dag.edges[v, u]["comm"] + rank[u] for u in succ)
    return rank


def downward_rank(dag: nx.DiGraph) -> Dict[int, float]:
    rank: Dict[int, float] = {}
    for v in nx.topological_sort(dag):
        pred = list(dag.predecessors(v))
        if not pred:
            rank[v] = avg_cost(dag, v)
        else:
            rank[v] = avg_cost(dag, v) + max(dag.edges[u, v]["comm"] + rank[u] for u in pred)
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
        ac, mic, mac = avg_cost(dag, v), min_cost(dag, v), max_cost(dag, v)
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
            "succ_count_weighted": float(sum(1.0 + dag.edges[v, u]["comm"] for u in dag.successors(v))),
            "pred_count_weighted": float(sum(1.0 + dag.edges[u, v]["comm"] for u in dag.predecessors(v))),
            "comm_pressure": float(fanout * avg_comm),
            "mobility_proxy": float((max_depth - depth[v]) + fanout - 0.5 * indegree),
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
# Lower bounds
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


# ============================================================
# List scheduling
# ============================================================

def earliest_start_finish(dag, task, proc, proc_available, finish_times, assignment):
    ready_time = 0.0
    for parent in dag.predecessors(task):
        comm = 0.0 if assignment.get(parent) == proc else float(dag.edges[parent, task]["comm"])
        ready_time = max(ready_time, finish_times[parent] + comm)
    start = max(proc_available[proc], ready_time)
    finish = start + dag.nodes[task]["costs"][proc]
    return start, finish


def choose_best_processor(dag, task, proc_available, finish_times, assignment):
    candidates = []
    for p in range(len(proc_available)):
        st, ft = earliest_start_finish(dag, task, p, proc_available, finish_times, assignment)
        candidates.append((ft, st, proc_available[p], p))
    best = min(candidates, key=lambda x: (x[0], x[1], x[2], x[3]))
    return int(best[3]), float(best[1]), float(best[0])


def list_schedule(dag, num_processors, priority_fn, method="list") -> ScheduleResult:
    completed, scheduled = set(), set()
    assignment, start_times, finish_times = {}, {}, {}
    proc_available = [0.0] * num_processors

    while len(completed) < dag.number_of_nodes():
        ready = [v for v in dag.nodes() if v not in scheduled and all(p in completed for p in dag.predecessors(v))]
        if not ready:
            raise RuntimeError("No ready tasks; not a DAG?")
        task = max(ready, key=lambda v: (priority_fn(v), -v))
        proc, st, ft = choose_best_processor(dag, task, proc_available, finish_times, assignment)
        assignment[task], start_times[task], finish_times[task] = proc, st, ft
        proc_available[proc] = ft
        scheduled.add(task)
        completed.add(task)

    return ScheduleResult(
        makespan=max(finish_times.values()) if finish_times else 0.0,
        assignment=assignment, start_times=start_times, finish_times=finish_times, method=method,
    )


def schedule_with_ready_policy(dag, num_processors, policy_fn, method) -> ScheduleResult:
    completed, scheduled = set(), set()
    assignment, start_times, finish_times = {}, {}, {}
    proc_available = [0.0] * num_processors

    while len(completed) < dag.number_of_nodes():
        ready = [v for v in dag.nodes() if v not in scheduled and all(p in completed for p in dag.predecessors(v))]
        if not ready:
            raise RuntimeError("No ready tasks; not a DAG?")
        task = policy_fn(ready, completed)
        if task not in ready:
            task = ready[0]
        proc, st, ft = choose_best_processor(dag, task, proc_available, finish_times, assignment)
        assignment[task], start_times[task], finish_times[task] = proc, st, ft
        proc_available[proc] = ft
        scheduled.add(task)
        completed.add(task)

    return ScheduleResult(
        makespan=max(finish_times.values()) if finish_times else 0.0,
        assignment=assignment, start_times=start_times, finish_times=finish_times, method=method,
    )


# ============================================================
# Baselines
# ============================================================

def heft_schedule(dag, num_processors) -> ScheduleResult:
    ur = upward_rank(dag)
    return list_schedule(dag, num_processors, lambda v: ur[v], method="HEFT")


def cpop_schedule(dag, num_processors) -> ScheduleResult:
    ur, dr = upward_rank(dag), downward_rank(dag)
    return list_schedule(dag, num_processors, lambda v: ur[v] + dr[v], method="CPOP")


def greedy_eft_schedule(dag, num_processors) -> ScheduleResult:
    completed, scheduled = set(), set()
    assignment, start_times, finish_times = {}, {}, {}
    proc_available = [0.0] * num_processors

    while len(completed) < dag.number_of_nodes():
        ready = [v for v in dag.nodes() if v not in scheduled and all(p in completed for p in dag.predecessors(v))]

        def best_eft(v):
            return min(earliest_start_finish(dag, v, p, proc_available, finish_times, assignment)[1]
                       for p in range(num_processors))

        task = min(ready, key=lambda v: (best_eft(v), v))
        proc, st, ft = choose_best_processor(dag, task, proc_available, finish_times, assignment)
        assignment[task], start_times[task], finish_times[task] = proc, st, ft
        proc_available[proc] = ft
        scheduled.add(task)
        completed.add(task)

    return ScheduleResult(
        makespan=max(finish_times.values()),
        assignment=assignment, start_times=start_times, finish_times=finish_times, method="Greedy-EFT",
    )


def weighted_feature_schedule(dag, num_processors, weights, method="Weighted") -> ScheduleResult:
    feats = normalize_features(extract_node_features(dag))
    result = list_schedule(
        dag, num_processors,
        lambda v: sum(weights.get(name, 0.0) * feats[v][name] for name in FEATURE_NAMES),
        method=method,
    )
    result.weights = dict(weights)
    return result


# ============================================================
# LLM weight proposal (one optional call)
# ============================================================

def default_llm_fallback_weights() -> Dict[str, float]:
    return {
        "upward_rank": 1.0,
        "downward_rank": 0.25,
        "depth": 0.30,
        "fanout": 0.20,
        "indegree": 0.05,
        "avg_comm_out": 0.35,
        "avg_cost": 0.10,
        "min_cost": 0.05,
        "max_cost": 0.05,
        "cost_spread": 0.10,
        "succ_count_weighted": 0.25,
        "pred_count_weighted": -0.05,
        "comm_pressure": 0.30,
        "mobility_proxy": 0.10,
    }


def llm_propose_weights(dag: nx.DiGraph, model: str = "gpt-4o-mini") -> Dict[str, float]:
    fallback = default_llm_fallback_weights()
    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        print("[LLM] OPENAI_API_KEY missing or openai package unavailable. Using engineered fallback weights.")
        return fallback

    prompt = f"""
You are designing an interpretable priority function for DAG list scheduling.
Return JSON only. No prose.

Priority: H(v) = sum_i theta_i * normalized_feature_i(v). Higher H(v) means schedule earlier.

Feature names (return a value for each):
{FEATURE_NAMES}

Rules:
- Values are floats in [-2, 2].
- Favor critical path (upward_rank, downward_rank), communication-heavy edges
  (avg_comm_out, comm_pressure), and structural priority.
- Use small or negative weights for features that may cause contention if scheduled too early.

DAG summary:
{json.dumps(graph_summary(dag), indent=2)}
"""
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model=model, temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        parsed = safe_json_loads(response.choices[0].message.content or "")
        if not parsed:
            return fallback
        return {name: clamp(float(parsed.get(name, fallback[name])), -2.0, 2.0) for name in FEATURE_NAMES}
    except Exception as exc:
        print(f"[LLM] OpenAI call failed: {exc}. Using fallback.")
        return fallback


# ============================================================
# Bayesian Optimization (warm and cold start variants)
# Both train ONLY on training DAGs.
# ============================================================

def evaluate_weights_on_dataset(data, num_processors, weights) -> float:
    return mean([weighted_feature_schedule(dag, num_processors, weights).makespan for _, dag in data])


def bo_optimize_weights(
    train_data, num_processors, initial_weights, n_calls, seed, label,
) -> BOResult:
    if gp_minimize is None:
        print(f"[WARN][{label}] scikit-optimize not installed. Returning initial weights.")
        weights = initial_weights or default_llm_fallback_weights()
        return BOResult(weights, evaluate_weights_on_dataset(train_data, num_processors, weights), [])

    dims = [Real(-2.0, 2.0, name=name) for name in FEATURE_NAMES]
    history: List[float] = []

    @use_named_args(dims)
    def objective(**weights):
        value = evaluate_weights_on_dataset(train_data, num_processors, weights)
        history.append(value)
        return value

    kwargs = dict(
        func=objective, dimensions=dims,
        n_calls=max(10, n_calls),
        random_state=seed, acq_func="EI",
        n_initial_points=8,
    )
    if initial_weights is not None:
        kwargs["x0"] = [float(initial_weights.get(name, 0.0)) for name in FEATURE_NAMES]

    result = gp_minimize(**kwargs)
    best_weights = {name: float(value) for name, value in zip(FEATURE_NAMES, result.x)}

    running_best, best = [], float("inf")
    for value in history:
        best = min(best, value)
        running_best.append(best)

    print(f"[{label}] best training makespan = {result.fun:.2f}")
    return BOResult(best_weights, float(result.fun), running_best)


# ============================================================
# Genetic search on TRAINING SET ONLY
# Produces a SINGLE weight vector evaluated on the test set.
# This is fundamentally different from per-DAG GA (which would train on test).
# ============================================================

def random_weights(scale: float = 2.0) -> Dict[str, float]:
    return {name: random.uniform(-scale, scale) for name in FEATURE_NAMES}


def mutate_weights(weights, sigma=0.4, prob=0.3, scale=2.0) -> Dict[str, float]:
    child = dict(weights)
    for name in FEATURE_NAMES:
        if random.random() < prob:
            child[name] = clamp(child[name] + random.gauss(0.0, sigma), -scale, scale)
    return child


def crossover(a, b) -> Dict[str, float]:
    return {name: (a[name] if random.random() < 0.5 else b[name]) for name in FEATURE_NAMES}


def ga_train_weights(
    train_data, num_processors, seeds, pop_size, generations, elite_frac=0.15,
) -> Dict[str, float]:
    """
    Genetic search over weight vectors using TRAIN set fitness.
    Returns one weight vector; will be evaluated on test set separately.
    """
    pop_size = max(12, pop_size)
    population = [dict(w) for w in seeds]
    while len(population) < pop_size:
        if seeds and random.random() < 0.4:
            population.append(mutate_weights(random.choice(seeds), sigma=0.8, prob=0.8))
        else:
            population.append(random_weights())

    cache: Dict[Tuple[float, ...], float] = {}

    def key(w):
        return tuple(round(w[name], 4) for name in FEATURE_NAMES)

    def fitness(w):
        k = key(w)
        if k not in cache:
            cache[k] = evaluate_weights_on_dataset(train_data, num_processors, w)
        return cache[k]

    best_w = min(population, key=fitness)
    best_fit = fitness(best_w)

    for gen in range(generations):
        ranked = sorted(population, key=fitness)
        if fitness(ranked[0]) < best_fit:
            best_w, best_fit = dict(ranked[0]), fitness(ranked[0])
        elite_n = max(3, int(elite_frac * pop_size))
        elites = ranked[:elite_n]
        sigma = 0.5 * (1.0 - 0.6 * gen / max(1, generations))
        next_pop = [dict(w) for w in elites]
        while len(next_pop) < pop_size:
            if random.random() < 0.7:
                p1, p2 = random.sample(elites, 2)
                child = crossover(p1, p2)
            else:
                child = dict(random.choice(elites))
            child = mutate_weights(child, sigma=sigma, prob=0.4)
            next_pop.append(child)
        population = next_pop

    print(f"[GA-Train] best training makespan = {best_fit:.2f}")
    return best_w


# ============================================================
# GNN imitation and Decima-like RL
# Both train ONLY on training DAGs.
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


def graph_to_tensors(dag):
    if torch is None:
        raise RuntimeError("PyTorch not installed.")
    feats = normalize_features(extract_node_features(dag))
    nodes = list(dag.nodes())
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    x = torch.tensor([[feats[v][name] for name in FEATURE_NAMES] for v in nodes], dtype=torch.float32)
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in dag.edges()]
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return nodes, node_to_idx, x, edge_index


def train_gnn_imitation(train_data, epochs, lr=1e-3, hidden_dim=64):
    if torch is None or nn is None or optim is None:
        print("[GNN] PyTorch unavailable; skipping.")
        return None
    model = TinyDAGGNN(len(FEATURE_NAMES), hidden_dim=hidden_dim)
    opt = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(max(1, epochs)):
        total_loss = 0.0
        random.shuffle(train_data)
        for _, dag in train_data:
            nodes, _, x, edge_index = graph_to_tensors(dag)
            scores = model(x, edge_index)
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
            print(f"[GNN-Imitation] epoch={epoch+1:03d}/{epochs}, loss={total_loss/len(train_data):.4f}")
    return model


def gnn_schedule(model, dag, num_processors, method="GNN-Imitation") -> ScheduleResult:
    if model is None:
        return heft_schedule(dag, num_processors)
    nodes, _, x, edge_index = graph_to_tensors(dag)
    model.eval()
    with torch.no_grad():
        scores = model(x, edge_index).detach().cpu().numpy()
    score_map = {v: float(scores[i]) for i, v in enumerate(nodes)}
    return list_schedule(dag, num_processors, lambda v: score_map[v], method=method)


def train_decima_like_rl(train_data, num_processors, episodes, lr=1e-3, hidden_dim=64):
    if torch is None or nn is None or optim is None:
        print("[Decima-RL] PyTorch unavailable; skipping.")
        return None
    model = TinyDAGGNN(len(FEATURE_NAMES), hidden_dim=hidden_dim)
    opt = optim.Adam(model.parameters(), lr=lr)
    baseline: Optional[float] = None
    for ep in range(max(1, episodes)):
        _, dag = random.choice(train_data)
        nodes, node_to_idx, x, edge_index = graph_to_tensors(dag)
        log_probs = []

        def policy(ready, completed):
            scores = model(x, edge_index)
            ready_idx = torch.tensor([node_to_idx[v] for v in ready], dtype=torch.long)
            ready_scores = scores[ready_idx]
            probs = torch.softmax(ready_scores, dim=0)
            dist = torch.distributions.Categorical(probs)
            action_pos = dist.sample()
            log_probs.append(dist.log_prob(action_pos))
            return ready[int(action_pos.item())]

        result = schedule_with_ready_policy(dag, num_processors, policy, "Decima-like-RL")
        reward = -float(result.makespan)
        baseline = reward if baseline is None else 0.9 * baseline + 0.1 * reward
        advantage = reward - baseline

        if log_probs:
            loss = -torch.stack(log_probs).sum() * float(advantage)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if (ep + 1) % max(1, episodes // 4) == 0:
            print(f"[Decima-RL] episode={ep+1:03d}/{episodes}, makespan={result.makespan:.2f}")
    if episodes < 500:
        print("[Decima-RL] WARNING: trained on few episodes; treat as under-converged reference.")
    return model


# ============================================================
# Evaluation and reporting
# ============================================================

def evaluate_method(name, data, num_processors, schedule_fn):
    rows = []
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
        })
    return rows


def add_diff_vs_heft(rows):
    """Compute makespan difference vs HEFT (not framed as 'improvement' to avoid the trap)."""
    heft_by_id = {r["dag_id"]: r["makespan"] for r in rows if r["method"] == "HEFT"}
    for r in rows:
        base = heft_by_id.get(r["dag_id"])
        if base and base > 0:
            r["diff_vs_heft_percent"] = (base - r["makespan"]) / base * 100.0
        else:
            r["diff_vs_heft_percent"] = float("nan")


def summarize_aggregate(rows):
    add_diff_vs_heft(rows)
    methods = list(dict.fromkeys(r["method"] for r in rows))
    print("\n=== Aggregate Results ===")
    print(f"{'Method':<22} {'Avg Makespan':>13} {'Avg SLR':>9} {'Gap-LB%':>9} {'vs HEFT%':>10} {'Runtime ms':>11}")
    for m in methods:
        rs = [r for r in rows if r["method"] == m]
        diffs = [r["diff_vs_heft_percent"] for r in rs if not math.isnan(r["diff_vs_heft_percent"])]
        print(
            f"{m:<22} {mean([r['makespan'] for r in rs]):>13.2f} "
            f"{mean([r['slr'] for r in rs]):>9.3f} "
            f"{mean([r['gap_to_lb_percent'] for r in rs]):>9.2f} "
            f"{mean(diffs):>+10.2f} "
            f"{mean([r['runtime_ms'] for r in rs]):>11.2f}"
        )
    print("\nNote: 'vs HEFT%' is the relative makespan difference; positive = lower makespan than HEFT.")
    print("This is a reference point, not a target. Comparable-to-HEFT performance is a legitimate outcome.")


def per_family_report(rows):
    families = sorted(set(r["family"] for r in rows))
    methods = list(dict.fromkeys(r["method"] for r in rows))
    print("\n=== Per-Family Avg Makespan ===")
    print(f"{'Method':<22}" + "".join(f"{f:>14}" for f in families))
    for m in methods:
        cells = []
        for fam in families:
            vals = [r["makespan"] for r in rows if r["method"] == m and r["family"] == fam]
            cells.append(f"{mean(vals):>14.2f}" if vals else f"{'-':>14}")
        print(f"{m:<22}" + "".join(cells))


def wilcoxon_report(rows, reference="BO-Warm-Heuristic"):
    if wilcoxon is None:
        print("\n[Stats] scipy unavailable; skipping Wilcoxon tests.")
        return
    by_method: Dict[str, Dict[int, float]] = {}
    for r in rows:
        by_method.setdefault(r["method"], {})[r["dag_id"]] = r["makespan"]
    if reference not in by_method:
        print(f"\n[Stats] reference '{reference}' not in results; skipping.")
        return
    print(f"\n=== Wilcoxon Signed-Rank Tests (reference: {reference}) ===")
    print(f"{'Method':<22} {'mean(ref-other)':>17} {'p-value':>12} {'verdict':>18}")
    ref = by_method[reference]
    for m, scores in by_method.items():
        if m == reference:
            continue
        common = sorted(set(ref) & set(scores))
        if len(common) < 5:
            print(f"{m:<22} (insufficient paired samples: {len(common)})")
            continue
        ref_arr = np.array([ref[i] for i in common], dtype=float)
        other_arr = np.array([scores[i] for i in common], dtype=float)
        diff = ref_arr - other_arr
        try:
            _, p = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            p = float("nan")
        if math.isnan(p):
            verdict = "n/a"
        elif p >= 0.05:
            verdict = "no sig. diff."
        elif mean(diff) < 0:
            verdict = "ref lower (better)"
        else:
            verdict = "ref higher (worse)"
        print(f"{m:<22} {mean(diff):>+17.3f} {p:>12.4g} {verdict:>18}")


def feature_ablation(weights, test_data, num_processors):
    print("\n=== Feature Ablation (zeroing each feature in the BO-Warm weights) ===")
    print(f"{'Removed feature':<22} {'Avg makespan':>14} {'Delta vs full':>16}")
    full = [weighted_feature_schedule(dag, num_processors, weights).makespan for _, dag in test_data]
    full_avg = mean(full)
    print(f"{'(none — full)':<22} {full_avg:>14.2f} {'+0.00':>16}")
    for feat in FEATURE_NAMES:
        ablated = dict(weights)
        ablated[feat] = 0.0
        avg = mean([weighted_feature_schedule(dag, num_processors, ablated).makespan for _, dag in test_data])
        print(f"{feat:<22} {avg:>14.2f} {avg - full_avg:>+16.2f}")


def save_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved results CSV: {path}")


def save_convergence_csv(warm, cold, path):
    if warm is None and cold is None:
        return
    max_len = max(len(warm.convergence_curve) if warm else 0,
                  len(cold.convergence_curve) if cold else 0)
    if max_len == 0:
        return
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iter", "warm_start_best", "cold_start_best"])
        for i in range(max_len):
            w = warm.convergence_curve[i] if warm and i < len(warm.convergence_curve) else ""
            c = cold.convergence_curve[i] if cold and i < len(cold.convergence_curve) else ""
            writer.writerow([i + 1, w, c])
    print(f"Saved BO convergence CSV: {path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-processors", type=int, default=4)
    parser.add_argument("--num-graphs", type=int, default=36)
    parser.add_argument("--n-values", type=str, default="20,30,50")
    parser.add_argument("--bo-calls", type=int, default=30)
    parser.add_argument("--ga-pop-size", type=int, default=48)
    parser.add_argument("--ga-generations", type=int, default=30)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--use-gnn", action="store_true")
    parser.add_argument("--use-decima", action="store_true")
    parser.add_argument("--use-ga-train", action="store_true",
                        help="Run train-set GA to produce one weight vector (deployable).")
    parser.add_argument("--skip-cold-bo", action="store_true")
    parser.add_argument("--gnn-epochs", type=int, default=80)
    parser.add_argument("--decima-episodes", type=int, default=120)
    parser.add_argument("--openai-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--out", type=str, default="mosaic_clean_results.csv")
    parser.add_argument("--convergence-out", type=str, default="bo_convergence.csv")
    parser.add_argument("--weights-out", type=str, default="bo_warm_weights.json")
    args = parser.parse_args()

    set_seed(args.seed)
    n_values = [int(x.strip()) for x in args.n_values.split(",") if x.strip()]
    dataset = build_dataset(args.num_graphs, n_values, args.num_processors)
    random.shuffle(dataset)
    split = int(0.7 * len(dataset))
    train_data = dataset[:split]
    test_data = dataset[split:]

    print(f"Train DAGs: {len(train_data)} | Test DAGs: {len(test_data)} | Processors: {args.num_processors}")
    print(f"BO calls: {args.bo_calls}")

    # ----- LLM Role 1 -----
    if args.use_llm:
        llm_weights = llm_propose_weights(train_data[0][1], args.openai_model)
    else:
        llm_weights = default_llm_fallback_weights()
    print("\nInitial LLM/fallback weights:")
    print(json.dumps(llm_weights, indent=2))

    # ----- BO Warm Start (main method) -----
    bo_warm = bo_optimize_weights(
        train_data, args.num_processors, llm_weights,
        n_calls=args.bo_calls, seed=args.seed, label="BO-Warm",
    )
    with open(args.weights_out, "w") as f:
        json.dump(bo_warm.weights, f, indent=2)
    print(f"Saved BO-Warm weights to {args.weights_out}")

    # ----- BO Cold Start (ablation) -----
    bo_cold = None
    if not args.skip_cold_bo:
        bo_cold = bo_optimize_weights(
            train_data, args.num_processors, None,
            n_calls=args.bo_calls, seed=args.seed + 1, label="BO-Cold",
        )

    save_convergence_csv(bo_warm, bo_cold, args.convergence_out)

    if bo_warm and bo_cold:
        print("\n=== BO Convergence (training-set best vs iteration) ===")
        print(f"{'Iter':>6} {'Warm':>10} {'Cold':>10}")
        m = max(len(bo_warm.convergence_curve), len(bo_cold.convergence_curve))
        marks = sorted({1, m // 4, m // 2, (3 * m) // 4, m} - {0})
        for i in marks:
            w = bo_warm.convergence_curve[i - 1] if i - 1 < len(bo_warm.convergence_curve) else float("nan")
            c = bo_cold.convergence_curve[i - 1] if i - 1 < len(bo_cold.convergence_curve) else float("nan")
            print(f"{i:>6} {w:>10.2f} {c:>10.2f}")

    # ----- GA-Train (deployable single-weight ablation) -----
    ga_weights = None
    if args.use_ga_train:
        seeds = [llm_weights, bo_warm.weights, default_llm_fallback_weights(),
                 {n: 0.0 for n in FEATURE_NAMES},
                 {n: (1.0 if n == "upward_rank" else 0.0) for n in FEATURE_NAMES}]
        ga_weights = ga_train_weights(
            train_data, args.num_processors, seeds,
            pop_size=args.ga_pop_size, generations=args.ga_generations,
        )

    # ----- GNN and Decima -----
    gnn_model = train_gnn_imitation(train_data, epochs=args.gnn_epochs) if args.use_gnn else None
    decima_model = train_decima_like_rl(train_data, args.num_processors, episodes=args.decima_episodes) if args.use_decima else None

    # ----- Evaluation -----
    all_rows: List[Dict[str, Any]] = []

    methods: List[Tuple[str, Callable]] = [
        ("HEFT", lambda dag: heft_schedule(dag, args.num_processors)),
        ("CPOP", lambda dag: cpop_schedule(dag, args.num_processors)),
        ("Greedy-EFT", lambda dag: greedy_eft_schedule(dag, args.num_processors)),
        ("LLM-Heuristic", lambda dag: weighted_feature_schedule(dag, args.num_processors, llm_weights, "LLM-Heuristic")),
        ("BO-Warm-Heuristic", lambda dag: weighted_feature_schedule(dag, args.num_processors, bo_warm.weights, "BO-Warm-Heuristic")),
    ]
    if bo_cold is not None:
        methods.append(("BO-Cold-Heuristic", lambda dag: weighted_feature_schedule(dag, args.num_processors, bo_cold.weights, "BO-Cold-Heuristic")))
    if ga_weights is not None:
        methods.append(("GA-Train", lambda dag: weighted_feature_schedule(dag, args.num_processors, ga_weights, "GA-Train")))
    if gnn_model is not None:
        methods.append(("GNN-Imitation", lambda dag: gnn_schedule(gnn_model, dag, args.num_processors)))
    if decima_model is not None:
        methods.append(("Decima-like-RL", lambda dag: gnn_schedule(decima_model, dag, args.num_processors, "Decima-like-RL")))

    for name, fn in methods:
        print(f"Evaluating {name}...")
        all_rows.extend(evaluate_method(name, test_data, args.num_processors, fn))

    summarize_aggregate(all_rows)
    per_family_report(all_rows)
    wilcoxon_report(all_rows, reference="BO-Warm-Heuristic")
    feature_ablation(bo_warm.weights, test_data, args.num_processors)
    save_csv(all_rows, args.out)


if __name__ == "__main__":
    main()