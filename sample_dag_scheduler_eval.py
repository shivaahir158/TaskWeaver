"""
sample_dag_scheduler_eval.py

Single-file DAG scheduling evaluation framework.

What this script does:
1. Generates synthetic DAGs:
   - ER, BA, WS random DAGs
   - motif DAGs: chain, fork, join, diamond

2. Evaluates scheduling methods:
   - HEFT
   - CPOP
   - LLM-Heuristic
   - LLM+BO-Heuristic
   - GNN-Imitation
   - Decima-like-RL

3. Uses LLM in two roles:
   Role 1: Generate initial weights for an interpretable priority function.
   Role 2: Analyze schedule and provide PPA-aware insights.

4. Uses Bayesian Optimization:
   - Starts from LLM-generated weights
   - Optimizes the weights on training DAGs
   - Evaluates optimized weights on unseen test DAGs

Install:
    pip install networkx numpy torch openai scikit-optimize

PowerShell OpenAI key:
    $env:OPENAI_API_KEY="your_key_here"

Run:
    python sample_dag_scheduler_eval_rewritten.py --use-llm

Without LLM:
    python sample_dag_scheduler_eval_rewritten.py
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
# DAG generation
# ============================================================

def ensure_dag_by_orientation(g: nx.Graph) -> nx.DiGraph:
    """Convert an undirected graph into a DAG by orienting edges."""
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
    """Add heterogeneous processor execution costs and edge communication costs."""
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
    g = nx.barabasi_albert_graph(
        n,
        max(1, min(m, n - 1)),
        seed=random.randint(0, 10_000),
    )
    return add_costs(ensure_dag_by_orientation(g), num_processors)


def random_ws_dag(n: int, k: int, p: float, num_processors: int) -> nx.DiGraph:
    k = min(k if k % 2 == 0 else k + 1, n - 1)
    g = nx.watts_strogatz_graph(
        n,
        k,
        p,
        seed=random.randint(0, 10_000),
    )
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
    dag.add_edges_from([
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (3, 4),
        (3, 5),
    ])
    return add_costs(dag, num_processors)


def build_dataset(
    num_graphs: int,
    n_values: List[int],
    num_processors: int,
) -> List[Tuple[str, nx.DiGraph]]:
    dataset = []

    for _ in range(num_graphs):
        n = random.choice(n_values)
        kind = random.choice(["ER", "BA", "WS"])

        if kind == "ER":
            dag = random_er_dag(
                n,
                p=random.uniform(0.05, 0.18),
                num_processors=num_processors,
            )
        elif kind == "BA":
            dag = random_ba_dag(
                n,
                m=random.randint(1, min(4, n - 1)),
                num_processors=num_processors,
            )
        else:
            dag = random_ws_dag(
                n,
                k=min(4, n - 1),
                p=random.uniform(0.1, 0.4),
                num_processors=num_processors,
            )

        dataset.append((kind, dag))

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

FEATURE_NAMES = [
    "upward_rank",
    "downward_rank",
    "depth",
    "fanout",
    "indegree",
    "avg_comm_out",
    "avg_cost",
]


def avg_cost(dag: nx.DiGraph, v: int) -> float:
    return float(np.mean(dag.nodes[v]["costs"]))


def upward_rank(dag: nx.DiGraph) -> Dict[int, float]:
    rank = {}
    for v in reversed(list(nx.topological_sort(dag))):
        succ = list(dag.successors(v))
        if not succ:
            rank[v] = avg_cost(dag, v)
        else:
            rank[v] = avg_cost(dag, v) + max(
                dag.edges[v, u]["comm"] + rank[u]
                for u in succ
            )
    return rank


def downward_rank(dag: nx.DiGraph) -> Dict[int, float]:
    rank = {}
    for v in nx.topological_sort(dag):
        pred = list(dag.predecessors(v))
        if not pred:
            rank[v] = avg_cost(dag, v)
        else:
            rank[v] = avg_cost(dag, v) + max(
                dag.edges[u, v]["comm"] + rank[u]
                for u in pred
            )
    return rank


def node_depths(dag: nx.DiGraph) -> Dict[int, int]:
    depth = {}
    for v in nx.topological_sort(dag):
        preds = list(dag.predecessors(v))
        depth[v] = 0 if not preds else 1 + max(depth[p] for p in preds)
    return depth


def extract_node_features(dag: nx.DiGraph) -> Dict[int, Dict[str, float]]:
    ur = upward_rank(dag)
    dr = downward_rank(dag)
    dep = node_depths(dag)
    feats = {}

    for v in dag.nodes():
        out_comms = [dag.edges[v, u]["comm"] for u in dag.successors(v)]
        feats[v] = {
            "upward_rank": ur[v],
            "downward_rank": dr[v],
            "depth": float(dep[v]),
            "fanout": float(dag.out_degree(v)),
            "indegree": float(dag.in_degree(v)),
            "avg_comm_out": float(np.mean(out_comms)) if out_comms else 0.0,
            "avg_cost": avg_cost(dag, v),
        }

    return feats


def normalize_features(
    feats: Dict[int, Dict[str, float]]
) -> Dict[int, Dict[str, float]]:
    out = {v: {} for v in feats}
    for name in FEATURE_NAMES:
        vals = np.array([feats[v][name] for v in feats], dtype=float)
        lo, hi = float(vals.min()), float(vals.max())
        for v in feats:
            out[v][name] = 0.0 if hi == lo else (feats[v][name] - lo) / (hi - lo)
    return out


def motif_counts(dag: nx.DiGraph) -> Dict[str, int]:
    counts = {
        "chain_nodes": 0,
        "fork_nodes": 0,
        "join_nodes": 0,
        "diamond_like_nodes": 0,
    }

    for v in dag.nodes():
        indeg = dag.in_degree(v)
        outdeg = dag.out_degree(v)
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
        "motifs": motif_counts(dag),
        "max_depth": max(depths.values()) if depths else 0,
        "avg_fanout": float(np.mean([dag.out_degree(v) for v in dag.nodes()])),
        "avg_indegree": float(np.mean([dag.in_degree(v) for v in dag.nodes()])),
        "avg_task_cost": float(np.mean([avg_cost(dag, v) for v in dag.nodes()])),
        "avg_comm_cost": float(np.mean([dag.edges[e]["comm"] for e in dag.edges()]))
        if dag.number_of_edges() > 0 else 0.0,
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
# List scheduling
# ============================================================

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
        same_processor = assignment.get(parent) == proc
        comm = 0 if same_processor else dag.edges[parent, task]["comm"]
        ready_time = max(ready_time, finish_times[parent] + comm)

    start = max(proc_available[proc], ready_time)
    finish = start + dag.nodes[task]["costs"][proc]
    return start, finish


def list_schedule(
    dag: nx.DiGraph,
    num_processors: int,
    priority_fn,
) -> ScheduleResult:
    completed = set()
    scheduled = set()
    assignment = {}
    start_times = {}
    finish_times = {}
    proc_available = [0.0] * num_processors

    while len(completed) < dag.number_of_nodes():
        ready = [
            v for v in dag.nodes()
            if v not in scheduled
            and all(parent in completed for parent in dag.predecessors(v))
        ]
        if not ready:
            raise RuntimeError("No ready tasks found. Please check whether graph is a DAG.")

        task = max(ready, key=priority_fn)
        best_proc = None
        best_start = None
        best_finish = float("inf")

        for p in range(num_processors):
            start, finish = earliest_start_finish(
                dag, task, p, proc_available, finish_times, assignment
            )
            if finish < best_finish:
                best_proc = p
                best_start = start
                best_finish = finish

        assignment[task] = int(best_proc)
        start_times[task] = float(best_start)
        finish_times[task] = float(best_finish)
        proc_available[best_proc] = float(best_finish)
        scheduled.add(task)
        completed.add(task)

    return ScheduleResult(
        makespan=max(finish_times.values()),
        assignment=assignment,
        start_times=start_times,
        finish_times=finish_times,
    )


def schedule_with_priority_order(
    dag: nx.DiGraph,
    num_processors: int,
    priority_order_policy,
) -> ScheduleResult:
    completed = set()
    scheduled = set()
    assignment = {}
    start_times = {}
    finish_times = {}
    proc_available = [0.0] * num_processors

    while len(completed) < dag.number_of_nodes():
        ready = [
            v for v in dag.nodes()
            if v not in scheduled
            and all(parent in completed for parent in dag.predecessors(v))
        ]
        if not ready:
            raise RuntimeError("No ready tasks found. Please check whether graph is a DAG.")

        task = priority_order_policy(ready, completed)
        best_proc = None
        best_start = None
        best_finish = float("inf")

        for p in range(num_processors):
            start, finish = earliest_start_finish(
                dag, task, p, proc_available, finish_times, assignment
            )
            if finish < best_finish:
                best_proc = p
                best_start = start
                best_finish = finish

        assignment[task] = int(best_proc)
        start_times[task] = float(best_start)
        finish_times[task] = float(best_finish)
        proc_available[best_proc] = float(best_finish)
        scheduled.add(task)
        completed.add(task)

    return ScheduleResult(
        makespan=max(finish_times.values()),
        assignment=assignment,
        start_times=start_times,
        finish_times=finish_times,
    )


def critical_path_fastest_processor(dag: nx.DiGraph) -> float:
    fastest = {v: min(dag.nodes[v]["costs"]) for v in dag.nodes()}
    dist = {}

    for v in nx.topological_sort(dag):
        preds = list(dag.predecessors(v))
        if not preds:
            dist[v] = fastest[v]
        else:
            dist[v] = fastest[v] + max(dist[p] for p in preds)

    return max(dist.values()) if dist else 0.0


# ============================================================
# Baselines and weighted priority heuristic
# ============================================================

def heft_schedule(dag: nx.DiGraph, num_processors: int) -> ScheduleResult:
    ur = upward_rank(dag)
    return list_schedule(dag, num_processors, lambda v: ur[v])


def cpop_schedule(dag: nx.DiGraph, num_processors: int) -> ScheduleResult:
    ur = upward_rank(dag)
    dr = downward_rank(dag)
    return list_schedule(dag, num_processors, lambda v: ur[v] + dr[v])


def weighted_feature_schedule(
    dag: nx.DiGraph,
    num_processors: int,
    weights: Dict[str, float],
) -> ScheduleResult:
    feats = normalize_features(extract_node_features(dag))
    return list_schedule(
        dag,
        num_processors,
        lambda v: sum(weights.get(name, 0.0) * feats[v][name] for name in FEATURE_NAMES),
    )


# ============================================================
# Bayesian Optimization for LLM weights
# ============================================================

def bayes_optimize_llm_weights(
    train_data: List[Tuple[str, nx.DiGraph]],
    num_processors: int,
    initial_weights: Dict[str, float],
    n_calls: int = 30,
    seed: int = 7,
) -> Dict[str, float]:
    if gp_minimize is None:
        print("[WARN] scikit-optimize not installed. Using raw LLM weights.")
        return initial_weights

    search_space = [Real(-2.0, 2.0, name=name) for name in FEATURE_NAMES]
    initial_point = [float(initial_weights.get(name, 0.0)) for name in FEATURE_NAMES]

    @use_named_args(search_space)
    def objective(**weights):
        total_makespan = 0.0
        for _, dag in train_data:
            result = weighted_feature_schedule(
                dag=dag,
                num_processors=num_processors,
                weights=weights,
            )
            total_makespan += result.makespan
        return total_makespan / len(train_data)

    result = gp_minimize(
        func=objective,
        dimensions=search_space,
        x0=initial_point,
        n_calls=n_calls,
        random_state=seed,
        acq_func="EI",
    )

    optimized_weights = {
        name: float(value)
        for name, value in zip(FEATURE_NAMES, result.x)
    }

    print("\nBayesian Optimization completed.")
    print(f"Best training makespan: {result.fun:.2f}")
    print("BO-optimized weights:")
    print(json.dumps(optimized_weights, indent=2))
    return optimized_weights


# ============================================================
# LLM Role 1: propose priority weights
# ============================================================

def llm_propose_weights(dag: nx.DiGraph, model: str = "gpt-4o-mini") -> Dict[str, float]:
    fallback = {
        "upward_rank": 1.0,
        "downward_rank": 0.25,
        "depth": 0.30,
        "fanout": 0.20,
        "indegree": 0.05,
        "avg_comm_out": 0.35,
        "avg_cost": 0.10,
    }

    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        print("[LLM fallback] OPENAI_API_KEY missing or openai package unavailable.")
        return fallback

    stats = graph_summary(dag)
    stats["instruction"] = "Return JSON only: feature weights for a list-scheduling priority function."
    stats["template"] = "H(v)=sum_i theta_i*f_i(v). Higher score means schedule earlier."

    prompt = f"""
You are helping design an interpretable DAG scheduling heuristic for HLS/list scheduling.

Given the DAG summary, propose numeric weights for normalized structural features.

Rules:
- Return JSON only.
- Keys must be exactly these feature names:
  {FEATURE_NAMES}
- Values must be floats between -2 and 2.
- Favor critical-path operations, communication-heavy operations, and useful parallelism.
- Penalize features only when early scheduling of that feature may cause synchronization or resource contention.
- Do not produce explanation. Return only JSON.

DAG summary:
{json.dumps(stats, indent=2)}
"""

    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        return {name: float(data.get(name, fallback[name])) for name in FEATURE_NAMES}
    except Exception as e:
        print(f"[LLM fallback] Could not call OpenAI API: {e}")
        return fallback


# ============================================================
# Schedule analysis helpers for LLM Role 2
# ============================================================

def processor_intervals(schedule: ScheduleResult) -> Dict[int, List[Dict[str, float]]]:
    intervals: Dict[int, List[Dict[str, float]]] = {}
    for task, proc in schedule.assignment.items():
        intervals.setdefault(proc, []).append({
            "task": int(task),
            "start": float(schedule.start_times[task]),
            "finish": float(schedule.finish_times[task]),
            "duration": float(schedule.finish_times[task] - schedule.start_times[task]),
        })
    for proc in intervals:
        intervals[proc] = sorted(intervals[proc], key=lambda x: x["start"])
    return intervals


def find_critical_path(dag: nx.DiGraph) -> List[int]:
    """Approximate critical path using average task cost and communication cost."""
    dist = {}
    parent = {}

    for v in nx.topological_sort(dag):
        preds = list(dag.predecessors(v))
        if not preds:
            dist[v] = avg_cost(dag, v)
            parent[v] = None
        else:
            best_pred = max(preds, key=lambda p: dist[p] + dag.edges[p, v]["comm"])
            dist[v] = dist[best_pred] + dag.edges[best_pred, v]["comm"] + avg_cost(dag, v)
            parent[v] = best_pred

    if not dist:
        return []

    end = max(dist, key=dist.get)
    path = []
    while end is not None:
        path.append(end)
        end = parent[end]
    return list(reversed(path))


def detect_independent_parallel_groups(dag: nx.DiGraph, limit: int = 5) -> List[List[int]]:
    levels: Dict[int, List[int]] = {}
    depths = node_depths(dag)
    for v, d in depths.items():
        levels.setdefault(d, []).append(v)

    groups = []
    for _, nodes in sorted(levels.items()):
        group = []
        for v in nodes:
            independent = True
            for u in group:
                if nx.has_path(dag, u, v) or nx.has_path(dag, v, u):
                    independent = False
                    break
            if independent:
                group.append(v)
        if len(group) >= 2:
            groups.append([int(x) for x in group[:8]])
        if len(groups) >= limit:
            break
    return groups


def candidate_pipeline_edges(dag: nx.DiGraph, schedule: ScheduleResult, limit: int = 10) -> List[Dict[str, Any]]:
    cp = set(find_critical_path(dag))
    candidates = []
    for u, v in dag.edges():
        producer_finish = schedule.finish_times[u]
        consumer_start = schedule.start_times[v]
        gap = max(0.0, consumer_start - producer_finish)
        comm = dag.edges[u, v]["comm"]
        score = gap + comm
        if u in cp and v in cp:
            score += 10.0
        candidates.append({
            "edge": [int(u), int(v)],
            "producer_finish": float(producer_finish),
            "consumer_start": float(consumer_start),
            "schedule_gap": float(gap),
            "communication_cost": float(comm),
            "on_critical_path": bool(u in cp and v in cp),
            "reason": "possible register/FIFO/buffer boundary; useful only if timing, buffering, or throughput constraints justify it",
            "score": float(score),
        })
    candidates.sort(key=lambda x: x["score"], reverse=True)
    for c in candidates:
        c.pop("score", None)
    return candidates[:limit]


def resource_contention_summary(dag: nx.DiGraph, schedule: ScheduleResult, num_processors: int) -> Dict[str, Any]:
    intervals = processor_intervals(schedule)
    busy_time = {}
    for p in range(num_processors):
        busy_time[p] = sum(item["duration"] for item in intervals.get(p, []))

    makespan = max(schedule.finish_times.values()) if schedule.finish_times else 0.0
    utilization = {p: (busy_time[p] / makespan if makespan > 0 else 0.0) for p in range(num_processors)}
    sorted_util = sorted(utilization.items(), key=lambda x: x[1], reverse=True)

    return {
        "processor_busy_time": {int(k): float(v) for k, v in busy_time.items()},
        "processor_utilization": {int(k): float(v) for k, v in utilization.items()},
        "most_loaded_processors": [{"processor": int(p), "utilization": float(u)} for p, u in sorted_util[:3]],
        "least_loaded_processors": [{"processor": int(p), "utilization": float(u)} for p, u in sorted_util[-3:]],
    }


def build_schedule_insight_payload(dag: nx.DiGraph, schedule_result: ScheduleResult, method_name: str, num_processors: int) -> Dict[str, Any]:
    feats = extract_node_features(dag)
    cp = find_critical_path(dag)
    cp_set = set(cp)

    sorted_nodes = sorted(
        dag.nodes(),
        key=lambda v: (
            1 if v in cp_set else 0,
            feats[v]["upward_rank"],
            schedule_result.finish_times[v] - schedule_result.start_times[v],
            dag.out_degree(v),
            feats[v]["avg_comm_out"],
        ),
        reverse=True,
    )

    bottlenecks = []
    for v in sorted_nodes[:10]:
        bottlenecks.append({
            "op_id": int(v),
            "processor": int(schedule_result.assignment[v]),
            "start": float(schedule_result.start_times[v]),
            "finish": float(schedule_result.finish_times[v]),
            "duration": float(schedule_result.finish_times[v] - schedule_result.start_times[v]),
            "on_critical_path": bool(v in cp_set),
            "upward_rank": float(feats[v]["upward_rank"]),
            "downward_rank": float(feats[v]["downward_rank"]),
            "depth": float(feats[v]["depth"]),
            "fanout": int(dag.out_degree(v)),
            "indegree": int(dag.in_degree(v)),
            "avg_comm_out": float(feats[v]["avg_comm_out"]),
            "successors": [int(s) for s in dag.successors(v)],
            "predecessors": [int(p) for p in dag.predecessors(v)],
        })

    return {
        "method": method_name,
        "graph_summary": graph_summary(dag),
        "makespan": float(schedule_result.makespan),
        "critical_path_nodes": [int(v) for v in cp],
        "bottleneck_operations": bottlenecks,
        "independent_parallel_groups": detect_independent_parallel_groups(dag),
        "candidate_register_or_memory_edges": candidate_pipeline_edges(dag, schedule_result),
        "resource_contention_summary": resource_contention_summary(dag, schedule_result, num_processors),
        "processor_schedule_intervals": processor_intervals(schedule_result),
        "interpretation_note": (
            "All suggested parallelism must respect DAG dependencies. "
            "Register/FIFO/memory insertion suggestions should refer to candidate edges or dependency boundaries listed above. "
            "The DAG abstraction does not include operation type, RTL internals, timing slack, memory banking, or DSP/ALU type, "
            "so any internal pipelining, operation splitting, or resource-specific recommendation must be stated as a hypothesis."
        ),
    }


# ============================================================
# LLM Role 2: schedule insights and PPA recommendations
# ============================================================

def llm_schedule_insights(
    dag: nx.DiGraph,
    schedule_result: ScheduleResult,
    method_name: str,
    num_processors: int,
    model: str = "gpt-4o-mini",
) -> str:
    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        return "[LLM insights skipped] OPENAI_API_KEY not found or openai package unavailable."

    payload = build_schedule_insight_payload(
        dag=dag,
        schedule_result=schedule_result,
        method_name=method_name,
        num_processors=num_processors,
    )

    prompt = f"""
You are an expert in high-level synthesis (HLS), DAG scheduling, FPGA/ASIC optimization, and PPA-aware hardware design.

Analyze the scheduled DAG and provide practical scheduling and hardware optimization insights.

Your answer must focus on the following requested logic:

TOP SECTION: Heuristics -> Bottlenecks
Identify the operations that hurt the schedule the most.
Use these bottleneck categories:
- long-latency operations
- operations that cause schedule variation by blocking otherwise independent operations
- operations that belong to the critical path
- operations affected by resource availability or resource contention
- high-fanout or communication-heavy operations
- dependency-induced serialization points

MIDDLE SECTION: Parallelism and Scheduling
Explain how to improve parallelism.
Discuss:
- which independent operations can execute simultaneously
- which operations are unnecessarily serialized
- whether adding new resources can improve parallelism
- which bottlenecks justify adding new resources
- whether resource reuse is helping area/power or hurting latency
- how rescheduling can better overlap independent operations

PIPELINING SECTION:
Explain how rescheduling can be improved by adding memory elements.
Discuss:
- where registers, FIFOs, buffers, or memory elements should be introduced
- which producer-consumer edges are good candidates for pipeline/register insertion
- how pipelining may improve clock frequency, throughput, and stage balance without violating producer-consumer dependencies
- whether pipeline stages should split long operations or critical dependency chains

BOTTOM QUESTION:
Explicitly answer this question:
"Where do you introduce the register/memory to achieve pipelining/scheduling?"

PPA SECTION:
Give PPA-aware recommendations:
- latency reduction
- area optimization
- power reduction
- throughput improvement
- tradeoff between adding resources and reusing resources

Return the answer exactly in this structure:

A. Heuristic and Bottleneck Analysis
- Major bottleneck operations:
- Why they are bottlenecks:
- Critical-path operations:
- Resource availability issues:
- Dependency-induced serialization:

B. Parallelism and Scheduling Opportunities
- Independent operations that can run in parallel:
- Operations that should be rescheduled:
- Resources that should be added to increase parallelism:
- Cases where resource reuse is good:
- Cases where resource reuse hurts latency:

C. Pipelining and Register/Memory Insertion Guidance
- Best locations to insert registers/FIFOs/buffers/memory:
- Candidate dependency edges for pipelining:
- Suggested pipeline stages:
- How pipelining improves scheduling:

D. Direct Answer: Where to Introduce Register/Memory?
- Give a clear edge-level or boundary-level answer using operation IDs.
- Mention why that location helps.

E. PPA Optimization Recommendations
- Latency:
- Area:
- Power:
- Throughput:
- Tradeoff summary:

F. Next Actions for the HLS Designer
- Concrete next steps:
- Scheduling changes:
- Hardware/resource changes:
- What to verify after changes:

Important constraints:
- Do NOT suggest impossible parallelism that violates DAG precedence constraints.
- Use the provided independent_parallel_groups when discussing parallelism.
- Use the provided candidate_register_or_memory_edges when discussing register/FIFO/memory insertion.
- Do NOT claim that registers/memory allow a dependent consumer operation to start before its producer data is available.
- Explain that registers/FIFOs/buffers mainly help by breaking long combinational paths, improving timing closure, balancing pipeline stages, and improving throughput.
- Register insertion may improve clock frequency and throughput, but it may not always reduce single-instance DAG latency.
- Avoid recommending registers on every incoming edge of a join node. Prefer only critical-path, high-communication, high-gap, or high-delay candidate edges.
- If suggesting internal pipelining or splitting a long operation, explicitly state: "This assumes the operation can be decomposed or internally pipelined; the DAG alone does not prove this."
- If a pipelining or decomposition suggestion depends on operation internals that are not visible in the DAG abstraction, explicitly state the assumption.
- If operation type, RTL structure, memory banking, DSP/ALU type, or timing slack is unknown, say the recommendation is a hypothesis.
- Focus on scheduling-aware hardware optimization, not generic advice.
- Never say that register insertion allows a dependent consumer node to start earlier unless retiming, buffering, or protocol-level decoupling is explicitly modeled. For normal DAG scheduling, the consumer must still wait for producer data.
- Keep the answer concise but technically meaningful.

Scheduled DAG data:
{json.dumps(payload, indent=2)}
"""

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


# ============================================================
# GNN models
# ============================================================

class TinyDAGGNN(nn.Module):
    """
    Small message-passing GNN over DAG edges.
    Used for GNN imitation and Decima-like RL.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.input = nn.Linear(in_dim, hidden_dim)
        self.msg = nn.Linear(hidden_dim, hidden_dim)
        self.update = nn.GRUCell(hidden_dim, hidden_dim)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: "torch.Tensor", edge_index: "torch.Tensor") -> "torch.Tensor":
        h = torch.relu(self.input(x))
        for _ in range(3):
            agg = torch.zeros_like(h)
            if edge_index.numel() > 0:
                src, dst = edge_index[0], edge_index[1]
                messages = self.msg(h[src])
                agg.index_add_(0, dst, messages)
            h = self.update(agg, h)
        return self.out(h).squeeze(-1)


def graph_to_tensors(dag: nx.DiGraph):
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
    epochs: int = 100,
    lr: float = 1e-3,
) -> Optional[Any]:
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
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"[GNN imitation] epoch={epoch + 1:03d}, loss={total_loss / len(train_data):.4f}")

    return model


def gnn_schedule(model: Any, dag: nx.DiGraph, num_processors: int) -> ScheduleResult:
    nodes, _, x, edge_index = graph_to_tensors(dag)
    with torch.no_grad():
        scores = model(x, edge_index).detach().cpu().numpy()
    score_map = {v: float(scores[i]) for i, v in enumerate(nodes)}
    return list_schedule(dag, num_processors, lambda v: score_map[v])


def train_decima_like_rl(
    train_data: List[Tuple[str, nx.DiGraph]],
    num_processors: int,
    episodes: int = 80,
    lr: float = 1e-3,
) -> Optional[Any]:
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

        def policy(ready: List[int], completed: set) -> int:
            scores = model(x, edge_index)
            ready_idx = torch.tensor([node_to_idx[v] for v in ready], dtype=torch.long)
            ready_scores = scores[ready_idx]
            probs = torch.softmax(ready_scores, dim=0)
            dist = torch.distributions.Categorical(probs)
            action_pos = dist.sample()
            log_probs.append(dist.log_prob(action_pos))
            return ready[int(action_pos.item())]

        result = schedule_with_priority_order(dag, num_processors, policy)
        reward = -result.makespan
        baseline = reward if baseline is None else 0.9 * baseline + 0.1 * reward
        advantage = reward - baseline
        loss = -torch.stack(log_probs).sum() * float(advantage)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if (ep + 1) % max(1, episodes // 5) == 0:
            print(f"[Decima-like RL] episode={ep + 1:03d}, makespan={result.makespan:.2f}, reward={reward:.2f}")

    return model


# ============================================================
# Evaluation
# ============================================================

def evaluate_method(
    name: str,
    data: List[Tuple[str, nx.DiGraph]],
    num_processors: int,
    schedule_fn,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, nx.DiGraph, ScheduleResult]]]:
    rows = []
    schedules = []

    for family, dag in data:
        t0 = time.time()
        result = schedule_fn(dag)
        runtime = time.time() - t0
        cp = critical_path_fastest_processor(dag)
        rows.append({
            "method": name,
            "family": family,
            "nodes": dag.number_of_nodes(),
            "edges": dag.number_of_edges(),
            "makespan": result.makespan,
            "slr": result.makespan / cp if cp > 0 else float("nan"),
            "runtime_ms": runtime * 1000,
            "motifs": motif_counts(dag),
        })
        schedules.append((family, dag, result))

    return rows, schedules


def summarize(rows: List[Dict[str, Any]]) -> None:
    by_method = {}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)

    print("\n=== Aggregate Results ===")
    print(f"{'Method':<20} {'Avg Makespan':>14} {'Avg SLR':>10} {'Runtime ms':>12}")

    for method, rs in by_method.items():
        print(
            f"{method:<20} "
            f"{np.mean([r['makespan'] for r in rs]):>14.2f} "
            f"{np.mean([r['slr'] for r in rs]):>10.3f} "
            f"{np.mean([r['runtime_ms'] for r in rs]):>12.3f}"
        )

    if "HEFT" in by_method:
        heft = by_method["HEFT"]
        print("\n=== Win Count vs HEFT ===")
        for method, rs in by_method.items():
            if method == "HEFT":
                continue
            wins = sum(r["makespan"] < h["makespan"] for r, h in zip(rs, heft))
            ties = sum(abs(r["makespan"] - h["makespan"]) < 1e-9 for r, h in zip(rs, heft))
            print(f"{method:<20} wins={wins:>3}, ties={ties:>3}, total={len(heft)}")

    print("\n=== Example Per-DAG Rows ===")
    for row in rows[:min(12, len(rows))]:
        print(
            f"{row['method']:<20} "
            f"family={row['family']:<12} "
            f"nodes={row['nodes']:<3} "
            f"edges={row['edges']:<3} "
            f"makespan={row['makespan']:<8.2f} "
            f"slr={row['slr']:.3f}"
        )


def save_csv(rows: List[Dict[str, Any]], path: str) -> None:
    import csv
    if not rows:
        print("[WARN] No rows to save.")
        return
    flat_rows = []
    for row in rows:
        flat = dict(row)
        flat["motifs"] = json.dumps(flat["motifs"])
        flat_rows.append(flat)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"\nSaved detailed results to: {path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-processors", type=int, default=4)
    parser.add_argument("--num-graphs", type=int, default=36)
    parser.add_argument("--train-episodes", type=int, default=80)
    parser.add_argument("--gnn-epochs", type=int, default=80)
    parser.add_argument("--bo-calls", type=int, default=30)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--openai-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--out", type=str, default="sample_scheduler_results.csv")
    parser.add_argument(
        "--insight-method",
        type=str,
        default="LLM+BO-Heuristic",
        choices=["HEFT", "CPOP", "LLM-Heuristic", "LLM+BO-Heuristic", "GNN-Imitation", "Decima-like-RL"],
        help="Which method schedule should be sent to the LLM insight module.",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    dataset = build_dataset(
        num_graphs=args.num_graphs,
        n_values=[20, 30, 50],
        num_processors=args.num_processors,
    )
    random.shuffle(dataset)

    split = int(0.7 * len(dataset))
    train_data = dataset[:split]
    test_data = dataset[split:]

    print(f"Train DAGs: {len(train_data)}, Test DAGs: {len(test_data)}")
    print(f"Processors: {args.num_processors}")

    llm_weights = None
    bo_weights = None

    if args.use_llm:
        llm_weights = llm_propose_weights(test_data[0][1], model=args.openai_model)
        print("\nLLM proposed weights:")
        print(json.dumps(llm_weights, indent=2))

        bo_weights = bayes_optimize_llm_weights(
            train_data=train_data,
            num_processors=args.num_processors,
            initial_weights=llm_weights,
            n_calls=args.bo_calls,
            seed=args.seed,
        )

    gnn_model = train_gnn_imitation(train_data, epochs=args.gnn_epochs)
    rl_model = train_decima_like_rl(train_data, args.num_processors, episodes=args.train_episodes)

    all_rows: List[Dict[str, Any]] = []
    schedules_by_method: Dict[str, List[Tuple[str, nx.DiGraph, ScheduleResult]]] = {}

    rows, schedules = evaluate_method(
        "HEFT", test_data, args.num_processors,
        lambda dag: heft_schedule(dag, args.num_processors),
    )
    all_rows += rows
    schedules_by_method["HEFT"] = schedules

    rows, schedules = evaluate_method(
        "CPOP", test_data, args.num_processors,
        lambda dag: cpop_schedule(dag, args.num_processors),
    )
    all_rows += rows
    schedules_by_method["CPOP"] = schedules

    if llm_weights is not None:
        rows, schedules = evaluate_method(
            "LLM-Heuristic", test_data, args.num_processors,
            lambda dag: weighted_feature_schedule(dag, args.num_processors, llm_weights),
        )
        all_rows += rows
        schedules_by_method["LLM-Heuristic"] = schedules

    if bo_weights is not None:
        rows, schedules = evaluate_method(
            "LLM+BO-Heuristic", test_data, args.num_processors,
            lambda dag: weighted_feature_schedule(dag, args.num_processors, bo_weights),
        )
        all_rows += rows
        schedules_by_method["LLM+BO-Heuristic"] = schedules

    if gnn_model is not None:
        rows, schedules = evaluate_method(
            "GNN-Imitation", test_data, args.num_processors,
            lambda dag: gnn_schedule(gnn_model, dag, args.num_processors),
        )
        all_rows += rows
        schedules_by_method["GNN-Imitation"] = schedules

    if rl_model is not None:
        rows, schedules = evaluate_method(
            "Decima-like-RL", test_data, args.num_processors,
            lambda dag: gnn_schedule(rl_model, dag, args.num_processors),
        )
        all_rows += rows
        schedules_by_method["Decima-like-RL"] = schedules

    summarize(all_rows)
    save_csv(all_rows, args.out)

    if args.use_llm:
        print("\n=== LLM Schedule Insights ===")

        family, example_dag = test_data[0]

    # Use HEFT schedule as example, but you can change this to LLM-Heuristic or GNN.
        example_schedule = heft_schedule(example_dag, args.num_processors)

        insights = llm_schedule_insights(
            dag=example_dag,
            schedule_result=example_schedule,
            method_name=insight_method,
            num_processors=args.num_processors,
            model=args.openai_model,
        )

        print(f"\nDAG family: {family}")
        print(f"Insight method: {insight_method}")
        print(insights)


if __name__ == "__main__":
    main()
