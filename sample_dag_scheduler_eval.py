#!/usr/bin/env python3
"""
sample_dag_scheduler_eval.py

Single-file demo for DAG scheduling evaluation:
- Generates sample synthetic DAGs: ER, BA, WS, and motif DAGs: chain, fork, join, diamond.
- Extracts structural features and simple motif counts.
- Evaluates classical baselines: HEFT-like and CPOP-like list scheduling.
- Trains/evaluates a Decima-style RL scheduler using a small GNN policy with REINFORCE.
- Evaluates a GNN supervised scheduler trained to imitate HEFT priorities.
- Optionally calls an LLM using OPENAI_API_KEY to propose feature weights for a priority heuristic.

This is a research-demo script, not a full reproduction of Decima.
Decima is approximated here as: GNN embeddings + policy network + reinforcement learning over ready-task decisions.

Run:
    pip install networkx numpy torch openai
    export OPENAI_API_KEY="your-key"        # optional
    python sample_dag_scheduler_eval.py --use-llm --train-episodes 80

Without OpenAI:
    python sample_dag_scheduler_eval.py --train-episodes 80
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

import networkx as nx
import numpy as np

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


# -----------------------------
# Reproducibility
# -----------------------------

def set_seed(seed: int = 7) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)


# -----------------------------
# DAG generation
# -----------------------------

def ensure_dag_by_orientation(g: nx.Graph) -> nx.DiGraph:
    """Orient undirected edges from lower topological index to higher index."""
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
    return nx.transitive_reduction(dag) if nx.is_directed_acyclic_graph(dag) else dag


def add_costs(dag: nx.DiGraph, num_processors: int = 4) -> nx.DiGraph:
    """Add heterogeneous execution costs and communication costs."""
    for v in dag.nodes():
        base = random.randint(5, 30)
        dag.nodes[v]["costs"] = [max(1, int(base * random.uniform(0.7, 1.4))) for _ in range(num_processors)]
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
    g = nx.barabasi_albert_graph(n, max(1, min(m, n - 1)), seed=random.randint(0, 10_000))
    return add_costs(ensure_dag_by_orientation(g), num_processors)


def random_ws_dag(n: int, k: int, p: float, num_processors: int) -> nx.DiGraph:
    k = min(k if k % 2 == 0 else k + 1, n - 1)
    g = nx.watts_strogatz_graph(n, k, p, seed=random.randint(0, 10_000))
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
    dataset = []
    for i in range(num_graphs):
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


# -----------------------------
# Feature extraction
# -----------------------------

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
            rank[v] = avg_cost(dag, v) + max(dag.edges[v, u]["comm"] + rank[u] for u in succ)
    return rank


def downward_rank(dag: nx.DiGraph) -> Dict[int, float]:
    rank = {}
    for v in nx.topological_sort(dag):
        pred = list(dag.predecessors(v))
        if not pred:
            rank[v] = avg_cost(dag, v)
        else:
            rank[v] = avg_cost(dag, v) + max(dag.edges[u, v]["comm"] + rank[u] for u in pred)
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


def normalize_features(feats: Dict[int, Dict[str, float]]) -> Dict[int, Dict[str, float]]:
    out = {v: {} for v in feats}
    for name in FEATURE_NAMES:
        vals = np.array([feats[v][name] for v in feats], dtype=float)
        lo, hi = float(vals.min()), float(vals.max())
        for v in feats:
            out[v][name] = 0.0 if hi == lo else (feats[v][name] - lo) / (hi - lo)
    return out


def motif_counts(dag: nx.DiGraph) -> Dict[str, int]:
    counts = {
        "chain_edges": 0,
        "fork_nodes": 0,
        "join_nodes": 0,
        "diamond_like_nodes": 0,
    }
    for v in dag.nodes():
        indeg, outdeg = dag.in_degree(v), dag.out_degree(v)
        if indeg == 1 and outdeg == 1:
            counts["chain_edges"] += 1
        if outdeg >= 2:
            counts["fork_nodes"] += 1
        if indeg >= 2:
            counts["join_nodes"] += 1
        if indeg >= 2 and outdeg >= 2:
            counts["diamond_like_nodes"] += 1
    return counts


# -----------------------------
# List scheduling
# -----------------------------

@dataclass
class ScheduleResult:
    makespan: float
    assignment: Dict[int, int]
    start_times: Dict[int, float]
    finish_times: Dict[int, float]


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
        comm = 0 if assignment.get(parent) == proc else dag.edges[parent, task]["comm"]
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
    assignment, start_times, finish_times = {}, {}, {}
    proc_available = [0.0] * num_processors

    while len(completed) < dag.number_of_nodes():
        ready = [
            v for v in dag.nodes()
            if v not in scheduled and all(p in completed for p in dag.predecessors(v))
        ]
        if not ready:
            raise RuntimeError("No ready tasks found. Is the graph a DAG?")

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


def schedule_with_priority_order(
    dag: nx.DiGraph,
    num_processors: int,
    priority_order_policy,
) -> ScheduleResult:
    """Used by RL policy, which selects from ready tasks dynamically."""
    completed = set()
    scheduled = set()
    assignment, start_times, finish_times = {}, {}, {}
    proc_available = [0.0] * num_processors

    while len(completed) < dag.number_of_nodes():
        ready = [
            v for v in dag.nodes()
            if v not in scheduled and all(p in completed for p in dag.predecessors(v))
        ]
        task = priority_order_policy(ready, completed)

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


# -----------------------------
# Baseline priority functions
# -----------------------------

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


def bayes_optimize_llm_weights(
    train_data,
    num_processors,
    initial_weights,
    n_calls=30,
):
    """
    Bayesian Optimization improves the LLM-generated feature weights.

    Input:
        initial_weights = weights proposed by LLM

    Output:
        optimized_weights = BO-refined weights
    """

    search_space = [
        Real(-2.0, 2.0, name=name)
        for name in FEATURE_NAMES
    ]

    initial_point = [
        initial_weights.get(name, 0.0)
        for name in FEATURE_NAMES
    ]

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

        avg_makespan = total_makespan / len(train_data)

        return avg_makespan

    result = gp_minimize(
        func=objective,
        dimensions=search_space,
        x0=initial_point,
        n_calls=n_calls,
        random_state=7,
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

# -----------------------------
# LLM feature-weight proposal
# -----------------------------

def llm_propose_weights(dag: nx.DiGraph, model: str = "gpt-4o-mini") -> Dict[str, float]:
    """
    Optional LLM call. Requires:
        pip install openai
        export OPENAI_API_KEY=...
    """
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
        return fallback

    stats = {
        "num_nodes": dag.number_of_nodes(),
        "num_edges": dag.number_of_edges(),
        "motifs": motif_counts(dag),
        "feature_names": FEATURE_NAMES,
        "instruction": "Return JSON only: feature weights for a list-scheduling priority function.",
        "template": "H(v)=sum_i theta_i*f_i(v). Higher score means schedule earlier.",
    }

    prompt = f"""
You are helping design an interpretable DAG scheduling heuristic for HLS/list scheduling.
Given this DAG summary, propose numeric weights for the listed normalized features.

Rules:
- Return JSON only.
- Keys must be exactly the feature names.
- Values should be floats between -2 and 2.
- Prioritize critical path, communication pressure, and motif structure.

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


def llm_schedule_insights(
    dag: nx.DiGraph,
    schedule_result: ScheduleResult,
    method_name: str,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Second LLM role:
    Analyze the generated schedule and suggest PPA-aware optimization actions.

    Requires:
        export OPENAI_API_KEY=...
        pip install openai
    """

    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        return "[LLM insights skipped] OPENAI_API_KEY not found."

    feats = extract_node_features(dag)
    motifs = motif_counts(dag)

    finish_times = schedule_result.finish_times
    start_times = schedule_result.start_times
    assignment = schedule_result.assignment

    # Find top bottleneck operations
    sorted_nodes = sorted(
        dag.nodes(),
        key=lambda v: (
            feats[v]["upward_rank"],
            dag.out_degree(v),
            feats[v]["avg_comm_out"],
        ),
        reverse=True,
    )

    bottlenecks = []
    for v in sorted_nodes[:8]:
        bottlenecks.append({
            "op_id": int(v),
            "processor": int(assignment[v]),
            "start": float(start_times[v]),
            "finish": float(finish_times[v]),
            "duration": float(finish_times[v] - start_times[v]),
            "upward_rank": float(feats[v]["upward_rank"]),
            "depth": float(feats[v]["depth"]),
            "fanout": int(dag.out_degree(v)),
            "indegree": int(dag.in_degree(v)),
            "avg_comm_out": float(feats[v]["avg_comm_out"]),
        })

    summary = {
        "method": method_name,
        "num_nodes": dag.number_of_nodes(),
        "num_edges": dag.number_of_edges(),
        "makespan": schedule_result.makespan,
        "motif_counts": motifs,
        "bottleneck_operations": bottlenecks,
        "goal": "Suggest scheduling and HLS-level changes that may improve latency, area, power, and resource reuse.",
    }

    prompt = f"""
You are an expert in high-level synthesis, DAG scheduling, and PPA optimization.

Analyze this scheduled DAG and give practical optimization insights.

Focus on:
1. Which operations are bottlenecks and why.
2. Whether to increase parallelism or reduce parallelism.
3. Whether resource reuse is beneficial.
4. Whether pipelining would help.
5. Whether high-fanout or communication-heavy nodes should be optimized.
6. Possible PPA tradeoffs: latency, area, power.
7. Give concise actionable recommendations.

Return the answer in this structure:

A. Bottleneck Summary
B. Scheduling Insight
C. PPA Optimization Recommendation
D. Next Action for HLS Designer

Schedule data:
{json.dumps(summary, indent=2)}
"""

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    return resp.choices[0].message.content.strip()


# -----------------------------
# GNN models
# -----------------------------

class TinyDAGGNN(nn.Module):
    """
    Small message-passing GNN over DAG edges.
    It creates node scores. We use the same architecture for:
    - Decima-like RL policy
    - Supervised GNN policy
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
        n = h.shape[0]

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
    x = torch.tensor([[feats[v][name] for name in FEATURE_NAMES] for v in nodes], dtype=torch.float32)
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
            print(f"[GNN imitation] epoch={epoch+1:03d}, loss={total_loss/len(train_data):.4f}")

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
            print(f"[Decima-like RL] episode={ep+1:03d}, makespan={result.makespan:.2f}, reward={reward:.2f}")

    return model


# -----------------------------
# Evaluation
# -----------------------------

def evaluate_method(
    name: str,
    data: List[Tuple[str, nx.DiGraph]],
    num_processors: int,
    schedule_fn,
) -> List[Dict[str, Any]]:
    rows = []
    for family, dag in data:
        t0 = time.time()
        res = schedule_fn(dag)
        runtime = time.time() - t0
        cp = critical_path_fastest_processor(dag)
        rows.append({
            "method": name,
            "family": family,
            "nodes": dag.number_of_nodes(),
            "edges": dag.number_of_edges(),
            "makespan": res.makespan,
            "slr": res.makespan / cp if cp > 0 else float("nan"),
            "runtime_ms": runtime * 1000,
            "motifs": motif_counts(dag),
        })
    return rows


def summarize(rows: List[Dict[str, Any]]) -> None:
    by_method = {}
    for r in rows:
        by_method.setdefault(r["method"], []).append(r)

    print("\n=== Aggregate Results ===")
    print(f"{'Method':<18} {'Avg Makespan':>14} {'Avg SLR':>10} {'Runtime ms':>12}")
    for method, rs in by_method.items():
        print(
            f"{method:<18} "
            f"{np.mean([r['makespan'] for r in rs]):>14.2f} "
            f"{np.mean([r['slr'] for r in rs]):>10.3f} "
            f"{np.mean([r['runtime_ms'] for r in rs]):>12.3f}"
        )

    methods = list(by_method.keys())
    if "HEFT" in by_method:
        heft = by_method["HEFT"]
        print("\n=== Win Count vs HEFT ===")
        for method in methods:
            if method == "HEFT":
                continue
            wins = sum(r["makespan"] < h["makespan"] for r, h in zip(by_method[method], heft))
            ties = sum(abs(r["makespan"] - h["makespan"]) < 1e-9 for r, h in zip(by_method[method], heft))
            print(f"{method:<18} wins={wins:>3}, ties={ties:>3}, total={len(heft)}")

    print("\n=== Example Per-DAG Rows ===")
    for r in rows[:min(12, len(rows))]:
        print(
            f"{r['method']:<18} family={r['family']:<12} "
            f"nodes={r['nodes']:<3} edges={r['edges']:<3} "
            f"makespan={r['makespan']:<8.2f} slr={r['slr']:.3f}"
        )


def save_csv(rows: List[Dict[str, Any]], path: str) -> None:
    import csv
    flat_rows = []
    for r in rows:
        rr = dict(r)
        rr["motifs"] = json.dumps(rr["motifs"])
        flat_rows.append(rr)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"\nSaved detailed results to: {path}")


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-processors", type=int, default=4)
    parser.add_argument("--num-graphs", type=int, default=36)
    parser.add_argument("--train-episodes", type=int, default=80)
    parser.add_argument("--gnn-epochs", type=int, default=80)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--openai-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--out", type=str, default="sample_scheduler_results.csv")
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

    # LLM proposed heuristic weights: one call using first test DAG.
    # llm_weights = None
    # if args.use_llm:
    #     llm_weights = llm_propose_weights(test_data[0][1], model=args.openai_model)
    #     print("\nLLM proposed weights:")
    #     print(json.dumps(llm_weights, indent=2))

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
            n_calls=30,
      )

    # Train learned methods.
    gnn_model = train_gnn_imitation(train_data, epochs=args.gnn_epochs)
    rl_model = train_decima_like_rl(train_data, args.num_processors, episodes=args.train_episodes)

    all_rows = []

    all_rows += evaluate_method(
        "HEFT",
        test_data,
        args.num_processors,
        lambda dag: heft_schedule(dag, args.num_processors),
    )

    all_rows += evaluate_method(
        "CPOP",
        test_data,
        args.num_processors,
        lambda dag: cpop_schedule(dag, args.num_processors),
    )

    if llm_weights is not None:
        all_rows += evaluate_method(
            "LLM-Heuristic",
            test_data,
            args.num_processors,
            lambda dag: weighted_feature_schedule(dag, args.num_processors, llm_weights),
        )
    if bo_weights is not None:
        all_rows += evaluate_method(
            "LLM+BO-Heuristic",
            test_data,
            args.num_processors,
            lambda dag: weighted_feature_schedule(dag, args.num_processors, bo_weights),
    )

    if gnn_model is not None:
        all_rows += evaluate_method(
            "GNN-Imitation",
            test_data,
            args.num_processors,
            lambda dag: gnn_schedule(gnn_model, dag, args.num_processors),
        )

    if rl_model is not None:
        all_rows += evaluate_method(
            "Decima-like-RL",
            test_data,
            args.num_processors,
            lambda dag: gnn_schedule(rl_model, dag, args.num_processors),
        )

    #summarize(all_rows)
    #save_csv(all_rows, args.out)

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
            method_name="HEFT",
            model=args.openai_model,
        )

        print(f"\nDAG family: {family}")
        print(insights)


if __name__ == "__main__":
    main()
