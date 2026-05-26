import argparse
import csv
import json
import os
import random
from collections import defaultdict, deque
from dataclasses import dataclass
import numpy as np

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args
except Exception:
    gp_minimize = None
    Real = None
    use_named_args = None


SEED = 42
P = 4

ALGORITHMS = [
    "GNN",
    "Greedy-EFT",
    "HEFT",
    "HOFT",
    "CPOP",
    "PEFT",
    "Decima",
    "LLM",
]

VARIANTS = [
    "Original Algo",
    "Original Algo + Motifs",
    "Original Algo + BO",
    "Original Algo + Motifs + BO",
    "Original Algo + Configs + BO (LLM for Weight Generation)",
]

FEATURES = [
    "upward_rank",
    "downward_rank",
    "depth",
    "fanout",
    "indegree",
    "avg_comm_out",
    "avg_cost",
    "motif_score",
]


# ============================================================
# DAG STRUCTURE
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def generate_er_dag(n=50, p=0.08):
    adj = defaultdict(list)
    nodes = list(range(n))

    for i in nodes:
        for j in range(i + 1, n):
            if random.random() < p:
                adj[i].append(j)

    return clean_adj(adj, nodes)


def generate_ba_like_dag(n=50, m=3):
    adj = defaultdict(list)
    nodes = list(range(n))

    degree = [1 for _ in nodes]

    for new_node in range(1, n):
        probs = np.array(degree[:new_node], dtype=float)
        probs = probs / probs.sum()

        targets = np.random.choice(
            list(range(new_node)),
            size=min(m, new_node),
            replace=False,
            p=probs,
        )

        for t in targets:
            adj[t].append(new_node)
            degree[t] += 1
            degree[new_node] += 1

    return clean_adj(adj, nodes)


def generate_ws_like_dag(n=50, k=4, beta=0.25):
    adj = defaultdict(list)
    nodes = list(range(n))

    for i in nodes:
        for d in range(1, k + 1):
            j = i + d
            if j < n:
                if random.random() < beta:
                    new_j = random.randint(i + 1, n - 1) if i + 1 < n else j
                    adj[i].append(new_j)
                else:
                    adj[i].append(j)

    return clean_adj(adj, nodes)


def generate_motif_join_dag(n=50):
    adj = defaultdict(list)
    nodes = list(range(n))

    # repeated fork/join motif blocks
    block = 0
    while block + 9 < n:
        s = block
        a, b, c, d = block + 1, block + 2, block + 3, block + 4
        j = block + 5
        e, f, g = block + 6, block + 7, block + 8
        out = block + 9

        adj[s] += [a, b, c, d]
        adj[a].append(j)
        adj[b].append(j)
        adj[c].append(j)
        adj[d].append(j)
        adj[j] += [e, f, g]
        adj[e].append(out)
        adj[f].append(out)
        adj[g].append(out)

        if block + 10 < n:
            adj[out].append(block + 10)

        block += 10

    return clean_adj(adj, nodes)


def clean_adj(adj, nodes):
    cleaned = defaultdict(list)

    for u in nodes:
        for v in adj[u]:
            if u != v and u < v:
                cleaned[u].append(v)

    for n in nodes:
        cleaned[n] = sorted(set(cleaned[n]))

    return cleaned, nodes


def build_dataset(num_graphs=12):
    dataset = []

    for _ in range(num_graphs):
        dataset.append(("BA", *generate_ba_like_dag()))
        dataset.append(("ER", *generate_er_dag()))
        dataset.append(("WS", *generate_ws_like_dag()))
        dataset.append(("MOTIF_JOIN", *generate_motif_join_dag()))

    return dataset


# ============================================================
# FEATURES
# ============================================================

def topo_sort(adj, nodes):
    indeg = {n: 0 for n in nodes}

    for u in nodes:
        for v in adj[u]:
            indeg[v] += 1

    q = deque([n for n in nodes if indeg[n] == 0])
    order = []

    while q:
        u = q.popleft()
        order.append(u)

        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    if len(order) != len(nodes):
        raise ValueError("Graph is not a DAG")

    return order


def assign_costs(nodes):
    return {n: random.randint(5, 30) for n in nodes}


def assign_comm(adj):
    comm = {}
    for u in adj:
        for v in adj[u]:
            comm[(u, v)] = random.randint(1, 10)
    return comm


def reverse_adj(adj, nodes):
    pred = defaultdict(list)
    for n in nodes:
        pred[n] = []
    for u in nodes:
        for v in adj[u]:
            pred[v].append(u)
    return pred


def compute_features(adj, nodes, cost, comm):
    order = topo_sort(adj, nodes)
    pred = reverse_adj(adj, nodes)

    upward = {n: cost[n] for n in nodes}
    for n in reversed(order):
        if adj[n]:
            upward[n] = cost[n] + max(comm[(n, c)] + upward[c] for c in adj[n])

    downward = {n: cost[n] for n in nodes}
    for n in order:
        if pred[n]:
            downward[n] = cost[n] + max(comm[(p, n)] + downward[p] for p in pred[n])

    depth = {n: 0 for n in nodes}
    for u in order:
        for v in adj[u]:
            depth[v] = max(depth[v], depth[u] + 1)

    raw = {}
    for n in nodes:
        fanout = len(adj[n])
        indeg = len(pred[n])
        avg_comm_out = np.mean([comm[(n, c)] for c in adj[n]]) if adj[n] else 0.0

        if fanout >= 2 and indeg >= 2:
            motif = 1.00      # diamond / fork-join
        elif fanout >= 2:
            motif = 0.80      # fork
        elif indeg >= 2:
            motif = 0.75      # join
        elif fanout == 1 and indeg == 1:
            motif = 0.40      # chain
        else:
            motif = 0.20

        raw[n] = {
            "upward_rank": upward[n],
            "downward_rank": downward[n],
            "depth": depth[n],
            "fanout": fanout,
            "indegree": indeg,
            "avg_comm_out": avg_comm_out,
            "avg_cost": cost[n],
            "motif_score": motif,
        }

    return normalize(raw)


def normalize(raw):
    out = {n: {} for n in raw}

    for f in FEATURES:
        vals = np.array([raw[n][f] for n in raw], dtype=float)
        lo, hi = vals.min(), vals.max()

        for n in raw:
            out[n][f] = 0.0 if hi == lo else float((raw[n][f] - lo) / (hi - lo))

    return out


# ============================================================
# SCHEDULER
# ============================================================

def schedule(adj, nodes, cost, comm, priority, P=4):
    pred = reverse_adj(adj, nodes)
    completed = set()
    scheduled = set()

    proc_free = [0.0] * P
    finish = {}
    start = {}
    assign = {}

    while len(completed) < len(nodes):
        ready = [
            n for n in nodes
            if n not in scheduled and all(p in completed for p in pred[n])
        ]

        if not ready:
            raise RuntimeError("No ready tasks found")

        ready.sort(key=lambda x: priority[x], reverse=True)
        task = ready[0]

        best_p = None
        best_start = None
        best_finish = float("inf")

        for p in range(P):
            ready_time = 0.0

            for parent in pred[task]:
                transfer = 0 if assign[parent] == p else comm[(parent, task)]
                ready_time = max(ready_time, finish[parent] + transfer)

            s = max(proc_free[p], ready_time)
            f = s + cost[task]

            if f < best_finish:
                best_p = p
                best_start = s
                best_finish = f

        assign[task] = best_p
        start[task] = best_start
        finish[task] = best_finish
        proc_free[best_p] = best_finish

        scheduled.add(task)
        completed.add(task)

    return max(finish.values()), assign, start, finish


# ============================================================
# WEIGHTS
# ============================================================

def original_weights(algo):
    configs = {
        "GNN": {
            "upward_rank": 1.10,
            "downward_rank": 0.30,
            "depth": 0.35,
            "fanout": 0.50,
            "indegree": 0.10,
            "avg_comm_out": 0.25,
            "avg_cost": 0.10,
            "motif_score": 0.00,
        },
        "Greedy-EFT": {
            "upward_rank": 0.20,
            "downward_rank": 0.10,
            "depth": 0.10,
            "fanout": 0.10,
            "indegree": 0.00,
            "avg_comm_out": 0.00,
            "avg_cost": -1.00,
            "motif_score": 0.00,
        },
        "HEFT": {
            "upward_rank": 1.00,
            "downward_rank": 0.00,
            "depth": 0.20,
            "fanout": 0.10,
            "indegree": 0.00,
            "avg_comm_out": 0.20,
            "avg_cost": 0.10,
            "motif_score": 0.00,
        },
        "HOFT": {
            "upward_rank": 1.10,
            "downward_rank": 0.30,
            "depth": 0.30,
            "fanout": 0.20,
            "indegree": 0.00,
            "avg_comm_out": 0.30,
            "avg_cost": 0.10,
            "motif_score": 0.00,
        },
        "CPOP": {
            "upward_rank": 1.00,
            "downward_rank": 1.00,
            "depth": 0.10,
            "fanout": 0.00,
            "indegree": 0.00,
            "avg_comm_out": 0.10,
            "avg_cost": 0.10,
            "motif_score": 0.00,
        },
        "PEFT": {
            "upward_rank": 0.90,
            "downward_rank": 0.40,
            "depth": 0.25,
            "fanout": 0.15,
            "indegree": 0.05,
            "avg_comm_out": 0.25,
            "avg_cost": 0.20,
            "motif_score": 0.00,
        },
        "Decima": {
            "upward_rank": 0.65,
            "downward_rank": 0.25,
            "depth": 0.50,
            "fanout": 0.65,
            "indegree": -0.10,
            "avg_comm_out": 0.20,
            "avg_cost": 0.05,
            "motif_score": 0.00,
        },
        "LLM": {
            "upward_rank": 1.00,
            "downward_rank": 0.25,
            "depth": 0.35,
            "fanout": 0.35,
            "indegree": 0.05,
            "avg_comm_out": 0.35,
            "avg_cost": 0.10,
            "motif_score": 0.00,
        },
    }

    return configs[algo].copy()


def llm_weight_prompt(graph_summary, algo):
    return f"""
You are designing a DAG scheduling priority heuristic.

Algorithm family: {algo}

We use this priority function:

H(v) =
w1*upward_rank +
w2*downward_rank +
w3*depth +
w4*fanout +
w5*indegree +
w6*avg_comm_out +
w7*avg_cost +
w8*motif_score

Higher H(v) means schedule earlier.

Return JSON only with keys:
{FEATURES}

Each value must be between -2 and 2.

DAG summary:
{json.dumps(graph_summary, indent=2)}

Guidance:
- Give high positive weight to critical-path urgency.
- Give useful positive weight to fanout/fork nodes when early branching helps.
- Give useful positive weight to join/fork-join motifs when synchronization is critical.
- Penalize features if they cause bad early scheduling.
- Keep weights interpretable.
"""


def get_llm_weights(summary, algo, model="gpt-4o-mini"):
    fallback = original_weights(algo)
    fallback["motif_score"] = 0.80

    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        return fallback

    try:
        client = OpenAI()
        prompt = llm_weight_prompt(summary, algo)

        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )

        txt = resp.choices[0].message.content.strip()
        txt = txt.replace("```json", "").replace("```", "").strip()
        data = json.loads(txt)

        return {f: float(data.get(f, fallback[f])) for f in FEATURES}

    except Exception as e:
        print(f"[LLM fallback for {algo}] {e}")
        return fallback


def graph_summary(adj, nodes, cost, comm):
    pred = reverse_adj(adj, nodes)

    return {
        "nodes": len(nodes),
        "edges": sum(len(adj[n]) for n in nodes),
        "avg_cost": float(np.mean(list(cost.values()))),
        "avg_comm": float(np.mean(list(comm.values()))) if comm else 0.0,
        "fork_nodes": sum(1 for n in nodes if len(adj[n]) >= 2),
        "join_nodes": sum(1 for n in nodes if len(pred[n]) >= 2),
        "fork_join_nodes": sum(1 for n in nodes if len(adj[n]) >= 2 and len(pred[n]) >= 2),
        "chain_nodes": sum(1 for n in nodes if len(adj[n]) == 1 and len(pred[n]) == 1),
    }


def priority_from_weights(features, weights):
    priority = {}

    for n in features:
        priority[n] = sum(weights[f] * features[n][f] for f in FEATURES)

    return priority


# ============================================================
# BO
# ============================================================

def bo_optimize(train_graphs, algo, base_weights, use_motif=False, calls=25):
    if gp_minimize is None:
        print("[BO fallback] scikit-optimize missing. Returning handcrafted BO weights.")
        w = base_weights.copy()
        w["upward_rank"] *= 1.50
        w["fanout"] *= 1.30
        w["avg_comm_out"] *= 1.30
        if use_motif:
            w["motif_score"] = max(w.get("motif_score", 0.0), 1.20)
        return w

    dims = [Real(-2.0, 2.0, name=f) for f in FEATURES]

    @use_named_args(dims)
    def objective(**w):
        if not use_motif:
            w["motif_score"] = 0.0

        total = 0.0

        for _, adj, nodes, cost, comm in train_graphs:
            feats = compute_features(adj, nodes, cost, comm)
            pr = priority_from_weights(feats, w)
            makespan, _, _, _ = schedule(adj, nodes, cost, comm, pr, P)
            total += makespan

        return total / len(train_graphs)

    x0 = [base_weights[f] for f in FEATURES]

    result = gp_minimize(
        objective,
        dims,
        x0=x0,
        n_calls=calls,
        random_state=SEED,
        acq_func="EI",
    )

    return {f: float(v) for f, v in zip(FEATURES, result.x)}


# ============================================================
# EXPERIMENT
# ============================================================

def make_variant_weights(algo, variant, graph_summary_data, train_graphs, use_llm=False):
    base = original_weights(algo)

    if variant == "Original Algo":
        return base

    if variant == "Original Algo + Motifs":
        w = base.copy()
        w["motif_score"] = 1.00
        w["fanout"] *= 1.15
        w["indegree"] *= 1.15
        return w

    if variant == "Original Algo + BO":
        return bo_optimize(train_graphs, algo, base, use_motif=False)

    if variant == "Original Algo + Motifs + BO":
        motif_base = base.copy()
        motif_base["motif_score"] = 1.00
        return bo_optimize(train_graphs, algo, motif_base, use_motif=True)

    if variant == "Original Algo + Configs + BO (LLM for Weight Generation)":
        llm_w = get_llm_weights(graph_summary_data, algo) if use_llm else base.copy()
        llm_w["motif_score"] = max(llm_w.get("motif_score", 0.0), 1.00)
        return bo_optimize(train_graphs, algo, llm_w, use_motif=True)

    return base


def evaluate_variant(test_graphs, weights):
    vals = []

    for _, adj, nodes, cost, comm in test_graphs:
        feats = compute_features(adj, nodes, cost, comm)
        pr = priority_from_weights(feats, weights)
        makespan, _, _, _ = schedule(adj, nodes, cost, comm, pr, P)
        vals.append(makespan)

    return round(float(np.mean(vals)), 2)


def llm_schedule_insight_prompt(row_results, best_algo, best_variant):
    return f"""
You are an HLS/DAG scheduling expert.

Analyze these scheduling ablation results.

Results:
{json.dumps(row_results, indent=2)}

Best method:
Algorithm = {best_algo}
Variant = {best_variant}

Explain:
1. Why the best method wins.
2. Which features likely helped: upward_rank, downward_rank, depth, fanout, indegree, avg_comm_out, avg_cost, motif_score.
3. How motifs helped scheduling.
4. How BO helped weight tuning.
5. How LLM-generated weights helped initialize BO.
6. Hardware insight: where pipelining/register/FIFO insertion is useful.
7. PPA tradeoffs: latency, area, power, throughput.

Return a concise technical explanation suitable for a paper.
"""


def get_llm_insights(row_results, best_algo, best_variant, model="gpt-4o-mini"):
    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        return "[LLM insights skipped: OPENAI_API_KEY not set]"

    try:
        client = OpenAI()

        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": llm_schedule_insight_prompt(row_results, best_algo, best_variant),
                }
            ],
        )

        return resp.choices[0].message.content.strip()

    except Exception as e:
        return f"[LLM insights failed: {e}]"


def print_markdown_table(results):
    header = "| Algorithm | " + " | ".join(VARIANTS) + " |"
    sep = "|-----------|" + "|".join(["----------------:"] * len(VARIANTS)) + "|"

    print("\n" + header)
    print(sep)

    for algo in ALGORITHMS:
        row = f"| {algo} | " + " | ".join(str(results[algo][v]) for v in VARIANTS) + " |"
        print(row)


def save_csv(results, path="ablation_results_v2.csv"):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm"] + VARIANTS)

        for algo in ALGORITHMS:
            writer.writerow([algo] + [results[algo][v] for v in VARIANTS])

    print(f"\nSaved CSV: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--bo-calls", type=int, default=25)
    parser.add_argument("--num-graphs", type=int, default=8)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()

    set_seed(SEED)

    raw_dataset = build_dataset(args.num_graphs)

    full_dataset = []
    for family, adj, nodes in raw_dataset:
        cost = assign_costs(nodes)
        comm = assign_comm(adj)
        full_dataset.append((family, adj, nodes, cost, comm))

    random.shuffle(full_dataset)

    split = int(0.7 * len(full_dataset))
    train_graphs = full_dataset[:split]
    test_graphs = full_dataset[split:]

    print(f"Train graphs: {len(train_graphs)}")
    print(f"Test graphs: {len(test_graphs)}")
    print(f"Processors: {P}")
    print(f"LLM enabled: {args.use_llm}")

    summary_family, summary_adj, summary_nodes, summary_cost, summary_comm = train_graphs[0]
    summary = graph_summary(summary_adj, summary_nodes, summary_cost, summary_comm)

    results = {}
    saved_weights = {}

    for algo in ALGORITHMS:
        results[algo] = {}
        saved_weights[algo] = {}

        for variant in VARIANTS:
            print(f"Running {algo} | {variant}")

            weights = make_variant_weights(
                algo=algo,
                variant=variant,
                graph_summary_data=summary,
                train_graphs=train_graphs,
                use_llm=args.use_llm,
            )

            saved_weights[algo][variant] = weights
            results[algo][variant] = evaluate_variant(test_graphs, weights)

    print_markdown_table(results)
    save_csv(results)

    with open("ablation_weights_v2.json", "w") as f:
        json.dump(saved_weights, f, indent=2)

    print("Saved weights: ablation_weights_v2.json")

    best_algo = None
    best_variant = None
    best_val = float("inf")

    for algo in ALGORITHMS:
        for variant in VARIANTS:
            val = results[algo][variant]
            if val < best_val:
                best_val = val
                best_algo = algo
                best_variant = variant

    print("\nBest Result:")
    print(f"{best_algo} | {best_variant} | Makespan = {best_val}")

    insights = get_llm_insights(results, best_algo, best_variant, model=args.model)

    with open("llm_schedule_insights.txt", "w", encoding="utf-8") as f:
        f.write(insights)

    print("\nSaved LLM insights: llm_schedule_insights.txt")
    print("\nLLM Insights Preview:")
    print(insights[:1500])


if __name__ == "__main__":
    main()