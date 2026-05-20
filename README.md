PS C:\Users\Shiva Ahir\Desktop\rkhs_revised_baselines> python sample_dag_scheduler_eval.py --use-llm
Train DAGs: 28, Test DAGs: 12
Processors: 4

LLM proposed weights:
{
  "upward_rank": 1.5,
  "downward_rank": 1.0,
  "depth": 2.0,
  "fanout": 0.5,
  "indegree": -0.5,
  "avg_comm_out": 1.5,
  "avg_cost": -1.0
}

Bayesian Optimization completed.
Best training makespan: 156.04
BO-optimized weights:
{
  "upward_rank": 2.0,
  "downward_rank": -1.729156400002047,
  "depth": -2.0,
  "fanout": 1.2015047333944877,
  "indegree": -1.8520244420275773,
  "avg_comm_out": 0.14654034663227433,
  "avg_cost": 1.6553153410421868
}
[GNN imitation] epoch=016, loss=0.0630
[GNN imitation] epoch=032, loss=0.0586
[GNN imitation] epoch=048, loss=0.0551
[GNN imitation] epoch=064, loss=0.0680
[GNN imitation] epoch=080, loss=0.0430
[Decima-like RL] episode=016, makespan=106.00, reward=-106.00
[Decima-like RL] episode=032, makespan=281.00, reward=-281.00
[Decima-like RL] episode=048, makespan=140.00, reward=-140.00
[Decima-like RL] episode=064, makespan=267.00, reward=-267.00
[Decima-like RL] episode=080, makespan=88.00, reward=-88.00

=== Aggregate Results ===
Method               Avg Makespan    Avg SLR   Runtime ms
HEFT                       174.92      1.427        1.006
CPOP                       206.42      1.690        2.386
LLM-Heuristic              216.33      1.779        2.865
LLM+BO-Heuristic           178.58      1.459        2.674
GNN-Imitation              176.75      1.438        3.543
Decima-like-RL             205.00      1.688        3.369

=== Win Count vs HEFT ===
CPOP               wins=  0, ties=  2, total=12
LLM-Heuristic      wins=  1, ties=  0, total=12
LLM+BO-Heuristic   wins=  3, ties=  1, total=12
GNN-Imitation      wins=  3, ties=  5, total=12
Decima-like-RL     wins=  0, ties=  0, total=12

=== Example Per-DAG Rows ===
HEFT               family=MOTIF_JOIN   nodes=11  edges=10  makespan=71.00    slr=1.449
HEFT               family=ER           nodes=30  edges=25  makespan=106.00   slr=1.828
HEFT               family=ER           nodes=50  edges=118 makespan=264.00   slr=1.354
HEFT               family=ER           nodes=20  edges=16  makespan=76.00    slr=1.118
HEFT               family=ER           nodes=30  edges=59  makespan=153.00   slr=1.366
HEFT               family=BA           nodes=50  edges=85  makespan=209.00   slr=1.504
HEFT               family=WS           nodes=30  edges=44  makespan=151.00   slr=1.452
HEFT               family=BA           nodes=30  edges=45  makespan=260.00   slr=1.244
HEFT               family=WS           nodes=30  edges=40  makespan=126.00   slr=1.537
HEFT               family=BA           nodes=50  edges=82  makespan=228.00   slr=1.399
HEFT               family=ER           nodes=50  edges=107 makespan=199.00   slr=1.382
HEFT               family=ER           nodes=50  edges=107 makespan=256.00   slr=1.488

Saved detailed results to: sample_scheduler_results.csv

=== LLM Schedule Insights ===

DAG family: MOTIF_JOIN
### A. Bottleneck Summary
The bottleneck operations in the scheduled DAG are:
1. **Operation 6** (Processor 2): Duration of 33.0, high upward rank (59.0), and a fanout of 1. This operation has the longest duration, which contributes significantly to the overall makespan.
2. **Operation 2** (Processor 1): Duration of 22.0, upward rank of 57.5, and a fanout of 1. This operation also has a considerable duration and contributes to the overall latency.
3. **Operation 3** (Processor 0): Duration of 25.0, upward rank of 52.75, and a fanout of 1. Similar to the previous operations, it has a significant duration.
4. **Operation 4** (Processor 3): Duration of 28.0, upward rank of 47.25, and a fanout of 1. This operation is also a significant contributor to the makespan.

### B. Scheduling Insight
The current scheduling shows that several operations are sequentially dependent, leading to a high makespan of 71.0. The operations with the longest durations are bottlenecks, and their execution times are not overlapping effectively. Increasing parallelism could help reduce the makespan, but care must be taken to ensure that dependencies are respected.

### C. PPA Optimization Recommendation
1. **Increase Parallelism**: Consider re-evaluating the scheduling of operations to allow more parallel execution. This could involve rescheduling operations that are currently waiting for the completion of the bottleneck operations.
2. **Resource Reuse**: Analyze the potential for resource reuse among operations, especially those with similar resource requirements. This can help reduce area and power consumption.
3. **Pipelining**: Implement pipelining for operations that can be broken down into smaller stages. This can help improve throughput and reduce latency.
4. **Optimize High-Fanout Nodes**: Focus on optimizing nodes with high communication overhead, particularly those with higher average communication outputs. This can help reduce the overall communication latency.

### D. Next Action for HLS Designer
1. **Revisit the Scheduling**: Analyze the dependencies and explore rescheduling operations to maximize parallel execution. Consider using techniques like loop unrolling or operation fusion where applicable.
2. **Evaluate Resource Allocation**: Assess the resource requirements of each operation and identify opportunities for resource sharing to minimize area and power.
3. **Implement Pipelining**: Identify operations that can be pipelined and implement this in the HLS tool to improve performance.
4. **Profile Communication**: Use profiling tools to analyze communication patterns and optimize high-fanout nodes to reduce communication overhead.

By following these recommendations, the HLS designer can effectively reduce latency, area, and power consumption while improving the overall performance of the design.
PS C:\Users\Shiva Ahir\Desktop\rkhs_revised_baselines> 