# HNSW Experimental Framework (L2, Synthetic Clusters)

> **Executable name:** `HNSW`

This project is a **research and validation framework** for evaluating  
**HNSW (Hierarchical Navigable Small World graphs)** against **Exact KNN (L2)**  
using **synthetic, well-separated high-dimensional clusters**.

It is designed to analyze:
- Recall and accuracy
- Per-cluster classification behavior
- Normalized confusion matrices
- Sensitivity to HNSW parameters (`M`, `ef_construction`, `ef_search`)

---

## Features

- Exact KNN (L2) baseline
- Configurable HNSW index
- Well-separated Gaussian clusters in `dim=128`
- Two unit tests:
  - **UT1** — HNSW vs Exact KNN recall
  - **UT2** — Per-cluster precision + normalized confusion matrix
- Single-threaded or multi-threaded index build
- Simple and explicit command-line interface

---

## Build & Run

Run without arguments to see usage:
```bash
mkdir Release
cd Release
cmake -DCMAKE_BUILD_TYPE=Release ..
make
./HNSW

```

Run **UT1** (HNSW vs Exact KNN):

```
./HNSW --ut1
```

## Command-Line Arguments

### Index parameters

| Flag    | Meaning                | Default |
| ------- | ---------------------- | ------- |
| `--dim` | Vector dimension       | 128     |
| `--M`   | Max neighbors per node | 16      |
| `--efc` | `ef_construction`      | 200     |

### Search parameters

| Flag        | Meaning             | Default |
| ----------- | ------------------- | ------- |
| `--k`       | K in KNN            | 15      |
| `--efs`     | `ef_search`         | 80      |
| `--queries` | Queries per cluster | 30      |

### Cluster generation

| Flag            | Meaning                         | Default |
| --------------- | ------------------------------- | ------- |
| `--clusters`    | Number of clusters              | 6       |
| `--pts`         | Points per cluster              | 200     |
| `--sigma`       | Intra-cluster std-dev           | 0.004   |
| `--center-dist` | Min L2 distance between centers | 8.0     |
| `--seed`        | RNG seed                        | 42      |

### Execution

| Flag        | Meaning       | Default |
| ----------- | ------------- | ------- |
| `--threads` | Build threads | 1       |
| `--ut1`     | Run UT1       | off     |
| `--ut2`     | Run UT2       | off     |

------

## Synthetic Data Model

- Cluster centers are **random but enforced to be well-separated in L2**

- Each cluster sample:

  ```
  x = center + Normal(0, σ)
  ```

- This avoids overlap in high-dimensional space

- Designed to test **index quality**, not ambiguous data

------

## Unit Tests Explained

------

## UT1 — HNSW vs Exact KNN

**Goal:**
 Verify that approximate search closely matches exact L2 KNN.

**Metrics:**

- **Top-1 accuracy**
- **Recall@K**

**Recall@K definition:**

```
Recall@K = |Approx ∩ Exact| / K
```

**Pass condition:**

```
assert(avg_recall > 0.95f);
```

This validates **search correctness**, independent of clustering.

------

## UT2 — Per-Cluster Precision & Confusion Matrix

**Goal:**
 Evaluate whether ANN search preserves **cluster identity**.

**Procedure:**

1. Query generated near cluster `c`
2. Perform HNSW KNN search
3. Convert neighbors → cluster labels
4. Predict cluster via **majority vote**
5. Update confusion matrix

------

## Confusion Matrix

### Layout

```
Rows    → Predicted cluster
Columns → True cluster
```

### Normalization

Each column is normalized to **sum to 1.0**:

```
CM_norm[p][t] = CM[p][t] / Σ_p CM[p][t]
```

Interpretation:

- Each column = recall distribution for a true cluster
- Diagonal = per-cluster recall
- Off-diagonal = confusion / leakage

------

### Example

```
Normalized confusion matrix (rows = predicted, cols = true)

      T0    T1    T2
P0   0.98  0.01  0.00
P1   0.02  0.97  0.01
P2   0.00  0.02  0.99
```

------

## Recall Derived from Confusion Matrix

The recall printed in **UT2** is **micro-average recall**:

```
Recall = Σ(diagonal) / Σ(all entries)
```

This equals **Recall@1** when:

- One prediction per query
- Equal number of queries per cluster

Thus:

> **UT1 Recall@K and UT2 confusion-derived recall measure related but distinct aspects of ANN quality**

------

## HNSW Parameters — Intuition

### `M`

- Graph degree
- Larger → higher recall, more memory
- Typical range: `8–32`

### `ef_construction`

- Build-time search width
- Larger → better graph, slower build
- Typical range: `100–400`

### `ef_search`

- Query-time search width
- Larger → higher recall, slower queries
- Usually `5–10 × K`

------

## Notes & Expected Behavior

- UT2 precision may be **low even when UT1 recall is high**
- This is expected when `K` spans multiple clusters
- Increase:
  - `center_dist`
  - or `ef_search`
- Decrease:
  - `sigma`
     to improve cluster purity

------

## Performance Notes & Empirical Results

This section documents **real benchmark runs** on **Mac Pro M1** of the `HNSW` executable on a large synthetic dataset.
The goal is to illustrate **build-time scalability**, **search latency**, and **recall trade-offs**
under heavy load.

---

### Test Configuration

Command:
```bash
./HNSW --ut1 --pts 100000 --efc 100 --efs 500 --M 26 --threads {1|8}


[UT] HNSW vs Exact KNN (L2)
Starting single-threaded index build...
[TIME] Total index insert: 308.072 sec
Top-1 accuracy: 0.927778
Recall@15: 0.881112
[TIME] Avg search per query: 0.00273322 sec
[FAIL] Recall is too low: 0.881112



[UT] HNSW vs Exact KNN (L2)
Starting parallel index build with 8 threads...
[TIME] Total index insert: 43.9286 sec
Top-1 accuracy: 0.922222
Recall@15: 0.880371
[TIME] Avg search per query: 0.00272346 sec
[FAIL] Recall is too low: 0.880371

```

| Threads | Build Time (s) | Recall@15 | Top-1 | Avg Query Time |
| ------- | -------------- | --------- | ----- | -------------- |
| 8       | 43.93          | 0.880     | 0.922 | 2.72 ms        |
| 1       | 308.07         | 0.881     | 0.928 | 2.73 ms        |
