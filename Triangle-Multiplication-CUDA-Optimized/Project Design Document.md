
# CUDA Optimized Triangle Counting & Matrix Multiplication

# Introduction

Triangle counting is a fundamental graph analysis problem with applications spanning social network analysis, fraud detection, community discovery, and biological network modeling. A triangle in a graph is formed when three vertices are mutually connected, representing the simplest form of clustering where "friends of friends are also friends”.

Use cases: In social networks, triangle density indicates the strength of community bonds and can help identify tightly-knit groups. In financial networks, unusual triangle patterns may signal collusive behavior or fraud rings, where multiple entities coordinate suspicious activities. 

# Bottlenecks

While conceptually straightforward: count all sets of three mutually connected nodes triangle, but counting becomes computationally challenging at scale. A graph with n vertices has O(n³) potential triangles to check, making naive approaches infeasible for modern networks containing millions or billions of nodes. 

# Technical Breakdown 

## Naive Way:

**Number of Triangles** = trace(A³) / 6

**How It Works**: Given the adjacency matrix A of an undirected graph, A³ counts all paths of length 3 that start and end at vertex i. Each triangle contributes 6 such paths (2 directions × 3 starting vertices), so summing the diagonal (trace) and dividing by 6 gives the exact triangle count.

**Pseudocode**: 

```
def naive_triangle_count(A, n): 
	A² = matrix_multiply(A, A) 
	A³ = matrix_multiply(A², A) 
	trace = sum of A³[i][i] for all i 
	return trace / 6
```

**Complexity:**

- Time: O(n³)  - two dense matrix multiplications

## CUDA-Optimized Approach: (L × L) .* L

**Idea**: Instead of computing the full A³, we use the lower triangular matrix L, where L[i][j] = 1 only if i > j and an edge exists. The formula becomes:

D = (L × L) .* L Triangle Count = sum of all entries in D

[Approach: https://sites.cs.ucsb.edu/~gilbert/cs219/cs219Spr2018/Notes/WolfEtAlTriangles.pdf]

  **Algorithm Steps:**

1. **Reorder vertices** by decreasing degree (improves load balancing)
2. **Build L in CSR format** — sparse representation storing only edges where row > col
3. **Compute C = L × L**: each element of C at i, j counts wedges (2-hop paths) from i to j through intermediate vertex k, where i > k > j
4.  Keep only wedges where endpoints are connected (**D = C.L**)
5. **Sum entries of D** gives the total triangle count

  **Pseudocode**
  
  ```
1: for each row (vertex) v ∈ L do
2:     Create a hashmap H, and insert columns L(v) into H.
3:     for each nonzero column (vertex) u ∈ L(v) do
4:          for each nonzero column (vertex) y ∈ L(u) do
5:               Query y in H
  ```

Each GPU thread processes one vertex i, iterating through its neighbors in parallel. The hash set lookup (or binary search) for the masking step is performed in shared memory for faster access. Atomic operations aggregate the final count.
# Comparision 

We will be comparing our version of the CUDA optimization algorithm against the naive way of doing it, on multiple threads (128, 512, 1024) to see the performance optimization graphed against it. 

We will also be comparing our results against [KokosKernels](https://kokkos.org/kokkos-kernels/docs/) Graph Algorithm (and its own triangle counting algorithm. More specifically, we will try to write our own API that can perform  X% of KokoKernel’s output on [datasets](https://sparse.tamu.edu/).

*Optimisation and benchmark percentages not guaranteed yet, to be decided*

### References

1.  [https://www.cs.cmu.edu/~15750/notes/lec1.pdf](https://www.cs.cmu.edu/~15750/notes/lec1.pdf)
    
2. [https://sites.cs.ucsb.edu/~gilbert/cs219/cs219Spr2018/Notes/WolfEtAlTriangles.pdf](https://sites.cs.ucsb.edu/~gilbert/cs219/cs219Spr2018/Notes/WolfEtAlTriangles.pdf)
    
3. [https://kokkos.org/kokkos-kernels/docs/](https://kokkos.org/kokkos-kernels/docs/