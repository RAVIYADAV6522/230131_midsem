## Reproducing “A Dual Coordinate Descent Method for Large-scale Linear SVM” (ICML 2008)

**Authors**: Cho-Jui Hsieh, Kai-Wei Chang, Chih-Jen Lin, S. Sathiya Keerthi, S. Sundararajan  
**Paper**: *A Dual Coordinate Descent Method for Large-scale Linear SVM* (ICML 2008)

### 1. Goal

The goal of this reproduction is to implement the paper’s **dual coordinate descent (DCD)** solver for **large-scale linear SVM** and evaluate it on the **RCV1** dataset. We aim to match the paper’s optimization perspective: **time to reach a target accuracy of optimization**, measured by the **relative primal objective error** compared to a high-accuracy reference solution.

All experiments and plots are produced by the notebook:

- `partB/task_1.1.ipynb`

Plots should be saved into:

- `partB/results/`

---

### 2. Background: linear SVM and the dual

Given training pairs \((x_i, y_i)\) where \(y_i \in \{-1, +1\}\), the primal linear SVM objective is:

\[
\min_{w} \;\; f_P(w) = \frac{1}{2}\|w\|^2 + C \sum_{i=1}^l \xi(w; x_i, y_i)
\]

The paper focuses on two common losses:

- **L1-loss SVM**: \(\xi = \max(1 - y_i w^\top x_i, 0)\)
- **L2-loss SVM**: \(\xi = \max(1 - y_i w^\top x_i, 0)^2\)

The corresponding dual (paper Eq. (4)) is:

\[
\min_\alpha \;\; f(\alpha) = \frac{1}{2}\alpha^\top \bar{Q}\alpha - e^\top \alpha
\quad \text{s.t.}\quad 0 \le \alpha_i \le U
\]

where \(\bar{Q} = Q + D\), \(Q_{ij} = y_i y_j x_i^\top x_j\), and:

- **L1-SVM**: \(U=C\), \(D_{ii}=0\)
- **L2-SVM**: \(U=\infty\), \(D_{ii}=1/(2C)\)

---

### 3. Method: Dual Coordinate Descent (DCD)

#### 3.1 Core idea (Algorithm 1)

DCD updates one coordinate \(\alpha_i\) at a time by solving the 1‑D subproblem exactly. The key linear‑SVM trick is to maintain:

\[
w = \sum_{j=1}^l y_j \alpha_j x_j
\]

so that the gradient component can be computed cheaply (paper Eq. (12)):

\[
G_i = y_i w^\top x_i - 1 + D_{ii}\alpha_i
\]

The update has closed form (paper Eq. (9)):

\[
\alpha_i \leftarrow \Pi_{[0,U]}\left(\alpha_i - \frac{G_i}{\bar{Q}_{ii}}\right),
\quad \bar{Q}_{ii} = x_i^\top x_i + D_{ii}
\]

and \(w\) is updated efficiently (paper Eq. (13)):

\[
w \leftarrow w + (\alpha_i^{new} - \alpha_i^{old}) y_i x_i
\]

The paper also recommends a **random permutation** of coordinates per outer iteration (Section 3.1), which we follow.

#### 3.2 Implementation details in our reproduction

In `task_1.1.ipynb`, the algorithm is implemented as `DCDSVM` with:

- L1/L2 configuration through \((U, D)\)
- Sparse inner loop: compute \(w^\top x_i\) as a **sparse dot** with the CSR row `X[i]`
- Dense conversion only when updating \(w\) (because \(w\) is dense)
- Optional objective logging for convergence curves

**No bias term** is used (matching the paper’s experiment note).

---

### 4. Dataset and experimental protocol

#### 4.1 Dataset: RCV1

We use `sklearn.datasets.fetch_rcv1()` which returns a sparse CSR matrix with high-dimensional text features. A single label column is used to create a binary task, with labels mapped from \(\{0,1\}\) to \(\{-1,+1\}\).

We follow a **stratified 80/20 train/test split**, consistent with the paper’s 4/5–1/5 split.

#### 4.2 What we measure (paper-style)

The paper measures optimization progress via **relative primal objective error** (paper Eq. (20)):

\[
\text{rel\_err}(w) = \frac{|f_P(w) - f_P(w^*)|}{|f_P(w^*)|}
\]

where \(w^*\) is an (approximate) optimum. In this reproduction:

1. We compute a **reference solution** \(w^*\) by running DCD for many epochs (tight training).
2. We run DCD epoch-by-epoch and track `rel_err` vs time.
3. We plot `rel_err` on a **log scale** vs time (similar to the paper’s Figure 1 curves).

---

### 5. Results (what to include when exporting to PDF)

From `task_1.1.ipynb`, export and place the following figures into `partB/results/`:

- **DCD L2 convergence plot**: log-scale `relative error vs time`
- (Optional) **DCD L1 vs L2**: objective / error curves
- (Optional) **Test accuracy vs time** comparisons

In the paper, DCD is notably competitive in time-to-accuracy on large sparse datasets such as RCV1; with the full dataset and sparse inner loop, DCD should show strong scaling behavior.

---

### 6. Discussion and limitations

- **Shrinking (Algorithm 3)**: The paper’s shrinking heuristic (DCDL1‑S / DCDL2‑S) can further accelerate training. This reproduction focuses on the core DCD updates; shrinking can be added as an extension.
- **Baselines**: The paper compares against Pegasos, TRON, PCD, and SVMperf with consistent stopping criteria. In this reproduction we use readily available scikit-learn baselines; reproducing *all* paper baselines exactly would require external solvers and careful matching of stopping conditions.
- **Stopping criterion**: We report convergence using the paper’s relative primal error metric with a reference optimum computed by a longer DCD run. A stricter reference can be obtained by increasing the reference epochs and/or implementing a duality-gap based stopping test.

---

### 7. How to run

See `README.md` (repo root) and `partB/requirements.txt`. The main notebook is `partB/task_1.1.ipynb`.

