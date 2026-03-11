## Dual Coordinate Descent SVM Reproduction (RCV1 / Hsieh et al. 2008)

This repository contains a reproduction of the paper:

> Cho-Jui Hsieh, Kai-Wei Chang, Chih-Jen Lin, S. Sathiya Keerthi, S. Sundararajan,  
> **“A Dual Coordinate Descent Method for Large-scale Linear SVM”**, ICML 2008.

The main work is in the notebook `svm_dcd_reproduction.ipynb`, which:

- Implements the **dual coordinate descent (DCD)** algorithm for **linear SVM** (L1- and L2-loss) as described in Algorithm 1 of the paper.
- Uses the **RCV1** text dataset (as in the paper) via `sklearn.datasets.fetch_rcv1`.
- Compares DCD to standard linear SVM baselines.
- Computes a **reference optimal solution** `w*` and plots **relative primal error** over time, similar to the convergence figures in the paper.

---

### Algorithm Overview

- **Problem**: Linear SVM in primal form  
  \[
    \min_w \frac12 \|w\|^2 + C \sum_i \xi(w;x_i,y_i)
  \]
  with L1-loss \(\xi = \max(1 - y_i w^\top x_i, 0)\) or L2-loss \(\xi = \max(1 - y_i w^\top x_i, 0)^2\).

- **Dual form**: Quadratic problem in dual variables \(\alpha_i\) with box constraints:
  \[
    \min_\alpha f(\alpha) = \frac12 \alpha^\top \bar{Q} \alpha - e^\top \alpha,\quad 0 \le \alpha_i \le U
  \]
  where \(\bar{Q} = Q + D\), \(Q_{ij} = y_i y_j x_i^\top x_j\), and \(D\) depends on the loss (L1 vs L2).

- **DCD idea**:
  - Maintain the **primal weight** \(w = \sum_j y_j \alpha_j x_j\).
  - At each inner iteration, pick one coordinate \(i\) (random permutation per epoch), and solve the **1‑D quadratic subproblem** in \(\alpha_i\) exactly:
    \[
      \alpha_i \leftarrow \text{clip}\left(\alpha_i - \frac{\nabla_i f(\alpha)}{\bar{Q}_{ii}},\; 0,\; U\right)
    \]
  - Update \(w\) cheaply using:
    \[
      w \leftarrow w + (\alpha_i^{\text{new}} - \alpha_i^{\text{old}}) y_i x_i
    \]
  - Use **sparse operations** and avoid materializing the kernel matrix.

- **Key properties**:
  - Each update is \(O(\bar{n})\), where \(\bar{n}\) is the average number of nonzeros per instance.
  - Proven **linear convergence** in terms of dual objective.
  - Works for both **L1‑SVM** and **L2‑SVM** (via different `D` and `U`).

In the notebook this is implemented as the `DCDSVM` class, with:

- Support for **L1** and **L2** loss (`loss="l1"` / `"l2"`).
+- A sparse inner loop (`xi_sparse.dot(w)`) and dense `w` update only when \(\alpha_i\) changes.
+- Optional logging of the **primal objective** during training.
+- A convergence driver that runs until the **relative error** \(|f_P(w) - f_P(w^*)| / |f_P(w^*)|\) reaches a target (e.g. 1%).

---

### How to Run the Notebook Locally

#### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/230131_midsem.git
cd 230131_midsem
```

Replace `<your-username>` with the actual GitHub owner if different.

#### 2. Create and activate a Python environment (recommended)

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
```

Or using `conda`:

```bash
conda create -n dcd_svm python=3.10 -y
conda activate dcd_svm
```

#### 3. Install dependencies

This project uses only standard scientific Python packages:

```bash
pip install numpy scipy scikit-learn matplotlib jupyter
```

#### 4. Start Jupyter and open the notebook

```bash
jupyter notebook
```

Then in the browser, open:

- `svm_dcd_reproduction.ipynb`

and run all cells from top to bottom.

> **Note**: Downloading and processing the full RCV1 dataset (677,399 instances) can take some time and memory. If your machine is limited, you can edit the notebook cell that sets `sample_size` and choose a smaller value (e.g. `sample_size = 80000`) for quicker experiments.

---

### Reproduced Results and Plots

The notebook produces:

- **Training time and test accuracy** for:
  - DCD L2‑SVM (`DCDSVM(loss="l2")`)
  - DCD L1‑SVM (`DCDSVM(loss="l1")`)
  - `LinearSVC` (squared hinge, no bias)
  - `SGDClassifier` (hinge)
- A **convergence plot** of **relative primal error vs. time** for DCD L2‑SVM, on a log scale, analogous to Figure 1 in the paper:
  - First, a long DCD run computes an approximate optimum `w*` (with small dual gap).
  - Then, a convergence loop runs DCD epoch by epoch, tracking
    \[
      \text{rel\\_err}(w) = \\frac{|f_P(w) - f_P(w^*)|}{|f_P(w^*)|}
    \]
    until it drops below a chosen tolerance (e.g. 1%).

These allow you to compare both **optimization speed** (time to a given relative error) and **predictive performance** (test accuracy) of DCD versus standard baselines.

---

### Notes and Possible Extensions

- The current reproduction focuses on **batch DCD** without shrinking; Algorithm 3 (shrinking, DCDL1‑S / DCDL2‑S) can be added as a follow‑up.
- The paper also compares against Pegasos, TRON, PCD, and SVMperf; reproducing all these baselines would require additional external solvers and is not included here.
