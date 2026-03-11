# Dual Coordinate Descent SVM Reproduction (RCV1 / Hsieh et al. 2008)

This repository contains a reproduction of the paper:

> Cho-Jui Hsieh, Kai-Wei Chang, Chih-Jen Lin, S. Sathiya Keerthi, S. Sundararajan
> **“A Dual Coordinate Descent Method for Large-scale Linear SVM”**, ICML 2008.

The main work is in the notebook `svm_dcd_reproduction.ipynb`, which:

* Implements the **Dual Coordinate Descent (DCD)** algorithm for **linear SVM** (L1- and L2-loss) as described in Algorithm 1 of the paper.
* Uses the **RCV1 text dataset** via `sklearn.datasets.fetch_rcv1`.
* Compares DCD with standard **linear SVM baselines**.
* Computes a **reference optimal solution `w*`** and plots **relative primal error over time**, similar to the convergence plots shown in the paper.

---

# Algorithm Overview

## 1. Primal Problem

The **linear SVM optimization problem** in primal form is:

```
min_w   1/2 ||w||² + C Σ_i ξ(w; x_i, y_i)
```

where the loss term is defined as:

### L1-loss (hinge loss)

```
ξ = max(1 − y_i (wᵀx_i), 0)
```

### L2-loss (squared hinge loss)

```
ξ = max(1 − y_i (wᵀx_i), 0)²
```

---

## 2. Dual Problem

The corresponding **dual optimization problem** becomes:

```
min_α   f(α) = 1/2 αᵀ Q̄ α − eᵀ α
```

subject to:

```
0 ≤ α_i ≤ U
```

Where:

* `Q_ij = y_i y_j x_iᵀ x_j`
* `Q̄ = Q + D`
* `D` depends on the loss function (L1 or L2)

---

## 3. Dual Coordinate Descent (DCD)

The key idea of **Dual Coordinate Descent** is to optimize one dual variable `α_i` at a time while maintaining the **primal weight vector**:

```
w = Σ_j y_j α_j x_j
```

For each coordinate update:

```
α_i ← clip( α_i − ∇_i f(α) / Q̄_ii , 0 , U )
```

After updating `α_i`, the weight vector is updated efficiently:

```
w ← w + (α_i(new) − α_i(old)) y_i x_i
```

---

# Key Properties of the Algorithm

* Each coordinate update costs **O(n̄)**
  where `n̄` = average number of **non-zero features** per example.

* The algorithm achieves **linear convergence** in terms of the **dual objective**.

* The implementation avoids constructing the full **kernel matrix**, making it suitable for **large-scale datasets**.

* Works for both:

  * **L1-SVM**
  * **L2-SVM**

---

# Implementation Details

The notebook implements a class:

```
DCDSVM
```

with features including:

* Support for **L1-loss** and **L2-loss**

```
DCDSVM(loss="l1")
DCDSVM(loss="l2")
```

* Sparse inner loop computation:

```
xi_sparse.dot(w)
```

* Efficient weight update only when `α_i` changes

* Optional logging of the **primal objective** during training

* A convergence driver that tracks **relative primal error**

```
rel_err(w) = |f_P(w) − f_P(w*)| / |f_P(w*)|
```

---

# How to Run the Notebook Locally

## 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/230131_midsem.git
cd 230131_midsem
```

Replace `<your-username>` with the actual GitHub owner if different.

---

## 2. Create a Python Environment (Recommended)

Using **venv**:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Or using **conda**:

```bash
conda create -n dcd_svm python=3.10 -y
conda activate dcd_svm
```

---

## 3. Install Dependencies

```bash
pip install numpy scipy scikit-learn matplotlib jupyter
```

---

## 4. Start Jupyter Notebook

```bash
jupyter notebook
```

Then open:

```
svm_dcd_reproduction.ipynb
```

and run all cells.

---

# Dataset Notes

The notebook downloads the **RCV1 dataset** automatically using:

```
sklearn.datasets.fetch_rcv1
```

Dataset size:

```
677,399 instances
47,236 features
```

Because of the dataset size, initial loading may take time.

If your system has limited memory, modify:

```
sample_size = 80000
```

inside the notebook to run faster experiments.

---

# Reproduced Experiments

The notebook compares several models:

### Dual Coordinate Descent

* DCD L2-SVM

```
DCDSVM(loss="l2")
```

* DCD L1-SVM

```
DCDSVM(loss="l1")
```

### Baselines

* `LinearSVC` (squared hinge loss)

* `SGDClassifier` (hinge loss)

---

# Evaluation Metrics

The notebook reports:

### Training Time

Total optimization runtime.

### Test Accuracy

Classification accuracy on held-out data.

### Convergence Speed

Measured using **relative primal error**:

```
rel_err(w) = |f_P(w) − f_P(w*)| / |f_P(w*)|
```

A long DCD run first computes an approximate optimal solution `w*`.

Then training is repeated epoch-by-epoch while measuring how quickly the algorithm approaches the optimum.

---

# Convergence Plot

The notebook produces a plot of:

```
Relative Primal Error vs Time
```

on a **log scale**, similar to **Figure 1 in the original paper**.

This demonstrates the **optimization efficiency of DCD** for large-scale linear SVMs.

---

# Possible Extensions

This reproduction focuses on **basic batch DCD**.

Possible improvements include:

### Shrinking Heuristics

Implement Algorithm 3 from the paper:

* `DCDL1-S`
* `DCDL2-S`

### Additional Baselines

Compare with other solvers mentioned in the paper:

* Pegasos
* TRON
* Parallel Coordinate Descent (PCD)
* SVMperf

These require additional external implementations and are not included in this repository.

---

# Reference

Cho-Jui Hsieh, Kai-Wei Chang, Chih-Jen Lin,
S. Sathiya Keerthi, S. Sundararajan.

**A Dual Coordinate Descent Method for Large-scale Linear SVM**

International Conference on Machine Learning (ICML), 2008.
