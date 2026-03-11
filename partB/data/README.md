## Dataset: RCV1 (Reuters Corpus Volume I)

This project uses the **RCV1** text dataset via scikit-learn:

```python
from sklearn.datasets import fetch_rcv1
data = fetch_rcv1()
```

- **How it is obtained**: `fetch_rcv1()` downloads/caches the dataset automatically using scikit-learn’s dataset utilities.
- **How it is used**: in `partB/task_1.1.ipynb`, we:
  - use `data.data` (sparse CSR feature matrix),
  - construct a binary task from one label column: `y = data.target[:, 0]`,
  - map labels from `{0, 1}` to `{-1, +1}` to match the paper’s formulation.

