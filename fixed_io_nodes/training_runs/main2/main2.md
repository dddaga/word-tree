* Dataset: Iris (Full dataset, loaded from sklearn)
* Loss Function: Binary Cross Entropy
* input was given to single input node having vectordim=4

Thought of trying with a smaller and simpler dataset to check if it converges. This was a combined experiment with `main3`, to check if vector dimension matters more or the number of nodes (keeping number of parameters the same)

Basically, this expt is having 100 nodes, each of vectordim=4, and `main3` has 400 nodes, each of vectordim=1

### Dataset details


* Code for loading dataset (replace in the top of the dataloader process function)

```python
from torch.utils.data import TensorDataset
from sklearn.datasets import load_iris

iris_data = load_iris()
X = iris_data.data  # Features: (150, 4) - sepal length, sepal width, petal length, petal width
y = iris_data.target  # Labels: (150,) - 0, 1, 2 for setosa, versicolor, virginica

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32).reshape(-1, 1, 4)
y_tensor = torch.tensor(y, dtype=torch.long)

# Create PyTorch Dataset
dataset = TensorDataset(X_tensor, y_tensor)
```

