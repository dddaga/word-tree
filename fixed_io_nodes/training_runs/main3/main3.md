* Dataset: Iris (Full dataset, loaded from sklearn)
* Loss Function: Binary Cross Entropy
* input was given to 

This was a combined experiment with `main2`, to check if vector dimension matters more or the number of nodes (keeping number of parameters the same)

Basically, this expt is having 400 nodes, each of vectordim=1, and `main2` has 100 nodes, each of vectordim=4

### Dataset details


* Code for loading dataset (replace in the top of the dataloader process function)

```python
iris_data = load_iris()
X = iris_data.data  # Features: (150, 4) - sepal length, sepal width, petal length, petal width
y = iris_data.target  # Labels: (150,) - 0, 1, 2 for setosa, versicolor, virginica

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32).reshape(-1, 4, 1)
y_tensor = torch.tensor(y, dtype=torch.long)

# Create PyTorch Dataset
dataset = TensorDataset(X_tensor, y_tensor)
```

