## **This is an exact copy of main2, just ran using pytorch backend for storing model weights. So just testing with it right now**

* Dataset: Iris (Full dataset, loaded from sklearn)
* Loss Function: Binary Cross Entropy
* input was given to single input node having vectordim=4

Thought of trying with a smaller and simpler dataset to check if it converges. This was a combined experiment with `main3`, to check if vector dimension matters more or the number of nodes (keeping number of parameters the same)

Basically, this expt is having 100 nodes, each of vectordim=4, and `main3` has 400 nodes, each of vectordim=1

### Dataset details


* Code for loading dataset (replace in the top of the dataloader process function)
* **The difference between this dataloader and the one in main2/main3 is that this does normalization of input data + taking arccos**

```python
# Load Iris dataset from scikit-learn
iris_data = load_iris()
X = iris_data.data  # Features: (150, 4) - sepal length, sepal width, petal length, petal width
y = iris_data.target  # Labels: (150,) - 0, 1, 2 for setosa, versicolor, virginica
    
# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
max_X = X_tensor.max(dim=0).values
min_X = X_tensor.min(dim=0).values
X_tensor = (X_tensor - min_X) / (max_X - min_X)
X_tensor = torch.arccos(X_tensor).reshape(-1, 1, 4)

y_tensor = torch.tensor(y, dtype=torch.long)

# Create PyTorch Dataset
dataset = TensorDataset(X_tensor, y_tensor)

print(f"Loaded Iris dataset: {len(dataset)} samples, {X.shape[1]} features, {len(iris_data.target_names)} classes")
```

