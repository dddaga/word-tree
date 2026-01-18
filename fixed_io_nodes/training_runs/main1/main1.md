* Dataset: MNIST (full 50k training set)
* Loss Function: Binary Cross Entropy
* Input was resized from (28, 28) to (14, 14) and each row of the image was fed to each input node (hence a total of 14 input nodes)

Did this experiment to find out what might happen when training with a bigger dataset. Main observation I wanted to do was whether or not all the nodes in the graph get activated or not. 

Previously, I only did training with single sample or with a very small subset (1 sample per class), so not all nodes in the graph were getting activated. With a bigger dataset, hypothesis was that more nodes would get activated in the dataset, which was True. In fact, almost all the nodes got activated in the graph


### Dataset details


* Code for loading dataset (replace in the top of the dataloader process function)

```python
transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((14, 14)),
    transforms.Lambda(lambda x: x.squeeze()),
])
dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transformations)
```

