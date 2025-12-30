The main entry point is through `main.py`

## File descriptions


* `custom_functions.py` - functions which which need custom backprop


* `input_adapter.py` - Contains `LinearInputAdapter` class, used as an adapter b/w the input data and the GNN if needed.
* `gnn_model.py` - Defined the class for GNN model
* `quantization.py` - The quantizer layers b/w adpater or input and the GNN layer
* `full_model.py` - Contains the Adapter + Quantizer + GNN model

$\newline$

* `lookup_table.py` - Lookup values from stored indices to the real values
* `gradient_accumulator.py` - 
* `node.py` - defines Node, used for representing a single Node in the graph. Manages loading of weights, incoming activations and activation strength
* `nodestore.py` - The node store used by each individual worker. 
* `main.py` - Training functions for defining different workers and the training script
