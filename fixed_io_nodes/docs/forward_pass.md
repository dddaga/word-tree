# Forward pass of the current model

The model has 3 layers, *Input Adapter*, *Quantizer Layer*, and *GNN*

* <u>**Input Adapter**</u>: It is a simple MLP with ReLU activtions and a Tanh activation at the end to ensure output is between $-1$ and $1$

* <u>**Quantizer**</u>: The input to the GNN is supposed to have discrete values. So quantizer picks the nearest value to assign the input to, then return the index of that. 

<details>
<summary>For the math of quantizer, click here</summary>

For input value, $x$, closest discrete value $v$, and the corresponding index $i$, we have the following relations

$$
v = \cos{(\frac{2\pi}{N}i)}
\implies i = \frac{N}{2\pi}\cos^{-1}(v)
$$

if we know gradient of $i$ (through gradients of GNN), we get gradient of $v$, calculated using lookup table. But, $v$ is not directly connected to $x$, so what we instead do it as follows (for pytorch)

$$
v_1 = (v - x).detach() + x
$$
During forward pass, $v_1 = v$, which works normally. But in backward pass, as $(v-x)$ is out of the computation graph, the gradient is passed from $v_1$ to $x$ directly, and $v_1$ recieves gradients from $i$ as discussed above. 

</details>

* <u>**GNN**</u>: In each forward pass, we repeat the following a fixed (N) number of times.

1. If any input data is given to the network at the current time step, then inject the input to the input nodes, otherwise proceed to step 2
2. Obtain the radiation targets for each node in the network. 
3. Collect both radiation and direct targets together, then for each node, update its values (currently done sequentially, but can be optimized for parallel processing). 

After all the iterations, the activation strengths of the output nodes are returned as the output of the overall network. The maths of how updating values of a node works is described in [MATHS.md](./maths.md)