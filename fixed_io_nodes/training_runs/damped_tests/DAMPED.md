This file contains the experiments information performed after some modifications to the the existing idea of phase/mag, and applying damping to the phase activation. 

- firstly, we have phase/mag activation as $\Phi$ & $M$ respectively, and phase/mag weights as $\theta$ & $m$ respectively, and activation strength $a$
- We define each node to hold 2 complex valued vectors - Activation & weight. The angle of the complex vectors are given by the corresponding phase vector, and magnitude by the corresponding magnitude vector
- While the angle is taken directly to be the phase, the magnitude is stored as log-like, i.e. actual magnitude used during computation $= e^{M}$

#### Forward pass details

- firstly, gather all the incoming inputs (inputs are the activation vectors and activation strength) to the node (regardless of conduction/radidation), and also include the current node's activation vectors in it
- calculate the complex activation and weight vectors in cartesian from the polar form. 
- Calculate the weights to be given to each node based on its activation strength (using softmax)
- Take weighted superposition of all the complex activation vectors
  
  $\mathbf{z_{out}} = \sum w_i \mathbf{z_i}$

- Multiply the weighted superposition a complex valued "bias" that is calculated using current node's weight as follows

  $\mathbf{z_{out}^{'}} = \mathbf{z_{out}} \cdot (M e^\theta)$

- Now, calculate its magnitude and phase - this is our new mag/phase activation for the current node. (note that the above is a vector operation, so we get a vector output to store)

- Additionally, a phase damping can be applied here on the obtained phase activation vector 

    $\Phi^{'} = \alpha \Phi_{new} + (1-\alpha) \Phi_{old} $


#### Activation Strength

We have two methods to calculate activation strength. 

1)  $a =  \sum M_i cos(\Phi_i)$ - sum across the vector dimension 
2)  $a = \sqrt{\frac{1}{D}\sum M_i^2} $ - D = vector dimension

#### Random Noise

A random noise was added to activation vectors after the single iteration of the forward pass - more details in the experiments (unless mentioned, assume that this is not applied in the experiment)

#### Stochastic Radiation

Randomly radiate to other nodes at the start of training and slowly reduce this towards the end (unless mentioned, assume that this is not applied in the expt)

#### Dynamic Loss

Let activation of node $i$, on timestep $t$ be $a_i^t$, we define dynamic loss as 
$L_{dyn} = \sum_{i=1}^N \sum_{t=1}^{T-1} \|a_i^t - a_i^{t+1}\|_2$

Final loss will be a weighted combination of these two (unless mentioned - assume that this loss is not applied in training)