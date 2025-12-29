### Terminologies

* Phase values - The actual numertical value of phase, denoted by $\Phi$
* Mag Values - The actual numerical value of magnitude, denoted by $M$
* Phases - The indices for phase, integer in $[0, 255]$, denoted by $\theta$
* Mags - indices for magnitude,  integer in $[0, 255]$, denoted by $m$

$\newline$



<!-- ### Lookup Table -->

<!-- Stores the mapping from Phases/Mags to Phase Values/Mag Values. The stored values are defined as following: -$\newline$ -->

$$\begin{equation}
\Phi = \cos{(\frac{2\pi}{N}\theta)} 
\end{equation}$$


$$\begin{equation}
M = e^{\gamma  \sin{(\frac{2\pi}{N}(m-\frac{N}{2}))}}
\end{equation}$$

Consequently, we can define gradients to store as well, 

$$\begin{equation}
\frac{d\Phi}{d\theta}  = -\frac{2\pi}{N} \cdot \sin{(\frac{2\pi}{N}\theta)}
\end{equation}$$

$$\begin{equation}
\frac{dM}{dm} = M \cdot \gamma  \sin{(\frac{2\pi}{N}(m-\frac{N}{2}))} \cdot \cos{(\frac{2\pi}{N}(m-\frac{N}{2}))}
\end{equation}$$

As it is inefficient to always apply equation (1) and (2) for conversion everytime, the values are taken from lookup table. But then that makes the process non-differentiable. For that purpose, different functions are defined for lookup with custom autograd behaviour according to equations (3) and (4) in [*custom_functions.py*](../custom_functions.py)



Although $\theta$ and $m$ are integers, during training, they are kept as floats as PyTorch supports gradients only for float/double tensors. However, after doing gradient updates, we quantize them again before storing in the database for other workers to access.


### Vector store

Mags, Phases and an index vector is stored in the vector database. Index vector can be defined as following

$$
I = [\cos{(\frac{2\pi}{N}\theta)} ; \sin{(\frac{2\pi}{N}\theta)}]
$$

which means for a vector dimension of say, 64, the index vector would be of length 2*64=128, having the first half as cosine, and 2nd half as sine. 
During searching in HNSW, we use the negative of the sine, keeping the first half same. (The distance metric is COSINE). 
Suppose $\theta = [\theta_i]$, so under the COSINE metric, the formula simplifies as follows ($I_2$ is the query vector, $I_1$ is the indexed vector)
$$
I_1 \cdot I_2 = [\cos{(\frac{2\pi}{N}\theta)} ; \sin{(\frac{2\pi}{N}\theta)}] \cdot [\cos{(\frac{2\pi}{N}\theta)} ; -\sin{(\frac{2\pi}{N}\theta)}] 
\newline
= \sum_{i=1}^{D} \cos{(\frac{2\pi}{N}\theta_{i}^{(1)})} \cos{(\frac{2\pi}{N}\theta_{i}^{(2)})} - \sin{(\frac{2\pi}{N}\theta_{i}^{(1)})} \sin{(\frac{2\pi}{N}\theta_{i}^{(2)})} 
\newline
= \sum_{i=1}^{D} \cos(\frac{2\pi}{N}(\theta_i^{(1)}+\theta_i^{(2)})) 
$$



Also, 

$$
|I_1|^2 = \sum_{i=1}^D \cos^2+\sin^2 = D
$$

hence, 
$$
\begin{equation}
\frac{I_1 \cdot I_2 }{|I_1||I_2|} = \frac{1}{D} \sum_{i=1}^{D} \cos(\frac{2\pi}{N}(\theta_i^{(1)}+\theta_i^{(2)})) 
\end{equation}
$$

So the HNSW index will return the node having the highest value for equation (5)


### Activation Strength



$$
a = \Phi \cdot M 
\newline
= \sum_{i=1}^N M_i \cos{(\frac{2\pi}{N}\theta_i)}
$$

### Managing incoming inputs to a node

Say the phase/mag activations and activation strength of the current node are $\theta_0$, $m_0$ and $a_0$ respectively, and the phase/mag weights are $\theta_w$ & $m_w$ respectively. 
Assume that, for n incoming connections, the phase/mag activations and activation strengths are $(\theta_i, m_i, a_i)$ for $i=1,...,n$, then, the new phase/mag activations are calculated as follows:-

1) Firstly, we weigh all the nodes by their activation strength. This weight is calculated by applying a softmax, (weight denoted by $A_i$)

$$
A_i = \frac{e^{a_i}}{\sum_{j=0}^ne^{a_j}}
$$

2) The phase activation will be calculated as 

$$
\theta_0^{(new)} = \sum_{i=0}^n A_i*(\theta_i+\theta_w) 
$$

To ensure its appropriate range, $\theta_0^{(new)}$ = $\theta_0^{(new)}\%phase\_bins$

We do similarly for magnitude activation updates as well. 



$\theta_i$ - phase activation
$m_i$ - mag activation
$a_i$ - activation strength
$\theta_w$ - phase weight
$m_w$ - mag weight





A single node can be repsented as a complex number in the followiing format, 
$$
M \cdot e^{i \frac{2\pi}{N}\theta} = e^{i \frac{2\pi}{N}\theta+\gamma  \sin{(\frac{2\pi}{N}(m-\frac{N}{2}))}}
$$


For two complex numbers, multiplying them is akin to rotation of one vector by the phase of other, and scaling the magnitude

$$
A_1 e^{i\theta_1} \cdot A_2 e^{i\theta_2} = A_1A_2e^{i(\theta_1+\theta_2)}
$$


## Some hyperparameter decisions



<details>
<summary>Why is LR a much higher value than standard architecture?</summary>

Let's analyse the operation of "phase lookup" from lookup table. It can be described as below, 

$$
\Phi = \cos(\frac{2\pi}{N}\theta)
$$$$
\frac{d\Phi}{d\theta} = -\sin(\frac{2\pi}{N}\theta)\cdot\frac{2\pi}{N}
$$$$
\frac{dJ}{d\theta} = \frac{dJ}{d\Phi}\frac{d\Phi}{d\theta} = - \frac{dJ}{d\Phi} \cdot \sin(\frac{2\pi}{N}\theta)\cdot\frac{2\pi}{N}
$$

What's worth noting here is, $\frac{2\pi}{N} \approx 0.025$ for $N=256$, which makes the gradients quite small, so we would need atleast 40 gradient sums to make one discrete change, that is without accounting for $\frac{dJ}{d\Phi}$, and taking LR=1. 

So I figured it would be better to take LR much higher, hence making the updates more frequent.

</details>
