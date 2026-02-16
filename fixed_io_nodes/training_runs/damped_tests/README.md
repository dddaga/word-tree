* First see [documentation for Damped](./DAMPED.md)


## Experiment 1

* Out of the 2 activation strength method described in [Damped](./DAMPED.md), this used the first type
* The phase damping was set to 0.5
* Result loss curve on training set - [image](./test1.jpg)

## Experiment 2

* Out of the 2 activation strengths described in [Damped](./DAMPED.md), this used the 2nd type (with just magnitude)
* phase damping was set to 0.2 
* Result - [image](./test2.jpg)

## Experiment 3

* Activation type first (same as expt 1)
* phase damping 0.5 (same as expt 1)
* Added a random noise of variance 0.01 to phase_activation vectors 
* Result - [image](./test3.jpg)


## Experiment 4

* Activation type first (same as expt 2)
* phase damping, dynamic loss, random noise - all disabled (set to 0)
* Added stochastic radiation - initially 80% of the radiation neighbours are selected randomly, and this is scaled down linearly to 0% for the first half of the training, after which the radiation neighbours are purely from vector search
* This had an output projection - hence GNN outputted 10 nodes, and then a projection from 10 to 3 numbers. 
* Epoch 16/16  train_acc=70.00%  val_acc=66.67%
* result - [image](./test4.png)

## Experiment 5

* same as expt 4, only change is activation strength calculation changed from type (2) to type (1) 
* result - [image](./test5.png)

## Experiment 6

* Same as expt 4 - but without the output projection
* train_acc=50.00%  val_acc=73.33%
* result - [image](./test6.png)

