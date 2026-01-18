* Dataset: MNIST (Just a single sample)
* Loss Function:  Binary Cross Entropy
* Input was resized from (28, 28) to (14, 14) and each row of the image was fed to each input node (hence a total of 14 input nodes)
* Used Unquantized model 
* For inactive output nodes, it returned the default activation strength calculated from just the phase/mag weight. This allowed weights to be updated even when node was inactive


The only difference between this and expt 4, is that this expt used multiple workers instead of just one.


### Dataset details

* Code for loading dataset is present in `single_sample_main.py`

