Our goal is to have an energy-efficient version of deep learning. The properties of this system are:
- It is compute-efficient.
- It does the same amount of information processing and gets the same output while using fewer compute cycles.
- It requires fewer parameters, so the threshold of the hardware required to run it decreases.

There are other colleaguges doing their own direction research 


Your goal here is to explore complex numbers and the rich tools available in complex algebra to enrich the field of deep learning. Specifically, we are proving whether we can replace multiplications with additions. 


Primarily, the famous Euler's identity, e^(iθ), when multiplying two complex numbers (let's say of the same magnitude or even different magnitude), we end of adding phases. 


First pickup this network fgsegnet_v2 Because it is fully convolutional and a small network to begin with, we will do ablation studies by converting this real-valued network to a complex network. 


A few things to build here: in the convolutions in the CNN operations where multiplication is happening, we would convert them to addition as the phase addition. After all those additions, you 
You take a cosine to simulate an activation function which does a real value filtering that only real value is propagated forward. (Is this a good idea? Does this break the face continuity or even the data structure that we have defined, its continuity and utility? )i
Also, we do experiment with certain things, like rotating numbers. When you do addition at a binary level on two numbers, it would auto-rotate when the value exceeds. This is beneficial to us because we allow the numbers to overflow, which gives us a native rotation by default.

We do a certain level of caching of costs because we are looking for efficiency. Changing multiplications to additions and then adding the cost might not be very effective. We will do some kind of quantization where, given an input value θ, we have a hyperparameter that quantizes it within the roots of unity and maps it to the closest root of unity. We get the cost value for that.

Or should we propagate the phase and magnitude in two different channels? 

Does it make sense to transform the image in a different phase rather than RGB, which is more phase native? 

Also, we do experiment with certain things, like rotating numbers. When you do addition at a binary level on two numbers, it would auto-rotate when the value exceeds. This is beneficial to us because we allow the numbers to overflow, which gives us a native rotation by default.  
  
We do a certain level of caching of costs because we are looking for efficiency. Changing multiplications to additions and then adding the cost might not be very effective. We will do some kind of quantization where, given an input value θ, we have a hyperparameter that quantizes it within the roots of unity and maps it to the closest root of unity. We get the cost value for that.  
  
For backprop, we can have a custom backprop with the mapping stored. For this particular value, this is the derivative. During the forward pass and backward pass of activation, we serve the precomputed values from a hashmap, and the degree of precision could be a hyperparameter.

Keep it very simple to begin with, maybe even from the choice of language to use. I will primarily test this on the 5060ti. You can write the scripts and push them there via SCP, and use tmux to manage those remote sessions. 

 You are okay to explore using custom CUDA kernels using PyTorch or even using JAX. 
 
 Not very simple. Let's pick up the network and let's try modifying one layer with this operation and see how it goes. What do you suggest? 




