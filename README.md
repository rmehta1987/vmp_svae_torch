# vmp_svae_torch
This is an implementation of Variational Message Passing for Structured VAE in pyTorch 

Variational Message Passing for Structured VAE (Code for the ICLR 2018 paper by Wu Lin, Nicolas Hubacher and Mohammad Emtiyaz Khan)

The main file is gmmsvae and it is heavily commented to help understand the model
and the relationship calculations to the equations in the paper.

Execute code you must use train.py.  It currently uses MNIST and a fake Pinwheel dataset.  Tensorboard and TensorbardX can be used for 
logging and creating the graph respectively.  Tensorboard cannot handle all torch functions yet, so it fails in some cases to create the graph,
thus it's only use to create plots.  Also TensorboardX, you must disable assertion check of Just in time compiler otherwise it will fail.  
