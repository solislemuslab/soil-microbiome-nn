#!/bin/bash

# set up the model:
# The model has 5 hidden layers, 1 input layer and one output layer
# 42 neurons for the input layer, 150 neurons for the hidden layers, 1 for the output layer
# The input to 1st hidden layer weights are associated with a hyperparameter that controls all weights from 1 input
# The rest weights are associated with a hyperparameter only differed by groups (layer & bias)
net-spec "$1".net 63 180 180 180 180 180 1 / ih0=2:0.5:0.5 bh0=5:0.5 h0h1=2:0.5 bh1=5:0.5 h1h2=2:0.5 bh2=5:0.5 h2h3=2:0.5 bh3=5:0.5 h3h4=2:0.5 bh4=5:0.5 h4o=x2:0.5 bo=100 
# Specify the model has binary outcomes
model-spec "$1".net binary
# Specify the model has 42 inputs, one output, and 2 possible value for the output
# specify the training and the testing set for the model
data-spec "$1".net 63 1 2 / "$1"_train.csv@1: . "$1"_test.csv@1: .
# Start with an intial HMC to get the weight chains to a reasonable starting position
# Fix all hyperparameter to 5 so that only the actual weight is changing
net-gen "$1".net fix 5
# use a leapfrog length of 50 and a step size of 0.15, this gives small rejection rate
mc-spec "$1".net repeat 10 sample-noise heatbath hybrid 50:10 0.1
# run the mcmc for one iteration to set the weights to a good start
net-mc "$1".net 1
# run the whole HMC, using leapfrog length of 10, window size 2, and step size 0.02
# This gives reasonable rejection rate (< 0.3 for most iterations)
mc-spec "$1".net repeat 10 sample-sigmas heatbath 0.95 hybrid 10:2 0.01 negate
# run 50,000 iterations
net-mc "$1".net 30000

# use the last 25,000 iterations for prediction
net-pred tmp "$1".net 25000: > family_result_"$1".txt

# trim the result file so that the predictions could be readable in R
sed -i 1,5d family_result_"$1".txt
sed -i "$(($(wc -l < family_result_"$1".txt)-3)),\$d" family_result_"$1".txt

# remove the network, we could always recreate it with the same script
rm "$1".net

