# mlp-standard
This program is used to train an ordinary MLP (multilayer perceptron or feed-forward neural network) with customizable layers and a single output. It is written in C++ and intended to run on multicore CPUs. The program uses Qt library to parse input arguments and measure execution time. Current state of the project allows to run it out of the box using a built-in dataset of 2500 samples.

### Main function
The program is written in a relatively compact way as a single function optimized for perfomance and precision. There are no separate classes for layers, neurons, gradients etc. All stages of the algorithm are included into the main function with respective comments. The trained network is assumed to be fully connected and supports customization of the hidden layers. The parallelization is implemented by creating a pool of training/validation samples. The application is intended to run on modern 64-bit CPUs.

### Recursive accumulator class
This program supports very large training sets (millions of samples). Therefore, to avoid loss of precision during gradient calculation, the summation is implemented as a separate function conceptually similar to finding a sum of leaf elements when performing a bypass of a binary tree. Such approach reduces loss of precision when summing very long sequences of small numbers.

### Backpropagation
The backpropagation algorithm was implemented in a time-efficient manner according to the book of Christopher Bishop - Neural Networks in Pattern Recognition. Its main feature is that the derivative components are calculated simultaneously with a forward network pass at every iteration. The program supports two gradient descent techniques: Adam (adaptive moment estimation, https://arxiv.org/abs/1412.6980) and the gradient descent with momentum.

### How to run
In order to compile this program the Qt library needs to be installed. Use Qt Creator to build the project or alternatively use Linux command line: `qmake mlp-standard.pro && make`. Type `mlp-standard --help` to see the list of supported arguments. The program can be compiled and executed without any additional external data. At every iteration the application prints network weights, gradient vector and training/validation errors (the time overhead for this operation is about 0.9 seconds for 1000 iterations).
