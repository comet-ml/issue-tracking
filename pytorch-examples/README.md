# comet-pytorch-example

#### Using Comet.ml to track PyTorch experiments

The following code snippets shows how to use PyTorch with Comet.ML. Based on the tutorial from [Yunjey](https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/feedforward_neural_network/main.py) this code trains an RNN to detect hand writted digits from the MNIST dataset.

By initialzing the Experiment() object comet will log stdout and source code. To log hyper-parameters, metrics and visualization we add a few function calls such as experiment.log_metric() and experiment.log_multiple_params().
