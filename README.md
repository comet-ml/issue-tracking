<img src="https://comet.ml/images/logo_comet_light.png" alt="Drawing" style="width: 350px;"/>


## Documentation 

Full documentation and additional training examples are available on http://www.comet.ml/docs/


- **Install Comet.ml from PyPI:**

```sh
pip install comet_ml
```
Comet.ml python SDK is compatible with: __Python 2.7-3.6__.



## Getting started: 30 seconds to Comet.ml 

The core class of Comet.ml is an  __Experiment__, a specific run of a script that generated a result such as training a model
  on a single set of hyper parameters. An [`Experiment`](Experiment/#experiment). will automatically
   log scripts output (stdout/stderr), code, and command line arguments on __any__ script and for the supported libraries will also log
    hyper parameters, metrics and model configuration. 

Here is the `Experiment` object:

```python
from comet_ml import Experiment
experiment = Experiment(api_key="YOUR_API_KEY")

# Your code.
```

The `Experiment` object logs various parameters of your experiment to Comet.ml
```python
from comet_ml import Experiment
experiment = Experiment(api_key="YOUR_API_KEY")
batch_size = 4 # A hyperparameter used somewhere in the code.

experiment.log_parameter("batch_size", batch_size) 
```

By default your experiment will be added to the project `Uncategorized Experiments`. You can also log your experiment to a specific project.
```python
from comet_ml import Experiment

#if "my project name" does not already exist, it will be created.
experiment = Experiment(api_key="YOUR_API_KEY",
                        project_name="my project name")
batch_size = 4 

experiment.log_parameter("batch_size", batch_size) 
```


You can also log a custom list of hyperparameters to your experiment via a dictionary.
```python
from comet_ml import Experiment
experiment = Experiment(api_key="YOUR_API_KEY",
                        project_name="my project name",
                        auto_param_logging=False)
batch_size = 128
num_classes = 10
epochs = 20

params={
    "batch_size":batch_size,
    "epochs":epochs,
    "num_classes":num_classes}

experiment.log_multiple_params(params)
```

We all strive to be data driven and yet every day valuable experiments results are just lost and forgotten. Comet.ml provides
a dead simple way of fixing that. Works with any workflow, any ML task, any machine and any piece of code.

For a more in-depth tutorial about Comet.ml, you can check out:

- [Getting started with the Experiment class](Experiment/#experiment)

----------------


## Installation

- [Sign up (free) on comet.ml and obtain an API key](https://www.comet.ml)


