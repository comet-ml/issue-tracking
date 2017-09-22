# comet-quickstart-guide

### view an example account at: https://www.comet.ml/view/Jon-Snow
### get your api key from: https://www.comet.ml/

#### 1) Installing Comet on your machine:
```
    pip3 install comet_ml
    pip  install comet_ml
```
#### 2) Import Comet in the top of your code
```
   from comet_ml import Experiment
   #other imports
```
#### 3) Create an experiment
```
   experiment = Experiment(api_key="YOUR-API-KEY", log_code=True)
```

#### 4) Extended usage:
+ report dataset hash:
```
    train_data = ....
    experiment.log_dataset_hash(train_data)
```
+ manual report parameters or metrics:
```
    hyper_params = {"learning_rate": 0.5, "steps": 100000, "batch_size": 50}
    experiment.log_multiple_params(hyper_params)

    some_param = ...
    experiment.log_parameter("param name", some_param)

    train_accuracy = ...
    experiment.log_metric("acc", train_accuracy)
    
```

#### Keras example: https://github.com/comet-ml/comet-keras-example
#### Tensorflow example: https://github.com/comet-ml/comet-tensorflow-example
#### Scikit example: https://github.com/comet-ml/comet-scikit-example

    
