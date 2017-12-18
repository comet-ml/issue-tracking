### Comet ML & GitHub tutorial

#### In this tutorial we will:
* create a new project in Comet ML
* link our project to GitHub
* train a model
* create a pull request

#### 1. create a new project in Comet ML

![new project](https://www.comet.ml/githubTutorial/new_projectcomet.png)
![new project](https://www.comet.ml/githubTutorial/new_projectcomet2.png)
![new project](https://www.comet.ml/githubTutorial/new_projectcomet3.png)


#### 2. link GitHub account to Comet ML project

![new project](https://www.comet.ml/githubTutorial/link1.png)
   * pick repository and branch

#### 3. train models with Comet ML

   * in training code include project name in Experiment
```python
    experiment = Experiment(api_key="YourApiKey", project_name="YourProjectName")
```

#### 4. create pull request for desired experiment

![new project](https://www.comet.ml/githubTutorial/pr1.png)
![new project](https://www.comet.ml/githubTutorial/pr2.png)
![new project](https://www.comet.ml/githubTutorial/pr3.png)
![new project](https://www.comet.ml/githubTutorial/pr4.png)

#### 5. deep link back to Comet ML from GitHub pull request
![new project](https://www.comet.ml/githubTutorial/focus.png)

#### done
