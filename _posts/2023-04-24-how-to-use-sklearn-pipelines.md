---
title: how to use sklearn pipelines
date: 2023-04-24 22:18:45 +02:00
categories: [AI]
tags: [machinelearning, sklearn]
---
Sklearn pipelines are a powerful way to streamline data preprocessing and model training in machine learning projects. Pipelines are particularly useful when you have to perform multiple preprocessing steps, such as scaling, encoding, and imputing missing values, before training a model. 

To create a pipeline, we use the `Pipeline` class from the `sklearn.pipeline` module. We pass a list of tuples, where each tuple contains a name for the step, and an instance of the transformer or estimator that we want to apply. 

For example, suppose we have a dataset with three types of features: ordinal, nominal, and numerical. We could create three separate pipelines, one for each type of feature, and then combine them using the `ColumnTransformer` class. Here's how we could do this:

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

ordinal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder())
])

nominal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(sparse=True, handle_unknown="ignore"))
])

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

preprocessing_pipeline = ColumnTransformer([
    ("nominal_preprocessor", nominal_pipeline, nominal),
    ("ordinal_preprocessor", ordinal_pipeline, ordinal),
    ("numerical_preprocessor", numerical_pipeline, numerical)
])
```

In this example, we first define three pipelines: `ordinal_pipeline`, `nominal_pipeline`, and `numerical_pipeline`, which apply different preprocessing steps to each type of feature. We then combine these pipelines into a single `preprocessing_pipeline` using `ColumnTransformer`. 

We can also add a model to the end of the pipeline, like this:

```python
from sklearn.linear_model import LogisticRegression
complete_pipeline = Pipeline([
    ("preprocessor", preprocessing_pipeline),
    ("estimator", LogisticRegression())
])

complete_pipeline.fit(train_features, train_label)
score = complete_pipeline.score(val_features, val_label)
predictions = complete_pipeline.predict(test_features)
```

Here, we define a `complete_pipeline` that first applies the `preprocessing_pipeline`, and then fits a logistic regression model. We can then call `fit` and `predict` on the `complete_pipeline` object just like we would with any other model.

One useful feature of pipelines is that we can use the `GridSearchCV` class to perform hyperparameter tuning across both the preprocessing steps and the model itself. Here's an example of how to do this:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "preprocessor__numerical_preprocessor__imputer__strategy": ["mean", "median"],
    "estimator__C": [0.1, 1, 10]
}

grid_search = GridSearchCV(complete_pipeline, param_grid=param_grid, cv=5)
grid_search.fit(train_features, train_label)
```

In this example, we define a `param_grid` dictionary with hyperparameters for both the preprocessing steps and the logistic regression estimator. We then pass this dictionary to `GridSearchCV`, along with the `complete_pipeline` object and the training data. `GridSearchCV` will then perform a cross-validated grid search over the hyperparameters, returning the best hyperparameters and model performance.

Overall, Sklearn pipelines are a powerful tool for streamlining machine learning workflows. By combining multiple preprocessing steps and models into a single pipeline, we can make our code more readable, reproducible, and scalable.
