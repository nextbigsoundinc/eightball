eightball
==========

A python machine learning classification toolbox.

Packages estimator object, imputator object, and training data into a model object. Includes a suite of tools to assist in performing common evaluation and model optimization routines (cross validation, grid search, feature reduction, etc.) and visualizing the results.

Installation
------------

Dependencies
~~~~~~~~~~~~
- scikit-learn (>=0.19.1)
- pandas (>=0.22.0)
- numpy (>=1.14.0)
- matplotlib (>=1.5.1)

Installation
~~~~~~~~~~~~
    
    pip install -U eightball

Main Features
-------------
- designed to work with pandas dataframes for easier data alignment
- pipelines
- evaluation
- parameter tuning w/ grid search
- auto parameter tuning
- feature reduction
- built-in plotting for feature importance, parameter tuning, and feature reduction