# Model Evaluation and Validation

## Intro: Model Evaluation and Validation
* Start with the data!
* statistics and model building/comparison
* "The ultimate goal of Machine Learning is to have data models that can learn and improve over time. In essence machine learning is making inferences on data from previous examples." (cmp. previous course part)

## Prerequisites
* programming experience (check!)
  * especially in Python (just basics - will learn as I go...)
* familiarity with statistics (check, I guess)
* curiosity (double check)
* interest in getting my hands on data related problems (check)
* git experience (check)

## Measures of Central Tendency
* just very basic terms of descriptive statistics: mode, mean, median, ...

## Variability of Data
* range, inter quartile range, outlier < Q_1 - 1.5 * IQR OR outlier > Q_3 + 1.5 * IQR, boxplots
* deviation from mean, standard deviation = sqrt(variance) = sqrt(sum(x-x_bar)/n)
* In standard deviation, 68% of the data lie within median+/- standard deviation
* In standard deviation, 95% of the data lie within median+/- 2 times standard deviation
* Bessel's Correction for approximating the standard deviation of the population of a sample: use n-1 for calculating variance and thus standard deviation (s instead of sigma)

## Numpy & Pandas Tutorials
* [Intro Docs for numpy](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html)
* [Intro Docs for pandas](http://pandas.pydata.org/pandas-docs/version/0.17.0/)

Pretty basic description of some functions from numpy and pandas.

## scikit-learn Tutorial
* Very rudimentary introduction to sklearn...

## Evaluation Metrics
* Accuracy here is described as the proportion of items classified or labeled correctly:
  * accuracy = number of correctly identified instances / all instances
  * .score() in *sklearn*
* Confusion matrix: actual class (pos/neg) VS predicted (pos/neg)
* Recall: True Positive / (True Positive + False Negative). Out of all the items that are truly positive, how many were correctly classified as positive. Or simply, how many positive items were 'recalled' from the dataset.
* Precision: True Positive / (True Positive + False Positive). Out of all the items labeled as positive, how many truly belong to the positive class.
* F1-Score: 2 * (precision * recall) / (precision + recall)

## Causes of Error
* Main causes of error
  * Bias due to a model being unable to represent the complexity of the underlying data / underfitting
  * variance due to a model being overly sensitive to the limited data it has been trained on / overfitting

## Nature of Data and Model Building
* Numeric Data (quantitative data), may be discrete or continuous
* Categorical Data (nominal data/ordinal data)
* Time Series, data collected repeatedly

## Training & Testing
* use independent datasets for training and testing for performance checks and checks for overfitting

## Cross Validation
* when splitting data into training and testing sets: use k datasets for k different learning experiments, afterwards average the k results = k-fold cross validation
* shuffling the data might be a good idea
* where's the central theme of this lesson?

## Representative Power of a Model
* if the number of features grows, the amount of data needed to generalize accurately grows exponentially

## Learning Curves and Model Complexity
* Learning Curves
  * training error and testing error converge (and are high): bias
  * training error and testing error have a gap: high variance
  * testing and training curves converge at similar values: ideal
