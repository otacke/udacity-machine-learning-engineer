# Supervised Learning

## Supervised Learning Intro
* Supervised Learning: You know the answers and can tell the algorithm how well it did its job.
* Features => Labels, different combinations of features may be labeled differently
* Visual representations of features can give valuable insight before applying machine learning techniques

## Regression and Classification
* Continuous inputs and outputs
* regression originally meant "regressing to the mean"
* regression = minimizing the mean squared error (MSE) with a a conctant function or linear function or something more complex (higher polynomial)
* There ara many error/loss functions, MSE is just one of them

* matrix: X * w ~ y where X = matrix of 1, x^1, ... and w = coefficent vector and y = target vector
* We're looking for w = (X^T X)^{-1} X^T y

* errors come from... sensors/measuring, transcription, transfer, maliciously, merging data sets that don't match perfectly, unmodeled influence, ... => again: cross validation

* 'The goal of Machine Learning is "Generalization"'. Science!

## Regressions
* vocabulary: slope (Steigung) + intercept (Abschnitt)
* two algorithms for finding the smallest error: ordinary least squares and gradient descent
* error metrics: sum of squared errors (SSE) and R².
* multi-variate-regression = multi-dimensional

## More Regressions
* k nearest neighbor (KNN) = non paramatric regegression
  * find k nearest data points to an x and use their average y for prediction
  * this way, find predictions for every x in X
* kernel regression
  * weighted KNN using the distance of the values
