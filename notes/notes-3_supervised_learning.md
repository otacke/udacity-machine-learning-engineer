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

## Decision Trees
* Terms of Classification
  * Instances = Input
  * Concept = Function that'll do the mapping to discrete values
  * Target Concept = Answer
  * Hypothesis Class = Set of all possible Concepts that we could choose
  * Sample = Training Set
  * Candidate = Concept that might be the target concept
  * Testing Set = will determine if the candidate is the target concept
* Decision Tree: What you would expect...
  # pick best attribute and ask a question
  # follow the path
  # repeat (until answer)
* Expressiveness
  * any (or) = linear VS parity (x-or, odd or even) = O(2^N)
* D3
  * [Wikipedia](https://de.wikipedia.org/wiki/ID3)
  * using entropy/gain to identify nodes to be chosen next
  * Bias: restriction bias (we limit the number of functions we're considering) VS preference bias (what type of functions from the hypothethis class do we prefer, e.g. those that split better at the top for ID3)
* For continuous values...
    * it can make sense to ask different questions about it deeper down in the tree
    * break criteria: everything classified correctly, no more attributes, no overfitting; Or pruning
    * Regression: average, variance, voting...
