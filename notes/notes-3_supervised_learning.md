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

## More Decision Trees
* Calculate the decision trees by hand "via a table"...
* Using Decision Trees is like using a linear boundary several times
* Entropy
  * measure of impurity
  * = sum_i{-p_i * log_2(p_i)} where p_i is the fraction of samples within a class
* Information Gain
  * = entropy(parent) - [weighted average] entropy(children) (weighted means the fraction that went to that node)
  * to be maximized
* prone to overfitting

## Neural Networks
* Perceptron
  * input = sum_i(w_i * x_i)
  * output = (input >= theta) ? 1 : 0
  * can represent AND, OR and NOT
* Perceptron Rule
  * use theta as an "additional parameter"
  * calculate delta_w = learning_rate * (y - y*) * input_value (where y* = actual output > 0; not theta, because we have theta as a parameter now)
  * w = w + delta w
  * do until done
* Gradient descent
  * a = input
  * Error(w) = 1/2 * sum_{x,y}( (y-a)² )
  * minimize the error
* sigmoid function (as a threshold)
  * \sigma(a) = \frac{1}{1+e^{-a}}
  * \sigma(a)' = \frac{1}{1+e^{-a}} (1 - \frac{1}{1+e^{-a}})
* Backpropagation

## Kernel Methods & SVMs
* The idea behind SVMs is to find a linear margin between different items within a sample.
* The distance between the items and the margin shall be as large as possible.
* SVMs only take into account the nearest items to the margin.
* Items are represented by a vector in a vector space, that are the foundation of the margin - hence support vector machine.

* Since not every sample can be split in a linear fashion, the problem can be transferred to higher dimensional vector spaces so linear splits are possible. Afterwards, the problem is projected to the original vector space - allowing non linear splits
* Computation costs of the "dimension transfer" are high, but there's the kernel trick that allows to avoid that. A kernel function may describe the split in a higher dimension, but look "nice" in lower dimensions - and computation between dimensions is not necessary.

## SVM
* Correct classification is the prime directive for a SVM
* Outliers can be tolerated
* parameters
  * kernel
  * C: penalty, controlling smooth vs. precise boundary (larger => more precise)
  * gamma

## Nonparametric Models
* Instance Based Learning
  * store the data, just lookup results
  * + no training, keep original values, it's simple
  * - prone to overfitting, no generalization, fix by k-nearest-neighbors with reasonable notion of distance (~similarity in general) and feasible notion of "mean/vote"
  * learning is quick, but querying is not (lazy)

## Bayesian Methods

### Naive Bayes
* Example: Cancer Test
  * Probability of having cancer: P(C) = 0.01 or P(!C) = 0.99
  * Probability of the test showing positive if you have cancer: P(pos|C) = 0.9
  * Probability of the test showing negative if you don't have cancer: P(neg|!C) = 0.9 or P(pos|!C) = 0.1
  * What't the probability P(pos) of actually having cancer if you get a positive result?
    * P(pos) = P(C|pos) / ( P(C|pos) + P(!C|pos) )
    * 1) calculate P(C|pos) = P(C) * P(pos|C) = 0.01 * 0.9 = 0.09
    * 2) calculate P(!C|pos) = P(!C) * P(pos|!C) = 0.99 * 0.1 = 0.099
    * 3) divide P(C|pos) by sum of P(C|pos) and P(!C|pos), e.g. normalizing the probability.
    * P(pos) = 1/12
* **Bayes in general: P(A|B) = P(A) * P(B|A) / P(B)**

### Bayesian Learning
* Best hypothesis h given the data D that we see -- or maximize P(h|D) for h in H
* intuitive approach:
  * ```for h in H { calculate P(h|D) }; output h = maxarg ( P(h|D) )```
  * not practical for large H
* lots of math stuff...
