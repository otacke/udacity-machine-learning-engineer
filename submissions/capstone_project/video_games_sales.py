# coding: utf-8

# # Predicting Video Game Sales
# First of all, we need to import some libraries.

# Import libraries necessary for this project
import numpy as np
import pandas as pd
import scipy as sp

import visuals as vs # External file, check re-engineering necessity
import matplotlib.pyplot as plt

# Allows the use of display() for DataFrames
from IPython.display import display

# make pretty
plt.style.use('ggplot')

# Pretty display for notebooks
get_ipython().magic(u'matplotlib inline')


# # Data Schema
#
# ## Dataset 1 (Video_Games_Sales_as_at_22_Dec_2016.csv)
# This is my main dataset. It will be used to predict video game sales.
#
# | title           | description                                                   | data type |
# |-----------------|---------------------------------------------------------------|-----------|
# | Name            | Name of the game                                              | String    |
# | Platform        | Platform                                                      | String    |
# | Year_of_Release | Year of release                                               | Numeric   |
# | Genre           | Genre                                                         | String    |
# | Publisher       | Publisher                                                     | String    |
# | NA_Sales        | Game sales in North America (in millions of units)            | Numeric   |
# | EU_Sales        | Game sales in the European Union (in millions of units)       | Numeric   |
# | JP_Sales        | Game sales in Japan (in millions of units)                    | Numeric   |
# | Other_Sales     | Game sales in the rest of the world (in millions of units)    | Numeric   |
# | Global_Sales    | Total sales in the world (in millions of units)               | Numeric   |
# | Critic_Score    | Aggregate score compiled by Metacritic staff                  | Numeric   |
# | Critic_Count    | The number of critics used in coming up with the Critic_score | Numeric   |
# | User_Score      | Score by Metacritic's subscribers                             | Numeric   |
# | User_Count      | Number of users who gave the user_score                       | Numeric   |
# | Developer       | Party responsible for creating the game                       | String    |
# | Rating          | The [ESRB](https://www.esrb.org/) ratings                     | String    |
#

# We can remove several colums in dataset2 that we don't need:
# * **ID** which is just an ID that's not used in dataset 1
# * **score_phrase**, because it redundant to _score_ and less precise
# * **url**, because that's just the origin of the data
# The next step will be to load the datasets.

# Load dataset #1
try:
    dataset1 = pd.read_csv("data/Video_Games_Sales_as_at_22_Dec_2016.csv")
    print "Dataset #1 has {} samples with {} features.".format(*dataset1.shape)
except:
    print "Dataset #1 could not be loaded. Is the dataset missing?"

print('\n')

# Display a description of the datasets
display(dataset1.dtypes)
display(dataset1.head())
display(dataset1.tail())
display(dataset1.describe())
print('\n')

# Sanitization

# beautify strings
for i in range(0, len(dataset1.axes[1])):
    if (dataset1.ix[:,i].dtype == object):
        dataset1.ix[:,i] = dataset1.ix[:,i].str.strip()

# tbd to NaN
dataset1 = dataset1.replace('tbd', float('NaN'))

# Remove rows with empty values, we want full information
for column in dataset1.axes[1]:
    dataset1 = dataset1[dataset1[column].notnull()]

# correct data types
dataset1['User_Score'] = dataset1['User_Score'].apply(pd.to_numeric)

# dataset1 is now clean, make a copy for possibe use later
clean_dataset1 = dataset1.copy()

# display some infos
display(clean_dataset1.head())
print ('REMAINING ROWS IN DATASET #1: {}'.format(*clean_dataset1.shape))
print ('Dataset #1 has {} unique game name values'.format(clean_dataset1['Name'].unique().size))
print ('Dataset #1 has {} unique platforms values'.format(clean_dataset1['Platform'].unique().size))
print ('Dataset #1 has games from {} unique years'.format(clean_dataset1['Year_of_Release'].unique().size))
print ('Dataset #1 has {} unique genre values'.format(clean_dataset1['Genre'].unique().size))
print ('Dataset #1 has {} unique publisher values'.format(clean_dataset1['Publisher'].unique().size))
print ('Dataset #1 has {} unique developer values'.format(clean_dataset1['Developer'].unique().size))
print ('Dataset #1 has {} unique rating values'.format(clean_dataset1['Rating'].unique().size))

# Feature Generation and further harmonizing data
# make a short dataset that will contain all rows considered for further analysis
short_dataset1 = clean_dataset1.copy()

# => too many unique values per feature for the dataset size, so features should be removed
short_dataset1 = short_dataset1.drop(['Name', 'Publisher', 'Developer'], axis=1)

"""
# create Boolean columns for each remaining nominal variable
for platform in short_dataset1['Platform'].unique():
    short_dataset1['Platform_' + platform] = short_dataset1['Platform'].map(lambda x: True if x == platform else False)

for genre in short_dataset1['Genre'].unique():
    short_dataset1['Genre_' + genre] = short_dataset1['Genre'].map(lambda x: True if x == genre else False)

for rating in short_dataset1['Rating'].unique():
    short_dataset1['Rating_' + rating] = short_dataset1['Rating'].map(lambda x: True if x == rating else False)

# Platform manufacturer based on the platform
man_microsoft = ['PC','X360','XB','XOne']
man_nintendo  = ['3DS','DS','GBA','GC','N64','Wii','WiiU']
man_sega      = ['DC']
man_sony      = ['PS','PS2','PS3','PS4' ,'PSP','PSV']

short_dataset1['Platform_Manufacturer'] = short_dataset1['Platform'].map(lambda x:
    'Microsoft' if x in man_microsoft else
    'Nintendo' if x in man_nintendo else
    'Sega' if x in man_sega else
    'Sony' if x in man_sony else
    float('NaN'))

# Portable based on the platform
portables = ['3DS','DS','GBA','PSP','PSV']

short_dataset1['Portable'] = short_dataset1['Platform'].map(lambda x: True if x in portables else False)
"""

# remove old columns
short_dataset1 = short_dataset1.drop(['Platform', 'Genre', 'Rating'], axis=1)

# Scale User_Score to boundaries of Critic_Score
short_dataset1[['User_Score']] = short_dataset1[['User_Score']] * 10

display(short_dataset1.head())
print ('REMAINING ROWS IN DATASET #1: {}'.format(*short_dataset1.shape))
print ('Dataset #1 has games from {} unique years'.format(short_dataset1['Year_of_Release'].unique().size))


# Outliers

# There are only few rows before 2000, because MetaCritic was founded in July 16, 1999
# I consider the those earlier rows to be outliers to be removed
short_dataset1 = short_dataset1[short_dataset1.Year_of_Release > 1999]

# display some infos
display(short_dataset1.head())
print ('REMAINING ROWS IN DATASET #1: {}'.format(*short_dataset1.shape))
print ('Dataset #1 has games from {} unique years'.format(short_dataset1['Year_of_Release'].unique().size))

# Show correlations between all variables
display(short_dataset1.corr()[short_dataset1.corr() > 0.7])
# display(short_dataset1.corr())

# In order to understand the dataset better, we should have a look at possible values.


# Critic_Score and User_Score
short_dataset1.plot.scatter(x='Critic_Score', y='User_Score', xlim=(0, 100), ylim=(0,100))

"""
# Just for testing
groups = short_dataset1.groupby('Platform_Manufacturer')

# Plot
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.Critic_Score, group.User_Score, marker='.', linestyle='', label=name)
ax.legend()

plt.show()
"""


# stats
print('MEDIAN')
display(short_dataset1.median())
print('STANDARD DEVIATION')
display(short_dataset1.std())
display(short_dataset1.describe())

# TODO: Check for outliers, see below visually

# TESTING: OUTLIER DETECTION
# We can't simply throw away just by a threshold, we gotta look at the data

# TODO: visual checks first!

OUTLIER_THRESHOLD = 1.5
# For each feature find the data points with extreme high or low values

# TODO: CHECK FOR MULTIPLE OUTLIERS

outliers = []

for feature in short_dataset1.keys():
    if short_dataset1[feature].dtype == float:

        # Calculate Q1 (25th percentile of the data) for the given feature
        Q1 = short_dataset1[feature].quantile(0.25)

        # Calculate Q3 (75th percentile of the data) for the given feature
        Q3 = short_dataset1[feature].quantile(0.75)

        # Use the interquartile range to calculate an outlier step (OUTLIER_THRESHOLD times the interquartile range)
        step = (Q3 - Q1) * OUTLIER_THRESHOLD

        # Display the outlier count
        print "Data points considered outliers for the feature '{}': {}".format(feature, short_dataset1[~((short_dataset1[feature] >= Q1 - step) & (short_dataset1[feature] <= Q3 + step))].shape[0])
        # outliers.extend(short_dataset1[~((short_dataset1[feature] >= Q1 - step) & (short_dataset1[feature] <= Q3 + step))].index.values)

        # remove outliers
        # short_dataset1 = short_dataset1[((short_dataset1[feature] >= Q1 - step) & (short_dataset1[feature] <= Q3 + step))]

display(short_dataset1.describe())
display(short_dataset1.head())

# Critic_Score and User_Score
short_dataset1.plot.scatter(x='Critic_Score', y='User_Score', xlim=(0, 100), ylim=(0,100))

# separate features and target values (sales volume)
sales = pd.DataFrame()
sales['Global'] = short_dataset1['Global_Sales'].copy()
sales['NA'] = short_dataset1['NA_Sales'].copy()
sales['EU'] = short_dataset1['EU_Sales'].copy()
sales['JP'] = short_dataset1['JP_Sales'].copy()
sales['Other'] = short_dataset1['Other_Sales'].copy()

# remove target columns
features = short_dataset1.drop(['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales'], axis=1)

# temporarily remove columns that disturb the functions

# TODO: include them again!
# features = features.drop(['Platform_Manufacturer'], axis=1)

# split data into training set and test set
from sklearn.model_selection import train_test_split

RANDOM_STATE =  31415
TEST_SIZE    =  0.2

# shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(features, sales['Global'], test_size=TEST_SIZE, random_state=RANDOM_STATE)

# Success
print "Training and testing split was successful."

from sklearn.metrics import r2_score

def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between
        true and predicted values based on the metric chosen. """

    # TODO: Calculate the performance score between 'y_true' and 'y_predict'
    score = r2_score(y_true, y_predict)

    # Return the score
    return score

from sklearn.model_selection import ShuffleSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn import svm

# create a decision tree regressor object
#regressor = DecisionTreeRegressor(max_depth = 5)
#params = {'max_depth':(1,2,3,4,5)}

# create a support vector regressor object
regressor = svm.SVR()
params = {'C': [0.1]}

def fit_model(X, y, regressor, params):
    """ Performs grid search over the 'max_depth' parameter for a
        decision tree regressor trained on the input data [X, y]. """

    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(n_splits = 20, test_size=TEST_SIZE, random_state = RANDOM_STATE)

    # Transform 'performance_metric' into a scoring function using 'make_scorer'
    scoring_fnc = make_scorer(performance_metric)

    # Create the grid search object
    grid = GridSearchCV(regressor, params, scoring=scoring_fnc, cv=cv_sets)

    # Fit he grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_

# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train, regressor, params)

vs.DisplayLearningCurve(X_train, y_train, regressor, params)

# Produce the value for 'max_depth'
print (reg.get_params())

# Print the results of prediction for both training and testing
print "For training set, model has a coefficient of determination, R^2, of {:.4f}.".format(performance_metric(y_train, reg.predict(X_train)))
print "For test set, model has a coefficient of determination, R^2, of {:.4f}.".format(performance_metric(y_test, reg.predict(X_test)))
