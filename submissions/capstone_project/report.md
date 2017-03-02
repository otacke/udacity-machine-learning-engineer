# Predicting Video Game Sales
## Capstone Project "Machine Learning Engineer"
Oliver Tacke  
_December 31st, 2050_

## I. Definition

### Project Overview
One of the first video games that I have ever played was [Munchkin](https://en.wikipedia.org/wiki/Munchkin_(video_game)). It was released in 1981 when the video game industry was still in its infancy. Today, it is a multi-billion dollar business. In 2014 in the U.S. alone, 155 million people played video games (cmp. [Entertainment Software Association, 2015](http://www.theesa.com/wp-content/uploads/2015/04/ESA-Essential-Facts-2015.pdf), p. 2). In total, they spent 15.4 billion US dollars (cmp. [Entertainment Software Association, 2015](http://www.theesa.com/wp-content/uploads/2015/04/ESA-Essential-Facts-2015.pdf), p. 12).

In an industry, success is not only measured in good critics, but the amount of money earned or unit sold. In order to make decisions about future productions, publishers may want to predict the sales figures which they can expect after a new game has been released. Those decisions could possibly be based on historical data and a suitable regression model. For instance, one could hypothesize that good scores in reviews correlate positively with high sales figures. Also I could assume, that those reviews can help us to project future sales. In fact, the general plausibility of this approach has been investigated and proven for the movie picture industry a decade ago: "Online movie reviews are available in large numbers within hours of a new movie’s theatrical release. Their use, thus, allows the generation of reliable forecasts much sooner than before." ([Dellarocas, Zhang & Awad, 2007](http://onlinelibrary.wiley.com/doi/10.1002/dir.20087/abstract), p. 39). For the video game industry, [Beaujon (2012)](https://www.few.vu.nl/nl/Images/werkstuk-beaujon_tcm243-264134.pdf) developed a third-degree polynomial formula for predicting sales from historical data - basically manually using a spreadsheet.

In this report, I present a machine learning approach to this problem that uses more features and more data. It is based on dataset about video game sales based on data scraped from VGChartz ([Video Game Sales with Ratings](https://www.kaggle.com/rush4ratio/video-game-sales-with-ratings)). It contains information about games, including name, platform, year of release, genre, and sales figures for several regions. This dataset has been extended with several features from Metacritic, adding e.g. quality ratings from metacritic's staff and from users, the amount of reviews, and also adding age/content ratings from the entertainment software rating board. In total, there are more that 5.500 complete cases. Also, some more features might be added, e.g. a flag indicating whether a game is part of a franchise such as "Super Mario".

### Problem Statement
The earlier a video game company knows how many copies it can expect to sell of a game, the earlier it can estimate the revenue and the earlier it knows whether the game will be a financial success or not. For example, a decision about starting the production of a sequel or the production of a port to a different platform might depend on this information. In consequence, early knowledge about the sales figures can speed up the decision process.

The solution to the problem will be a regression model can be fed with the variables that have been mentioned in the previous section, and it will output a prediction of sales figures for a video game.

Based on the problem definition above, the dataset will be analyzed and prepared. First, data quality will be checked on a syntactical level. For example, columns that are expected to contain numbers such as the predicted sales volume should not contains strings. After that, it is crucial to understand the data before experimenting with algorithms. Data will be checked for plausibility, because some feature values might be off the chart for whatever reason, etc. Also, by visualizing the data in different ways, we will get some more insight into the problem. This will probably also involve a principal component analysis to filter for features that might not be relevant and thus speed down the algorithms.

Subsequently, suitable algorithms will be selected. There are different methods and approaches that might be appropriate for building a regression model and should be considered. Although quite a lot algorithms might suit the problem, the right choice depends on different criteria, such as efficiency and scalability. Since we are dealing with more than 50 but less than 100.000 samples for predicting a quantity, the [scikit-learn cheat-sheet](http://scikit-learn.org/stable/tutorial/machine_learning_map/) suggests to consider Lasso, ElasticNet, Ridge Regression, Support Vector machines or even Ensemble Regressors. Out of curiosity, we might also tinker with Neural Networks.

The next step will be to run and evaluate the algorithms that we have chosen in the previous step. By comparing each algorithm's performance using e.g. k-fold cross validation, we can identify the algorithms that are best at "understanding" the data and at attacking the prediction of video game sales.

As a final step we are going to improve and finalize the results with focused experiments and fine tuning. Each algorithm has certain parameters that can be tweaked in order to get better results. Approaches such as a grid search will be helpful to support this task systematically. Possibly, even ensemble methods will be tried out for fine tuning the results.

### Metrics
The goal of this project is to predict the sales volume of video games, which simply is an integer. We can apply common statistic approaches to compare and visualize the deviation of the predictions from the correct results. The explained variance score or the R² score could be used to quantify the performance of my model. It indicates how much of the variance within a dataset can be explained by a regression model.

Using this score, we can check our model to be prone to high bias or high variance. One method is to split our data into a training set that's used for training the model and a test set for testing its performance, e.g. using k-fold cross validation. This way we can identify and reduce underfitting or overfitting. Also, plotting the learning curves (training error and the cross validation error in relation to the training set size) can offer insight related to high bias and high variance and appropriate options for improvement such as collecting more samples or more features.

Instead of splitting the dataset, we can also use new data from the data sources (VGChartz, Metacritics) that are not available yet. This would basically be a real world test.

## II. Analysis

### Data Exploration
The raw dataset (Video_Games_Sales_as_at_22_Dec_2016.csv) offers 16 features and 16579 data points:

| TITLE           | DESCRIPTION                                                   | DATA TYPE |
|-----------------|---------------------------------------------------------------|-----------|
| Name            | Name of the game                                              | String    |
| Platform        | Hardware Platform                                             | String    |
| Year_of_Release | Year of release                                               | Numeric   |
| Genre           | Game genre                                                    | String    |
| Publisher       | Publisher                                                     | String    |
| NA_Sales        | Game sales in North America (in millions of units)            | Numeric   |
| EU_Sales        | Game sales in the European Union (in millions of units)       | Numeric   |
| JP_Sales        | Game sales in Japan (in millions of units)                    | Numeric   |
| Other_Sales     | Game sales in the rest of the world (in millions of units)    | Numeric   |
| Global_Sales    | Total sales in the world (in millions of units)               | Numeric   |
| Critic_Score    | Aggregate score compiled by Metacritic staff                  | Numeric   |
| Critic_Count    | The number of critics who generated the Critic_Score          | Numeric   |
| User_Score      | Score by Metacritic's subscribers                             | Numeric   |
| User_Count      | Number of subscribers who gave the User_Score                 | Numeric   |
| Developer       | Party responsible for creating the game                       | String    |
| Rating          | The [ESRB](https://www.esrb.org/) ratings                     | String    |
_Table: Dataset Features_

Looking for high correlations within the dataset hardly reveals anything surprising. There are very high correlations between the global sales volume and the sales volume of North America (0.94) and Europe (0.90) respectively, because those two are the biggest markets contributing the most to the total sum. Also, there's a fairly high correlation between Europe and North America (0.77) and other countries excluding Japan (0.72), suggesting that Europe could be a linking pin between the tastes of the other regions. That's it. The Japanese market seems to work differently.

#### Formal abnormalities
There are some obvious formal abnormalities. For instance, the years of release range from 1977 to 2020, so there must be at least one invalid entry. We can also clearly see that Metacritics doesn't provide scores for all the games that have been listed at VGChartz. There are 8466 rows without a Critic Score and even 9013 rows without a User Score. In addition, not all games contain information about their publisher, developer or their rating.

Having a closer look at the data reveals that there are 269 games without a year of release, mostly for Japanese releases or for old Atari 2600 games. We can also note that there are quite some games seem to have been releases in Japan only, and some that have an extra row for Japan sales instead of using just one.

Furthermore, we can come across a fun fact: In Germany, until 2002 it was forbidden to sell the game "River Raid" from 1982 with its clunky graphics and sound. [It was said to resemble a paramilitary training and could cause muscle cramps, anger, aggressiveness, absent-mindedness, and headaches.](http://www.simulationsraum.de/blog/2011/03/31/river-raid-rage/) The game sold more than 1.6 million copies which is quite a lot even for modern games.

Finally, there are some cells containing the string 'tbd' instead of a proper NaN/None, some cells contain double spaces within strings, and the Critic Score and User Score are scaled differently. While the former ranges from 0 to 100, the latter ranges from 0 to 10.

#### Statistical abnormalities
One of the aspects of statistical abnormalities is the subject of outlier identification. 
"Sample outliers can be identified asof two basic types. Here we are concerned with the firstype, which may conveniently betermed representative outliers. These are sample elements with values that have been correctly recorded and that cannot be assumed to be unique. That is, there is no good reason to assume there are no more similar outliers in the nonsampled part of the target population. The remaining sample outliers, which by default are termed nonrepresentative, are sample elements whose data values are incorrect or unique in some sense." (Chambers, 1986, p. 1063). Some of those nonrepresentative elements have already been mentioned above.

When we investigate the feature of global sales, we can easily note some games with very high sales volumes. For example, the top selling game is said to have been sold more than 82 million times and tenth highest selling game still more than 28 million times -- whereas the mean value of global sales is at merely 0.54 million copies and the median at 0.17. Fortunately, there's an explanation. The "seemingly" top selling game is "Wii Sports", which in fact has changed owners that often, but probably only because it came as a bundle with the Wii console in 2006. People bought it whether they wanted or not and without paying respect to critics. The same is true for a lot of the top ranking titles.

| RANK | TITLE                    | GLOBAL SALES |
|------|--------------------------|--------------|
|    1 | Wii Sports               |        82.53 |
|    2 | Super Mario Bros. NES    |        40.24 |
|    3 | Mario Kart Wii           |        35.52 |
|    4 | Wii Sports Resort        |        32.77 |
|    5 | Pokemon Red/Pokemon Blue |        31.37 |
_Table: Top 5 selling games in raw dataset_

Those and other titles have probably been recorded correctly. Yet they could be considered outliers or suggest to add a feature that indicates that the game had been bundled with hardware and might account for the tremendous sales volume.

We can also detect large very values for the number of votes that come from customers. For example, the game "The Witcher 3" for the PC received over 10,000 votes while the mean value is at roughly 162 votes and the median at 24. It is eye-catching that the games with the highest user count also have a very high user score -- at least most of the time. There are some exceptions, e.g. Diablo III for the PC. It received a critic score of 88 and was praises, yet the user score is at only average. It seems that many people gave a very bad score, presumably because the game forced users to be online in order to be able to even play the game in single-player mode. The high count seems to either confirm love or hate for a game.

| RANK | TITLE                            | CRITIC SCORE | CRITIC COUNT | USER SCORE | USER COUNT |
|------|----------------------------------|--------------|--------------|------------|------------|
|    1 | The Witcher 3: Wild Hunt	(PC)    |           93 |           32 |        9.3 |      10665 |
|    2 | The Witcher 3: Wild Hunt (PS4)   |           92 |           79 |        9.2 |      10179 |
|    3 | Counter-Strike: Source (PC)      |           88 |            9 |        8.9 |       9851 |
|    4 | Diablo III	(PC)                  |           88 |           86 |        4.0 |       9629 |
|    5 | The Elder Scrolls V: Skyrim (PC) |           94 |           32 |        8.1 |       9073 |
_Table: Top 5 games concerning user count_

Given some of the statistical information above we can already assume that some of the features are skewed, e.g. sales volume or user count. The visualizations in the next chapter will show this fact more clearly, but just looking at the median, mean and standard deviation of the numerical values suggests that merely critic score, critic count and user score could be fairly normally distributed.

### Exploratory Visualization
In this section, you will need to provide some form of visualization that summarizes or extracts a relevant characteristic or feature about the data. The visualization should adequately support the data being used. Discuss why this visualization was chosen and how it is relevant. Questions to ask yourself when writing this section:
- _Have you visualized a relevant characteristic or feature about the dataset or input data?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

TODO: What is useful? What neglectable? 
### Algorithms and Techniques
In this section, you will need to discuss the algorithms and techniques you intend to use for solving the problem. You should justify the use of each one based on the characteristics of the problem and the problem domain. Questions to ask yourself when writing this section:
- _Are the algorithms you will use, including any default variables/parameters in the project clearly defined?_
- _Are the techniques to be used thoroughly discussed and justified?_
- _Is it made clear how the input data or datasets will be handled by the algorithms and techniques chosen?_

- optional: statistical outlier removal
- optional: feature scaling
- optional: feature standardization
- optional: feature transforming

### Benchmark
There are some people at Kaggle who seem to be experimenting with the dataset, too. For example, [Jonathan Bouchet built a polynomial regression model in R and reports an R² score of 0.098404](https://www.kaggle.com/jonathanbouchet/d/rush4ratio/video-game-sales-with-ratings/vg-sales-score-prediction/notebook). Since Bouchet barely preprocessed the data and did not try other algorithms, I am quite optimistic that I can come up with a better result.


## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

- Remove unnecessary columns
- ...

### Implementation
In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

## VI. Sources
* Beaujon, Walter S. (2012). _Predicting Video Game Sales in the European Market._ Retrieved from https://www.few.vu.nl/nl/Images/werkstuk-beaujon_tcm243-264134.pdf (January 12, 2016).

* Chambers, Raymond L. (1986). Outlier Robust Finite Population Estimation. _Journal of the American Statistical Association_, 81(396), 1063-1069.

* Dellarocas, Chrysanthos, Zhang, Xiaoquan (Michael) & Awad, Neveen F. (2007). [Exploring the value of online product reviews in forecasting sales: The case of motion pictures.](http://onlinelibrary.wiley.com/doi/10.1002/dir.20087/abstract) _Journal of Interactive Marketing, 21(4)_, 23-45.

* Entertainment Software Association (2015). _Essential Facts About The Computer And Video Game Industry. 2015 Sales, Demographic And Usage Data._ Retrieved from http://www.theesa.com/wp-content/uploads/2015/04/ESA-Essential-Facts-2015.pdf (January 12, 2016).

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
