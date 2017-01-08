# Machine Learning Engineer Nanodegree
## Capstone Proposal
Oliver Tacke  
_December 31st, 2050_

## Proposal
It was more difficult than I thought to come up with a proposal for a capstone project. There are tons of interesting data sets out there and dozens of questions that you could possibly ask, but I'd love to create something that someone actually needed.

I contacted a friend of mine who co-created [OpenSNP](https://opensnp.org), a platform hosting raw genotype data. In fact, he could need someone to use labeled data from [1000Genomes](http://www.internationalgenome.org) in order to create a classifier that would tell you what's the origin of genetic data. Unfortunately, my knowledge of biology is too poor. Anyway, what I learned form this inquiry is how important domain knowledge and interdisciplinary teams are for data science and machine learning.

I chose to have a closer look at the field of education where I know a thing or two. I found a paper that dealt with undergraduate student generic problem-solving skills. It was based on an empirical study that had produced some data. It could have been used for creating a predictive model for problem solving skills performance, for detecting/classifying sub-groups of students, etc. Unfortunately, the data was not freely available. We need more Open Science! I contacted the author, but I guess he was more afraid of "losing" his data than excited about getting new tools he could use. He wanted to have my resume - that he received - but I never heard of him again.

Well, I browsed the web for some more data sets and found something that would be interesting for me, that might have an appropriate level of difficulty, and that doesn't require too much domain knowledge.

### Domain Background
_(approx. 1-2 paragraphs)_

_In this section, provide brief details on the background information of the domain from which the project is proposed. Historical information relevant to the project should be included. It should be clear how or why a problem in the domain can or should be solved. Related academic research should be appropriately cited in this section, including why that research is relevant. Additionally, a discussion of your personal motivation for investigating a particular problem in the domain is encouraged but not required._

One of the first video games that I have ever played was [Munchkin](https://en.wikipedia.org/wiki/Munchkin_(video_game)). It was released in 1981 when the video game industry was still in its infancy. Today, it is a multi-billion dollar business. In 2014 in the U.S. alone, 155 million people played video games. In total, they spent 15.4 billion US dollars (cmp. http://www.theesa.com/wp-content/uploads/2015/04/ESA-Essential-Facts-2015.pdf).

In an industry, success is not only measured in good critics, but the amount of money earned. In order to make decisions about future productions, publishers may want to predict the sales figures which they can expect after a new game has been released. Those decisions could possibly be based on historical data and a suitable regression model.

### Problem Statement
_(approx. 1 paragraph)_

_In this section, clearly describe the problem that is to be solved. The problem described should be well defined and should have at least one relevant potential solution. Additionally, describe the problem thoroughly such that it is clear that the problem is quantifiable (the problem can be expressed in mathematical or logical terms) , measurable (the problem can be measured by some metric and clearly observed), and replicable (the problem can be reproduced and occurs more than once)._

The earlier a video game company knows how many copies it can expect to sell of a game, the earlier it can estimate the revenue and the earlier it knows whether the game will be a financial success or not. For example, a decision about starting the production of a sequel or the production of a port to a different platform might depend on this information. In consequence, early knowledge about the sales figures can speed up the decision process.

The problem can clearly be measured, because the success of a video game can be expressed in sales volume or revenue if the retail price is known. Also, it should be possible to create a regression model that takes input data and transforms those to a prediction of revenue. Since there are no random parameters involved, the results will also be reproducible given the same model and the same input parameters.

### Datasets and Inputs
_(approx. 2-3 paragraphs)_

_In this section, the dataset(s) and/or input(s) being considered for the project should be thoroughly described, such as how they relate to the problem and why they should be used. Information such as how the dataset or input is (was) obtained, and the characteristics of the dataset or input, should be included with relevant references and citations as necessary It should be clear how the dataset(s) or input(s) will be used in the project and whether their use is appropriate given the context of the problem._

There are at least two relevant datasets that I'd like to use for creating a regression model for predicting the sales figures of a video game.

- [Video Game Sales with Ratings](https://www.kaggle.com/rush4ratio/video-game-sales-with-ratings):
There's a dataset about video game sales based of data scraped from VGChartz. It contains information about games, including name, platform, year of release, genre, and sales figures for several regions. This dataset has been extended with several features from Metacritic, adding e.g. quality ratings from metacritic's staff and from users, and also adding age/content ratings from the entertainment software rating board. In total, there are more that 5.500 complete cases.

- [IGN scores](https://www.kaggle.com/egrinstein/20-years-of-games/discussion):
There's also a dataset that contains ratings from Imagine Games Network (IGN).

Both data sets might be merged, and there may be similar sets that could be obtained by scraping some other sources. By combining them it could be possible to train a model that uses information about the game's genre, platform, year of release, and the different ratings to predict the sales volume.

Merging several datasets will require to cleanse them. Also, some fuzzy string comparison might be beneficial in order to identify matching entries across the datasets. As a side project, using machine learning for tuning string distance metrics such as [Damerau-Levenshtein distance](https://en.wikipedia.org/wiki/Damerau–Levenshtein_distance) or [Jaro-Winkler distance](https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance) might be interesting and useful ([I recently used both in a completely different project](https://github.com/otacke/h5p-text-utilities)).

### Solution Statement
_(approx. 1 paragraph)_

_In this section, clearly describe a solution to the problem. The solution should be applicable to the project domain and appropriate for the dataset(s) or input(s) given. Additionally, describe the solution thoroughly such that it is clear that the solution is quantifiable (the solution can be expressed in mathematical or logical terms) , measurable (the solution can be measured by some metric and clearly observed), and replicable (the solution can be reproduced and occurs more than once)._

### Benchmark Model
_(approximately 1-2 paragraphs)_

_In this section, provide the details for a benchmark model or result that relates to the domain, problem statement, and intended solution. Ideally, the benchmark model or result contextualizes existing methods or known information in the domain and problem given, which could then be objectively compared to the solution. Describe how the benchmark model or result is measurable (can be measured by some metric and clearly observed) with thorough detail._

### Evaluation Metrics
_(approx. 1-2 paragraphs)_

_In this section, propose at least one evaluation metric that can be used to quantify the performance of both the benchmark model and the solution model. The evaluation metric(s) you propose should be appropriate given the context of the data, the problem statement, and the intended solution. Describe how the evaluation metric(s) are derived and provide an example of their mathematical representations (if applicable). Complex evaluation metrics should be clearly defined and quantifiable (can be expressed in mathematical or logical terms)._

### Project Design
_(approx. 1 page)_

_In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project._
