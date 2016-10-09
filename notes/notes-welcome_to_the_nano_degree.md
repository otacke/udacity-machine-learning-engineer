# Course notes

Might be useful: [old notes from a Stanford online course at Coursera](https://de.wikiversity.org/wiki/Benutzer:O.tacke/2015/Machine_Learning)

## Getting started: From Artificial Intelligence to Machine Learning
* heuristic by *Thrun*: Inteprtet AI as uncertainty management: "What to do, when you don't know what to do"
* heuristic by *Albus*: ""A useful definition of intelligence ... should include both biological and machine embodiments, and these should span an intellectual range from that of an insect to that of an Einstein, from that of a thermostat to that of the most sophisticated computer system that could ever be built."

### Knowledge Based AI
* Deliberation = Reasoning + Learning + Memorizing
* Cognitive System = Metacognition + Deliberation + Reaction
* As in cognitivism, the cognitive system might be a black box that accepts input and returns output

* Engineering AI (build intelligent agents that may or may not resemble a natural equivalent) vs. Cognitive AI (mimic natural intelligence)
* overview using an agent that must navigate a city in http://dilab.gatech.edu/test/wp-content/uploads/2014/11/AI-GoelDavies2011-Final.pdf

* Four Schools of AI: optimally vs. like humans + acting vs. thinking => agents that act/think optimally/like humans (https://people.eecs.berkeley.edu/~russell/aima1e/chapter01.pdf)
  * think optimally: machine learning, logic, Google Maps
  * act optimally: airplane autopilot, Roomba
  * act like humans: improvisional robots, Turing test related stuff, C3PO
  * think like humans: semantic web, Siri
  
### Bayes Rule
* posterior = P(A|B) = P(B|A) * P(A) / P(B) where
  * P(B|A) = likelihood
  * P(A) = prior
  * P(B) = marginal likelihood
* look up: Bayes Networt -> not explained well

### Machine Learning and Data Science
* Data Scientist (Analyst + Statistician + Software Engineering) or as a Venn diagram...
  * Hacking -(Machine Learning)- Math/Statistics
  * Math/Statistics -(Traditional Research)- Substantive Expertise
  * Substantive Expertise -(Danger Zone)- Hacking
* Definition of Machine Learning: Dealing with computational artifacts that learn over time based on experience using mathematics, science and engineering.
* Supervised Learning
  * I LOVE the "hip to be square" reference in https://www.youtube.com/watch?v=Ki2iHgKxRBo :-D
  * supervised learning as (inductive) function approximation that fits to data, e.g. regression
  * Nice explanation for deduction, induction, and abduction: Using cause, effect and a rule...
    * deduction: cause + rule => *deduce* an effect (truth preserving)
    * induction: cause + effect => *induce* a rule (prone to error)
    * abduction: rule + effect => *abduce* a cause (prone to error)
* Unsupervised Learning
  * Describing "a problem" compactly, making up categories, e.g. clustering
* Reinforcement Learning
  * Learning from delayed reward
* Taxonomy
  * *What?* parameters, structure, hidden concepts
  * *What from?* target labels (supervised), replacement principles (unsupervised), feedback (reinforcement)
  * *What for?* prediction, diagnostics, explanation, summarization
  * *How?* passive, active, on-the-fly, after data generation
  * *Outputs?* classification, regression
  * *Details?* generative, discrimitative
* Occam's Razor
  * Beware of overfitting!!!
