# Reinforcement Learning

Making decisions based on behavior-esque training using rewards and punishments

## Markov Decision Processes
- Markovian Property: only the present matters
- Rules don't change

### in a nutshell
- States: S
- Actions: A(s), A
- Model: T(s,a,s') ~ Pr(s'|s,a) = probability of being in s' given s,a after transition T
- Reward: R(s), R(s,a), R(s,a,s')
- Policy: pi(s) -> a = 'rule' what to do
- Policy: pi* = policy that maximizes the long term expected reward

### rewards
* (temporal) credit assignment problem: the final outcome isn't known until the end, but you have to figure out where in the sequence something went wrong.
* outcome is sensitive to change of rules!
* Utility of sequences of rewards: U(s_0, s_1, s_2, ...) = sum_{t=0}^{\infty}{y^t R(s_t)} with 0 <= y < 1 <= sum_{t=0}^{\infty}{y^t R_max} = frac{R_max}{1-y} (discounted rewards)
* Utility (long term) != Reward (short term)

### Policies
* pi* = argmax( E[ sum_{t=0}^{\infty}{y^t R(s_t)} | pi ] )
* U^{pi}(s) = E[ sum_{t=0}^{\infty}{y^t R(s_t)} | pi, s_0 = s ]
* pi*(s) = argmax_a( sum_{s'}{ T(s,a,s') U(s') } )
* U(s) = R(s) + y * max_a sum_s' T(s, a, s') U(s')

## Reinforcement Learning
- policy search: state -> (Policy) -> action
- value search: state -> (Utility) -> value
- model based: state, action -> (Transaction, Reward) -> state', reward

- Q-Learning
- Learning rate adaption

## Game Theory
- "mathematics of conflicts of interests"
- multiple agents

### Minimax
- [Minimax algorithm](https://en.wikipedia.org/wiki/Minimax) viable in 2-player, zero-sum games with perfect information (deterministic and non-deterministic)

### Strategy
- pure VS mixed = distribution over strategies
- Nash equilibrium
