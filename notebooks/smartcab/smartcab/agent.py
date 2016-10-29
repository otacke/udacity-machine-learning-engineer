import random
import numpy as np

import operator
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        
        # TODO: Initialize any additional variables here
        self.state = None
        
        # q uses tuples of action and state for identifying a reward, e.g. {(state, action): reward}
        self.q = {}
        
        # learning rate [0 = "Doh!" (Homer Simpson) up to 1 = "New is always better!" (Barney Stinson)]
        self.alpha = 0 # will be set by grid search later
        
        # discount factor [0 = "I want it all, I want it now" (Queen) vs. 1 = "Someday Never Comes" (Creedence Clearwater Revival)]
        self.gamma = 0 # will be set by grid search later
        
        # exploration rate [0 = average Hobbit up to 1 = Kirk]
        self.epsilon = 0 # will be set by grid search later
        
        # number of our successes in trial
        self.success = 0
        
        # number of actions leading to a penalty in trial
        self.violations = 0
        
        # net reward of trial
        self.net_reward = []
        
        # number of actions diverging from the shortest path within a trial
        self.moves_made = 0
        
        # minimal distance
        self.min_distance = 0
        
        # indicates if the agent was reset
        self.was_reset = True
        
    def reset(self, destination=None):
        self.planner.route_to(destination)
        
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.was_reset = True

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state   
        self.state = self.build_state(inputs, self.next_waypoint)
        
        # TODO: Select action according to your policy
        action = self.compute_next_action(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        self.q[(self.state, action)] = self.q_learn(self.state, action, reward, self.build_state(self.env.sense(self), self.planner.next_waypoint()))        
     
        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

        # do computation for stats
        self.do_stats(reward, deadline)

    def do_stats(self, reward, deadline):
        # Final Destination?
        if self.planner.next_waypoint() is None:
            self.success += 1

        # store net reward
        self.net_reward.append(reward)
        
        # register traffic violations
        if reward == -1:
            self.violations += 1

        # compute minimal trip distance
        if self.was_reset:
            self.min_distance += deadline / 5
            self.was_reset = False
        
        # store the number of actions "over par"
        self.moves_made += 1

        return

    # build state tuple from relevant variables
    def build_state(self, inputs, waypoint):
        return (inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint) # initial version without 'right' and 'deadline'
        #return (inputs['light'], self.next_waypoint) # ignore other cars totally???

    # compute next action depending on daring
    def compute_next_action(self, state):
        return self.compute_random_action() if random.random() < self.epsilon else self.compute_best_action(state)     

    # compute a random action
    def compute_random_action(self):
        return random.choice(Environment.valid_actions)

    # compute best action based on previous rewards
    def compute_best_action(self, state):
        candidates = [self.q.get((state, a)) for a in Environment.valid_actions]
        index = candidates.index(max(candidates))
        return Environment.valid_actions[index]        

    # q-learning if not in uncharted territory
    def q_learn(self, state, action, reward, next_state):
        old_q = self.q.get((state, action))
        return reward if (old_q is None) else self.compute_bellman(old_q, state, action, reward, next_state)

    # compute the bellman equation
    # cmp. https://classroom.udacity.com/nanodegrees/nd009/parts/0091345409/modules/e64f9a65-fdb5-4e60-81a9-72813beebb7e/lessons/5446820041/concepts/6348990570923
    def compute_bellman(self, old_q, state, action, reward, next_state):
        max_q = max([self.q.get((next_state, a), 0.0) for a in Environment.valid_actions])
        estimate_q = reward + self.gamma * max_q
        return (1 - self.alpha) * old_q + self.alpha * estimate_q

def run():
    """Run the agent for a finite number of trials."""

    # parameters for grid search #1
    #alphas   = [1]
    #gammas   = [0]
    #epsilons = [1]

    # parameters for grid search #2
    #alphas   = [random.random()]
    #gammas   = [random.random()]
    #epsilons = [random.random()]

    # parameters for grid search #3
    #alphas   = [0.5]
    #gammas   = [0.5]
    #epsilons = [0.25]

    # parameters for grid search #4
    alphas   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    gammas   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # parameters for grid search #5
    #alphas   = [0.03, 0.07, 0.1, 0.13, 0.17]
    #gammas   = [0.23, 0.27, 0.2, 0.23, 0.27]
    #epsilons = [0.23, 0.27, 0.2, 0.23, 0.27]

    # number of runs to average the trials
    n_runs = 100
    
    # number of trials/cab rides for training
    n_trials = 100
    
    success    = [] # number of successes in all trials
    violations = [] # average violations per cab ride within whole trial
    net_reward = [] # net_reward of whole trial
    detour     = [] # average actions "over par" per cab ride within whole trial    
    
    # grid results
    grid_success     = {}
    grid_violations  = {}
    grid_net_rewards = {}
    grid_detour      = {}
    
    # the naive grid search
    for trial_alpha in alphas:
        for trial_gamma in gammas:
            for trial_epsilon in epsilons:
                
                # for letting me know where we are...
                print 'probing alpha: {:.2f}, gamma: {:.2f}, epsilon: {:.2f}'.format(trial_alpha, trial_gamma, trial_epsilon)
                
                for run in range(n_runs):
                
                    # Set up environment and agent
                    e = Environment()  # create environment (also adds some dummy traffic)
                    a = e.create_agent(LearningAgent)  # create agent
                    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
                    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

                    # give parameters to agent
                    a.alpha = trial_alpha
                    a.gamma = trial_gamma
                    a.epsilon = trial_epsilon                   

                    # Now simulate it
                    sim = Simulator(e, update_delay=0.0, display=False)  # create simulator (uses pygame when display=True, if available)
                    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

                    sim.run(n_trials=n_trials)  # run for a specified number of trials
                    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
                    
                    success.append(float(a.success)/n_trials)
                    violations.append(float(a.violations)/n_trials)
                    net_reward.append(np.sum(a.net_reward))
                    # we cannot detect the last move of a trial directly, but by adding the number of failed trials
                    detour.append(float(a.moves_made - a.min_distance)/n_trials)
                
                # store each result for a combination
                grid_success[(a.alpha, a.gamma, a.epsilon)] = np.mean(success)
                grid_violations[(a.alpha, a.gamma, a.epsilon)] = np.mean(violations)
                grid_net_rewards[(a.alpha, a.gamma, a.epsilon)] = np.mean(net_reward)
                grid_detour[(a.alpha, a.gamma, a.epsilon)] = np.mean(detour)
                       
    # get best results
    best_tupel = max(grid_success.iteritems(), key=operator.itemgetter(1))[0]

    print "========================================"
    print "RESULTS FOR {} RUNS WITH {} TRIALS EACH".format(n_runs, n_trials)
    print "Highest success rate is {} for alpha={}, gamma={}, and epsilon={}.".format(grid_success[(best_tupel)], best_tupel[0], best_tupel[1], best_tupel[2],)
    print "with average traffic violations per cab ride: {:.2f}". format(grid_violations[(best_tupel)])
    print "with total net reward over all {} trials: {:.2f}". format(n_trials, grid_net_rewards[(best_tupel)])
    print "with average moves above optimum per cab ride: {:.2f}". format(grid_detour[(best_tupel)])


if __name__ == '__main__':
    run()
