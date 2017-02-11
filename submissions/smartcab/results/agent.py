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
        
        # value for initializing the q table
        self.q_default = 5

        # final q table of best results with initial q = reward
        #self.q = {(('red', None, None, 'right'), 'left'): -0.9461915040873616, (('green', 'left', 'forward', 'forward'), None): 0.0, (('green', 'left', None, 'forward'), 'forward'): 2.0769662559368993, (('green', None, 'left', 'right'), None): 0.0, (('red', 'left', None, 'forward'), None): 0.0, (('green', None, None, 'right'), None): 0.06003990223563961, (('red', None, None, 'forward'), 'forward'): -1.0, (('red', 'right', None, 'forward'), None): 0.0, (('green', None, None, 'forward'), 'left'): -0.3825676325917002, (('red', 'forward', None, 'right'), 'forward'): -1.0, (('green', 'left', 'left', 'forward'), None): 0.0, (('red', None, 'forward', 'right'), None): 0.0, (('red', 'left', None, 'forward'), 'forward'): -1.0, (('red', None, None, 'forward'), 'right'): -0.28984611501191493, (('red', None, 'forward', 'forward'), None): 0.0, (('red', None, None, 'left'), 'forward'): -1.0, (('green', None, 'forward', 'forward'), None): 0.0, (('green', None, 'right', 'forward'), 'right'): -0.5, (('green', 'left', None, 'right'), None): 0.0, (('red', None, None, 'right'), None): 0.06664317301401293, (('red', None, 'left', 'forward'), None): 0.0, (('red', 'left', None, 'right'), None): 0.0, (('green', 'left', None, 'forward'), None): 0.0, (('green', None, None, 'right'), 'forward'): -0.45476794876879145, (('red', None, None, 'left'), 'right'): -0.4623040027650762, (('red', None, None, 'left'), 'left'): -1.0, (('red', None, None, 'left'), None): 0.0, (('green', None, None, 'left'), None): 0.14350871184045882, (('red', None, None, 'right'), 'right'): 3.2047056750995457, (('red', None, 'right', 'left'), None): 0.0, (('green', 'left', None, 'forward'), 'right'): -0.5, (('red', 'right', None, 'right'), None): 0.0, (('green', None, None, 'forward'), None): 0.22886213824736912, (('green', None, None, 'forward'), 'forward'): 4.162929168271038, (('green', 'right', None, 'left'), None): 0.0, (('green', None, None, 'forward'), 'right'): -0.4182159915856784, (('red', None, 'right', 'forward'), None): 0.0, (('red', 'left', None, 'forward'), 'left'): -1.0, (('red', None, None, 'right'), 'forward'): -0.9204180100584439, (('green', None, None, 'left'), 'right'): -0.4639989927242393, (('green', None, None, 'left'), 'forward'): -0.482, (('red', 'left', 'right', 'forward'), None): 0.0, (('green', None, 'forward', 'left'), 'right'): -0.5, (('green', None, 'left', 'forward'), None): 0.0, (('green', None, None, 'right'), 'right'): 2.6992010514084797, (('green', 'left', None, 'right'), 'forward'): -0.5, (('red', 'forward', None, 'right'), 'left'): -1.0, (('green', 'forward', None, 'left'), None): 0.0, (('red', None, None, 'forward'), 'left'): -1.0, (('red', 'forward', None, 'forward'), None): 0.0, (('green', None, None, 'left'), 'left'): 5.428300368904872, (('green', None, None, 'right'), 'left'): -0.45395326785025214, (('red', None, None, 'forward'), None): 0.0, (('green', 'right', None, 'forward'), 'right'): -0.42548484225145505, (('red', 'forward', None, 'right'), None): 0.0, (('green', 'left', None, 'left'), 'left'): 2.022234215001157, (('red', 'right', None, 'left'), None): 0.0, (('green', 'left', 'right', 'forward'), 'right'): -0.5}
        
        # final q table of best results with initial q = 5 (might violate traffic sometimes)
        #self.q = {(('red', None, None, 'right'), 'left'): -0.3024206389545996, (('red', None, 'right', 'right'), None): 5, (('green', 'left', None, 'right'), None): 3.93295805, (('red', None, 'left', 'forward'), None): 3.4881404945449996, (('green', None, None, 'right'), None): 0.8555011249117009, (('red', 'left', None, 'right'), None): 3.4881404945449996, (('red', None, None, 'forward'), 'forward'): -0.9955673115563245, (('red', 'right', None, 'forward'), None): 5, (('green', None, None, 'right'), 'right'): 3.0090979077212117, (('green', None, 'left', 'forward'), None): 2.1582044745758604, (('green', None, 'right', 'left'), None): 5, (('green', None, 'forward', 'right'), None): 3.93295805, (('green', 'right', None, 'forward'), None): 4.4345, (('red', 'left', None, 'forward'), 'forward'): 2.65538624245379, (('red', None, None, 'forward'), 'right'): -0.25196171465387174, (('red', None, 'forward', 'forward'), None): 4.4345, (('red', None, None, 'left'), 'forward'): -0.7173040224973123, (('green', None, 'forward', 'forward'), None): 2.1582044745758604, (('red', None, 'right', 'forward'), 'right'): 5, (('green', 'right', None, 'right'), None): 5, (('red', 'left', None, 'forward'), 'right'): 2.4113488161286427, (('green', None, 'forward', 'left'), None): 3.09363180461196, (('red', 'left', None, 'forward'), None): 2.7437420475103473, (('red', None, None, 'right'), None): 0.6794344147737408, (('red', None, 'forward', 'left'), None): 5, (('red', None, 'left', 'right'), None): 5, (('green', None, None, 'forward'), 'left'): -0.17142315293395632, (('green', None, None, 'right'), 'forward'): 0.7282309446845889, (('green', 'left', None, 'forward'), None): 1.9141115485013305, (('green', None, None, 'left'), None): 1.5407409641532746, (('red', None, None, 'left'), None): 9.984997195891648e-05, (('red', None, None, 'left'), 'right'): -0.04633182005315557, (('red', None, None, 'left'), 'left'): -0.501888914801135, (('red', None, None, 'right'), 'right'): 3.1816836514669364, (('green', 'forward', None, 'forward'), None): 4.4345, (('green', None, 'left', 'forward'), 'right'): 2.650936855, (('red', 'right', None, 'right'), None): 5, (('green', None, None, 'forward'), None): 0.5194278303776698, (('green', None, None, 'forward'), 'forward'): 5.06338558002154, (('green', 'right', None, 'left'), None): 4.4345, (('green', None, 'forward', 'forward'), 'right'): 3.7453621250134295, (('green', None, None, 'forward'), 'right'): 0.12841901587686397, (('red', 'forward', None, 'left'), None): 3.93295805, (('green', None, 'right', 'right'), None): 5, (('red', None, None, 'right'), 'forward'): -0.2664737698823249, (('green', None, None, 'right'), 'left'): 0.9872779719359829, (('green', None, None, 'left'), 'forward'): 0.7897815099111861, (('red', 'left', 'right', 'forward'), None): 5, (('red', None, 'forward', 'right'), 'right'): 5, (('red', None, 'right', 'left'), None): 5, (('green', None, 'forward', 'forward'), 'forward'): 5, (('green', None, None, 'left'), 'left'): 2.493319971504075, (('red', 'right', None, 'forward'), 'right'): 5, (('green', 'left', None, 'left'), None): 5, (('red', None, None, 'forward'), 'left'): -0.9930697446003097, (('green', 'left', None, 'right'), 'left'): 3.7658360799273543, (('red', 'forward', None, 'left'), 'forward'): 3.6876610499999996, (('green', None, None, 'left'), 'right'): 0.7449793641663388, (('red', None, None, 'forward'), None): 1.256324494036063e-24, (('green', None, 'left', 'right'), None): 4.4345, (('green', None, 'forward', 'right'), 'right'): 4.092817945776977, (('green', None, 'left', 'right'), 'right'): 5}
        
        # perfect q table for "safety first, but then move quick"
        #self.q = {(('red', None, None, None), None): 0, (('red', None, None, None), 'left'): -1, (('red', None, None, None), 'forward'): -1, (('red', None, None, None), 'right'): -0.5, (('red', None, None, 'left'), None): 0, (('red', None, None, 'left'), 'left'): -1, (('red', None, None, 'left'), 'forward'): -1, (('red', None, None, 'left'), 'right'): -0.5, (('red', None, None, 'forward'), None): 0, (('red', None, None, 'forward'), 'left'): -1, (('red', None, None, 'forward'), 'forward'): -1, (('red', None, None, 'forward'), 'right'): -0.5, (('red', None, None, 'right'), None): 0, (('red', None, None, 'right'), 'left'): -1, (('red', None, None, 'right'), 'forward'): -1, (('red', None, None, 'right'), 'right'): 2, (('red', None, 'left', None), None): 0, (('red', None, 'left', None), 'left'): -1, (('red', None, 'left', None), 'forward'): -1, (('red', None, 'left', None), 'right'): -0.5, (('red', None, 'left', 'left'), None): 0, (('red', None, 'left', 'left'), 'left'): -1, (('red', None, 'left', 'left'), 'forward'): -1, (('red', None, 'left', 'left'), 'right'): -0.5, (('red', None, 'left', 'forward'), None): 0, (('red', None, 'left', 'forward'), 'left'): -1, (('red', None, 'left', 'forward'), 'forward'): -1, (('red', None, 'left', 'forward'), 'right'): -0.5, (('red', None, 'left', 'right'), None): 0, (('red', None, 'left', 'right'), 'left'): -1, (('red', None, 'left', 'right'), 'forward'): -1, (('red', None, 'left', 'right'), 'right'): 2, (('red', None, 'forward', None), None): 0, (('red', None, 'forward', None), 'left'): -1, (('red', None, 'forward', None), 'forward'): -1, (('red', None, 'forward', None), 'right'): -1, (('red', None, 'forward', 'left'), None): 0, (('red', None, 'forward', 'left'), 'left'): -1, (('red', None, 'forward', 'left'), 'forward'): -1, (('red', None, 'forward', 'left'), 'right'): -1, (('red', None, 'forward', 'forward'), None): 0, (('red', None, 'forward', 'forward'), 'left'): -1, (('red', None, 'forward', 'forward'), 'forward'): -1, (('red', None, 'forward', 'forward'), 'right'): -1, (('red', None, 'forward', 'right'), None): 0, (('red', None, 'forward', 'right'), 'left'): -1, (('red', None, 'forward', 'right'), 'forward'): -1, (('red', None, 'forward', 'right'), 'right'): -1, (('red', None, 'right', None), None): 0, (('red', None, 'right', None), 'left'): -1, (('red', None, 'right', None), 'forward'): -1, (('red', None, 'right', None), 'right'): -0.5, (('red', None, 'right', 'left'), None): 0, (('red', None, 'right', 'left'), 'left'): -1, (('red', None, 'right', 'left'), 'forward'): -1, (('red', None, 'right', 'left'), 'right'): -0.5, (('red', None, 'right', 'forward'), None): 0, (('red', None, 'right', 'forward'), 'left'): -1, (('red', None, 'right', 'forward'), 'forward'): -1, (('red', None, 'right', 'forward'), 'right'): -0.5, (('red', None, 'right', 'right'), None): 0, (('red', None, 'right', 'right'), 'left'): -1, (('red', None, 'right', 'right'), 'forward'): -1, (('red', None, 'right', 'right'), 'right'): 2, (('red', 'left', None, None), None): 0, (('red', 'left', None, None), 'left'): -1, (('red', 'left', None, None), 'forward'): -1, (('red', 'left', None, None), 'right'): -0.5, (('red', 'left', None, 'left'), None): 0, (('red', 'left', None, 'left'), 'left'): -1, (('red', 'left', None, 'left'), 'forward'): -1, (('red', 'left', None, 'left'), 'right'): -0.5, (('red', 'left', None, 'forward'), None): 0, (('red', 'left', None, 'forward'), 'left'): -1, (('red', 'left', None, 'forward'), 'forward'): -1, (('red', 'left', None, 'forward'), 'right'): -0.5, (('red', 'left', None, 'right'), None): 0, (('red', 'left', None, 'right'), 'left'): -1, (('red', 'left', None, 'right'), 'forward'): -1, (('red', 'left', None, 'right'), 'right'): 2, (('red', 'left', 'left', None), None): 0, (('red', 'left', 'left', None), 'left'): -1, (('red', 'left', 'left', None), 'forward'): -1, (('red', 'left', 'left', None), 'right'): -0.5, (('red', 'left', 'left', 'left'), None): 0, (('red', 'left', 'left', 'left'), 'left'): -1, (('red', 'left', 'left', 'left'), 'forward'): -1, (('red', 'left', 'left', 'left'), 'right'): -0.5, (('red', 'left', 'left', 'forward'), None): 0, (('red', 'left', 'left', 'forward'), 'left'): -1, (('red', 'left', 'left', 'forward'), 'forward'): -1, (('red', 'left', 'left', 'forward'), 'right'): -0.5, (('red', 'left', 'left', 'right'), None): 0, (('red', 'left', 'left', 'right'), 'left'): -1, (('red', 'left', 'left', 'right'), 'forward'): -1, (('red', 'left', 'left', 'right'), 'right'): 2, (('red', 'left', 'forward', None), None): 0, (('red', 'left', 'forward', None), 'left'): -1, (('red', 'left', 'forward', None), 'forward'): -1, (('red', 'left', 'forward', None), 'right'): -1, (('red', 'left', 'forward', 'left'), None): 0, (('red', 'left', 'forward', 'left'), 'left'): -1, (('red', 'left', 'forward', 'left'), 'forward'): -1, (('red', 'left', 'forward', 'left'), 'right'): -1, (('red', 'left', 'forward', 'forward'), None): 0, (('red', 'left', 'forward', 'forward'), 'left'): -1, (('red', 'left', 'forward', 'forward'), 'forward'): -1, (('red', 'left', 'forward', 'forward'), 'right'): -1, (('red', 'left', 'forward', 'right'), None): 0, (('red', 'left', 'forward', 'right'), 'left'): -1, (('red', 'left', 'forward', 'right'), 'forward'): -1, (('red', 'left', 'forward', 'right'), 'right'): -1, (('red', 'left', 'right', None), None): 0, (('red', 'left', 'right', None), 'left'): -1, (('red', 'left', 'right', None), 'forward'): -1, (('red', 'left', 'right', None), 'right'): -0.5, (('red', 'left', 'right', 'left'), None): 0, (('red', 'left', 'right', 'left'), 'left'): -1, (('red', 'left', 'right', 'left'), 'forward'): -1, (('red', 'left', 'right', 'left'), 'right'): -0.5, (('red', 'left', 'right', 'forward'), None): 0, (('red', 'left', 'right', 'forward'), 'left'): -1, (('red', 'left', 'right', 'forward'), 'forward'): -1, (('red', 'left', 'right', 'forward'), 'right'): -0.5, (('red', 'left', 'right', 'right'), None): 0, (('red', 'left', 'right', 'right'), 'left'): -1, (('red', 'left', 'right', 'right'), 'forward'): -1, (('red', 'left', 'right', 'right'), 'right'): 2, (('red', 'forward', None, None), None): 0, (('red', 'forward', None, None), 'left'): -1, (('red', 'forward', None, None), 'forward'): -1, (('red', 'forward', None, None), 'right'): -0.5, (('red', 'forward', None, 'left'), None): 0, (('red', 'forward', None, 'left'), 'left'): -1, (('red', 'forward', None, 'left'), 'forward'): -1, (('red', 'forward', None, 'left'), 'right'): -0.5, (('red', 'forward', None, 'forward'), None): 0, (('red', 'forward', None, 'forward'), 'left'): -1, (('red', 'forward', None, 'forward'), 'forward'): -1, (('red', 'forward', None, 'forward'), 'right'): -0.5, (('red', 'forward', None, 'right'), None): 0, (('red', 'forward', None, 'right'), 'left'): -1, (('red', 'forward', None, 'right'), 'forward'): -1, (('red', 'forward', None, 'right'), 'right'): 2, (('red', 'forward', 'left', None), None): 0, (('red', 'forward', 'left', None), 'left'): -1, (('red', 'forward', 'left', None), 'forward'): -1, (('red', 'forward', 'left', None), 'right'): -0.5, (('red', 'forward', 'left', 'left'), None): 0, (('red', 'forward', 'left', 'left'), 'left'): -1, (('red', 'forward', 'left', 'left'), 'forward'): -1, (('red', 'forward', 'left', 'left'), 'right'): -0.5, (('red', 'forward', 'left', 'forward'), None): 0, (('red', 'forward', 'left', 'forward'), 'left'): -1, (('red', 'forward', 'left', 'forward'), 'forward'): -1, (('red', 'forward', 'left', 'forward'), 'right'): -0.5, (('red', 'forward', 'left', 'right'), None): 0, (('red', 'forward', 'left', 'right'), 'left'): -1, (('red', 'forward', 'left', 'right'), 'forward'): -1, (('red', 'forward', 'left', 'right'), 'right'): 2, (('red', 'forward', 'forward', None), None): 0, (('red', 'forward', 'forward', None), 'left'): -1, (('red', 'forward', 'forward', None), 'forward'): -1, (('red', 'forward', 'forward', None), 'right'): -1, (('red', 'forward', 'forward', 'left'), None): 0, (('red', 'forward', 'forward', 'left'), 'left'): -1, (('red', 'forward', 'forward', 'left'), 'forward'): -1, (('red', 'forward', 'forward', 'left'), 'right'): -1, (('red', 'forward', 'forward', 'forward'), None): 0, (('red', 'forward', 'forward', 'forward'), 'left'): -1, (('red', 'forward', 'forward', 'forward'), 'forward'): -1, (('red', 'forward', 'forward', 'forward'), 'right'): -1, (('red', 'forward', 'forward', 'right'), None): 0, (('red', 'forward', 'forward', 'right'), 'left'): -1, (('red', 'forward', 'forward', 'right'), 'forward'): -1, (('red', 'forward', 'forward', 'right'), 'right'): -1, (('red', 'forward', 'right', None), None): 0, (('red', 'forward', 'right', None), 'left'): -1, (('red', 'forward', 'right', None), 'forward'): -1, (('red', 'forward', 'right', None), 'right'): -0.5, (('red', 'forward', 'right', 'left'), None): 0, (('red', 'forward', 'right', 'left'), 'left'): -1, (('red', 'forward', 'right', 'left'), 'forward'): -1, (('red', 'forward', 'right', 'left'), 'right'): -0.5, (('red', 'forward', 'right', 'forward'), None): 0, (('red', 'forward', 'right', 'forward'), 'left'): -1, (('red', 'forward', 'right', 'forward'), 'forward'): -1, (('red', 'forward', 'right', 'forward'), 'right'): -0.5, (('red', 'forward', 'right', 'right'), None): 0, (('red', 'forward', 'right', 'right'), 'left'): -1, (('red', 'forward', 'right', 'right'), 'forward'): -1, (('red', 'forward', 'right', 'right'), 'right'): 2, (('red', 'right', None, None), None): 0, (('red', 'right', None, None), 'left'): -1, (('red', 'right', None, None), 'forward'): -1, (('red', 'right', None, None), 'right'): -0.5, (('red', 'right', None, 'left'), None): 0, (('red', 'right', None, 'left'), 'left'): -1, (('red', 'right', None, 'left'), 'forward'): -1, (('red', 'right', None, 'left'), 'right'): -0.5, (('red', 'right', None, 'forward'), None): 0, (('red', 'right', None, 'forward'), 'left'): -1, (('red', 'right', None, 'forward'), 'forward'): -1, (('red', 'right', None, 'forward'), 'right'): -0.5, (('red', 'right', None, 'right'), None): 0, (('red', 'right', None, 'right'), 'left'): -1, (('red', 'right', None, 'right'), 'forward'): -1, (('red', 'right', None, 'right'), 'right'): 2, (('red', 'right', 'left', None), None): 0, (('red', 'right', 'left', None), 'left'): -1, (('red', 'right', 'left', None), 'forward'): -1, (('red', 'right', 'left', None), 'right'): -0.5, (('red', 'right', 'left', 'left'), None): 0, (('red', 'right', 'left', 'left'), 'left'): -1, (('red', 'right', 'left', 'left'), 'forward'): -1, (('red', 'right', 'left', 'left'), 'right'): -0.5, (('red', 'right', 'left', 'forward'), None): 0, (('red', 'right', 'left', 'forward'), 'left'): -1, (('red', 'right', 'left', 'forward'), 'forward'): -1, (('red', 'right', 'left', 'forward'), 'right'): -0.5, (('red', 'right', 'left', 'right'), None): 0, (('red', 'right', 'left', 'right'), 'left'): -1, (('red', 'right', 'left', 'right'), 'forward'): -1, (('red', 'right', 'left', 'right'), 'right'): 2, (('red', 'right', 'forward', None), None): 0, (('red', 'right', 'forward', None), 'left'): -1, (('red', 'right', 'forward', None), 'forward'): -1, (('red', 'right', 'forward', None), 'right'): -1, (('red', 'right', 'forward', 'left'), None): 0, (('red', 'right', 'forward', 'left'), 'left'): -1, (('red', 'right', 'forward', 'left'), 'forward'): -1, (('red', 'right', 'forward', 'left'), 'right'): -1, (('red', 'right', 'forward', 'forward'), None): 0, (('red', 'right', 'forward', 'forward'), 'left'): -1, (('red', 'right', 'forward', 'forward'), 'forward'): -1, (('red', 'right', 'forward', 'forward'), 'right'): -1, (('red', 'right', 'forward', 'right'), None): 0, (('red', 'right', 'forward', 'right'), 'left'): -1, (('red', 'right', 'forward', 'right'), 'forward'): -1, (('red', 'right', 'forward', 'right'), 'right'): -1, (('red', 'right', 'right', None), None): 0, (('red', 'right', 'right', None), 'left'): -1, (('red', 'right', 'right', None), 'forward'): -1, (('red', 'right', 'right', None), 'right'): -0.5, (('red', 'right', 'right', 'left'), None): 0, (('red', 'right', 'right', 'left'), 'left'): -1, (('red', 'right', 'right', 'left'), 'forward'): -1, (('red', 'right', 'right', 'left'), 'right'): -0.5, (('red', 'right', 'right', 'forward'), None): 0, (('red', 'right', 'right', 'forward'), 'left'): -1, (('red', 'right', 'right', 'forward'), 'forward'): -1, (('red', 'right', 'right', 'forward'), 'right'): -0.5, (('red', 'right', 'right', 'right'), None): 0, (('red', 'right', 'right', 'right'), 'left'): -1, (('red', 'right', 'right', 'right'), 'forward'): -1, (('red', 'right', 'right', 'right'), 'right'): 2, (('green', None, None, None), None): 0, (('green', None, None, None), 'left'): -0.5, (('green', None, None, None), 'forward'): -0.5, (('green', None, None, None), 'right'): -0.5, (('green', None, None, 'left'), None): 0, (('green', None, None, 'left'), 'left'): 2, (('green', None, None, 'left'), 'forward'): -0.5, (('green', None, None, 'left'), 'right'): -0.5, (('green', None, None, 'forward'), None): 0, (('green', None, None, 'forward'), 'left'): -0.5, (('green', None, None, 'forward'), 'forward'): 2, (('green', None, None, 'forward'), 'right'): -0.5, (('green', None, None, 'right'), None): 0, (('green', None, None, 'right'), 'left'): -0.5, (('green', None, None, 'right'), 'forward'): -0.5, (('green', None, None, 'right'), 'right'): 2, (('green', None, 'left', None), None): 0, (('green', None, 'left', None), 'left'): -0.5, (('green', None, 'left', None), 'forward'): -0.5, (('green', None, 'left', None), 'right'): -0.5, (('green', None, 'left', 'left'), None): 0, (('green', None, 'left', 'left'), 'left'): 2, (('green', None, 'left', 'left'), 'forward'): -0.5, (('green', None, 'left', 'left'), 'right'): -0.5, (('green', None, 'left', 'forward'), None): 0, (('green', None, 'left', 'forward'), 'left'): -0.5, (('green', None, 'left', 'forward'), 'forward'): 2, (('green', None, 'left', 'forward'), 'right'): -0.5, (('green', None, 'left', 'right'), None): 0, (('green', None, 'left', 'right'), 'left'): -0.5, (('green', None, 'left', 'right'), 'forward'): -0.5, (('green', None, 'left', 'right'), 'right'): 2, (('green', None, 'forward', None), None): 0, (('green', None, 'forward', None), 'left'): -0.5, (('green', None, 'forward', None), 'forward'): -0.5, (('green', None, 'forward', None), 'right'): -0.5, (('green', None, 'forward', 'left'), None): 0, (('green', None, 'forward', 'left'), 'left'): 2, (('green', None, 'forward', 'left'), 'forward'): -0.5, (('green', None, 'forward', 'left'), 'right'): -0.5, (('green', None, 'forward', 'forward'), None): 0, (('green', None, 'forward', 'forward'), 'left'): -0.5, (('green', None, 'forward', 'forward'), 'forward'): 2, (('green', None, 'forward', 'forward'), 'right'): -0.5, (('green', None, 'forward', 'right'), None): 0, (('green', None, 'forward', 'right'), 'left'): -0.5, (('green', None, 'forward', 'right'), 'forward'): -0.5, (('green', None, 'forward', 'right'), 'right'): 2, (('green', None, 'right', None), None): 0, (('green', None, 'right', None), 'left'): -0.5, (('green', None, 'right', None), 'forward'): -0.5, (('green', None, 'right', None), 'right'): -0.5, (('green', None, 'right', 'left'), None): 0, (('green', None, 'right', 'left'), 'left'): 2, (('green', None, 'right', 'left'), 'forward'): -0.5, (('green', None, 'right', 'left'), 'right'): -0.5, (('green', None, 'right', 'forward'), None): 0, (('green', None, 'right', 'forward'), 'left'): -0.5, (('green', None, 'right', 'forward'), 'forward'): 2, (('green', None, 'right', 'forward'), 'right'): -0.5, (('green', None, 'right', 'right'), None): 0, (('green', None, 'right', 'right'), 'left'): -0.5, (('green', None, 'right', 'right'), 'forward'): -0.5, (('green', None, 'right', 'right'), 'right'): 2, (('green', 'left', None, None), None): 0, (('green', 'left', None, None), 'left'): -0.5, (('green', 'left', None, None), 'forward'): -0.5, (('green', 'left', None, None), 'right'): -0.5, (('green', 'left', None, 'left'), None): 0, (('green', 'left', None, 'left'), 'left'): 2, (('green', 'left', None, 'left'), 'forward'): -0.5, (('green', 'left', None, 'left'), 'right'): -0.5, (('green', 'left', None, 'forward'), None): 0, (('green', 'left', None, 'forward'), 'left'): -0.5, (('green', 'left', None, 'forward'), 'forward'): 2, (('green', 'left', None, 'forward'), 'right'): -0.5, (('green', 'left', None, 'right'), None): 0, (('green', 'left', None, 'right'), 'left'): -0.5, (('green', 'left', None, 'right'), 'forward'): -0.5, (('green', 'left', None, 'right'), 'right'): 2, (('green', 'left', 'left', None), None): 0, (('green', 'left', 'left', None), 'left'): -0.5, (('green', 'left', 'left', None), 'forward'): -0.5, (('green', 'left', 'left', None), 'right'): -0.5, (('green', 'left', 'left', 'left'), None): 0, (('green', 'left', 'left', 'left'), 'left'): 2, (('green', 'left', 'left', 'left'), 'forward'): -0.5, (('green', 'left', 'left', 'left'), 'right'): -0.5, (('green', 'left', 'left', 'forward'), None): 0, (('green', 'left', 'left', 'forward'), 'left'): -0.5, (('green', 'left', 'left', 'forward'), 'forward'): 2, (('green', 'left', 'left', 'forward'), 'right'): -0.5, (('green', 'left', 'left', 'right'), None): 0, (('green', 'left', 'left', 'right'), 'left'): -0.5, (('green', 'left', 'left', 'right'), 'forward'): -0.5, (('green', 'left', 'left', 'right'), 'right'): 2, (('green', 'left', 'forward', None), None): 0, (('green', 'left', 'forward', None), 'left'): -0.5, (('green', 'left', 'forward', None), 'forward'): -0.5, (('green', 'left', 'forward', None), 'right'): -0.5, (('green', 'left', 'forward', 'left'), None): 0, (('green', 'left', 'forward', 'left'), 'left'): 2, (('green', 'left', 'forward', 'left'), 'forward'): -0.5, (('green', 'left', 'forward', 'left'), 'right'): -0.5, (('green', 'left', 'forward', 'forward'), None): 0, (('green', 'left', 'forward', 'forward'), 'left'): -0.5, (('green', 'left', 'forward', 'forward'), 'forward'): 2, (('green', 'left', 'forward', 'forward'), 'right'): -0.5, (('green', 'left', 'forward', 'right'), None): 0, (('green', 'left', 'forward', 'right'), 'left'): -0.5, (('green', 'left', 'forward', 'right'), 'forward'): -0.5, (('green', 'left', 'forward', 'right'), 'right'): 2, (('green', 'left', 'right', None), None): 0, (('green', 'left', 'right', None), 'left'): -0.5, (('green', 'left', 'right', None), 'forward'): -0.5, (('green', 'left', 'right', None), 'right'): -0.5, (('green', 'left', 'right', 'left'), None): 0, (('green', 'left', 'right', 'left'), 'left'): 2, (('green', 'left', 'right', 'left'), 'forward'): -0.5, (('green', 'left', 'right', 'left'), 'right'): -0.5, (('green', 'left', 'right', 'forward'), None): 0, (('green', 'left', 'right', 'forward'), 'left'): -0.5, (('green', 'left', 'right', 'forward'), 'forward'): 2, (('green', 'left', 'right', 'forward'), 'right'): -0.5, (('green', 'left', 'right', 'right'), None): 0, (('green', 'left', 'right', 'right'), 'left'): -0.5, (('green', 'left', 'right', 'right'), 'forward'): -0.5, (('green', 'left', 'right', 'right'), 'right'): 2, (('green', 'forward', None, None), None): 0, (('green', 'forward', None, None), 'left'): -1, (('green', 'forward', None, None), 'forward'): -0.5, (('green', 'forward', None, None), 'right'): -0.5, (('green', 'forward', None, 'left'), None): 0, (('green', 'forward', None, 'left'), 'left'): -1, (('green', 'forward', None, 'left'), 'forward'): -0.5, (('green', 'forward', None, 'left'), 'right'): -0.5, (('green', 'forward', None, 'forward'), None): 0, (('green', 'forward', None, 'forward'), 'left'): -1, (('green', 'forward', None, 'forward'), 'forward'): 2, (('green', 'forward', None, 'forward'), 'right'): -0.5, (('green', 'forward', None, 'right'), None): 0, (('green', 'forward', None, 'right'), 'left'): -1, (('green', 'forward', None, 'right'), 'forward'): -0.5, (('green', 'forward', None, 'right'), 'right'): 2, (('green', 'forward', 'left', None), None): 0, (('green', 'forward', 'left', None), 'left'): -1, (('green', 'forward', 'left', None), 'forward'): -0.5, (('green', 'forward', 'left', None), 'right'): -0.5, (('green', 'forward', 'left', 'left'), None): 0, (('green', 'forward', 'left', 'left'), 'left'): -1, (('green', 'forward', 'left', 'left'), 'forward'): -0.5, (('green', 'forward', 'left', 'left'), 'right'): -0.5, (('green', 'forward', 'left', 'forward'), None): 0, (('green', 'forward', 'left', 'forward'), 'left'): -1, (('green', 'forward', 'left', 'forward'), 'forward'): 2, (('green', 'forward', 'left', 'forward'), 'right'): -0.5, (('green', 'forward', 'left', 'right'), None): 0, (('green', 'forward', 'left', 'right'), 'left'): -1, (('green', 'forward', 'left', 'right'), 'forward'): -0.5, (('green', 'forward', 'left', 'right'), 'right'): 2, (('green', 'forward', 'forward', None), None): 0, (('green', 'forward', 'forward', None), 'left'): -1, (('green', 'forward', 'forward', None), 'forward'): -0.5, (('green', 'forward', 'forward', None), 'right'): -0.5, (('green', 'forward', 'forward', 'left'), None): 0, (('green', 'forward', 'forward', 'left'), 'left'): -1, (('green', 'forward', 'forward', 'left'), 'forward'): -0.5, (('green', 'forward', 'forward', 'left'), 'right'): -0.5, (('green', 'forward', 'forward', 'forward'), None): 0, (('green', 'forward', 'forward', 'forward'), 'left'): -1, (('green', 'forward', 'forward', 'forward'), 'forward'): 2, (('green', 'forward', 'forward', 'forward'), 'right'): -0.5, (('green', 'forward', 'forward', 'right'), None): 0, (('green', 'forward', 'forward', 'right'), 'left'): -1, (('green', 'forward', 'forward', 'right'), 'forward'): -0.5, (('green', 'forward', 'forward', 'right'), 'right'): 2, (('green', 'forward', 'right', None), None): 0, (('green', 'forward', 'right', None), 'left'): -1, (('green', 'forward', 'right', None), 'forward'): -0.5, (('green', 'forward', 'right', None), 'right'): -0.5, (('green', 'forward', 'right', 'left'), None): 0, (('green', 'forward', 'right', 'left'), 'left'): -1, (('green', 'forward', 'right', 'left'), 'forward'): -0.5, (('green', 'forward', 'right', 'left'), 'right'): -0.5, (('green', 'forward', 'right', 'forward'), None): 0, (('green', 'forward', 'right', 'forward'), 'left'): -1, (('green', 'forward', 'right', 'forward'), 'forward'): 2, (('green', 'forward', 'right', 'forward'), 'right'): -0.5, (('green', 'forward', 'right', 'right'), None): 0, (('green', 'forward', 'right', 'right'), 'left'): -1, (('green', 'forward', 'right', 'right'), 'forward'): -0.5, (('green', 'forward', 'right', 'right'), 'right'): 2, (('green', 'right', None, None), None): 0, (('green', 'right', None, None), 'left'): -1, (('green', 'right', None, None), 'forward'): -0.5, (('green', 'right', None, None), 'right'): -0.5, (('green', 'right', None, 'left'), None): 0, (('green', 'right', None, 'left'), 'left'): -1, (('green', 'right', None, 'left'), 'forward'): -0.5, (('green', 'right', None, 'left'), 'right'): -0.5, (('green', 'right', None, 'forward'), None): 0, (('green', 'right', None, 'forward'), 'left'): -1, (('green', 'right', None, 'forward'), 'forward'): 2, (('green', 'right', None, 'forward'), 'right'): -0.5, (('green', 'right', None, 'right'), None): 0, (('green', 'right', None, 'right'), 'left'): -1, (('green', 'right', None, 'right'), 'forward'): -0.5, (('green', 'right', None, 'right'), 'right'): 2, (('green', 'right', 'left', None), None): 0, (('green', 'right', 'left', None), 'left'): -1, (('green', 'right', 'left', None), 'forward'): -0.5, (('green', 'right', 'left', None), 'right'): -0.5, (('green', 'right', 'left', 'left'), None): 0, (('green', 'right', 'left', 'left'), 'left'): -1, (('green', 'right', 'left', 'left'), 'forward'): -0.5, (('green', 'right', 'left', 'left'), 'right'): -0.5, (('green', 'right', 'left', 'forward'), None): 0, (('green', 'right', 'left', 'forward'), 'left'): -1, (('green', 'right', 'left', 'forward'), 'forward'): 2, (('green', 'right', 'left', 'forward'), 'right'): -0.5, (('green', 'right', 'left', 'right'), None): 0, (('green', 'right', 'left', 'right'), 'left'): -1, (('green', 'right', 'left', 'right'), 'forward'): -0.5, (('green', 'right', 'left', 'right'), 'right'): 2, (('green', 'right', 'forward', None), None): 0, (('green', 'right', 'forward', None), 'left'): -1, (('green', 'right', 'forward', None), 'forward'): -0.5, (('green', 'right', 'forward', None), 'right'): -0.5, (('green', 'right', 'forward', 'left'), None): 0, (('green', 'right', 'forward', 'left'), 'left'): -1, (('green', 'right', 'forward', 'left'), 'forward'): -0.5, (('green', 'right', 'forward', 'left'), 'right'): -0.5, (('green', 'right', 'forward', 'forward'), None): 0, (('green', 'right', 'forward', 'forward'), 'left'): -1, (('green', 'right', 'forward', 'forward'), 'forward'): 2, (('green', 'right', 'forward', 'forward'), 'right'): -0.5, (('green', 'right', 'forward', 'right'), None): 0, (('green', 'right', 'forward', 'right'), 'left'): -1, (('green', 'right', 'forward', 'right'), 'forward'): -0.5, (('green', 'right', 'forward', 'right'), 'right'): 2, (('green', 'right', 'right', None), None): 0, (('green', 'right', 'right', None), 'left'): -1, (('green', 'right', 'right', None), 'forward'): -0.5, (('green', 'right', 'right', None), 'right'): -0.5, (('green', 'right', 'right', 'left'), None): 0, (('green', 'right', 'right', 'left'), 'left'): -1, (('green', 'right', 'right', 'left'), 'forward'): -0.5, (('green', 'right', 'right', 'left'), 'right'): -0.5, (('green', 'right', 'right', 'forward'), None): 0, (('green', 'right', 'right', 'forward'), 'left'): -1, (('green', 'right', 'right', 'forward'), 'forward'): 2, (('green', 'right', 'right', 'forward'), 'right'): -0.5, (('green', 'right', 'right', 'right'), None): 0, (('green', 'right', 'right', 'right'), 'left'): -1, (('green', 'right', 'right', 'right'), 'forward'): -0.5, (('green', 'right', 'right', 'right'), 'right'): 2}
        
        # learning rate [0 = "Doh!" (Homer Simpson) up to 1 = "New is always better!" (Barney Stinson)]
        self.alpha = 0 # will be set by grid search later
        
        # discount factor [0 = "I want it all, I want it now" (Queen) vs. 1 = "Someday Never Comes" (Creedence Clearwater Revival)]
        self.gamma = 0 # will be set by grid search later
        
        # exploration rate [0 = average Hobbit up to 1 = Kirk]
        self.epsilon = 0 # will be set by grid search later
        
        # time for decays
        self.t = 0
        
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

        # update time dpr decays
        self.t = t + 1

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
        # either use fixed epsilon or degrading epsilon
        epsilon = self.epsilon if self.epsilon >= 0 else 1.0/(self.t)
        return self.compute_random_action() if random.random() < epsilon else self.compute_best_action(state)     

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
        return self.q_default if (old_q is None) else self.compute_bellman(old_q, state, action, reward, next_state)

    # compute the bellman equation
    # cmp. https://classroom.udacity.com/nanodegrees/nd009/parts/0091345409/modules/e64f9a65-fdb5-4e60-81a9-72813beebb7e/lessons/5446820041/concepts/6348990570923
    def compute_bellman(self, old_q, state, action, reward, next_state):
        #alpha decay
        alpha = self.alpha if self.alpha > 0 else 1.0/(self.t)
        max_q = max([self.q.get((next_state, a), 0.0) for a in Environment.valid_actions])
        estimate_q = reward + self.gamma * max_q
        return (1 - alpha) * old_q + alpha * estimate_q

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
    #alphas   = [-1]
    #gammas   = [random.random()]
    #epsilons = [-1]

    # parameters for grid search #4
    #alphas   = [0.5]
    #gammas   = [0.5]
    #epsilons = [0.25]

    # parameters for grid search #5
    #alphas   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, -1]
    #gammas   = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, -1]
   
    # parameters for grid search #6
    #alphas   = [0.03, 0.07, 0.1, 0.13, 0.17]
    #gammas   = [0.03, 0.07, 0.1, 0.13, 0.17]
    #epsilons = [0.13, 0.17, 0.2, 0.23, 0.27]    
    
    # parameters for grid search #7
    alphas   = [0.13]
    gammas   = [0.13]
    epsilons = [0.23]

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
                
                # for letting us know where we are...
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
                    
                    # compute stats for this run
                    success.append(float(a.success)/n_trials)
                    violations.append(float(a.violations)/n_trials)
                    net_reward.append(np.sum(a.net_reward))
                    detour.append(float(a.moves_made - a.min_distance)/n_trials)
                
                # store stats for a combination of parameters
                grid_success[(a.alpha, a.gamma, a.epsilon)] = np.mean(success)
                grid_violations[(a.alpha, a.gamma, a.epsilon)] = np.mean(violations)
                grid_net_rewards[(a.alpha, a.gamma, a.epsilon)] = np.mean(net_reward)
                grid_detour[(a.alpha, a.gamma, a.epsilon)] = np.mean(detour)
                       
    # get best results of all runs
    best_tupel = max(grid_success.iteritems(), key=operator.itemgetter(1))[0]

    print "========================================"
    print "RESULTS FOR {} RUN(S) WITH {} TRIAL(S) EACH".format(n_runs, n_trials)
    print "Highest success rate is {} for\n  alpha={},\n  gamma={}, and\n  epsilon={}.".format(grid_success[(best_tupel)], best_tupel[0], best_tupel[1], best_tupel[2],)
    print "with average traffic violations per cab ride: {:.2f}". format(grid_violations[(best_tupel)])
    print "with total net reward over all {} trial(s): {:.2f}". format(n_trials, grid_net_rewards[(best_tupel)])
    print "with average moves above optimum per cab ride: {:.2f}". format(grid_detour[(best_tupel)])
  
    # If we set n_runs=1 for fixed parameters, get a good result and use the
    # q table as a default for an agent with epsilon = 0, we should have quite
    # a smartcab...
    print "========================================"
    # print a.q

if __name__ == '__main__':
    run()
