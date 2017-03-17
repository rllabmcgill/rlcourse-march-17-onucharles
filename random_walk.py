import numpy as np
# from sklearn.metrics import mean_squared_error
# from math import sqrt
# import matplotlib.pyplot as plt
# import pickle

N_ACTIONS = 2

WALK_LEFT = 0
WALK_RIGHT = 1

REWARD_TERM_GOAL = 1
REWARD_TERM_NONGOAL = 0
REWARD_NONTERM = 0

class random_walk(object):

    def __init__(self, n_states):
        self.n_states = n_states
        # self.N_ACTIONS = 2
        #
        # self.WALK_LEFT = 0
        # self.WALK_RIGHT = 1
        #
        # self.TERM_REWARD_GOAL = 1
        # self.TERM_REWARD_NONGOAL = -1
        # self.NON_TERM_REWARD = 0


        states = np.arange(n_states)
        #actions = np.arange(self.N_ACTIONS)
        self.actual_state_values = (states + 1) / (n_states + 1)

    def walk(self, state, action):
        nextState = -1
        isEnd = False
        curReward = REWARD_NONTERM

        if WALK_LEFT == action:
            nextState = state - 1
            if nextState == -1:
                isEnd = True
                curReward = REWARD_TERM_NONGOAL
                nextState = -1
        elif WALK_RIGHT == action:
            nextState = state + 1
            if nextState == self.n_states:
                isEnd = True
                curReward = REWARD_TERM_GOAL
                nextState = -1
        return nextState, isEnd, curReward


    def generateEpisode(self):
        '''
        Generates episodes according equiprobable random policy.
        Episodes are a list of tuples of the form (state, reward, nextState)
        '''

        episode = []

        start_state = np.floor(self.n_states / 2)
        curState = start_state
        while True:
            action = np.random.binomial(1, 0.5)
            nextState, isEnd, curReward = self.walk(curState, action)
            episode.append((int(curState), curReward, int(nextState)))
            if isEnd == True:
                break
            curState = nextState

        return episode



