# -*- coding: utf-8 -*-
'''
Environement for a Multi-armed bandit problem 
with arms given in the 'arms' list 
'''

__author__ = "Olivier Cappé,Aurélien Garivier"
__version__ = "$Revision: 1.5 $"

from Result import *
from Environment import Environment
from collections import deque

class MAB_delayed(Environment):
    """Multi-armed bandit problem with arms given in the 'arms' list"""
    
    def __init__(self, arms, delay=1000):
        self.arms = arms
        self.nbArms = len(arms)
        self.delay = delay
        # supposed to have a property nbArms

    def play(self, policy, horizon):
        policy.startGame()
        result = Result(self.nbArms, horizon)
        cola = deque([])
        for t in range(horizon):
            choice = policy.choice()
            reward = self.arms[choice].draw()
            if t < self.delay:
                cola.append((choice, reward))
            else:
                delayed_choice, delayed_reward = cola.popleft()

                policy.getReward(delayed_choice, delayed_reward)
                cola.append((choice, reward))
                result.store(t, delayed_choice, delayed_reward)

        return result
    def restart(self):
        for i in self.arms:
            i.restart()
