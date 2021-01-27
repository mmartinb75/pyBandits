# -*- coding: utf-8 -*-
'''A utility class for evaluating the performance of a policy in multi-armed bandit problems.'''

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.10 $"


import numpy as np
#from translate.misc.progressbar import ProgressBar

class Evaluation:
  
    def __init__(self, env, pol, nbRepetitions, horizon, tsav=[]):
        if len(tsav)>0:
            self.tsav = tsav
        else:
            self.tsav = np.arange(horizon)

        self.env = env
        self.pol = pol
        self.nbRepetitions = nbRepetitions
        self.horizon = horizon
        self.nbArms = env.nbArms
        self.nbPulls = np.zeros((self.nbRepetitions, self.nbArms))
        self.cumReward = np.zeros((self.nbRepetitions, len(self.tsav)))
        self.env.restart()
        self.choices = np.zeros((self.nbRepetitions, horizon))
        self.rewards = np.zeros((self.nbRepetitions, horizon))
        self.vars = np.zeros((self.nbRepetitions, horizon))
        self.bads = np.zeros((self.nbRepetitions, horizon))
        self.worsts = np.zeros((self.nbRepetitions, horizon))
                 
        # progress = ProgressBar()
        for k in range(nbRepetitions): # progress(range(nbRepetitions)):
            if nbRepetitions < 10 or k % (nbRepetitions/10)==0:
                print k
            result = env.play(pol, horizon)
            self.nbPulls[k, :] = result.getNbPulls()
            self.cumReward[k, :] = np.cumsum(result.rewards)[self.tsav]
            self.choices[k, :] = result.choices
            self.rewards[k, :] = result.rewards
            self.vars[k, :] = result.vars
            self.bads[k, :] = np.cumsum(result.bads)[self.tsav]
            self.worsts[k, :] = result.worst
            

        # progress.finish()
     
    def meanReward(self):
        return sum(self.cumReward[:,-1])/len(self.cumReward[:,-1])

    def meanNbDraws(self):
        return np.mean(self.nbPulls ,0) 

    def meanRegret(self):
        
        #return (1+self.tsav)*np.mean(self.bestExpect) - np.mean(self.cumReward, 0)
        return (1+self.tsav)*max([arm.expectation for arm in self.env.arms]) - np.mean(self.cumReward, 0)
    def regret(self):
        print "Max"
        print (1+self.tsav)*max([arm.expectation for arm in self.env.arms])
        print "Reward"
        print self.cumReward
        print "regret"
        print ((1+self.tsav)*max([arm.expectation for arm in self.env.arms]) - self.cumReward)

        return (1+self.tsav)*max([arm.expectation for arm in self.env.arms]) - self.cumReward

    def meanVar(self):
        return np.mean(self.vars, 0)

    def getBads(self):
        return np.mean(self.worsts, 0)



