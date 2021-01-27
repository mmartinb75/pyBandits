# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
'''The Possibilistic Policy.
  Reference: [Miguel Martin].'''
import numpy

__author__ = "Miguel Martin"
__version__ = "1.0"


from math import sqrt, log, exp
import random as rand

from IndexPolicy import IndexPolicy
from functools import cmp_to_key
from numpy import *

def winBesa(arm1, arm2):
        la1 = len(arm1)
        la2 = len(arm2)
        truncateList = list()
        if la1 == 0:
            return 1
        elif la2 == 0:
            return -1
        elif la1 <= la2:
            for i in range(la1):
                truncateList.append(rand.choice(arm2))

            sample_mean_arm1 = float(sum(arm1)/la1)
            sample_mean_arm2 = float(sum(truncateList)/la1)

        else:
            for i in range(la2):
                truncateList.append(rand.choice(arm1))

            sample_mean_arm1 = float(sum(truncateList)/la2)
            sample_mean_arm2 = float(sum(arm2)/la2)

        #print "arms: " + str(arm1) + ":" + str(arm2)
        #print "length arms: " + str(la1) + ":" + str(la2)
        #print "means: " + str(sample_mean_arm1) + ":" + str(sample_mean_arm2)
        return sample_mean_arm1 - sample_mean_arm2



class besa(IndexPolicy):
    """Class that implements the UCB-V policy.
    """

    def __init__(self, nbArms, amplitude=1., lower=0., scale=1):
        self.nbArms = nbArms
        self.amplitude = amplitude
        self.lower = lower
        self.nbDraws = dict()
        self.cumReward = dict()
        self.cumReward2 = dict()
        self.scale = scale
        self.secuence = dict()
        self.sortedArms = list()


    def startGame(self):
        self.t = 1
        self.sortedArms = list()
        for arm in range(self.nbArms):
            self.nbDraws[arm] = 0
            self.cumReward[arm] = 0.0
            self.cumReward2[arm] = 0.0
            self.secuence[arm] = list([1])
            self.sortedArms.append(self.secuence[arm])



    def computeIndex(self, arm):
        return (self.nbArms - self.sortedArms.index(self.secuence[arm]))



    def getReward(self, arm, reward):
        self.nbDraws[arm] += 1
        self.cumReward[arm] += reward
        self.cumReward2[arm] += reward**2
        self.secuence[arm].append(reward)
        #print "Before sorted: " + str(self.sortedArms)
        self.sortedArms = sorted(self.sortedArms, key=cmp_to_key(winBesa), reverse=True)
        #print "sorted: " + str(self.sortedArms)
        self.t += 1

