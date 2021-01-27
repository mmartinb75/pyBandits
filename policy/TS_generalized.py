# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
'''The Possibilistic Policy.
  Reference: [Miguel Martin].'''

__author__ = "Miguel Martin"
__version__ = "1.0"


from math import sqrt, log, exp
import random as rand


from IndexPolicy import IndexPolicy

class TS_generalized(IndexPolicy):
    """Class that implements the UCB-V policy.
    """

    def __init__(self, nbArms, amplitude=1., lower=0., scale=1):
        self.nbArms = nbArms
        self.factor = amplitude
        self.amplitude = 1
        self.lower = lower
        self.nbDraws = dict()
        self.cumReward = dict()
        self.cumReward2 = dict()
        self.scale = scale

    def startGame(self):
        self.t = 1
        for arm in range(self.nbArms):
            self.nbDraws[arm] = 0
            self.cumReward[arm] = 0.0
            self.cumReward2[arm] = 0.0

    def fuzzyTransformed(self, x, arm):
        m1 = self.cumReward[arm]/self.nbDraws[arm]
        m = m1/self.factor
        s = self.nbDraws[arm]
        if (x >= self.amplitude) or (x <= 0):
            return 0
        else:
            relative_entropy_a = m*log((m)/x) if m/x > 0 else 0
            relative_entropy_b = (1-m)*log((1-m)/(1-x)) if (1-m)/(1-x) > 0 else 0
            relative_entropy = relative_entropy_a + relative_entropy_b
            # return min([1, 2*exp(-s*self.scale*relative_entropy)])
            try:
                return exp(-s*self.scale*relative_entropy)
            except OverflowError:
                return float('inf')

    def fuzzy(self, x, arm):
        m1 = self.cumReward[arm]/self.nbDraws[arm]
        m = m1/self.factor
        s = self.nbDraws[arm]
        return exp(-2*self.scale*s*((m-x)/self.amplitude)**2)

    def computeIndex(self, arm):
        if self.nbDraws[arm] < 1:
            return rand.random()*self.amplitude + self.lower
        else:
            mu1 = self.cumReward[arm]/self.nbDraws[arm]
            mu = mu1/self.factor
            s = self.nbDraws[arm]
            a = mu*s
            b = s - a
            bet = rand.betavariate(1+a, 1+b)
            sigma = self.amplitude/sqrt(4*s*self.scale)
            #r1 = mu+sigma*rand.gauss(0,1)
            #r2 = rand.random()
            #while r2 > self.fuzzyTransformed(r1, arm)/self.fuzzy(r1,arm):
            #    r1 = mu+sigma*rand.gauss(0,1)
            #    r2 = rand.random()
            return bet


    def getReward(self, arm, reward):
        self.nbDraws[arm] += 1
        self.cumReward[arm] += float(random() < reward)
        self.cumReward2[arm] += self.cumReward[arm]**2
        self.t += 1

