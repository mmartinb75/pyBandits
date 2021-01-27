# -*- coding: utf-8 -*-
'''Gaussian distributed arm.'''

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.4 $"

import random as rand
from Arm import Arm

class MixtureGaussian(Arm):
    """Gaussian distributed arm."""
    def __init__(self, params):
        self.params = params
        self.advance = [self.generate() for i in range(1000000)]
        self.expectation = sum(self.advance)/len(self.advance)
        self.t = 0
        print self.expectation
        
    def draw(self):
        res = self.advance[self.t]
        self.t +=1
        return res

    def generate(self):
    	data = (0.4*(self.params[0]+self.params[1]*rand.gauss(0,1)) +
    	        0.3*(self.params[2]+self.params[3]*rand.gauss(0,1)) +
    	        0.2*(self.params[4]+self.params[5]*rand.gauss(0,1)) +
    	        0.1*(self.params[6]+self.params[7]*rand.gauss(0,1)))
    	return (max(0,min(10,data)))
    def restart(self):
    	self.t = 0
