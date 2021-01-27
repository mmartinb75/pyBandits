# -*- coding: utf-8 -*-
'''Exponentially distributed arm.'''

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.5 $"

from random import random
from math import isinf,exp,log
import numpy as np

from Arm import Arm

class ExponentialUnit(Arm):
	"""Exponentially distributed arm, possibly truncated"""
	def __init__(self, p, trunc = float('inf')):
		self.p = p
		self.trunc = trunc
		self.t = 0
		self.advance = [min(-1./self.p*log(random()), self.trunc)/10 for i in range(10000000)]
		self.expectation = sum(self.advance)/len(self.advance)


		#self.expectation  =  np.mean(self.rewards)


	def draw(self):
		res = self.advance[self.t]
		self.t +=1
		return res

	def restart(self):
		self.t = 0