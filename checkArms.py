# -*- coding: utf-8 -*-
'''Demonstration file for the pyBandits package'''

__author__ = "Olivier Cappé, Aurélien Garivier, Emilie Kaufmann"
__version__ = "$Revision: 1.6 $"

from environment.MAB import MAB
from arm.Bernoulli import Bernoulli
from arm.Bernoulli_Normal import Bernoulli_Normal
from arm.Poisson import Poisson
from arm.Exponential import Exponential
from arm.ExponentialUnit import ExponentialUnit
from arm.MixtureGaussian import MixtureGaussian
from arm.Gamma import Gamma
from policy.UCB import UCB
from numpy import *
from matplotlib.pyplot import *

from policy.UCBV import UCBV
from policy.DMED import DMED
from policy.klUCB import klUCB
from policy.klUCBplus import klUCBplus
from policy.KLempUCB import KLempUCB
from policy.Thompson import Thompson
from policy.BayesUCB import BayesUCB
from policy.PossibilisticReward import PossibilisticReward
from policy.PossibilisticRewardLambda import PossibilisticRewardLambda
from policy.PossibilisticReward_II import PossibilisticReward_II
from policy.PossibilisticReward_III import PossibilisticReward_III
from policy.PossibilisticReward_IV import PossibilisticReward_IV
from policy.PossibilisticReward_V import PossibilisticReward_V
from policy.PossibilisticReward_var import PossibilisticReward_var
from policy.PossibilisticReward_chernoff import PossibilisticReward_chernoff
from policy.besa import besa
from Evaluation import *
from kullback import *
from posterior.Beta import Beta


env = [Bernoulli(p) for p in [0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01]]
optimal = np.zeros(50000)
for j in range(50000):
    best = 0
    for ev in env:
        r = 0
        for i in range(20000):
            r = r + ev.draw()
        if r > best:
            best = r
    if j % 1000 == 0: print j
    optimal[j] = best
save("data/" + "optimals", optimal)
a = load("data/optimals.npy")
print shape(a)
print a

exit(0)

