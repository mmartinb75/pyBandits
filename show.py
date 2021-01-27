# -*- coding: utf-8 -*-
'''Demonstration file for the pyBandits package'''

__author__ = "Olivier Cappé, Aurélien Garivier, Emilie Kaufmann"
__version__ = "$Revision: 1.6 $"


from environment.MAB import MAB
from arm.Bernoulli import Bernoulli
from arm.Poisson import Poisson
from arm.Exponential import Exponential
from arm.MixtureGaussian import MixtureGaussian
from arm.Gamma import Gamma
from policy.UCB import UCB
from numpy import *
from matplotlib.pyplot import *

from policy.UCBV import UCBV
from policy.klUCB import klUCB
from policy.KLempUCB import KLempUCB
from policy.Thompson import Thompson
from policy.BayesUCB import BayesUCB
from policy.PossibilisticReward import PossibilisticReward
from policy.PossibilisticRewardLambda import PossibilisticRewardLambda
from policy.besa import besa
from Evaluation import *
from kullback import *
from posterior.Beta import Beta


colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'black','yellow','grey']
graphic = 'yes'
scenario = 5
nbRep =2
horizon = 20000

if graphic == 'yes':
    figure(1)


scenario = 'Exponential'

# if scenario == 'bernoulli':
#     regret = np.load("bernoulli/UCB-200-200000_none_none_none_none_regret.npy")
#     regret2 = np.load("bernoulli/Thompson-200-200000_none_none_none_none_regret.npy")
#     regret3 = np.load("bernoulli/BayesUCB-200-200000_none_none_none_none_regret.npy")
#     regret4 = np.load("bernoulli/klUCB-200-200000_none_none_none_none_regret.npy")
#     regret5 = np.load("bernoulli/PossibilisticReward-200-200000_none_1_none_none_regret.npy")
#     regret6 = np.load("bernoulli/PossibilisticReward-200-200000_none_2_none_none_regret.npy")
#     regret7 = np.load("bernoulli/PossibilisticReward_V-200-200000_none_8.2_4_0.2_regret.npy")
#
#
#
# elif scenario == 'bernoulli2':
#     regret = np.load("bernoulli2/UCB-200-200000_none_none_none_none_regret.npy")
#     regret2 = np.load("bernoulli2/Thompson-200-200000_none_none_none_none_regret.npy")
#     regret3 = np.load("bernoulli2/BayesUCB-200-200000_none_none_none_none_regret.npy")
#     regret4 = np.load("bernoulli2/klUCB-200-200000_none_none_none_none_regret.npy")
#     regret5 = np.load("bernoulli2/PossibilisticReward-200-200000_none_1_none_none_regret.npy")
#     regret6 = np.load("bernoulli2/PossibilisticReward-200-200000_none_8_none_none_regret.npy")
#     regret7 = np.load("bernoulli2/PossibilisticReward_V-200-200000_none_5.04_4_0.02_regret.npy")
# else:
#     #regret = np.load("Exponential2/klUCB-200-200000_base_none_none_none_regret.npy")
#     #regret2 = np.load("Exponential2/klUCB-200-200000_exp_none_none_none_regret.npy")
#     #regret3 = np.load("Exponential2/PossibilisticReward-200-200000_none_1_none_none_regret.npy")
#     #regret4 = np.load("Exponential2/PossibilisticReward-200-200000_none_4_none_none_regret.npy")
#     regret5 = np.load("Exponential2/PossibilisticReward-200-200000_none_6_none_none_regret.npy")
#     regret6 = np.load("Exponential2/PossibilisticReward-200-200000_none_10_none_none_regret.npy")
#





# #meanRegret = np.mean(regret, 0)
# #meanRegret2 = np.mean(regret2, 0)
# #meanRegret3 = np.mean(regret3, 0)
# #meanRegret4 = np.mean(regret4, 0)
# meanRegret5 = np.mean(regret5, 0)
# meanRegret6 = np.mean(regret6, 0)
# #meanRegret7 = np.mean(regret7, 0)
#
# #plot(np.arange(len(meanRegret)),   meanRegret, color = colors[0])
# #plot(np.arange(len(meanRegret2)), meanRegret2, color = colors[1])
# #plot(np.arange(len(meanRegret3)), meanRegret3, color = colors[2])
# #plot(np.arange(len(meanRegret4)), meanRegret4, color = colors[3])
# plot(np.arange(len(meanRegret5)), meanRegret5, color = colors[4])
# plot(np.arange(len(meanRegret6)), meanRegret6, color = colors[5])
# #plot(np.arange(len(meanRegret7)), meanRegret7, color = colors[6])

regret = np.load("data/bernoulli_normal/PossibilisticReward-200-20000_none_1_none_none_regret.npy")
meanRegret = np.mean(regret, 0)
semilogx(np.arange(len(meanRegret)),   meanRegret, color = colors[0])

regret = np.load("data/bernoulli_normal/PossibilisticReward-200-20000_none_10_none_none_regret.npy")
meanRegret = np.mean(regret, 0)
semilogx(np.arange(len(meanRegret)),   meanRegret, color = colors[1])

regret = np.load("data/bernoulli_normal/PossibilisticReward-200-20000_none_50_none_none_regret.npy")
meanRegret = np.mean(regret, 0)
semilogx(np.arange(len(meanRegret)),   meanRegret, color = colors[2])

regret = np.load("data/bernoulli_normal/PossibilisticReward-200-20000_none_70_none_none_regret.npy")
meanRegret = np.mean(regret, 0)
semilogx(np.arange(len(meanRegret)),   meanRegret, color = colors[3])

regret = np.load("data/bernoulli_normal/PossibilisticReward-200-20000_none_90_none_none_regret.npy")
meanRegret = np.mean(regret, 0)
semilogx(np.arange(len(meanRegret)),   meanRegret, color = colors[4])

regret = np.load("data/bernoulli_normal/PossibilisticReward-200-20000_none_120_none_none_regret.npy")
meanRegret = np.mean(regret, 0)
semilogx(np.arange(len(meanRegret)),   meanRegret, color = colors[5])

regret = np.load("data/bernoulli_normal/PossibilisticReward-200-20000_none_200_none_none_regret.npy")
meanRegret = np.mean(regret, 0)
semilogx(np.arange(len(meanRegret)),   meanRegret, color = colors[6])

#regret = np.load("bernoulli_scales_1/PossibilisticReward-200-200000_none_12_none_none_regret.npy")
#meanRegret = np.mean(regret, 0)
#plot(np.arange(len(meanRegret)),   meanRegret, color = colors[6])




xlabel('Time')
ylabel('Regret')
#legend(["klUCB","klUCB-Exp", "Pos1", "Pos4","UCB","UCBV", "Poss Reward dinamic"], loc=0)
legend(["PR " + r'$\alpha$' + "=1",
        "PR " + r'$\alpha$' + "=10",
        "PR " + r'$\alpha$' + "=50",
        "PR " + r'$\alpha$' + "=70",
        "PR " + r'$\alpha$' + "=90",
        "PR " + r'$\alpha$' + "=120",
        "PR " + r'$\alpha$' + "=200"], loc=0)
if graphic == 'yes':
    show()
