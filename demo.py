# -*- coding: utf-8 -*-
'''Demonstration file for the pyBandits package'''

__author__ = "Olivier Cappé, Aurélien Garivier, Emilie Kaufmann"
__version__ = "$Revision: 1.6 $"

from environment.MAB import MAB
from environment.MAB_delayed import MAB_delayed
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
from policy.klUCB import klUCB
from policy.KLempUCB import KLempUCB
from policy.Thompson import Thompson
from policy.BayesUCB import BayesUCB
from policy.PossibilisticReward import PossibilisticReward
from policy.PossibilisticRewardLambda import PossibilisticRewardLambda
from policy.PossibilisticReward_II import PossibilisticReward_II
from policy.PossibilisticReward_selfOpt import PossibilisticReward_selfOpt
from policy.PossibilisticReward_IV import PossibilisticReward_IV
from policy.PossibilisticReward_V import PossibilisticReward_V
from policy.PossibilisticReward_var import PossibilisticReward_var
from policy.PossibilisticReward_var2 import PossibilisticReward_var2
from policy.PossibilisticReward_var2 import PossibilisticReward_var2
from policy.DMED import DMED

from policy.PossibilisticReward_var2 import PossibilisticReward_var2
from policy.PossibilisticRewardTuned import  PossibilisticRewardTuned
from policy.PossibilisticRewardTunedDyn import  PossibilisticRewardTunedDyn
from policy.klUCBplus import  klUCBplus

from policy.besa import besa
from Evaluation import *
from kullback import *
from posterior.Beta import Beta



colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'black', 'yellow', 'grey', 'orange', 'violet']
graphic = 'yes'
nbRep = 20
horizon = 20000

scenarios = ['bernoulli','Poisson', 'MixGaussians', 'Gamma', 'Exponential']

scenario = 'bernoulli'

if scenario == 'bernoulli':
    # First scenario (default): Bernoulli experiment with ten arms
    # (figure 2 in [Garivier & Cappé, COLT 2011])
    #env = MAB([Bernoulli_Normal(p, 10, mu=n) for (p,n) in [(0.1, 1) ,(0.05, 2) ,(0.05, 1) ,(0.05, 3) ,(0.02, 5), (0.02, 1), (0.02, 10), (0.01, 1), (0.01, 8) ,(0.01, 1)]])
    #env = MAB([Bernoulli(p, samples=nbRep*horizon) for p in [0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01]])
    env = MAB([Bernoulli(p,samples=nbRep*horizon) for p in [0.5, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45]])
    #policies = [klUCB(env.nbArms), BayesUCB(env.nbArms,Beta)]
    #policies = [PossibilisticReward(env.nbArms, 1, scale=8), PossibilisticReward(env.nbArms, 1, scale=6), PossibilisticReward(env.nbArms, 1, scale=10)]
    trunc = 1
    # policies = [DMED(env.nbArms, trunc),klUCB(env.nbArms), UCB(env.nbArms, trunc),
    #                 PossibilisticReward_var2(env.nbArms, trunc, weight=0.5),
    #                 PossibilisticReward(env.nbArms, trunc, scale=8),
    #                 PossibilisticReward_var(env.nbArms, trunc, weight=0.5)]
    # policies = [BayesUCB(env.nbArms,Beta),
    #             Thompson(env.nbArms, Beta),
    #             PossibilisticReward_var(env.nbArms, trunc, weight=0.5)]
    #policies = [PossibilisticReward_var(env.nbArms, 1, weight=0.5),
               # PossibilisticReward(env.nbArms, 1, scale=8)]
    policies = [PossibilisticReward_var(env.nbArms, trunc), PossibilisticReward_var2(env.nbArms, trunc)]

elif scenario == 'Poisson':
    # Second scenario: Truncated Poissson distrubtions
    trunc = 10
    env = MAB([Poisson(0.5 + 0.25 * a, trunc, samples=nbRep*horizon) for a in range(1, 7)])
    K = env.nbArms;
    policies = [UCB(K, trunc), UCBV(K, trunc), klUCB(K, trunc), klUCB(K, klucb=klucbPoisson), KLempUCB(K, trunc)]
    policies = [PossibilisticReward(env.nbArms, trunc, scale=4),
                PossibilisticReward_IV(env.nbArms, trunc, gap=14, delta=0.6),
                PossibilisticReward_IV(env.nbArms, trunc, gap=4, delta=0.6)]
    policies = [PossibilisticReward(env.nbArms, trunc, scale=12),
                PossibilisticReward_var(env.nbArms, trunc, weight=1)]
    policies = [PossibilisticReward_var(env.nbArms, trunc, weight=0.5)]

elif scenario == 'MixGaussians':
    # Second scenario: Truncated Poissson distrubtions
    trunc = 10
    env = MAB([MixtureGaussian(a) for a in [[0.2, 1, 3, 1.1, 7, 1.2, 0.8, 0.5], [0.15, 1, 2.7, 1.1, 7, 1.2, 0.8, 0.5],
                                            [0.2, 1, 3, 1.1, 5.5, 1.2, 0.8, 0.5], [0.2, 1, 3, 1.1, 7, 1.2, 0.8, 0.5],
                                            [0.4, 0.9, 0.6, 1.3, 4, 0.7, 6, 0.5]]])
    K = env.nbArms;
    policies = [UCB(K, trunc), klUCB(K, trunc), klUCB(K, klucb=klucbPoisson), KLempUCB(K, trunc),
                PossibilisticReward(env.nbArms, trunc, scale=15)]
elif scenario == 'Gamma':
    # Second scenario: Truncated Poissson distrubtions
    trunc = 10
    env = MAB([Gamma(a) for a in [[1, 2], [3, 4], [5, 6], [7, 8], [3.5, 4.5]]])
    K = env.nbArms;
    policies = [UCB(K, trunc), klUCB(K, trunc), klUCB(K, klucb=klGamma), KLempUCB(K, trunc),
                PossibilisticReward(env.nbArms, trunc, scale=45)]
else:
    # Third scenario: Truncated exponential distributions
    trunc = 10
    env = MAB([Exponential(1. / p, trunc, samples=nbRep*horizon) for p in range(1, 6)])
    K = env.nbArms;
    #policies = [UCB(K, trunc), UCBV(K, trunc), klUCB(K, trunc), klUCB(K, klucb=klucbExp), KLempUCB(K, trunc), PossibilisticReward(env.nbArms, trunc, scale=1), PossibilisticReward(env.nbArms, trunc, scale=10)]
    # policies = [klUCB(K, klucb=klucbExp), PossibilisticReward(env.nbArms, trunc, scale=1), PossibilisticReward(env.nbArms, trunc, scale=10), besa(K)]
    #policies = [PossibilisticReward(env.nbArms, trunc, scale=6), PossibilisticReward_var(env.nbArms, trunc, scale=0.7)]
    #policies = [PossibilisticReward(env.nbArms, trunc, scale=1), PossibilisticReward(env.nbArms, trunc, scale=2), PossibilisticReward(env.nbArms, trunc, scale=4), PossibilisticReward(env.nbArms, trunc, scale=6), PossibilisticReward(env.nbArms, trunc, scale=8), PossibilisticReward(env.nbArms, trunc, scale=10),PossibilisticReward(env.nbArms, trunc, scale=12)]
    policies = [PossibilisticReward(env.nbArms, trunc, scale=12),
                PossibilisticReward_var(env.nbArms, trunc, weight=1)]
    policies = [PossibilisticReward_var(env.nbArms, trunc, weight=0.5)]

tsav = int_(linspace(100, horizon - 1, 200))

if graphic == 'yes':
    figure(1)

k = 0
for policy in policies:
    ev = Evaluation(env, policy, nbRep, horizon)
    print ev.meanReward()
    print ev.meanNbDraws()
    meanRegret = ev.meanRegret()
    regret = ev.regret()
    cumReward = ev.cumReward
    if graphic == 'yes':
        # semilogx(1+tsav, meanRegret, color = colors[k])
        plot(np.arange(len(meanRegret)), meanRegret, color=colors[k])
        xlabel('Time')
        ylabel('Regret')

    k = k + 1

legend(["DMED+","KL-UCB+", "BESA", "UCB","KL-UCB", "BAYES-UCB", "TS", "PR 1", "PR 8", "DPR"], loc=0)

show()

