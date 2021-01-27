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


colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'black', 'yellow', 'grey']
graphic = 'no'
nbRep = int(sys.argv[3])
horizon = int(sys.argv[4])

print 'nbRep :' + str(nbRep)
print 'horizon :' + str(horizon)

scenarios = ['bernoulli_low_var', 'bernoulli_normal', 'bernoulli_high_var', 'poisson', 'exponential']

scenario = scenarios[int(sys.argv[1])]

print 'scenario: ' + scenario


if scenario == 'bernoulli_low_var':
    # First scenario (default): Bernoulli experiment with ten arms
    # (figure 2 in [Garivier & Cappé, COLT 2011])
    env = MAB([Bernoulli(p, samples=nbRep*horizon) for p in [0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01]])
    trunc = 1
    all_policies = [DMED(env.nbArms, trunc), klUCBplus(env.nbArms, trunc),
                    besa(env.nbArms, trunc),
                    UCB(env.nbArms, trunc),
                    klUCB(env.nbArms, trunc),
                    PossibilisticReward(env.nbArms, trunc, scale=70),
                    PossibilisticReward_var(env.nbArms, trunc, weight=0.5),
                    PossibilisticReward_chernoff(env.nbArms, trunc, scale=1),
                    PossibilisticReward_chernoff(env.nbArms, trunc, scale=4)]

elif scenario == 'bernoulli_normal':
    # First scenario (default): Bernoulli experiment with ten arms
    # (figure 2 in [Garivier & Cappé, COLT 2011])
    #env = MAB([Bernoulli(p, samples=nbRep*horizon) for p in [0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01]])
    env = MAB([Bernoulli_Normal(p, 10, mu=n) for (p,n) in [(0.1, 1) ,(0.05, 2) ,(0.05, 1) ,(0.05, 3) ,(0.02, 5), (0.02, 1), (0.02, 10), (0.01, 1), (0.01, 8) ,(0.01, 1)]])
    #env = MAB([Bernoulli(p,samples=nbRep*horizon) for p in [0.5, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45]])
    #policies = [klUCB(env.nbArms), BayesUCB(env.nbArms,Beta)]
    #policies = [PossibilisticReward(env.nbArms, 1, scale=8), PossibilisticReward(env.nbArms, 1, scale=6), PossibilisticReward(env.nbArms, 1, scale=10)]
    #policies = [PossibilisticReward_V(env.nbArms, 1, gap=4, delta=0.2)]
    #policies = [PossibilisticReward(env.nbArms, 1, scale=12)]
    trunc = 10
    all_policies = [DMED(env.nbArms, trunc), klUCBplus(env.nbArms, trunc),
                    besa(env.nbArms, trunc),
                    UCB(env.nbArms, trunc),
                    klUCB(env.nbArms, trunc),
                    PossibilisticReward(env.nbArms, trunc, scale=70),
                    PossibilisticReward_var(env.nbArms, trunc, weight=0.5),
                    PossibilisticReward_chernoff(env.nbArms, trunc, scale=1),
                    PossibilisticReward_chernoff(env.nbArms, trunc, scale=1)]

elif scenario == 'bernoulli_high_var':
    # First scenario (default): Bernoulli experiment with ten arms
    # (figure 2 in [Garivier & Cappé, COLT 2011])
    #env = MAB([Bernoulli(p, samples=nbRep*horizon) for p in [0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01]])
    env = MAB([Bernoulli(p,samples=nbRep*horizon) for p in [0.5, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45]])
    #policies = [klUCB(env.nbArms), BayesUCB(env.nbArms,Beta)]
    #policies = [PossibilisticReward(env.nbArms, 1, scale=8), PossibilisticReward(env.nbArms, 1, scale=6), PossibilisticReward(env.nbArms, 1, scale=10)]
    #policies = [PossibilisticReward_V(env.nbArms, 1, gap=4, delta=0.2)]
    #policies = [PossibilisticReward(env.nbArms, 1, scale=12)]
    trunc = 1
    all_policies = [DMED(env.nbArms, trunc), klUCBplus(env.nbArms),
                    besa(env.nbArms),
                    UCB(env.nbArms),
                    klUCB(env.nbArms),
                    BayesUCB(env.nbArms,Beta),
                    Thompson(env.nbArms,Beta),
                    PossibilisticReward(env.nbArms, trunc, scale=1),
                    PossibilisticReward(env.nbArms, trunc, scale=8),
                    PossibilisticReward_var(env.nbArms, trunc, weight=1),
                    PossibilisticReward_chernoff(env.nbArms, trunc, scale=1),
                    PossibilisticReward_chernoff(env.nbArms, trunc, scale=3)]


elif scenario == 'poisson':
    # Second scenario: Truncated Poissson distrubtions
    trunc = 10
    env = MAB([Poisson(0.5 + 0.25 * a, trunc, samples=nbRep*horizon) for a in range(1, 7)])
    K = env.nbArms;
    all_policies = [DMED(env.nbArms, trunc), klUCBplus(K, trunc),
                besa(K),
                UCB(K, trunc),
                UCBV(K, trunc),
                klUCB(K, trunc),
                klUCB(K, klucb=klucbPoisson),
                PossibilisticReward(env.nbArms, trunc, scale=1),
                PossibilisticReward(env.nbArms, trunc, scale=12),
                PossibilisticReward_var(env.nbArms, trunc, weight=1),
                PossibilisticReward_chernoff(env.nbArms, trunc, scale=1),
                PossibilisticReward_chernoff(env.nbArms, trunc, scale=14)]

else:
    # Third scenario: Truncated exponential distributions
    trunc = 10
    env = MAB([Exponential(1. / p, trunc, samples=nbRep*horizon) for p in range(1, 6)])
    K = env.nbArms;
    all_policies = [DMED(env.nbArms, trunc), klUCBplus(K, trunc),
                    besa(K),
                    UCB(K, trunc),
                UCBV(K, trunc),
                klUCB(K, trunc),
                klUCB(K, klucb=klucbExp),
                PossibilisticReward(env.nbArms, trunc, scale=1),
                PossibilisticReward(env.nbArms, trunc, scale=6),
                PossibilisticReward_var(env.nbArms, trunc, weight=0.5),
                PossibilisticReward_chernoff(env.nbArms, trunc, scale=1),
                PossibilisticReward_chernoff(env.nbArms, trunc, scale=4)]


policies = [all_policies[int(sys.argv[2])]]
print 'policiy: ' + str(policies)
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

    name_klucb = str(getattr(policy, "klucb" ,"none"))
    name_scale = str(getattr(policy, "scale" ,"none"))
    name_gap = str(getattr(policy, "gap" ,"none"))
    name_delta = str(getattr(policy, "delta" ,"none"))
    base_name = "data/" + scenario + "/" + policy.__class__.__name__ + "-" + str(nbRep) + "-" + str(horizon) + "_" + name_klucb + "_"  + name_scale +"_" + name_gap + "_" +  name_delta
    save(base_name + "_" + "regret", regret)
    if graphic == 'yes':
        # semilogx(1+tsav, meanRegret, color = colors[k])
        semilogx(np.arange(len(meanRegret)), meanRegret, color=colors[k])
        xlabel('Time')
        ylabel('Regret')
    k = k + 1

exit(0)

