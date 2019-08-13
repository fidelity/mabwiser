import random
from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy
from mabwiser.simulator import Simulator

from sklearn.preprocessing import StandardScaler
from time import time

size = 1000

decisions = [random.randint(0, 2) for _ in range(size)]
rewards = [random.randint(0, 1000) for _ in range(size)]
contexts = [[random.random() for _ in range(50)] for _ in range(size)]


####################################
# Different Bandits for Simulation
####################################

print('Starting simulation 1\n')


def binarize(decision, reward):

    if decision == 0:
        return reward <= 50
    else:
        return reward >= 220

n_jobs=2
contextual_mabs = [('Random', MAB([0, 1], LearningPolicy.Random(), NeighborhoodPolicy.Radius(10), n_jobs=n_jobs)),
                   ('UCB1', MAB([0, 1], LearningPolicy.UCB1(1), NeighborhoodPolicy.Radius(10), n_jobs=n_jobs)),
                   ('ThompsonSampling', MAB([0, 1], LearningPolicy.ThompsonSampling(binarize),
                                            NeighborhoodPolicy.Radius(10), n_jobs=n_jobs)),
                   ('EpsilonGreedy', MAB([0, 1], LearningPolicy.EpsilonGreedy(epsilon=.15),
                                         NeighborhoodPolicy.Radius(10), n_jobs=n_jobs)),
                   ('Softmax', MAB([0, 1], LearningPolicy.Softmax(), NeighborhoodPolicy.Radius(10), n_jobs=n_jobs))]

context_free_mabs = [('Random', MAB([0, 1], LearningPolicy.Random(), n_jobs=n_jobs)),
                     ('UCB1', MAB([0, 1], LearningPolicy.UCB1(1), n_jobs=n_jobs)),
                     ('ThompsonSampling', MAB([0, 1], LearningPolicy.ThompsonSampling(binarize), n_jobs=n_jobs)),
                     ('EpsilonGreedy', MAB([0, 1], LearningPolicy.EpsilonGreedy(epsilon=.15), n_jobs=n_jobs)),
                     ('Softmax', MAB([0, 1], LearningPolicy.Softmax(), n_jobs=n_jobs))]

mixed = [('RandomRadius', MAB([0, 1], LearningPolicy.Random(), NeighborhoodPolicy.Radius(10), n_jobs=n_jobs)),
          ('Random', MAB([0, 1], LearningPolicy.Random(), n_jobs=n_jobs))]

hyper_parameter_tuning = []
for radius in range(6, 10):
    hyper_parameter_tuning.append(('Radius'+str(radius),
                                  MAB([0, 1], LearningPolicy.UCB1(1), NeighborhoodPolicy.Radius(radius),
                                      n_jobs=n_jobs)))

####################################
# Contextual Simulation
####################################

start = time()
sim = Simulator(contextual_mabs, decisions, rewards, contexts,
                scaler=StandardScaler(), test_size=0.5, is_ordered=False, batch_size=0, seed=123456)
sim.run()
end = time()

runtime = (end - start) / 60
print('Complete', str(runtime) + ' minutes')
print('\n')

for mab_name, mab in sim.bandits:
    print(mab_name)

    # Since simulation is offline, print the bandit stats directly
    print('Worst Case Scenario', sim.bandit_to_arm_to_stats_min[mab_name])
    print('Average Case Scenario', sim.bandit_to_arm_to_stats_avg[mab_name])
    print('Best Case Scenario:', sim.bandit_to_arm_to_stats_max[mab_name])

    print('\n\n')

sim.plot('max', True)

####################################
# Context-Free Simulation
####################################
start = time()
sim = Simulator(context_free_mabs, decisions, rewards, contexts=None,
                scaler=None, test_size=0.5, is_ordered=False, batch_size=100, seed=123456)
sim.run()
end = time()

runtime = (end - start) / 60
print('Complete', str(runtime) + ' minutes')
print('\n')

for mab_name, mab in sim.bandits:
    print(mab_name)

    # Since simulation is _online, print the 'total' stats
    print('Worst Case Scenario:', sim.bandit_to_arm_to_stats_min[mab_name]['total'])
    print('Average Case Scenario:', sim.bandit_to_arm_to_stats_avg[mab_name]['total'])
    print('Best Case Scenario:', sim.bandit_to_arm_to_stats_max[mab_name]['total'])

    print('\n\n')

sim.plot('min', False)

####################################
# Mixed Simulation
####################################
start = time()
sim = Simulator(mixed, decisions, rewards, contexts,
                scaler=StandardScaler(), test_size=0.5, is_ordered=False, batch_size=0, seed=123456)
sim.run()
end = time()

runtime = (end - start) / 60
print('Complete', str(runtime) + ' minutes')
print('\n')

for mab_name, mab in sim.bandits:
    print(mab_name)

    # Since simulation is offline, print the bandit stats directly
    print('Worst Case Scenario', sim.bandit_to_arm_to_stats_min[mab_name])
    print('Average Case Scenario', sim.bandit_to_arm_to_stats_avg[mab_name])
    print('Best Case Scenario:', sim.bandit_to_arm_to_stats_max[mab_name])


    print('\n\n')

sim.plot('avg', False)

####################################
# Hyper-Parameter Tuning Simulation
####################################

start = time()
sim = Simulator(hyper_parameter_tuning, decisions, rewards, contexts,
                scaler=StandardScaler(), test_size=0.5, is_ordered=False, batch_size=10, seed=123456)
sim.run()
end = time()

runtime = (end - start) / 60
print('Complete', str(runtime) + ' minutes')
print('\n')

for mab_name, mab in sim.bandits:
    print(mab_name)

    # Since simulation is _online, print the 'total' stats
    print('Worst Case Scenario:', sim.bandit_to_arm_to_stats_min[mab_name]['total'])
    print('Average Case Scenario:', sim.bandit_to_arm_to_stats_avg[mab_name]['total'])
    print('Best Case Scenario:', sim.bandit_to_arm_to_stats_max[mab_name]['total'])

    print('\n\n')

sim.plot('avg', True)

####################################
# Quick Run
####################################

start = time()
sim = Simulator(contextual_mabs, decisions, rewards, contexts,
                scaler=StandardScaler(), test_size=0.5, is_ordered=False, batch_size=0, seed=123456, is_quick=True)
sim.run()
end = time()

runtime = (end - start) / 60
print('Complete', str(runtime) + ' minutes')
print('\n')

for mab_name, mab in sim.bandits:
    print(mab_name)

    # Since simulation is offline, print the bandit stats directly
    print('Worst Case Scenario', sim.bandit_to_arm_to_stats_min[mab_name])
    print('Average Case Scenario', sim.bandit_to_arm_to_stats_avg[mab_name])
    print('Best Case Scenario:', sim.bandit_to_arm_to_stats_max[mab_name])

    print('\n\n')

sim.plot('max', True)

print('All simulations complete')

