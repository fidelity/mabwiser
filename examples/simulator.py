import random
from mabwiser.mab import MAB
from mabwiser.configs.mab import MABConfig, SimulatorConfig
from mabwiser.configs.learning import Random, UCB1, ThompsonSampling, EpsilonGreedy, Softmax
from mabwiser.configs.neighborhood import Radius

from mabwiser.simulator.simulator import Simulator
from mabwiser.simulator.evaluators import DefaultEvaluator

from sklearn.preprocessing import StandardScaler
from time import time

size = 1000

decisions = [str(random.randint(0, 1)) for _ in range(size)]
rewards = [float(random.randint(0, 1000)) for _ in range(size)]
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

arms = ["0", "1"]

contextual_mabs = [
    ("Random", MAB(MABConfig(
        arms=arms,
        learning_policy=Random(),
        neighborhood_policy=Radius(radius=10.0),
        n_jobs=n_jobs
    ))),
    ("ThompsonSampling", MAB(MABConfig(
        arms=arms,
        learning_policy=ThompsonSampling(binarizer=binarize),
        neighborhood_policy=Radius(radius=10.0),
        n_jobs=n_jobs
    ))),
    ("UCB1", MAB(MABConfig(
        arms=arms,
        learning_policy=UCB1(alpha=1.0),
        neighborhood_policy=Radius(radius=10.0),
        n_jobs=n_jobs
    ))),
    ("EpsilonGreedy", MAB(MABConfig(
        arms=arms,
        learning_policy=EpsilonGreedy(epsilon=0.15),
        neighborhood_policy=Radius(radius=10.0),
        n_jobs=n_jobs
    ))),
    ("Softmax", MAB(MABConfig(
        arms=arms,
        learning_policy=Softmax(),
        neighborhood_policy=Radius(radius=10.0),
        n_jobs=n_jobs
    ))),
]

context_free_mabs = [
    ("Random", MAB(MABConfig(
        arms=arms,
        learning_policy=Random(),
        n_jobs=n_jobs
    ))),
    ("UCB1", MAB(MABConfig(
        arms=arms,
        learning_policy=ThompsonSampling(binarizer=binarize),
        n_jobs=n_jobs
    ))),
    ("ThompsonSampling", MAB(MABConfig(
        arms=arms,
        learning_policy=UCB1(alpha=1.0),
        n_jobs=n_jobs
    ))),
    ("EpsilonGreedy", MAB(MABConfig(
        arms=arms,
        learning_policy=EpsilonGreedy(epsilon=0.15),
        n_jobs=n_jobs
    ))),
    ("Softmax", MAB(MABConfig(
        arms=arms,
        learning_policy=Softmax(),
        n_jobs=n_jobs
    ))),
]

mixed = [
    ("RandomRadius", MAB(MABConfig(
        arms=arms,
        learning_policy=Random(),
        neighborhood_policy=Radius(radius=10.0),
        n_jobs=n_jobs
    ))),
    ("Random", MAB(MABConfig(
        arms=arms,
        learning_policy=Random(),
        n_jobs=n_jobs
    ))),
]


hyper_parameter_tuning = []
for radius in range(6, 10):
    hyper_parameter_tuning.append(
        (f'Radius{radius}', MAB(MABConfig(
            arms=arms,
            learning_policy=UCB1(alpha=1.0),
            neighborhood_policy=Radius(radius=float(radius)),
            n_jobs=n_jobs
        )))
    )

####################################
# Contextual Simulation
####################################

start = time()

sim = Simulator(
    bandits=contextual_mabs,
    decisions=decisions,
    rewards=rewards,
    contexts=contexts,
    scaler=StandardScaler(),
    config=SimulatorConfig(
        test_size=0.5,
        is_ordered=False,
        batch_size=0,
        seed=123456,
        evaluator=DefaultEvaluator.evaluator
    )
)
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


sim = Simulator(
    bandits=context_free_mabs,
    decisions=decisions,
    rewards=rewards,
    scaler=StandardScaler(),
    config=SimulatorConfig(
        test_size=0.5,
        is_ordered=False,
        batch_size=0,
        seed=123456,
        evaluator=DefaultEvaluator.evaluator
    )
)

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

sim = Simulator(
    bandits=mixed,
    decisions=decisions,
    rewards=rewards,
    contexts=contexts,
    scaler=StandardScaler(),
    config=SimulatorConfig(
        test_size=0.5,
        is_ordered=False,
        batch_size=0,
        seed=123456,
        evaluator=DefaultEvaluator.evaluator
    )
)


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

