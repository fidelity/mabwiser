# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from mabwiser.mab import MAB, LearningPolicy

######################################################################################
#
# MABWiser
# Scenario: A/B Testing for Website Layout Design
#
# An e-commerce website experiments with 2 different layouts options for their homepage
# Each layouts decision leads to generating different revenues
#
# What should the choice of layouts be based on historical data?
#
######################################################################################

# Arms
options = [1, 2]

# Historical data of layouts decisions and corresponding rewards
layouts = [1, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1]
revenues = [10, 17, 22, 9, 4, 0, 7, 8, 20, 9, 50, 5, 7, 12, 10]

###################################
# Epsilon Greedy Learning Policy
###################################

# Epsilon Greedy learning policy with random exploration set to 15%
greedy = MAB(arms=options,
             learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15),
             seed=123456)

# Learn from previous layouts decisions and revenues generated
greedy.fit(decisions=layouts, rewards=revenues)

# Predict the next best layouts decision
prediction = greedy.predict()

# Expected revenues of each layouts learnt from historical data based on epsilon greedy policy
expectations = greedy.predict_expectations()

# Results
print("Epsilon Greedy: ", prediction, " ", expectations)
assert(prediction == 1)

# Additional historical data becomes available which allows _online learning
additional_layouts = [1, 2, 1, 2]
additional_revenues = [0, 12, 7, 19]

# Online updating of the model
greedy.partial_fit(additional_layouts, additional_revenues)

# Adding a new layout option
greedy.add_arm(3)

#################################################
# Randomized Popularity Learning Policy
#################################################

# Randomized Popularity learning policy that select arms
# with weighted probability based on the mean reward for each arm
popularity = MAB(arms=options,
                 learning_policy=LearningPolicy.Popularity(),
                 seed=123456)

# Learn from previous layouts decisions and revenues generated
popularity.fit(decisions=layouts, rewards=revenues)

# Predict the next best layouts decision
prediction = popularity.predict()

# Expected revenues of each layouts learnt from historical data
expectations = popularity.predict_expectations()

# Results
print("Randomized Popularity: ", prediction, " ", expectations)
assert(prediction == 1)

###################################
# Softmax Learning Policy
###################################

# Softmax learning policy with tau set to 1
sfm = MAB(arms=options,
          learning_policy=LearningPolicy.Softmax(tau=1),
          seed=123456)
sfm.fit(decisions=layouts, rewards=revenues)
prediction = sfm.predict()
expectations = sfm.predict_expectations()
print("Softmax: ", prediction, " ", expectations)
assert(prediction == 2)

# Online updating of the model
sfm.partial_fit(additional_layouts, additional_revenues)

# Update the model with new arm
sfm.add_arm(3)

###########################################################
# Thompson Sampling with Binary Rewards Learning Policy
###########################################################

# Thompson Sampling learning policy with binary rewards
thompson = MAB(arms=options,
               learning_policy=LearningPolicy.ThompsonSampling(),
               seed=123456)
thompson.fit(decisions=layouts, rewards=[1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1])
prediction = thompson.predict()
expectations = thompson.predict_expectations()
print("Thompson Sampling (0/1): ", prediction, " ", expectations)
assert(prediction == 2)

# Online updating of the model
thompson.partial_fit(additional_layouts, [0, 1, 0, 1])

# Update the model with new arm
thompson.add_arm(3)

#############################################################
# Thompson Sampling with Non-Binary Rewards Learning Policy
#############################################################

# Thompson Sampling learning policy with function for converting rewards to binary
def binary_func(decision, reward):
    decision_to_threshold = {1: 10, 2: 10}
    return 1 if reward > decision_to_threshold[decision] else 0


assert(callable(binary_func))

thompson = MAB(arms=options,
               learning_policy=LearningPolicy.ThompsonSampling(binarizer=binary_func),
               seed=123456)
thompson.fit(decisions=layouts, rewards=revenues)
prediction = thompson.predict()
expectations = thompson.predict_expectations()
print("Thompson Sampling: ", prediction, " ", expectations)
assert(prediction == 2)

# Online updating of the model
thompson.partial_fit(additional_layouts, additional_revenues)

# Update the model with new arm
def binary_func2(decision, reward):
    decision_to_threshold = {1: 10, 2: 10, 3: 15}
    return 1 if reward > decision_to_threshold[decision] else 0

thompson.add_arm(3, binary_func2)
assert(3 in thompson.arms)

##############################################
# Upper Confidence Bound1 Learning Policy
#############################################

# UCB1 learning policy with alpha set to 1.25
ucb = MAB(arms=options,
          learning_policy=LearningPolicy.UCB1(alpha=1.25),
          seed=123456)
ucb.fit(decisions=layouts, rewards=revenues)
prediction = ucb.predict()
expectations = ucb.predict_expectations()
print("UCB1: ", prediction, " ", expectations)
assert(prediction == 2)

# Online updating of the model
ucb.partial_fit(additional_layouts, additional_revenues)

# Update the model with new arm
ucb.add_arm(3)

##############################################
# Data Series as Input
#############################################

# Data Series as training input
df = pd.DataFrame({'layouts': [1, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1],
                   'revenues': [10, 17, 22, 9, 4, 0, 7, 8, 20, 9, 50, 5, 7, 12, 10]})

greedy = MAB(arms=options,
             learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15),
             seed=123456)
greedy.fit(decisions=df['layouts'], rewards=df['revenues'])
prediction = greedy.predict()
expectations = greedy.predict_expectations()
print("Greedy (Data Series): ", prediction, " ", expectations)
assert(prediction == 1)

##############################################
# Numpy Arrays as Input
#############################################

# Numpy Array as training
greedy = MAB(arms=options,
             learning_policy=LearningPolicy.EpsilonGreedy(epsilon=0.15),
             seed=123456)
greedy.fit(decisions=np.array([1, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1]),
           rewards=np.array([10, 17, 22, 9, 4, 0, 7, 8, 20, 9, 50, 5, 7, 12, 10]))
prediction = greedy.predict()
expectations = greedy.predict_expectations()
print("Greedy (Numpy Arrays): ", prediction, " ", expectations)
assert(prediction == 1)
