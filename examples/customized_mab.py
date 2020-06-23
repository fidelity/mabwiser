# -*- coding: utf-8 -*-

from typing import List

import numpy as np
from scipy.spatial.distance import cdist

from mabwiser.linear import _Linear
from mabwiser.mab import _EpsilonGreedy
from mabwiser.utils import argmax, Arm, Num, create_rng, _BaseRNG


######################################################################################
#
# MABWiser
# Scenario: Customized Greedy Learning Policy
#
# Imagine a scenario where the operational cost of making decisions comes into play.
# Due to the high price attached to some decisions, we cannot take the decision with
# highest expected reward unless the reward is higher than some operational cost margin
# compared to the second best decision.
#
# To implement such a custom greedy learning policy
# We can inherent from the original _EpsilonGreedy class
# and use the original fit() and prediction_expectation() method of the parent class as is.
# For the predict() method, we leverage from the expectations learnt by the greedy policy
# And change the final decision to respect the given operational cost margin.
#
######################################################################################


class CustomGreedy(_EpsilonGreedy):

    def __init__(self, rng: _BaseRNG, arms: List[Arm], n_jobs: int, margin: Num, backend: str = None):
        # initialize the parent class as is
        super().__init__(rng, arms, n_jobs, backend)

        # save the given operational cost margin
        self.margin = margin

    # Override the predict method of the parent class
    def predict(self, contexts: np.ndarray=None):

        # Generate expectations based on epsilon greedy policy
        arm_to_exp = super().predict_expectations()

        # Arm with the highest expectation
        max_arm = argmax(arm_to_exp)
        max_exp = arm_to_exp[max_arm]
        print("Max arm: ", max_arm, " with expectation: ", max_exp)

        # Arm with second highest expectation
        del arm_to_exp[max_arm]
        next_max_arm = argmax(arm_to_exp)
        next_max_exp = arm_to_exp[next_max_arm]
        print("Next max arm: ", next_max_arm, " with expectation: ", next_max_exp)

        # Regret between best and the second best decision
        regret = max_exp - next_max_exp
        print("Regret: ", regret, " margin: ", self.margin)

        # Return the arm with maximum expectation
        # if and only if the regret beats the given operational margin
        return max_arm if regret >= self.margin else next_max_arm


# Random number generator
rng = create_rng(123456)

# Arms
options = [1, 2]

# Historical data of layouts decisions and corresponding rewards
layouts = np.array([1, 1, 1, 2, 1, 2, 2, 1, 2, 1, 2, 2, 1, 2, 1])
revenues = np.array([10, 17, 22, 9, 4, 0, 7, 8, 20, 9, 50, 5, 7, 12, 10])

# Custom greedy learning policy with high operational cost margin
greedy = CustomGreedy(rng, options, 1, 5.0)

# Learn from previous layouts decisions and revenues generated
greedy.fit(decisions=layouts, rewards=revenues)

# Predict the next best layouts decision
prediction = greedy.predict()

# Expected revenues of each layouts learnt from historical data based on epsilon greedy policy
expectations = greedy.predict_expectations()

# Results
print("Custom Margin 5 Greedy: ", prediction, " ", expectations, "\n")
assert(prediction == 1)

# Custom greedy learning policy with low operational cost margin
greedy = CustomGreedy(rng, options, 1, 3)
greedy.fit(decisions=layouts, rewards=revenues)
prediction = greedy.predict()
print("Custom Margin 3 Greedy: ", prediction, " ", expectations, "\n")
assert(prediction == 2)


######################################################################################
#
# MABWiser
# Scenario: Article Recommendation
#
# Imagine a scenario where there is external data about arms which can be used to identify arms
# that are similar, such as the content of articles. We can use this data to improve predictions
# for arms that have no historic data (articles that are new, or have never been used before) by
# using the model of the closest arm to make the predictions rather than waiting for data to
# become available for the new article, effectively the cold-start problem.
#
# To implement such a custom learning policy we can built on top LinUCB
# We can inherent from the _Linear factory class
# The class requires a custom constructor to introduce feature vector for each arm.
# The customized fit() method utilizes the features vectors to find article similar in content
#
######################################################################################


class LinUCBColdStart(_Linear):
    def __init__(self, rng, arms, n_jobs, backend, alpha, l2_lambda=1.0, features=None):
        # initialize the parent class as is
        super().__init__(rng, arms, n_jobs, backend, alpha, l2_lambda, 'ucb')

        # save the feature vectors
        self.features = features

        # List of arms with no decision/reward history
        self.untrained_arms = []

    # Overwrite the add arm method to allow new arm initialize its features
    def add_arm(self, arm, binarizer=None, scaler=None, features=None):
        self.arm_to_expectation[arm] = 0
        self._uptake_new_arm(arm, binarizer, scaler, features)

    # Overwrite the fit method to seed cold start arms
    def fit(self, decisions, rewards, contexts=None):

        self.num_features = contexts.shape[1]

        # Fit a regression for each arm
        for arm in self.arms:

            # Initialize the model
            self.arm_to_model[arm].init(num_features=self.num_features)

            # If there is data for the arm, fit the regression
            indices = np.where(decisions == arm)

            # Track the arms with no decision/reward history
            if indices[0].size == 0:
                self.untrained_arms.append(arm)
            # Fit the arms that do have decision/reward history
            else:
                X = contexts[indices]
                y = rewards[indices]
                self.arm_to_model[arm].fit(X, y)

        self._cold_start_initialize()

    def _cold_start_initialize(self):

        # Initialize the models for the untrained arms with the data from the most similar arm
        for arm in self.untrained_arms:

            distances = {}
            for n in self.arms:
                if n not in self.untrained_arms:
                    distances[n] = cdist(np.asarray([self.features[n]]), np.asarray([self.features[arm]]),
                                         metric='cosine')

            # Identify the closest arm
            closest_arm = min(distances, key=distances.get)
            print('Cold Start Arm:', arm, 'Closest arm:', closest_arm)

            # Set the arm to use the values of the closets arm
            self.arm_to_model[arm].beta = self.arm_to_model[closest_arm].beta.copy()
            self.arm_to_model[arm].A = self.arm_to_model[closest_arm].A.copy()
            self.arm_to_model[arm].A_inv = self.arm_to_model[closest_arm].A_inv.copy()
            self.arm_to_model[arm].Xty = self.arm_to_model[closest_arm].Xty.copy()

            self.untrained_arms.remove(arm)

    def _uptake_new_arm(self, arm, binarizer=None, scaler=None, features=None):

        # Save the given features for the new arm
        self.features[arm] = features[arm].copy()

        # Create the model of the new arm
        self.arm_to_model[arm] = _Linear.factory.get(self.regression)(self.rng, self.l2_lambda, self.alpha)
        self.arm_to_model[arm].init(self.num_features)

        # Mark it as untrained and initialize based on the closest arm
        self.untrained_arms.append(arm)
        self._cold_start_initialize()


# Random number generator
rng = create_rng(123456)

# Arms
articles = [1, 2, 3, 4, 5, 6]

# Historical data of layouts decisions and corresponding rewards
shown = np.array([1, 3, 4, 2, 4, 2, 5, 1, 2, 3, 5, 5, 1, 4, 3])
clicked = np.array([0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0])
contexts = np.array([[0, 0.5, 0.7, 0.15], [1, 0.25, -0.7, 1.0], [0, -0.5, 0.7, 1], [1, 0.33, 0.65, 0.2],
                     [0, -0.1, 0.23, 0.25], [0, 0.44, 0.75, 0.75], [1, -0.25, -0.9, 0.1], [1, 0.01, 0.87, 1.0],
                     [1, 0.36, 0.67, 0.14], [0, 0.45, 0.77, 1.0], [1, 0.6, 0.3, 0.9], [0, -0.12, -0.22, 0.6],
                     [0, 0.31, 0.99, 0.2], [0, 0.8, -0.5, .9], [1, 0.02, 0.7, 0.5]])
article_content = {1: [1, 1, 1], 2: [0, 0, 0.01], 3: [0.1, 0, 1], 4: [1, .5, .2], 5: [0.2, 0, 0.1], 6: [1, 1, 1]}

# Custom LinUCB with content features
linucb = LinUCBColdStart(rng=rng, arms=articles, n_jobs=1, backend=None, alpha=1.0, l2_lambda=1.0,
                         features=article_content)

# Learn from previous article decisions and clicks
linucb.fit(decisions=shown, rewards=clicked, contexts=contexts)

# Predict the next best article recommendation
prediction = linucb.predict(np.array([[1, 0.2, -0.1, 0.75]]))

# Expected clicks for each article
expectations = linucb.predict_expectations(np.array([[1, 0.2, -0.1, 0.75]]))

# Results
print("Cold Start LinUCB:", prediction, " ", expectations, "\n")
assert(prediction == 4)

new_features = {7: [0.1, 0.2, 0.3]}
linucb.add_arm(arm=7, features=new_features)
assert(7 in linucb.features.keys())
assert([0.1, 0.2, 0.3] == linucb.features[7])
