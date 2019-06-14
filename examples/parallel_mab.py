# -*- coding: utf-8 -*-

import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from mabwiser.mab import MAB, LearningPolicy

######################################################################################
#
# MABWiser
# Scenario: Playlist recommendation for music streaming service
#
# An _online music streaming service wants to recommend a playlist to a user
# based on a user's listening history and user features. There is a large amount
# of data available to train this recommender model, which means the parallel
# functionality in MABWiser can be useful.
#
#
######################################################################################

# Seed
seed = 111

# Arms
arms = list(np.arange(100))

# Historical on user contexts and rewards (i.e. whether a user clicked
# on the recommended playlist or not)
contexts, rewards = make_classification(n_samples=100000, n_features=200,
                                        n_informative=20, weights=[0.01], scale=None)

# Independently simulate the recommended playlist for each event
decisions = np.random.choice(arms, size=100000)

# Split data into train and test data sets
contexts_train, contexts_test = train_test_split(contexts, test_size=0.3, random_state=seed)
rewards_train, rewards_test = train_test_split(rewards, test_size=0.3, random_state=seed)
decisions_train, decisions_test = train_test_split(decisions, test_size=0.3, random_state=seed)

# Fit standard scaler for each arm
arm_to_scaler = {}
for arm in arms:
    # Get indices for arm
    indices = np.where(decisions_train == arm)

    # Fit standard scaler
    scaler = StandardScaler()
    scaler.fit(contexts[indices])
    arm_to_scaler[arm] = scaler

########################################################
# LinUCB Learning Policy
########################################################

# LinUCB learning policy with alpha 1.25 and n_jobs = -1 (maximum available cores)
linucb = MAB(arms=arms,
             learning_policy=LearningPolicy.LinUCB(alpha=1.25, arm_to_scaler=arm_to_scaler),
             n_jobs=-1)

# Learn from playlists shown and observed click rewards for each arm
linucb.fit(decisions=decisions_train, rewards=rewards_train, contexts=contexts_train)

# Predict the next best playlist to recommend
prediction = linucb.predict(contexts_test)

# Results
print("LinUCB: ", prediction[:10])

