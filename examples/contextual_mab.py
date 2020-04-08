# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.preprocessing import StandardScaler

from mabwiser.mab import MAB, LearningPolicy, NeighborhoodPolicy

######################################################################################
#
# MABWiser
# Scenario: Advertisement Optimization
#
# An e-commerce website needs to solve the problem of which ad to display to _online users
# Each advertisement decision leads to generating different revenues
#
# What should the choice of advertisement be given the context of an _online user
# based on customer data such as age, click rate, subscriber?
#
######################################################################################

# Arms
ads = [1, 2, 3, 4, 5]

# Historical data of ad decisions with corresponding revenues and context information
train_df = pd.DataFrame({'ad': [1, 1, 1, 2, 4, 5, 3, 3, 2, 1, 4, 5, 3, 2, 5],
                         'revenues': [10, 17, 22, 9, 4, 20, 7, 8, 20, 9, 50, 5, 7, 12, 10],
                         'age': [22, 27, 39, 48, 21, 20, 19, 37, 52, 26, 18, 42, 55, 57, 38],
                         'click_rate': [0.2, 0.6, 0.99, 0.68, 0.15, 0.23, 0.75, 0.17,
                                        0.33, 0.65, 0.56, 0.22, 0.19, 0.11, 0.83],
                         'subscriber': [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0]}
                        )

# Test data to for new prediction
test_df = pd.DataFrame({'age': [37, 52], 'click_rate': [0.5, 0.6], 'subscriber': [0, 1]})
test_df_revenue = pd.Series([7, 13])

# Scale the training and test data
scaler = StandardScaler()
train = scaler.fit_transform(train_df[['age', 'click_rate', 'subscriber']].values.astype('float64'))
test = scaler.transform(test_df.values.astype('float64'))

########################################################
# Radius Neighborhood Policy with UCB1 Learning Policy
########################################################

# Radius contextual policy with radius equals to 5 and ucb1 learning with alpha 1.25
radius = MAB(arms=ads,
             learning_policy=LearningPolicy.UCB1(alpha=1.25),
             neighborhood_policy=NeighborhoodPolicy.Radius(radius=5))

# Learn from previous ads shown and revenues generated
radius.fit(decisions=train_df['ad'], rewards=train_df['revenues'], contexts=train)

# Predict the next best ad to show
prediction = radius.predict(test)

# Expectation of each ad based on learning from past ad revenues
expectations = radius.predict_expectations(test)

# Results
print("Radius: ", prediction, " ", expectations)
assert(prediction == [4, 4])

# Online update of model
radius.partial_fit(decisions=prediction, rewards=test_df_revenue, contexts=test)

# Updating of the model with new arm
radius.add_arm(6)

########################################################
# KNearest Neighborhood Policy with UCB1 Learning Policy
########################################################

# KNearest context policy with k equals to 5 and ucb1 learning with alpha of 1.25
knearest = MAB(arms=ads,
               learning_policy=LearningPolicy.UCB1(alpha=1.25),
               neighborhood_policy=NeighborhoodPolicy.KNearest(k=5))

# Learn from previous ads shown and revenues generated
knearest.fit(decisions=train_df['ad'], rewards=train_df['revenues'], contexts=train)

# Predict the next best ad to show
prediction = knearest.predict(test)

# Expectation of each ad based on learning from past ad revenues
expectations = knearest.predict_expectations(test)

# Results
print("KNearest: ", prediction, " ", expectations)
assert(prediction == [5, 1])

# Online update of model
knearest.partial_fit(decisions=prediction, rewards=test_df_revenue, contexts=test)

# Updating of the model with new arm
knearest.add_arm(6)

########################################################
# KMeans Neighborhood Policy with UCB1 Learning Policy
########################################################

# KMeans clustering context policy with 4 clusters and ucb1 learning with alpha of 1.25
clusters = MAB(arms=ads,
               learning_policy=LearningPolicy.UCB1(alpha=1.25),
               neighborhood_policy=NeighborhoodPolicy.Clusters(n_clusters=4))

# Learn from previous ads shown and revenues generated
clusters.fit(decisions=train_df['ad'], rewards=train_df['revenues'], contexts=train)

# Predict the next best ad to show
prediction = clusters.predict(test)

# Expectation of each ad based on learning from past ad revenues
expectations = clusters.predict_expectations(test)

# Results
print("KMeans: ", prediction, " ", expectations)
assert(prediction == [1, 2])

# Online update of model
clusters.partial_fit(decisions=prediction, rewards=test_df_revenue, contexts=test)

# Updating of the model with new arm
clusters.add_arm(6)

#################################################################
# MiniBatchKMeans Neighborhood Policy with UCB1 Learning Policy
#################################################################

# MiniBatchKMeans clusters context policy with 4 clusters and ucb1 learning with alpha of 1.25
clusters = MAB(arms=ads,
               learning_policy=LearningPolicy.UCB1(alpha=1.25),
               neighborhood_policy=NeighborhoodPolicy.Clusters(n_clusters=4, is_minibatch=True))

# Learn from previous ads shown and revenues generated
clusters.fit(decisions=train_df['ad'], rewards=train_df['revenues'], contexts=train)

# Predict the next best ad to show
prediction = clusters.predict(test)

# Expectation of each ad based on learning from past ad revenues
expectations = clusters.predict_expectations(test)

# Results
print("MiniBatchKMeans: ", prediction, " ", expectations)
assert(prediction == [5, 2])

# Online update of model
clusters.partial_fit(decisions=prediction, rewards=test_df_revenue, contexts=test)

# Updating of the model with new arm
clusters.add_arm(6)
