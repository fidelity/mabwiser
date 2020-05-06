.. _about:

About Multi-Armed Bandits
=========================

There are many real-world situations in which we have to decide between multiple options yet we are only able to learn the best course of action by testing each option sequentially.

**Multi-armed bandit (MAB)** algorithms are suitable for such sequential, online decision making problems under uncertainty.
As such, they play an important part in many machine learning applications in internet advertising, recommendation engines, and clinical trials among many others.

.. admonition:: Exploration vs. Exploitation

	In this setting, for each and every renewed decision we face an underlying question: Do we stick to what we know and receive an expected result ("**exploit**") or choose an option we do not know much about and potentially learn something new ("**explore**")? 

**Problem Definition:** In a multi-armed bandits problem, the model of outcomes is unknown, and the outcomes can be deterministic
or stochastic. The agent needs to make a sequence of decisions in time *1, 2, ..., T*.
At each time *t* the agent is given a set of *K* arms, and it has to decide which arm to pull. 
After pulling an arm, it receives a *reward* of that arm, and the rewards of other arms are unknown. 
In a stochastic setting the reward of an arm is sampled from some unknown distribution. There exist situations where we also observe side information at each time *t*.
This side information is referred to as *context*. The arm that has the highest expected reward may be different given different contexts.
This variant is called **contextual multi-armed bandits**. Overall, the objective is to maximize the cumulative expected reward in the long run.

------------

For more information, we refer to these excellent resources:

1. `Contextual Multi-Armed Bandits`_, Tyler Lu *et. al*. Proc. of Machine Learning Research, 2010
2. `A Survey on Contextual Multi-armed Bandits`_, Li Zhou, Carnegie Mellon University, arXiv, 2016


.. _Contextual Multi-Armed Bandits: http://proceedings.mlr.press/v9/lu10a/lu10a.pdf
.. _A Survey on Contextual Multi-armed Bandits: https://arxiv.org/pdf/1508.03326.pdf
