"""
strategies.py


Created by limsi on 12/04/2019
"""

import numpy as np


class BasicEvolutionStrategy:

    def __init__(self, params, w_0):
        self.alpha = params['learning_rate']
        self.sigma = params['sigma']
        self.num_models = params['population_size']

        self.population_weight = w_0  # initialise weights
        self.local_adjustments = 0

    @property
    def num_weights(self):
        return self.population_weight.shape[0]

    def mutate(self):

        w = self.population_weight

        perturbations = np.random.standard_normal([self.num_models, self.num_weights])
        particle_weights = w + self.sigma * perturbations

        self.local_adjustments = perturbations

        return particle_weights

    def update(self, rewards):

        perturbations = self.local_adjustments

        # Filter out valid reward values
        valid_locations = np.isfinite(rewards)

        rewards = rewards.copy()[valid_locations]
        stacked_rewards = np.tile(rewards[:, np.newaxis], (1, self.num_weights))

        # Update population average
        weighed_perturbations = stacked_rewards * perturbations[valid_locations, :]

        self.population_weight = self.population_weight \
                                 + self.alpha / self.sigma * weighed_perturbations.mean(axis=0)


class ParticleSwarmOptimisationStrategy(BasicEvolutionStrategy):

    def __init__(self, params, w_0):
        self.W = params['dropout_rate']  # learning rate should be < 1
        self.sigma_v = 0.0
        self.sigma_w = params['w_sigma']
        self.c1 = 2
        self.c2 = 2
        self.num_models = params['population_size']

        # Particle swarm optimisation required variables
        self.population_weight = np.zeros(w_0.shape) #np.random.randn(w_0.shape[0])  # Global best
        self.population_best_reward = -np.inf
        self.best_rewards_per_traj = np.zeros(self.num_models) - np.inf
        self.local_adjustments = self.sigma_w * np.random.randn(self.num_models, self.num_weights)
        self.best_trajectory_weights = self.local_adjustments.copy()
        self.velocities = self.sigma_v * np.random.randn(self.num_models, self.num_weights)

    def mutate(self):

        # Update velocities
        self.velocities = self.W * self.velocities \
                    + self.c1*np.random.rand(self.num_models, 1) * (self.population_weight - self.local_adjustments) \
                    + self.c2*np.random.rand(self.num_models, 1)*(self.best_trajectory_weights - self.local_adjustments)

        self.local_adjustments = self.local_adjustments + self.velocities

        return self.local_adjustments

    def update(self, rewards):

        # Update local weights
        trajectory_weights_to_update = [i for i in range(rewards.shape[0])
                                        if rewards[i] > self.best_rewards_per_traj[i]]

        for i in trajectory_weights_to_update:
            self.best_rewards_per_traj[i] = rewards[i]
            self.best_trajectory_weights[i, :] = self.local_adjustments[i, :]

        # Update global weights
        best_reward = rewards.max()
        if best_reward > self.population_best_reward:
            self.population_best_reward = best_reward
            best_index = np.where(rewards == best_reward)[0][0]  # only need one
            self.population_weight = self.best_trajectory_weights[best_index, :]
