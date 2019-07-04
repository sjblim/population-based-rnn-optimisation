"""
particle.py


Created by limsi on 12/04/2019
"""

import datetime as dte
import numpy as np

class NeuroevolutionParticle:

    particles = []
    @classmethod
    def get_particle(cls, i):
        return cls.particles[i]

    @classmethod
    def update_particles(cls, particles):
        cls.particles = particles

    def __init__(self,
                 model,
                 handles,
                 batched_data):

        self.model = model
        self.handles = handles
        self.sess = model.sess
        self.batched_data = batched_data

    def update_data(self, batched_data):
        self.batched_data = batched_data

    def update_weights(self, rnn_weights):
        sess = self.sess
        weight_placeholders = self.handles['weight_placeholder']
        weight_update = self.handles['weight_assignment_ops']
        _ = sess.run(weight_update,
                     {weight_placeholders: rnn_weights})

    def compute_loss(self, verbose=False, additional_info=""):
        batched_data = self.batched_data
        handles = self.handles
        sess = self.sess
        state_initialiser = self.model.state_initialiser
        # Start doing feed-forward pass to evaluate
        placeholders = handles['data_placeholders']

        states = None
        total_loss = 0.0
        total_counts = 0
        try:

            for i, data_map in enumerate(batched_data):

                # Setup placeholders
                placeholder_mapping = {placeholders[k]: data_map[k]
                                       for k in placeholders}

                # Add initial states
                if states is None:
                    placeholder_mapping[handles['initial_states']] = state_initialiser.get_states(
                        state_indices=data_map['trajectory_index'])
                else:
                    placeholder_mapping[handles['initial_states']] = states

                # Actual loss function
                loss, counts, states \
                    = sess.run([handles['total_loss'],
                                handles['counts'],
                                handles['final_states']],
                               placeholder_mapping)

                total_loss += loss
                total_counts += counts

                valid_loss = total_loss / total_counts

                if verbose:
                    print("{} | {} | Evaluation {} of {}: Loss={} | {}".format(dte.datetime.now().strftime("%Y%m%d %H:%M:%S"),
                                                                          self.model.variable_scope_name,
                                                                          i,
                                                                          len(batched_data),
                                                                          valid_loss,
                                                                          additional_info))

                if np.isnan(valid_loss):
                    raise ValueError("NAN found!")

        except Exception as e:
            valid_loss = np.Inf
            print("Unstable particle found for {} particle {}".format(self.model.variable_scope_name, i))

        return valid_loss