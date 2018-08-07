import numpy as np
from .utils import *

class Agent:
    def __init__(self, game):
        self.game = game
        self.value_model = game.Model()
        self.state_history = []
        self.reward_history = []
        self.action_prediction_history = []

    def generate_predictions(self, state):
        action_predictions = list(zip(*self.game.generate_action_choices(state)))
        _, state_transitions, _, _, _ = action_predictions
        action_predictions.append([self.value_model.predict(self.game.input_transform(state_transition, False))
                        for state_transition in state_transitions])
        return action_predictions

    def generate_action(self, state = None):
        if state is None:
            predictions = self.action_prediction_history[-1]
        else:
            predictions = self.generate_predictions(state)
        actions, _, _, rewards, reset_counts, value_pdfs = prediction
        value_samples = np.array((np.random.choice(pdf.shape, p=pdf) if reset_count is 0 else reward
                for value_pdf, reward, reset_count in zip(value_pdfs, rewards, reset_counts)))
        return actions[np.argmax(value_samples)]

    def propagate_reward(self, steps, terminal_state = True):
        training_input_set = (self.game.input_transform(input_state)
                for input_state in self.state_history[-steps:])
        _, _, _, _, _, end_value_pdfs = self.action_prediction_history[-1]
        if terminal_state:
            training_target_set = [one_hot_pdf(self.reward_history[-1] / self.game.max_value,
                    self.game.output_dimension)]
        else:
            training_target_set = [shift_pdf(max_pdf(end_value_pdfs),
                    self.reward_history[-1] / self.game.max_value)]
        for chosen_state, reward, (_, state_transitions, state_transition_bytes, _, _, value_pdfs) in zip(
                reversed(self.state_history[-steps+1:]),
                reversed(self.reward_history[-steps:-1]),
                reversed(self.action_prediction_history[-steps:-1])):
            if self.game.min_max:
                _, chosen_state_bytes = self.game.reduce_symetry(-chosen_state)
                action_index = state_transition_bytes.index(chosen_state_bytes)
                value_pdfs[action_index] = np.flip(training_target_set[-1])
            else:
                _, chosen_state_bytes = self.game.reduce_symetry(chosen_state)
                action_index = state_transition_bytes.index(chosen_state_bytes)
                value_pdfs[action_index] = training_target_set[-1]
            value_update = shift_pdf(max_pdf(value_pdfs), reward / self.game.max_value)
            training_target_set.append(value_update)
        training_target_set = reversed(training_target_set)
        return (training_input_set, training_target_set)

    def update_session(self, state, reward, reset_count):
        self.state_history.append(state)
        self.reward_history.append(reward)
        self.action_prediction_history.append(self.generate_predictions(state))
        if reset_count is not 0:
            self.inputs, self.pdfs = self.propagate_reward(reset_count)
                #todo train model here
            self.state_history = self.state_history[:-reset_count]
            self.reward_history = self.reward_history[:-reset_count]
            self.action_prediction_history = self.action_prediction_history[:-reset_count-1]
            self.action_prediction_history.append(self.generate_predictions(self.state_history[-1]))
        elif reward is not 0:
            #this is subjective
            self.propagate_reward(min(3, len(self.state_history)), False)
            # train more
