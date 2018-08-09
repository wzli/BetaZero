import numpy as np
from keras.models import load_model
from .utils import *

class Agent:
    def __init__(self, game, model):
        self.game = game
        try:
            self.value_model = load_model(model)
        except OSError:
            print("Failed to load", model, "-> create new model")
            self.value_model = game.ValueModel()
        self.state_history = []
        self.reward_history = []
        self.action_prediction_history = []
        self.x_train = None
        self.y_train = None

    def generate_predictions(self, state):
        action_predictions = list(zip(*self.game.generate_action_choices(state)))
        if not action_predictions:
            return [()] * 6
        _, state_transitions, _, _, _ = action_predictions
        if self.game.min_max:
            state_transitions = [-state_transition for state_transition in state_transitions]
        action_predictions.append(self.value_model.predict(np.vstack(
                [self.game.input_transform(state_transition, False) for state_transition in state_transitions])))
        return action_predictions

    def generate_action(self, state = None):
        if state:
            predictions = self.generate_predictions(state)
        else:
            predictions = self.action_prediction_history[-1]
        actions, _, _, rewards, reset_counts, value_pdfs = predictions
        value_samples = np.array([np.random.choice(value_pdf.shape[0], p=value_pdf)
                if reset_count == 0 else value_to_index(reward/self.game.max_value, self.game.output_dimension)
                        for value_pdf, reward, reset_count in zip(value_pdfs, rewards, reset_counts)])
        if value_samples.shape[0] == 0:
            return None
        max_value_sample_indexes = np.argwhere(value_samples == np.amax(value_samples)).flat
        return actions[np.random.choice(max_value_sample_indexes)]

    def generate_training_set(self, steps, terminal_state = True):
        if steps < 1:
            raise ValueError("step number < 1")
        training_input_set = (self.game.input_transform(input_state)
                for input_state in self.state_history[-steps:])
        _, _, _, _, _, end_value_pdfs = self.action_prediction_history[-1]
        if terminal_state or not end_value_pdfs:
            training_target_set = [one_hot_pdf(self.reward_history[-1] / self.game.max_value,
                    self.game.output_dimension)]
        else:
            training_target_set = [shift_pdf(max_pdf(end_value_pdfs),
                    self.reward_history[-1] / self.game.max_value)]
        if self.game.min_max:
            training_target_set[0] = np.flip(training_target_set[0], axis=0)
        for chosen_state, reward, (_, state_transitions, state_transition_bytes, _, _, value_pdfs) in zip(
                reversed(self.state_history[-steps+1:]),
                reversed(self.reward_history[-steps:-1]),
                reversed(self.action_prediction_history[-steps:-1])):
            if not state_transitions:
                raise ValueError("action prediction history has missing links")
            _, chosen_state_bytes = self.game.reduce_symetry(chosen_state)
            action_index = state_transition_bytes.index(chosen_state_bytes)
            value_pdfs[action_index] = training_target_set[-1]
            value_update = shift_pdf(max_pdf(value_pdfs), reward / self.game.max_value)
            if self.game.min_max:
                value_update = np.flip(value_update, axis=0)
            training_target_set.append(value_update)
        training_target_set = reversed(training_target_set)
        return (np.vstack(training_input_set), np.vstack(training_target_set))

    def update_session(self, state, reward, reset_count):
        if reset_count < 0:
            raise ValueError("reset_count < 0")
        if self.game.min_max:
            state = -state
            reward = -reward
        self.state_history.append(state)
        self.reward_history.append(reward)
        self.action_prediction_history.append(self.generate_predictions(state))
        if reset_count != 0:
            self.x_train, self.y_train = self.generate_training_set(reset_count)
            self.value_model.fit(self.x_train, self.y_train, verbose=0)
            self.state_history = self.state_history[:-reset_count]
            self.reward_history = self.reward_history[:-reset_count]
            self.action_prediction_history = self.action_prediction_history[:-reset_count-1]
            self.action_prediction_history.append(self.generate_predictions(self.state_history[-1]))
        elif not self.action_prediction_history[-1][0]:
            raise ValueError("no more actions but game doesn't reset")
        elif reward != 0:
            # train more
            #self.propagate_reward(min(3, len(self.state_history)), False)
            pass
