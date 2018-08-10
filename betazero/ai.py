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
        self.action_selection_history = []
        self.action_index_lookup = None
        self.value_samples = None
        self.x_train = None
        self.y_train = None

    def generate_predictions(self, state):
        action_choices = self.game.generate_action_choices(state)
        if not action_choices:
            return None, None
        state_byte_keys, action_predictions = zip(*action_choices.items())
        action_index_lookup = {state_byte_key:i for i, state_byte_key in enumerate(state_byte_keys)}
        action_predictions = list(zip(*action_predictions))
        action_predictions.append(self.value_model.predict(np.vstack(
                (self.game.input_transform(state_transition, False) for state_transition in action_predictions[1]))))
        return (action_predictions, action_index_lookup)

    def generate_action(self, state = None):
        predictions = self.generate_predictions(state)[0] if state else self.action_prediction_history[-1]
        if not predictions:
            return None
        actions, _, rewards, reset_counts, value_pdfs = predictions
        value_samples = np.array([np.random.choice(value_pdf.shape[0], p=value_pdf)
                if reset_count == 0 else value_to_index(reward/self.game.max_value, self.game.output_dimension)
                        for value_pdf, reward, reset_count in zip(value_pdfs, rewards, reset_counts)])
        max_value_sample_index = np.random.choice(np.argwhere(value_samples == np.amax(value_samples)).flat)
        self.value_samples = value_samples
        return actions[max_value_sample_index]

    def generate_training_set(self, steps, terminal_state = True):
        if steps < 1:
            raise ValueError("step number < 1")
        training_input_set = (self.game.input_transform(input_state)
                for input_state in self.state_history[-steps:])
        if terminal_state or not self.action_prediction_history[-1]:
            training_target_set = [one_hot_pdf(self.reward_history[-1] / self.game.max_value,
                    self.game.output_dimension)]
        else:
            training_target_set = [shift_pdf(max_pdf(self.action_prediction_history[-1][-1]),
                    self.reward_history[-1] / self.game.max_value)]
        if self.game.min_max:
            training_target_set[0] = np.flip(training_target_set[0], 0)
        for reward, action_index, action_predictions in zip(
                reversed(self.reward_history[-steps:-1]),
                reversed(self.action_selection_history[-steps+1:]),
                reversed(self.action_prediction_history[-steps:-1])):
            action_predictions[-1][action_index] = training_target_set[-1]
            value_update = shift_pdf(max_pdf(action_predictions[-1]), reward / self.game.max_value)
            if self.game.min_max:
                value_update = np.flip(value_update, 0)
            training_target_set.append(value_update)
        training_target_set = reversed(training_target_set)
        return np.vstack(training_input_set), np.vstack(training_target_set)

    def update_session(self, state, reward, reset_count):
        if reset_count < 0:
            raise ValueError("reset_count < 0")
        if self.action_index_lookup:
            _, state_bytes = self.game.reduce_symetry(state)
            self.action_selection_history.append(self.action_index_lookup[state_bytes])
        self.state_history.append(state)
        self.reward_history.append(-reward if self.game.min_max else reward)
        action_predictions, self.action_index_lookup = self.generate_predictions(-state if self.game.min_max else state)
        self.action_prediction_history.append(action_predictions)
        if reset_count != 0:
            self.x_train, self.y_train = self.generate_training_set(reset_count)
            self.value_model.fit(self.x_train, self.y_train, verbose=0)
            self.state_history = self.state_history[:-reset_count]
            self.reward_history = self.reward_history[:-reset_count]
            self.action_selection_history = self.action_selection_history[:-reset_count]
            self.action_prediction_history = self.action_prediction_history[:-reset_count-1]
            action_predictions, self.action_index_lookup = self.generate_predictions(self.state_history[-1])
            self.action_prediction_history.append(action_predictions)
        elif not self.action_prediction_history[-1]:
            raise ValueError("no more actions but game doesn't reset")
