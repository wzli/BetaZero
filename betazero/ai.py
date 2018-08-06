import numpy as np
from .utils import *

class Agent:
    def __init__(self, game):
        self.game = game
        self.value_model = game.Model()
        self.state_history = []
        self.action_prediction_history = []

    def generate_predictions(self, state):
        action_predictions = list(zip(*self.game.generate_action_choices(state)))
        action_predictions.append([self.value_model.predict(self.game.input_transform(state_transition, False))
                if value is None else one_hot_pdf(value, self.game.max_value, self.game.output_dimension)
                        for _, state_transition, _, value, _ in zip(*action_predictions)])
        return action_predictions

    def generate_action(self, state = None):
        if state is None:
            predictions = self.action_prediction_history[-1]
        else:
            predictions = self.generate_predictions(state)
        actions, _, _, _, _, value_pdfs = predictions
        value_samples = np.array((self.game.sample_pdf(pdf) for pdf in value_pdfs))
        return actions[np.argmax(value_samples)]

    def min_max_propagate(self, value, reset_count):
        training_input_set = (self.game.input_transform(input_state)
                for input_state in self.state_history[-reset_count:])
        training_target_set = [one_hot_pdf(value, self.game.max_value, self.game.output_dimension)]
        for chosen_state, (_, state_transitions, state_transition_bytes, _, _, value_pdfs) in zip(
                reversed(self.state_history[-reset_count + 1:]),
                reversed(self.action_prediction_history[-reset_count + 1:]
                )):
            _, chosen_state_bytes = self.game.reduce_symetry(-chosen_state)
            action_index = state_transition_bytes.index(chosen_state_bytes)
            value_pdfs[action_index] = np.flip(training_target_set[-1])
            training_target_set.append(max_pdf(value_pdfs))
        training_target_set = reversed(training_target_set)
        return (training_input_set, training_target_set)

    def update_session(self, state, value, reset_count):
        self.state_history.append(state)
        if reset_count is not 0:
            if value is not None:
                self.inputs, self.pdfs = self.min_max_propagate(value, reset_count)
                #todo train model here
            self.state_history = self.state_history[:-reset_count]
            self.action_prediction_history = self.action_prediction_history[:-reset_count]
        self.action_prediction_history.append(self.generate_predictions(self.state_history[-1]))
