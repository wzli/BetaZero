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
        # only for debug
        self.value_samples = None
        self.x_train = None
        self.y_train = None

    def generate_predictions(self, state):
        """from current state generate a tuple of:
        (actions, state_transitions, rewards, reset_counts, value_pdfs)
        and the action index lookup table keyed by bytes of the state transition
        """
        # expect state in self perspective if min-max
        action_space = self.game.generate_action_space(state)
        if not action_space:
            return None, None
        # create action index lookup
        state_byte_keys, action_predictions = zip(*action_space.items())
        action_index_lookup = {
            state_byte_key: i
            for i, state_byte_key in enumerate(state_byte_keys)
        }
        # create predictions
        action_predictions = list(zip(*action_predictions))
        # use model to predict the value pdf of each action in action space
        action_predictions.append(
            self.value_model.predict(
                np.vstack((self.game.input_transform(state_transition, False)
                           for state_transition in action_predictions[1]))))
        return (action_predictions, action_index_lookup)

    def generate_action(self, state=None):
        """generate an intelligent action given current state in perspective of action making player"""
        # based on prevously updated state by default
        predictions = self.generate_predictions(state)[
            0] if state else self.action_prediction_history[-1]
        # action space is empty
        if not predictions:
            return None
        actions, _, rewards, reset_counts, value_pdfs = predictions
        # Thompson Sampling based action selection:
        # Samples the predicted value distribution of each possible action
        # and take the max, if max has multiple, randomly choose one
        value_samples = np.array([
            np.random.choice(value_pdf.shape[0], p=value_pdf)
            if reset_count == 0 else value_to_index(
                reward / self.game.max_value, self.game.output_dimension)
            for value_pdf, reward, reset_count in zip(value_pdfs, rewards,
                                                      reset_counts)
        ])
        max_value_sample_index = np.random.choice(
            np.argwhere(value_samples == np.amax(value_samples)).flat)
        self.value_samples = value_samples
        return actions[max_value_sample_index]

    def generate_training_set(self, steps, terminal_state=True):
        """generate a training set by propagative rewards back in state history"""
        if steps < 1:
            raise ValueError("step number < 1")
        # generate input set based on recent history
        training_input_set = (self.game.input_transform(input_state)
                              for input_state in self.state_history[-steps:])
        # if already at terminal state, there is no future value only rewrad
        if terminal_state or not self.action_prediction_history[-1]:
            training_target_set = [
                one_hot_pdf(self.reward_history[-1] / self.game.max_value,
                            self.game.output_dimension)
            ]
        else:
            training_target_set = [
                shift_pdf(
                    max_pdf(self.action_prediction_history[-1][-1]),
                    self.reward_history[-1] / self.game.max_value)
            ]
        # if min_max, flip value_pdfs to other player's perspective
        if self.game.min_max:
            training_target_set[0] = np.flip(training_target_set[0], 0)
        for reward, action_index, action_predictions in zip(
                reversed(self.reward_history[-steps:-1]),
                reversed(self.action_selection_history[-steps + 1:]),
                reversed(self.action_prediction_history[-steps:-1])):
            action_predictions[-1][action_index] = training_target_set[-1]
            # shift pdf to add reward, max_pdf is the pdf equivalent of max()
            value_update = shift_pdf(
                max_pdf(action_predictions[-1]), reward / self.game.max_value)
            # if min_max, flip value_pdfs to other player's perspective
            if self.game.min_max:
                value_update = np.flip(value_update, 0)
            training_target_set.append(value_update)
        training_target_set = reversed(training_target_set)
        return np.vstack(training_input_set), np.vstack(training_target_set)

    def update_session(self, state, reward, reset_count):
        if reset_count < 0:
            raise ValueError("reset_count < 0")
        # associate new state with previously predictied action that caused the transition
        if self.action_index_lookup:
            _, state_bytes = self.game.reduce_symetry(state)
            self.action_selection_history.append(
                self.action_index_lookup[state_bytes])
        # save in history
        self.state_history.append(state)
        self.reward_history.append(-reward if self.game.min_max else reward)
        # generate predictions for possible actions in the new state
        # if min max, assumes input state is in opponent's perspective
        # need to change to current player's perspective as input to generate perdictions
        action_predictions, self.action_index_lookup = self.generate_predictions(
            -state if self.game.min_max else state)
        self.action_prediction_history.append(action_predictions)
        if reset_count != 0:
            # if the game resets, train the network
            self.x_train, self.y_train = self.generate_training_set(
                reset_count)
            self.value_model.fit(self.x_train, self.y_train, verbose=0)
            # discard history of the previous game after it's been trained
            self.state_history = self.state_history[:-reset_count]
            self.reward_history = self.reward_history[:-reset_count]
            self.action_selection_history = self.action_selection_history[:
                                                                          -reset_count]
            # re-predict actions for initial state after training
            self.action_prediction_history = self.action_prediction_history[:
                                                                            -reset_count
                                                                            -
                                                                            1]
            action_predictions, self.action_index_lookup = self.generate_predictions(
                self.state_history[-1])
            self.action_prediction_history.append(action_predictions)
        elif not self.action_prediction_history[-1]:
            raise ValueError("no more actions but game doesn't reset")
