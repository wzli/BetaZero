import numpy as np
from keras.models import load_model
from .utils import *

ACTIONS = 0
STATE_TRANSITIONS = 1
REWARDS = 2
RESET_COUNTS = 3
VALUE_PDFS = 4


class Agent:
    def __init__(self, game, model):
        self.game = game
        # get value model
        try:
            self.value_model = load_model(model)
        except OSError:
            print("Failed to load", model, "-> create new model")
            self.value_model = game.ValueModel()
        print(self.value_model.summary())

        # compute constants
        self.symetric_set_size = ((int(game.rotational_symetry) + 1) * (int(
            game.vertical_symetry) + 1) * (int(game.horizontal_symetry) + 1))

        self.value_range = np.linspace(
            -game.max_value, game.max_value,
            self.value_model.layers[-1].output.shape[-1])

        # history lists
        self.state_history = []
        self.reward_history = []
        self.action_prediction_history = []

        # only for debug
        self.value_samples = None
        self.x_train = None
        self.y_train = None

    def generate_predictions(self, state):
        """from current state generate a tuple of:
        (actions, state_transitions, rewards, reset_counts, value_pdfs)
        """
        # expect state in self perspective if min-max
        actions = self.game.get_actions(state)
        if not actions:
            return None
        # create predictions
        state_transitions, rewards, reset_counts = list(
            zip(*(self.game.predict_action(state, action)
                  for action in actions)))
        # use model to predict the value pdf of each action in action space
        value_pdfs = self.value_model.predict(
            np.vstack((np.rollaxis(state_transition.array(), 1, 4)
                       for state_transition in state_transitions)))
        return actions, state_transitions, rewards, reset_counts, value_pdfs

    def generate_action(self, explore=True, state=None):
        """generate an intelligent action given current state in perspective of action making player"""
        # based on prevously updated state by default
        predictions = self.generate_predictions(
            state) if state else self.action_prediction_history[-1]
        # action space is empty
        if not predictions:
            return None
        actions, _, rewards, reset_counts, value_pdfs = predictions
        # if explore, Thompson Sampling based action selection:
        # Samples the predicted value distribution of each possible action
        # othersize, don't sample, just take the expected value
        if explore:
            value_sample = lambda value_pdf: np.random.choice(self.value_range, p=value_pdf)
        else:
            value_sample = lambda value_pdf: np.average(self.value_range, weights=value_pdf)
        value_samples = np.array([
            reward + value_sample(value_pdf) if reset_count == 0 else
            reward for value_pdf, reward, reset_count in zip(
                value_pdfs, rewards, reset_counts)
        ])
        # take the max, if max has multiple, randomly choose one
        max_value_index = np.random.choice(
            np.argwhere(value_samples == np.amax(value_samples)).flat)
        self.value_samples = value_samples
        return actions[max_value_index]

    def generate_training_set(self, steps, terminal_state=True):
        """generate a training set by propagative rewards back in state history"""
        if steps < 1:
            raise ValueError("step number < 1")
        elif steps > len(self.state_history) - 1:
            steps = len(self.state_history) - 1
        # generate input set based on recent history
        training_input_set = np.vstack(
            input_state.array() for input_state in self.state_history[-steps:])
        self.x_train = self.state_history[-steps:]
        # generate symetric input arrays
        training_input_set = np.vstack([
            np.rollaxis(array, 1, 4) for array in symetric_arrays(
                training_input_set, self.game.rotational_symetry,
                self.game.vertical_symetry, self.game.horizontal_symetry)
        ])
        # if already at terminal state, there is no future advantage
        if terminal_state or not self.action_prediction_history[-1]:
            training_target_set = [
                one_hot_pdf(0,
                            int(self.value_model.layers[-1].output.shape[-1]))
            ]
        else:
            training_target_set = [
                max_pdf(self.action_prediction_history[-1][VALUE_PDFS])
            ]
        # if min_max, flip value_pdfs to other player's perspective
        if self.game.min_max:
            training_target_set[0] = np.flip(training_target_set[0], 0)
        for chosen_state, reward, action_predictions in zip(
                reversed(self.state_history[-steps + 1:]),
                reversed(self.reward_history[-steps + 1:]),
                reversed(self.action_prediction_history[-steps:-1])):
            action_index = [
                state_transition.key()
                for state_transition in action_predictions[STATE_TRANSITIONS]
            ].index(chosen_state.key())
            action_predictions[VALUE_PDFS][action_index] = shift_pdf(
                training_target_set[-1], reward / self.game.max_value)
            # max_pdf is the pdf equivalent of max()
            value_update = max_pdf(action_predictions[VALUE_PDFS])
            # if min_max, flip value_pdfs to other player's perspective
            if self.game.min_max:
                value_update = np.flip(value_update, 0)
            training_target_set.append(value_update)
        training_target_set = list(reversed(training_target_set))
        self.y_train = training_target_set
        # expand target set to match input set
        training_target_set = np.vstack(
            training_target_set * self.symetric_set_size)
        return training_input_set, training_target_set

    def update_session(self, state, reward, reset_count):
        if reset_count < 0:
            raise ValueError("reset_count < 0")
        # save in history
        self.state_history.append(state)
        self.reward_history.append(reward)
        # generate predictions for possible actions in the new state
        # if min max, assumes input state is in opponent's perspective
        # need to change to current player's perspective as input to generate perdictions
        action_predictions = self.generate_predictions(
            state.flip() if self.game.min_max else state)
        self.action_prediction_history.append(action_predictions)
        if reset_count != 0:
            # if the game resets, train the network
            training_set = self.generate_training_set(reset_count,
                                                      self.game.terminal_state)
            self.value_model.fit(*training_set, verbose=0)
            # discard history of the previous game after it's been trained
            self.state_history = self.state_history[:-reset_count]
            self.reward_history = self.reward_history[:-reset_count]
            self.action_prediction_history = self.action_prediction_history[:
                                                                            -reset_count
                                                                            -
                                                                            1]
            # re-predict actions for initial state after training
            action_predictions = self.generate_predictions(self.state_history[
                -1].flip() if self.game.min_max else self.state_history[-1])
            self.action_prediction_history.append(action_predictions)
        elif reward != 0 and self.game.reward_span > 1:
            # reward received, train the network
            training_set = self.generate_training_set(self.game.reward_span,
                                                      False)
            self.value_model.fit(*training_set, verbose=0)
        elif not self.action_prediction_history[-1]:
            raise ValueError("no more actions but game doesn't reset")
