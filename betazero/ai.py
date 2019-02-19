import threading, queue, os, time
import numpy as np
from .utils import *

#tempoary fix for:
#https://github.com/tensorflow/tensorflow/issues/14048
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


ACTIONS = 0
STATE_TRANSITIONS = 1
REWARDS = 2
RESET_COUNTS = 3
VALUE_PDFS = 4


class Agent:
    def __init__(self,
                 game,
                 name='agent',
                 model_path='model.h5',
                 save_interval=0,
                 save_dir='.'):
        self.game = game
        self.name = name
        self.model_path = model_path
        self.save_interval = save_interval
        self.save_time = time.time()
        self.save_counter = 0
        self.total_moves = 0
        self.total_rewards = 0

        # get value model
        from keras.models import load_model
        # if not training, bubble up error
        if self.save_interval <= 0:
            self.value_model = load_model(model_path, compile=False)
        else:
            try:
                self.value_model = load_model(model_path)
            except OSError as e:
                # create blank model
                print(e, "\nFailed to load", model_path, "-> create new model")
                self.value_model = game.ValueModel()
            print(self.value_model.summary())

        self.model_save_dir = os.path.join(
            save_dir, os.path.basename(self.model_path)) + ".save"

        # compute constants
        self.symetric_set_size = ((int(game.rotational_symetry) + 1) *
                                  (int(game.vertical_symetry) + 1) *
                                  (int(game.horizontal_symetry) + 1))

        self.value_range = np.linspace(
            -game.max_value, game.max_value,
            self.value_model.layers[-1].output.shape[-1])

        # history lists
        self.state_history = []
        self.action_prediction_history = []

        # continue below only if training is required
        if self.save_interval <= 0:
            return

        # setup tensorboard callback
        from keras.callbacks import TensorBoard
        self.training_callbacks = [
            TensorBoard(
                log_dir=self.model_save_dir,
                histogram_freq=1,
                write_graph=False,
                write_grads=True,
                write_images=True)
        ]

        # train on a seperate thread
        self.value_model._make_train_function()
        self.value_model._make_test_function()
        self.training_queue = queue.Queue(10)
        self.training_thread = threading.Thread(target=self.training_loop)
        self.training_thread.daemon = True
        self.training_thread.start()

    def training_loop(self):
        # main loop for training thread
        while True:
            training_set, original_training_set = self.training_queue.get()
            self.value_model.fit(*training_set, verbose=0)
            n_moves = len(original_training_set[0])
            self.total_moves += n_moves
            self.save_counter += n_moves
            self.training_queue.task_done()

    def queue_training_set(self, training_set):
        while True:
            try:
                self.training_queue.put(training_set, timeout=5)
                break
            except queue.Full:
                print("Queue Full at ", self.total_moves)

    def save_model(self, training_set, original_training_set):
        # create save directory if doesn't exist
        if not os.path.exists(self.model_save_dir):
            print("Save directory", self.model_save_dir,
                  "doesn't exist -> create new folder")
            os.makedirs(self.model_save_dir)
        # reset save counter
        self.save_counter = 0
        # use keras tensorboard callback for logging
        self.value_model.fit(
            *training_set,
            verbose=0,
            validation_split=0.99,
            callbacks=self.training_callbacks)
        # print most recent game history
        print(self.name, "model saved at move", self.total_moves)
        print("reward/move", self.total_rewards / self.total_moves)
        print("time elapsed", time.time() - self.save_time, "seconds")
        self.save_time = time.time()
        self.value_model.save(
            os.path.join(self.model_save_dir,
                         "model_" + str(int(self.save_time)) + '.h5'))
        self.value_model.save(self.model_path)
        for i, (x, y) in enumerate(zip(*original_training_set)):
            expected, variance = expected_value(y, self.value_range, True)
            expected = round(expected, 3)
            deviation = round(variance**0.5, 3)
            print(x, "\nexpected value", expected, "deviation", deviation,
                  "step", i + 1, "\n")

    def generate_predictions(self, state):
        """from current state generate a tuple of:
        (actions, state_transitions, rewards, reset_counts, value_pdfs)
        """
        # expect state in self perspective
        actions = self.game.get_actions(state)
        if not actions:
            return None
        # create predictions
        state_transitions, rewards, reset_counts = list(
            zip(*(self.game.predict_action(state, action)
                  for action in actions)))
        # generate input arrays, usually this takes high CPU so parallelize
        input_arrays = [
            state_transition.array() for state_transition in state_transitions
        ]
        # use model to predict the value pdf of each action in action space
        value_pdfs = self.value_model.predict(
            np.vstack(input_arrays), batch_size=len(input_arrays))
        return actions, state_transitions, rewards, reset_counts, value_pdfs

    def generate_action(self, state=None, explore=True, verbose=False):
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
        action = actions[np.random.choice(
            np.argwhere(value_samples == np.amax(value_samples)).flat)]
        # print some debug info
        if verbose:
            for action_choice, _, action_reward, _, value_pdf, value_sample in sorted(
                    zip(*predictions, value_samples), key=lambda x: x[-1]):
                expected, variance = expected_value(value_pdf,
                                                    self.value_range, True)
                expected = round(expected, 3)
                deviation = round(variance**0.5, 3)
                print('ACT:', action_choice, '\tRWD:', action_reward, ""
                      if not explore else
                      '\t\tSMP ' + str(round(value_sample, 3)), '\tEXP',
                      expected, '\tSTD:', deviation)
            print(self.name, "played", action)
        return action

    def generate_training_set(self, steps, terminal_state=True):
        """generate a training set by propagative rewards back in state history"""
        if steps < 1:
            raise ValueError(self.name + ": step number < 1")
        elif steps > len(self.state_history) - 1:
            steps = len(self.state_history) - 1
        # generate input set based on recent history
        original_input_set = self.state_history[-steps:]
        training_input_set = np.vstack(
            [input_state.array() for input_state in original_input_set])
        # generate symetric input arrays
        training_input_set = np.vstack(
            symetric_arrays(training_input_set, self.game.rotational_symetry,
                            self.game.vertical_symetry,
                            self.game.horizontal_symetry))

        def get_value_update(predictions):
            value_update = max_pdf([
                shift_pdf(value_pdf, reward / self.game.max_value)
                for value_pdf, reward in zip(predictions[VALUE_PDFS],
                                             predictions[REWARDS])
            ])
            # if min_max, flip value_pdfs to other player's perspective
            return np.flip(value_update,
                           0) if self.game.min_max else value_update

        # if already at terminal state, there is no future advantage
        if terminal_state or not self.action_prediction_history[-1]:
            original_target_set = [
                one_hot_pdf(0,
                            int(self.value_model.layers[-1].output.shape[-1]))
            ]
        else:
            original_target_set = [
                get_value_update(self.action_prediction_history[-1])
            ]
        for chosen_state, action_predictions in zip(
                reversed(self.state_history[-steps + 1:]),
                reversed(self.action_prediction_history[-steps:-1])):
            # reverse search action index from resultant state
            action_index = [
                state_transition.key()
                for state_transition in action_predictions[STATE_TRANSITIONS]
            ].index(chosen_state.key())
            # back propagate value from child state
            action_predictions[VALUE_PDFS][action_index] = original_target_set[
                -1]
            # adjust value distribution to reward and take max
            original_target_set.append(get_value_update(action_predictions))
        original_target_set = list(reversed(original_target_set))
        # expand target set to match input set
        training_target_set = np.vstack(
            original_target_set * self.symetric_set_size)
        return (training_input_set, training_target_set), (original_input_set,
                                                           original_target_set)

    def update_session(self, state, reward, reset_count):
        if reset_count < 0:
            raise ValueError(self.name + ": reset_count < 0")
        # save in history
        self.total_rewards += reward
        self.state_history.append(state)
        # generate predictions for possible actions in the new state
        # if min max, assumes input state is in opponent's perspective
        # need to change to current player's perspective as input to generate perdictions
        action_predictions = self.generate_predictions(
            state.flip() if self.game.min_max else state)
        self.action_prediction_history.append(action_predictions)
        train = self.save_interval > 0
        if reset_count != 0:
            if train:
                # if the game resets, train the network
                training_set = self.generate_training_set(
                    reset_count, self.game.terminal_state)
                # save model if counter is reached
                if self.save_counter < self.save_interval:
                    self.queue_training_set(training_set)
                else:
                    self.save_model(*training_set)
            # discard history of the previous game after it's been trained
            self.state_history = self.state_history[:-reset_count]
            self.action_prediction_history = self.action_prediction_history[:
                                                                            -reset_count
                                                                            - 1]
            # re-predict actions for initial state after training
            action_predictions = self.generate_predictions(
                self.state_history[-1].flip()
                if self.game.min_max else self.state_history[-1])
            self.action_prediction_history.append(action_predictions)
        elif train and reward != 0 and self.game.reward_span > 1:
            # reward received, train the network
            training_set = self.generate_training_set(self.game.reward_span,
                                                      False)
            self.queue_training_set(training_set)
        elif not self.action_prediction_history[-1]:
            raise ValueError(
                self.name + ": no more actions but game doesn't reset")
