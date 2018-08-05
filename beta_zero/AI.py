import numpy as np

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def max_pdf(pdfs):
    pdfs = np.asarray(pdfs)
    cdfs = np.zeros((pdfs.shape[0], pdfs.shape[1] + 1))
    np.cumsum(pdfs, axis=1, out=cdfs[:,1:])
    max_cdf = np.prod(cdfs, axis=0)
    max_pdf = np.diff(max_cdf)
    max_pdf = max_pdf/np.sum(max_pdf)
    return max_pdf


class Agent:
    def __init__(self, game):
        self.game = game
        self.value_model = game.Model()
        self.state_history = []
        self.action_prediction_history = [None]

    def generate_predictions(self, state):
        action_predictions = list(zip(*self.game.generate_action_choices(state)))
        action_predictions.append([self.value_model.predict(self.game.input_transform(state_transition, False))
                if value is None else value for _, state_transition, value, _ in zip(*action_predictions)])
        return action_predictions

    def generate_action(self, state = None):
        if state is None:
            predictions = self.action_prediction_history[-1]
        else:
            predictions = self.generate_predictions(state)
        actions, _, _, _, value_pdfs = predictions
        value_samples = np.array((self.game.sample_pdf(pdf) for pdf in value_pdfs))
        return actions[np.argmax(value_samples)]

    def update_session(self, state, value_pdf, reset_count):
        self.state_history.append(state)
        if reset_count is 0:
            self.action_prediction_history.append(self.generate_predictions(state))
        else:
            if value_pdf is not None:
                training_input_set = [self.game.input_transform(input_state)
                        for input_state in self.state_history[-reset_count:]]
                training_target_set = [value_pdf]
                for chosen_state, (_, state_transitions, _, _, value_pdfs) in zip(
                        reversed(self.state_history[-reset_count:]),
                        reversed(self.action_prediction_history[-reset_count:]
                        )):
                    _, chosen_state_bytes = self.game.reduce_symetry(-chosen_state)
                    state_transition_bytes = [state_transition.tobytes()
                            for state_transition in state_transitions]
                    action_index = state_transition_bytes.index(chosen_state_bytes)
                    value_pdfs[action_index] = np.flip(training_target_set[-1])
                    training_target_set.append(max_pdf(value_pdfs))
                training_target_set = reversed(training_target_set)
                #TODO train
                self.pdfs = training_target_set
                self.state_history = self.state_history[:-reset_count]
                self.action_prediction_history = self.action_prediction_history[:-reset_count+1]
