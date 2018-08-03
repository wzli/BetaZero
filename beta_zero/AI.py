import numpy as np

class Agent:
    def __init__(self, Game):
        self.Game = Game
        self.value_model = Game.Model()
        self.state_history = []

    def update_session(self, state, value, reset_count):
        self.state_history.append(state)

        if value:
            if reset_count > 0:
                training_input_set = np.array([self.Game.input_transform(input_state)
                        for input_state in self.state_history[-reset_count:]])
                self.state_history = self.state_history[:-reset_count]

                #print(training_input_set)
                # learn based on end value

                #get predictions values
                #mix max propagate
                #train model


    def generate_action(self, state = None):
        if not state:
            state = self.state_history[-1]
        predictions = self.Game.generate_action_inputs(state)
        value_distributions = np.array([
                np.array([value, 0]) if value else self.value_model.predict(state_transition)
                        for _, state_transition, value, _ in predictions])
        return predictions[np.argmax(np.random.normal(value_distributions[:,0], value_distributions[:,1]))][0]
