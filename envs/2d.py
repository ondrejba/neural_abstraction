import numpy as np
import matplotlib.pyplot as plt
import utils
plt.style.use("seaborn-colorblind")


class Env2D:

    def __init__(self, json_path):

        json = utils.load_json(json_path)

        self.transitions = None
        self.parse_transitions(json["transitions"])

        self.rewards = None
        self.parse_rewards(json["rewards"])

        self.means = None
        self.parse_means(json["means"])

        self.var = json["var"]
        self.cov = [[self.var, 0], [0, self.var]]

    def parse_transitions(self, obj):

        self.transitions = {}

        for key, value in obj.items():

            state = int(key.split(",")[0])
            action = int(key.split(",")[1])
            next_state = int(value)

            self.transitions[(state, action)] = next_state

    def parse_rewards(self, obj):

        self.rewards = {}

        for value in obj:

            state = int(value.split(",")[0])
            action = int(value.split(",")[1])

            self.rewards[(state, action)] = 1

    def parse_means(self, obj):

        self.means = {}

        for key, value in obj.items():

            state = int(key)
            self.means[state] = value

    def plot(self):

        for state, means in self.means.items():

            state_samples = []

            for mean in means:
                state_samples.append(np.random.multivariate_normal(mean, self.cov, size=100))

            state_samples = np.concatenate(state_samples, axis=0)

            plt.scatter(state_samples[:, 0], state_samples[:, 1], label="s{:d}".format(state))

        plt.legend()
        plt.show()
