import os
import numpy as np
import agent_utils
import constants
import utils
import matplotlib.pyplot as plt
plt.style.use("seaborn-colorblind")
import seaborn as sns


LOAD_DIR = "results/env_2d"
LOAD_FILE = "1a_easy_norm_lambda.pickle"
LOAD_PATH = os.path.join(LOAD_DIR, LOAD_FILE)

NORMS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
LAMBDAS = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
RUNS = 100

results = utils.read_pickle(LOAD_PATH)
results_array = np.zeros((len(NORMS), len(LAMBDAS)))

for i, norm in enumerate(NORMS):

    for j, lambda_1 in enumerate(LAMBDAS):

        accuracies = []

        for run_idx in range(RUNS):

            key = (norm, lambda_1, run_idx)

            if key in results:

                accuracies.append(results[key][0])

        if len(accuracies) > 0:
            mean_accuracy = np.mean(accuracies)
            results_array[i, j] = mean_accuracy

sns.heatmap(results_array, xticklabels=LAMBDAS, yticklabels=NORMS, annot=True,
            cbar=False)
plt.xlabel("lambda")
plt.ylabel("norm")
plt.show()
