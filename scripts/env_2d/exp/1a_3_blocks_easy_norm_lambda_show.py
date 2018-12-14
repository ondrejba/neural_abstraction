import os
import numpy as np
import utils
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("seaborn-colorblind")


LOAD_DIR = "results/env_2d"
LOAD_FILE = "1a_3_blocks_easy_0.5_norm_lambda.pickle"
LOAD_PATH = os.path.join(LOAD_DIR, LOAD_FILE)

NORMS = [0.5]
LAMBDAS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5]
RUNS = 20

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
