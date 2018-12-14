import os
import numpy as np
import agent_utils
import constants
import utils
from envs.env_2d import Env2D
from nets.fully_connected import FullyConnected


os.environ["CUDA_VISIBLE_DEVICES"] = ""

ENV_NAME = constants.ENV_1A_MEDIUM
NUM_TRAIN = 1000
NUM_TEST = 1000
Z_HIDDENS = [20, 20]
T_HIDDENS = []
R_HIDDENS = []
A_HIDDENS = None
LEARNING_RATE = 0.01
Z_SIZE = 4
BATCH_SIZE = 32
TRAIN_STEPS = 250

NORMS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
LAMBDAS = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
RUNS = 20

SAVE_DIR = "results/env_2d"
utils.maybe_makedirs(SAVE_DIR)
SAVE_FILE = "1a_medium_norm_lambda.pickle"
SAVE_PATH = os.path.join(SAVE_DIR, SAVE_FILE)

results = utils.maybe_read_pickle(SAVE_PATH)

for norm in NORMS:

    for lambda_1 in LAMBDAS:

        for run in range(RUNS):

            key = (norm, lambda_1, run)

            if key not in results:

                # collect experience
                env = Env2D(constants.env_to_path(ENV_NAME))

                train_exp = agent_utils.collect_exp(env, NUM_TRAIN)
                test_exp = agent_utils.collect_exp(env, NUM_TEST)

                # set up and train a neural network
                net = FullyConnected(
                    2, len(env.actions), Z_HIDDENS, T_HIDDENS, R_HIDDENS, a_hiddens=A_HIDDENS,
                    learning_rate=LEARNING_RATE, z_size=Z_SIZE, norm=norm, lambda_1=lambda_1
                )
                net.state_session()

                losses = agent_utils.train(
                    net, TRAIN_STEPS, BATCH_SIZE, train_exp[0], train_exp[1], train_exp[2], train_exp[3], train_exp[4]
                )

                zs = net.session.run(net.z_t, feed_dict={net.state_pl: test_exp[0]})

                # calculate overlap
                predictions = np.argmax(zs, axis=1)
                hits, total = agent_utils.overlap(predictions, test_exp[5])

                overlap = hits / total

                # save results
                results[key] = (overlap, losses["transition"], losses["reward"])
                utils.write_pickle(SAVE_PATH, results)
                print("done")

                # clean up
                net.stop_session()
