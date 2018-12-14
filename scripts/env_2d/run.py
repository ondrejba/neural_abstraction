import argparse
import agent_utils
import constants
from envs.env_2d import Env2D
from nets.fully_connected import FullyConnected


def main(args):

    # set up an environment and collect experience
    env = Env2D(constants.env_to_path(args.env_name))

    train_exp = agent_utils.collect_exp(env, args.num_train)
    test_exp = agent_utils.collect_exp(env, args.num_test)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("env_name", help=", ".join(constants.ENVS))

    parser.add_argument("--num-train", default=1000, help="number of samples for training")
    parser.add_argument("--num-test", default=1000, help="number of samples for testing")

    parsed = parser.parse_args()
    main(parsed)
