import argparse
import agent_utils
import constants
from envs.env_2d import Env2D
from nets.fully_connected import FullyConnected


def main(args):

    # get settings
    z_size = args.z_size
    if z_size is None:
        if args.env_name in [constants.ENV_1A_EASY, constants.ENV_4A_EASY]:
            z_size = 3
        else:
            z_size = 5

    # set up an environment and collect experience
    env = Env2D(constants.env_to_path(args.env_name))

    train_exp = agent_utils.collect_exp(env, args.num_train)
    test_exp = agent_utils.collect_exp(env, args.num_test)

    # set up and train a neural network
    net = FullyConnected(
        2, len(env.actions), args.z_hiddens, args.t_hiddens, args.r_hiddens, a_hiddens=args.a_hiddens,
        learning_rate=args.learning_rate, z_size=z_size, norm=args.norm, lambda_1=args.lambda_1
    )
    net.state_session()

    losses = agent_utils.train(
        net, args.train_steps, args.batch_size, train_exp[0], train_exp[1], train_exp[2], train_exp[3], train_exp[4]
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("env_name", help=", ".join(constants.ENVS))

    parser.add_argument("--z-hiddens", type=int, nargs="+", default=[20, 20], help="hidden layers for the encoder")
    parser.add_argument("--t-hiddens", type=int, nargs="+", default=[], help="hidden layers for the transition model")
    parser.add_argument("--r-hiddens", type=int, nargs="+", default=[], help="hidden layers for the reward model")
    parser.add_argument("--a-hiddens", type=int, nargs="+", default=None, help="action encoder")

    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--z-size", type=int, default=None, help="size of the latent space; leave it blank")
    parser.add_argument("--norm", type=float, default=0.5,
                        help="L-something norm for the bottleneck (forces the latent vector to approach "
                             "a one-hot vector)")
    parser.add_argument("--lambda-1", type=float, default=0.5, help="strength of the L-something norm loss")

    parser.add_argument("--train-steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--num-train", default=1000, help="number of samples for training")
    parser.add_argument("--num-test", default=1000, help="number of samples for testing")

    parsed = parser.parse_args()
    main(parsed)
