import argparse
import numpy as np
import matplotlib.pyplot as plt
import agent_utils
import constants
import viz_utils
from envs.env_2d import Env2D
from nets.fully_connected import FullyConnected
plt.style.use("seaborn-colorblind")


def main(args):

    # get settings
    z_size = args.z_size
    if z_size is None:
        if args.env_name in [constants.ENV_1A_EASY, constants.ENV_4A_EASY]:
            z_size = 2
        elif args.env_name in [constants.ENV_1A_3_BLOCKS, constants.ENV_1A_3_BLOCKS_EASY]:
            z_size = 3
        else:
            z_size = 4

    # set up an environment and collect experience
    env = Env2D(constants.env_to_path(args.env_name))

    train_exp = agent_utils.collect_exp(env, args.num_train)
    test_exp = agent_utils.collect_exp(env, args.num_test)

    if args.plot_exp:
        for data, labels in [[train_exp[0], train_exp[5]], [test_exp[0], test_exp[5]]]:
            plt.scatter(data[:, 0], data[:, 1], c=labels)
            plt.colorbar()
            plt.show()

    # set up and train a neural network
    net = FullyConnected(
        2, len(env.actions), args.z_hiddens, args.t_hiddens, args.r_hiddens, a_hiddens=args.a_hiddens,
        learning_rate=args.learning_rate, z_size=z_size, l_norm=args.l_norm, l0_norm=args.l0_norm,
        lambda_1=args.lambda_1, entropy=args.entropy, sparse=args.sparse, sparse_target=args.sparse_target
    )
    net.state_session()

    losses = agent_utils.train(
        net, args.train_steps, args.batch_size, train_exp[0], train_exp[1], train_exp[2], train_exp[3], train_exp[4],
        save_latent_space_every_step=args.plot_latent_space_every_step, test_exp=test_exp
    )

    if args.step_2_train_steps is not None:
        net.update_lambda_1(args.step_2_lambda_1)
        losses2 = agent_utils.train(
            net, args.train_steps, args.batch_size, train_exp[0], train_exp[1], train_exp[2], train_exp[3], train_exp[4]
        )
        losses = {key: np.concatenate([losses[key], losses2[key]], axis=0) for key in losses.keys()}

    zs = net.session.run(net.z_t, feed_dict={
        net.state_pl: test_exp[0],
        net.is_training_pl: False
    })

    # plot losses and latent space
    print("plotting losses")
    viz_utils.plot_losses(losses)

    if args.plot_latent_space:
        print("plotting latent space")
        viz_utils.plot_latent_space(zs, test_exp[5], z_size)

    # calculate overlap
    predictions = np.argmax(zs, axis=1)
    hits, total = agent_utils.overlap(predictions, test_exp[5])

    print("overlap: {:.2f}% ({:d}/{:d})".format((hits / total) * 100, hits, total))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("env_name", help=", ".join(constants.ENVS))

    parser.add_argument("--plot-exp", default=False, action="store_true", help="plot collected experience")
    parser.add_argument("--plot-latent-space", default=False, action="store_true", help="only works up to 3D")
    parser.add_argument("--plot-latent-space-every-step", default=False, action="store_true")

    parser.add_argument("--z-hiddens", type=int, nargs="+", default=[20, 20], help="hidden layers for the encoder")
    parser.add_argument("--t-hiddens", type=int, nargs="+", default=[], help="hidden layers for the transition model")
    parser.add_argument("--r-hiddens", type=int, nargs="+", default=[], help="hidden layers for the reward model")
    parser.add_argument("--a-hiddens", type=int, nargs="+", default=None, help="action encoder")

    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--z-size", type=int, default=None, help="size of the latent space; leave it blank")
    parser.add_argument("--l-norm", type=float, default=None,
                        help="L-something norm for the bottleneck (forces the latent vector to approach "
                             "a one-hot vector)")
    parser.add_argument("--l0-norm", default=False, action="store_true",
                        help="use the L0 norm")
    parser.add_argument("--lambda-1", type=float, default=0.5, help="strength of the L-something norm loss")
    parser.add_argument("--entropy", default=False, action="store_true", help="minimize entropy of z")

    parser.add_argument("--sparse", default=False, action="store_true")
    parser.add_argument("--sparse-target", type=float, default=0.05)

    parser.add_argument("--train-steps", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--num-train", type=int, default=1000, help="number of samples for training")
    parser.add_argument("--num-test", type=int, default=1000, help="number of samples for testing")

    parser.add_argument("--step-2-train-steps", type=int, default=None)
    parser.add_argument("--step-2-lambda-1", type=float, default=0.0)

    parsed = parser.parse_args()
    main(parsed)
