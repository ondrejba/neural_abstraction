import collections
import os
import random
import copy as cp
import numpy as np
import viz_utils


def collect_exp(env, num_steps):

    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    states_real = []
    next_states_real = []

    for i in range(num_steps):
        state = env.get_observation()
        action = random.choice(env.actions)
        real_state = cp.deepcopy(env.state)
        next_state, reward, done = env.step(action)
        real_next_state = cp.deepcopy(env.state)

        states.append(state)
        actions.append(action)
        states_real.append(real_state)
        rewards.append(reward)
        next_states.append(next_state)
        next_states_real.append(real_next_state)
        dones.append(done)

        if done:
            env.reset()

    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int32)
    rewards = np.array(rewards, dtype=np.float32)
    next_states = np.array(next_states, dtype=np.float32)
    dones = np.array(dones, dtype=np.float32)
    states_real = np.array(states_real, dtype=np.int32)
    next_states_real = np.array(next_states_real, dtype=np.int32)

    return states, actions, rewards, next_states, dones, states_real, next_states_real


def overlap(predictions, states_real):

    min_pred_class = np.min(predictions)
    min_gt_class = np.min(states_real)

    num_pred_classes = np.max(predictions) - min_pred_class + 1
    num_gt_classes = np.max(states_real) - min_gt_class + 1

    # create an assignment matrix for the two partitions
    num_matches = np.zeros((num_gt_classes, num_pred_classes), dtype=np.int32)

    for p, r in zip(predictions, states_real):
        num_matches[r - min_gt_class, p - min_pred_class] += 1

    # match ground-truth partitions to predicted partitions
    matches = {}
    hits = 0

    for _ in range(min(num_gt_classes, num_pred_classes)):
        coords = np.unravel_index(num_matches.argmax(), num_matches.shape)
        matches[coords[0]] = coords[1]

        hits += num_matches[coords]

        num_matches[coords[0], :] = -1
        num_matches[:, coords[1]] = -1

    return hits, len(predictions)


def train(model, steps, batch_size, states, actions, rewards, next_states, dones, save_latent_space_every_step=False,
          test_exp=None):

    assert test_exp is not None or not save_latent_space_every_step

    run_folder = viz_utils.get_run_folder()

    epoch_size = len(states) // batch_size

    losses = collections.defaultdict(list)
    epoch_losses = collections.defaultdict(list)

    for i in range(steps):

        epoch_idx = i % epoch_size
        batch_slice = np.index_exp[epoch_idx * batch_size: (epoch_idx + 1) * batch_size]

        z = model.session.run(model.target_z_t, feed_dict={
            model.state_pl: next_states[batch_slice],
            model.is_training_pl: False
        })

        feed_dict = {
            model.state_pl: states[batch_slice],
            model.reward_pl: rewards[batch_slice],
            model.done_pl: dones[batch_slice],
            model.target_pl: z,
            model.is_training_pl: True
        }

        if actions is not None and model.action_pl is not None:
            feed_dict[model.action_pl] = actions[batch_slice]

        _, total_loss, transition_loss, reward_loss, norm_loss = model.session.run(
            [model.train_step, model.loss_t, model.transition_loss_t, model.reward_loss_t, model.norm_loss_t],
            feed_dict=feed_dict
        )

        epoch_losses["total"].append(total_loss)
        epoch_losses["transition"].append(transition_loss)
        epoch_losses["reward"].append(reward_loss)
        epoch_losses["norm"].append(norm_loss)

        if model.target_encoder and i % model.target_encoder_update_freq == 0:
            model.session.run(model.update_op)

        if save_latent_space_every_step:
            zs = model.session.run(model.z_t, feed_dict={
                model.state_pl: test_exp[0],
                model.is_training_pl: False
            })
            viz_utils.plot_latent_space(zs, test_exp[5], zs.shape[1],
                                        os.path.join(run_folder, "step_{:d}.png".format(i + 1)))

        if epoch_idx == 0 and i > 0:

            for key, value in epoch_losses.items():
                losses[key].append(np.mean(value))

            epoch_losses = collections.defaultdict(list)

    losses = dict(losses)

    return losses
