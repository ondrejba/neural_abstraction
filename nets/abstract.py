import abc


class Model(abc.ABC):
    """
    A model for bisimulation learning should define all of the variables below.
    """

    state_pl = NotImplemented
    reward_pl = NotImplemented
    done_pl = NotImplemented
    target_pl = NotImplemented
    action_pl = NotImplemented
    is_training_pl = NotImplemented

    z_t = NotImplemented
    z_bar_t = NotImplemented
    r_bar_t = NotImplemented

    norm_loss_t = NotImplemented
    transition_loss_t = NotImplemented
    reward_loss_t = NotImplemented
    loss_t = NotImplemented
    train_step = NotImplemented

    session = NotImplemented
