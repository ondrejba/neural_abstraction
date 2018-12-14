from nets.abstract import Model
import tensorflow as tf


class FullyConnected(Model):

    MOMENTUM = 0.9

    def __init__(self, state_size, num_actions, z_hiddens, t_hiddens, r_hiddens, a_hiddens=None, learning_rate=0.01,
                 z_size=3, norm=0.5, lambda_1=0.5):

        self.state_size = state_size
        self.num_actions = num_actions
        self.z_hiddens = z_hiddens
        self.t_hiddens = t_hiddens
        self.r_hiddens = r_hiddens
        self.a_hiddens = a_hiddens

        self.learning_rate = learning_rate
        self.z_size = z_size
        self.norm = norm
        self.lambda_1 = lambda_1

        self.state_pl = None
        self.reward_pl = None
        self.done_pl = None
        self.target_pl = None
        self.action_pl = None

        self.z_t = None
        self.z_bar_t = None
        self.r_bar_t = None

        self.norm_loss_t = None
        self.transition_loss_t = None
        self.reward_loss_t = None
        self.loss_t = None
        self.train_step = None

        self.build()

        self.session = None

    def build(self):

        # placeholders
        self.state_pl = tf.placeholder(tf.float32, shape=(None, self.state_size), name="state_pl")
        self.reward_pl = tf.placeholder(tf.float32, shape=(None,), name="reward_pl")
        self.done_pl = tf.placeholder(tf.float32, shape=(None,), name="done_pl")
        self.target_pl = tf.placeholder(tf.float32, shape=(None, self.z_size), name="target_pl")

        if self.num_actions > 1:
            self.action_pl = tf.placeholder(tf.int32, shape=(None,), name="action_pl")

        # encoder
        x = self.state_pl

        for i, hidden in enumerate(self.z_hiddens):
            x = tf.layers.dense(x, hidden, activation=tf.nn.relu)

        x = tf.layers.dense(x, self.z_size, activation=tf.nn.softmax)

        self.z_t = x

        # action encoder
        if self.action_pl is not None:

            a = tf.one_hot(self.action_pl, self.num_actions, dtype=tf.float32)

            if self.a_hiddens is not None:
                for hidden in self.a_hiddens:
                    a = tf.layers.dense(a, hidden, activation=tf.nn.relu)

            z_plus_a_t = tf.concat([self.z_t, a], axis=1)
        else:
            z_plus_a_t = self.z_t

        # transition function
        x = z_plus_a_t

        for hidden in self.t_hiddens:
            x = tf.layers.dense(x, hidden, activation=tf.nn.relu)

        self.z_bar_t = tf.layers.dense(x, self.z_size, activation=None)

        # reward function
        x = z_plus_a_t

        for hidden in self.r_hiddens:
            x = tf.layers.dense(x, hidden, activation=tf.nn.relu)

        self.r_bar_t = tf.layers.dense(x, 1, activation=None)[:, 0]

        # loss
        norm_t = tf.reduce_sum(tf.pow(tf.abs(self.z_t), self.norm), axis=1)
        self.norm_loss_t = tf.reduce_mean(norm_t)

        self.transition_loss_t = (1.0 - self.done_pl) * tf.reduce_mean((self.z_bar_t - self.target_pl) ** 2, axis=1)
        self.reward_loss_t = (self.r_bar_t - self.reward_pl) ** 2

        self.loss_t = self.transition_loss_t + self.reward_loss_t
        self.loss_t = tf.reduce_mean(self.loss_t) + self.lambda_1 * self.norm_loss_t

        # training
        self.train_step = tf.train.MomentumOptimizer(self.learning_rate, self.MOMENTUM).minimize(self.loss_t)

    def state_session(self):

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def stop_session(self):

        if self.session is not None:
            self.session.close()
            self.session = None
