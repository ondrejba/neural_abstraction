import tensorflow as tf
from nets.abstract import Model
import l0_norm


class FullyConnected(Model):

    MOMENTUM = 0.9

    def __init__(self, state_size, num_actions, z_hiddens, t_hiddens, r_hiddens, a_hiddens=None, learning_rate=0.01,
                 z_size=3, l_norm=None, l0_norm=False, lambda_1=1.0, lambda_2=1.0, entropy=False, sparse=False,
                 sparse_target=0.05, log_sparse=False, target_encoder=True, target_encoder_update_freq=50,
                 z_activation=tf.nn.softmax):

        self.state_size = state_size
        self.num_actions = num_actions
        self.z_hiddens = z_hiddens
        self.t_hiddens = t_hiddens
        self.r_hiddens = r_hiddens
        self.a_hiddens = a_hiddens

        self.learning_rate = learning_rate
        self.z_size = z_size
        self.l_norm = l_norm
        self.l0_norm = l0_norm
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.entropy = entropy
        self.sparse = sparse
        self.sparse_target = sparse_target
        self.log_sparse = log_sparse
        self.target_encoder = target_encoder
        self.target_encoder_update_freq = target_encoder_update_freq
        self.z_activation = z_activation

        self.state_pl = None
        self.reward_pl = None
        self.done_pl = None
        self.target_pl = None
        self.action_pl = None
        self.lambda_1_v = None

        self.z_t = None
        self.target_z_t = None
        self.z_bar_t = None
        self.r_bar_t = None

        self.norm_loss_t = None
        self.transition_loss_t = None
        self.reward_loss_t = None
        self.loss_t = None
        self.train_step = None
        self.update_op = None

        self.build()

        self.session = None

    def build(self):

        # placeholders
        self.state_pl = tf.placeholder(tf.float32, shape=(None, self.state_size), name="state_pl")
        self.reward_pl = tf.placeholder(tf.float32, shape=(None,), name="reward_pl")
        self.done_pl = tf.placeholder(tf.float32, shape=(None,), name="done_pl")
        self.target_pl = tf.placeholder(tf.float32, shape=(None, self.z_size), name="target_pl")
        self.is_training_pl = tf.placeholder(tf.bool, shape=[], name="is_training_pl")

        if self.num_actions > 1:
            self.action_pl = tf.placeholder(tf.int32, shape=(None,), name="action_pl")

        self.lambda_1_v = tf.Variable(initial_value=self.lambda_1, trainable=False, name="lambda_1")

        # encoder
        with tf.variable_scope("encoder"):

            x = self.state_pl

            for i, hidden in enumerate(self.z_hiddens):
                x = tf.layers.dense(x, hidden, activation=tf.nn.relu)

            self.z_t = tf.layers.dense(x, self.z_size, activation=self.z_activation)

        if self.target_encoder:

            with tf.variable_scope("target_encoder"):

                x = self.state_pl

                for i, hidden in enumerate(self.z_hiddens):
                    x = tf.layers.dense(x, hidden, activation=tf.nn.relu)

                self.target_z_t = tf.layers.dense(x, self.z_size, activation=tf.nn.softmax)

            encoder_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="encoder")
            target_encoder_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_encoder")

            update_ops = []
            for source, target in zip(encoder_vars, target_encoder_vars):
                update_ops.append(tf.assign(target, source))
            self.update_op = tf.group(*update_ops)

        else:

            self.target_z_t = self.z_t

        norm_t = tf.constant(0.0)

        # sparsity penalty
        if self.sparse:
            avg_activation = tf.reduce_mean(self.z_t, axis=0)
            norm_t = self.sparse_target * tf.log(self.sparse_target / avg_activation) + \
                (1 - self.sparse_target) * tf.log((1 - self.sparse_target) / (1 - avg_activation))
            norm_t = tf.reduce_sum(norm_t)

        # log sparsity
        if self.log_sparse:
            norm_t = tf.reduce_sum(tf.log(1 + tf.square(self.z_t)), axis=1)

        # L0 norm
        if self.l0_norm:
            self.z_t, l0_reg = l0_norm.l0_norm(self.z_t, self.is_training_pl)

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

        self.z_bar_t = tf.layers.dense(x, self.z_size, activation=self.z_activation)

        # reward function
        x = z_plus_a_t

        for hidden in self.r_hiddens:
            x = tf.layers.dense(x, hidden, activation=tf.nn.relu)

        self.r_bar_t = tf.layers.dense(x, 1, activation=tf.nn.sigmoid)[:, 0]

        # loss
        if self.entropy:
            norm_t = - tf.reduce_sum(self.z_t * tf.log(self.z_t), axis=1)
        elif self.l_norm is not None:
            norm_t = tf.reduce_sum(tf.pow(tf.abs(self.z_t), self.l_norm), axis=1)

        self.norm_loss_t = tf.reduce_mean(norm_t)

        self.transition_loss_t = (1.0 - self.done_pl) * tf.reduce_sum((self.z_bar_t - self.target_pl) ** 2, axis=1)
        self.reward_loss_t = (self.r_bar_t - self.reward_pl) ** 2

        self.loss_t = self.lambda_2 * self.transition_loss_t + self.reward_loss_t
        self.loss_t = tf.reduce_mean(self.loss_t) + self.lambda_1_v * self.norm_loss_t

        # training
        self.train_step = tf.train.MomentumOptimizer(self.learning_rate, self.MOMENTUM).minimize(self.loss_t)

    def state_session(self):

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def stop_session(self):

        if self.session is not None:
            self.session.close()
            self.session = None

    def update_lambda_1(self, value):

        self.lambda_1 = value
        self.session.run(tf.assign(self.lambda_1_v, value))
