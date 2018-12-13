import unittest
import numpy as np
from nets.fully_connected import FullyConnected


class TestFullyConnected(unittest.TestCase):

    def test_train_step(self):

        net = FullyConnected(2, 1, [20, 20], [], [])
        net.state_session()

        net.session.run(net.train_step, feed_dict={
            net.state_pl: np.random.uniform(-1, 1, [10, 2]),
            net.reward_pl: np.random.randint(0, 2, [10]),
            net.done_pl: np.random.randint(0, 2, [10], dtype=np.bool),
            net.target_pl: np.random.uniform(-1, 1, [10, 3])
        })
