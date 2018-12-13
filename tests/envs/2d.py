import unittest
from envs.env_2d import Env2D
import paths


class TestEnv2D(unittest.TestCase):

    def test_1a_easy(self):

        env = Env2D(paths.ENV_1A_EASY)

        self.assertEqual(env.state, 1)

        _, r, d, n = env.step(1)

        self.assertEqual(r, 0)
        self.assertEqual(d, False)
        self.assertEqual(n, 2)
        self.assertEqual(n, env.state)

        _, r, d, n = env.step(1)

        self.assertEqual(r, 1)
        self.assertEqual(d, True)
        self.assertEqual(n, 3)
        self.assertEqual(n, env.state)

        env.reset()

        self.assertEqual(env.state, 1)
