"""Simple unit tests for students."""

import unittest
import numpy as np
from models import gan
import tensorflow as tf

class ModelTests(unittest.TestCase):
    def setUp(self):
        self.model = gan.Gan()

    def test_ouput_shape(self):
        np.testing.assert_array_equal(self.model.session.run(tf.shape(self.model.x_hat)), (10, 1))

    def test_generator_loss_shape(self):
        tf.assert_scalar(self.model.g_loss)

if __name__ == '__main__':
    unittest.main()
