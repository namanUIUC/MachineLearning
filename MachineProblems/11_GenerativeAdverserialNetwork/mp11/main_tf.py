"""Generative Adversarial Networks
"""

import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models.gan import Gan
import time
import datetime


def image_save(model, path='GAN.png'):
        # Plot out latent space, for +/- 3 std.
    std = 1
    x_z = np.linspace(-3 * std, 3 * std, 20)
    y_z = np.linspace(-3 * std, 3 * std, 20)

    out = np.empty((28 * 20, 28 * 20))
    for x_idx, x in enumerate(x_z):
        for y_idx, y in enumerate(y_z):
            z_mu = np.random.uniform(-1, 1, [16, 10])
            img = model.generate_samples(z_mu)
            out[x_idx * 28:(x_idx + 1) * 28,
                y_idx * 28:(y_idx + 1) * 28] = img[0].reshape(28, 28)
    plt.imsave(path, out, cmap="gray")


def train(model, mnist_dataset, learning_rate=0.001, batch_size=16,
          num_steps=50000):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size and
    learning_rate.

    Args:
        model(GAN): Initialized generative network.
        mnist_dataset: input_data.
        learning_rate(float): Learning rate.
        batch_size(int): batch size used for training.
        num_steps(int): Number of steps to run the update ops.
    """
    # Iterations for generator
    g_iters = 1

    # Iterations for discriminator
    d_iters = 1

    print('Batch Size: %d, Total epoch: %d, Learning Rate : %f' %
          (batch_size, num_steps, learning_rate))

    filename = "./summary_log/run" + datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%s")
    writer = tf.summary.FileWriter(filename, model.session.graph)

    print('Start training ...')
    tic = time.time()
    for epoch in range(0, num_steps):

        batch_x, _ = mnist_dataset.train.next_batch(batch_size)
        batch_z = np.random.uniform(-1, 1, [batch_size, 10])
        # Train generator and discriminator

        for train_discriminator in range(d_iters):
            _, d_loss = model.session.run(
                [model.d_optimizer, model.d_loss],
                feed_dict={model.x_placeholder: batch_x,
                           model.z_placeholder: batch_z,
                           model.learning_rate_placeholder: learning_rate}
            )

        batch_z = np.random.uniform(-1, 1, [batch_size, 10])
        for train_generator in range(g_iters):
            _, g_loss, sum_out = model.session.run(
                [model.g_optimizer, model.g_loss, model.mergedSummary],
                feed_dict={model.z_placeholder: batch_z,
                           model.learning_rate_placeholder: learning_rate}
            )
        writer.add_summary(sum_out, epoch)
        if (epoch + 1) % 500 == 0:
            toc = time.time() - tic
            tic = time.time()
            print("%d training steps completed out of %d in %d sec" % (epoch + 1, num_steps, toc))
            print("g_loss :", g_loss)
            print("d_loss :", d_loss)
            print('-' * 50)
            image_save(model, './output/' + str(epoch + 1))


def main(_):
    """High level pipeline.

    This scripts performs the training for GANs.
    """
    # Get dataset.
    mnist_dataset = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Build model.
    model = Gan(nlatent=10)

    # Start training
    train(model, mnist_dataset)


if __name__ == "__main__":
    tf.app.run()
