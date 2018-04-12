"""Implements the Gaussian Mixture model, and trains using EM algorithm."""
import numpy as np
import scipy
from scipy.stats import multivariate_normal
from scipy.cluster.vq import kmeans2


class GaussianMixtureModel(object):
    """Gaussian Mixture Model"""

    def __init__(self, n_dims, n_components=1,
                 max_iter=10,
                 reg_covar=1e-6):
        """
        Args:
            n_dims: The dimension of the feature.
            n_components: Number of Gaussians in the GMM.
            max_iter: Number of steps to run EM.
            reg_covar: Amount to regularize the covariance matrix, (i.e. add
                to the diagonal of covariance matrices).
        """

        self._n_dims = n_dims
        self._n_components = n_components
        self._max_iter = max_iter
        self._reg_covar = reg_covar

        # Randomly Initialize model parameters
        self._mu = np.random.rand(n_components, n_dims)  # np.array of size (n_components, n_dims)

        # Initialized with uniform distribution.
        tmp = np.random.dirichlet(np.ones(self._n_components), size=1)
        self._pi = tmp.reshape(tmp.shape[1], 1)

        # Initialized with identity.
        self._sigma = 10 * np.array(self._n_components * [np.eye(self._n_dims)])  # np.array of size (n_components, n_dims, n_dims)

    def fit(self, x):
        """Runs EM steps.

        Runs EM steps for max_iter number of steps.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        """

        # Reinitializing mu
        self._mu = x[np.random.choice(x.shape[0], self._n_components, False), :]

        for i in range(self._max_iter):
            #print("iteration : ", i + 1)

            # E Step
            z_ik = self._e_step(x)

            #M Step
            self._m_step(x, z_ik)

    def _e_step(self, x):
        """E step.

        Wraps around get_posterior.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """
        return self.get_posterior(x)

    def _m_step(self, x, z_ik):
        """M step, update the parameters.

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
                (Alternate way of representing categorical distribution of z_i)
        """
        # Update the parameters.
        z_ik = self.get_posterior(x)
        mu = self._mu
        sigma = self._sigma
        z_k = np.sum(z_ik, axis=0)

        # Updating pi
        self._pi = z_k / x.shape[0]

        # Updating mu
        num = np.matmul(np.transpose(z_ik), x)
        self._mu = np.transpose(np.transpose(num) / z_k)

        # Updating sigma
        temp = np.zeros((self._n_components, self._n_dims, self._n_dims))
        for k in range(self._n_components):

            demean = x - self._mu[k, :]
            zik_times_x = (z_ik[:, k][:, np.newaxis] * demean)
            temp[k, :, :] = np.dot(np.transpose(demean), zik_times_x) / z_k[k]
            for i in range(self._n_dims):
                temp[k, i, i] += self._reg_covar
        self._sigma = temp


    def get_conditional(self, x):
        """Computes the conditional probability.

        p(x^(i)|z_ik=1)

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            ret(numpy.ndarray): The conditional probability for each example,
                dimension (N,, n_components).
        """

        p = []
        for k in range(self._n_components):
            temp = self._multivariate_gaussian(x, self._mu[k], self._sigma[k])
            p.append(temp)
        p = np.transpose(np.array(p))

        return np.array(p)

    def get_marginals(self, x):
        """Computes the marginal probability.

        p(x^(i)|pi, mu, sigma)

        Args:
             x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            (1) The marginal probability for each example, dimension (N,).
        """
        # Calculate p(x_i | mu_k, sigma_k) for all i and k
        p = self.get_conditional(x)

        # Calculate basian denominator
        denominator = np.multiply(p, np.transpose(self._pi))

        # Sum all k from 1 - K for each i
        marginal = np.sum(denominator, axis=1)

        return marginal + np.finfo(float).eps

    def get_posterior(self, x):
        """Computes the posterior probability.

        p(z_{ik}=1|x^(i))

        Args:
            x(numpy.ndarray): Feature array of dimension (N, ndims).
        Returns:
            z_ik(numpy.ndarray): Array containing the posterior probability
                of each example, dimension (N, n_components).
        """
        # Get the denominator for basian probability
        marginals = self.get_marginals(x)

        # Calculate p(x_i | mu_k, sigma_k) for all i and k
        N = self.get_conditional(x)

        # Calculate basian numerator
        numerator = np.multiply(N, np.transpose(self._pi))

        # Calculate basian matrix
        z_ik = np.transpose(np.transpose(numerator) / marginals)


        return z_ik

    def _multivariate_gaussian(self, x, mu_k, sigma_k):
        """Multivariate Gaussian, implemented for you.
        Args:
            x(numpy.ndarray): Array containing the features of dimension (N,
                ndims)
            mu_k(numpy.ndarray): Array containing one single mean (ndims,1)
            sigma_k(numpy.ndarray): Array containing one signle covariance matrix
                (ndims, ndims)
        """
        return multivariate_normal.pdf(x, mu_k, sigma_k)

    def supervised_fit(self, x, y):
        """Assign each cluster with a label through counting.
        For each cluster, find the most common digit using the provided (x,y)
        and store it in self.cluster_label_map.
        self.cluster_label_map should be a list of length n_components,
        where each element maps to the most common digit in that cluster.
        (e.g. If self.cluster_label_map[0] = 9. Then the most common digit
        in cluster 0 is 9.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
            y(numpy.ndarray): Array containing the label of dimension (N,)
        """

        self.fit(x)

        self.cluster_label_map = np.random.rand(self._n_components).tolist()

        cluster_index = np.argmax(self.get_posterior(x), axis=1)
        for k in range(self._n_components):
            points_index = np.where(cluster_index == k)
            if points_index[0].size:
                associated_labels = y[points_index]
                most_common_label = scipy.stats.mode(associated_labels)[0]
                self.cluster_label_map[k] = most_common_label

    def supervised_predict(self, x):
        """Predict a label for each example in x.
        Find the get the cluster assignment for each x, then use
        self.cluster_label_map to map to the corresponding digit.
        Args:
            x(numpy.ndarray): Array containing the feature of dimension (N,
                ndims).
        Returns:
            y_hat(numpy.ndarray): Array containing the predicted label for each
            x, dimension (N,)
        """

        z_ik = self.get_posterior(x)
        y_hat = []
        cluster_index = np.argmax(z_ik, axis=1)
        y_hat = [self.cluster_label_map[indx][0] for indx in cluster_index]
        return np.array(y_hat)
