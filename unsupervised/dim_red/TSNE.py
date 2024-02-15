import numpy as np

class TSNE:
    def __init__(self, n_components, perplexity=30.0, learning_rate=100.0, n_iter=1000, random_state=None):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self.embedding = None

    def fit(self, X):
        # Initialize embedding randomly or based on random state
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.embedding = np.random.normal(0., 1e-4, (X.shape[0], self.n_components))

        # Compute pairwise distances
        pairwise_distances = self._pairwise_distances(X)

        # Compute pairwise similarities using a Gaussian kernel
        P = self._joint_probabilities(pairwise_distances)

        # Initialize low-dimensional representation randomly
        Y = np.random.normal(0., 1e-4, (X.shape[0], self.n_components))

        # Perform gradient descent
        for _ in range(self.n_iter):
            # Compute Q distribution
            Q = self._student_t_distribution(Y)

            # Compute gradient
            grad = self._gradient(P, Q, Y)

            # Update embedding
            Y -= self.learning_rate * grad

        self.embedding = Y

    def transform(self, X):
        return self.embedding

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def _pairwise_distances(self, X):
        return np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=-1)

    def _joint_probabilities(self, distances):
        # Compute conditional probabilities using perplexity
        P = np.exp(-distances * self.perplexity)
        P /= np.sum(P, axis=1, keepdims=True)
        P = (P + P.T) / (2. * distances.shape[0])  # Symmetrize
        return np.maximum(P, 1e-12)

    def _student_t_distribution(self, Y):
        distances = self._pairwise_distances(Y)
        inv_distances = 1. / (1. + distances)
        np.fill_diagonal(inv_distances, 0.)
        return inv_distances / np.sum(inv_distances, axis=1, keepdims=True)

    def _gradient(self, P, Q, Y):
        # Compute gradient using the t-SNE formula
        pq_diff = P - Q
        grad = np.zeros_like(Y)
        for i in range(Y.shape[0]):
            grad[i] = 4. * np.sum(
                (pq_diff[i] * (Y[i, np.newaxis] - Y)) *
                (1. - np.sqrt(np.sum((Y[i, np.newaxis] - Y) ** 2, axis=1)))[:, np.newaxis],
                axis=0)
        return grad
