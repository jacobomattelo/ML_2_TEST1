import numpy as np

class TSNE:
    def __init__(self, n_components, perplexity=30.0, learning_rate=200.0, n_iter=1000):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.embedding = None

    def _symmetric_sne(self, P):
        # Compute symmetric SNE probability matrix Q
        Q = 1 / (1 + np.sum((P[:, None, :] - P[None, :, :]) ** 2, axis=-1) / self.perplexity)
        np.fill_diagonal(Q, 0)
        Q /= np.sum(Q)
        return Q

    def _gradient(self, X, P, Q, Y):
        # Compute gradient of t-SNE cost function
        pq_diff = P - Q
        grad = np.zeros_like(Y)
        for i in range(len(Y)):
            grad[i, :] = np.sum((pq_diff[:, i] * (Y[i, :] - Y)), axis=0)
        grad *= 4
        grad -= 2 * self.learning_rate * np.sum((1 / (1 + np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)))[:, :, None] * (Y[:, None, :] - Y), axis=0)
        return grad

    def _kl_divergence(self, P, Q):
        # Compute KL divergence between P and Q
        return np.sum(P * np.log(P / Q))

    def _tsne(self, X):
        # Initialize Y randomly
        Y = np.random.randn(X.shape[0], self.n_components)

        # Compute pairwise probability matrix P for a subset of points
        num_samples = min(1000, X.shape[0])  # Limit the number of samples to reduce memory usage
        indices = np.random.choice(X.shape[0], num_samples, replace=False)
        X_subset = X[indices]

        P = np.zeros((num_samples, num_samples))
        for i in range(num_samples):
            distances = np.sum((X_subset[i, :] - X_subset) ** 2, axis=1)
            P[i, :] = self._conditional_probabilities(distances, i)
        P = (P + P.T) / (2 * num_samples)

        # Train t-SNE using gradient descent
        for _ in range(self.n_iter):
            Q = self._symmetric_sne(P)
            grad = self._gradient(X, P, Q, Y)
            Y -= self.learning_rate * grad
            P = self._symmetric_sne(Y)

        return Y

    def _conditional_probabilities(self, distances, i):
        # Compute conditional probabilities using binary search
        beta_min, beta_max = -np.inf, np.inf
        tol = 1e-5
        target_entropy = np.log(self.perplexity)

        while True:
            beta = (beta_min + beta_max) / 2
            exp_distances = np.exp(-beta * distances)
            sum_exp_distances = np.sum(exp_distances)
            P = exp_distances / sum_exp_distances
            entropy = -np.sum(P * np.log2(P + 1e-12))
            error = entropy - target_entropy

            if np.abs(error) < tol:
                break
            elif error > 0:
                beta_max = beta
            else:
                beta_min = beta

        return P

    def fit_transform(self, X):
        self.embedding = self._tsne(X)
        return self.embedding