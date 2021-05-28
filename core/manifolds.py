import numpy as np
from sklearn.metrics import pairwise_distances
import time

# This Riemannian metric is the inverse of the aggregated posterior.
# M(z) = (rho + q(z))^(2/D) * I_d, so the sqrt(|M(z)|) = 1 / (q(z) + rho)
class AggregatedPosteriorMetric:

    def __init__(self, data, sigmas, rho=1e-2):
        self.rho = rho
        self.sigmas = sigmas
        self.centers = data

    @staticmethod
    def is_diagonal():
        return True

    # Compute the measure
    def measure(self, z):
        # z: d x N
        M = self.metric_tensor(z)  # N x D x D
        return np.sqrt(np.prod(M, axis=1)).reshape(-1, 1)  # N x 1

    def metric_tensor(self, c, nargout=1):
        D, N = c.shape
        c = c.T  # We need in the computations N x D format

        dists = pairwise_distances(c, self.centers) / self.sigmas.T
        Phi = np.exp(-0.5 * dists ** 2) / ((2 * np.pi * (self.sigmas**2).T) ** (D / 2))  # N x S
        y = 1 / (Phi.mean(axis=1, keepdims=True) + self.rho)  # N x 1
        M = np.ones((N, D)) * (y ** (2 / D))

        if nargout == 2:
            dif_data_c = -(self.centers[:, np.newaxis] - c[np.newaxis, :])  # S x N x D
            factor = (y ** ((2 + D) / D)) / self.centers.shape[0]  # N x 1
            term_1 = (np.expand_dims(Phi.T / (self.sigmas ** 2), axis=2).repeat(D, axis=2)) * dif_data_c
            temp = term_1.sum(axis=0) * factor  # N x D
            dMdc = temp[:, np.newaxis, :].repeat(D, axis=1)
            return M, dMdc
        return M

    def asymptotic_measure(self):
        return 1/self.rho


# Note: local diagonal PCA with projection
# This is the classical local diagonal PCA metric
class LocalDiagPCA:

    def __init__(self, data, sigma, rho, with_projection=False, A=None, b=None):
        self.with_projection = with_projection
        if with_projection:
            self.data = (data - b.reshape(1, -1)) @ A  # NxD
            self.A = A
            self.b = b.reshape(-1, 1)  # D x 1
        else:
            self.data = data  # NxD
        self.sigma = sigma
        self.rho = rho
        

    @staticmethod
    def is_diagonal():
        return True

    def measure(self, z):
        # z: d x N
        M = self.metric_tensor(z)  # N x D x D
        return np.sqrt(np.prod(M, axis=1)).reshape(-1, 1)  # N x 1
    
    # the asymptotic value of the measure far away from data
    # does not exist for all manifolds
    def asymptotic_measure(self):
        return np.sqrt(1/self.rho**self.data.shape[1])
    

    def metric_tensor(self, c, nargout=1):
        # c is D x N
        if self.with_projection:
            c = ((c.T - self.b.T) @ self.A).T

        sigma2 = self.sigma ** 2
        D, N = c.shape

        M = np.empty((N, D))
        M[:] = np.nan
        if nargout == 2:  # compute the derivative of the metric
            dMdc = np.empty((N, D, D))
            dMdc[:] = np.nan

        for n in range(N):
            cn = c[:, n]  # Dx1
            delta = self.data - cn.transpose()  # N x D
            delta2 = delta ** 2  # pointwise square
            dist2 = np.sum(delta2, axis=1, keepdims=True)  # Nx1, ||X-c||^2
            # wn = np.exp(-0.5 * dist2 / sigma2) / ((2 * np.pi * sigma2) ** (D / 2))  # Nx1
            wn = np.exp(-0.5 * dist2 / sigma2)
            s = np.dot(delta2.transpose(), wn) + self.rho  # Dx1
            m = 1 / s  # D x1
            M[n, :] = m.transpose()

            if nargout == 2:
                dsdc = 2 * np.diag(np.squeeze(np.matmul(delta.transpose(), wn)))
                weighted_delta = (wn / sigma2) * delta
                dsdc = dsdc - np.matmul(weighted_delta.transpose(), delta2)
                dMdc[n, :, :] = dsdc.transpose() * m ** 2  # The dMdc[n, D, d] = dMdc_d

        if nargout == 1:
            return M
        elif nargout == 2:
            return M, dMdc

