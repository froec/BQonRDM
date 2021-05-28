import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
# import matplotlib.transforms as transforms
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from core import utils
import scipy.io as sio
#import torch
from scipy.stats import multivariate_normal
from core import geodesics
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
from scipy.interpolate import CubicSpline

# for the logger
from os import listdir
from os.path import isfile, join
import dill

# for logging information produced by the integration procedures
# naming scheme: data-iteration_0.pkl, ..., data-iteration_n.pkl
def logger(savedict, output_directory, startswith="data-iteration_"):
    logfiles_ids = [-1] + [int((f.split("_")[1]).split(".")[0]) \
                            for f in listdir(output_directory) if isfile(join(output_directory, f)) \
                            and f.startswith(startswith)]

    _id = max(logfiles_ids) + 1
    dill.dump(savedict, open(output_directory + startswith + str(_id)  + ".pkl", "wb"))
    return True

    

def predict_gmm(model, data):
    K = model['means'].shape[0]
    N = data.shape[0]
    resps = np.zeros((N, K))
    for k in range(K):
        resps[:, k] = model['Weights'][k] * multivariate_normal.pdf(data,
                                                                    mean=model['means'][k, :],
                                                                    cov=model['Sigmas'][k, :, :])
    resps = resps / resps.sum(axis=1, keepdims=True)
    return resps


def semi_supervised_gmm(data_labeled, data_unlabeled, labels, K, max_iter=10):

    N_labeled, D = data_labeled.shape
    N_ublabeled = data_unlabeled.shape[0]
    N = N_ublabeled + N_labeled

    means = np.zeros((K, D))  # K x D
    Sigmas = np.zeros((K, D, D))
    Weights = np.zeros((K, 1))

    # Initialize the centers and the covariances
    resp_labeled = np.zeros((N_labeled, K))
    for k in range(K):
        inds_k = (labels == k)
        resp_labeled[inds_k.flatten(), k] = 1  # We set 1 for the points in this class
        Weights[k, 0] = np.sum(inds_k)  # The number of points in this class
        data_labeled_k = data_labeled[inds_k.flatten(), :]
        means[k, :] = data_labeled_k.mean(0).flatten()
        Sigmas[k, :, :] = np.cov(data_labeled_k.T)
    Weights = Weights / N_labeled

    resp_unlabeled = np.zeros((N_ublabeled, K))
    for iteration in range(max_iter):

        # E-step
        for k in range(K):
            resp_unlabeled[:, k] = Weights[k] * multivariate_normal.pdf(data_unlabeled,
                                                                        mean=means[k, :],
                                                                        cov=Sigmas[k, :, :])
        resp_unlabeled = resp_unlabeled / resp_unlabeled.sum(axis=1, keepdims=True)

        # Combine the data for simplicity
        resp_all = np.concatenate((resp_labeled, resp_unlabeled), axis=0)
        data_all = np.concatenate((data_labeled, data_unlabeled), axis=0)

        # M-step
        for k in range(K):
            resp_k = resp_all[:, k].reshape(-1, 1)  # (N_u + N_l) x 1
            N_k = np.sum(resp_k, axis=0)
            means[k, :] = (resp_k.T @ data_all) / N_k  # 1 x D
            Sigmas[k, :, :] = ((data_all - means[k, :].reshape(1, -1)).T
                                    @ np.diag(resp_k.flatten())
                                    @ (data_all - means[k, :].reshape(1, -1))) / N_k
            Weights[k, 0] = N_k / N

    result = {'means': means, 'Sigmas': Sigmas, 'Weights': Weights, 'responsibilities': resp_all}
    return result


# Plots a graph from the dataset and the edges between the points
def print_graph(data, graph):
    N = data.shape[0]
    plt.figure()
    utils.my_plot(data, s=10, c='k')
    graph = graph / graph.max()
    for ni in range(N):
        for nj in range(ni+1, N):
            if graph[ni, nj] > 0:
                points = np.concatenate((data[ni, :].reshape(1, -1), data[nj, :].reshape(1, -1)), axis=0)
                plt.plot(points[:, 0], points[:, 1], c='r', linewidth=(1 + 5*graph[ni, nj]), alpha=1)
                # plt.plot(points[:, 0], points[:, 1], c='r', linewidth=1, alpha=1)


# Uniform distribution on a disk, returns 1 if the points x in the radius of the disk
def uniform_p_z(x, r=3.5):
    # x: N x D
    dists = np.sqrt(np.sum(x ** 2, axis=1, keepdims=True))  # N x 1,
    result = np.zeros((x.shape[0], 1))
    result[(dists < r).flatten(), :] = 1  # Only those in the disk get 1.
    return result.flatten()


# Numerical estimation of the Riemannian metric, using decoder based on numpy.
def numerical_metric_computation(manifold, z, D_dim, d_dim, eps=1e-4):
    Jacobians = np.zeros((D_dim, d_dim))
    F = manifold.decode(z)  # D x 1
    M_ambient = manifold.manifold_ambient.metric_tensor(F)
    eps = eps
    for d in range(d_dim):
        temp_matrix = np.zeros_like(z)
        temp_matrix[d, 0] = 1 * eps
        F_d = manifold.decode(z + temp_matrix)  # D x 1
        Jacobians[:, d] = ((F - F_d) / eps).flatten()
    if manifold.manifold_ambient.with_projection:
        A = manifold.manifold_ambient.A
        if manifold.manifold_ambient.is_diagonal():
            Metrics = Jacobians.T @ A @ np.diag(M_ambient.flatten()) @ A.T @ Jacobians + np.eye(d_dim)
        else:
            Metrics = Jacobians.T @ A @ M_ambient.squeeze() @ A.T @ Jacobians + np.eye(d_dim)
    else:
        if manifold.manifold_ambient.is_diagonal():
            Metrics = Jacobians.T @ np.diag(M_ambient.flatten()) @ Jacobians + np.eye(d_dim)
        else:
            Metrics = Jacobians.T @ M_ambient.squeeze() @ Jacobians + np.eye(d_dim)

    return Metrics, Jacobians

"""
# Numerical estimation of the Riemannian metric, using decoder based on Torch.
def numerical_metric_computation_torch(model, manifold_ambient, z, D_dim, d_dim, eps):
    with torch.no_grad():
        Jacobians = np.zeros((D_dim, d_dim))
        F = model.generate(torch.from_numpy(z.T.astype(np.float32))).detach().numpy().T  # D x 1
        M_ambient = manifold_ambient.metric_tensor(F)
        for d in range(d_dim):
            temp_matrix = np.zeros_like(z)
            temp_matrix[d, 0] = 1 * eps
            F_d = model.generate(torch.from_numpy((z + temp_matrix).T.astype(np.float32))).detach().numpy().T
            Jacobians[:, d] = ((F - F_d) / eps).flatten()
        if manifold_ambient.with_projection:
            A = manifold_ambient.A
            if manifold_ambient.is_diagonal():
                Metrics = Jacobians.T @ A @ np.diag(M_ambient.flatten()) @ A.T @ Jacobians + np.eye(d_dim)
            else:
                Metrics = Jacobians.T @ A @ M_ambient.squeeze() @ A.T @ Jacobians + np.eye(d_dim)
        else:
            if manifold_ambient.is_diagonal():
                Metrics = Jacobians.T @ np.diag(M_ambient.flatten()) @ Jacobians + np.eye(d_dim)
            else:
                Metrics = Jacobians.T @ M_ambient.squeeze() @ Jacobians + np.eye(d_dim)

    return Metrics, Jacobians
"""

# A function to open a 3d plot
def my_3d_plot_axis():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    return ax


# Load the dataset
def load_data(file_name, batches_size=256, test_per=0.1):
    # Load the data and get the parameters
    try:
        mat_contents = sio.loadmat(file_name)
    except:
        mat_contents = dill.load(open(file_name,"rb"))
    data = mat_contents['data'].astype(np.float32)
    try:
        labels = mat_contents['labels'].astype(np.float32)
    except Exception:
        labels = np.ones((data.shape[0], 1)) * np.nan

    # Initialize the test and train data and batches
    train_data = np.nan
    test_data = np.nan
    train_labels = np.nan
    test_labels = np.nan
    train_batches_inds = np.nan
    test_batches_inds = np.nan

    # The number of points for the train and test data
    N = data.shape[0]  # Number of points in the dataset
    N_test = int(np.rint(test_per * N))  # Number of test samples
    N_train = N - N_test  # Number of training samples

    # The number of batches
    train_batch_size = batches_size  # This is given by the user
    batches_num = int(N_train / train_batch_size)  # The number of batches
    # The batch sizes for train and test
    test_batch_size = int(N_test / batches_num)

    N_train = int(batches_num * train_batch_size)
    N_test = int(batches_num * test_batch_size)

    # Sample randomly the dataset
    subsample_inds = np.random.choice(N, size=N_train + N_test, replace=False)
    data = data[subsample_inds, :]  # Subsample the dataset, to have equal sets
    labels = labels[subsample_inds, :]  # Subsample the labels, to have equal sets

    # Separate train-test set and shuffle first
    ind_list = [i for i in range(N_train + N_test)]  # All the indices as a list [1,2,3,...,N]
    np.random.shuffle(ind_list)  # Shuffle the dataset
    data = data[ind_list, :]
    labels = labels[ind_list, :]

    train_data = data[:N_train, :]  # Keep the train points
    test_data = data[N_train:, :]  # Keep the test points

    train_labels = labels[:N_train, :]  # Keep the train points
    test_labels = labels[N_train:, :]  # Keep the test points

    # If we need to use multiple batches
    if batches_num > 1:

        # Produce some sets of indices without replacement
        train_batches_inds = np.random.choice(N_train,
                                              size=(batches_num, train_batch_size),
                                              replace=False)

        test_batches_inds = np.random.choice(N_test,
                                             size=(batches_num, test_batch_size),
                                             replace=False)

        return train_data, train_labels, train_batches_inds, test_data, test_labels, test_batches_inds

    # If we have to return the whole dataset
    return train_data, train_labels, train_batches_inds, test_data, test_labels, test_batches_inds


# Plot an image tha contains all the images given.
def plot_image_of_images(Imgs, N_rows, N_cols, height, width):
    # Imgs: N x D the array of images, each row is one image.
    # height x width: the iamge that we want to process.

    for n_rows in range(N_rows):
        # Construct a row of images
        for n_cols in range(N_cols):
            img_counter = n_rows * N_cols + n_cols  # The image indicator
            # when a new row starts
            if n_cols == 0:
                row_imgs = Imgs[:, img_counter].reshape(height, width)
            else:
                row_imgs = np.concatenate((row_imgs, Imgs[:, img_counter].reshape(height, width)), axis=1)

        # If is the first row do nothing, else concatinate
        if n_rows == 0:
            Imgs_ours_final = row_imgs
        else:
            Imgs_ours_final = np.concatenate((Imgs_ours_final, row_imgs), axis=0)

    plt.figure()
    plt.imshow(utils.clip_values(Imgs_ours_final, min_val=-1, max_val=1), cmap=cm.binary_r, interpolation='bicubic')

    return True


# Clip the values of all the elements in x.
def clip_values(x, min_val, max_val):
    temp_x = x.copy()
    temp_x[x > max_val] = max_val
    temp_x[x < min_val] = min_val
    return temp_x


# Plots an image from the values in vals.
def my_imshow(vals, z_intervals=None, cmapSel=cm.RdBu_r):
    # vals: N*N x 1 vector
    # z_intervals = (z1min, z1max, z2min, z2max)
    N = int(np.sqrt(vals.shape[0]))
    if z_intervals is None:
        plt.imshow(vals.reshape(N, N), interpolation='none', origin='lower',
                   cmap=cmapSel, aspect='equal')
    else:
        plt.imshow(vals.reshape(N, N), interpolation='none', origin='lower',
                   extent=z_intervals, cmap=cmapSel, aspect='equal')


# Returns a matrix with the meshgrid points
def my_meshgrid(x1min, x1max, x2min, x2max, N=10):
    X1, X2 = np.meshgrid(np.linspace(x1min, x1max, N), np.linspace(x2min, x2max, N))
    X = np.concatenate((np.expand_dims(X1.flatten(), axis=1), np.expand_dims(X2.flatten(), axis=1)), axis=1)
    return X


# This function implements the PCA, and returns eigenvectors, eigenvalues sqrt and the mean
def my_pca(X, d):
    X_mean = X.mean(axis=0)
    Cov_X = (X - X_mean).T @ (X - X_mean) / X.shape[0]
    eigenValues, eigenVectors = np.linalg.eigh(Cov_X)
    idx = eigenValues.argsort()[::-1]  # Sort the eigenvalues from max -> min
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]
    V = eigenVectors[:, 0:d]  # The projection matrix  # The projection matrix
    L = eigenValues[0:d]
    mu = X_mean.reshape(-1, 1)  # The center of the dataset
    return V, L, mu


# Give the boundaries around the given 2d dataset
def my_boundaries(data=None, offset=0):
    if data.shape[1] > 2:
        print('Data dimension cannot be used (D > 2)!')
        return None
    else:
        z1min = data[:, 0].min() - offset
        z1max = data[:, 0].max() + offset
        z2min = data[:, 1].min() - offset
        z2max = data[:, 1].max() + offset

        return z1min, z1max, z2min, z2max


# Fast quiver implementation in 2d
def my_quiver(x, v):
    plt.quiver(x[0], x[1], v[0], v[1], angles='xy', scale_units='xy', scale=0.1, color='r')


# Makes a Dx1 vector given x
def my_vector(x):
    return np.asarray(x).reshape(-1, 1)


# Synthetic datasets
def generate_data(N=200, data_type=1):
    # The semi-circle data in 2D
    if data_type == 1:
        theta = np.pi * np.random.rand(N, 1)
        data = np.concatenate((np.cos(theta), np.sin(theta)), axis=1) + 0.1 * np.random.randn(N, 2)

    # Semi-sphere, the upper part in 3D
    elif data_type == 2:
        data = np.random.randn(2 * N, 3)
        data = data / np.sqrt(np.sum(data ** 2, axis=1, keepdims=True))
        data = data[data[:, 2] >= 0, :]  # keep only the ones with positive z dimension
        data = data + 0.1 * np.random.randn(data.shape[0], 3)

    # Circle data in 2D
    elif data_type == 3:
        theta = 0 + np.pi * 2 * np.random.rand(N, 1)
        data = np.concatenate((np.cos(theta), np.sin(theta)), axis=1) + 0.1 * np.random.randn(N, 2)

    # The two-moons data in 2D
    elif data_type == 4:
        N = int(N / 2)
        theta = np.pi * np.random.rand(N, 1)
        data1 = np.concatenate((np.cos(theta), np.sin(theta)), axis=1) + 0.0 * np.random.randn(N, 2)
        data2 = np.concatenate((np.cos(theta), -np.sin(theta)), axis=1) + 0.0 * np.random.randn(N, 2) \
                + utils.my_vector([1.0, +0.25]).T
        data = np.concatenate((data1, data2), axis=0)
        data = data - data.mean(0).reshape(1, -1)
        data += 0.15 * np.random.randn(int(N * 2), 2)

    elif data_type == 5:
        N = int(N / 2)
        Id = np.eye(2)
        T = np.ones((2, 2)) - Id
        Id[1, 1] = 0.4
        Id[0, 0] = 3
        S1 = Id + 0.85 * T
        S2 = Id - 0.85 * T
        data1 = np.random.multivariate_normal([1, 1], S1, N)
        data2 = np.random.multivariate_normal([1, -1], S2, N)
        data = np.concatenate((data1, data2), axis=0)

    return data


# Plots easily data in 2d or 3d
def my_plot(x, **kwargs):
    if x.shape[1] == 2:
        plt.scatter(x[:, 0], x[:, 1], **kwargs)
        plt.axis('equal')
    if x.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], **kwargs)


# Plots the measure of the Riemannian manifold
def plot_measure(manifold, linspace_x1, linspace_x2, isLog=True, cmapSel=cm.RdBu_r, ax=None):
    X1, X2 = np.meshgrid(linspace_x1, linspace_x2)
    X = np.concatenate((np.expand_dims(X1.flatten(), axis=1), np.expand_dims(X2.flatten(), axis=1)), axis=1)
    M = manifold.metric_tensor(X.transpose(), nargout=1)

    if manifold.is_diagonal():
        img = np.reshape(np.sqrt(np.prod(M, axis=1)), X1.shape)
    elif not manifold.is_diagonal():
        N = M.shape[0]
        img = np.zeros((N, 1))
        for n in range(N):
            img[n] = np.sqrt(np.linalg.det(np.squeeze(M[n, :, :])))
        img = img.reshape(X1.shape)

    if isLog:
        img = np.log(img + 1e-10)
    else:
        img = img

    if ax is None:
        plt.figure(figsize=(10,10))
        plt.imshow(img, interpolation='gaussian', origin='lower',
                   extent=(linspace_x1.min(), linspace_x1.max(), linspace_x2.min(), linspace_x2.max()),
                   cmap=cmapSel, aspect='equal')
    else:
       ax.imshow(img, interpolation='gaussian', origin='lower',
                   extent=(linspace_x1.min(), linspace_x1.max(), linspace_x2.min(), linspace_x2.max()),
                   cmap=cmapSel, aspect='auto') 


# Draws an elipsoid that correspond to the metric
def plot_metric(x, cov, color='r', inverse_metric=False, plot=True):
    eigvals, eigvecs = np.linalg.eig(cov)
    N = 100
    theta = np.linspace(0, 2 * np.pi, N)
    theta = theta.reshape(N, 1)
    points = np.concatenate((np.cos(theta), np.sin(theta)), axis=1)
    points = points * np.sqrt(eigvals)
    points = np.matmul(eigvecs, points.transpose()).transpose()
    points = points + x.flatten()
    if plot:
        plt.plot(points[:, 0], points[:, 1], c=color)
    return points


# Draws an elipsoid that correspond to the metric
def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    from matplotlib.patches import Ellipse
    ax = ax or plt.gca()

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))
