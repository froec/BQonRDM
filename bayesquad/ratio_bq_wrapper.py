import GPy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy import newaxis

from .bayesquad.batch_selection import select_batch, LOCAL_PENALISATION, KRIGING_BELIEVER, KRIGING_OPTIMIST
from .bayesquad.gps import WsabiLGP, WsabiMGP, LogMGP
from .bayesquad.priors import Gaussian
from .bayesquad.quadrature import IntegrandModel
from .bayesquad.acquisition_functions import model_variance
from .bayesquad._transformations import log_of_function
from .bayesquad._optimisation import multi_start_maximise

import time
import matplotlib.colors
import pickle
import sys

import scipy.stats as stats


class RatioBQWrapper():

    def __init__(self, dim,  x_i, y_i, mu=None, Z=None, n_grad=200, plot=True, logger=None, land=None):
        self.dim = dim
        self.X = np.empty((0,dim), float)
        self.Y = np.empty((0,1), float)
        
        self.k = k
        
        if mu is not None:
            self.mu = mu
        
        if Z is not None:
            self.Z = Z
            
        
        self.n_grad = n_grad
        
        self.plot = plot
        self.logger = logger
        self.land = land # useful for the logmaps

        self.x_i = x_i
        self.y_i = y_i

        print("x_i:")
        print(self.x_i)
        print("y_i:")
        print(self.y_i)
        
        self.time_start = time.time()


    def wsabi_integrate(self, f, last_fi_xs, last_fi, last_fj_xs, last_fj, n_batches, batch_size, \
                        variance=1.,lengthscale=1.,\
                        prior_mean=np.array([0., 0.]), prior_cov=np.eye(2),\
                        grad=False, integration_params={'bq_method' : 'WSABI-L'}, component_k=0):
        print("calling wsabi_integrate...")
        verbose = integration_params.get('verbose') if integration_params.get('verbose') is not None else False
        savedir = integration_params.get('savedir')
        print(self.mu)
        print(prior_cov)
        self.prior_cov = prior_cov


        batch_method = KRIGING_OPTIMIST
        #batch_method = KRIGING_BELIEVER
        D = self.dim
        
        k = GPy.kern.RBF(D, variance=variance, lengthscale=lengthscale) 
        lik = GPy.likelihoods.Gaussian(variance=1e-10)
        
        k['.*lengthscale'].constrain_bounded(0.2,20.)


        prior = Gaussian(mean=prior_mean, covariance=prior_cov)

        mf = GPy.core.Mapping(D,1)
        mf.f = lambda x: 1.
        mf.update_gradients = lambda a,b: None

        print("the last f_i's are:")
        print(last_fi)

        print("the last f_j's are:")
        print(last_fj)

        gpy_gp = GPy.core.GP(last_xs, np.sqrt(2*last_fj/lastfi), kernel=kernel, likelihood=lik, mean_function=mf)
        warped_gp = WsabiLGP(gpy_gp)

        model = IntegrandModel(warped_gp, prior)
        


        for i in range(n_batches):

            tb = time.time()
            #batch = select_batch(model, batch_size, batch_method)
            batch = self.select_point(model)
            #print(batch)
            #if verbose:
            #    tb = time.time() - tb
            #    print("time for batch selection (%s): %s" % (batch_size, tb))
            batch = np.vstack(batch)
            
            X = np.array(batch)
            Y = f(X)
            #print(Y)
                        
            self.X = np.vstack([self.X, X])
            self.Y = np.vstack([self.Y, Y])
            
            model.update(X, Y)