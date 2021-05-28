from typing import Dict, Any

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
from .bayesquad._maths_helpers import jacobian_of_f_squared_times_g

# for DCV
from pymanopt.manifolds import Sphere
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent, ConjugateGradient, NelderMead, TrustRegions
from pymanopt.solvers.linesearch import LineSearchBackTracking, LineSearchAdaptive

from scipy.integrate import simps, quad
import time
import matplotlib.colors
import pickle
import sys
import dill

import scipy.stats as stats
from scipy.optimize import minimize


import scipy.special as ss
from sklearn.decomposition import PCA # for higher-dimensional visualization

import io
from contextlib import redirect_stdout



class BQWrapper:
    # this is a wrapper class, accesses from land_quadrature.py
    # provides all the necessary BQ functionality
    # makes use itself of the bayesquad library
    def __init__(self, dim, mu=None, Z=None, k=None, n_grad=200, plot=True, logger=None, land=None):
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
        
        self.time_start = time.time()
        
    
    
    def select_point(self, model):
        init = stats.multivariate_normal(mean=np.zeros(self.dim).reshape(-1,), cov=self.prior_cov).rvs(10)
        point, _ = multi_start_maximise(log_of_function(model_variance(model)), init)
        return np.atleast_2d(point)



    def dcv(self, model, gpy_gp, warped_gp, expmap, f_manifold, X, Y):
        """ Directional Cumulative Acquisition
        """        

        Gamma = np.linalg.inv(self.prior_cov)
        """ First we define some helper functions for the optimization 
        of the acquisition function
        """
        def objective_scaledunitvec(b, r):
            """
            Uncertainty sampling objective expressed through a scalar b and a unit vector r,
            i.e. x=b*r
            This is the acquisition function
            """
            _, var = warped_gp.posterior_mean_and_variance(b*r)
            return np.float(var) * self.int_measure.pdf(b*r)**2

        def gradient_scaledunitvec(b, r, d=None):
            """
            Gradient of the uncertainty sampling objective expressed through a scalar b and a unit vector r,
            i.e. x=b*r
            This is wrt to r and it is a euclidean gradient, which will then be projected
            on the hypersphere.
            """
            x = b*r
            nx = self.int_measure.pdf(x)
            gp_mean, gp_variance = warped_gp.posterior_mean_and_variance(x)
            vx = np.float(gp_variance)
            nx_prime = (-nx * Gamma @ x.flatten()).flatten()
            vx_prime = warped_gp.posterior_variance_jacobian(x).flatten()

            """
            # alternative (inlined), but hardly any speed gain
            gp_mean_jacobian, gp_variance_jacobian = gpy_gp.predictive_gradients(np.atleast_2d(x))
            gp_mean_jacobian = np.squeeze(gp_mean_jacobian, axis=-1)


            vx_prime = jacobian_of_f_squared_times_g(
                f=np.array([gp_mean]).reshape(1,), f_jacobian=gp_mean_jacobian,
                g=np.array([gp_variance]), g_jacobian=gp_variance_jacobian).flatten()
            """
            
            output = b * (2 * nx * vx * nx_prime + nx**2 * vx_prime).flatten()
            if d is None:
                return output
            else:
                return output[d]


        #### DRAW CUSTOM ELLIPSE at chi2=0.998
        # https://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(self.prior_cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:,order]

        eigvals, eigvecs = eigsorted(self.prior_cov)
        chivalue = stats.chi2(df=2).ppf(0.995) # was 0.998
        

        def getBetaLimit(r):
            r /= np.linalg.norm(r)
            radius = np.sqrt(1 / (r.T @ Gamma @ r.T) * chivalue)
            return radius


        global cost_evals
        cost_evals = 0
        def cost(r):
            global cost_evals
            cost_evals += 1
            #print(r)
            c = quad(objective_scaledunitvec, 0., getBetaLimit(r), limit=30, epsrel=0.01, args=(r))
            return -c[0]

        global grad_evals
        grad_evals = 0
        # this is very slow!
        def egrad(r):
            global grad_evals 
            grad_evals += 1
            qs = []
            for d in range(r.shape[0]):
                tt = time.time()
                g = quad(gradient_scaledunitvec, 0., getBetaLimit(r), limit=30, epsrel=0.01, args=(r,d))
                qs.append(g[0])
            qs = -np.hstack(qs)
            return qs

        # this numeric gradient is much faster
        # all dimensions can be integrated at once
        # uses simpsons rule
        def egrad_num(r):
            t = np.linspace(0.0, getBetaLimit(r), 50)
            fevals = [gradient_scaledunitvec(t_, r0) for t_ in t]
            return np.array([-simps(np.array(fevals)[:,d], t) for d in range(r.shape[0])])


        manifold = Sphere(self.dim)
        problem = Problem(manifold=manifold, cost=cost, egrad=egrad_num, verbosity=0) # , verbosity=0
        linesearch = LineSearchBackTracking(maxiter=5, optimism=2.0, initial_stepsize=1.0)
        solver = SteepestDescent(maxiter=15, linesearch=linesearch, minstepsize=1e-10)


        for n_iter in range(1):
            r0 = stats.multivariate_normal(mean=np.zeros(self.dim).reshape(-1,), cov=self.prior_cov).rvs(1)
            r0 /= np.linalg.norm(r0)
            
            tt = time.time()
            Xopt = solver.solve(problem, x=r0.reshape(-1,), reuselinesearch=False)


            # now solve the exponential map to Xopt
            # TODO: determine scaling in a reasonable way
            scaling = float(getBetaLimit(Xopt))
            n_points = 6
            n_discrete = 30 # discretize the vector to check for highest variance
            # TODO: this is too long!
            # select the first point, then compute the expmap to that point
            # or perhaps we could compute half/three quarters of the expmap already
            # because setting up expmaps might take time
            # then select the second point
            # if it's further away, extend the expmap
            # to do so, compute a new partial expmap from the last farthest point
            # continue in this fashion
            ttt = time.time()
            curve = expmap(scaling*Xopt)


            ts = np.linspace(0.02,1.,n_discrete) 
            for x_i in range(n_points):
                maxv = float('-inf')
                maxt = None
                tt = time.time()
                for t in ts:
                    l = (scaling*t*Xopt.reshape(-1,1)).T
                    _,v = model.posterior_mean_and_variance(l) # was warped_bq before
                    if v > maxv:
                        maxv = v
                        maxt = t
                #print(maxt)
                maxl_manifold = curve(maxt)[0] # where the point lies on the manifold
                newx = (scaling*maxt*Xopt.reshape(-1,1)).T
                #plt.scatter(newx[:,0],newx[:,1],c='b')
                X = np.vstack([X,newx])
                fx = f_manifold(maxl_manifold.reshape(-1,1))
                Y = np.vstack([Y,fx])
                model.update(newx, fx)

        
        
        return X,Y
        
    

    def wsabi_integrate(self, f, initial_x, initial_y, n_batches, batch_size, \
                        variance=1.,lengthscale=1., constant_mean=0.,\
                        prior_mean=np.array([0., 0.]), prior_cov=np.eye(2),\
                        grad=False, integration_params={'bq_method' : 'WSABI-L'}, component_k=0, expmap=None, f_manifold=None):
        # this is the access point for the land_quadrature.py module

        verbose = integration_params.get('verbose') if integration_params.get('verbose') is not None else False
        savedir = integration_params.get('savedir')
        print(self.mu)
        print(prior_cov)
        # we save both the covariance matrix
        self.prior_cov = prior_cov
        # and the integration measure as a callable
        # use: self.int_measure.pdf(x)
        self.int_measure = stats.multivariate_normal(mean=np.zeros(self.dim).reshape(-1,), cov=self.prior_cov)

        # is there a timelimit? (used for experiments)
        if integration_params.get('timelimit') is None:
            self.timelimit = None
        else:
            self.timer = time.time()
            self.timelimit = integration_params.get('timelimit')
        
        # currently not used anyway
        batch_method = KRIGING_OPTIMIST
        #batch_method = KRIGING_BELIEVER
        D = self.dim

        # if dcv is specified, then we need the exponential map
        if integration_params.get("dcv") or integration_params.get("DCV"):
            if expmap is None or f_manifold is None:
                sys.exit("for DCV, the BQWrapper needs access to the exponential map and function, which \
                    is evaluated on the manifold.")

        ard = False # default: isotropic kernel
        if integration_params.get('kernel') is None or integration_params.get('kernel') == 'rbf':
            kernel_name = "rbf"
            if integration_params.get('ard') is None or integration_params.get('ard') is False:
                k = GPy.kern.RBF(D, variance=variance, lengthscale=lengthscale) 
            elif integration_params.get('ard'):
                ard = True
                k = GPy.kern.RBF(D, variance=variance, lengthscale=lengthscale*np.ones(D), ARD=True) 
        elif integration_params.get('kernel') == 'matern52':
            kernel_name = "matern52"
            if integration_params.get('ard') is None or integration_params.get('ard') is False:
                k = GPy.kern.Matern52(D, variance=variance, lengthscale=lengthscale) 
            elif integration_params.get('ard'):
                ard = True
                k = GPy.kern.Matern52(D, variance=variance, lengthscale=lengthscale*np.ones(D), ARD=True) 
        elif integration_params.get('kernel') == 'ratquad':
            k = GPy.kern.RatQuad(D, variance=variance, lengthscale=lengthscale, power=1.0)
        else:
            sys.exit("oops, no kernel found matching the integration parameters.")

        lik = GPy.likelihoods.Gaussian(variance=1e-10)


        k['.*lengthscale'].fix(warning=False)
        k['.*variance'].fix(warning=False)
        k['.*lengthscale'].constrain_bounded(0.1,20.) # was 0.2 before
        prior = Gaussian(mean=prior_mean, covariance=prior_cov)

        if initial_x.shape[0] > 0 and self.k is not None and (verbose or True):
            # we reuse the kernel only if the mean didnt change
            # so that pathological hyperparameters are not propagated for too long
            print("reusing kernel!")
            k = self.k

        if self.k is None:
            print("no previous kernel")
        
        # TODO: something smarter
        # need first sample to initialize GP
        initial_x_ = 0.2*stats.multivariate_normal().rvs(D).reshape(1,D)
        initial_y_ = f(initial_x_)
        
        if verbose:        
            print("initial_x_:")
            print(initial_x_)
            initial_y_ = f(initial_x_)
            print("initial_y_:")
            print(initial_y_)
        
        
        if integration_params.get('prior_mean_scaling'):
            constant_mean *= integration_params.get('prior_mean_scaling')
            #print("scaled constant mean: %s" % constant_mean)
        
        kernel = k
        
        kwargs = {}
        bq_method = integration_params['bq_method'] if integration_params['bq_method'] is not None else 'WSABI-L'
        if constant_mean > 0.:
            if bq_method == 'WSABI-L' or bq_method == 'WSABI-M':
                constant_mean_warped = np.sqrt(2*constant_mean)
            elif bq_method == 'LOG':
                constant_mean_warped = np.log(constant_mean)
            if verbose:
                print("using constant mean: %.2f" % constant_mean)
                print("warped mean:")
                print(constant_mean_warped)

            mf = GPy.core.Mapping(D,1)
            mf.f = lambda x: constant_mean_warped
            mf.update_gradients = lambda a,b: None
            kwargs = {'mean_function' : mf}
        
        #### IMPORTANT: APPLY THE INITIAL WARPING (alpha=0.0 for WSABI)
        if bq_method == 'WSABI-L' or bq_method == 'WSABI-M':
            gpy_gp = GPy.core.GP(initial_x_, np.sqrt(2*initial_y_), kernel=kernel, likelihood=lik, **kwargs)
            if bq_method == 'WSABI-L':
                warped_gp = WsabiLGP(gpy_gp)
            elif bq_method == 'WSABI-M':
                warped_gp = WsabiMGP(gpy_gp)
            
        elif bq_method == 'LOG':
            gpy_gp = GPy.core.GP(initial_x_, np.log(initial_y_), kernel=kernel, likelihood=lik, **kwargs)
            warped_gp = LogMGP(gpy_gp)
            
        
        model = IntegrandModel(warped_gp, prior)

        if len(initial_x) > 0:
            model.update(initial_x, initial_y)
            self.X = np.vstack([self.X, initial_x])
            self.Y = np.vstack([self.Y, initial_y])
               
        if verbose:
            print("kernel before first batch:")
            print(k)
        
        
        for i in range(n_batches):
            if self.timelimit is not None:
                if time.time() - self.timer > self.timelimit:
                    print("timelimit reached, break!")
                    break

            tb = time.time()
            #batch = select_batch(model, batch_size, batch_method)
            if integration_params.get("DCV") or integration_params.get("dcv"):
                self.X, self.Y = self.dcv(model, gpy_gp, warped_gp, expmap, f_manifold, self.X, self.Y) 
                with io.StringIO() as buf, redirect_stdout(buf):
                    gpy_gp.optimize() # after every new line
            else:
                batch = self.select_point(model)
                batch = np.vstack(batch)
            
                X = np.array(batch)
                Y = f(X)
                self.X = np.vstack([self.X, X])
                self.Y = np.vstack([self.Y, Y])
                model.update(X, Y)

                if i > 5:
                    with io.StringIO() as buf, redirect_stdout(buf):
                        gpy_gp.optimize()

            
            

            if verbose and i % 5 == 0:
                print("kernel after batch:")
                print(k['.*lengthscale'])
                print(k['.*variance'])
                if integration_params.get("kernel") == 'ratquad':
                    print(k['.*power'])
        
        # save the kernel for later use
        self.k = k
        
        
        if verbose:
            print("final lengthscale: ")
            print(k['.*lengthscale'])
            print("final variance: ")
            print(k['.*variance'])
            
        print("final number of collected observations: %s" % self.X.shape[0])

        
        # trick:  predict with monte carlo
        predict_using_mc = True
        if predict_using_mc:
                        
            
            L, U = np.linalg.eigh(prior_cov)
            A = U @ np.diag(np.sqrt(L))
            if verbose:
                print("using MC samples: %s" % self.land.V_samples.shape[1])
            randsT = (A @ np.random.randn(D,self.land.V_samples.shape[1])).T

            preds_r_ms, preds_r_vs = gpy_gp.predict(randsT)
            #print(randsT.shape)
            preds_unwarped = warped_gp._unwarp(preds_r_ms)


            
            if verbose:
                faraway = 1000*np.ones(D).reshape(1,-1)
                print("underlying GP asymptotic:")
                print(gpy_gp.predict(faraway))
                print("alpha:")
                print(warped_gp._alpha[0,0])
                print("asymptotic warped gp mean and variance:")
                print(warped_gp.posterior_mean_and_variance(faraway))
                
            
            
            postmeans, postvars = warped_gp.posterior_mean_and_variance(randsT)


            
            post_int_estimate = np.mean(postmeans)
            print("integral estimate:")
            print(post_int_estimate*self.Z)
            integral = post_int_estimate
            
            #print("exact integral estimate:")
            #exact_estimate = np.mean(preds_unwarped)
            #print(np.mean(preds_unwarped)*self.Z)


            if bq_method == 'LOG':
                integral = exact_estimate # the other one is too bad

            
        else:
            if bq_method == 'WSABI-L':

                integral = model.integral_mean()
                print("integral calculation successfull")
            else:
                sys.exit("analytic interation not supported for method %s" % bq_method)


       

        # calculate the variance for the 2nd term
        # which is also the variance for the whole gradient
        # assuming that the logmaps are noise-free
        # here we consider the lower term fixed
        outers = []
        for it, rt in enumerate(randsT):
            rt = rt.reshape(-1,1)
            outers.append(preds_unwarped[it] * rt @ rt.T)
        
        Vtild = np.var(outers, axis=0)
        mc_variance = 1/preds_unwarped.shape[0] * Vtild



        if D == 2:
            # predict within a certain range
            n_pred = 100
            scale_fac = 0.1
            xdiff = np.max(self.X[:,0]) - np.min(self.X[:,0])
            ydiff = np.max(self.X[:,0]) - np.min(self.X[:,0])
            x = np.linspace(np.min(self.X[:,0]) - scale_fac*xdiff, np.max(self.X[:,0])+scale_fac*xdiff, n_pred)
            y = np.linspace(np.min(self.X[:,1]) - scale_fac*xdiff, np.max(self.X[:,1])+scale_fac*ydiff, n_pred)
            vs = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
            

            means, variances = model.posterior_mean_and_variance(vs)
            means = means.reshape(n_pred,n_pred)
        else:
            
            x = None
            y = None
            means = None
            variances = None
            obs_logs_pca = None
            
            # this can be commented in to save debug data for higher-dimensional cases
            # using a PCA to visulize the posterior
            """
            # we do a PCA on the data
            pca = PCA(n_components=2)
            X = pca.fit_transform(self.land.data)
            print("explained var:")
            print(pca.explained_variance_ratio_)
            mean_pca = pca.transform(np.atleast_2d(self.land.means[component_k]))
            plt.figure()
            plt.scatter(X[:,0],X[:,1],s=7)
            observations_manifold = np.vstack([np.array(expmap(x)(1)[0]).reshape(1,-1) for x in self.X])
            print(observations_manifold.shape)
            obs_pca = pca.transform(observations_manifold)
            plt.scatter(obs_pca[:,0], obs_pca[:,1], c='r',s=10,marker="x")
            plt.scatter(mean_pca[:,0], mean_pca[:,1],c='r',marker="D",s=50)
            #plt.show()
            plt.close()

            
            pca_logmaps = PCA(n_components=2)
            logs = self.land.logmaps[component_k]
            inds = np.isnan(logs[:, 0])
            logs = logs[~inds, :] 
            X_logs = pca_logmaps.fit_transform(logs)
            print("logmaps pca explained var:")
            print(pca_logmaps.explained_variance_ratio_)
            plt.figure()
            plt.scatter(X_logs[:,0], X_logs[:,1])
            #plt.show()
            plt.close()

            obs_logs_pca = pca_logmaps.transform(self.X)
            n_pred = 100
            scale_fac = 0.1
            xdiff = np.max(obs_logs_pca[:,0]) - np.min(obs_logs_pca[:,0])
            ydiff = np.max(obs_logs_pca[:,0]) - np.min(obs_logs_pca[:,0])
            x = np.linspace(np.min(obs_logs_pca[:,0]) - scale_fac*xdiff, np.max(obs_logs_pca[:,0])+scale_fac*xdiff, n_pred)
            y = np.linspace(np.min(obs_logs_pca[:,1]) - scale_fac*xdiff, np.max(obs_logs_pca[:,1])+scale_fac*ydiff, n_pred)
            vs = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])

            means, variances = model.posterior_mean_and_variance(pca_logmaps.inverse_transform(vs))
            means = means.reshape(n_pred,n_pred)
            """

        # higher dimensional plots
        #plt.figure()
        #plt.title("Z=%.4f, preds_unwarped_mean=%.2f" % (self.Z, np.mean(preds_unwarped)))
        #plt.hist(preds_unwarped,bins=40)
        #plt.show()

        #print("max observation: %.2f" % np.max(self.Y))
        #print("max prediction: %.2f" % np.max(preds_unwarped))
        
        
        if self.logger is not None:
            if savedir is None:
                savedir = "data-test/"

            _means = self.land.means.copy()
            _sigmas = self.land.sigmas.copy()
            _consts = self.land.consts
            _means[component_k] = self.mu
            _sigmas[component_k] = prior_cov 
            _consts[component_k] = integral*self.Z
            
            if ard:
                lscale = np.array(k['.*lengthscale'])
                print("lscale:")
                print(lscale)
            else:
                lscale = float(k['.*lengthscale'])
            # TODO: save the lengthscale
            res_dict = {'mu' : self.mu, 'Sigma' : prior_cov, 'Z' : self.Z, 'X' : self.X, 'Y' : self.Y, 'm_int' : integral, \
                        'exact_estimate' : exact_estimate,\
                        'X_pca' : obs_logs_pca if D > 2 else None,\
                        'x_preds' : x, 'y_preds' : y, 'pred_means' : means, 'pred_variances' : variances,\
                       'lengthscale' : lscale, 'variance' : float(k['.*variance']),\
                       'runtime' : (time.time() - self.time_start),\
                       'k' : component_k,\
                       'all_mus' : _means,\
                       'all_sigmas' : _sigmas,\
                       'all_logmaps' : self.land.logmaps,\
                       'all_consts' : _consts,\
                       'weights' : self.land.weights}
            self.logger(res_dict, savedir)
            
            if self.plot:

                import matplotlib.cm as cm
                plt.figure(figsize=(12,8))
                plt.contourf(x,y,means,cmap='magma',levels=100)
                print("posterior mean range:")
                print((np.min(means),np.max(means)))
                plt.scatter(self.X[:,0],self.X[:,1],c='white',s=4)
                m = plt.cm.ScalarMappable(cmap=cm.magma)
                m.set_array(means)
                #m.set_clim(1.0826224448865052e-15,12.075679276496833)
                plt.colorbar(m)
                plt.axis('off')
                plt.tight_layout()
                plt.show()


                plt.figure(figsize=(20,4))
                plt.subplot(1,5,1)
                plt.title("posterior mean")
                plt.contourf(x,y,means,cmap='magma',levels=100)
                plt.scatter(self.X[:,0],self.X[:,1],c='white',marker='x',s=7)
                for yi, y_ in enumerate(self.Y):
                    if y_[0] > 5 and y_[0] < 20:
                        plt.scatter(self.X[yi,0],self.X[yi,1],c='b',marker='x',s=7)
                    if y_[0] > 20:
                        plt.scatter(self.X[yi,0],self.X[yi,1],c='r',marker='x',s=7)
                        
                #### DRAW CUSTOM ELLIPSE at chi2=0.998
                # https://stackoverflow.com/questions/12301071/multidimensional-confidence-intervals
                def eigsorted(cov):
                    vals, vecs = np.linalg.eigh(cov)
                    order = vals.argsort()[::-1]
                    return vals[order], vecs[:,order]

                eigvals, eigvecs = eigsorted(prior_cov)
                chivalue = stats.chi2(df=2).ppf(0.995)
                N = 200
                theta = np.linspace(0, 2 * np.pi, N)
                theta = theta.reshape(N, 1)
                points = np.concatenate((np.cos(theta), np.sin(theta)), axis=1)
                points = points * np.sqrt(eigvals * chivalue)
                points = np.matmul(eigvecs, points.transpose()).transpose()
                xmin = np.min(points[:,0])
                xmax = np.max(points[:,0])
                #plt.xlim(xmin,xmax)
                plt.plot(points[:, 0], points[:, 1], c='red', alpha=.6)

                plt.subplot(1,5,2)
                plt.title("raw GP means")
                # raw gp
                meansraw, varsraw = gpy_gp.predict(vs)
                plt.contourf(x,y,meansraw.reshape(n_pred,n_pred),cmap='magma',levels=50)
                plt.colorbar()      
                
                plt.subplot(1,5,3)
                plt.title("unwarped GP means")
                meansw, varsw = warped_gp.posterior_mean_and_variance(vs)
                plt.contourf(x,y,meansw.reshape(n_pred,n_pred),cmap='magma',levels=50)
                plt.colorbar()     
                
                plt.subplot(1,5,4)
                plt.title("unwarped posterior variance")
                plt.contourf(x,y,variances.reshape(n_pred,n_pred),cmap='magma',levels=100)
                #plt.scatter(self.X[:,0],self.X[:,1],c='white',marker='x',s=30)
                plt.colorbar()
                
                plt.subplot(1,5,5)
                plt.title("raw GP variance")
                plt.contourf(x,y,varsraw.reshape(n_pred,n_pred),cmap='magma',levels=100)
                plt.colorbar()
                
                plt.show()
            
    
            
        return integral, randsT, postmeans, mc_variance

