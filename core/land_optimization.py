import numpy as np
from core import geodesics
import time
from pymanopt.manifolds import PositiveDefinite
from pymanopt import Problem
from pymanopt.solvers.linesearch import LineSearchAdaptive
from core.LineSearchCustom import LineSearchCustom, LineSearchBackTrackingCustom
from core.steepest_descent import SteepestDescent
import operator
from abc import ABC
import matplotlib.pyplot as plt
import pickle
import sys

class LandOptimizer(ABC):
    """ This abstract class represents an abstract LAND optimizer
    which supplies methods to update the mean and to update Sigma.
    """
    
    def __init__(self, land):
        """ Initializes the optimizer with a LAND object
        """
        self.land = land


    def update_mean(self, k):
        """ Update the mean of component k.

        Parameters
        ----------
        k: int
            component for which the mean is updated.
        """
        raise NotImplementedError

    def update_Sigma(self,k):
        """ Update Sigma of component k.

        Parameters
        ----------
        k: int
            component for which Sigma is updated.
        """
        raise NotImplementedError

    def iteration_callback(self, negLogLikelihood, newNegLogLikelihood):
        """ This method is called after one "super-iteration" of updating 
        the mean, Sigma and the weights is complete.

        Parameters
        ----------
        negLogLikelihood
            the old negative log likelihood
        newNegLogLikelihood
            the new negative log likelihood, i.e., after the whole iteration is complete.
        """
        raise NotImplementedError



class AdaptiveRiemannianDescent(LandOptimizer):
    """ This adaptive Riemannian descent optimizer inherits from the LandOptimizer base class.
    It is adaptive in the sense that steepest descent is combined with adaptive line searches
    both for Sigma and mu.
    """

    def __init__(self, land):
        """ Initializes the optimizer with a LAND object
        and a linesearch object for sigma and another one for mu
        this enables reusing the linesearch parameters

        Parameters
        ----------
        land
            A LAND mixture model object.
        """
        self.land = land
        self.sigma_linesearch=LineSearchCustom(maxiter=self.land.model_params['max_iter_Sigma'], initial_stepsize=1.0)

        self.v = VanillaDescent(self.land)

        # for each component separately
        self.sigma_oldalpha = {k:None for k in range(self.land.K)}

        self.alpha0 = None
        self.alpha_stats = None

    
    def update_mean(self, k):
        return self.v.update_mean(k)


    def iteration_callback(self, negLogLikelihood, newNegLogLikelihood):
        """ Update the step-sizes after one "super-iteration" is complete, i.e.,
        mu, Sigma and the weights have been updated.
        """
        if newNegLogLikelihood < negLogLikelihood:
            self.land.model_params['step_size'] = 1.1 * self.land.model_params['step_size']
            print("increasing step size to %s" % self.land.model_params['step_size'])
        else:
            self.land.model_params['step_size'] = 0.75 * self.land.model_params['step_size']
            print("decreasing step size to %s" % self.land.model_params['step_size'])

        return True
    
        
    def update_Sigma(self, k):
        data = self.land.data
        N, D = data.shape
        Logmaps = self.land.logmaps[k, :, :]  # N x D
        mu = self.land.means[k, :].reshape(-1, 1)  # D x 1
        Sigma = self.land.sigmas[k, :, :].copy()  # D x D
        Resp = self.land.resp[:, k].reshape(-1, 1)  # N x 1
        Z_eucl = self.land.Z_eucl[k].copy()
        Const = self.land.consts[k].copy()
        V_samples = self.land.V_samples[k, :, :].copy()  # S x D, Use the samples from the last Const estimation
        volM_samples = self.land.volM_samples[:, k].reshape(-1, 1).copy()  # S x 1
        manifold = self.land.manifold
        solver = self.land.solver

        # Keep only the non-failed Logmaps
        inds = np.isnan(Logmaps[:, 0])
        # First term of the gradient doesn't change during iterations
        grad_term1 = Logmaps[~inds, :].T @ np.diag(Resp[~inds].flatten()) @ Logmaps[~inds, :] / Resp[~inds].sum()  
        
        # compute old likelihood first
        old_likelihood = self.land.compute_negLogLikelihood()
        print("")
        print("")
        print("updating Sigma:")
        print(Sigma)
        print("old likelihood: %s" % old_likelihood)
        #print("volM_samples:")
        #print(volM_samples.shape)

        Gamma = np.linalg.inv(Sigma)


        # we store both the likelihoods of the optimization steps
        # and the corresponding Gamma, Z_eucl, V_samples, volM_samples
        self.optimization_container = {}
        
        # this is a bit ugly, Pymanopt does not accept a function
        # for cost and gradient at the same time
        # so we memoize the results
        self.grad_container = {}
        self.cost_container = {}
        def cost(Sigma):
            c = self.cost_container.get(Sigma.tobytes())
            if c is None:
                c, g = cost_and_grad_Sigma(Sigma)
                #self.grad_container[Sigma.tobytes()] = g
                return c
            else:
                return c

        def grad(Sigma):
            g = self.grad_container.get(Sigma.tobytes())
            if g is None:
                print("The likelihood for Sigma=%s has not been calculated yet" % Sigma)
                # compute both but only return the gradient
                return cost_and_grad_Gamma(Sigma)[1]
            else:
                return g


        # this function computes the neg log likelihood (cost) for a given Sigma
        # for this, the normalization constant has to be estimated
        # then the gradient is essentially for free
        # the first term for the gradient does not change here
        # Keep only the non-failed Logmaps
        inds = np.isnan(Logmaps[:, 0])
        #term1 = 1/2*Logmaps[~inds, :].T @ np.diag(Resp[~inds].flatten()) @ Logmaps[~inds, :] # / Resp[~inds].sum()


        def cost_and_grad_Sigma(Sigma, compute_const=True):
            print("iteration: %s" % (len(self.optimization_container)))
            Gamma = np.linalg.inv(Sigma)
            Gamma = Gamma.reshape(D,D)


            # Compute the new normalization constant
            self.land.sigmas[k] = Sigma
            Z_eucl = np.sqrt(((2 * np.pi) ** D) * np.linalg.det(Sigma))
            self.land.Z_eucl[k] = Z_eucl
            old_vs = self.land.V_samples.copy()
            if compute_const:
                Const, int_variance = self.land.quad_obj.estimate_norm_constant(k)
            else:
                Const = self.land.consts[k]
                # TODO: fix this...if we need it at all
                int_variance = float('inf')

            new_likelihood = self.land.compute_negLogLikelihood()
            """
            print("new constant is:")
            print(Const)
            print("new likelihood is %s" % new_likelihood)
            print("scaled: %s" % (new_likelihood / Resp[~inds].sum()))
            """

            V_samples = self.land.V_samples[k,:,:].copy()
            volM_samples = self.land.volM_samples[:,k].copy()      
            term1 = -1/2 * Gamma.T @ Logmaps[~inds, :].T @ np.diag(Resp[~inds].flatten()) @ Logmaps[~inds, :] @ Gamma.T
            term1 /= Resp[~inds].sum()
            #term1 = 1/2 * Logmaps[~inds, :].T @ Logmaps[~inds, :] / Resp[~inds].sum()

            gvvt = np.einsum('nd,ne,n->de', V_samples, V_samples, volM_samples.flatten())
            term2 = 1/2 * Z_eucl / (Const * self.land.model_params['S']) \
            * (-Gamma.T @ gvvt @ Gamma.T)


            gradient = term1 - term2
            #gradient /= Resp[~inds].sum()
            if D <= 5:
                print("gradient:")
                print(gradient)

            # variance for the 2nd term of the gradient
            term2_var = 1/4 * Z_eucl**2 * int_variance / Const**2
            """
            print("gradient var:")
            print(np.sqrt(term2_var))
            """
            gradient_var = term2_var


            # the gradient is already averaged, i.e., we have the mean gradient
            # so we must do the same for the likelihood
            # actually the divison should be by Resp[~inds].sum()
            new_likelihood_mean = new_likelihood / self.land.data.shape[0] #Resp[~inds].sum()

            # here we save the total likelihood
            self.optimization_container[Sigma.tobytes()] = {'likelihood' : new_likelihood, 'Gamma' : Gamma,\
                                                            'Sigma' : Sigma,\
                                                             'V_samples' : V_samples.copy(),\
                                                            'volM_samples' : volM_samples.copy(), 'Z_eucl' : Z_eucl, 'Const' : Const}
            self.grad_container[Sigma.tobytes()] = (gradient.copy(), gradient_var)
            self.cost_container[Sigma.tobytes()] = new_likelihood_mean
            
            return new_likelihood_mean, (gradient, gradient_var)


        # deterministic line search
        
        print("")
        print("")
        print("deterministic line search:")
        print("Sigma is:")
        print(Sigma)
        sigma_manifold = PositiveDefinite(D)
        
        # TODO: for the initial Gamma, we already know cost and gradient
        # do not recompute the current constant
        # the result is cached in the grad_container
        cost_and_grad_Sigma(Sigma, compute_const=False)

        problem = Problem(manifold=sigma_manifold, cost=cost, egrad=(lambda x: grad(x)[0]))
        print("initializing steepest descent...")
        solver = SteepestDescent(maxiter=2,linesearch=None)
        solver._linesearch = self.sigma_linesearch
        if self.sigma_oldalpha[k] is not None and self.sigma_oldalpha[k] > 1e-5:
            print("reusing oldalpha:")
            print(self.sigma_oldalpha[k])
            # this was for the adaptive line search
            solver._linesearch._oldalpha = self.sigma_oldalpha[k]
            #solver._linesearch._oldf0 = self.sigma_oldalpha[k]
        print("")
        print("now solving...")
        res = solver.solve(problem, x=Sigma, reuselinesearch=True)

        # save alpha
        self.sigma_oldalpha[k] = solver.linesearch._oldalpha
        #self.sigma_oldalpha[k] = solver.linesearch._oldf0
        print("saving alpha for next run:")
        print(self.sigma_oldalpha[k])

        # instead of using the last result (maybe maxiter was reached)
        # we want that with the lowest neg log likelihood
        # https://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
        def minkey(d):
            vs=[x['likelihood'] for x in d.values()]
            ks=list(d.keys())
            return ks[vs.index(min(vs))]

        mk = minkey(self.optimization_container)
        maxvals = self.optimization_container[mk]
        self.land.V_samples[k,:,:] = maxvals['V_samples']
        self.land.volM_samples[:,k] = maxvals['volM_samples']
        self.land.Z_eucl[k] = maxvals['Z_eucl']
        self.land.consts[k] = maxvals['Const']
        
        Sigma = maxvals['Sigma']
        print("final Sigma:")
        print(Sigma)
        Gamma = np.linalg.inv(Sigma)
        #print("final Gamma:")
        #print(Gamma)
        self.land.sigmas[k] = Sigma

        print("resulting likelihood:")
        print(maxvals['likelihood'])
             

        return Sigma



        
        
class VanillaDescent(LandOptimizer):
    """ This is the "vanilla" LAND Optimizer
    which essentially performs block-coordinate steepest descent.
    """

    def __init__(self, land):
        self.land = land
        self.gti = None

    def update_mean(self, k, step_selection='vanilla'):
        data = self.land.data
        N, D = data.shape
        Logmaps = self.land.logmaps[k, :, :]  # N x D
        Resp = self.land.resp[:, k].reshape(-1, 1)  # N x 1
        # we need to copy mu because self.land.means changes[k, :] in the loop
        mu = self.land.means[k, :].reshape(-1, 1).copy()  # D x 1
        Sigma = self.land.sigmas[k, :, :]  # D x D
        Z_eucl = self.land.Z_eucl[k]
        manifold = self.land.manifold
        solver = self.land.solver

        S = self.land.model_params['S']

        print("mu is:")
        print(mu)


        for iteration in range(self.land.model_params['max_iter_mu']):
            # these might have changed from the last iteration
            Const = self.land.consts[k].copy()
            V_samples = self.land.V_samples[k, :, :].copy()  # S x D, Use the samples from the last Const estimation
            volM_samples = self.land.volM_samples[:, k].reshape(-1, 1).copy()  # S x 1
        

            print('[Updating mean: {}] [Iteration: {}/{}]'.format(k+1, iteration+1, self.land.model_params['max_iter_mu']))

            # Use the non-failed only
            inds = np.isnan(Logmaps[:, 0])  # The failed logmaps
            grad_term1 = Logmaps[~inds, :].T @ Resp[~inds, :] / Resp[~inds, :].sum()
            grad_term2 = V_samples.T @ volM_samples.reshape(-1, 1) * Z_eucl / (Const * S) #param['S']


            # We do not multiply in front the invSigma because we use the steepest direction
            grad = -(grad_term1 - grad_term2)
            print("gradient is:")
            print(grad)


            # check if gradient norm is below tolerance
            if np.linalg.norm(grad) < self.land.model_params['mu_grad_tol']:
                print("gradient norm below tolerance, break!")
                break

            # unclear if this is theoretically sound
            # since we use riemannian normal coordinates
            if step_selection == 'adagrad':
                if self.gti is None:
                    self.gti = np.zeros(D)
                    ada_init = True
                else:
                    ada_init = False
                epsilon = 1e-6
                grad2 = grad @ grad.T
                grad2d = np.diag(grad2)

                self.gti += (grad2d).reshape(D,)
                adagrad = grad.reshape(D,) / (epsilon + np.sqrt(self.gti)).reshape(D,)
                print("adagrad:")
                print(adagrad)
                adagrad = adagrad.reshape(D,1)
                # for adagrad, we keep the step size fixed
                if ada_init:
                    # for the first iteration, we follow the regular gradient
                    adagrad = grad 
                curve, failed = geodesics.expmap(manifold, mu, -adagrad * 0.2)

            else:
                curve, failed = geodesics.expmap(manifold, mu, -grad * self.land.model_params['step_size'])
            mu_new = curve(1)[0]
            
            print("mu_new:")
            print(mu_new)
            # Compute the logmaps for the new mean
            print("computing new logmaps...")
            t_ = time.time()
            for n in range(N):
                _, logmap, curve_length, failed, sol \
                    = geodesics.compute_geodesic(solver, manifold, mu_new, data[n, :].reshape(-1, 1))
                if failed:
                    Logmaps[n, :] = np.nan  # Note: here we NOT use the straight geodesic
                else:
                    Logmaps[n, :] = logmap.flatten()
            print("duration: %s" % (time.time() - t_))


            # change the mean and logmaps
            self.land.means[k,:] = mu_new.flatten()
            self.land.logmaps[k,:,:] = Logmaps
            
            # Compute the constant for the new mean
            # this will access the land object and change the V_samples etc.
            Const, int_variance = self.land.quad_obj.estimate_norm_constant(k)

            # compute new likelihood
            new_likelihood = self.land.compute_negLogLikelihood()
            print("new likelihood:")
            print(new_likelihood)
            

            cond = np.sum((mu - mu_new) ** 2)  # The condition to stop
            mu = mu_new.copy()
            if cond < self.land.model_params['tol']:   # Stop if the change is lower than the tolerance
                print("break: %s < %s" % (cond, self.land.model_params['tol']))
                break

        print("new mean: %s" % mu.flatten())
        return mu.flatten()



    # Update Sigma using gradient descent
    def update_Sigma(self, k):
        data = self.land.data
        N, D = data.shape
        Logmaps = self.land.logmaps[k, :, :]  # N x D
        mu = self.land.means[k, :].reshape(-1, 1)  # D x 1
        Sigma = self.land.sigmas[k, :, :].copy()  # D x D
        Resp = self.land.resp[:, k].reshape(-1, 1)  # N x 1
        Z_eucl = self.land.Z_eucl[k]
        Const = self.land.consts[k]
        V_samples = self.land.V_samples[k, :, :]  # S x D, Use the samples from the last Const estimation
        volM_samples = self.land.volM_samples[:, k].reshape(-1, 1)  # S x 1
        manifold = self.land.manifold
        solver = self.land.solver
        
        S = self.land.model_params['S']

        # Keep only the non-failed Logmaps
        inds = np.isnan(Logmaps[:, 0])
        grad_term1 = Logmaps[~inds, :].T @ np.diag(Resp[~inds].flatten()) @ Logmaps[~inds, :] / Resp[~inds].sum()  # First term of the gradient
        for iteration in range(self.land.model_params['max_iter_Sigma']):
            print('[Updating Sigma: {}] [Iteration: {}/{}]'.format(k+1, iteration+1, self.land.model_params['max_iter_Sigma']))

            # Get the matrix A
            L, U = np.linalg.eigh(np.linalg.inv(Sigma))
            A = np.diag(np.sqrt(L)) @ U.T

            # Compute the gradient
            gvvt = np.einsum('nd,ne,n->de', V_samples, V_samples, volM_samples.flatten())
            grad_term2 = (gvvt * Z_eucl / (Const * S))
            grad = A @ (grad_term1 - grad_term2)

            print("gradient for A:")
            print(grad)

            # Update the precision on the tangent space and get the covariance
            A_new = A - grad * self.land.model_params['step_size']
            Sigma_new = np.linalg.inv(A_new.T @ A_new)

            print("the new Sigma is:")
            print(Sigma_new)
            # Compute the new normalization constant
            self.land.sigmas[k] = Sigma_new.copy()
            Z_eucl = np.sqrt(((2 * np.pi) ** D) * np.linalg.det(Sigma_new))
            self.land.Z_eucl[k] = Z_eucl
            Const, int_variance = self.land.quad_obj.estimate_norm_constant(k)

            V_samples = self.land.V_samples[k,:,:].copy()
            volM_samples = self.land.volM_samples[:,k].copy()

            new_likelihood = self.land.compute_negLogLikelihood()
            print("new constant is:")
            print(Const)
            print("new likelihood is %s" % new_likelihood)


            # The stopping condition
            # cond = np.sum((A - A_new) ** 2)
            cond = 0.5 * (np.log(np.linalg.det(Sigma) / np.linalg.det(Sigma_new))
                          + np.trace(np.linalg.inv(Sigma) @ Sigma_new) - D)  # KL-divergence for same mean Gaussians

            Sigma = Sigma_new.copy()
            if cond < self.land.model_params['tol']:
                print("break: %s < %s" % (cond, self.land.model_params['tol']))
                break

        return Sigma
    
    def iteration_callback(self, negLogLikelihood, newNegLogLikelihood):
        # Update the step-sizes
        print("callback after iteration...")
        if newNegLogLikelihood < negLogLikelihood:
            self.land.model_params['step_size'] = 1.1 * self.land.model_params['step_size']
            print("increasing step size to %s" % self.land.model_params['step_size'])
        else:
            self.land.model_params['step_size'] = 0.75 * self.land.model_params['step_size']
            print("decreasing step size to %s" % self.land.model_params['step_size'])
        return True


