import numpy as np
from core import manifolds
from core import geodesics
from core import utils
from core.land_quadrature import LandQuadrature, MCQuadrature, BQuadrature
from core.land_optimization import VanillaDescent, AdaptiveRiemannianDescent
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.mixture import GaussianMixture
import time
import sklearn.neighbors as skn
#import networkx as nx
import operator
from scipy.optimize import minimize
from sklearn.cluster import SpectralClustering
import dill
import os
import copy

class LandMixture():
    """
    Represents a LAND mixture model, containing all the relevant data.   
        
    """
    
    def __init__(self, manifold, solver, data):
        """ Initialize the LAND object.
        
        Parameters
        ----------
        manifold
            The manifold that the LAND is fit on.
        solver 
            A solver object for geodesics computation.
        data : ndarray[N,D]
            The data used to construct the manifold and to fit the LAND.
        """
        self.manifold = manifold
        self.solver = solver
        self.data = data
        
    def setup(self, model_params, integration_params, initial_means=None):
        """ Before the LAND can be fit, model and integration parameters
        need to be set up and initial means may be specified.
        If no initial means are provided, a euclidean Gaussian Mixture model used
        to initialize them.
        
        Parameters
        ----------
        
        model_params:
            a dictionary specifying model parameters, e.g. 'maxiter' and 'tol'
        integration_params:
            parameters specific to the used integration method.
        initial_means: ndarray[K,D]
            initial means for the K components
        
        """
        self.model_params = model_params
        self.integration_params = integration_params
        self.times = {'start' : time.time()}
        
        K = model_params['K']
        N, D = self.data.shape
        self.K = K
        self.N = N
        self.D = D
        
        # initialize the parameters for the LAND
        self.resp = np.zeros((N, K))  # 1 failed, 0 not failed
        self.logmaps = np.zeros((K, N, D))
        self.logmaps[:] = np.nan
        self.consts = np.zeros((K, 1))
        self.sigmas = np.zeros((K, D, D))
        self.manifold_sigmas = np.zeros((K, D, D))
        self.weights = np.zeros((K, 1))  # The mixing components
        self.failed = np.zeros((N, K))  # 1 failed, 0 not failed
        self.V_samples = np.zeros((K, self.model_params['S'], D))
        self.volM_samples = np.zeros((self.model_params['S'], K))
        self.Z_eucl = np.zeros((K, 1))


        # initialize the means
        if initial_means is None:
            self.means = GaussianMixture(n_components=K, covariance_type='full', n_init=10).fit(self.data).means_
        else:
            self.means = np.atleast_2d(initial_means)
        if model_params.get('verbose'):
            print('initial means: %s' % str(self.means))
            
        # evaluate the measure at the data points - we need this if logmaps are evaluated at data points
        # TODO: it's enough to evaluate at the inducing points if the measure is expensive
        self.data_measures = np.apply_along_axis(lambda v: self.manifold.measure(v.reshape(D,1)), 1, self.data).reshape(-1,1)
        # for each component K, this holds information for reusing information during integration
        self.transfer_dict = {}
        for k in range(K):
            self.transfer_dict[k] = {}
        if 'method' not in integration_params.keys():
            print("specify an integration method! choices: ['MC','BQ']")
            sys.exit()
        if integration_params['method'] == 'BQ':
            self.quad_obj = BQuadrature(self)
        elif integration_params['method'] == 'MC':
            self.quad_obj = MCQuadrature(self)
        else:
            print("unsupported integration method: %s" % integration_params['method'])
        
        # choose some inducing points for BQ
        # to reduce the size of the dataset
        n_inducing = integration_params.get('n_inducing') if integration_params.get('n_inducing') is not None else min(30,self.data.shape[0])
        clusters = KMeans(n_clusters=n_inducing).fit(self.data)
        centers = clusters.cluster_centers_
        inducing_points = np.zeros(self.data.shape[0])
        for c in centers:
            mind = float('+inf')
            minp = None
            for i,d in enumerate(self.data):
                dist = np.linalg.norm(c-d)
                if dist < mind:
                    mind = dist
                    minp = i
            inducing_points[minp] = 1

        # array of 0s and 1s (if point belongs to the inducing points)
        self.inducing_points = inducing_points
        if model_params.get('verbose'):
            print("number of inducing points: %s" % (inducing_points.sum()))
            
        print("finished setting up the parameters, initializing components now.")
        
        # Initialize components
        solutions = {}
        for k in range(K):
            for n in range(N):
                if model_params.get('verbose'):
                    print('[Initialize: {}/{}] [Process point: {}/{}]'.format(k+1, K, n+1, N))
                key = 'k_' + str(k) + '_n_' + str(n)
                curve, logmap, curve_length, failed, sol \
                    = geodesics.compute_geodesic(self.solver, self.manifold,
                                                 self.means[k, :].reshape(-1, 1), self.data[n, :].reshape(-1, 1))
                #geodesics.plot_curve(curve, linewidth=1, c='r')
                if failed:
                    self.failed[n, k] = True
                    self.logmaps[k, n, :] = logmap.flatten()  # Note: The straight line geodesic
                    self.resp[n, k] = 1/curve_length  # If points are far lower responsibility
                    solutions[key] = None
                else:
                    self.failed[n, k] = False
                    self.logmaps[k, n, :] = logmap.flatten()
                    self.resp[n, k] = 1/curve_length
                    solutions[key] = curve(np.linspace(0,1,200)) # sol

            #plt.show()
            #plt.close()

        self.resp = self.resp / self.resp.sum(axis=1, keepdims=True)   # Compute the responsibilities
        self.weights = np.sum(self.resp, axis=0).reshape(-1, 1) / N

        #print("the max resps are:")
        #print(self.resp.argmax(axis=1))

        # create the directory for saving the LAND objects
        if integration_params["save_lands"]:
            land_savedir = integration_params["savedir"] + "lands/" 
            try:
                os.mkdir(land_savedir)
            except FileExistsError:
                print("Directory " , land_savedir ,  " already exists")

        # Use the closest points to estimate the Sigmas and the normalization constants
        for k in range(K):
            print(k)
            inds_k = (self.resp.argmax(axis=1) == k)  # The closest points to the k-th center
            self.sigmas[k, :, :] = np.cov(self.logmaps[k, inds_k, :].T)
            self.quad_obj.estimate_norm_constant(k, consider_failed=True)


        print("initialization complete.")
        print("LAND means:")
        print(self.means)
        print("weights:")
        print(self.weights)
        print("")
        print("consts:")
        print(self.consts)

        if integration_params["save_lands"]:
            try:
                plt.figure()
                plt.scatter(self.data[:,0], self.data[:,1], c=self.resp.argmax(axis=1))
                plt.scatter(self.means[:,0],self.means[:,1],marker="D",c='r',s=50)
                plt.savefig(integration_params["savedir"] + "initialization.png", dpi=300)
                plt.close()
            except Exception as e:
                print(e)

        if integration_params["save_lands"]:
            try:
                dill.dump({'mean' : self.means, 'sigmas' : self.sigmas, 'weights' : self.weights, 'consts' : self.consts},\
                    open(integration_params["savedir"] + "initialization.pkl","wb"))
            except Exception as e:
                print(e)

        # dump the LAND object after initialization is finished
        if integration_params["save_lands"]:
            try:
                dill.dump(solutions, open(integration_params["savedir"] + "solutions.pkl","wb"))
                landcopy = self.getDumpableLand()
                dill.dump(landcopy, open(integration_params["savedir"] + "land-initialization.pkl","wb"))
            except Exception as e:
                print(e)

        self.negLogLikelihoods = []
        self.times['initialization'] = time.time() - self.times['start']


        

        
    def fit(self):
        """ Fit the LAND mixture model with the previously specified parameters.
        the setup method has to be called first.
        
        Returns
        ------
        negLogLikelihood
            the final negative log likelihood
        """
        K = self.K
        N = self.data.shape[0]
        negLogLikelihood = self.compute_negLogLikelihood()
        self.negLogLikelihoods = [negLogLikelihood]
        
        #optimizer = VanillaDescent(self)
        optimizer = AdaptiveRiemannianDescent(self)
        
        for iteration in range(self.model_params['max_iter']):
            print("")
            print("-------------------------------------------------------------------")
            print('[Iteration: {}/{}] [Negative log-likelihood: {}]'.format(iteration+1, self.model_params['max_iter'], negLogLikelihood))
        
            # ----- E-step ----- #
            for k in range(K):
                self.resp[:, k] = self.weights[k] \
                                     * self.evaluate_pdf(k).flatten()
            self.resp = self.resp / self.resp.sum(axis=1, keepdims=True)


            # ----- M-step ----- #
            # Update the means
            for k in range(K):
                optimizer.update_mean(k)
            print("")

            # Update the covariances
            for k in range(K):
                optimizer.update_Sigma(k)
            print("")
                
            # Update the mixing components
            self.weights = np.sum(self.resp, axis=0).reshape(-1, 1) / N
            print("weights are now:")
            print(self.weights)

            # Compute the new likelihood and store it
            newNegLogLikelihood = self.compute_negLogLikelihood()
            self.negLogLikelihoods.append(newNegLogLikelihood)
            
            optimizer.iteration_callback(negLogLikelihood, newNegLogLikelihood)

            # Check the difference in log-likelihood between updates
            ldiff = np.abs(newNegLogLikelihood - negLogLikelihood) 
            if ldiff < self.model_params['likelihood_tol']:
                print("absolute difference in neg log likelihoods was %s " % ldiff)
                print("finished!")
                break
            else:
                negLogLikelihood = newNegLogLikelihood

            self.times['iteration_' + str(iteration)] = time.time() - self.times['start']

        if self.integration_params["save_lands"]:
            landcopy = self.getDumpableLand()
            dill.dump(landcopy, open(self.integration_params["savedir"] + "final-land.pkl","wb"))
        return negLogLikelihood

                
                
    def evaluate_pdf(self, k, replace_args=None):
        """ Evaluate the probability density function of the LAND
        
        Parameters
        ----------
        k : int
            component for which the density should be evaluated.
        replace_args: (ndarray[N,D], ndarray[D,D], float)
            an optional 3-tuple containing the logmaps, Sigma and the constant
            for which the density should be evaluated. 
                Default: use the current LAND parameters.
        """
        if replace_args is None:
            Logmaps = self.logmaps[k]
            Sigma = self.sigmas[k]
            Const = self.consts[k]
        else:
            (Logmaps, Sigma, Const) = replace_args
        N, D = Logmaps.shape
        inds = np.isnan(Logmaps[:, 0])  # For the logmaps which are NaN return a zero pdf
        #print("failed logmaps: %s" % inds.sum())
        result = np.zeros((N, ))
        result[~inds] = (np.exp(-0.5 * pairwise_distances(Logmaps[~inds, :], np.zeros((1, D)),
                                                metric='mahalanobis',
                                                VI=np.linalg.inv(Sigma)) ** 2) / Const).flatten()

        result[inds] = 1e-9  # For the failed ones put a very small pdf

        return result.reshape(-1, 1)  # N x 1
    
  
    def compute_negLogLikelihood(self, return_individuals=False):
        """ Given the current logmaps, compute the
        negative log likelihood (for all components in the mixture).
        """
        K = self.means.shape[0]
        result = np.zeros((self.logmaps.shape[1], 1))
        for k in range(K):
            result += self.weights[k] * self.evaluate_pdf(k)

        if return_individuals:
            return -np.log(result)
        else:
            return -np.sum(np.log(result))


    def predict(self, test_data):
        """ Using a trained land model, predict the cluster for the given data

        Parameters
        ----------
        test_data:
            the data for which the prediction is desired.

        """
        N, D = test_data.shape
        K = self.K
        responsibilities = np.zeros((N, K))
        Logmaps = np.zeros((K, N, D))

        for k in range(K):
            mu = self.means[k, :].reshape(-1, 1)
            for n in range(N):
                print('[Center: {}/{}] [Point: {}/{}]'.format(k+1, K, n+1, N))
                _, logmap, _, failed, _ \
                    = geodesics.compute_geodesic(self.solver, self.manifold, mu, test_data[n, :].reshape(-1, 1))
                Logmaps[k, n, :] = logmap.flatten()  # Use the geodesic even if it is the failed line

        for k in range(K):
            responsibilities[:, k] = self.weights[k] \
                                 * self.evaluate_pdf(Logmaps[k, :, :],
                                                           self.sigmas[k, :, :],
                                                           self.consts[k]).flatten()

        responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)
        return responsibilities


    def getDumpableLand(self):
        """ Get a copy of the LAND mixture object which can be dumped using dill
        """
        landcopy = copy.copy(self)
        landcopy.quad_obj = None
        landcopy.transfer_dict = None
        return landcopy

    
    def saveResult(self, file):
        """ Save the resulting LAND fit parameters to a dill file (binary).

        Parameters
        ----------
        file: String
            the directory+file name 
        """
        res = {'means' : self.means, 'sigmas' : self.sigmas, 'logmaps' : self.logmaps, 'consts' : self.consts,\
               'weights' : self.weights, 'resp' : self.resp, 'negLogLikelihoods' : self.negLogLikelihoods,\
               'times' : self.times, 'manifold_sigmas' : self.manifold_sigmas, 'Z_eucl' : self.Z_eucl}
        dill.dump(res, open(file, "wb"))
        return True


    def loadResult(self, file):
        """ Save the resulting LAND fit parameters from a dill file (binary).

        Parameters
        ----------
        file: String
            the directory+file name 
        """
        res = dill.load(open(file, "rb"))
        self.means = res["means"]
        self.sigmas = res['sigmas']
        self.logmaps = res["logmaps"]
        self.consts = res["consts"]
        self.weights = res["weights"]
        self.resp = res["resp"]
        self.negLogLikelihoods = res["negLogLikelihoods"]
        try:
            self.Z_eucl = res["Z_eucl"]
            self.manifold_sigmas = res["manifold_sigmas"]
        except Exception as e:
            print(e)
        self.times = res['times'] if 'times' in res.keys() else {}
        return self
    
    def initialize_means_gmm(self,k=1):
        gm = GaussianMixture(n_components=k).fit(self.data)
        means = gm.means_
        return means
      
        
    def initialize_means_knn(self, k=1):
        """ Initialize the means for the LAND mixture
        using kNN+spectral clustering

        Parameters
        ----------
        k: Int
            number of desired components
        """

        print("initializing LAND means using spectral clustering on kNN-graph, edges reweighted by geodesic distances.")
        data = self.data
        dim = data.shape[1]

        # construct solver graph based on the data
        solver_graph = geodesics.SolverGraph(manifold=self.manifold, data=self.data+1e-4, kNN_num=7, tol=1e-3)
        dists = solver_graph.dist_matrix


        print("is the kNN-graph disconnected?")
        print(np.isinf(dists).any())

        maxel = np.max(np.where(np.isinf(dists),-np.inf,dists))
        # dirty trick:
        # if the distance is infinity (which shouldn't usually happen...)
        # set the distance to twice the maximum distance
        dists_ = dists.copy() # save a copy for later
        dists[dists == np.inf] = 2*maxel
        rho = 1.
        kxx = np.exp(- dists ** 2 / (2. * rho ** 2))

        # spectral clustering
        sc = SpectralClustering(k, affinity='precomputed')
        labels = sc.fit_predict(kxx)
        print("spectral clustering finished.")

        mus = np.zeros((k,dim))

        for _k in range(k):
            nodes_k = data[labels==_k]
            print("component %s is responsible for %s data points." % (_k, nodes_k.shape[0]))

            dmin = float('+inf')
            dmin_n = None
            for n in nodes_k:
                i_n = None
                for i_d, d in enumerate(data):
                    if np.allclose(d, n):
                        i_n = i_d
                same_cluster_dists = dists_[i_n][labels==_k]
                total_dist = same_cluster_dists[same_cluster_dists != np.inf].sum()
                if total_dist < dmin:
                    dmin = total_dist 
                    dmin_n = n  
         

            mus[_k] = dmin_n+1e-3   


        return mus