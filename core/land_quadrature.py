import numpy as np
from core import geodesics
from core import utils
from bayesquad.bq_wrapper import BQWrapper
from bayesquad.ratio_bq_wrapper import RatioBQWrapper
import time
import copy
import dill

class LandQuadrature():
    
    def __init__(self, land):
        self.land = land
        self.integration_params = land.integration_params
        
    def estimate_norm_constant(self, K):
        pass
    

class MCQuadrature(LandQuadrature):
    # This function estimates the normalization constant using Monte Carlo sampling on the tangent space
    def estimate_norm_constant(self, k, consider_failed=False):
        if self.integration_params['method'] != 'MC':
            print("oops, this shouldn't happen. Integration parameters are probably misspecified.")
        mu = self.land.means[k]
        Sigma = self.land.sigmas[k]
        S = self.land.model_params["S"]
        start = time.time()

        D = Sigma.shape[0]
        Z_eucl = np.sqrt(((2 * np.pi) ** D) * np.linalg.det(Sigma))  # The Euclidean normalization constant

        # Initialize the matrices for the samples and the matrix A
        L, U = np.linalg.eigh(Sigma)
        A = U @ np.diag(np.sqrt(L))
        V_samples = np.zeros((S, D))
        V_samples[:] = np.nan
        X_samples = np.zeros((S, D))
        X_samples[:] = np.nan

        s = 0
        while True:
            try:
                v = A @ np.random.randn(D, 1)  # D x 1, get white noise to sample from N(0, Sigma)
                curve, failed = geodesics.expmap(self.land.manifold, x=mu.reshape(-1, 1), v=v.reshape(-1, 1))
                if not failed:
                    X_samples[s, :] = curve(1)[0].flatten()
                    V_samples[s, :] = v.flatten()
                    s = s + 1
                else:
                    print('Expmap failed for v %s' % v)
            except Exception:
                print('Expmap failed for v %s' % v)

            if s == S:  # We have collected all the samples we need
                break

        inds = np.isnan(X_samples[:, 0])  # The failed exponential maps
        X_samples = X_samples[~inds, :]  # Keep the non-failed ones
        V_samples = V_samples[~inds, :]

        volM_samples = self.land.manifold.measure(X_samples.T).flatten()  # Compute the volume element sqrt(det(M))
        norm_constant = np.mean(volM_samples) * Z_eucl  # Estimate the normalization constant

        end = time.time()
        time_mc = end - start
        print("Const: %s" % norm_constant)
        print("Runtime for MC: %s" % time_mc)


        # save some debug information
        if self.integration_params["logger"]:
            savedir = self.integration_params["savedir"]

            _means = self.land.means.copy()
            _sigmas = self.land.sigmas.copy()
            _consts = self.land.consts
            _means[k] = mu
            _sigmas[k] = Sigma
            _consts[k] = norm_constant

            res_dict = {'mu' : mu, 'Sigma' : Sigma, 'Z' : Z_eucl, 'X' : None, 'Y' : None, 'm_int' : norm_constant / Z_eucl, \
                        'x_preds' : None, 'y_preds' : None, 'pred_means' : None, 'pred_variances' : None,\
                        'lengthscale' : None, 'variance' : None,\
                        'V_samples' : V_samples, 'volM_samples' : volM_samples, 'X_samples' : X_samples,\
                        'runtime' : time_mc, 'logmaps' : self.land.logmaps[k], 'k' : k,\
                        'all_mus' : _means,\
                        'all_sigmas' : _sigmas,\
                        'all_logmaps' : self.land.logmaps,\
                        'all_consts' : _consts,\
                        'weights' : self.land.weights}
            utils.logger(res_dict, savedir)

        self.land.consts[k] = norm_constant
        self.land.V_samples[k] = V_samples
        self.land.volM_samples[:,k] = volM_samples.flatten()
        self.land.Z_eucl[k] = Z_eucl

        # we can compute the "true" manifold sigma = int v v^t exp(-0.5 <v, Gamma v>) dv
        try:
            self.land.manifold_sigmas[k] = 1/norm_constant * 1/V_samples.shape[0] * V_samples.T @ np.diag(volM_samples.flatten()) @ V_samples
            #print("true Sigma is:")
            #print(self.land.manifold_sigmas[k])
        except:
            pass

        if self.integration_params.get("save_lands"):
            # save the land object
            landcopy = self.land.getDumpableLand()
            utils.logger(landcopy, self.integration_params["savedir"] + "lands/", startswith="land_")


        # TODO: variance for MC estimator
        return norm_constant, 0 
        
        
        
        
class BQuadrature(LandQuadrature):
    # estimating the normalization constant using BQ
    def estimate_norm_constant(self, k, consider_failed=False):
        if self.integration_params['method'] != 'BQ':
            print("oops, this shouldn't happen. Integration parameters are probably misspecified.")

        mu = self.land.means[k]
        Sigma = self.land.sigmas[k]
        D = Sigma.shape[0]
        S = self.land.model_params['S']

        Z = np.sqrt(((2 * np.pi) ** D) * np.linalg.det(Sigma))  # The Euclidean normalization constant

        # these containers are needed for the gradients
        # but this requires that the BQ method uses MC inside to predict
        V_samples = np.zeros((S+1, D))
        V_samples[:] = np.nan

        volM_samples = np.zeros(S+1)
        volM_samples[:] = np.nan

        # the integrand
        # works directly on a tangent vector    
        # needs to be integrated against a gaussian with covariance matrix Sigma
        def fTangent(v): 
            curve, failed = geodesics.expmap(self.land.manifold, x=mu, v=v)  # Solve the IVP problem for expmap
            if failed:
                print("expmap failed for v: %s" % v)
                print("the failed result is %s" % curve(1)[0].reshape(-1,1))
            x = curve(1)[0].reshape(-1, 1)  # D x 1, the point on the manifold
            meas = self.land.manifold.measure(x) # The Riemannian measure |M(x)|^1/2
            if np.isnan(np.array(meas)):
                print("result is NaN")
                print("v:")
                print(v)
            return meas

        # for multiple tangent vectors at once
        def fTangent_multiple(vs): 
            return np.apply_along_axis(fTangent, 1, vs).reshape(-1,1,)


        def ExpmapCurve(v):
            curve, failed = geodesics.expmap(self.land.manifold, x=mu, v=v)  # Solve the IVP problem for expmap
            if failed:
                print("expmap failed for v: %s" % v)
                print("the failed result is %s" % curve(1)[0].reshape(-1,1))
            return curve

        start = time.time()
        fun = fTangent_multiple

        # reuse information from the last iteration
        last_mu = self.land.transfer_dict[k].get("last_mu")
        last_X = self.land.transfer_dict[k].get("last_X")
        last_Y = self.land.transfer_dict[k].get("last_Y")
        last_kernel = self.land.transfer_dict[k].get("last_kernel")

        reusing = False # indicates whether mu changed or not, we can reuse old observations if it didnt

        if self.integration_params.get("ever_reuse"):
            if last_mu is not None:
                print("can BQ reuse information?")
                print("last mu: %s " % last_mu)
                print("new mu: %s" % mu)
                if np.linalg.norm(last_mu - mu) < 1e-6:
                    print("reusing last " + str(last_X.shape[0]) + " observations.")
                    reusing = True
                    X = last_X
                    Y = last_Y
                else:
                    print("not reusing")
            else:
                print("last mu is None!")

        if True:#try:
            if not reusing:
                # if we don't reuse, then mu changed
                # but in this case, we can use the logmaps!
                # to do so, we have to evaluate the measure at the data points
                # but only use non-failed logmaps

                # if consider_failed=True, we have to check land.failed to see which
                # logmaps succeeded. this is only for the initialization of the LAND
                # afterwards, failed logmaps are NaN
                if self.integration_params.get('use_logmaps') is None or self.integration_params.get('use_logmaps'):
                    logmaps = []
                    measures = []
                    if consider_failed:
                        #print("considering failed logmaps..")
                        for il, l in enumerate(self.land.logmaps[k]):
                            if self.land.inducing_points[il] == 1 and not self.land.failed[il,k]:
                                logmaps.append(l)
                                measures.append(self.land.data_measures[il])

                    else:
                        for il, l in enumerate(self.land.logmaps[k]):
                            if self.land.inducing_points[il] == 1 and not np.isnan(l).any():
                                logmaps.append(l)
                                measures.append(self.land.data_measures[il])

                    measures = np.array([measures]).reshape(-1,1)
                    logmaps = np.vstack(logmaps).reshape(-1,D)
                    X = logmaps
                    Y = measures
                else:
                    X = np.array([])
                    Y = np.array([])

                if self.integration_params.get('verbose'):
                    print("y init: %s" % str(Y.shape))
                #last_kernel = None

            if True:
                print("last_mu is none!")
                logger = utils.logger if (self.integration_params.get('logger') is not None \
                    and self.integration_params.get('logger') is not False) else None
                w = BQWrapper(D, mu=mu, Z=Z, k=last_kernel, n_grad=S, plot=False, logger=logger, land=self.land)
                n_samples = self.integration_params["reusing_samples"] if reusing else self.integration_params["new_samples"]
                n_batches = self.integration_params["reusing_batches"] if reusing else self.integration_params["new_batches"]
                gp_mean = self.land.manifold.asymptotic_measure() if self.integration_params.get("asymptotic_mean") else 0.
                m_int, V_samples, volM_samples, int_variance = w.wsabi_integrate(fun, X, Y, n_batches, n_samples, \
                                                                variance=200., lengthscale=2.,\
                                                                prior_mean=np.zeros(D), prior_cov=Sigma, \
                                                                constant_mean = gp_mean,\
                                                                grad=True, \
                                                                integration_params=self.integration_params,\
                                                                component_k=k,
                                                                expmap=ExpmapCurve, f_manifold=self.land.manifold.measure)

            # update the last mu container
            last_mu = mu
            print("collected observations: %s" % (w.X).shape[0])
            last_X = w.X
            last_Y = w.Y

        else:#except Exception as e: 
            print("error in BQ")
            print(e)
            m_int = 0.0
            w = None

        # multiply with euclidean normalization constant
        Const = Z * m_int

        end = time.time()
        time_bq = end - start
        if self.integration_params.get("verbose"):
            print("Const: %s" % Const)
        print('Runtime for BQ : ' + str(time_bq))

        self.land.transfer_dict[k] = {'last_mu' : last_mu.copy(), 'last_X' : last_X.copy(), 'last_Y' : last_Y.copy(),\
                                      'last_kernel' : (w.k if w is not None else None)}

        # we do not have any X samples to return, thus we return None
        # but they are not used anyway
        self.land.V_samples[k,:,:] = V_samples
        self.land.volM_samples[:,k] = volM_samples.flatten()
        self.land.consts[k,:] = Const
        self.land.Z_eucl[k] = Z

        # we can compute the "true" manifold sigma = int g(v) v v^t exp(-0.5 <v, Gamma v>) dv
        #self.land.manifold_sigmas[k] = 1/Const * 1/V_samples.shape[0] * V_samples.T @ np.diag(volM_samples.flatten()) @ V_samples
        self.land.manifold_sigmas[k] = 1/Const * 1/V_samples.shape[0] * np.einsum('nd,ne,n->de', V_samples, V_samples, volM_samples.flatten())


        # save the land object
        if self.integration_params.get("save_lands"):
            landcopy = self.land.getDumpableLand()
            utils.logger(landcopy, self.integration_params["savedir"] + "lands/", startswith="land_")

        # return both the constant (mean) and its variance estimator
        return Const, int_variance