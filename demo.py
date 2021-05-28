from core import geodesics
from core import manifolds 
from core.land_mixture import LandMixture
import numpy as np
import time
import dill


np.random.seed(42)

# a toy circle with 1000 data points
#data = np.load('data/circle.npy') # takes much longer

data = np.load('data/example1.npy') # for a quick demo

# construct the unsupervised kernel metric
manifold = manifolds.LocalDiagPCA(data, sigma=0.15, rho=1e-3)


solver_fp = geodesics.SolverFP(D=data.shape[1], N=10, tol=1e-1)  # Approximate fixed-point solver
# for speed, we here only use the FP solver.
#solver_bvp = geodesics.SolverBVP(tol=1e-1, NMax=100, T=20)  # Python bvp5c solver
#solver_fpbvp = geodesics.SolverFPBVP(solver_fp, solver_bvp)
#solver_wrapper = geodesics.SolverReusingWrapper(solver_fpbvp, solver_bvp)
Logmaps_data = np.zeros_like(data)  # N x D

# fit LAND with 1 component, S=30,000 MC emulator samples
# maxiter=number of 'superiteration' (outer loop of LAND mixture algorithm), fixed to 2 (so this demo does not take too long)
# in each 'superiteration' (outer loop of LAND mixture algorithm), update each mean once
# then update each covariance 4x
# initial step size for updating the mean is 0.1
# stop if likelihood change between 'superiterations' is below 2
# only take a gradient step for the mean if gradient l2-norm is > 0.01
# 'tol' parameter is not needed here, only if max_iter_mu > 1
model_params = {'K' : 1, 'S' : 30000,  \
                'max_iter' : 4, 'max_iter_mu' : 1, 'max_iter_Sigma' : 4,\
                'step_size' : 0.1, 'tol' : 1e-3, 'likelihood_tol' : 2., 'mu_grad_tol' : 0.01, 'verbose' : False}

# MC
mc_integration_params = {'method' : 'MC', 'logger' : False, 'savedir' : '', 'save_lands' : False}

# BQ
# the 'savedir' can be set to a directory path to generate extensive log files (for each integration)
# these log files include the integration problems (mu, Sigma, ...) and GP information (e.g., predictions on grid)
# check bayesquad/bq_wrapper.py for more details
# we use 80/10 samples (depending on whether the mean changed or not)
# with the matern52 kernel
bq_integration_params = {'method' : 'BQ', 'bq_method' : 'WSABI-L', \
                      'logger' : False, 'savedir' : '',\
                      'new_samples' : 1, 'new_batches' : 80, \
                      'reusing_samples' : 1, 'reusing_batches' : 10,\
                      'kern' : 'matern52',\
                      'asymptotic_mean' : True,\
                      'use_logmaps' : False,\
                      'ever_reuse' : True, \
                      'verbose' : False,\
                      'save_lands' : False,\
                      'prior_mean_scaling' : 1.0}

durations = {}
for method, integration_params in [('BQ', bq_integration_params), ('MC', mc_integration_params)]:
    print("integration method: %s" % method)
    if method == 'MC':
        # we use 1000 monte carlo samples
        model_params['S'] = 1000
    elif method == 'BQ':
        # for BQ, the S parameter means simply
        # how many MC samples are used on top of the GP emulator
        # the time investment for this is negligible,
        # since these do not entail the computation of expmaps.
        model_params['S'] = 30000

    start = time.time()
    land = LandMixture(manifold, solver_fp, data)
    init_mean = land.initialize_means_knn(model_params['K'])

    print("initial LAND means:")
    print(init_mean)

    land.setup(model_params, integration_params, init_mean)
    land.fit()
    duration = time.time() - start
    durations[method] = duration
    print("finished")
    print("duration: %s seconds" % duration)
    print("")
    print("")

    land.saveResult("%s_land.pkl" % method)

    # the final LAND object contains lots of debug information
    #land = dill.load(open("%s_land.pkl" % method,"rb"))
    #print(land)

print("")
print("-----------------------------")
print("LAND fit duration using MC: %.2f min" % (durations["MC"]/60.))
print("LAND fit duration using BQ: %.2f min" % (durations["BQ"]/60.))
print("note that the difference is only in the integration subroutine.")