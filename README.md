# Bayesian Quadrature on Riemannian Data Manifolds

This code includes the core of the LAND mixture model (geodesic methods + optimization + quadrature) and 
a Bayesian quadrature (BQ) implementation based on the ''bayesquad'' library (see bayesquad/LICENSE).
A small demo is included in the file demo.py, which shows how to fit a LAND mixture using MC or BQ. 

Documentation will still be added.

Requirements:
* python==3.7.7
* gpy==1.9.9
* pymanopt==0.2.5
* scikit-learn==0.23.1
* scipy==1.5.2
* dill==0.3.3
* multimethod==1.4
