"""
Adapted from the Pymanopt toolbox

Copyright (c) 2015-2016, Pymanopt Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of pymanopt nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


class LineSearchBackTrackingCustom:
    """
    Back-tracking line-search based on linesearch.m in the manopt MATLAB
    package.
    """

    def __init__(self, contraction_factor=.5, optimism=2,
                 suff_decr=0.05, maxiter=25, initial_stepsize=1):
        self.contraction_factor = contraction_factor
        self.optimism = optimism
        self.suff_decr = suff_decr
        self.maxiter = maxiter
        self.initial_stepsize = initial_stepsize

        self._oldf0 = None

    def search(self, objective, manifold, x, d, f0, df0):
        """
        Function to perform backtracking line-search.
        Arguments:
            - objective
                objective function to optimise
            - manifold
                manifold to optimise over
            - x
                starting point on the manifold
            - d
                tangent vector at x (descent direction)
            - f0
                starting cost at x
            - df0
                directional derivative at x along d
        Returns:
            - stepsize
                norm of the vector retracted to reach newx from x
            - newx
                next iterate suggested by the line-search
        """
        # Compute the norm of the search direction
        norm_d = manifold.norm(x, d)
        # normalize the search direction
        #unit_d = d / norm_d


        if self._oldf0 is not None:
            #print("reusing _oldf0=%s" % self._oldf0)
            # Pick initial step size based on where we were last time.
            alpha = 2 * (f0 - self._oldf0) / df0
            # Look a little further
            alpha *= self.optimism
        else:
            alpha = self.initial_stepsize / norm_d
        alpha = float(alpha)

        print("alpha: %s" % alpha)


        # Make the chosen step and compute the cost there.
        newx = manifold.retr(x, alpha * d)
        newf = objective(newx)
        step_count = 1

        # Backtrack while the Armijo criterion is not satisfied
        #print("our goal is: %s" % (f0 + self.suff_decr * alpha * df0))
        while (newf > f0 + self.suff_decr * alpha * df0 and
               step_count <= self.maxiter):

            # Reduce the step size
            alpha = self.contraction_factor * alpha
            #print("trying new alpha: %s" % alpha)

            # and look closer down the line
            newx = manifold.retr(x, alpha * d)
            newf = objective(newx)
            #print("newf is %s" % newf)

            step_count = step_count + 1

        # If we got here without obtaining a decrease, we reject the step.
        if newf > f0:
            print("oops, no improvement in linesearch.")
            alpha = 0
            newx = x

        stepsize = alpha * norm_d
        print("set _oldf0 to %s" % f0)
        self._oldf0 = f0

        return stepsize, newx




class LineSearchCustom:
    '''
    Customized Adaptive line-search
    '''

    def __init__(self, contraction_factor=.5, suff_decr=0.1, maxiter=10,
                 initial_stepsize=1):
        self._contraction_factor = contraction_factor
        self._suff_decr = suff_decr
        self._maxiter = maxiter
        self._initial_stepsize = initial_stepsize
        self._oldalpha = None

    def search(self, objective, man, x, d, f0, df0):
        norm_d = man.norm(x, d)

        if self._oldalpha is not None and self._oldalpha > 0:
            #print("we have oldalpha: %s" % self._oldalpha)
            alpha = self._oldalpha
        else:
            #print("oldalpha is none...")
            alpha = self._initial_stepsize / norm_d
        alpha = float(alpha)
        print("initialising alpha:")
        print(alpha)

        newx = man.retr(x, alpha * d)
        newf = objective(newx)
        cost_evaluations = 1

        while (newf > f0 + self._suff_decr * alpha * df0 and
               cost_evaluations <= self._maxiter):
            print("reducing alpha...")
            # Reduce the step size.
            alpha *= self._contraction_factor
            
            print("alpha:")
            print(alpha)

            # Look closer down the line.
            newx = man.retr(x, alpha * d)
            newf = objective(newx)

            cost_evaluations += 1

        print("linesearch loop ended")

        if newf > f0:
            print("oops, no improvement!")
            alpha = 0
            newx = x

        stepsize = alpha * norm_d

        # Store a suggestion for what the next initial step size trial should
        # be. On average we intend to do only one extra cost evaluation. Notice
        # how the suggestion is not about stepsize but about alpha. This is the
        # reason why this line search is not invariant under rescaling of the
        # search direction d.

        # If things go reasonably well, try to keep pace.
        if cost_evaluations == 2:
            self._oldalpha = alpha
        # If things went very well or we backtracked a lot (meaning the step
        # size is probably quite small), speed up.
        else:
            self._oldalpha = 1.3 * alpha # was 2 originally
        print("set _oldalpha to %s" % self._oldalpha)

        return stepsize, newx