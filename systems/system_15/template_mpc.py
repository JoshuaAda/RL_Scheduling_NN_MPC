#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version system_3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

#import numpy as np
from casadi import *
import sys
sys.path.append('../../')
import do_mpc
import control
import cvxpy as cp
import matplotlib.pyplot as plt

def template_mpc(model):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)



    setup_mpc = {
        'n_robust': 0,
        'n_horizon': 10,
        't_step': 0.5,
        'store_full_solution':True,

    }

    A = np.array([[ 0.763,  0.460,  0.115,  0.020],
                  [-0.899,  0.763,  0.420,  0.115],
                  [ 0.115,  0.020,  0.763,  0.460],
                  [ 0.420,  0.115, -0.899,  0.763]])

    B = np.array([[0.014],
                  [0.063],
                  [0.221],
                  [0.367]])
    Q=np.array([[1,  0,   0,  0],
                  [0,  0,   0,  0],
                  [ 0,  0,  0,  0],
                  [ 0,  0,  0,  0]])

    # get a stabilizing controller gain matrix with an simple LQR approach
    R=np.array([0])
    P,S,K=control.dare(A,B,Q,R)

    mpc.set_param(**setup_mpc)

    # setup of parameters of the system which can be changed during the process
    tvp_template = mpc.get_tvp_template()

    def tvp_fun(t_curr):

        for k in range(mpc.n_horizon + 1):
            tvp_template['_tvp', k, 'x_tilde'] = np.array([[0], [0], [0], [0]])
            tvp_template['_tvp', k, 'A'] = A
            tvp_template['_tvp', k, 'B'] = B
            tvp_template['_tvp', k, 'Q'] = Q
            # tvp_template['_tvp', k, 'R'] = R
            tvp_template['_tvp', k, 'K'] = K.T
            tvp_template['_tvp', k, 'P'] = P

        return tvp_template
    template=tvp_fun(0)
    mpc.set_tvp_fun(tvp_fun)

    # set costs
    mterm = model.aux['ter_cost']
    lterm = model.aux['cost'] # terminal cost

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=0.1)

    # set constraints
    max_x = np.array([[10], [10], [10], [10]])#np.array([[10], [10]])

    mpc.bounds['lower','_x','x'] = -max_x
    mpc.bounds['upper','_x','x'] =  max_x

    mpc.bounds['lower','_u','u'] = -5
    mpc.bounds['upper','_u','u'] =  5
    suppress_ipopt  = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
    mpc.set_param(nlpsol_opts=suppress_ipopt)
    # set terminal set
    F=np.array([[0.1, 0,0,0],[-0.1, 0,0,0],[0, 0.1,0,0],[0, -0.1,0,0],[0, 0,0.1,0],[0, 0,-0.1,0],[0, 0,0,0.1],[0, 0,0,-0.1]])
    G=np.array([[0.2],[-0.2]])

    nc=len(F)
    G=np.repeat(G,4,axis=0)
    E=np.zeros((1,10))
    E[0,0]=1
    H=cp.Variable((nc,nc),symmetric=True)
    F_tilde=np.concatenate((F+G@K,G@E),axis=1)

    M=np.concatenate((np.concatenate((np.zeros((9,1)),np.eye(9)),axis=1),np.zeros((1,10))),axis=0)
    Phi=A+B@K
    mtx1 = np.concatenate((Phi, np.zeros((np.shape(M)[0], np.shape(Phi)[1]))))
    mtx2=np.concatenate((B@E,M))
    Psi=np.concatenate((mtx1,mtx2),axis=1)
    S=cp.Variable((14,14),PSD=True)
    mtx1=cp.vstack([H,S@F_tilde.T])
    mtx2=cp.vstack([F_tilde@S,S])

    cond_1=cp.hstack([mtx1,mtx2])
    cond_2=cp.hstack([cp.vstack([S,S@Psi.T]),cp.vstack([Psi@S,S])])
    constraints=[cond_1>> 0]
    constraints+=[cond_2 >> 0]
    list_of_vectors=[np.zeros((nc,1)) for k in range(nc)]
    for k in range(nc):
        list_of_vectors[k][k]=1
    constraints+=[vec.T@H@vec<=1 for k,vec in enumerate(list_of_vectors)]
    prob=cp.Problem(cp.Maximize(cp.log_det(S[0:4,0:4])),constraints)
    #prob.solve(max_iters=10000)
    #S_xx=S.value[system_0:system_4,system_0:system_4]
    #P=np.linalg.inv(S_xx)
    #print(P)
    P=np.array([[ 8.50543612e-06,-6.04789205e-06,-8.33176398e-06,-4.39895763e-06],
 [-6.04789205e-06 ,9.93318732e-06 ,9.78329032e-06 ,3.78525362e-06],
 [-8.33176398e-06 , 9.78329032e-06 , 1.30540025e-05 , 4.79938239e-06],
 [-4.39895763e-06,  3.78525362e-06,  4.79938239e-06 , 3.85928974e-06]])
    mtx = np.zeros((2, 2))
    mtx[0, 0] = P[0, 0]
    mtx[0, 1] = P[0, 1]
    mtx[1, 0] = P[1,0]
    mtx[1, 1] = P[1, 1]
    eigenvalues, eigenvectors = np.linalg.eig(mtx)
    theta = np.linspace(0, 2 * np.pi, 1000)
    ellipsis = (1 / np.sqrt(eigenvalues[None, :]) * eigenvectors) @ [np.sin(theta), np.cos(theta)]
    #plt.plot(ellipsis[system_0, :], ellipsis[system_1, :])
    #plt.show()

    mpc.prepare_nlp()

    # Create new constraint:Terminal constraint
    extra_cons = mpc.opt_x['_x', -1, 0][0][0:4].T@P @ mpc.opt_x['_x', -1, 0][0][0:4] - 1
    mpc.nlp_cons.append(
        extra_cons
    )
    mtx=np.zeros(extra_cons.shape)
    mtx.fill(-1)
    # Create appropriate upper and lower bound (here they are both system_0 to create an equality constraint)
    mpc.nlp_cons_lb.append(mtx)
    mpc.nlp_cons_ub.append(np.zeros(extra_cons.shape))

    mpc.create_nlp()

    return mpc
