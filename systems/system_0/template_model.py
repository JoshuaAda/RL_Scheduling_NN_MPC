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

import numpy as np
from casadi import *
from casadi.tools import *

import sys
sys.path.append('../../')
import do_mpc


def template_model(symvar_type='SX'):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'discrete' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    # Simple oscillating masses example with two masses and two inputs.
    # States are the position and velocitiy of the two masses.

    # States struct (optimization variables):
    _x = model.set_variable(var_type='_x', var_name='x', shape=(2,1))
    x_tilde = model.set_variable(var_type='_tvp', var_name='x_tilde',shape=(2,1))


    K=model.set_variable('_tvp', var_name='K',shape=(2,1))
    T=model.set_variable('_tvp', var_name='T',shape=(2,2))
    P = model.set_variable('_tvp', var_name='P',shape=(2,2))
    F = model.set_variable('_tvp', var_name='F',shape=(2,2))
    A = model.set_variable('_tvp', var_name='A',shape=(2,2))
    B = model.set_variable('_tvp', var_name='B',shape=(2,1))
    Q = model.set_variable('_tvp', var_name='Q',shape=(2,2))


    # Input struct (optimization variables):
    _u = model.set_variable(var_type='_u', var_name='u', shape=(1,1))

    # Set expression. These can be used in the cost function, as non-linear constraints
    # or just to monitor another output.
    model.set_expression(expr_name='cost', expr=(_x-x_tilde).T@Q@(_x-x_tilde))
    model.set_expression(expr_name='ter_cost',expr=(_x-x_tilde).T@Q@(_x-x_tilde))


    x_next = A@_x+B@_u+0.1*sqrt(_x.T@_x+0.1)
    model.set_rhs('x', x_next,process_noise=True)

    model.setup()

    return model
