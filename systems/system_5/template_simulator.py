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
import pdb
import sys
sys.path.append('../../')
import do_mpc


def template_simulator(model):
    """
    --------------------------------------------------------------------------
    template_optimizer: tuning parameters
    --------------------------------------------------------------------------
    """
    simulator = do_mpc.simulator.Simulator(model)
    A = np.array([[1, 0.5],[0,1]]) #np.array([[0.763, 0.460, 0.115, 0.020],
         #         [-0.899, 0.763, 0.420, 0.115],
         #         [0.115, 0.020, 0.763, 0.460],
         #         [0.420, 0.115, -0.899, 0.763]])

    B = np.array([[0],[1]])#np.array([[0.014],
        #          [0.063],
        #          [0.221],
        #          [0.367]])
    F=np.array([[1,0],[0,1]])#np.array([[1, 0, 0, 0],
      #            [0, 1, 0., 0],
      #            [0, 0, 1, 0],
      #            [0, 0, 0, 1]])

    simulator.set_param(t_step = 0.5)
    tvp_template = simulator.get_tvp_template()

    # Define the function (indexing is much simpler ...)
    def tvp_fun(t_now):

        tvp_template['A'] = A
        tvp_template['B'] = B
        tvp_template['F'] = F
        return tvp_template

    # Set the tvp_fun:
    simulator.set_tvp_fun(tvp_fun)

    simulator.setup()

    return simulator
