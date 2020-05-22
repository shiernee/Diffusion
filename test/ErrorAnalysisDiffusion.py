import os
import sys
from os import sys, path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

import unittest
import time
import numpy as np
import copy
import pandas as pd
from src.utils.RbfInterpolator import RbfInterpolatorSphCubic, RbfInterpolatorCartesian, RbfInterpolatorIDW, \
    RbfInterpolatorGaussian, NearestNBInterpolator, LinearNBInterpolator, IDWNBInterpolator
from src.variables.QuadraticFormDivCalculator import QuadraticFormDivCalculator
from src.variables.DiffusionOperatorFitMethod import DiffusionOperatorFitMethod
from src.utils.IDWInterpolator import IDWInterpolator
from src.utils.ExactInterpolator4test import ExactInterpolator4test
from src.pointcloud.PointCloud import PointCloud
from src.variables.LinearOperator import LinearOperator
from src.variables.CubicOperator import CubicOperator
from src.variables.DiffusionOperator import DiffusionOperator
from src.variables.BaseVariables import BaseVariables
from src.variables.Variables import Variables
from src.utils.DataFrame import DataFrame
from Utils.Utils import xyz2r_phi_theta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.variables.FHNeq import FHNeq

'''
this code is to test the error of u at time t using 
FHN equation diffusion term

'''

def test_u_value_at_time_n(duration, dt):
    print('========= start test_u_value_at_time_n ==========')
    print('========= u0 = np.cos(theta) ====================')
    print('== u_exact_at_time = np.exp(-2 * t) * np.cos(theta)====')

    dataframe = get_variable()
    grid_length = 0.2
    local_grid_resolution = 19

    pt_cld = PointCloud(dataframe.get_coord(), grid_length, local_grid_resolution)
    print('interpolated_spacing = ', pt_cld.interpolated_spacing)

    x, y, z = pt_cld.coord[:, 0], pt_cld.coord[:, 1], pt_cld.coord[:, 2]
    r, phi, theta = xyz2r_phi_theta(x, y, z)

    u0 = np.cos(theta)

    interp = RbfInterpolatorSphCubic(pt_cld)

    u = Variables(pt_cld, interp, 0)
    w = Variables(pt_cld, interp, 0)
    u.set_val(u0)  # set to some initial values
    w.set_val(np.zeros(u0.shape))  # set to some initial values

    u0 = np.cos(theta)

    # set up the variables
    D = Variables(pt_cld, interp, 0)
    D.set_val(np.ones(u0.shape))

    a = Variables(pt_cld, interp, 0)
    a.set_val(np.zeros(u0.shape))

    b = Variables(pt_cld, interp, 0)
    b.set_val(np.zeros(u0.shape))

    epsilon_beta = Variables(pt_cld, interp, 0)
    epsilon_beta.set_val(np.zeros(u0.shape))

    neg_epsilon_gamma = Variables(pt_cld, interp, 0)
    neg_epsilon_gamma.set_val(np.zeros(u0.shape))

    neg_epsilon_delta = Variables(pt_cld, interp, 0)
    neg_epsilon_delta.set_val(np.zeros(u0.shape))

    fhn = FHNeq(a, b, epsilon_beta, neg_epsilon_gamma, neg_epsilon_delta, D, pt_cld, interp, dt)
    print('dt: ', fhn.dt)

    nstep = int(duration / fhn.dt)

    # === fhn.integrate(u, w, nsteps) =====
    fhn.u0.copy(u)  # initialize the variables
    fhn.u1.copy(u)
    fhn.w0.copy(w)
    fhn.w1.copy(w)

    text = ''
    for itr in range(nstep):

        start = time.time()

        idx_cur = (itr) % 2
        idx_nxt = (itr + 1) % 2
        u_cur = fhn.U[idx_cur]
        w_cur = fhn.W[idx_cur]

        u_exact_at_time = np.exp(-2 * itr * fhn.dt) * np.cos(theta)  # for every point in grid
        max_u_error = np.max(u_exact_at_time - u_cur.get_val())
        max_rel_u_error = np.max(abs((u_exact_at_time - u_cur.get_val()) / u_exact_at_time))
        mean_rel_u_error = np.mean(abs((u_exact_at_time - u_cur.get_val()) / u_exact_at_time))
        median_rel_u_error = np.median(abs((u_exact_at_time - u_cur.get_val()) / u_exact_at_time))

        dudt = fhn.deUu.eval(u_cur)
        dudt += fhn.deUw.eval(w_cur)

        dwdt = fhn.deWu.eval(u_cur)
        dwdt += fhn.deWw.eval(w_cur)
        dwdt += fhn.deWc.eval()

        printout = 'itr: {}, max_u_error: {}, max_rel_u_error: {} , mean_rel_u_error: {}, ' \
                   'median_rel_u_error: {}'.format(
            itr, max_u_error, max_rel_u_error, mean_rel_u_error, median_rel_u_error)

        u_cur.eval(dudt, fhn.dt)
        w_cur.eval(dwdt, fhn.dt)

        fhn.U[idx_nxt].copy(u_cur)
        fhn.W[idx_nxt].copy(w_cur)

        end = time.time()
        process_time = end-start
        printout = printout + ',  nn time: ' + str(process_time) + '\n'
        print(printout)

        text = text + printout

        if itr % 1 == 0:
            with open('error_analysis.txt', 'w') as fp:
                fp.write(text)
            fp.close()

def get_variable():
    dataframe = DataFrame("database5000_normdist.csv")

    print('Test Diffusion Code')
    print('A unit Sphere of {} points'.format(len(dataframe.get_coord())))

    return dataframe


if __name__ == '__main__':
    duration = 1e-4  #2sec
    dt = 5e-5
    test_u_value_at_time_n(duration, dt)

