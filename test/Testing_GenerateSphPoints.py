import os
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
sys.path.append(path.dirname(path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))))


from src.pointcloud.GeneratePoints import *
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from Utils.Utils import r_phi_theta2xyz, xyz2r_phi_theta


def plot_theta_phi(theta, phi, title):
    plt.figure()
    plt.plot(theta, phi, '.')
    plt.xlabel('theta')
    plt.ylabel('phi')
    plt.title(title)


def plot_3d_sphere(x, y, z, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title(title)


def uniform_distribution_phi_theta(no_pt):
    title = 'uniform_distribution_phi_theta'
    phi = np.random.uniform(-np.pi, np.pi, no_pt)
    theta = np.random.uniform(0, np.pi, no_pt)
    r = np.ones(phi.shape)
    x, y, z = r_phi_theta2xyz(r, phi, theta)
    plot_theta_phi(theta, phi, title)
    plot_3d_sphere(x, y, z, title)
    return x, y, z


def normal_distribution_xyz(no_pt):
    title = 'normal_distribution_xyz'
    x = np.random.standard_normal(no_pt)
    y = np.random.standard_normal(no_pt)
    z = np.random.standard_normal(no_pt)
    r = np.sqrt(x**2 + y**2 + z**2)
    x, y, z = x/r, y/r, z/r
    r, phi, theta = xyz2r_phi_theta(x, y, z)
    plot_theta_phi(theta, phi, title)
    plot_3d_sphere(x, y, z, title)
    return x, y, z


if __name__== '__main__':
    no_pt = 1000
    sph_radius = 1

    uniform_distribution_phi_theta(no_pt)
    normal_distribution_xyz(no_pt)

    plt.show()


