import sys
sys.path.insert(1, '/home/sawsn/Shiernee/Utils/src/utils')

import numpy as np
import matplotlib.pyplot as plt
from Utils import Utils
from scipy.interpolate import Rbf


class ViewResultsUtils:
    def __init__(self):
        # x, y, z = 1D array
        # t = 1D array. eg; 0, 0.01, 0.02, 0.03, 0.04
        # u: length of u = no_pt * t

        self.x = None
        self.y = None
        self.z = None
        self.t = None
        self.Xi = None
        self.Yi = None
        self.no_pt = None
        self.max_x = None
        self.max_y = None

        self.u = None
        self.t = None
        self.ut = Utils()

        # cart_coord = np.array([self.x, self.y, self.z]).transpose()
        # self.r, self.phi, self.theta = self.ut.xyz2sph(cart_coord)

    def assign_no_pt(self, no_pt):
        self.no_pt = no_pt
        return

    def assign_x_y_z(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.max_x, self.max_y = round(x.max()), round(y.max())
        self.Xi, self.Yi = self.generate_meshgrid()
        return

    def generate_meshgrid(self):
        x, y, z = self.x.copy(), self.y.copy(), self.z.copy()
        Xi = np.arange(round(x.min()), round(x.max() + 2), 2)
        Yi = np.arange(round(y.min()), round(y.max() + 2), 2)
        Xi, Yi = np.meshgrid(Xi, Yi)
        return Xi, Yi

    def assign_u_t(self, u, t):
        self.u = np.reshape(u, [-1, self.no_pt])
        self.t = t
        return

    def get_var_specific_period(self, start_time, end_time, variable):
        dt = self.t[2] - self.t[1]
        start_time_step, end_time_step = int(start_time / dt), int(end_time / dt) + 1
        return variable[start_time_step:end_time_step].copy()

    def get_u_t_at_specific_time(self, start_time, end_time):
        u_specific_period = self.get_u_specific_period(start_time, end_time)
        t = self.get_t_specific_period(start_time, end_time)
        return u_specific_period, t

    def plot_theta_phi_V(self, start_time, end_time, variable, no_of_plot):
        dt = self.t[2] - self.t[1]

        sqrt_no_of_plot = int(np.sqrt(no_of_plot))
        start_time_step, end_time_step = int(start_time / dt), int(end_time / dt) + 1
        step_plot = int((end_time_step - start_time_step + 1) // no_of_plot)
        var_specific_period = self.get_var_specific_period(start_time, end_time, variable)

        coord = np.array([self.x, self.y, self.z]).transpose()
        coord = coord - coord.mean(axis=0)
        _, phi, theta = self.ut.xyz2sph(coord)

        thetaI = np.linspace(theta.min(), theta.max())
        phiI = np.linspace(phi.min(), phi.max())
        thetaI, phiI = np.meshgrid(thetaI, phiI)
        cbar_min, cbar_max = var_specific_period.min(), var_specific_period.max()

        fig1 = plt.figure()

        for i in range(no_of_plot):
            print(i)
            ax = fig1.add_subplot(sqrt_no_of_plot, sqrt_no_of_plot, i + 1)
            ax.set_xlabel('\u03F4')  # THETA
            ax.set_ylabel('\u03C6')  # PHI

            u_tmp = var_specific_period[i * step_plot]

            # === plot raw ===
            cbar = ax.scatter(theta, phi, c=u_tmp, vmin=cbar_min, vmax=cbar_max)
            # === plot smooth before plotting ====
            # rbf = Rbf(theta, phi, u_tmp, function='linear', smooth=1)
            # V_I = rbf(thetaI, phiI)
            # cbar = ax.imshow(V_I, origin='lower', extent=[thetaI.min(), thetaI.max(), phiI.min(), phiI.max()],
            #                  vmin=cbar_min, vmax=cbar_max)
            ax.set_title('t={:.5f}'.format(self.t[i * step_plot]))
            fig1.colorbar(cbar, ax=ax)

        fig1.tight_layout()
        # plt.show()
        return

    def plot_u_theta(self, start_time, end_time, skip_dt, variable):
        dt = self.t[2] - self.t[1]
        var_specific_period = self.get_var_specific_period(start_time, end_time, variable)

        assert np.ndim(var_specific_period) == 2, 'u_update should be 2D array (time_pt x n_coord)'

        coord = np.array([self.x, self.y, self.z]).transpose()
        _, phi, theta = self.ut.xyz2sph(coord)

        # == sort theta for plotting raw data  (not used, sort theta) ====
        ind = np.argsort(theta)
        skip_time_step = int(skip_dt * (var_specific_period.shape[0] - 1) / self.t[-1])

        # === plot smooth ===
        thetaI = np.linspace(theta.min(), theta.max())
        phiI = np.linspace(phi.min(), phi.max())
        thetaI, phiI = np.meshgrid(thetaI, phiI)
        cbar_min, cbar_max = var_specific_period.min(), var_specific_period.max()

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        for n in range(0, np.shape(var_specific_period)[0], skip_time_step):
            u_tmp = var_specific_period[n]
            # ==== sort theta for plotting raw data  (not used, sort theta)=====
            ax1.plot(theta[ind], u_tmp, label='time:{0:0.2f}'.format(n * dt))
            # === smooth ====
            # rbf = Rbf(theta, phi, u_tmp, function='linear', smooth=1)
            # V_I = rbf(thetaI, phiI)
            # ax1.plot(thetaI[0], V_I[0], label='time:{0:0.2f}'.format(n * dt), vmin=0, vmax=1)

        ax1.set_xlabel('\u03F4')  #THETA
        ax1.set_ylabel('u')
        # ax1.set_xlim(thetaI.min(), thetaI.max())
        ax1.set_ylim(0, 100)
        ax1.legend()
        fig1.tight_layout()
        return

    def plot_u_phi(self, skip_dt):
        assert np.ndim(self.u_) == 2, 'u_update should be 2D array (time_pt x n_coord)'
        ind = np.argsort(self.phi)
        dt = self.t[2] - self.t[1]
        skip_time_step = int(skip_dt * (self.t.shape[0] - 1) / self.t[-1])

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        for n in range(0, np.shape(self.u_)[0], skip_time_step):
            ax1.plot(self.phi[ind], self.u_[n, ind], label='time:{0:0.2f}'.format(n * dt))
        ax1.set_xlabel('phi')
        ax1.set_ylabel('u')
        ax1.legend()
        return

    def plot_u_1D(self, skip_dt):
        dt = self.t[2] - self.t[1]
        skip_time_step = int(skip_dt * (self.t.shape[0] - 1) / self.t[-1])

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        for n in range(0, np.shape(self.u_)[0], skip_time_step):
            ax1.plot(self.x, self.u_[n], label='time:{0:0.2f}'.format(n * dt))
        ax1.set_xlabel('x')
        ax1.set_ylabel('u')
        ax1.legend()
        return
    
    def plot_u_2D(self, skip_dt):
        dt = self.t[2] - self.t[1]
        skip_time_step = int(skip_dt * (self.t.shape[0] - 1) / self.t[-1])

        for n in range(0, np.shape(self.u_)[0], skip_time_step):
            fig1 = plt.figure(n + 1)
            ax1 = fig1.add_subplot(111)
            c = ax1.scatter(self.x, self.y, c=self.u_[n])
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_title('time: {0:0.2f}'.format(n * dt))
            fig1.colorbar(c)

        return
