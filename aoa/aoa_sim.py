import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eigh
from numpy.linalg import eigvals
from numpy.linalg import inv


class Array():
    def __init__(self, type, phis, thetas, SNR=10, M=4, N=5):
        """
        :param type: type of array: linear, square or circular
        :param phis: list of phi angles for the sources
        :param thetas: list of theta angles for the sources
        :param SNR: signal noise ratio
        :param M: number of elements for a array(M*M elements for a square array)
        :param N: number of observations for estimating covariance matrix
        """
        self.type = type
        self.phis = phis
        self.thetas = thetas
        self.SNR = SNR
        self.M = M
        self.N = N
        self.f = 2.4 * 10e9
        self.c = 3 * 10e8

    def steering(self, phi, theta):
        """
        calculating the corresponding steering vector
        :param phi: phi angle in degree
        :param theta: theta angle in degree
        :return: steering vector
        """
        phi = phi / 180 * np.pi
        theta = theta / 180 * np.pi
        if self.type == 'linear':
            self.d = self.c / self.f / 2
            res = np.exp(-1j * 2 * np.pi * self.f / self.c *
                         self.d * np.cos(phi) * np.arange(self.M))
        elif self.type == 'square':
            self.d = self.c / self.f / 2
            res = []
            for i in np.arange(1, self.M + 1):
                for j in np.arange(1, self.M + 1):
                    res.append(np.exp(-1j * 2 * np.pi * self.f / self.c *
                                      ((i - 1) * self.d * np.sin(theta) * np.cos(phi) +
                                       (j - 1) * self.d * np.sin(theta) * np.sin(phi))))
        else:
            self.a = self.c / self.f
            res = []
            for i in np.arange(1, self.M + 1):
                cos = np.cos(2 * np.pi * i / self.M)
                sin = np.sin(2 * np.pi * i / self.M)
                res.append(np.exp(-1j * 2 * np.pi * self.f / self.c *
                                  (a * np.sin(theta) * np.cos(phi) * cos +
                                   a * np.sin(theta) * np.sin(phi) * sin)))
        return np.array(res)

    def sim_1(self):
        """
        simulating one observation vector by adding a certain level of noise
        :return:
        """
        for i, (phi, theta) in enumerate(zip(self.phis, self.thetas)):
            s = np.power(10, self.SNR / 10) * np.exp(1j * np.random.rand() * 2 * np.pi)
            b = np.exp(1j * np.random.rand() * 2 * np.pi)
            if i == 0:
                x = s * self.steering(phi, theta) + b
            else:
                x += s * self.steering(phi, theta) + b
        return x

    def sim_N(self):
        """
        simulating the observation matrix(N columns)
        :return:
        """
        if self.type == 'square':
            X = np.zeros([self.M * self.M, self.N], dtype='complex128')
        else:
            X = np.zeros([self.M, self.N], dtype='complex128')
        for i in np.arange(self.N):
            X[:, i] = self.sim_1()
        return X

    def capon(self, X, phi_list=np.arange(0, 90, 1), theta_list=np.arange(0, 90, 1)):
        """
        calculating the spectrum using CAPON method
        :param X: observation matrix(N columns)
        :param phi_list: search grid for phi angle
        :param theta_list: search grid for theta angle
        :return: spectrum matrix
        """
        R = np.matmul(X, X.conj().T) / self.N
        iRx = inv(R)
        P = np.zeros([len(theta_list), len(phi_list)])
        for i in np.arange(len(phi_list)):
            for j in np.arange(len(theta_list)):
                a = np.expand_dims(self.steering(phi_list[i], theta_list[j]), axis=1)
                aH = a.conj().T
                P[j, i] = 1 / np.abs(aH.dot(iRx).dot(a))
        return np.log(P / np.max(np.max(P)))

    def music(self, X, phi_list, theta_list):
        """
        calculating the spectrum using MUSIC method
        :param X: observation matrix
        :param phi_list: search grid for phi angle
        :param theta_list: search grid for theta angle
        :return: spectrum matrix
        """
        R = np.matmul(X, X.conj().T) / self.N
        eig_values, eig_vectors = eigh(R)
        eig_values = np.abs(eig_values)
        eig_values_ratio = eig_values / np.sum(eig_values)
        M_K = np.max([np.sum(eig_values_ratio < 0.01), 1])
        Eb = eig_vectors[:, :M_K]
        EbH = Eb.conj().T
        P = np.zeros([len(theta_list), len(phi_list)])
        for i in np.arange(len(phi_list)):
            for j in np.arange(len(theta_list)):
                a = np.expand_dims(self.steering(phi_list[i], theta_list[j]), axis=1)
                aH = a.conj().T
                P[j, i] = 1 / np.abs(aH.dot(Eb).dot(EbH).dot(a))
        print('eigen values: ', eig_values_ratio)
        print('K = ', str(self.M - M_K))
        return np.log(P / np.max(np.max(P)))

    def esprit(self, X):
        """
        calculating the direction of angle using ESPRIT method
        :param X: observation matrix
        :return: angles
        """
        R = np.matmul(X, X.conj().T) / self.N
        eig_values, eig_vectors = eigh(R)
        eig_values = np.abs(eig_values)
        eig_values_ratio = eig_values / np.sum(eig_values)
        M_K = np.max([np.sum(eig_values_ratio < 0.01), 1])
        Es = eig_vectors[:, M_K:]
        Es1 = Es[:-1, :]
        Es2 = Es[1:, :]
        Es1H = Es1.conj().T
        F = inv(Es1H.dot(Es1)).dot(Es1H).dot(Es2)
        res_eigvals = np.angle(eigvals(F))
        res_cos = -res_eigvals / (2 * np.pi * self.f * self.d / self.c)
        print('eigen values: ', eig_values_ratio)
        print('K = ', str(self.M - M_K))
        return np.arccos(res_cos) / np.pi * 180

    @staticmethod
    def plot_spectrum_1D(ax, spectrum, phi_list, label):
        """
        plot spectrum for linear array(spectrum vs phi)
        :param ax: figure ax
        :param spectrum: spectrum vector
        :param phi_list: search grid for phi
        :return: figure
        """
        ax.plot(phi_list, spectrum, label=label)
        ax.set_xlim([np.min(phi_list), np.max(phi_list)])
        ax.set_xlabel(r'$\varphi$: degree')
        ax.set_ylabel('Spectrum: dB')
        ax.grid()
        ax.legend()

    @staticmethod
    def plot_spectrum_2D(ax, spectrum, phi_list, theta_list):
        """
        plot spectrum for square or circular array(spectrum vs phi and theta)
        :param ax: figure ax
        :param spectrum: spectrum matrix
        :param phi_list: search grid for phi
        :param theta_list: search grid for theta
        :return: figure
        """
        phi_grid, theta_grid = np.meshgrid(phi_list, theta_list)
        ax.pcolor(phi_grid, theta_grid, spectrum)
        ax.set_xlabel(r'$\varphi$')
        ax.set_ylabel(r'$\theta$')


if __name__ == '__main__':
    # linear array
    phis = [30, 45]
    thetas = np.ones(len(phis)) * 90
    phi_list = np.arange(1, 180, 1)
    theta_list = [90]
    linear = Array('linear', phis, thetas, SNR=20, M=4, N=5)
    X = linear.sim_N()
    P_music = linear.music(X, phi_list, theta_list)
    fig, ax = plt.subplots()
    Array.plot_spectrum_1D(ax, P_music[0], phi_list, 'Music')
    res_esprit = linear.esprit(X)
    print('angles using ESPRIT: ', res_esprit)

    # square and circular array
    # phis = [30, 60]
    # thetas = [20, 40]
    # phi_list = np.arange(1, 360, 2)
    # theta_list = np.arange(1, 90, 2)
    # square = Array('square', phis, thetas)
    # X = square.sim_N()
    # P_music = square.music(X, phi_list, theta_list)
    # fig, ax = plt.subplots()
    # Array.plot_spectrum_2D(ax, P_music, phi_list, theta_list)
    # circular = Array('circular', phis, thetas, M=8)
    # X = circular.sim_N()
    # P_music = circular.music(X, phi_list, theta_list)
    # fig, ax = plt.subplots()
    # Array.plot_spectrum_2D(ax, P_music, phi_list, theta_list)

    plt.show()
