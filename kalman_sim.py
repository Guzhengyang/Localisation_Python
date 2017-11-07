from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

dt = 1 / 100


def kf_predict(X, P, F, G, Q):
    """
    kalman predict step
    :param X: state vector
    :param P: error covariance matrix
    :param F: process transition matrix
    :param G: process noise matrix
    :param Q: process noise covariance matrix
    :return: predicted state vector and error covariance matrix
    """
    X = F.dot(X)
    P = F.dot(P.dot(F.T)) + G.dot(Q.dot(G.T))
    return X, P


def kf_update(X, P, Z, H, R):
    """
    kalman update step
    :param X: state vector
    :param P: error covariance matrix
    :param Z: observation vector
    :param H: observation transition matrix
    :param R: observation noise matrix
    :return: updated state vector, error covariance matrix and kalman filter
    """
    K = P.dot(H.T.dot(inv(H.dot(P.dot(H.T)) + R)))
    X = X + K.dot(Z - H.dot(X))
    P = P - K.dot(H.dot(P))
    return X, P, K


def init_CV(q=1):
    """
    initialize matrix for Constant Velocity Model(CV)
    :param q: process noise value
    :return: matrix for process and observation
    """
    global dt
    X = np.array([[0], [0], [0], [0]])
    P = np.eye(X.shape[0])
    F = np.array([[1, dt, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, dt],
                  [0, 0, 0, 1]])
    G = np.array([[dt * dt / 2, 0],
                  [dt, 0],
                  [0, dt * dt / 2],
                  [0, dt]])
    H = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0]])
    Q = q * np.eye(2)
    R = np.eye(H.shape[0])
    return X, P, F, G, H, Q, R, 'CV'


def init_CA(q=1):
    """
    initialize matrix for Constant Acceleration Model(CA)
    :param q: process noise value
    :return:
    """
    global dt
    X = np.array([[0], [0], [0], [0], [0], [0]])
    P = np.eye(X.shape[0])
    F = np.array([[1, dt, dt * dt / 2, 0, 0, 0],
                  [0, 1, dt, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, dt, dt * dt / 2],
                  [0, 0, 0, 0, 1, dt],
                  [0, 0, 0, 0, 0, 1]])
    G = np.array([[0, 0],
                  [0, 0],
                  [1, 0],
                  [0, 0],
                  [0, 0],
                  [0, 1]])
    H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 1]])
    Q = q * np.eye(2)
    R = np.eye(H.shape[0])
    return X, P, F, G, H, Q, R, 'CA'


def sim_process(N, type=0):
    """
    simulate real trajectory and observed info
    :param N: number of sampling times
    :param type: type of process(0->static position,
                                1->constant velocity for x and y,
                                2->constant velocity for x and constant acceleration for y,
                                3->random acceleration for x and y)
    :return: real trajectory and observed info(coordinate and acceleration)
    """
    global dt
    Z_real = np.array([[0], [0]])
    Z_observed = Z_real + np.array([[np.random.randn()], [np.random.randn()]])
    v = np.array([[0], [0]])
    a_temp = np.array([[0], [0]])
    a_observed = a_temp + np.array([[np.random.randn()], [np.random.randn()]])
    for i in range(N):
        T = (i + 1) * dt
        if type == 0:
            a_temp = np.array([[0], [0]])
            Z_temp = np.array([[0], [0]])
            a_observed = np.append(a_observed,
                                   a_temp + np.array([[np.random.randn()], [np.random.randn()]]), axis=1)
        elif type == 1:
            a_temp = np.array([[0], [0]])
            Z_temp = np.array([[T * 2], [T * 1]])
            a_observed = np.append(a_observed,
                                   a_temp + np.array([[np.random.randn()], [np.random.randn()]]), axis=1)
        elif type == 2:
            a_temp = np.array([[0], [1]])
            Z_temp = np.array([[T * 2], [T * T / 2]])
            a_observed = np.append(a_observed,
                                   a_temp + np.array([[np.random.randn()], [np.random.randn()]]), axis=1)
        else:
            a_temp = 10 * np.array([[np.random.randn()], [np.random.randn()]])
            v = v + a_temp * dt
            Z_temp = np.array([[Z_real[0, -1] + v[0, 0] * dt],
                               [Z_real[1, -1] + v[1, 0] * dt]])
            a_observed = np.append(a_observed,
                                   a_temp + np.array([[np.random.randn()], [np.random.randn()]]), axis=1)
        Z_real = np.append(Z_real, Z_temp, axis=1)
        Z_observed = np.append(Z_observed, Z_temp + np.array([[np.random.randn()], [np.random.randn()]]), axis=1)
    return Z_real, np.vstack((Z_observed[0, :],
                              a_observed[0, :],
                              Z_observed[1, :],
                              a_observed[1, :]))


def plot_result(Z_observed, Z_estimed, Z_real, ax, title=''):
    """
    util function: show the result of applying kalman filter
    :param Z_observed: observed trajectory
    :param Z_estimed: estimated trajectory with kalman filter
    :param Z_real: real trajectory
    :param ax: plot ax
    :param title: title of
    :return:
    """
    rmse0 = np.mean(np.linalg.norm(Z_observed - Z_real, axis=0))
    rmse1 = np.mean(np.linalg.norm(Z_estimed - Z_real, axis=0))
    ax.plot(Z_observed[0, :], Z_observed[1, :], 'r.', label='observed')
    ax.plot(Z_estimed[0, :], Z_estimed[1, :], color='b', lw=3, label='estimated')
    ax.plot(Z_real[0, :], Z_real[1, :], 'ks', ms=5, label='real')
    ax.legend()
    ax.grid()
    ax.set_title("{} : RMSE = {:.3f} --> {:.3f}".format(title, rmse0, rmse1))
    ax.set_xlabel('X: m')
    ax.set_ylabel('Y: m')


def plot_error(error):
    """
    util function: plot a error figure
    :param error: RMSE
    :return: figure
    """
    plt.figure()
    plt.plot(error)
    plt.grid()
    plt.xlabel('Iteration')
    plt.title('Error Covariance Matrix')


def sim_CV(Z_real, Z_observed):
    """
    simulate the result using CV model
    :param Z_real: real trajectory
    :param Z_observed: observed info
    :return: estimated trajectory
    """
    X, P, F, G, H, Q, R, model = init_CV()
    error = np.matrix.trace(P)
    N = Z_real.shape[1]
    for i in range(N):
        X, P = kf_predict(X, P, F, G, Q)
        Z_temp = Z_observed[:2, i][:, np.newaxis]
        X, P, K = kf_update(X, P, Z_temp, H, R)
        Z_estimed_temp = np.array([[X[0, 0]], [X[2, 0]]])
        error = np.append(error, np.matrix.trace(P))
        if i == 0:
            Z_estimated = Z_estimed_temp
        else:
            Z_estimated = np.append(Z_estimated, Z_estimed_temp, axis=1)
    return Z_estimated


def sim_CA(Z_real, Z_observed):
    """
    simulate the result using CA model
    :param Z_real: real trajectory
    :param Z_observed: observed info
    :return: estimated trajectory
    """
    X, P, F, G, H, Q, R, model = init_CA()
    error = np.matrix.trace(P)
    N = Z_real.shape[1]
    for i in range(N):
        X, P = kf_predict(X, P, F, G, Q)
        Z_temp = Z_observed[:, i][:, np.newaxis]
        X, P, K = kf_update(X, P, Z_temp, H, R)
        Z_estimed_temp = np.array([[X[0, 0]], [X[3, 0]]])
        error = np.append(error, np.matrix.trace(P))
        if i == 0:
            Z_estimated = Z_estimed_temp
        else:
            Z_estimated = np.append(Z_estimated, Z_estimed_temp, axis=1)
    return Z_estimated


if __name__ == '__main__':
    N = 1000
    type = 3
    for i in range(1):
        Z_real, Z_observed = sim_process(N, type)
        Z_estimated_CV = sim_CV(Z_real, Z_observed[[0, 2]])
        Z_estimated_CA = sim_CA(Z_real, Z_observed)
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
        plot_result(Z_observed[[0, 2]], Z_estimated_CV, Z_real, axs[0], 'CV')
        plot_result(Z_observed[[0, 2]], Z_estimated_CA, Z_real, axs[1], 'CA')
    plt.show()
