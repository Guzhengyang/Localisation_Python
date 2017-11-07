import os
import time
from threading import Lock
from threading import Thread

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import serial
from matplotlib import style
from numpy.linalg import eigh, inv

style.use("ggplot")


class AOA():
    def __init__(self, ser, file_name='test', dir_name='data'):
        """
        :param ser: serial object
        :param file_name: saved file name
        :param dir_name: directory for saving the files
        """
        self.ser = ser
        self.file_name = file_name
        self.dir = dir_name
        self.file_start = True
        self.M = 4
        self.N = 5
        self.f = 2.380 * 10e9
        self.c = 3 * 10e8
        self.d = 0.125 / 4
        self.line_gain = None
        self.lines_x = None
        self.lines_prefix = None
        self.gain = 0
        self.prefix = np.zeros([544, 2])
        self.DCOC = np.zeros(2)
        self.X = np.zeros([self.M, self.N], dtype='complex128')
        self.arg = np.zeros(self.M)
        self.arg_diff = np.zeros(self.M - 1)
        self.X_corrected = np.zeros([self.M, self.N], dtype='complex128')
        self.arg_corrected = np.zeros(self.M)
        self.arg_diff_corrected = np.zeros(self.M - 1)
        self.phi_list = np.arange(-180, 181, 1)
        self.P_capon = np.zeros(self.phi_list.shape)
        self.P_music = np.zeros(self.phi_list.shape)
        self.lock = Lock()
        self.start_uart()
        self.init_plot()

    def init_plot(self):
        """
        initialize plot figure: set arrangements, titles, sticks, labels...
        :return: plot figure
        """
        self.fig = plt.figure(figsize=(16, 16))
        ax1 = self.fig.add_subplot(221)
        ax2 = self.fig.add_subplot(223)
        ax3 = self.fig.add_subplot(222, projection='polar')
        self.ax4 = self.fig.add_subplot(224)

        # plot arg for each antenna
        self.plot_arg, = ax1.plot(range(len(self.arg)), self.arg, c='b', marker='o', label='Original')
        self.plot_arg_corrected, = ax1.plot(range(len(self.arg_corrected)), self.arg_corrected, c='g',
                                            marker='o', label='Corrected')
        ax1.set_ylim([0, 360])
        ax1.set_title('Arg')
        ax1.set_xticks(range(len(self.arg)))
        ax1.set_xticklabels(['arg1', 'arg2', 'arg3', 'arg4'])
        ax1.legend()

        # plot arg difference for each two successive antenna
        self.plot_arg_diff, = ax2.plot(range(len(self.arg_diff)), self.arg_diff, c='b', marker='o',
                                       label='Original')
        self.plot_arg_diff_corrected, = ax2.plot(range(len(self.arg_diff_corrected)),
                                                 self.arg_diff_corrected, c='g', marker='o',
                                                 label='Corrected')
        ax2.set_ylim([0, 360])
        ax2.set_title('Arg Difference')
        ax2.set_xticks(range(len(self.arg_diff)))
        ax2.set_xticklabels(['arg2 - arg1', 'arg3 - arg2', 'arg4 - arg3'])
        ax2.legend()

        # plot spectrum for MUSIC and CAPON method
        # self.plot_spectrum_capon, = ax3.plot(self.phi_list / 180 * np.pi, self.P_capon, c='b', label='CAPON')
        self.plot_spectrum_music, = ax3.plot(self.phi_list / 180 * np.pi, self.P_music, c='g', label='MUSIC')
        ax3.set_title('Spectrum')
        ax3.set_ylim([0, 1.2])
        ax3.set_xlabel('Degree')
        ax3.legend()

        # plot prefix signal for debug purpose
        self.plot_prefix_I, = self.ax4.plot(self.prefix[:, 0], c='b', label='I')
        self.plot_prefix_Q, = self.ax4.plot(self.prefix[:, 1], c='g', label='Q')
        self.ax4.set_ylim([-2000, 2000])
        self.ax4.legend()
        self.ax4.set_title('gain: {}, DC_I: {}, DC_Q: {}'.format(self.gain, self.DCOC[0], self.DCOC[1]))

        plt.draw()
        plt.pause(10e-10)

    def start_uart(self):
        """
        write b'\r', b'1', b'3', b'\r' in the order for the first connection
        :return: None
        """
        self.ser.write(b'\r')
        self.ser.readlines()
        self.ser.write(b'1')
        self.ser.readlines()
        self.ser.write(b'3')
        self.ser.readlines()
        self.ser.write(b'\r')

    @staticmethod
    def parse_IQ(line):
        """
        parse a line for IQ values for prefix signal
        :param line: bytes
        :return: I Q value
        """
        res = line.decode().replace('\r', '').replace('\n', '').split(' ')
        res = np.array([int(res[0]), int(res[1])])
        return res

    @staticmethod
    def parse_IQ2x(line):
        """
        parse a line for IQ values for X observation matrix
        :param line: bytes
        :return: two observation values
        """
        res = line.decode().replace('\r', '').replace('\n', '').split(' ')
        res = [int(res[0]) + 1j * int(res[1]), int(res[2]) + 1j * int(res[3])]
        return res

    def parse_gain(self, line):
        """
        parse a line for gain value
        :param line: received line in bytes
        :return: update gain value
        """
        try:
            self.gain = int(line.decode().replace('\r', '').replace('\n', ''))
        except:
            print('parse gain error')

    def parse_prefix(self, lines_prefix):
        """
        parse lines for prefix signal
        :param lines_prefix: received lines in bytes
        :return: update prefix signal
        """
        try:
            for i in range(len(lines_prefix)):
                self.prefix[i, :] = self.parse_IQ(lines_prefix[i])
        except:
            print('parse prefix error')

    def parse_x(self, lines_x):
        """
        parse lines for observation matrix X
        :param lines_x: received lines in bytes
        :return: update observation matrix X
        """
        X_list = []
        try:
            for line in lines_x:
                res = self.parse_IQ2x(line)
                X_list.extend(res)
            X = np.array(X_list).reshape(self.N, 6).T
            self.X = X[[0, 1, 3, 5], :] / 16
        except:
            print('parse x error')

    def save_X(self):
        """
        save IQ in a csv
        :return: None
        """
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        path = os.path.join(self.dir, self.file_name + '.csv')
        X_real = np.vstack((np.real(self.X[0, :]), np.imag(self.X[0, :]),
                            np.real(self.X[1, :]), np.imag(self.X[1, :]),
                            np.real(self.X[2, :]), np.imag(self.X[2, :]),
                            np.real(self.X[3, :]), np.imag(self.X[3, :])))
        df_new = pd.DataFrame(X_real.T, columns=['Antenna1_I', 'Antenna1_Q', 'Antenna2_I', 'Antenna2_Q',
                                                 'Antenna3_I', 'Antenna3_Q', 'Antenna4_I', 'Antenna4_Q'])
        if self.file_start:
            df_new.to_csv(path, sep=';', index=False)
            self.file_start = False
        else:
            df_old = pd.read_csv(path, sep=';')
            df_new = pd.concat([df_old, df_new], ignore_index=True)
            df_new.to_csv(path, sep=';', index=False)

    def read_packet(self):
        """
        read a whole packet for gain, prefix signal and obervation matrix
        :return: update received packet lines
        """
        lines_X = []
        lines_prefix = []
        line = self.ser.readline()
        while b'START' not in line:
            line = self.ser.readline()
        line_gain = self.ser.readline()
        line = self.ser.readline()
        while b'END' not in line:
            lines_X.append(line)
            line = ser.readline()
        line = self.ser.readline()
        while b'STOP' not in line:
            lines_prefix.append(line)
            line = ser.readline()
        lines_prefix.pop()
        lines_prefix.pop()
        with self.lock:
            self.line_gain, self.lines_x, self.lines_prefix = line_gain, lines_X, lines_prefix

    def process_thread(self):
        """
        use a thread to update data
        :param lines: bytes
        :return: update gain, prefix signal and observation matrix
        """
        while True:
            time.sleep(0.5)
            with self.lock:
                if (self.line_gain != None) & (self.lines_x != None) & (self.lines_prefix != None):
                    self.parse_gain(self.line_gain)
                    self.parse_prefix(self.lines_prefix)
                    self.parse_x(self.lines_x)
                    self.save_X()
                    self.arg = np.angle(self.X[:, 2]) / np.pi * 180 % 360
                    self.arg_diff = np.array([self.arg[1] - self.arg[0],
                                              self.arg[2] - self.arg[1],
                                              self.arg[3] - self.arg[2]]) % 360
                    self.correct_data()
                    # self.capon()
                    self.music()

    def correct_data(self):
        """
        correct observation matrix taking DCOC into consideration
        :return: update corrected observation matrix
        """
        self.DCOC = (np.max(self.prefix[:100], axis=0) + np.min(self.prefix[:100], axis=0)) / 2
        offset = self.DCOC[0] + 1j * self.DCOC[1]
        self.X_corrected = self.X - offset
        self.arg_corrected = np.angle(self.X_corrected[:, 2]) / np.pi * 180 % 360
        self.arg_diff_corrected = np.array([self.arg_corrected[1] - self.arg_corrected[0],
                                            self.arg_corrected[2] - self.arg_corrected[1],
                                            self.arg_corrected[3] - self.arg_corrected[2]]) % 360

    def steering(self, phi):
        """
        steering vector for a linear array
        :param phi:
        :return:
        """
        phi = phi / 180 * np.pi
        return np.exp(-1j * 2 * np.pi * self.f / self.c *
                      self.d * np.cos(phi) * np.arange(self.M))

    def music(self):
        """
        use MUSIC method for DOA
        :return: update spectrum
        """
        R = np.matmul(self.X, self.X.conj().T) / self.X.shape[1]
        eig_values, eig_vectors = eigh(R)
        eig_values = np.abs(eig_values)
        eig_values_ratio = eig_values / np.sum(eig_values)
        M_K = np.max([np.sum(eig_values_ratio < 0.01), 1])
        Eb = eig_vectors[:, :M_K]
        EbH = Eb.conj().T
        P = np.zeros(len(self.phi_list))
        for i in np.arange(len(self.phi_list)):
            a = np.expand_dims(self.steering(self.phi_list[i]), axis=1)
            aH = a.conj().T
            P[i] = 1 / np.abs(aH.dot(Eb).dot(EbH).dot(a))
        # print('eig value ration: ', eig_values_ratio)
        # print('K = ', self.M - M_K)
        self.P_music = P / np.max(np.max(P))

    def capon(self):
        """
        use CAPON method for DOA
        :return: update spectrum
        """
        R = np.matmul(self.X, self.X.conj().T) / self.X.shape[1]
        iRx = inv(R)
        P = np.zeros(len(self.phi_list))
        for i in np.arange(len(self.phi_list)):
            a = np.expand_dims(self.steering(self.phi_list[i]), axis=1)
            aH = a.conj().T
            P[i] = 1 / np.abs(aH.dot(iRx).dot(a))
        self.P_capon = P / np.max(np.max(P))


if __name__ == '__main__':

    with serial.Serial(port='COM4', baudrate=115200, timeout=0.1) as ser:
        aoa = AOA(ser)
        thread_process = Thread(target=aoa.process_thread)
        thread_process.start()
        while True:
            aoa.read_packet()
            aoa.plot_arg.set_ydata(aoa.arg)
            aoa.plot_arg_corrected.set_ydata(aoa.arg_corrected)
            aoa.plot_arg_diff.set_ydata(aoa.arg_diff)
            aoa.plot_arg_diff_corrected.set_ydata(aoa.arg_diff_corrected)
            # aoa.plot_spectrum_capon.set_ydata(aoa.P_capon)
            aoa.plot_spectrum_music.set_ydata(aoa.P_music)
            aoa.plot_prefix_I.set_ydata(aoa.prefix[:, 0])
            aoa.plot_prefix_Q.set_ydata(aoa.prefix[:, 1])
            aoa.ax4.set_title('gain: {}, DC_I: {}, DC_Q: {}'.format(aoa.gain, aoa.DCOC[0], aoa.DCOC[1]))
            aoa.fig.canvas.draw()
            aoa.fig.canvas.flush_events()
