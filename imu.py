import socket
import time
from threading import Thread

import matplotlib.pyplot as plt
import numpy as np


class IMU():
    def __init__(self, ip):
        """
        :param ip: ip of smartphone
        """
        self.ip = ip
        self.port = 8888
        self.acc_z_step = 3
        self.n_sample_step = 5
        self.n_sample_static = 20
        self.da = np.zeros(3)
        self.dt = 0.1
        self.n_sample = 0
        self.n_window = 100
        self.step_length = 0.75
        self.acc = np.zeros((self.n_window, 3))
        self.velocity = np.zeros((self.n_window, 3))
        self.position = np.zeros((self.n_window, 3))
        self.dist = np.zeros((1, 2))
        self.step = np.zeros(self.n_window)
        self.orientation = np.zeros((self.n_window, 3))
        self.tag_acc = '#GLOBALACC#'
        self.tag_orientation = '#ORIENTATION#'
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.init_plot()

    def init_plot(self):
        """
        initialize a figure
        :return: figure
        """
        self.fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 16))
        self.acc_x, = axs[0, 0].plot(self.acc[:, 0], label='Global Acc_X', c='b')
        self.acc_y, = axs[0, 0].plot(self.acc[:, 1], label='Global Acc_Y', c='g')
        self.acc_z, = axs[0, 0].plot(self.acc[:, 2], label='Global Acc_Z', c='r')
        self.line_step1, = axs[0, 0].plot(self.step, label='Step', c='k')
        axs[0, 0].plot(np.ones(self.n_window) * self.acc_z_step)
        axs[0, 0].set_xlim([0, self.n_window])
        axs[0, 0].set_xlabel(r'Packet')
        axs[0, 0].set_ylim([-10, 10])
        axs[0, 0].set_ylabel(r'$m/s^2$')
        axs[0, 0].legend()
        axs[0, 0].grid()
        axs[0, 0].set_title('Global Acc')

        self.vel_x, = axs[0, 1].plot(self.velocity[:, 0], label='Global Velocity_X', c='b')
        self.vel_y, = axs[0, 1].plot(self.velocity[:, 1], label='Global Velocity_Y', c='g')
        self.line_step2, = axs[0, 1].plot(self.step, label='Step', c='k')
        axs[0, 1].set_xlim([0, self.n_window])
        axs[0, 1].set_xlabel(r'Packet')
        axs[0, 1].set_ylim([-2, 2])
        axs[0, 1].set_ylabel(r'$m/s$')
        axs[0, 1].legend()
        axs[0, 1].grid()
        axs[0, 1].set_title('Global Velocity')

        self.azimuth, = axs[1, 0].plot(self.orientation[:, 0], label='Azimuth', c='b')
        axs[1, 0].set_xlim([0, self.n_window])
        axs[1, 0].set_xlabel(r'Packet')
        axs[1, 0].set_ylim([0, 360])
        axs[1, 0].set_ylabel(r'degree')
        axs[1, 0].legend()
        axs[1, 0].grid()
        axs[1, 0].set_title('Orientation')

        self.pos_xy, = axs[1, 1].plot(self.position[:, 0], self.position[:, 1], marker='+', c='b', label='Through Acc')
        self.dist_xy, = axs[1, 1].plot(self.dist[:, 0], self.dist[:, 1], marker='s', c='g', label='Through Step')
        axs[1, 1].set_xlim([-10, 10])
        axs[1, 1].set_xlabel(r'X: m')
        axs[1, 1].set_ylim([-10, 10])
        axs[1, 1].set_ylabel(r'Y: m')
        axs[1, 0].legend()
        axs[1, 1].grid()
        axs[1, 1].set_title('Position')

        plt.draw()
        plt.pause(10e-10)

    def udp_request(self):
        """
        send a request for IMU data to a smartphone
        :return: none
        """
        request = 'REQUEST'
        self.socket.sendto(request.encode(), (self.ip, self.port))

    @staticmethod
    def parse_packet(packet, tag):
        """
        get specific IMU values in a packet according to the tag
        :param packet: UPD packet
        :param tag: '#GLOBALACC#' etc
        :return: timestamp, IMU specific value in a received packet
        """
        items = packet.decode().split('|')
        timestamp = items[0]
        for item in items[1:]:
            res = item.split(' ')
            if res[0] == tag:
                return timestamp, np.array([float(res[1]), float(res[2]), float(res[3])])

    @staticmethod
    def update_velocity(velocity, acc_temp, dt):
        """
        update current velocity
        :param velocity: variable saving velocity values
        :param acc_temp: current acceleration
        :param dt: acceleration sampling period
        :return: current velocity
        """
        return velocity[-1] + acc_temp * dt

    @staticmethod
    def update_position(position, velocity, acc_temp, dt):
        """
        update current position(X, Y)
        :param position: variable saving position
        :param velocity: variable saving velocity
        :param acc_temp: current acceleration
        :param dt: acceleration sampling period
        :return: current position
        """
        return position[-1] + velocity[-1] * dt + acc_temp * dt * dt / 2

    def detect_step(self, acc_temp, n_sample):
        """
        step detection
        :param acc_temp: current acceleration value
        :param n_sample: minimum samples from last detected step
        :return: whether current time corresponds to a step
        """
        if (abs(acc_temp[2]) > self.acc_z_step) & (n_sample > self.n_sample_step):
            return 100
        else:
            return 0

    def udp_thread(self):
        """
        thread for receiving IMU data and saving corresponding results
        :return:
        """
        while True:
            packet = self.socket.recv(128)
            self.n_sample += 1
            timestamp, acc_temp = self.parse_packet(packet, self.tag_acc)
            print(timestamp)
            _, orientation_temp = self.parse_packet(packet, self.tag_orientation)
            acc_temp = acc_temp - self.da
            velocity_temp = self.update_velocity(self.velocity, acc_temp, self.dt)
            position_temp = self.update_position(self.position, self.velocity, acc_temp, self.dt)
            step_temp = self.detect_step(acc_temp, self.n_sample)
            if step_temp != 0:
                self.da = velocity_temp / self.n_sample / self.dt
                self.n_sample = 0
                velocity_xy = np.sqrt(velocity_temp[0] * velocity_temp[0] + velocity_temp[1] * velocity_temp[1])
                dist_temp = np.array([self.step_length * velocity_temp[0] / velocity_xy,
                                      self.step_length * velocity_temp[1] / velocity_xy])
                self.dist = np.vstack((self.dist, self.dist[-1] + dist_temp))
            if self.n_sample > self.n_sample_static:
                velocity_temp = np.zeros(3)
            self.acc = np.delete(self.acc, 0, 0)
            self.acc = np.vstack((self.acc, acc_temp))
            self.velocity = np.delete(self.velocity, 0, 0)
            self.velocity = np.vstack((self.velocity, velocity_temp))
            self.position = np.delete(self.position, 0, 0)
            self.position = np.vstack((self.position, position_temp))
            self.step = np.delete(self.step, 0)
            self.step = np.append(self.step, step_temp)
            self.orientation = np.delete(self.orientation, 0, 0)
            self.orientation = np.vstack((self.orientation, orientation_temp))


if __name__ == '__main__':

    # smartphone and PC need to be in the same network
    imu_client = IMU(ip='192.168.10.29')
    imu_client.udp_request()

    thread_udp = Thread(target=imu_client.udp_thread)
    thread_udp.start()

    while True:
        start = time.time()
        imu_client.acc_x.set_ydata(imu_client.acc[:, 0])
        imu_client.acc_y.set_ydata(imu_client.acc[:, 1])
        imu_client.acc_z.set_ydata(imu_client.acc[:, 2])
        imu_client.line_step1.set_ydata(imu_client.step)
        imu_client.vel_x.set_ydata(imu_client.velocity[:, 0])
        imu_client.vel_y.set_ydata(imu_client.velocity[:, 1])
        imu_client.line_step2.set_ydata(imu_client.step)
        imu_client.pos_xy.set_xdata(imu_client.position[:, 0])
        imu_client.pos_xy.set_ydata(imu_client.position[:, 1])
        imu_client.dist_xy.set_xdata(imu_client.dist[:, 0])
        imu_client.dist_xy.set_ydata(imu_client.dist[:, 1])
        imu_client.azimuth.set_ydata(imu_client.orientation[:, 0])
        imu_client.fig.canvas.draw()
        imu_client.fig.canvas.flush_events()
        print('update plot: ', time.time() - start)
