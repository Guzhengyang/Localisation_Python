# -*- coding: utf-8 -*-
"""
Created on Thu Nov 03 11:26:57 2016

@author: zgu4
"""
from __future__ import division

import itertools
import os
import re

import h2o
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from h2o.estimators import H2ORandomForestEstimator, H2ODeepLearningEstimator
from matplotlib import style

style.use("ggplot")
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle

from mpl_toolkits.mplot3d import Axes3D
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

"""
constants used in the utils
@:param FREQ_BLE: central frequency of bluetooth
@:param C: speed of light
@:param P0: transmitted power
@:param N_WIDTH: width of map for the square method
@:param N_HIGHT: height of map for the square method
@:param FREQ_LOG: frequency for the log file
"""
FREQ_BLE = 2.45e9
C = 3e8
P0 = -22
N_WIDTH = 11
N_HIGHT = 10
FREQ_LOG = 10


def walk_files(directory):
    """
    get file path for all the files in a directory
    :param directory: name of a directory to explore
    :return: list containing all the file paths
    """
    files_path = []
    for (root, dirs, files) in os.walk(directory):
        if files:
            for i in range(len(files)):
                files_path.append(root + "\\" + files[i])
    return files_path


def read_file_rssi(file, n=8, verbose=True, correct_index=False):
    """
    read the rssi values of a csv file
    :param file: file path of a csv file
    :param n: number of beacons to read
    :param verbose: whether to show reading debug info
    :param correct_index: whether to correct index to corresponding time
    :return: dataframe containing rssi values
    """
    log = pd.read_csv(file, delimiter=';', skiprows=2)
    if n == 8:
        cols = ["RSSI LEFT_ORIGIN", "RSSI MIDDLE_ORIGIN", "RSSI RIGHT_ORIGIN", "RSSI TRUNK_ORIGIN",
                "RSSI FRONTLEFT_ORIGIN", "RSSI FRONTRIGHT_ORIGIN",
                "RSSI REARLEFT_ORIGIN", "RSSI REARRIGHT_ORIGIN"]
    elif n == 6:
        cols = ["RSSI FRONTLEFT_ORIGIN", "RSSI FRONTRIGHT_ORIGIN",
                "RSSI REARLEFT_ORIGIN", "RSSI REARRIGHT_ORIGIN",
                "RSSI MIDDLE_ORIGIN", "RSSI TRUNK_ORIGIN"]
    elif n == 4:
        cols = ["RSSI LEFT_ORIGIN", "RSSI MIDDLE_ORIGIN", "RSSI RIGHT_ORIGIN", "RSSI TRUNK_ORIGIN"]
    elif n == 3:
        cols = ["RSSI LEFT_ORIGIN", "RSSI MIDDLE_ORIGIN", "RSSI RIGHT_ORIGIN"]
    elif n == 2:
        cols = ["RSSI MIDDLE_ORIGIN", "RSSI TRUNK_ORIGIN"]
    elif n == 1:
        cols = ["RSSI MIDDLE_ORIGIN"]
    if verbose:
        print("Read RSSI of " + file)
    if correct_index:
        log.index = np.arange(log.shape[0]) / FREQ_LOG
    return log[cols]


def read_file_rssi_37(file, verbose=True, correct_index=False):
    """
    read the rssi values in channel 37
    :param file: file path of a csv file
    :param verbose: whether to show reading info
    :param correct_index: whether to correct index to corresponding time
    :return: dataframes containing rssi values in channel 37
    """
    colum_rssi = ["RSSI LEFT_ORIGIN", "RSSI MIDDLE_ORIGIN", "RSSI RIGHT_ORIGIN", "RSSI TRUNK_ORIGIN"]
    colum_channel = 'BLE CHANNEL MIDDLE'
    rssi = read_file_rssi(file)[colum_rssi]
    channel = read_file_channel(file)[colum_channel]
    index = np.array(channel) == 'BLE_CHANNEL_37'
    rssi_subset = rssi[index]
    if verbose:
        print('Read RSSI in Channel 37 of' + file)
    if correct_index:
        rssi_subset.index = np.arange(rssi_subset.shape[0]) / FREQ_LOG
    return rssi_subset


def read_counter(file):
    """
    read the counter of a csv file
    :param file: file path of a csv file
    :return: dataframe containing counter values
    """
    col = ['COUNTER FLAG']
    return pd.read_csv(file, delimiter=';', skiprows=2, usecols=col)


def read_file_orientation(file, n=3, verbose=True, correct_index=False):
    """
    read the orientation of the smartphone in a csv file
    :param file: file path of a csv file
    :param n: number of orientations to read
    :param verbose: whether to show reading debug info
    :param correct_index: whether to correct index to corresponding time
    :return: dataframe containing orientation values
    """
    log = pd.read_csv(file, delimiter=';', skiprows=2)
    if n == 3:
        cols = ["ORIENTATION_X", "ORIENTAION_Y", "ORIENTATION_Z"]
    elif n == 1:
        cols = ["ORIENTATION_X"]
    if verbose:
        print("Read ORIENTATION of " + file)
    if correct_index:
        log.index = np.arange(log.shape[0]) / FREQ_LOG
    return log[cols]


def read_file_channel(file, n=8, verbose=True, correct_index=False):
    """
    read the channel information of each beacon
    :param file: file path of a csv file
    :param n: number of beacons to read
    :param verbose: whether to show reading debug info
    :param correct_index: whether to correct index to corresponding time
    :return: dataframe containing channel infos
    """
    if n == 8:
        cols = ["BLE CHANNEL FRONTLEFT", "BLE CHANNEL LEFT", "BLE CHANNEL REARLEFT", "BLE CHANNEL TRUNK",
                "BLE CHANNEL REARRIGHT", "BLE CHANNEL RIGHT", "BLE CHANNEL FRONTRIGHT", "BLE CHANNEL MIDDLE"]
    elif n == 4:
        cols = ["BLE CHANNEL LEFT", "BLE CHANNEL RIGHT", "BLE CHANNEL MIDDLE", "BLE CHANNEL TRUNK"]
    elif n == 3:
        cols = ["BLE CHANNEL LEFT", "BLE CHANNEL RIGHT", "BLE CHANNEL MIDDLE"]
    elif n == 1:
        cols = ["BLE CHANNEL MIDDLE"]
    log = pd.read_csv(file, delimiter=';', skiprows=2, usecols=cols)
    if verbose:
        print("Read Channel of " + file)
    if correct_index:
        log.index = np.arange(log.shape[0]) / FREQ_LOG
    return log


def resample_df(df, n=200):
    """
    resample a dataframe to have the same size
    :param df: dataframe to be resampled
    :param n: number of rows to resample
    :return: dataframe containing n rows
    """
    rows = np.random.choice(df.index, n)
    rssi_resampled = pd.DataFrame(df.ix[rows])
    return rssi_resampled


def radian2degree(radian):
    """
    convert radian to degree
    :param radian: orientation in radian
    :return: orientation in degree
    """
    return radian / np.pi * 180


def degree2radian(degree):
    """
    convert degree to radian
    :param degree: orientation in degree
    :return: orientation in radian
    """
    return degree * np.pi / 180


def construct_set(dir, pattern, pattern_valid, filter=0):
    """
    construct train, valid set
    :param dir: name of a directory containing all the log files
    :param pattern: regular expression for data set
    :param pattern_valid: regular expression for valid set
    :param filter: use different filter
                    0:Raw
                    1:Unilateral limiter
                    2:bilateral limiter
                    3:TA n=5
                    4:TA n=10
                    5:LSA n=5
                    6:LSA n=10
    :return:  void
    """
    X = pd.DataFrame()
    y = []
    X_train = pd.DataFrame()
    y_train = []
    X_valid = pd.DataFrame()
    y_valid = []
    N_beacons = 8
    N_samples = 500
    files = walk_files(dir)
    for file in files:
        match_obj = re.search(pattern, file)
        match_obj_valid = re.search(pattern_valid, file)
        if match_obj:
            rssi_raw = read_file_rssi(file, N_beacons, verbose=True)
            if filter == 0:
                rssi_corrected = rssi_raw
            elif filter == 1:
                rssi_corrected = rssi_raw.apply(correct_rssi_unilateral, args=(1,))
            elif filter == 2:
                rssi_corrected = rssi_raw.apply(correct_rssi_bilateral, args=(1,))
            elif filter == 3:
                rssi_corrected = rssi_raw.apply(filter_ta, args=(5,))
            elif filter == 4:
                rssi_corrected = rssi_raw.apply(filter_ta, args=(10,))
            elif filter == 5:
                rssi_corrected = rssi_raw.apply(filter_lsa, args=(5,))
            elif filter == 6:
                rssi_corrected = rssi_raw.apply(filter_lsa, args=(10,))
            else:
                rssi_corrected = rssi_raw.apply(correct_rssi_unilateral, args=(1,))
            X_temp = resample_df(rssi_corrected, N_samples)
            X = X.append(X_temp, ignore_index=True)
            y_temp = np.repeat(match_obj.groups()[0], X_temp.shape[0])
            y.extend(y_temp)
            if match_obj_valid:
                X_valid = X_valid.append(X_temp, ignore_index=True)
                y_valid.extend(y_temp)
            else:
                X_train = X_train.append(X_temp, ignore_index=True)
                y_train.extend(y_temp)
    X, y = shuffle(X, y, random_state=0)
    X_train, y_train = shuffle(X_train, y_train, random_state=0)
    X_valid, y_valid = shuffle(X_valid, y_valid, random_state=0)
    if not os.path.exists('data'):
        os.makedirs('data')
    joblib.dump(X, "data/X")
    joblib.dump(y, "data/y")
    joblib.dump(X_valid, "data/X_valid")
    joblib.dump(y_valid, "data/y_valid")
    joblib.dump(X_train, "data/X_train")
    joblib.dump(y_train, "data/y_train")


def construct_set_same_label(dir, label='internal', N_samples=2000):
    """
    construct data set for the same label in a directory
    :param dir: name of a directory which contains all log files with same label
    :param label: name of label: internal/access/lock...
    :param N_samples: number of samples to be used
    :return: X, y
    """
    X = pd.DataFrame()
    N_beacons = 8
    files = walk_files(dir)
    for file in files:
        rssi_raw = read_file_rssi(file, N_beacons, verbose=True)
        rssi_corrected = rssi_raw.apply(correct_rssi_unilateral, args=(1,))
        X = X.append(rssi_corrected, ignore_index=True)
    X = resample_df(X, N_samples)
    y = pd.Series(np.repeat(label, X.shape[0]))
    y.name = 'class'
    return X, y


def changement_channel(file, type, beacon):
    """
    calculate the variation of rssi value in each change of channel
    :param file: file path of a csv file
    :param type: type of change of channel, 0:37->38, 1:38->39, 2:39->37
    :param beacon: name of beacon
    :return: list containing the changement for certain type and cerain beacon
    """
    column_channel = "BLE CHANNEL " + beacon
    column_rssi = "RSSI " + beacon + "_ORIGIN"
    channel = read_file_channel(file)[column_channel]
    rssi = read_file_rssi(file)[column_rssi]
    changement = list()
    if type == 0:  # 37->38
        for index in range(len(channel) - 1):
            if (channel[index] == "BLE_CHANNEL_37") & \
                    (channel[index + 1] == "BLE_CHANNEL_38"):
                changement.append(rssi[index + 1] - rssi[index])
    elif type == 1:  # 38->39
        for index in range(len(channel) - 1):
            if (channel[index] == "BLE_CHANNEL_38") & \
                    (channel[index + 1] == "BLE_CHANNEL_39"):
                changement.append(rssi[index + 1] - rssi[index])
    else:  # 39->37
        for index in range(len(channel) - 1):
            if (channel[index] == "BLE_CHANNEL_39") & \
                    (channel[index + 1] == "BLE_CHANNEL_37"):
                changement.append(rssi[index + 1] - rssi[index])
    return changement


def correct_channel(rssi, channel):
    """
    compensate the rssi by the variation caused by changement of channel
    :param rssi: dataframe containing rssi values
    :param channel: dataframe containg channel infos
    :return: dataframe containing corrected rssi values
    """
    rssi_corrected = rssi.copy()
    n, p = rssi.shape[0], rssi.shape[1]
    for index_column in range(p):
        for index_row in range(n - 1):
            if (channel.iloc[index_row, index_column] == "BLE_CHANNEL_37") & \
                    (channel.iloc[index_row + 1, index_column] == "BLE_CHANNEL_38"):
                offset38 = rssi_corrected.iloc[index_row, index_column] - rssi_corrected.iloc[
                    index_row + 1, index_column]
                for i in range(index_row + 1, n):
                    if channel.iloc[i, index_column] == "BLE_CHANNEL_38":
                        rssi_corrected.iloc[i, index_column] += offset38
                    else:
                        break
            if (channel.iloc[index_row, index_column] == "BLE_CHANNEL_38") & \
                    (channel.iloc[index_row + 1, index_column] == "BLE_CHANNEL_39"):
                offset39 = rssi_corrected.iloc[index_row, index_column] - rssi_corrected.iloc[
                    index_row + 1, index_column]
                for i in range(index_row + 1, n):
                    if channel.iloc[i, index_column] == "BLE_CHANNEL_39":
                        rssi_corrected.iloc[i, index_column] += offset39
                    else:
                        break
    return rssi_corrected


def rssi2distance(rssi):
    """
    convert rssi to distance using propagation function
    :param rssi: dataframe containing rssi values
    :return: dataframe containing distance values
    """
    return C / FREQ_BLE / 4 / np.pi * np.power(10, -(rssi - P0) * 1.0 / 20)


def distance2rssi(distance):
    """
    convert distance to rssi using propagation function
    :param distance: dataframe containing distance values
    :return: dataframe containg rssi values
    """
    return P0 - 20 * np.log10(4 * np.pi * FREQ_BLE * distance / C)


def distance2rssi_int(distance):
    """
    convert the distance to rssi and then quantify the result to int
    :param distance: dataframe containing distance values
    :return: dataframe containing rssi values in int
    """
    return np.around(distance2rssi(distance))


def quantify_distance(distance, step=0.1):
    """
    quantify the distance with a certain step
    :param distance: dataframe containing distance values
    :param step: step for the quantification
    :return: dataframe containing quantified distance
    """
    return np.around(distance / step) * step


def correct_rssi_unilateral(rssi, threshold_drop=1, n_window=50):
    """
    correct rssi values using unilateral limit of variation between two successive time
    :param rssi: dataframe containing rssi values
    :param threshold_drop: limit of variation for decreasing signal power between two successive time
    :param n_window: window size for applying this correction
    :return: datafame containing corrected rssi values
    """
    rssi_corrected = list(rssi)
    for i in range(len(rssi)):
        if i % n_window == 0:
            rssi_corrected[i] = rssi[i]
        else:
            if rssi[i] < rssi_corrected[i - 1]:
                rssi_corrected[i] = rssi_corrected[i - 1] - min(threshold_drop, rssi_corrected[i - 1] - rssi[i])
    return rssi_corrected


def correct_rssi_bilateral(rssi, threshold_drop=1, threshold_augment=5, n_window=50):
    """
    correct rssi values using bilateral limit of variation between two successive time
    :param rssi: dataframe containing rssi values
    :param threshold_drop: limit of variation for decreasing signal power between two successive time
    :param threshold_augment: limit of variation for increasing signal power between two successive time
    :param n_window: window size for applying this correction
    :return: dataframe containing corrected rssi values
    """
    rssi_corrected = list(rssi)
    for i in range(len(rssi)):
        if i % n_window == 0:
            rssi_corrected[i] = rssi[i]
        else:
            if rssi[i] < rssi_corrected[i - 1]:
                rssi_corrected[i] = rssi_corrected[i - 1] - min(threshold_drop, rssi_corrected[i - 1] - rssi[i])
            elif rssi[i] > rssi_corrected[i - 1]:
                rssi_corrected[i] = rssi_corrected[i - 1] + min(threshold_augment, rssi[i] - rssi_corrected[i - 1])
    return rssi_corrected


def correct_distance_unilateral(distance, threshold_drop=0.1, n_window=50):
    """
    correct distance values using unilateral limit of variation between two successive time
    :param distance: dataframe containing distance values
    :param threshold_drop: limit of variation for keeping away from the car between two successive time
    :param n_window: window size for applying this correction
    :return: datafame containing corrected distance values
    """
    distance_corrected = list(distance)
    for i in range(len(distance)):
        if i % n_window == 0:
            distance_corrected[i] = distance[i]
        else:
            if distance[i] > distance_corrected[i - 1]:
                distance_corrected[i] = distance_corrected[i - 1] + min(threshold_drop,
                                                                        distance[i] - distance_corrected[i - 1])
    return distance_corrected


def correct_distance_bilateral(distance, threshold_far=0.15, threshold_augment=1, n_window=50):
    """
    correct distance values using bilateral limit of variation between wo successive time
    :param distance: dataframe containing distance values
    :param threshold_far: limit of variation for keeping away from the car between tow successive time
    :param threshold_augment: limit of variation for approaching the car between two successive time
    :param n_window: window size for applying this correction
    :return: dataframe containing corrected distance
    """
    distance_corrected = list(distance)
    for i in range(len(distance)):
        if i % n_window == 0:
            distance_corrected[i] = distance[i]
        else:
            if distance[i] < distance_corrected[i - 1]:
                distance_corrected[i] = distance_corrected[i - 1] - min(threshold_augment,
                                                                        distance_corrected[i - 1] - distance[i])
            elif distance[i] > distance_corrected[i - 1]:
                distance_corrected[i] = distance_corrected[i - 1] + min(threshold_far,
                                                                        distance[i] - distance_corrected[i - 1])
    return distance_corrected


def average(df, n_average=3):
    """
    apply moving average filter to rssi
    :param df: dataframe containing rssi values
    :param n_average: window size for applying moving average filter
    :return: dataframe containing averaging values in a window
    """
    df_rooling = df.rolling(window=n_average, center=False).mean()
    index = range(n_average, df_rooling.shape[0], n_average)
    return df_rooling.iloc[index, :]


def plot_file(file, n=4):
    """
    plot the rssi of a log file
    :param file: file name
    :param n: number of beacon to plot
    :return: figure showing rssi varaition over time
    """
    rssi = read_file_rssi(file, n)
    rssi.plot(kind="line")
    plt.title(file)


def plot_directory(dir, n=4):
    """
    plot the rssi of a directory
    :param dir: directory name
    :param n: number of beacon to plot
    :return: figures showing rssi variation over time
    """
    files = walk_files(dir)
    for i in range(len(files)):
        file = files[i]
        rssi = read_file_rssi(file, n)
        rssi.plot(kind="line")
        plt.title(file)


def plot_dataframe(df, title=""):
    """
    plot rssi figure in a dataframe
    :param df: dataframe containing rssi values
    :param title: title of figure
    :return: show a figure
    """
    plt.plot(df['RSSI LEFT_ORIGIN'], 'b', label='LEFT')
    plt.plot(df['RSSI RIGHT_ORIGIN'], 'y', label='RIGHT')
    plt.plot(df['RSSI MIDDLE_ORIGIN'], 'r', label='MIDDLE')
    plt.plot(df['RSSI TRUNK_ORIGIN'], 'g', label='TRUNK')
    plt.title(title)
    plt.legend()
    plt.xlabel("Time: 100ms")


def plot_correction_RSSI(file):
    """
    plot a figure containing raw rssi, unilaterally corrected rssi,
    raw distance, unilateraly corrected distance
    :param file: file path for a csv file
    :return: show a figure
    """
    plt.figure()
    rssi_raw = read_file_rssi(file, 4)
    rssi_corrected = rssi_raw.apply(correct_distance_unilateral, args=(1,))
    distance_raw = rssi2distance(rssi_raw)
    distance_corrected = distance_raw.apply(correct_distance_unilateral, args=(0.1,))
    # plot rssi
    plt.subplot(4, 1, 1)
    plot_dataframe(rssi_raw, "Raw RSSI: " + file)
    plt.ylabel("RSSI: dBm")
    # plot unilaterally corrected rssi
    plt.subplot(4, 1, 2)
    plot_dataframe(rssi_corrected, "Corrected RSSI: " + file)
    plt.ylabel("RSSI: dBm")
    # plot distance
    plt.subplot(4, 1, 3)
    plot_dataframe(distance_raw, "Raw Distance: " + file)
    plt.ylabel("Distance: m")
    # plot unilaterally corrected distance
    plt.subplot(4, 1, 4)
    plot_dataframe(distance_corrected, "Corrected Distance: " + file)
    plt.ylabel("Distance: m")


def plot_distribution(dir_base, dir_list, type=1):
    """
    plot rssi or distance distribution in each zone
    :param dir_base: path of the base directory
    :param dir_list: list of zones to plot
    :param type: type of distribution, 1:rssi distribution, 2:distance distribution
    :return: show distribution figure
    """
    for dir in dir_list:
        X = pd.DataFrame()
        files = walk_files(dir_base + dir)
        if type == 1:
            for file in files:
                rssi_raw = read_file_rssi(file, 4)
                rssi_corrected = rssi_raw.apply(correct_rssi_unilateral, args=(1,))
                X_temp = pd.DataFrame(rssi_corrected)
                X = X.append(X_temp, ignore_index=True)
            ax = X.plot(kind="density")
            ax.set_xlim([-120, -40])
            plt.xticks(np.linspace(start=-120, stop=-40, num=21))
            plt.title("zone " + dir)
        elif type == 2:
            for file in files:
                rssi_raw = read_file_rssi(file, 4)
                distance = rssi2distance(rssi_raw)
                distance_corrected = distance.apply(correct_distance_unilateral, args=(0.1,))
                X_temp = pd.DataFrame(distance_corrected)
                X = X.append(X_temp, ignore_index=True)
            ax = X.plot(kind="density")
            ax.set_xlim([0, 6])
            plt.xticks(np.linspace(start=0, stop=6, num=31))
            plt.title("zone " + dir)


def load_all():
    """
    load X, y data from the disk
    :return: void
    """
    try:
        X = joblib.load("data/X")
        y = joblib.load("data/y")
    except:
        print("X, y don't exist")
    return X, y


def load_train_valid():
    """
    load train set, valid set from the disk
    :return: void
    """
    try:
        X_train = joblib.load("data/X_train")
        X_valid = joblib.load("data/X_valid")
        y_train = joblib.load("data/y_train")
        y_valid = joblib.load("data/y_valid")
    except:
        print("X_train, X_valid, y_train, y_valid don't exist")
    return X_train, X_valid, y_train, y_valid


def save_to_csv():
    """
    save train set, valid set to the disk
    :return: void
    """
    try:
        # save all the data set
        X = joblib.load("data/X")
        y = joblib.load("data/y")
        data = pd.DataFrame(X)
        data["class"] = y
        data.to_csv("data/data.csv", index=False, sep=",")
        # save train set
        X_train = joblib.load("data/X_train")
        y_train = joblib.load("data/y_train")
        data_train = pd.DataFrame(X_train)
        data_train["class"] = y_train
        data_train.to_csv("data/data_train.csv", index=False, sep=",")
        # save valid set
        X_valid = joblib.load("data/X_valid")
        y_valid = joblib.load("data/y_valid")
        data_valid = pd.DataFrame(X_valid)
        data_valid["class"] = y_valid
        data_valid.to_csv("data/data_valid.csv", index=False, sep=",")
    except:
        print("Save error ")


def vote(y_pred, classes, n_vote=3):
    """
    vote in the successive predictions
    :param y_pred: list of predictions
    :param classes: list containing the names of all the classes
    :param n_vote: number of successive predictions to be voted
    :return: list containing voted result
    """
    y_voted = list(y_pred)
    for i in range(len(y_pred)):
        dict = {}
        for i_class in range(len(classes)):
            dict[classes[i_class]] = 0
        if i < n_vote:
            for item in y_pred[0:i + 1]:
                dict[item] += 1
        else:
            for item in y_pred[i + 1 - n_vote:i + 1]:
                dict[item] += 1
        y_voted[i] = max(dict, key=dict.get)
    return y_voted


def vote_prob(y_pred_prob, classes, n_vote=3):
    """
    vote in the sum of the probabilities in the successive predictions
    :param y_pred_prob: list of probability vectors
    :param classes: list containing the names of all the classes
    :param n_vote: number of successive predictions to be voted
    :return: list containing voted result
    """
    y_prob_voted = list(y_pred_prob)
    for i in range(len(y_pred_prob)):
        if i < n_vote:
            prob = pd.DataFrame(y_pred_prob[0:i + 1]).apply(sum, axis=0)
        else:
            prob = pd.DataFrame(y_pred_prob[i + 1 - n_vote:i + 1]).apply(sum, axis=0)
        y_prob_voted[i] = classes[np.argmax(prob)]
    return y_prob_voted


def save_probability(y_pred_prob, y_pred, y_valid, file, classes):
    """
    save probability analysis file
    :param y_pred_prob: list of probability vector
    :param y_pred: list of predictions
    :param y_valid: list of reference
    :param file: name of the saved file
    :param classes: list containing the names of all the classes
    :return: void
    """
    result = pd.DataFrame(y_pred_prob)
    result.columns = classes
    result["Max Prob"] = result.apply(max, axis=1)
    result["Prediction"] = y_pred
    result["Reference"] = y_valid
    result_true = result[y_pred == y_valid]
    result_false = result[y_pred != y_valid]
    result.to_csv("data/" + file + ".csv", index=False, float_format='%.2f')
    result_true.to_csv("data/" + file + "_true.csv", index=False, float_format='%.2f')
    result_false.to_csv("data/" + file + "_false.csv", index=False, float_format='%.2f')


def plot_max_probablity_distribution(file, title=''):
    """
    plot the distribution of the max probability for correctly predicted samples and incorrectly predicted samples
    :param file: csv file to plot
    :return: void
    """
    file_true = "data/" + file + "_true.csv"
    file_false = "data/" + file + "_false.csv"
    df_true = pd.read_csv(file_true, usecols=["Max Prob"])
    df_true.columns = ["Correctly Labeled"]
    df_false = pd.read_csv(file_false, usecols=["Max Prob"])
    df_false.columns = ["Incorrectly Labeled"]
    ax = df_true.plot(kind="density", color="b")
    df_false.plot(kind="density", ax=ax, color="r")
    plt.xlabel('Max Probalibity')
    plt.title(title)


def plot_PCA(X, y, title="", show_components=False):
    """
    plot the data after PCA
    :param X: dataframe of size n_sample * n_features (n_features >= 3)
    :param y: list containing the real class
    :param title: title for the figure
    :param show_components: whether to show 3 components vectors
    :return: void
    """
    pca = PCA(n_components=X.shape[1])
    X_pca = pca.fit_transform(X)
    index_lock = np.array([item == 'lock' for item in y])
    index_access = np.array([item in ['access', 'left', 'right', 'front', 'back'] for item in y])
    index_internal = np.array([item in ['internal', 'start', 'trunk'] for item in y])
    ax = Axes3D(plt.figure())
    ax.scatter(X_pca[index_lock, 0], X_pca[index_lock, 1], X_pca[index_lock, 2],
               c='red', marker='s', s=5, label='lock')
    ax.scatter(X_pca[index_access, 0], X_pca[index_access, 1], X_pca[index_access, 2],
               c='green', marker='o', s=5, label='access')
    ax.scatter(X_pca[index_internal, 0], X_pca[index_internal, 1], X_pca[index_internal, 2],
               c='blue', marker='h', s=5, label='internal')
    ax.set_xlabel('1st eigenvector')
    ax.set_ylabel('2nd eigenvector')
    ax.set_zlabel('3rd eigenvector')
    ax.legend()
    plt.title(title)
    if show_components:
        print("First Principal Axes: ")
        print(", ".join(["{:.3f}".format(element) for element in pca.components_[0]]))
        print("Second Principal Axes: ")
        print(", ".join(["{:.3f}".format(element) for element in pca.components_[1]]))
        print("Third Principal Axes: ")
        print(", ".join(["{:.3f}".format(element) for element in pca.components_[2]]))


def plot_TSNE(X, y, title='', n_subset=4000):
    """
    plot dimension reduced data after t-SNE (t-distributed Stochastic Neighbor Embedding)
    :param X: dataframe of size n_sample * n_features (n_features >= 3)
    :param y: list containing the real class
    :param title: title for the figure
    :param n_subset: number of subset data due to memory constraint
    :return: void
    """
    tsne = TSNE(n_components=2, init='pca', verbose=2)
    if n_subset < X.shape[0]:
        index_sub = np.random.choice(X.shape[0], 4000, replace=False)
    else:
        index_sub = np.random.choice(X.shape[0], X.shape[0], replace=False)
    X_tsne = tsne.fit_transform(X.iloc[index_sub])
    y_tsne = np.array(y)[index_sub]
    index_lock = np.array([item == 'lock' for item in y_tsne])
    index_access = np.array([item in ['access', 'left', 'right', 'front', 'back'] for item in y_tsne])
    index_internal = np.array([item in ['internal', 'start', 'trunk'] for item in y_tsne])
    plt.figure()
    plt.scatter(X_tsne[index_lock, 0], X_tsne[index_lock, 1], marker='s', s=10, c='red', label='lock')
    plt.scatter(X_tsne[index_access, 0], X_tsne[index_access, 1], marker='o', s=10, c='green', label='access')
    plt.scatter(X_tsne[index_internal, 0], X_tsne[index_internal, 1], marker='h', s=10, c='blue', label='internal')
    plt.legend()
    plt.title(title)


def plot_2D_separation(X, y, title=""):
    """
    plot data in a 2D figure
    :param X: dataframe of size n_samples * 2
    :param y: list containing the real class
    :param title: title of the figure
    :return: void
    """
    plt.figure()
    color_dict = {'inside': "blue", 'outside': 'red',
                  'near': 'blue', 'far': 'red',
                  'left': 'green', 'right': 'green', 'front': 'green', 'back': 'green',
                  'start': 'blue', 'trunk': 'pink',
                  'lock': 'red',
                  'frontleft': 'blue', 'frontright': 'green', 'backleft': 'red', 'backright': 'yellow'}
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=[color_dict[i] for i in y])
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.title("2D separation: " + title)


def show_result(y_valid, y_pred, title, labels):
    """
    show the classification report and the confusion matrix
    :param y_valid: list of real class
    :param y_pred: list of prediction
    :param title: title of this result
    :param labels: labels of the classes
    :return:
    """
    print(title + '\n')
    print('Confusion Matrix:\n', confusion_matrix(y_valid, y_pred, labels=labels), '\n')
    print('Report:\n' + classification_report(y_valid, y_pred, labels=labels) + '\n')


def plot_map():
    """
    show a figure of map reference
    :param color: whether to color the different zones
    :return: return a map figure
    """
    index_car = [48, 49, 50, 51, 59, 60, 61, 62]
    index_1m = [36, 37, 38, 39, 40, 41, 52, 63,
                74, 73, 72, 71, 70, 69, 58, 47, 36]
    index_2m = [25, 26, 27, 28, 29, 30, 42,
                53, 64, 75, 85, 84, 83, 82, 81,
                80, 68, 57, 46, 35]
    plt.figure()
    for index in range(1, N_WIDTH * N_HIGHT + 1):
        x, y = square2XY_corner(index)
        window_size = 0.95
        delta = (1 - 0.95) / 2
        if index in index_car:
            rectangle = plt.Rectangle((x + delta, y + delta), window_size, window_size, fc='b')
        elif index in index_1m:
            rectangle = plt.Rectangle((x + delta, y + delta), window_size, window_size, fc='g')
        elif index in index_2m:
            rectangle = plt.Rectangle((x + delta, y + delta), window_size, window_size, fc='g')
        else:
            rectangle = plt.Rectangle((x + delta, y + delta), window_size, window_size, fc='r')
        plt.gca().add_patch(rectangle)
    plt.xlim(0, N_WIDTH)
    plt.ylim(0, N_HIGHT)
    plt.xticks(np.linspace(start=0, stop=N_WIDTH, num=N_WIDTH + 1))
    plt.yticks(np.linspace(start=0, stop=N_HIGHT, num=N_HIGHT + 1))
    plt.xlabel('X: m')
    plt.ylabel('Y: m')


def plot_square(title=''):
    """
    show a figure of map reference with index for each square
    :return: return a map figure
    """
    plot_map()
    plt.title(title)
    for index in range(1, N_WIDTH * N_HIGHT + 1):
        x, y = square2XY_center(index)
        plt.text(x, y, str(index), color="k")


def plot_point(title=''):
    """
    show a figure of mpa reference with index for each point
    :return: return a map figure
    """
    plot_map()
    plt.title(title)
    for index in range(1, (2 * N_WIDTH + 1) * (2 * N_HIGHT + 1) + 1):
        x, y = point2XY(index)
        plt.text(x - 0.05, y - 0.15, str(index), color="k")


def square2zone(index):
    """
    map the square index to zone
    :param index: index of square data
    :return: name of zone
    """
    if index in [25, 26, 27, 28, 29, 30, 37, 38, 39, 40]:
        return "left"
    elif index in [70, 71, 72, 73, 80, 81, 82, 83, 84, 85]:
        return "right"
    elif index in [35, 36, 46, 47, 57, 58, 68, 69]:
        return "front"
    elif index in [41, 42, 52, 53, 63, 64, 74, 75]:
        return "back"
    else:
        return "lock"


def point2zone_thatcham(index):
    """
    map the point index to zone for the thatcham purpose
    :param index: index of point data
    :return: name of the zone
    """
    if (index in range(121, 132)) | \
            (index in range(143, 156)) | \
            (index in range(167, 178)) | \
            (index in range(191, 200)):
        return 'left'
    elif (index in range(305, 316)) | \
            (index in range(327, 340)) | \
            (index in range(351, 362)) | \
            (index in range(283, 292)):
        return 'right'
    elif (index in range(165, 326, 23)) | \
            (index in range(143, 350, 23)) | \
            (index in range(167, 328, 23)) | \
            (index in [214, 237, 260]):
        return 'front'
    elif (index in range(177, 338, 23)) | \
            (index in range(155, 362, 23)) | \
            (index in range(179, 340, 23)) | \
            (index in [222, 245, 268]):
        return 'back'
    else:
        return 'lock'


def point2zone_normal(index):
    """
    map the point index to zone for the normal usage purpose
    :param index: index of the point data
    :return: name of the zone
    """
    if (index in range(97, 110)) | \
            (index in range(120, 133)) | \
            (index in range(143, 156)) | \
            (index in range(167, 178)) | \
            (index in range(191, 200)):
        return 'left'
    elif (index in range(305, 316)) | \
            (index in range(327, 340)) | \
            (index in range(350, 363)) | \
            (index in range(373, 386)) | \
            (index in range(283, 292)):
        return 'right'
    elif (index in range(141, 348, 23)) | \
            (index in range(142, 349, 23)) | \
            (index in range(143, 350, 23)) | \
            (index in range(167, 328, 23)) | \
            (index in [214, 237, 260]):
        return 'front'
    elif (index in range(177, 338, 23)) | \
            (index in range(155, 362, 23)) | \
            (index in range(156, 363, 23)) | \
            (index in range(157, 364, 23)) | \
            (index in [222, 245, 268]):
        return 'back'
    else:
        return 'lock'


def construct_prob_dict_zero(prob_dict):
    """
    construct a new dictionary whose keys are same to prob_dict and whose values are 0
    :param prob_dict: a dictionary containing the probability
    :return: zero dictionary
    """
    return {key: 0 for key in prob_dict.keys()}


def find_max(prob_dict, kernel):
    """
    find the key whose value is the max among the kernel keys
    :param prob_dict: probalibity dictionary
    :param kernel: list of keys to be considered
    :return: key
    """
    max = 0
    index_max = 0
    for i in range(len(kernel)):
        key = kernel[i]
        if prob_dict[key] > max:
            max = prob_dict[key]
            index_max = i
    return kernel[index_max]


def transfer_prob(prob_dict, index):
    """
    construct a new probability dictionary based on the previous result
    :param prob_dict: probability dictionary
    :param index: previous result index
    :return: new probability dictionary
    """
    w0 = 4
    w1 = 3
    w2 = 2
    kernel = []
    prob_new = construct_prob_dict_zero(prob_dict)
    surrounding1 = [index - N_WIDTH - 1, index - N_WIDTH, index - N_WIDTH + 1,
                    index + N_WIDTH - 1, index + N_WIDTH, index - N_WIDTH + 1,
                    index - 1, index + 1]
    surrounding2 = [index - 2 * N_WIDTH - 2, index - 2 * N_WIDTH - 1, index - 2 * N_WIDTH, index - 2 * N_WIDTH + 1,
                    index - 2 * N_WIDTH + 2,
                    index + 2 * N_WIDTH - 2, index + 2 * N_WIDTH - 1, index + 2 * N_WIDTH, index + 2 * N_WIDTH + 1,
                    index + 2 * N_WIDTH + 2,
                    index - N_WIDTH - 2, index - N_WIDTH + 2,
                    index - 2, index + 2,
                    index + N_WIDTH - 2, index + N_WIDTH + 2]
    kernel.append(index)
    prob_new[index] = int(prob_dict[index] * w0 * 100) * 1.0 / 100
    for key in surrounding1:
        if key in prob_dict.keys():
            kernel.append(key)
            prob_new[key] = int(prob_dict[key] * w1 * 100) * 1.0 / 100
    for key in surrounding2:
        if key in prob_dict.keys():
            kernel.append(key)
            prob_new[key] = int(prob_dict[key] * w2 * 100) * 1.0 / 100
    return prob_new, kernel


def list2dict(prob_list, clf):
    """
    convert probability vector to a dictionary
    :param prob_list: probability vector
    :param clf: classifier
    :return: probability dictionary
    """
    prob_dict = {key: int(value * 100) * 1.0 / 100 for key, value in zip(clf.classes_, prob_list)}
    return prob_dict


def predict_trace(clf, X):
    """
    object tracking given a list of samples and a classifier
    :param clf: classifier
    :param X: list of samples of size n_samples*n_features
    :return: list of square index
    """
    res = list()
    # initiation
    y_pred_prob = clf.predict_proba(X[:1])[-1]
    prob_dict = list2dict(y_pred_prob, clf)
    index_max = max(prob_dict, key=lambda k: prob_dict[k])
    res.append(index_max)
    # iteration
    for i in range(1, X.shape[0]):
        y_pred_prob = clf.predict_proba(X[i:i + 1])[-1]
        prob_dict = list2dict(y_pred_prob, clf)
        prob_dict_new, kernel = transfer_prob(prob_dict, index_max)
        index_max = find_max(prob_dict_new, kernel)
        res.append(index_max)
    return np.array(res)


def plot_prob(prob_dict, title=""):
    """
    plot a figure containing the probabilities of each square
    :param prob_dict: probability dictionary
    :param title: title of the figures
    :return: show a figure
    """
    X_car = [3, 7, 7, 3, 3]
    Y_car = [4, 4, 6, 6, 4]
    plt.plot(X_car, Y_car, "b")
    for key, value in prob_dict.items():
        x, y = square2XY_center(key)
        plt.text(x, y, str(value))
    for i in range(1, N_HIGHT * N_WIDTH + 1):
        x, y = square2XY_corner(i)
        plt.text(x, y, str(i), color="green")
    plt.xlim(0, N_WIDTH)
    plt.ylim(0, N_HIGHT)
    plt.xticks(np.linspace(start=0, stop=N_WIDTH, num=N_WIDTH + 1))
    plt.yticks(np.linspace(start=0, stop=N_HIGHT, num=N_HIGHT + 1))
    plt.title(title)


def square2XY_center(index_square):
    """
    convert square index to coordinates of the center of that square
    :param index_square: index of square
    :return: coordinate x and y
    """
    index_square = index_square - 1
    y = np.floor(index_square / N_WIDTH)
    x = index_square - y * N_WIDTH
    x = x + 0.5
    y = y + 0.5
    return x, y


def square2XY_corner(index_square):
    """
    convert square index to coordinates of the corner of that square
    :param index_square: index of square
    :return: coordinate x and y
    """
    index_square = index_square - 1
    y = np.floor(index_square / N_WIDTH)
    x = index_square - y * N_WIDTH
    return x, y


def point2XY(index_point):
    """
    convert point index to coordinate of that point
    :param index_point:
    :return: coordinate x and y
    """
    index_point = index_point - 1
    y = np.floor(index_point / (N_WIDTH * 2 + 1))
    x = index_point - y * (N_WIDTH * 2 + 1)
    return x / 2, y / 2


def str2int(str):
    """
    find the index in a string
    :param str: string in  the form of ("S"+index) or ("P"+index)
    :return: index
    """
    res = np.zeros(len(str))
    for i in range(len(str)):
        item = str[i]
        res[i] = int(item[1:])
    return res


def plot_trace(x_pred, y_pred, index_ref=[], title="", show_number=False):
    """
    plot the trace of an object
    :param x_pred: list of coordinate x
    :param y_pred: list of coordinate y
    :param index_ref: list of real index
    :param title: title of a figure
    :param show_number: whether to show the number in the figure
    :return:
    """
    plot_map()
    plt.title(title)
    if x_pred:
        plt.plot(x_pred, y_pred, 'k')
    if index_ref:
        X_ref, Y_ref = square2XY_center(np.array(index_ref))
        plt.plot(X_ref, Y_ref, "b", linewidth=2)
    if show_number:
        for i in range(N_HIGHT * N_WIDTH):
            index = i + 1
            x, y = square2XY_corner(index)
            plt.text(x, y, str(index), color="k")


def name2index(file):
    """
    find the list of index in a file name
    :param file: file name
    :return: list of index
    """
    pattern = r'.*\\(.*?).csv$'
    match_obj = re.match(pattern, file)
    index_ref = []
    if match_obj:
        res = match_obj.groups()[0]
        index_str = res.split(" ")
        for item in index_str:
            index_ref.append(int(item))
    return index_ref


def error_2D(PxPy, PxPy_pred):
    """
    calculate rmse base on 2D coordinates
    :param PxPy: list of real 2D coordinates
    :param PxPy_pred: list of predicted 2D coordinates
    :return: error
    """
    delta = np.array(PxPy) - np.array(PxPy_pred)
    norm = np.linalg.norm(delta, axis=1)
    return np.mean(norm)


def error_1D(P, P_pred):
    """
    calculate rmse based on 1D coordinates
    :param P: list of real 1D coordinates
    :param P_pred: list of predicted 1D coordinates
    :return: error
    """
    delta = np.array(P) - np.array(P_pred)
    return np.sqrt(np.mean(np.square(delta)))


def correct_PxPy(PxPy, threshold=0.5):
    """
    correct coordinates based on the limit of movement between two successive time
    :param PxPy: list of raw 2D coordinates
    :param threshold: limit of movement between two successive time
    :return: corrected coordinates
    """
    PxPy = np.array(PxPy)
    PxPy_corrected = np.zeros(PxPy.shape)
    PxPy_corrected[0] = PxPy[0]
    for i in range(1, PxPy.shape[0]):
        delta = PxPy[i] - PxPy_corrected[i - 1]
        dist = np.linalg.norm(delta)
        if dist < threshold:
            PxPy_corrected[i] = PxPy[i]
        else:
            PxPy_corrected[i] = PxPy_corrected[i - 1] + threshold / dist * delta
    return PxPy_corrected


def plot_n_PCA(X, y, y_list):
    """
    plot the chosen data after PCA
    :param X: dataframe containing rssi values
    :param y: list of classes
    :param y_list: list of chosen classes
    :return:
    """
    mask = np.repeat(False, len(y))
    for i in range(len(y)):
        if y[i] in y_list:
            mask[i] = True
    X_sub = X[mask]
    y_sub = y[mask]
    print(np.unique(y_sub))
    pca = PCA(n_components=X_sub.shape[1])
    X_reduced = pca.fit_transform(X_sub)
    ax1 = Axes3D(plt.figure())
    ax1.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y_sub, s=50)
    ax1.set_xlabel('1st eigenvector')
    ax1.set_ylabel('2nd eigenvector')
    ax1.set_zlabel('3rd eigenvector')
    plt.title("PCA for point data")


def average_in_degree(file, degree, degree_delta=22.5, n=4):
    """
    calculate the average rssi values in a given degree
    :param file: file path of a csv file
    :param degree: central degree
    :param degree_delta: degree interval: [degree-degree_delta, degree+degree_delta]
    :param n: number of beacons
    :return: dataframe containing the average rssi values
    """
    # read orientation of x
    df_radian = read_file_orientation(file, 1, verbose=False)
    # convert radian to degree
    df_degree = df_radian.apply(radian2degree)
    if degree < 0:
        if degree - degree_delta < -180:
            index1 = (df_degree >= -180) & (df_degree <= degree + degree_delta)
            delta = degree_delta - (degree + 180)
            index2 = (df_degree >= 180 - delta) & (df_degree <= 180)
            index = index1 | index2
            # print "degree = {0} ===> ".format(degree), \
            #     "[{0}, {1}] U [{2}, {3}]".format(-180, degree + degree_delta, 180 - delta, 180)
        else:
            index = (df_degree >= degree - degree_delta) & (df_degree <= degree + degree_delta)
            # print "degree = {0} ===> ".format(degree), \
            #     "[{0}, {1}]".format(degree - degree_delta, degree + degree_delta)
    else:
        if degree + degree_delta > 180:
            index1 = (df_degree >= degree - degree_delta) & (degree <= 180)
            delta = degree_delta - (180 - degree)
            index2 = (df_degree >= -180) & (df_degree <= -180 + delta)
            index = index1 | index2
            # print "degree = {0} ===> ".format(degree), \
            #     "[{0}, {1}] U [{2}, {3}]".format(degree - degree_delta, 180, -180, -180 + delta)
        else:
            index = (df_degree >= degree - degree_delta) & (df_degree <= degree + degree_delta)
            # print "degree = {0} ===> ".format(degree), \
            #     "[{0}, {1}]".format(degree - degree_delta, degree + degree_delta)
    index = np.array(index)
    rssi = read_file_rssi(file, n, verbose=False)
    return rssi[index].apply(np.mean).to_frame().transpose()


def average_in_degree_list(file, degree_list=np.linspace(-180, 180, 8, endpoint=False)):
    """
    calculate the average rssi values in a list of degree
    :param file: file path of a csv file
    :param degree_list: list of degrees
    :return: dataframe containing average rssi for each degree
    """
    df = pd.DataFrame()
    for degree in degree_list:
        df_temp = average_in_degree(file, degree)
        df = df.append(df_temp, ignore_index=True)
    df = df.apply(lambda x: x - x[0])
    df["degree"] = degree_list
    df["radis"] = df["degree"].apply(degree2radian)
    df = df.append(df[:1], ignore_index=True)
    return df


def mapping_radian(radian, radian_ref, loc='back'):
    """
    mapping radian orientation measured by smartphone to standard radian for a diagram
    :param radian: radian orientation measured directly by the smartphone
    :param radian0: fisst orientation measured
    :return: dataframe containing mapped degree
    """
    delta = radian_ref
    if loc == 'left':
        delta = delta + np.pi / 2
    elif loc == 'right':
        delta = delta - np.pi / 2
    elif loc == 'front':
        delta = delta + np.pi

    return np.mod(radian - delta, 2 * np.pi)


def tune(X_train, X_valid, y_train, y_valid, method, param_grid):
    """
    tune parameters
    :param X_train: train data set
    :param X_valid: valid data set
    :param y_train: train label
    :param y_valid: valid label
    :param method: machine learning method, mothod="KNN", "SVM", "RF" or "GBM"
    :param param_grid: tuning parameters for ml method,
                        "KNN": param_grid={'n_neighbors': [10, 20, 30, 40, 50, 100]}
                        "SVM": param_grid = {'C': [0.1, 0.25], 'gamma': [0.1, 0.25, 0.5]}
                        "RF": param_grid = {'max_features': [2, 3], 'n_estimators': [50, 100]}
                        "GBM": param_grid = {'n_estimators': [400], 'max_depth': [1, 2, 3]}
    :return: show tuning result
            for exemple, peps data set, best parameter
            "KNN": n_neighbors=10
            "SVM": C=0.25, gamma=0.5
            "RF": max_feature=2, n_estimators=100
            "GBM": n_estimators=400, max_depth=3

    """
    if method == "KNN":
        clf = GridSearchCV(KNeighborsClassifier(),
                           param_grid=param_grid, n_jobs=-1, verbose=3)
    elif method == "SVM":
        clf = make_pipeline(StandardScaler(),
                            GridSearchCV(
                                SVC(cache_size=3000, probability=True, decision_function_shape='ovr', random_state=0),
                                param_grid=param_grid, n_jobs=-1, verbose=3))
    elif method == "RF":
        clf = GridSearchCV(RandomForestClassifier(random_state=0),
                           param_grid=param_grid, n_jobs=-1, verbose=3)
    elif method == "GBM":
        clf = GridSearchCV(GradientBoostingClassifier(random_state=0),
                           param_grid=param_grid, n_jobs=-1, verbose=3)
    clf.fit(X_train, y_train)
    if method == "SVM":
        classes = clf.steps[1][1].best_estimator_.classes_
    else:
        classes = clf.best_estimator_.classes_
    y_pred = clf.predict(X_valid)
    print("===========================" + method + " ====================================================")
    show_result(y_valid, y_pred, "Result without vote", classes)
    print("===========================" + method + " ====================================================")


def train(X_train, X_valid, y_train, y_valid, method="Logistic", param=None, vote=False, vote_prob=False,
          save_prob=False):
    """
    train a model
    :param X_train: train data set
    :param X_valid: valid data set
    :param y_train: train label
    :param y_valid: valid label
    :param method: machine learning method, mothod="Logistic","LDA","QDA","KNN","SVM","RF","GBM" or "MLP"
    :param param: parameters for each model
                    "KNN": {n_neighbors=10}
                    "SVM": {C=0.25, gamma=0.5}
                    "RF": {max_features=2, n_estimators=100}
                    "GBM": {n_estimators=400, max_depth=3}
                    "MLP": {hidden_layer_sizes=(50, 10)}
    :param vote: whether to show the voted result in the predictions
    :param vote_prob: whether to show voted result in the probabilities
    :param save_prob: whether to save the probability analysis file
    :return: show train result and return predicted labels
    """
    if method == "Logistic":
        clf = LogisticRegression(random_state=0, class_weight='balanced', n_jobs=-1)
    elif method == "LDA":
        clf = LinearDiscriminantAnalysis()
    elif method == "QDA":
        clf = QuadraticDiscriminantAnalysis()
    elif method == "KNN":
        vote_prob = False
        clf = KNeighborsClassifier(n_jobs=-1)
    elif method == "SVM":
        clf = make_pipeline(StandardScaler(),
                            SVC(cache_size=3000, probability=True, decision_function_shape='ovr', random_state=0,
                                class_weight='balanced'))
    elif method == "RF":
        clf = RandomForestClassifier(random_state=0, class_weight='balanced', n_jobs=-1)
    elif method == "GBM":
        clf = GradientBoostingClassifier(random_state=0)
    elif method == "MLP":
        clf = make_pipeline(StandardScaler(),
                            MLPClassifier(tol=1e-6, verbose=True, random_state=0))
    if (method == "SVM") | (method == "MLP"):
        if param:
            clf.steps[1][1].set_params(**param)
        clf.fit(X_train, y_train)
        classes = clf.steps[1][1].classes_
    else:
        if param:
            clf.set_params(**param)
        clf.fit(X_train, y_train)
        classes = clf.classes_
    y_pred = clf.predict(X_valid)
    cm = confusion_matrix(y_valid, y_pred, labels=classes)
    report = precision_recall_fscore_support(y_valid, y_pred, labels=classes)
    df_report = pd.Series({'precision': report[0].mean(),
                           'recall': report[1].mean(),
                           'f1-score': report[2].mean(),
                           'method': method})
    print("===========================" + method + " ====================================================")
    if param:
        print('Parameters:', param)
    show_result(y_valid, y_pred, "Result without vote for Validation", classes)
    if vote:
        y_voted = vote(y_pred, classes, 3)
        show_result(y_valid, y_voted, "Result with vote for Validation", classes)
    if vote_prob:
        y_pred_prob = clf.predict_proba(X_valid)
        y_voted = vote_prob(y_pred_prob, classes, 3)
        show_result(y_valid, y_voted, "Result with vote in the probabilities", classes)
    if save_prob:
        y_pred_prob = clf.predict_proba(X_valid)
        save_probability(y_pred_prob, y_pred, y_valid, method, classes)
    print("===========================" + method + " ====================================================")
    return cm, df_report, classes


def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.grid(False)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def find_beacon_name(beacon):
    """
    find the name of beacon in column name, for exemple, 'RSSI LEFT_ORIGIN'->'LEFT'
    :param beacon: name of column containing beacon name
    :return: beacon name
    """
    pattern_beacon = r' (.*?)_'
    match_obj = re.search(pattern_beacon, beacon)
    return match_obj.groups()[0]


def correct_diagram(rssi, radian):
    """
    correct rssi with the a diagram of the smartphone
    :param rssi: dataframe of size (n_sample * n) containing original rssi values
    :param radian: dataframe of size (n_sample * 1) containing the corresponded orientation of smartphone
    :return: dataframe containing corrected rssi values
    """
    # diagram measured with analyser
    # offsets_H = np.array([0.2, 2.9, 4.6, 7.0, 8.3, 19.2, 15.5, 8.4,
    #                       2.1, 7.1, 2.0, 0.6, 0.5, 0.6, 4.0, 0.4, 0.2])
    # offsets_T = np.array([0.2, 2.9, 4.6, 7.0, 7.8, 8.9, 9.3, 8.4,
    #                       2.1, 5.3, 2.0, 0.6, 0.5, 0.6, 3.9, 0.4, 0.2])

    # diagram measured with circularly polarized antenna
    offsets_H = np.array([-61, -60, -60, -66, -65, -72, -61, -56,
                          -61, -57, -54, -63, -61, -65, -63, -64, -61])
    offsets_H = offsets_H - offsets_H[0]
    offsets_V = np.array([-59, -60, -64, -57, -52, -51, -51, -51,
                          -50, -50, -55, -63, -59, -60, -61, -60, -59])
    offsets_V = offsets_V - offsets_V[0]
    radian_centers = np.linspace(0, 2 * np.pi, 17)
    rssi_corrected = pd.DataFrame(rssi, copy=True)
    for j in range(rssi.shape[1]):
        for i in range(rssi.shape[0]):
            radian_deltas = np.abs(radian_centers - radian.iloc[i, 0])
            index_min = np.argmin(np.abs(radian_deltas))
            rssi_corrected.iloc[i, j] = offsets_H[index_min] + rssi_corrected.iloc[i, j]
    return rssi_corrected


def plot_correction_diagram(rssi, rssi_corrected, degree, beacon, location):
    """
    plot the result with the diagram correction
    :param rssi: dataframe containing the original rssi values
    :param rssi_corrected: dataframe containing the corrected rssi values
    :param beacon: beacon to be considered
    :param location: location where the mesure was taken place
    :return: figure showing the result
    """
    plt.subplot(3, 1, 1)
    rssi[beacon].plot(kind='line', label='Original')
    rssi_corrected[beacon].plot(kind='line', label='Corrected')
    plt.title("BEACON " + find_beacon_name(beacon) +
              ", Smartphone in " + location)
    plt.legend()

    ax1 = plt.subplot(3, 1, 2)
    degree.plot(kind='line', ax=ax1)
    plt.title('Orientation of Smartphone')
    plt.legend()

    plt.subplot(3, 2, 5)
    rssi[beacon].plot(kind='density', label='Original')
    rssi_corrected[beacon].plot(kind='density', label='Corrected')
    plt.legend()

    ax2 = plt.subplot(3, 2, 6)
    df = pd.DataFrame()
    df['Original'] = rssi[beacon]
    df['Corrected'] = rssi_corrected[beacon]
    df.plot(kind='box', ax=ax2)
    plt.legend()


def construct_heatmap_set(dir, pattern):
    """
    construct heatmap data set
    :param dir: directory of a heatmap files
    :param pattern: pattern for each heatmap file
    :return: saved data set
    """
    files = walk_files(dir)
    df = pd.DataFrame()
    for i in range(len(files)):
        file = files[i]
        match_obj = re.search(pattern, file)
        if match_obj:
            n_line = int(match_obj.groups()[0])
            rssi = read_file_rssi(file, 8)
            counter = read_counter(file)
            df_temp = pd.concat([rssi, counter], axis=1)
            index_temp = df_temp.iloc[:, 8] != 0
            df_temp = df_temp[index_temp]
            df_temp.iloc[:, 8] = 23 * (n_line - 1) + df_temp.iloc[:, 8]
            df = pd.concat([df, df_temp], ignore_index=True)
    # df = shuffle(df, random_state=0)
    if not os.path.exists('data'):
        os.makedirs('data')
    df.to_csv('data/df.csv', sep=';', index=False)
    heat = df.groupby(by=['COUNTER FLAG'], as_index=False).mean()
    heat.to_csv('data/heat.csv', sep=';', index=False)


def plot_heatmap(df_heat, beacon, title=''):
    """
    show a heatmap figure for a certain beacon
    :param df_heat: heatmap dataframe
    :param beacon: name of the beacon
    :param title: name of a figure
    :return: figure
    """
    n_X = 23
    n_Y = 21
    xs = np.arange(n_X) / 2.0
    ys = np.arange(n_Y) / 2.0
    xs, ys = np.meshgrid(xs, ys)
    intensity = np.zeros((n_Y, n_X))
    for index in np.arange(df_heat.shape[0]):
        counter = df_heat.loc[index, 'COUNTER FLAG']
        x, y = point2XY(counter)
        index_x = int(x / 0.5)
        index_y = int(y / 0.5)
        intensity[index_y, index_x] = df_heat.loc[index, beacon]
    plt.figure()
    plt.pcolor(xs, ys, intensity, cmap='jet',
               vmin=np.min(np.min(intensity)), vmax=np.max(np.max(intensity[intensity != 0])))
    plt.colorbar()
    index_car = [48, 49, 50, 51, 59, 60, 61, 62]
    for index in range(1, N_WIDTH * N_HIGHT + 1):
        x, y = square2XY_corner(index)
        if index in index_car:
            rectangle = plt.Rectangle((x, y), 1, 1, fc='k')
            plt.gca().add_patch(rectangle)
    plt.xlim(0, N_WIDTH)
    plt.ylim(0, N_HIGHT)
    plt.xticks(np.linspace(start=0, stop=N_WIDTH, num=N_WIDTH + 1))
    plt.yticks(np.linspace(start=0, stop=N_HIGHT, num=N_HIGHT + 1))
    plt.xlabel('X: m')
    plt.ylabel('Y: m')
    plt.title(title)


def plot_mlp_result(mlp, title=''):
    """
    plot mlp convergence result
    :param mlp: h2o mlp model
    :param title: titile for the figure
    :return:
    """
    plt.figure()
    plt.plot(mlp.scoring_history()['epochs'], np.sqrt(mlp.scoring_history()['training_deviance']), label='Train')
    plt.plot(mlp.scoring_history()['epochs'], np.sqrt(mlp.scoring_history()['validation_deviance']), label='Valid')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title(title)
    plt.legend()


def transform_data_label(df_set, thatcham=True, binary=True):
    """
    transform the label of point data to zone according to the strategy (thatcham or binary)
    :param df_set: dataframe containing rssi values and index of each point data
    :param thatcham: whether to use thatcham strategy
    :param binary: whether to use binary classification
    :return: X containing features (rssi values) and y containing zone labels
    """
    X = df_set.iloc[:, 0:8]
    if thatcham:
        y = df_set['COUNTER FLAG'].apply(point2zone_thatcham)
    else:
        y = df_set['COUNTER FLAG'].apply(point2zone_normal)
    y.name = 'class'
    if binary:
        y[y != 'lock'] = 'access'
    return X, y


def train_rf(weight_lock=2, model_id="Test", mtries=2, ntrees=50):
    """
    deploy Random Forest results using h2o package
    :param weight_lock: weight for lock zone to better guarantee the thatcham constriant
    :param model_id: name of jave model
    :param mtries: parameter of Random Forest
    :param ntrees: parameter of Random Forest
    :return: Random Forest using h2o package
    """
    df_train = h2o.upload_file('data/data_train.csv')
    df_valid = h2o.upload_file('data/data_valid.csv')

    df_train['weight'] = 1
    index_lock = df_train['class'] == 'lock'
    df_train[index_lock, 'weight'] = weight_lock
    rf = H2ORandomForestEstimator(mtries=mtries, ntrees=ntrees,
                                  model_id=model_id)
    rf.train(x=df_train.columns[:8],
             y='class',
             weights_column='weight',
             training_frame=df_train,
             validation_frame=df_valid)
    print(rf.model_performance(valid=True))
    return rf


def train_mlp(X_train, X_valid, Px_train, Px_valid, Py_train, Py_valid,
              model_id_px='MLP4Px', model_id_py='MLP4Py'):
    """
    deploy Multi Perceptron results using h2o package
    :param X_train: train set containing features (rssi values)
    :param X_valid: valid/test set containing features (rssi values)
    :param Px_train: train set containing x coordinate
    :param Px_valid: valid/test set containing x coordinate
    :param Py_train: train set containing y coordinate
    :param Py_valid: valid/test set containing y coordinate
    :param model_id_px: name of jave model for x coordinate
    :param model_id_py: name of jave model for y coordinate
    :return:
    """
    data_train = pd.concat([X_train, Px_train, Py_train], axis=1)
    data_valid = pd.concat([X_valid, Px_valid, Py_valid], axis=1)
    data_train.to_csv('data/data_train.csv', index=False)
    data_valid.to_csv('data/data_valid.csv', index=False)
    df_train = h2o.upload_file('data/data_train.csv')
    df_valid = h2o.upload_file('data/data_valid.csv')
    mlp_px = H2ODeepLearningEstimator(hidden=[64, 64], epochs=200,
                                      score_each_iteration=True,
                                      loss="Quadratic",
                                      stopping_rounds=0,
                                      mini_batch_size=64,
                                      input_dropout_ratio=0,
                                      l1=0,
                                      l2=0,
                                      model_id=model_id_px)
    mlp_py = H2ODeepLearningEstimator(hidden=[64, 64], epochs=200,
                                      score_each_iteration=True,
                                      loss="Quadratic",
                                      stopping_rounds=0,
                                      mini_batch_size=64,
                                      input_dropout_ratio=0,
                                      l1=0,
                                      l2=0,
                                      model_id=model_id_py)
    mlp_px.train(x=df_train.columns[:8],
                 y='Px',
                 training_frame=df_train,
                 validation_frame=df_valid)
    mlp_py.train(x=df_train.columns[:8],
                 y='Py',
                 training_frame=df_train,
                 validation_frame=df_valid)
    print(mlp_px.model_performance(valid=True))
    print(mlp_py.model_performance(valid=True))
    return mlp_px, mlp_py


def filter_ta(x, n=10):
    """
    time average filter
    :param x: time sequence signal
    :param n: window size of the filter
    :return: filtered sequence
    """
    x = np.array(x)
    res = list()
    for i in np.arange(len(x)):
        if i <= len(x) - n:
            res.append(np.mean(x[i:i + n]))
        else:
            res.append(np.mean(x[i:]))
    return np.array(res)


def autocorrelation(x):
    """
    calculate the autocorrelation of a signal
    :param x: time sequence signal
    :return: autocorrelation
    """
    x = np.array(x)
    N = len(x)
    r = list()
    for k in np.arange(N):
        temp = 0
        for n in np.arange(k, N):
            temp += x[n] * x[n - k]
        temp /= N - k
        r.append(temp)
    return np.array(r)


def dft(r):
    """
    calculate PSD(Power Spectral Density) using discrete fourier transformation of autocorrelation
    :param r: autocorrelation
    :return: PSD
    """
    r = np.array(r)
    N = len(r)
    s = list()
    for f in np.arange(N):
        temp = np.sum(r * np.exp(-1j * 2 * np.pi * f * np.arange(N) / N))
        s.append(temp)
    return np.mean(s)


def filter_lsa(x, n=10):
    """
    log spectrum average filter
    :param x: time sequence signal
    :param n: window size of the filter
    :return: filtered signal
    """
    x = np.array(x)
    res = list()
    for i in np.arange(len(x)):
        if i <= len(x) - n:
            x_temp = x[i:i + n]
        else:
            x_temp = x[i:]
        r = autocorrelation(x_temp)
        s = dft(r)
        res.append(np.mean(np.log(np.abs(s))))
    return np.array(res)
