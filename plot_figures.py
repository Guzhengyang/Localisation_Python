from __future__ import division

import os
import re

import h2o
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import utils

dir_fig = 'figures'
colums = ['RSSI LEFT_ORIGIN', 'RSSI MIDDLE_ORIGIN', 'RSSI RIGHT_ORIGIN', 'RSSI TRUNK_ORIGIN']


def plot_map(fig_name):
    """
    plot a location map containing squares
    :param fig_name: saved figure name
    :return: figure
    """
    utils.plot_map()
    plt.title(fig_name)
    if not os.path.exists(dir_fig):
        os.makedirs(dir_fig)
    plt.savefig('figures/' + fig_name + '.png')


def plot_point(fig_name):
    """
    plot location map containing all the points
    :param fig_name: saved figure name
    :return: figure
    """
    utils.plot_point()
    plt.title(fig_name)
    if not os.path.exists(dir_fig):
        os.makedirs(dir_fig)
    plt.savefig('figures/' + fig_name + '.png')


def plot_rssi_variation(fig_name):
    """
    plot a figure for RSSI variation in a static point
    :param fig_name: saved figure name
    :return: figure
    """
    dir = "log/heatmap"
    pattern = r'(\d{1,2}).csv$'
    utils.construct_heatmap_set(dir, pattern)
    df = pd.read_csv('data/df.csv', sep=';')
    df_point100 = df[[index == 100 for index in df['COUNTER FLAG']]]
    df_point100.index = np.arange(df_point100.shape[0]) / 10
    df_point100[colums].plot()
    plt.xlabel('Time: s')
    plt.ylabel('RSSI: dBm')
    plt.title(fig_name)
    if not os.path.exists(dir_fig):
        os.makedirs(dir_fig)
    plt.savefig(dir_fig + '/' + fig_name + '.png')


def plot_channel_offset(fig_name):
    """
    plot a figure showing channel offset in a static point
    :param fig_name: saved figure name
    :return:
    """
    file1 = 'log/influence/channel/table_left_150cm.csv'
    df = utils.read_file_rssi(file1, correct_index=True)
    df[colums].plot()
    plt.title(file1)
    plt.xlabel('Time: s')
    plt.ylabel('RSSI: dBm')
    if not os.path.exists(dir_fig):
        os.makedirs(dir_fig)
    plt.savefig(dir_fig + '/' + fig_name + '1.png')
    file2 = 'log/influence/channel/table_left_50cm.csv'
    df = utils.read_file_rssi(file2, correct_index=True)
    df[colums].plot()
    plt.title(file2)
    plt.xlabel('Time: s')
    plt.ylabel('RSSI: dBm')
    plt.savefig('figures/' + fig_name + '2.png')


def plot_path_loss_theoretical(fig_name):
    """
    plot a figure for the theoretical pass loss
    :param fig_name: saved figure name
    :return: figure
    """
    plt.figure()
    d = np.linspace(0.1, 5, 100)
    rssi = utils.distance2rssi(d)
    plt.plot(d, rssi)
    plt.title(fig_name)
    plt.xlabel('Distance: m')
    plt.ylabel('RSSI: dBm')
    if not os.path.exists(dir_fig):
        os.makedirs(dir_fig)
    plt.savefig(dir_fig + '/' + fig_name + '.png')


def plot_path_loss_experimental(fig_name):
    """
    plot a figure for the experimental path loss
    :param fig_name: saved figure name
    :return: figure
    """
    plt.figure()
    N = 26
    dist = 0.1 * np.arange(N) + 0.5
    pattern = r'(\d{1,2}).csv$'
    dir1 = "log/path loss/PIFA"
    files = utils.walk_files(dir1)
    rssi_mean = np.zeros(26)
    for file in files:
        match_obj = re.search(pattern, file)
        rssi = utils.read_file_rssi(file, 4, correct_index=True)
        index = int(match_obj.groups()[0]) - 1
        rssi_mean[index] = np.mean(rssi['RSSI LEFT_ORIGIN'])
    plt.plot(dist[2:], rssi_mean[2:])
    plt.xlabel('Distance: m')
    plt.ylabel('RSSI: dBm')
    plt.title(fig_name)
    if not os.path.exists(dir_fig):
        os.makedirs(dir_fig)
    plt.savefig(dir_fig + '/' + fig_name + '.png')


def plot_diagram_horizontal(fig_name):
    """
    plot a figure for horizontal diagram of a smartphone which a measured in an anechoic chamber
    :param fig_name: saved figure name
    :return: figure
    """
    plt.figure()
    offsets_H = np.array([-61, -60, -60, -66, -65, -72, -61, -56,
                          -61, -57, -54, -63, -61, -65, -63, -64, -61])
    offsets_H = offsets_H - np.mean(offsets_H)
    degree_centers = np.linspace(0, 360, 17)
    plt.subplot(projection='polar')
    plt.plot(utils.degree2radian(degree_centers), offsets_H, marker='o', label='Horizontal Diagram')
    plt.plot(utils.degree2radian(degree_centers), np.zeros(offsets_H.size), marker='o', label='Reference Circle')
    plt.xticks(utils.degree2radian(degree_centers))
    plt.yticks(np.linspace(-15, 10, 6))
    plt.ylim([-15, 10])
    plt.legend()
    plt.title('Diagram of the Smartphone')
    if not os.path.exists(dir_fig):
        os.makedirs(dir_fig)
    plt.savefig(dir_fig + '/' + fig_name + '.png')


def plot_correction_diagram(fig_name):
    """
    plot correctionn RSSI result using the diagram of the smartphone measured in an anechoic chamber
    :param fig_name: saved figure name
    :return: figure
    """
    dir = "log/circular antenna/around car"
    files = utils.walk_files(dir)
    pattern_loc = r'.*\\(.*?) (.*?).csv$'
    radian_ref = 0.2
    for i in range(len(files)):
        file = files[i]
        match_obj = re.match(pattern_loc, file)
        if (match_obj.groups()[0] == 'left') & (match_obj.groups()[1] == '2m50'):
            df = pd.read_csv(file, sep=';', decimal=',')
            df1 = df[df['RSSI ANTENNA PCB'] != 0]
            rssi = pd.DataFrame(df1['RSSI ANTENNA PCB'])
            radian = pd.DataFrame(df1['ORIENTATION_X'])
            radian_mapped = radian.apply(utils.mapping_radian, args=(radian_ref, match_obj.groups()[0],))
            rssi_corrected = utils.correct_diagram(rssi, radian_mapped)
            rssi.columns = ['Original']
            rssi_corrected.columns = ['Corrected']
            rssi.index = np.arange(rssi.shape[0]) / utils.FREQ_LOG
            rssi_corrected.index = np.arange(rssi_corrected.shape[0]) / utils.FREQ_LOG

            plt.figure(figsize=(12, 8))
            ax1 = plt.subplot(2, 1, 1)
            rssi.plot(kind='line', ax=ax1)
            rssi_corrected.plot(kind='line', ax=ax1)
            plt.title(match_obj.groups()[0] + ' ' + match_obj.groups()[1])
            plt.title('RSSI Variation over time')
            plt.xlabel('Time: s')
            plt.ylabel('RSSI: dBm')

            ax3 = plt.subplot(2, 2, 3)
            rssi.plot(kind='density', ax=ax3)
            rssi_corrected.plot(kind='density', ax=ax3)
            plt.title('Distribution')
            plt.xlabel('RSSI: dBm')

            ax4 = plt.subplot(2, 2, 4)
            df = pd.DataFrame()
            df['Original'] = rssi
            df['Corrected'] = rssi_corrected
            df.plot(kind='box', ax=ax4)
            plt.title('Box')
            plt.ylabel('RSSI: dBm')
            if not os.path.exists(dir_fig):
                os.makedirs(dir_fig)
            plt.savefig(dir_fig + '/' + fig_name + '.png')


def plot_influence_wifi(fig_name):
    """
    plot a figure illustrating the influence of wifi in the RSSI measurement
    :param fig_name: saved figure name
    :return: figure
    """
    plt.figure()
    file_with = 'log/influence/wifi/with wifi.csv'
    file_without = 'log/influence/wifi/without wifi.csv'
    rssi_with = utils.read_file_rssi(file_with, correct_index=True)[colums[0]]
    rssi_without = utils.read_file_rssi(file_without, correct_index=True)[colums[0]]
    time = min(rssi_with.index.max(), rssi_without.index.max())
    rssi_with[:time].plot(kind='line', label='With WiFi')
    rssi_without[:time].plot(kind='line', label='Without WiFi')
    plt.title(fig_name)
    plt.xlabel('Time: s')
    plt.ylabel('RSSI: dBm')
    plt.legend()
    if not os.path.exists(dir_fig):
        os.makedirs(dir_fig)
    plt.savefig(dir_fig + '/' + fig_name + '.png')


def plot_influence_smartphone(fig_name):
    """
    plot a figure illustrating the smartphone offset in the RSSI measurement
    :param fig_name: saved figure name
    :return: figure
    """
    plt.figure()
    file_A3 = 'log/influence/smartphone/A3/left/A3_left_1m.csv'
    file_A5 = 'log/influence/smartphone/A3/left/A5_left_1m.csv'
    rssi_A3 = utils.read_file_rssi(file_A3, correct_index=True)[colums[0]]
    rssi_A5 = utils.read_file_rssi(file_A5, correct_index=True)[colums[0]]
    time = min(rssi_A3.index.max(), rssi_A5.index.max())
    rssi_A3[:time].plot(kind='line', label='Samsung A3')
    rssi_A5[:time].plot(kind='line', label='Samsung A5')
    plt.title(fig_name)
    plt.xlabel('Time: s')
    plt.ylabel('RSSI: dBm')
    plt.legend()
    if not os.path.exists(dir_fig):
        os.makedirs(dir_fig)
    plt.savefig(dir_fig + '/' + fig_name + '.png')


def plot_influence_body(fig_name):
    """
    plot a figure illustrating the possible attenuation by body
    :param fig_name: saved figure name
    :return: figure
    """
    plt.figure()
    file_with = 'log/influence/body/Left 1m with body.csv'
    file_without = 'log/influence/body/Left 1m without body.csv'
    rssi_with_subset = utils.read_file_rssi_37(file_with, correct_index=True)[colums[0]]
    rssi_without_subset = utils.read_file_rssi_37(file_without, correct_index=True)[colums[0]]
    time = min(rssi_with_subset.index.max(), rssi_without_subset.index.max())
    rssi_with_subset[:time].plot(kind='line', label='With Body Attenuation')
    rssi_without_subset[:time].plot(kind='line', label='Without Body Attenuation')
    plt.xlabel('Time: s')
    plt.ylabel('RSSI: dBm')
    plt.title(fig_name)
    plt.legend()
    if not os.path.exists(dir_fig):
        os.makedirs(dir_fig)
    plt.savefig(dir_fig + '/' + fig_name + '.png')


def plot_influence_environment(fig_name):
    """
    plot a figure illustrating the inside/outside environment difference for RSSI measurement
    :param fig_name: saved figure name
    :return: figure
    """
    file_inside = 'log/influence/environment/right/right_inside.csv'
    file_outside = 'log/influence/environment/right/right_outside.csv'
    rssi_inside = utils.read_file_rssi(file_inside, 4, correct_index=True)
    rssi_outside = utils.read_file_rssi(file_outside, 4, correct_index=True)
    plt.figure(figsize=(16, 6))
    ax1 = plt.subplot(1, 2, 1)
    rssi_outside.plot(kind='line', ax=ax1)
    plt.title('Outside Environment')
    plt.xlabel('Time: s')
    plt.ylabel('RSSI: dBm')
    ax2 = plt.subplot(1, 2, 2)
    rssi_inside.plot(kind='line', ax=ax2, sharey=ax1)
    plt.title('Inside Environment')
    plt.xlabel('Time: s')
    if not os.path.exists(dir_fig):
        os.makedirs(dir_fig)
    plt.savefig(dir_fig + '/' + fig_name + '.png')


def plot_correct_body_effect():
    """
    plot figures showing the effectiveness of using measurement difference for body attenuation purpose
    :return: figures
    """
    file_with = 'log/influence/body/Left 2m with body.csv'
    file_without = 'log/influence/body/Left 2m without body.csv'
    rssi_with_subset = utils.read_file_rssi_37(file_with, correct_index=True)
    rssi_without_subset = utils.read_file_rssi_37(file_without, correct_index=True)
    plt.figure(figsize=(16, 6))
    ax1 = plt.subplot(1, 2, 1)
    rssi_with_subset.plot(kind='line', ax=ax1)
    plt.xlabel('Time: s')
    plt.ylabel('RSSI: dBm')
    plt.title('With Body Attenuation')
    ax2 = plt.subplot(1, 2, 2)
    rssi_without_subset.plot(kind='line', ax=ax2, sharey=ax1)
    plt.xlabel('Time: s')
    plt.title('Without Body Attenuation')
    plt.figure(figsize=(16, 6))
    rssi_with_subset = rssi_with_subset.apply(lambda x: x - rssi_with_subset[colums[1]])
    rssi_without_subset = rssi_without_subset.apply(lambda x: x - rssi_without_subset[colums[1]])
    ax3 = plt.subplot(1, 2, 1)
    rssi_with_subset.plot(kind='line', ax=ax3)
    plt.xlabel('Time: s')
    plt.ylabel('Difference: dB')
    plt.title('Difference With Body Attenuation')
    ax4 = plt.subplot(1, 2, 2)
    rssi_without_subset.plot(kind='line', ax=ax4, sharey=ax3)
    plt.xlabel('Time: s')
    plt.title('Difference Without Body Attenuation')


def plot_PCA(fig_name):
    """
    plot a figure for PCA dimension reduction
    :param fig_name: saved figure name
    :return: figure
    """
    dir = "log/peps mini"
    pattern = r'(internal|access|lock)\\\d{1,2}.csv$'
    pattern_valid = r'(3|6|9|12).csv$'
    utils.construct_set(dir, pattern, pattern_valid)
    X, y = utils.load_all()
    utils.plot_PCA(X, y)
    plt.title(fig_name)
    if not os.path.exists(dir_fig):
        os.makedirs(dir_fig)
    plt.savefig(dir_fig + '/' + fig_name + '.png')


def plot_TSNE(fig_name):
    """
    plot a figure for 2D t-SNE dimension reduction
    :param fig_name: saved figure name
    :return: figure
    """
    dir = "log/peps mini"
    pattern = r'(internal|access|lock)\\\d{1,2}.csv$'
    pattern_valid = r'(3|6|9|12).csv$'
    utils.construct_set(dir, pattern, pattern_valid)
    X, y = utils.load_all()
    utils.plot_TSNE(X, y)
    plt.title(fig_name)
    if not os.path.exists(dir_fig):
        os.makedirs(dir_fig)
    plt.savefig(dir_fig + '/' + fig_name + '.png')


def plot_report(fig_name, plot_cm=False):
    """
    plot the comparison result using different ML methods
    :param fig_name: saved figure name
    :param plot_cm: whether to plot confusion matrix result
    :return: figure
    """
    dir = "log/peps mini"
    pattern = r'(internal|access|lock)\\\d{1,2}.csv$'
    pattern_valid = r'(3|6|9|12).csv$'
    utils.construct_set(dir, pattern, pattern_valid)
    X_train, X_valid, y_train, y_valid = utils.load_train_valid()
    methods = ["Logistic", "LDA", "QDA", "KNN", "SVM", "RF", "GBM", "MLP"]
    params = [None,
              None,
              None,
              {"n_neighbors": 10},
              {"C": 0.25, "gamma": 0.5},
              {"max_features": 2, "n_estimators": 100},
              {"n_estimators": 400, "max_depth": 3},
              {"hidden_layer_sizes": (16, 8)}]
    df_report = pd.DataFrame()
    for method, param in zip(methods, params):
        cm, report_temp, classes = utils.train(X_train, X_valid, y_train, y_valid, method=method, param=param)
        df_report = df_report.append(report_temp, ignore_index=True)
        if plot_cm:
            plt.figure()
            utils.plot_confusion_matrix(cm, classes, normalize=True)
            plt.title(method)
            if not os.path.exists(dir_fig + '/methods/'):
                os.makedirs(dir_fig + '/methods/')
            plt.savefig(dir_fig + '/methods/' + method + '.png')
    df_report.set_index('method', inplace=True)
    df_report.plot(kind='bar', rot=0, figsize=(16, 6), ylim=(0.6, 1))
    plt.title(fig_name)
    if not os.path.exists(dir_fig):
        os.makedirs(dir_fig)
    plt.savefig(dir_fig + '/' + fig_name + '.png')


def plot_correction_RSSI(fig_name):
    """
    plot a figure for the RSSi correction using unilateral limiter
    :param fig_name: saved figure name
    :return: figure
    """
    file = 'log/peps mini/access/1.csv'
    rssi_raw = utils.read_file_rssi(file, 4)[:500]
    rssi_corrected = rssi_raw.apply(utils.correct_rssi_unilateral, args=(1,))[:500]
    plt.figure()
    plt.plot(np.arange(rssi_raw.shape[0]) / utils.FREQ_LOG, rssi_raw[colums[0]], label='Original')
    plt.plot(np.arange(rssi_corrected.shape[0]) / utils.FREQ_LOG, rssi_corrected[colums[0]], label='Corrected')
    plt.xlabel('Time: s')
    plt.ylabel('RSSI: dBm')
    plt.legend()
    plt.title(fig_name)
    if not os.path.exists(dir_fig):
        os.makedirs(dir_fig)
    plt.savefig(dir_fig + '/' + fig_name + '.png')


def plot_comparaison_filter(fig_name):
    """
    plot a figure illustrating improved performance using RSSI unilateral correction
    :param fig_name: saved figure name
    :return: figure
    """
    dir = "log/peps mini"
    pattern = r'(internal|access|lock)\\\d{1,2}.csv$'
    pattern_valid = r'(3|6|9|12).csv$'
    df_report = pd.DataFrame()
    utils.construct_set(dir, pattern, pattern_valid, filter=0)
    X_train, X_valid, y_train, y_valid = utils.load_train_valid()
    cm, report_temp, classes = utils.train(X_train, X_valid, y_train, y_valid, method='RF',
                                           param={"max_features": 2, "n_estimators": 100})
    df_report = df_report.append(report_temp, ignore_index=True)
    utils.construct_set(dir, pattern, pattern_valid, filter=1)
    X_train, X_valid, y_train, y_valid = utils.load_train_valid()
    cm, report_temp, classes = utils.train(X_train, X_valid, y_train, y_valid, method='RF',
                                           param={"max_features": 2, "n_estimators": 100})
    df_report = df_report.append(report_temp, ignore_index=True)
    df_report.index = ['Original', 'Corrected']
    df_report.plot(kind='bar', rot=0, ylim=(0.6, 1))
    plt.title(fig_name)
    if not os.path.exists(dir_fig):
        os.makedirs(dir_fig)
    plt.savefig(dir_fig + '/' + fig_name + '.png')


def plot_comparaison_filter2():
    """
    plot a figure comparing the performance using different correction filters
    (Raw, Unilateral Limiter, Bilateral Limiter, Time Averaging, Log Sepctrum Averaging)
    :return: figure
    """
    dir = "log/NY/peps mini"
    pattern = r'(internal|access|lock)\\\d{1,2}.csv$'
    pattern_valid = r'(3|6|9|12).csv$'
    df_report = pd.DataFrame()
    filters = ['Raw', 'ULimiter', 'BLimiter', 'TA:n=5', 'TA:n=10', 'LSA:n=5', 'LSA:n=10']
    for index, name in enumerate(filters):
        print('Filter: ' + name)
        utils.construct_set(dir, pattern, pattern_valid, filter=index)
        X_train, X_valid, y_train, y_valid = utils.load_train_valid()
        cm, report_temp, classes = utils.train(X_train, X_valid, y_train, y_valid, method='RF',
                                               param={"max_features": 2, "n_estimators": 100})
        df_report = df_report.append(report_temp, ignore_index=True)
    df_report.index = filters
    df_report.plot(kind='bar', rot=0, ylim=(0.8, 0.95))
    df_report.to_csv('data/comparaison_filter.csv', sep=';')


def compare_binary_multi():
    """
    plot a figure comparing the binary classification(access, lock) and multiclass classification(left, right, front, back, lock)
    :return: figure
    """
    dir = "log\\NY\\peps"
    pattern = r'(front|left|right|back|lock)\\\d.csv$'
    pattern_valid = r'3.csv$'
    utils.construct_set(dir, pattern, pattern_valid, filter=1)
    cm_list = list()
    labels_list = list()
    # multi classification
    X_train, X_valid, y_train, y_valid = utils.load_train_valid()
    cm, report_temp, classes = utils.train(X_train, X_valid, y_train, y_valid, method='RF',
                                           param={"max_features": 2, "n_estimators": 100})
    cm_list.append(cm)
    labels_list.append(classes)
    # binary classification
    y_train = np.array(y_train)
    y_train[y_train != 'lock'] = 'access'
    y_valid = np.array(y_valid)
    y_valid[y_valid != 'lock'] = 'access'
    cm, report_temp, classes = utils.train(X_train, X_valid, y_train, y_valid, method='RF',
                                           param={"max_features": 2, "n_estimators": 100})
    cm_list.append(cm)
    labels_list.append(classes)


def plot_distribution_prob(fig_name):
    """
    plot a figure showing the distribution of the max probability
    :param fig_name: saved figure name
    :return: figure
    """
    dir = "log/peps mini"
    pattern = r'(internal|access|lock)\\\d{1,2}.csv$'
    pattern_valid = r'(3|6|9|12).csv$'
    utils.construct_set(dir, pattern, pattern_valid, filter=1)
    X_train, X_valid, y_train, y_valid = utils.load_train_valid()
    utils.train(X_train, X_valid, y_train, y_valid, method='RF',
                param={"max_features": 2, "n_estimators": 100}, save_prob=True)
    utils.plot_max_probablity_distribution('RF')
    plt.title(fig_name)
    if not os.path.exists(dir_fig):
        os.makedirs(dir_fig)
    plt.savefig(dir_fig + '/' + fig_name + '.png')


def plot_heatmap():
    """
    plot a example of heatmap
    :return: figure
    """
    dir = "log/heatmap"
    pattern = r'Ligne (\d{1,2}).csv$'
    utils.construct_heatmap_set(dir, pattern)
    df_heat = pd.read_csv('data/heat.csv', sep=';')
    heatmap_dir = os.path.join('figures', 'heatmaps')
    for beacon in colums:
        utils.plot_heatmap(df_heat, beacon)
        plt.title(beacon)
        if not os.path.isdir(heatmap_dir):
            os.makedirs(heatmap_dir)
        plt.savefig(os.path.join(heatmap_dir, utils.find_beacon_name(beacon) + '.png'))


def plot_mlp(fig_name):
    """
    plot MLP training result
    :param fig_name: saved figure name
    :return: figure
    """
    dir = "log/heatmap"
    pattern = r'(\d{1,2}).csv$'
    utils.construct_heatmap_set(dir, pattern)
    df = pd.read_csv('data/df.csv', sep=';')
    Px, Py = utils.point2XY(df['COUNTER FLAG'])
    Px.name = 'Px'
    Py.name = 'Py'
    X = df.iloc[:, 0:8]
    X_train, X_valid, Px_train, Px_valid, Py_train, Py_valid = train_test_split(X, Px, Py)
    h2o.init()
    mlp_px, mlp_py = utils.train_mlp(X_train, X_valid, Px_train, Px_valid, Py_train, Py_valid)
    if not os.path.exists(dir_fig):
        os.makedirs(dir_fig)
    utils.plot_mlp_result(mlp_px)
    plt.title('Train result for Px')
    plt.savefig(dir_fig + '/' + fig_name + '4px.png')
    utils.plot_mlp_result(mlp_py)
    plt.title('Train result for Py')
    plt.savefig(dir_fig + '/' + fig_name + '4py.png')
    h2o.cluster().shutdown()


def plot_filter(fig_name):
    """
    illustrate unilateral correction
    :param fig_name: saved figure name
    :return: figure
    """
    x = [0, 0.8, 1]
    y = [0, 0.4, 0.5]
    plt.figure(figsize=(8, 3))
    plt.plot(x, y, ls='-.', marker='o', markersize=10, color='b')
    plt.grid(False)
    plt.xlim([-0.5, 1.5])
    plt.ylim([-0.5, 1])
    plt.xticks([])
    plt.yticks([])
    delta = 0.1
    plt.text(x[0], y[0] + delta, r'$P_{n-1}$')
    plt.text(x[1], y[1] + delta, r'$\tilde{P}_{n}$')
    plt.text(x[2], y[2] + delta, r'$P_{n}$')
    plt.title(fig_name)
    if not os.path.exists(dir_fig):
        os.makedirs(dir_fig)
    plt.savefig(dir_fig + '/' + fig_name + '.png')


if __name__ == '__main__':
    plot_map('map')
    plot_point('point')
    plot_diagram_horizontal('diagram')
    plot_correction_diagram('correction_diagram')
    plot_rssi_variation('rssi_variation')
    plot_channel_offset('channel')
    plot_path_loss_theoretical('pathloss_theoretical')
    plot_path_loss_experimental('pathloss_experimental')
    plot_influence_wifi('wifi')
    plot_influence_smartphone('smartphone')
    plot_influence_body('body')
    plot_influence_environment('environment')
    plot_correct_body_effect()
    plot_PCA('pca')
    plot_TSNE('tsne')
    plot_report('report', plot_cm=True)
    plot_correction_RSSI('correction_rssi')
    plot_comparaison_filter('comparaison_filter')
    plot_distribution_prob('distribution_prob')
    plot_heatmap()
    plot_mlp('mlp')
    plot_filter('limiter filter')
    plt.show()
