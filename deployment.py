import os

import h2o
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import utils


def deploy_zone_prediction():
    """
    generate java model for zone prediction
    :return: training result and java model
    """
    dir = "log/peps normal"
    pattern = r'(left|right|front|back|start|trunk|lock)\\\d{1,2}.csv$'
    pattern_valid = r'(3|6|9|12).csv$'
    utils.construct_set(dir, pattern, pattern_valid, filter=1)
    utils.save_to_csv()
    id = 'EightNormal'
    dir_path = 'model/'
    rf = utils.train_rf(model_id=id, ntrees=25, weight_lock=1)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    rf.download_pojo(path=dir_path, get_genmodel_jar=False)


def deploy_coord_prediction():
    """
    generate java model for coord_x and coord_y prediction
    :return: training result and java models
    """
    dir = "log/heatmap"
    pattern = r'(\d{1,2}).csv$'
    utils.construct_heatmap_set(dir, pattern)
    df = pd.read_csv('data/df.csv', sep=';')
    Px, Py = utils.point2XY(df['COUNTER FLAG'])
    Px.name = 'Px'
    Py.name = 'Py'
    X = df.iloc[:, 0:8]
    X, Px, Py = shuffle(X, Px, Py, random_state=0)
    X_train, X_valid, Px_train, Px_valid, Py_train, Py_valid = train_test_split(X, Px, Py)
    mlp_px, mlp_py = utils.train_mlp(X_train, X_valid, Px_train, Px_valid, Py_train, Py_valid,
                                     model_id_px='MLP4Px', model_id_py='MLP4Py')
    utils.plot_mlp_result(mlp_px, title=Px.name)
    utils.plot_mlp_result(mlp_py, title=Py.name)
    plt.show()
    dir_path = 'model/'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    mlp_px.download_pojo(path=dir_path, get_genmodel_jar=True)
    mlp_py.download_pojo(path=dir_path, get_genmodel_jar=True)


if __name__ == '__main__':
    h2o.init()
    deploy_zone_prediction()
    deploy_coord_prediction()
    h2o.cluster().shutdown()
