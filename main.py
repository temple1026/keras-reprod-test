#%%
# import required modules
import os 
import numpy as np
import random
import sys
import time
import datetime

from sklearn.model_selection import KFold, train_test_split

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input, Conv2D, MaxPooling2D, Reshape

def setSeed(seed, mode_ops):
    os.environ['PYTHONHASHSEED'] = '0'
    
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if mode_ops:
        # see https://suneeta-mall.github.io/2019/12/22/Reproducible-ml-tensorflow.html

        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

    else:
        os.environ['TF_DETERMINISTIC_OPS'] = '0'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '0'
        
        max_workers = 4
        tf.config.threading.set_inter_op_parallelism_threads(max_workers)
        tf.config.threading.set_intra_op_parallelism_threads(max_workers)


def loadModel(width, height):

    layer_input = Input(shape=(width, height))
    layers_hidden = Reshape(target_shape=(width, height, 1))(layer_input)
    layers_hidden = Conv2D(filters=8, kernel_size=(3, 3), strides=1, activation="relu")(layers_hidden)
    layers_hidden = MaxPooling2D(pool_size=(2,2), strides=(2,2))(layers_hidden)
    layers_hidden = Conv2D(filters=8, kernel_size=(3, 3), strides=1, activation="relu")(layers_hidden)
    layers_hidden = MaxPooling2D(pool_size=(2,2), strides=(2,2))(layers_hidden)
    # layers_hidden = Conv2D(filters=8, kernel_size=(3, 3), strides=1, activation="relu")(layers_hidden)
    # layers_hidden = MaxPooling2D(pool_size=(2,2), strides=(2,2))(layers_hidden)
    layers_hidden = Flatten()(layers_hidden)
    layers_hidden = Dense(128, activation="relu")(layers_hidden)
    layers_hidden = Dropout(0.2)(layers_hidden)
    layer_output = Dense(10, activation="softmax")(layers_hidden)
    
    return Model(inputs=layer_input, outputs=layer_output)

def run_kfold_validation(n_splits):
    # load train and test sets from mnist
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    kfold = KFold(n_splits=n_splits, shuffle=True)
    
    width, height = x_train[0].shape

    losses = []
    accs = []

    for idxs_train, idxs_test in kfold.split(x_train, y_train):
        # load the neural network model
        model = loadModel(width, height)

        # compile   
        model.compile(optimizer='adam',
                loss="sparse_categorical_crossentropy",
                metrics=['accuracy'])

        # model.summary()

        model.fit(x=x_train[idxs_train], y=y_train[idxs_train], epochs=10, validation_split=0.2, batch_size=100, verbose=2)
        
        loss, acc = model.evaluate(x=x_train[idxs_test], y=y_train[idxs_test], verbose=2)

        losses.append(loss)
        accs.append(acc)

    return losses, accs

def main():
    # VRAMを無駄に確保しないように設定
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # 引数に0か1を受け取る
    args = sys.argv

    if len(args) == 1:
        print("Please choose 1 (use tf.option.threading) or 0 (not use them).")
        exit(1)

    mode_ops = int(args[1])

    # 引数が0なら再現性を確保しないファイル名を指定する．0か1以外なら例外

    dict_files = {1:"with_tf_option.txt", 0:"without_tf_option.txt"}
    try:
        print(dict_files[mode_ops])
    except KeyError:
        print("Arg must have 0 or 1.")
        exit(1)

    # 乱数と再現性の確保の設定
    setSeed(seed=0, mode_ops=mode_ops)

    # 学習
    losses, accs = run_kfold_validation(n_splits=10)
    
    # 学習結果を保存するためのフォルダのパスを定義．フォルダが無ければつくる．
    dist_dir = os.path.join("results")
    os.makedirs(dist_dir, exist_ok=True)

    # 結果をテキストに書き込む
    with open(os.path.join(dist_dir, dict_files[mode_ops]), "a") as f:
        # 現在の時刻を取得
        date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        f.writelines(date + "\n" + "".join([f"{i} {loss} {acc}\n" for i, (loss, acc) in enumerate(zip(losses, accs))]))

if __name__=="__main__":
    main()