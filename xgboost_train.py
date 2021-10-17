#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pathlib
from pathlib import Path
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pickle

data_path = Path('data/')
artifact_path = Path('artifact/')

def prepare_data():
    """
    カリフォルニア住宅価格の予測を行います。
    ※こちらのブログを参考にしています。
      * カリフォルニア住宅価格データセットの使い方：https://zerofromlight.com/blogs/detail/65/
      * XGBoostの使い方：https://qiita.com/ganmo0911/items/478be76c5029fff15029#%E6%89%8B%E6%B3%951
    """
    data = fetch_california_housing()
    X = data['data']
    y = data['target']

    # 全データに対して、train, val, testを8:1:1分割を行います。
    X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.2, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=1)

    save_csv(data_path/'X_train.csv', X_train)
    save_csv(data_path/'X_val.csv', X_val)
    save_csv(data_path/'X_test.csv', X_test)
    save_csv(data_path/'y_train.csv', y_train)
    save_csv(data_path/'y_val.csv', y_val)
    save_csv(data_path/'y_test.csv', y_test)

    return


def train():
    print('started train.')
    X_train = load_csv(data_path/'X_train.csv')
    y_train = load_csv(data_path/'y_train.csv')

    X_val = load_csv(data_path/'X_val.csv')
    y_val = load_csv(data_path/'y_val.csv')
    print(X_train)
    print(X_train.shape)
    print(y_train.shape)
    print(X_val.shape)
    print(y_val.shape)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
       "objective" : "reg:squarederror",
        "eval_metric" : "rmse"
    }
    reg = xgb.train(params=params,
                    dtrain=dtrain,
                    evals=[(dtrain, 'train'), (dval, 'val')],
                    num_boost_round=100
    )

    return reg


def evaluate(reg):
    X_test = load_csv(data_path/'X_test.csv')
    y_test = load_csv(data_path/'y_test.csv')
    
    dtest = xgb.DMatrix(X_test)
    y_pred = reg.predict(dtest)
    print(mean_squared_error(y_test, y_pred))

    return


def save_csv(filename, data):
    """
    csvファイル書き込みのラッパー関数。
    """
    np.savetxt(filename, data, delimiter=',')
    return


def load_csv(filename):
    """
    csvファイル読み込みのラッパー関数。
    """
    data = np.loadtxt(filename, delimiter=',')
    return data


def save_artifact(filename, model):
    with open(filename, mode='wb') as fp:
        pickle.dump(model, fp)


def load_artifact(filename):
    model = None
    with open(filename, mode='rb') as fp:
        model = pickle.load(fp)

    if model is None:
        raise ValueError
        
    return model


if __name__ == "__main__":
    # prepare_data()
    reg = train()
    evaluate(reg)
    save_artifact(artifact_path/'xgboost_model.pickle', reg)
