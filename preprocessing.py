import os
import datetime
import time

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


def read_csv(path, test_size=0.15, categories=3):
    df = pd.read_csv(path).drop('Unnamed: 0', axis=1)
    df.sort_values('date', axis=0, ascending=True, inplace=True, kind='quicksort', na_position='last', ignore_index=True, key=None)
    if categories == 2:
        df['direction'] = df['direction'].map(lambda x : float(1 & int(x)))
    df = pd.concat([df, pd.get_dummies(df['direction'])], axis=1).drop(['direction'], axis=1)

    beg = datetime.datetime.strptime(df['date'][0], '%Y-%m-%d %H:%M:%S')

    def date_to_hours(date: str):
        d = datetime.datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
        return (d - beg).total_seconds() / 3600

    df['hours'] = df['date'].map(date_to_hours)
    df = df.drop(['date'], axis=1)

    shuffle = False
    train, test = train_test_split(df, shuffle=shuffle, test_size=test_size)
    train, validation = train_test_split(train, shuffle=shuffle, test_size=test_size)

    print('input shape\ntrain:    {}\nvalidate: {}\ntest:     {}\n'.format(train.shape, validation.shape, test.shape))

    return train, validation, test

def compute_class_weights(df):
    total = df.shape[0]
    c0 = df[df[0.0] == 1].shape[0]
    c1 = df[df[1.0] == 1].shape[0]
    c2 = df[df[2.0] == 1].shape[0]

    weight_for_0 = (1 / c0)*(total)/3.0
    weight_for_1 = (1 / c1)*(total)/3.0
    weight_for_2 = (1 / c2)*(total)/3.0

    class_weight = {0: weight_for_0, 1: weight_for_1, 2: weight_for_2}
    print('class_weights', class_weight)
    return class_weight