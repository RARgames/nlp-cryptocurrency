import datetime
import time

import numpy as np
import pandas as pd
import tensorflow as tf

class DataGenerator():

  def __init__(self, input_width, txt_width, label_width=1, shift=1, shuffle=True, batch_size=32,
               text_max_features=1000, text_sequence_length=40, hours_window=6, categories=3,
               train_df=None, val_df=None, test_df=None):
    """DataGenerator transforms input DataFrames to time-series data(text is transformed
        to the numerical format).

    Args:
        input_width (int): width of the window with previous labels
        txt_width (int): width of the window with text inputs
        label_width (int, optional): Width of the window with labels to predict. Defaults to 1.
        shift (int, optional): Shift between windows. Defaults to 1.
        shuffle (bool, optional): If True data is shuffled. Defaults to True.
        batch_size (int, optional): Defaults to 32.
        text_max_features (int, optional): Maximum number of tokens to genrate when
        vectorizing text inputs. Defaults to 1000.
        text_sequence_length (int, optional): Length of text sequence. Defaults to 40.
        hours_window (int, optional): Only records older than hours_window are slected for
        series input. Defaults to 6.
        categories (int, optional): Number of categories in labels. Defaults to 3.
        train_df (pd.DataFrame, optional): DataFrame with train data.
        Data should be in the format: text, categorical labels, timestamp. Defaults to None.
        val_df (pd.DataFrame, optional): DataFrame with validation data. Defaults to None.
        test_df (pd.DataFrame, optional): DataFrame with test data. Defaults to None.
    """

    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    self.text_max_features = text_max_features
    self.text_sequence_length =  text_sequence_length
    self.batch_size = batch_size
    self.shuffle = shuffle

    self.input_width = input_width
    self.label_width = label_width
    self.txt_width = txt_width
    self.shift = shift
    self.hours_window = hours_window
    self.categories = categories

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.txt_input_start = self.total_window_size - self.txt_width
    self.txt_input_slice = slice(self.txt_input_start, None)
    self.txt_input_indices = np.arange(self.total_window_size)[self.txt_input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    self._init_vectorizer()

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}'])

  def _init_vectorizer(self):
    self.vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens = self.text_max_features,
        output_sequence_length = self.text_sequence_length
    )

    self.vectorizer.adapt(self.train_df['text'].values)


  def _split_window(self, features):
    series_inputs = features[:, self.input_slice, -self.categories-1:-1]
    series_times = features[:, self.input_slice, -1]

    text_inputs = features[:, self.txt_input_slice, :-self.categories-1]
    labels = features[:, self.labels_slice, -self.categories-1:-1]
    current_time = features[:, self.labels_slice, -1]


    series_inputs.set_shape([None, self.input_width, None])
    text_inputs.set_shape([None, self.txt_width, None])
    labels.set_shape([None, self.label_width, None])

    series_times.set_shape([None, self.input_width])
    current_time.set_shape([None, self.label_width])

    mask = tf.math.less(series_times, current_time - self.hours_window)
    series_inputs = tf.ragged.boolean_mask(series_inputs, mask)


    text_inputs = tf.reshape(text_inputs, [-1, self.txt_width*text_inputs.shape[2]])
    labels = tf.reshape(labels, [-1, self.categories])

    return {'series': series_inputs, 'text': tf.cast(text_inputs, tf.int64)}, labels

  def make_dataset(self, data, shuffle):
    txt = np.array(self.vectorizer(data['text']).numpy(), dtype=np.float32)
    data = np.array(data.drop('text', axis=1), dtype=np.float32)
    data = np.concatenate([txt, data], axis=1)

    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=self.shift,
        shuffle=shuffle,
        batch_size=self.batch_size,)

    ds = ds.map(self._split_window)

    return ds

  def train(self):
    """Generates dataset for time-series models from train data.

    Returns:
        TF DataSet: Format of each batch ({'text': txt, 'series': series}, labels) where
        txt - vectorized text, vector's size is txt_width*text_sequence_length;
        series - previous labels data, size depends on time distribution. From `input_width` number of
        previous labels, only those are selected that happend `hours_window` before current record
        labels - categorical labels
    """
    return self.make_dataset(self.train_df, shuffle=self.shuffle)

  def val(self):
    return self.make_dataset(self.val_df, shuffle=False)

  def test(self):
    return self.make_dataset(self.test_df, shuffle=False)