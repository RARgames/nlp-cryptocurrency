import os
import datetime
import uuid
import json
import datetime
import time

import numpy as np
import pandas as pd

from tensorflow.keras import layers
from tensorflow.keras import losses

import tensorflow as tf
import sklearn.metrics as metrics

import models

class Analyzer:

    def __init__(self, reports_path):
        self.root_reports_path = reports_path

    def train_model(self, model, model_name, dg, epochs, class_weight=None,
                    early_stopping_min_delta=0.01, early_stopping_patience=2,
                    batch_size=32, model_parameters=None):

        callbacks = [models.early_stopping(early_stopping_min_delta, early_stopping_patience)]

        history = model.fit(dg.train(), validation_data=dg.val(), batch_size=batch_size, epochs=epochs, class_weight=class_weight, callbacks=callbacks)

        self.model = model

        timestamp = datetime.datetime.now()

        self.reports_path = self._report_directory(model_name, timestamp)

        self._compute_metrics_val(dg)
        self._compute_metrics_test(dg)

        self._print_metrics(model_name)

        self._generate_md_report(model_name, timestamp, model_parameters, history)
        self._generate_metrics_json()

        self._save_training_history(history)

        model.save(os.path.join(self.reports_path, 'model'))


    def _save_training_history(self, history):
        pd.DataFrame(history.history).to_csv(os.path.join(self.reports_path, 'history.csv'), index_label='epoch')

    def _compute_metrics_val(self, dg):
        val_ds = dg.val()
        y_true = []
        y_pred = []
        for x, y in val_ds:
            y_pred.append(self.model.predict(x))
            y_true.append(y.numpy())

        yt = np.argmax(np.concatenate(y_true), axis=1)
        yp = np.argmax(np.concatenate(y_pred), axis=1)

        self.accuracy_val = metrics.accuracy_score(yt, yp)
        self.conf_matrix_val = metrics.confusion_matrix(yt, yp)

    def _compute_metrics_test(self, dg):
        test_ds = dg.test()
        y_true = []
        y_pred = []
        for x, y in test_ds:
            y_pred.append(self.model.predict(x))
            y_true.append(y.numpy())

        yt = np.argmax(np.concatenate(y_true), axis=1)
        yp = np.argmax(np.concatenate(y_pred), axis=1)

        self.accuracy_test = metrics.accuracy_score(yt, yp)
        self.conf_matrix_test = metrics.confusion_matrix(yt, yp)

    def map_ds_to_array(self, ds):
        y = []
        for batch in ds:
            y.append(batch)
        return np.concatenate(y, axis=0)

    def _print_metrics(self, model_name):
        print('{} metrics:'.format(model_name))
        print('accuracy val: {:.3f}'.format(self.accuracy_val))
        print('accuracy test: {:.3f}'.format(self.accuracy_test))
        print('confusion matrics val:\n', self.conf_matrix_val)
        print('confusion matrics test:\n', self.conf_matrix_test)

    def _generate_metrics_json(self):
        m = {
            'accuracy_val': self.accuracy_val,
            'conf_matrix_val': self.conf_matrix_val.tolist(),
            'accuracy_test': self.accuracy_test,
            'conf_matrix_test': self.conf_matrix_test.tolist()
        }

        with open(os.path.join(self.reports_path, 'metrics.json'), 'w') as f:
            json.dump(m, f)

    def _generate_md_report(self, model_name, timestamp, model_parameters, history):
        with open(os.path.join(self.reports_path, 'report.md'), 'w') as f:
            f.write('# {}\n*{}*\n'.format(model_name, timestamp.strftime("%Y-%m-%d %H:%M:%S")))
            self._add_model_summary(f)

            if model_parameters is not None:
                self._add_model_parameters(f, model_parameters)

            self._add_metrics(f)
            self._add_confusion_matrix(f, 'val', self.conf_matrix_val)
            self._add_confusion_matrix(f, 'test', self.conf_matrix_test)
            self._add_history(f, history)

    def _add_confusion_matrix(self, file, mat_type, conf_mat):
        text = '## Confusion matrix {}\n'.format(mat_type)

        class_num = conf_mat.shape[0]
        text += ' | '.join([str(x) for x in range(1, class_num+1)]) + '\n'
        text += ' | '.join(['---']*class_num) + '\n'
        for row in conf_mat:
            text += ' | '.join([str(x) for x in row])
            text += '\n'
        file.write(text)

    def _add_metrics(self, file):
        text = '## Metrics\n'
        text += '| Metric | Value \n --- | ---\n'
        text += ' {} | {:.3f} \n'.format('accuracy val', self.accuracy_val)
        text += ' {} | {:.3f} \n'.format('accuracy test', self.accuracy_test)
        file.write(text)


    def _add_model_parameters(self, file, parameters):
        text = '### Model parameters\n'
        text += '| Prameters | Value \n --- | ---\n'
        for k, v in parameters.items():
            text += '{} | {}\n'.format(k,v)
        file.write(text)

    def _add_model_summary(self, file):
        file.write('## Model\n```')
        self.model.summary(print_fn=lambda x : file.write(x + '\n'))
        file.write('```\n')

    def _add_history(self, file, history):
        df = pd.DataFrame(history.history)
        header = ' | '.join(df.columns)
        text = '## History\n'
        text += header + '\n'
        text += ' | '.join(len(df.columns)*['---']) + '\n'
        for _, row in df.iterrows():
            text += ' | '.join([str(round(x,4)) for x in row.values])
            text += '\n'
        file.write(text)

    def _report_directory(self, model_name, timestamp):
        dir_name = 'results_{}_{}_{}/'.format(model_name, timestamp.strftime("%m-%d_%H-%M"), uuid.uuid1().hex[:7])
        path = os.path.join(self.root_reports_path, dir_name)
        try:
            os.mkdir(path)
        except FileExistsError:
            pass
        return path