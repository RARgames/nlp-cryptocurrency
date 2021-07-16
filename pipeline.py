import sys
import json

import preprocessing
import datagenerator
import models
import analyzer

def run(parameters_path):
    with open(parameters_path, 'r') as f:
        parameters = json.load(f)

    inputs_path = parameters.get('inputs_path')
    embeddings_path = parameters.get('embeddings_path')
    embeddings_dim = parameters.get('embeddings_dim')
    results_path = parameters.get('results_path')

    test_size = parameters.get('test_size', 0.15)
    categories = parameters.get('categories', 2)
    use_weights = parameters.get('use_wieghts', True)

    print('Reading data')

    train, validation, test =  preprocessing.read_csv(inputs_path, test_size, categories)

    if use_weights:
        weights = preprocessing.compute_class_weights(train)
    else:
        weights = None

    print('Preprocessing data')
    series_input_width = parameters.get('series_input_width', 100)
    txt_window_width = parameters.get('txt_window_width', 10)
    hours_window_width = parameters.get('hours_window', 6)
    shuffle = parameters.get('shuffle', True)
    text_max_features = parameters.get('text_max_features', 10000)
    text_sequence_length = parameters.get('text_sequence_length', 40)
    batch_size = parameters.get('batch_size', 32)

    dg = datagenerator.DataGenerator(series_input_width, txt_window_width, shuffle=shuffle, batch_size=batch_size,
                       hours_window=hours_window_width, categories=categories, train_df=train,
                       val_df=validation, test_df=test, text_max_features=text_max_features,
                       text_sequence_length=text_sequence_length)

    print("creating embeddings matrix")

    emb_matrix = models.create_embeddings_matrix(dg.vectorizer, embeddings_path, embeddings_dim)

    print("building model")
    optimizer = parameters.get('optimizer', 'adam')
    loss = parameters.get('loss', 'categorical_crossentropy')
    model_fn = parameters.get('model_fn', 'create_model_gru')

    model_fn = getattr(models, model_fn)
    model = models.build_model(emb_matrix, model_fn, categories, optimizer=optimizer, loss=loss)

    model_name = parameters.get('model_name', 'no_name_model')

    print("training and validating model")
    epochs = parameters.get("epochs", 5)
    early_stopping_min_delta = parameters.get("early_stopping_min_delta", 0.01)
    early_stopping_patience = parameters.get("early_stopping_patience", 2)

    an = analyzer.Analyzer(results_path)
    an.train_model(model, model_name, dg, epochs, weights,
                   early_stopping_min_delta, early_stopping_patience,
                   batch_size, parameters)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('No parameters file!')
    else:
        run(sys.argv[1])