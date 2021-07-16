import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import losses


def create_embeddings_matrix(vectorizer, embeddings_path, embedding_dim=100, mask_zero=True):
    embeddings_index = {}
    with open(embeddings_path) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))

    num_tokens = len(voc) + 2
    hits = 0

    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            hits += 1

    print("Converted %d words from %d" % (hits, len(voc)))

    return layers.Embedding(
            num_tokens,
            embedding_dim,
            embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
            trainable=False,
            mask_zero=mask_zero
    )

def create_model_lstm(embedding_layer, num_labels=3):

    text_input = layers.Input(shape=(None,), name='text')
    txt = embedding_layer(text_input)
    txt = layers.Bidirectional(tf.keras.layers.LSTM(64, recurrent_dropout=0.5, dropout=0.5))(txt)
    x = layers.Dropout(0.25)(txt)
    out = layers.Dense(num_labels, activation='softmax')(x)

    return tf.keras.Model(inputs=[text_input], outputs=[out])


def create_model_gru(embedding_layer, num_labels=3):

    text_input = layers.Input(shape=(None,), name='text')
    txt = embedding_layer(text_input)
    txt = tf.keras.layers.GRU(128)(txt)
    # txt = layers.Bidirectional(tf.keras.layers.GRU(64, recurrent_dropout=0.5, dropout=0.5))(txt)

    series_input = layers.Input(shape=(None, num_labels), name='series')
    series = layers.GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(series_input)
    series = layers.GRU(64)(series)
    series = layers.Reshape([-1])(series)

    x = layers.concatenate([txt, series])

    # txt = layers.Dropout(0.25)(x)

    x = layers.Dense(64)(x)

    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_labels, activation='softmax')(x)

    return tf.keras.Model(inputs=[text_input, series_input], outputs=[out])


def create_model_lstm_big(embedding_layer, num_labels=3):

    text_input = layers.Input(shape=(None,), name='text')
    txt = embedding_layer(text_input)
    txt = layers.Bidirectional(tf.keras.layers.LSTM(64, recurrent_dropout=0.5, dropout=0.5))(txt)
    txt = layers.Dense(32)(txt)

    series_input = layers.Input(shape=(None,num_labels), name='series')
    series = layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)(series_input)
    series = layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2)(series)
    series = layers.Dense(32)(series)
    series = layers.Reshape([-1])(series)

    x = layers.concatenate([txt, series])

    x = layers.Dropout(0.25)(txt)
    out = layers.Dense(num_labels, activation='softmax')(x)

    return tf.keras.Model(inputs=[text_input], outputs=[out])


def build_model(embeddings_layer, model_fn, categories=3, optimizer='adam',
                loss='categorical_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy()]):
    model = model_fn(embeddings_layer, categories)
    model.summary()
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model

def early_stopping(min_delta=1e-3, patience=3, monitor='val_categorical_accuracy'):
    return tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        min_delta=min_delta,
        patience=patience,
        verbose=1,
        restore_best_weights=True
    )
