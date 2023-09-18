# Author: Sung-Wook Park
# Date: 16 Jun 2022
# Last updated: 11 Jul 2022
# --- Ad hoc ---

import tensorflow as tf

from quantum_layer import QConv

def FC(img_shape, units, num_classes):
    fc_model = tf.keras.models.Sequential()
    fc_model.add(tf.keras.layers.Flatten(input_shape=img_shape))
    fc_model.add(tf.keras.layers.Dense(units, activation='relu'))
    fc_model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    fc_model.compile(optimizer=tf.keras.optimizers.Adam(),
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
    fc_model.summary()

    return fc_model

def CNN(img_shape, units, num_classes):
    cnn_model = tf.keras.models.Sequential()
    cnn_model.add(tf.keras.layers.Conv2D(8, (2, 2), activation='relu', name='conv1', input_shape=img_shape))
    cnn_model.add(tf.keras.layers.Flatten())
    cnn_model.add(tf.keras.layers.Dense(units, activation='relu'))
    cnn_model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    cnn_model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    cnn_model.summary()

    return cnn_model

def QCNN(img_shape, units, num_classes):
    qcnn_model = tf.keras.models.Sequential()
    qcnn_model.add(QConv(filter_size=2, depth=8, activation='relu', name='qconv1', input_shape=img_shape))
    qcnn_model.add(tf.keras.layers.Flatten())
    qcnn_model.add(tf.keras.layers.Dense(units, activation='relu'))
    qcnn_model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    qcnn_model.compile(optimizer=tf.keras.optimizers.Adam(),
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])
    qcnn_model.summary()

    return qcnn_model
