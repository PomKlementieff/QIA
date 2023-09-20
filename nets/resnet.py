# Author: Sung-Wook Park
# Date: 16 Jun 2022
# Last updated: 18 Sep 2023
# --- Ad hoc ---

import tensorflow as tf

from quantum_layer import QConv

def Resnet(img_shape, units, num_classes):
    inputs = tf.keras.Input(shape=img_shape)
    x = QConv(filter_size=2, depth=8, activation='relu', name='qconv1')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    shortcut = x
    x = QConv(filter_size=1, depth=8, activation='relu', name='qconv2')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)    
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units, activation='relu')(x)
    prediction_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    qcnn_model = tf.keras.Model(inputs, prediction_layer)
    qcnn_model.compile(optimizer=tf.keras.optimizers.Adam(),
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])
    qcnn_model.summary()

    return qcnn_model
