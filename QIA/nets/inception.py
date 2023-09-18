# Author: Sung-Wook Park
# Date: 16 Jun 2022
# Last updated: 11 Jul 2022
# --- Ad hoc ---

import tensorflow as tf

from quantum_layer import QConv

def Naive(img_shape, units, num_classes):
    # Inception module, naive version
    inputs = tf.keras.Input(shape=img_shape)
    conv_1x1 = QConv(filter_size=1, depth=8, activation='relu', name='qconv1')(inputs)
    conv_3x3 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), padding='same', activation='relu')(inputs)
    conv_5x5 = tf.keras.layers.Conv2D(filters=8, kernel_size=(5, 5), padding='same', activation='relu')(inputs)
    max_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    outputs = tf.keras.layers.concatenate([conv_1x1, conv_3x3, conv_5x5, max_pool], axis=3)
    x = tf.keras.layers.Flatten()(outputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(units, activation='relu')(x)
    prediction_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    qcnn_model = tf.keras.Model(inputs, prediction_layer)
    qcnn_model.compile(optimizer=tf.keras.optimizers.Adam(),
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])
    qcnn_model.summary()

    return qcnn_model

def Dimension_Reductions(img_shape, units, num_classes):
    # Inception module with dimension reductions
    inputs = tf.keras.Input(shape=img_shape)
    conv_1x1 = QConv(filter_size=1, depth=16, activation='relu', name='qconv1')(inputs)
    conv_3x3_reduce = QConv(filter_size=1, depth=8, activation='relu', name='qconv2')(inputs)
    conv_3x3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(conv_3x3_reduce)
    conv_5x5_reduce = QConv(filter_size=1, depth=8, activation='relu', name='qconv3')(inputs)
    conv_5x5 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu')(conv_5x5_reduce)
    max_pool = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    pool_proj = QConv(filter_size=1, depth=16, activation='relu', name='qconv4')(max_pool)
    outputs = tf.keras.layers.concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3)
    x = tf.keras.layers.Flatten()(outputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(units, activation='relu')(x)
    prediction_layer = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    qcnn_model = tf.keras.Model(inputs, prediction_layer)
    qcnn_model.compile(optimizer=tf.keras.optimizers.Adam(),
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])
    qcnn_model.summary()

    return qcnn_model
