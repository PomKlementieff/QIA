# Author: Sung-Wook Park
# Date: 16 Jun 2022
# Last updated: 18 Sep 2023
# --- Ad hoc ---

import argparse
import cirq
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq

from cirq.contrib.svg import SVGCircuit
from dataset.cifair_loader import load_cifair10, load_cifair100
from tensorflow.keras.utils import plot_model

np.random.seed(42)

class QConv(tf.keras.layers.Layer):
    def __init__(self, filter_size, depth, activation=None, name=None, kernel_regularizer=None, **kwangs):
        super(QConv, self).__init__(name=name, **kwangs)
        self.filter_size = filter_size
        self.depth = depth
        self.learning_params = []
        self.QCNN_layer_gen()
        # self.circuit_tensor = tfq.convert_to_tensor([self.circuit])
        self.activation = tf.keras.layers.Activation(activation)
        self.kernel_regularizer = kernel_regularizer

    def _next_qubit_set(self, original_size, next_size, qubits):
        step = original_size // next_size
        qubit_list = []
        for i in range(0, original_size, step):
            for j in range(0, original_size, step):
                qubit_list.append(qubits[original_size*i + j])
        return qubit_list

    def _get_new_param(self):
        """
        return new learnable parameter
        all returned parameter saved in self.learning_params
        """
        new_param = sympy.symbols("p"+str(len(self.learning_params)))
        self.learning_params.append(new_param)
        return new_param
    
    def _QConv(self, step, target, qubits):
        """
        apply learnable gates each quantum convolutional layer level
        """
        yield cirq.CZPowGate(exponent=self._get_new_param())(qubits[target], qubits[target+step])
        yield cirq.CXPowGate(exponent=self._get_new_param())(qubits[target], qubits[target+step])
        
    def QCNN_layer_gen(self):
        """
        make quantum convolutional layer in QConv layer
        """
        pixels = self.filter_size**2
        # filter size: 2^n only for this version!
        if np.log2(pixels) % 1 != 0:
            raise NotImplementedError("filter size: 2^n only available")
        cirq_qubits = cirq.GridQubit.rect(self.filter_size, self.filter_size)
        # mapping input data to circuit
        input_circuit = cirq.Circuit()
        input_params = [sympy.symbols('a%d' %i) for i in range(pixels)]
        for i, qubit in enumerate(cirq_qubits):
            input_circuit.append(cirq.rx(np.pi*input_params[i])(qubit))
        # apply learnable gate set to QCNN circuit
        QCNN_circuit = cirq.Circuit()
        step_size = [2**i for i in range(np.log2(pixels).astype(np.int32))]
        for step in step_size:
            for target in range(0, pixels, 2*step):
                QCNN_circuit.append(self._QConv(step, target, cirq_qubits))
        # merge the circuits
        full_circuit = cirq.Circuit()
        full_circuit.append(input_circuit)
        full_circuit.append(QCNN_circuit)
        self.circuit = full_circuit # save circuit to the QCNN layer obj.
        self.params = input_params + self.learning_params
        self.op = cirq.Z(cirq_qubits[0])
        
    def build(self, input_shape):
        self.width = input_shape[1]
        self.height = input_shape[2]
        self.channel = input_shape[3]
        self.num_x = self.width - self.filter_size + 1
        self.num_y = self.height - self.filter_size + 1
        
        self.kernel = self.add_weight(name="kenel", 
                                      shape=[self.depth, 
                                             self.channel, 
                                             len(self.learning_params)],
                                     initializer=tf.keras.initializers.glorot_normal(),
                                     regularizer=self.kernel_regularizer)
        self.circuit_tensor = tfq.convert_to_tensor([self.circuit] * self.num_x * self.num_y * self.channel)
        
    def call(self, inputs):
        # input shape: [N, width, height, channel]
        # slide and collect data
        stack_set = None
        for i in range(self.num_x):
            for j in range(self.num_y):
                slice_part = tf.slice(inputs, [0, i, j, 0], [-1, self.filter_size, self.filter_size, -1])
                slice_part = tf.reshape(slice_part, shape=[-1, 1, self.filter_size, self.filter_size, self.channel])
                if stack_set == None:
                    stack_set = slice_part
                else:
                    stack_set = tf.concat([stack_set, slice_part], 1)  
        # -> shape: [N, num_x*num_y, filter_size, filter_size, channel]
        stack_set = tf.transpose(stack_set, perm=[0, 1, 4, 2, 3])
        # -> shape: [N, num_x*num_y, channel, filter_size, fiter_size]
        stack_set = tf.reshape(stack_set, shape=[-1, self.filter_size**2])
        # -> shape: [N*num_x*num_y*channel, filter_size^2]
        
        # total input citcuits: N * num_x * num_y * channel
        circuit_inputs = tf.tile([self.circuit_tensor], [tf.shape(inputs)[0], 1])
        circuit_inputs = tf.reshape(circuit_inputs, shape=[-1])
        tf.fill([tf.shape(inputs)[0]*self.num_x*self.num_y, 1], 1)
        outputs = []
        for i in range(self.depth):
            controller = tf.tile(self.kernel[i], [tf.shape(inputs)[0]*self.num_x*self.num_y, 1])
            outputs.append(self.single_depth_QCNN(stack_set, controller, circuit_inputs))
            # shape: [N, num_x, num_y] 
            
        output_tensor = tf.stack(outputs, axis=3)
        output_tensor = tf.math.acos(tf.clip_by_value(output_tensor, -1+1e-5, 1-1e-5)) / np.pi
        # output_tensor = tf.clip_by_value(tf.math.acos(output_tensor)/np.pi, -1, 1)
        return self.activation(output_tensor)
          
    def single_depth_QCNN(self, input_data, controller, circuit_inputs):
        """
        make QCNN for 1 channel only
        """
        # input shape: [N*num_x*num_y*channel, filter_size^2]
        # controller shape: [N*num_x*num_y*channel, len(learning_params)]
        input_data = tf.concat([input_data, controller], 1)
        # input_data shape: [N*num_x*num_y*channel, len(learning_params)]
        QCNN_output = tfq.layers.Expectation()(circuit_inputs, 
                                               symbol_names=self.params,
                                               symbol_values=input_data,
                                               operators=self.op)
        # QCNN_output shape: [N*num_x*num_y*channel]
        QCNN_output = tf.reshape(QCNN_output, shape=[-1, self.num_x, self.num_y, self.channel])
        return tf.math.reduce_sum(QCNN_output, 3)

def plot_loss_curves(fc_loss, cnn_loss, qcnn_loss):
    fig = plt.figure()
    plt.plot(np.arange(len(fc_loss)) + 1, fc_loss, "go-", label="fully-connected")
    plt.plot(np.arange(len(cnn_loss)) + 1, cnn_loss, "rs-", label="CNN")
    plt.plot(np.arange(len(qcnn_loss)) + 1, qcnn_loss, "b^-", label="QCNN")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, 50, 0, 1])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Test set loss")
    plt.grid(True)
    fig.savefig('loss.png', dpi=300)
    
def plot_loss_curves_small(fc_loss, cnn_loss, qcnn_loss):
    fig = plt.figure()
    plt.plot(np.arange(len(fc_loss))[::2] + 1, fc_loss[::2], "go-", label="fully-connected")
    plt.plot(np.arange(len(cnn_loss))[::2] + 1, cnn_loss[::2], "rs-", label="CNN")
    plt.plot(np.arange(len(qcnn_loss))[::2] + 1, qcnn_loss[::2], "b^-", label="QCNN")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, 50, 0, 1])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Test set loss")
    plt.grid(True)
    fig.savefig('loss_small.png', dpi=300)

def plot_acc_curves(fc_acc, cnn_acc, qcnn_acc):
    fig = plt.figure()
    plt.plot(np.arange(len(fc_acc)) + 1, fc_acc, "go-", label="fully-connected")
    plt.plot(np.arange(len(cnn_acc)) + 1, cnn_acc, "rs-", label="CNN")
    plt.plot(np.arange(len(qcnn_acc)) + 1, qcnn_acc, "b^-", label="QCNN")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.grid()
    plt.axis([1, 50, 0.8, 1])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Test set accuracy")
    fig.savefig('accuracy.png', dpi=300)
    
def plot_acc_curves_small(fc_acc, cnn_acc, qcnn_acc):
    fig = plt.figure()
    plt.plot(np.arange(len(fc_acc))[::2] + 1, fc_acc[::2], "go-", label="fully-connected")
    plt.plot(np.arange(len(cnn_acc))[::2] + 1, cnn_acc[::2], "rs-", label="CNN")
    plt.plot(np.arange(len(qcnn_acc))[::2] + 1, qcnn_acc[::2], "b^-", label="QCNN")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.grid()
    plt.axis([1, 50, 0.8, 1])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Test set accuracy")
    fig.savefig('accuracy_small.png', dpi=300)

def main(config):
    
    BATCH_SIZE = config.batch_size
    EPOCHS = config.epochs
    NUM_CLASSES = config.num_classes

    if config.dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif config.dataset == 'fmnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    elif config.dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = load_cifair10()
    elif config.dataset == 'cifar100':
        (x_train, y_train), (x_test, y_test) = load_cifair100()
    else:
        (x_train, y_train), (x_test, y_test) = np.load(config.dataset+'.npy', allow_pickle= True)

    if config.dataset == 'mnist' or config.dataset == 'fmnist':
        # Rescale the images from [0,255] to the [0.0,1.0] range.
        x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
    elif config.dataset == 'cifar10' or config.dataset == 'cifar100' or config.dataset == 'isic' or config.dataset == 'eurosat' or config.dataset =='large_intestine':
        x_train = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in x_train])
        x_test = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in x_test])
        x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
    else:
        x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

    print('***************************************************************************')
    print("·Number of original training examples:", len(x_train))
    print("·Number of original test examples:", len(x_test))
    print('***************************************************************************')
    print('\n')

    x_train = tf.image.resize(x_train[:], (10,10)).numpy()
    x_test = tf.image.resize(x_test[:], (10,10)).numpy()

    y_train = y_train[:]
    y_test = y_test[:]

    width = np.shape(x_train)[1]
    height = np.shape(x_train)[2]
    channels = np.shape(x_train)[3]

    # Fully Connected Layer Only
    fc_model = tf.keras.models.Sequential()
    fc_model.add(tf.keras.layers.Flatten(input_shape=(width,height,channels)))
    fc_model.add(tf.keras.layers.Dense(32, activation='relu'))
    fc_model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))
    fc_model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

    fc_history = fc_model.fit(x_train,
                              y_train,
                              steps_per_epoch=int(len(x_train)/BATCH_SIZE),
                              validation_data=(x_test, y_test),
                              epochs=EPOCHS,
                              batch_size=BATCH_SIZE)

    # Convolutional Layer
    cnn_model = tf.keras.models.Sequential()
    cnn_model.add(tf.keras.layers.Conv2D(8, (2, 2), activation='relu', input_shape=(width, height, channels)))
    cnn_model.add(tf.keras.layers.Flatten())
    cnn_model.add(tf.keras.layers.Dense(32, activation='relu'))
    cnn_model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))
    cnn_model.summary()
    cnn_model.compile(optimizer=tf.optimizers.Adam(),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    cnn_history = cnn_model.fit(x_train,
                                y_train,
                                steps_per_epoch=int(len(x_train)/BATCH_SIZE),
                                validation_data=(x_test, y_test),
                                epochs=EPOCHS,
                                batch_size=BATCH_SIZE)
    
    # Make Quantum Convolution Layer
    qcnn_model = tf.keras.models.Sequential()
    qcnn_model.add(QConv(filter_size=2, depth=8, activation='relu', name='qconv1', input_shape=(width, height, channels)))
    qcnn_model.add(tf.keras.layers.Flatten())
    qcnn_model.add(tf.keras.layers.Dense(32, activation='relu'))
    qcnn_model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'))
    qcnn_model.summary()
    qcnn_model.compile(optimizer=tf.keras.optimizers.Adam(),
                       loss='sparse_categorical_crossentropy', # sparse_categorical_crossentropy
                       metrics=['accuracy'])

    qcnn_history = qcnn_model.fit(x_train,
                                  y_train,
                                  steps_per_epoch=int(len(x_train)/BATCH_SIZE),
                                  validation_data=(x_test, y_test),
                                  epochs=EPOCHS,
                                  batch_size=BATCH_SIZE)

    test_loss, test_acc = qcnn_model.evaluate(x_test, y_test, verbose=2)

    
    plot_loss_curves(fc_history.history['val_loss'], cnn_history.history['val_loss'], qcnn_history.history['val_loss'])
    plot_loss_curves_small(fc_history.history['val_loss'], cnn_history.history['val_loss'], qcnn_history.history['val_loss'])
    plot_acc_curves(fc_history.history['val_accuracy'], cnn_history.history['val_accuracy'], qcnn_history.history['val_accuracy'])
    plot_acc_curves_small(fc_history.history['val_accuracy'], cnn_history.history['val_accuracy'], qcnn_history.history['val_accuracy'])
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantum Convolutional Neural Networks')

    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset Name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs')
    parser.add_argument('--num_classes', type=int, default=7, help='Num Classes')
    
    config = parser.parse_args()
    main(config)
