# Author: Sung-Wook Park
# Date: 15 Jun 2022
# Last updated: 2 Jul 2022
# --- Ad hoc ---

# Update package resources to account for version changes.
import importlib, pkg_resources
importlib.reload(pkg_resources)

import argparse
import cirq
import matplotlib.pyplot as plt
import numpy as np
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq

from cirq.contrib.svg import SVGCircuit
from matplotlib.legend_handler import HandlerLine2D

np.random.seed(1234)

def filter_03(x, y):
        keep = (y == 0) | (y == 3)
        x, y = x[keep], y[keep]
        y = y == 0
        return x,y

def truncate_x(x_train, x_test, n_components=10):
    """Perform PCA on image dataset keeping the top `n_components` components."""
    n_points_train = tf.gather(tf.shape(x_train), 0)
    n_points_test = tf.gather(tf.shape(x_test), 0)

    # Flatten to 1D
    x_train = tf.reshape(x_train, [n_points_train, -1])
    x_test = tf.reshape(x_test, [n_points_test, -1])

    # Normalize.
    feature_mean = tf.reduce_mean(x_train, axis=0)
    x_train_normalized = x_train - feature_mean
    x_test_normalized = x_test - feature_mean

    # Truncate.
    e_values, e_vectors = tf.linalg.eigh(
        tf.einsum('ji,jk->ik', x_train_normalized, x_train_normalized))
    return tf.einsum('ij,jk->ik', x_train_normalized, e_vectors[:,-n_components:]), \
        tf.einsum('ij,jk->ik', x_test_normalized, e_vectors[:, -n_components:])

def single_qubit_wall(qubits, rotations):
    """Prepare a single qubit X,Y,Z rotation wall on `qubits`."""
    wall_circuit = cirq.Circuit()
    for i, qubit in enumerate(qubits):
        for j, gate in enumerate([cirq.X, cirq.Y, cirq.Z]):
            wall_circuit.append(gate(qubit) ** rotations[i][j])

    return wall_circuit

def v_theta(qubits):
    """Prepares a circuit that generates V(\theta)."""
    ref_paulis = [
        cirq.X(q0) * cirq.X(q1) + \
        cirq.Y(q0) * cirq.Y(q1) + \
        cirq.Z(q0) * cirq.Z(q1) for q0, q1 in zip(qubits, qubits[1:])
    ]
    exp_symbols = list(sympy.symbols('ref_0:'+str(len(ref_paulis))))
    return tfq.util.exponential(ref_paulis, exp_symbols), exp_symbols

def prepare_pqk_circuits(qubits, classical_source, n_trotter=10):
    """Prepare the pqk feature circuits around a dataset."""
    n_qubits = len(qubits)
    n_points = len(classical_source)

    # Prepare random single qubit rotation wall.
    random_rots = np.random.uniform(-2, 2, size=(n_qubits, 3))
    initial_U = single_qubit_wall(qubits, random_rots)

    # Prepare parametrized V
    V_circuit, symbols = v_theta(qubits)
    exp_circuit = cirq.Circuit(V_circuit for t in range(n_trotter))
    
    # Convert to `tf.Tensor`
    initial_U_tensor = tfq.convert_to_tensor([initial_U])
    initial_U_splat = tf.tile(initial_U_tensor, [n_points])

    full_circuits = tfq.layers.AddCircuit()(
        initial_U_splat, append=exp_circuit)
    # Replace placeholders in circuits with values from `classical_source`.
    return tfq.resolve_parameters(
        full_circuits, tf.convert_to_tensor([str(x) for x in symbols]),
        tf.convert_to_tensor(classical_source*(n_qubits/3)/n_trotter))

def get_pqk_features(qubits, data_batch):
    """Get PQK features based on above construction."""
    ops = [[cirq.X(q), cirq.Y(q), cirq.Z(q)] for q in qubits]
    ops_tensor = tf.expand_dims(tf.reshape(tfq.convert_to_tensor(ops), -1), 0)
    batch_dim = tf.gather(tf.shape(data_batch), 0)
    ops_splat = tf.tile(ops_tensor, [batch_dim, 1])
    exp_vals = tfq.layers.Expectation()(data_batch, operators=ops_splat)
    rdm = tf.reshape(exp_vals, [batch_dim, len(qubits), -1])
    return rdm

def compute_kernel_matrix(vecs, gamma):
    """Computes d[i][j] = e^ -gamma * (vecs[i] - vecs[j]) ** 2 """
    scaled_gamma = gamma / (
        tf.cast(tf.gather(tf.shape(vecs), 1), tf.float32) * tf.math.reduce_std(vecs))
    return scaled_gamma * tf.einsum('ijk->ij',(vecs[:,None,:] - vecs) ** 2)

def get_spectrum(datapoints, gamma=1.0):
    """Compute the eigenvalues and eigenvectors of the kernel of datapoints."""
    KC_qs = compute_kernel_matrix(datapoints, gamma)
    S, V = tf.linalg.eigh(KC_qs)
    S = tf.math.abs(S)
    return S, V

def get_stilted_dataset(S, V, S_2, V_2, lambdav=1.1):
    """Prepare new labels that maximize geometric distance between kernels."""
    S_diag = tf.linalg.diag(S ** 0.5)
    S_2_diag = tf.linalg.diag(S_2 / (S_2 + lambdav) ** 2)
    scaling = S_diag @ tf.transpose(V) @ \
                V_2 @ S_2_diag @ tf.transpose(V_2) @ \
                V @ S_diag

    # Generate new lables using the largest eigenvector.
    _, vecs = tf.linalg.eig(scaling)
    new_labels = tf.math.real(
        tf.einsum('ij,j->i', tf.cast(V @ S_diag, tf.complex64), vecs[-1])).numpy()
    # Create new labels and add some small amount of noise.
    final_y = new_labels > np.median(new_labels)
    noisy_y = (final_y ^ (np.random.uniform(size=final_y.shape) > 0.95))
    return noisy_y

# docs_infra: no_execute
def create_pqk_model(qubits):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(32, activation='sigmoid', input_shape=[len(qubits) * 3,]))
    model.add(tf.keras.layers.Dense(16, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(1))
    return model

# docs_infra: no_execute
def create_fair_classical_model(DATASET_DIM):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(32, activation='sigmoid', input_shape=[DATASET_DIM,]))
    model.add(tf.keras.layers.Dense(16, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(1))
    return model

# Defining Helper Functions
def update(handle, orig):
    handle.update_from(orig)
    handle.set_linewidth(12)
    
def plot_acc(pqk_history, classical_history):
    # docs_infra: no_execute
    plt.rcParams['figure.figsize'] = [12, 5]
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(classical_history.history['accuracy'], color='Salmon', label='Train Classical', linewidth=2)
    axes[0].plot(pqk_history.history['accuracy'], color='Gold', label='Train Quantum', linewidth=2)
    axes[0].set_xlabel('Epoch\n\n(a)')
    axes[0].set_xticks([0,200,400,600,800,1000])
    axes[0].set_xlim([0, 1000])
    axes[0].set_ylabel('Accuracy')
    axes[0].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axes[0].set_ylim([0.0, 1.0])
    axes[0].grid(axis='both', color='black', alpha=0.3, linestyle=':')
    
    axes[1].plot(classical_history.history['val_accuracy'], color='Orchid', label='Test Classical', linewidth=2)
    axes[1].plot(pqk_history.history['val_accuracy'], color='Orange', label='Test Quantum', linewidth=2)
    axes[1].set_xlabel('Epoch\n\n(b)')
    axes[1].set_xticks([0,200,400,600,800,1000])
    axes[1].set_xlim([0, 1000])
    axes[1].set_ylabel('Accuracy')
    axes[1].set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axes[1].set_ylim([0.0, 1.0])
    axes[1].grid(axis='both', color='black', alpha=0.3, linestyle=':')
    
    lines = []
    labels = []
    
    for ax in fig.axes:
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)

    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), fontsize=12, frameon=False, ncol=4, handler_map={plt.Line2D : HandlerLine2D(update_func=update)})

    plt.tight_layout(pad=2.58)
    plt.show()
    plt.close()
    
def plot_loss(pqk_history, classical_history):
    #docs_infra: no_execute
    plt.rcParams['figure.figsize'] = [12, 5]
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(classical_history.history['loss'], color='Salmon', label='Train Classical', linewidth=2)
    axes[0].plot(pqk_history.history['loss'], color='Gold', label='Train Quantum', linewidth=2)
    axes[0].set_xlabel('Epoch\n\n(a)')
    axes[0].set_xticks([0,200,400,600,800,1000])
    axes[0].set_xlim([0, 1000])
    axes[0].set_ylabel('Loss')
    axes[0].grid(axis='both', color='black', alpha=0.3, linestyle=':')
    
    axes[1].plot(classical_history.history['val_loss'], color='Orchid', label='Test Classical', linewidth=2)
    axes[1].plot(pqk_history.history['val_loss'], color='Orange', label='Test Quantum', linewidth=2)
    axes[1].set_xlabel('Epoch\n\n(b)')
    axes[1].set_xticks([0,200,400,600,800,1000])
    axes[1].set_xlim([0, 1000])
    axes[1].set_ylabel('Loss')
    axes[1].grid(axis='both', color='black', alpha=0.3, linestyle=':')
    
    lines = []
    labels = []
    
    for ax in fig.axes:
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)

    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), fontsize=12, frameon=False, ncol=4, handler_map={plt.Line2D : HandlerLine2D(update_func=update)})

    plt.tight_layout(pad=2.58)
    plt.show()
    plt.close()

def main(config):
    
    if config.dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif config.dataset == 'fmnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    elif config.dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    
    # Rescale the images from [0,255] to the [0.0,1.0] range.
    x_train, x_test = x_train/255.0, x_test/255.0
    
    print('*******************************************************')
    print("·Number of original training examples:", len(x_train))
    print("·Number of original test examples:", len(x_test))
    print('*******************************************************')
    print('\n')
        
    x_train, y_train = filter_03(x_train, y_train)
    x_test, y_test = filter_03(x_test, y_test)
    
    print('*******************************************************')
    print("·Number of filtered training examples:", len(x_train))
    print("·Number of filtered test examples:", len(x_test))
    print('*******************************************************')
    print('\n')
    
    DATASET_DIM = 10
    x_train, x_test = truncate_x(x_train, x_test, n_components=DATASET_DIM)
    print('*******************************************************')
    print(f'·New datapoint dimension:', len(x_train[0]))
    print('*******************************************************')
    print('\n')
    
    N_TRAIN = 1000
    N_TEST = 200
    x_train, x_test = x_train[:N_TRAIN], x_test[:N_TEST]
    y_train, y_test = y_train[:N_TRAIN], y_test[:N_TEST]
    
    print('*******************************************************')
    print("·New number of training examples:", len(x_train))
    print("·New number of test examples:", len(x_test))
    print('*******************************************************')
    print('\n')
    
    SVGCircuit(single_qubit_wall(cirq.GridQubit.rect(1,4), np.random.uniform(size=(4, 3))))
    
    test_circuit, test_symbols = v_theta(cirq.GridQubit.rect(1, 2))
    print('*******************************************************')
    print(f'·Symbols found in circuit:{test_symbols}')
    print('*******************************************************')
    print('\n')
    SVGCircuit(test_circuit)
    
    qubits = cirq.GridQubit.rect(1, DATASET_DIM + 1)
    q_x_train_circuits = prepare_pqk_circuits(qubits, x_train)
    q_x_test_circuits = prepare_pqk_circuits(qubits, x_test)
    
    x_train_pqk = get_pqk_features(qubits, q_x_train_circuits)
    x_test_pqk = get_pqk_features(qubits, q_x_test_circuits)
    print('*******************************************************')
    print('·New PQK training dataset has shape:', x_train_pqk.shape)
    print('·New PQK testing dataset has shape:', x_test_pqk.shape)
    print('*******************************************************')
    print('\n')
    
    S_pqk, V_pqk = get_spectrum(tf.reshape(tf.concat([x_train_pqk, x_test_pqk], 0), [-1, len(qubits) * 3]))
    
    S_original, V_original = get_spectrum(tf.cast(tf.concat([x_train, x_test], 0), tf.float32), gamma=0.005)
    
    print('*******************************************************')
    print('·Eigenvectors of pqk kernel matrix:', V_pqk)
    print('·Eigenvectors of original kernel matrix:', V_original)
    print('*******************************************************')
    print('\n')
    
    y_relabel = get_stilted_dataset(S_pqk, V_pqk, S_original, V_original)
    y_train_new, y_test_new = y_relabel[:N_TRAIN], y_relabel[N_TRAIN:]
    
    pqk_model = create_pqk_model(qubits)
    pqk_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
                      metrics=['accuracy'])
    pqk_model.summary()
    
    # docs_infra: no_execute
    pqk_history = pqk_model.fit(tf.reshape(x_train_pqk, [N_TRAIN, -1]),
                                           y_train_new,
                                           batch_size=32,
                                           epochs=1000,
                                           verbose=1,
                                           validation_data=(tf.reshape(x_test_pqk, [N_TEST, -1]), y_test_new))
            
    model = create_fair_classical_model(DATASET_DIM)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.03),
                  metrics=['accuracy'])
    model.summary()
    
    # docs_infra: no_execute
    classical_history = model.fit(x_train,
                                  y_train_new,
                                  batch_size=32,
                                  epochs=1000,
                                  verbose=1,
                                  validation_data=(x_test, y_test_new))
    
    plot_acc(pqk_history, classical_history)
    plot_loss(pqk_history, classical_history)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantum Data')
    
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset Name')
    
    config = parser.parse_args()
    main(config)
