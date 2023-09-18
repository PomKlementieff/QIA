# Author: Sung-Wook Park
# Date: 16 Jun 2022
# Last updated: 18 Sep 2023
# --- Ad hoc ---

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

from emnist import extract_training_samples, extract_test_samples
from nets.basic_nets import FC, CNN, QCNN
from nets.inception import Naive, Dimension_Reductions
from nets.resnet import Resnet
from sklearn.metrics import classification_report, f1_score,  precision_score, recall_score, precision_recall_fscore_support
from tensorflow.keras.models import save_model
from tf_explain.core.grad_cam import GradCAM
from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity

np.random.seed(42)

def exploring_dataset(Y_test, Y_pred):
    _, _, _, support = np.asarray(precision_recall_fscore_support(Y_test, Y_pred)).astype(int)
    support_dict = {str(i): [support[i]] for i in range(len(support))}

    support_len = len(support)
    support_max = max(support)
    
    df = pd.DataFrame(support_dict)

    fig, ax = plt.subplots()
    ax = sns.barplot(data=df, orient='h', palette='RdPu') 
    
    bbox = dict(boxstyle='square', facecolor='White', pad=0.3)

    plt.grid(axis='both', color='black', alpha=0.3, linestyle=':')
    plt.title('Class distribution EMNIST-{} test'.format(config.dataset))
    plt.xlabel('Frequency count')
    plt.yticks(np.arange(support_len), [i for i in range(support_len)])
    plt.ylabel('Class')
    plt.xticks(range(0, int(np.ceil(support_max/1000)*1000)+1, 1000))
    plt.tight_layout()
    plt.show()
    
def plot_example_images(x_train, y_train):
    _, indices = np.unique(y_train, return_index=True)

    num_row = 2
    num_column = round(len(indices)/2)

    fig, axes = plt.subplots(num_row, num_column, figsize=(1.5*num_column,2*num_row))
    fig.suptitle('Example images', fontsize=16)
    for i in range(len(indices)):
        ax = axes[i//num_column, i%num_column]
        ax.imshow(x_train[indices[i]], cmap=plt.cm.binary)
        if config.dataset == 'fmnist':
            class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
            ax.set_title('{}'.format(class_names[y_train[indices[i]]]))
        else:
            ax.set_title('Label: {}'.format(y_train[indices[i]]))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(config, model_name, Y_test, Y_pred):
    con_mat = tf.math.confusion_matrix(labels=Y_test,
                                       predictions=Y_pred).numpy() # Confusion matrix
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm)
    
    plt.figure() # Set Figure
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.RdPu, vmin=0, vmax=1) # The best cmap list=['PuRd', 'RdPu', 'BuPu']
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.title('Normalized confusion matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix({}_{}).png'.format(model_name, config.dataset))
    plt.close()

def run_explainer(config, model, model_name, x_test, y_test):
    g_explainer = GradCAM()
    
    _, indices = np.unique(y_test, return_index=True)

    num_row = 2
    num_column = round(len(indices)/2)

    fig, axes = plt.subplots(num_row, num_column, figsize=(1.5*num_column,2*num_row))
    fig.suptitle('Results of running Grad CAM with a trained {}'.format(model_name), fontsize=16)
    for i in range(len(indices)):
        ax = axes[i//num_column, i%num_column]
        data = ([x_test[indices[i]]], None)
        grid = g_explainer.explain(data, model, y_test[indices[i]])
        im=ax.imshow(grid, vmin=0, vmax=1)
        #plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if config.dataset == 'fmnist':
            class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
            ax.set_title('{}'.format(class_names[y_test[indices[i]]]))
        else:
            ax.set_title('Label: {}'.format(y_test[indices[i]]))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    #plt.show()
    plt.savefig('Results of running Grad CAM with a trained {}'.format(model_name))
    plt.close()

    o_explainer = OcclusionSensitivity()

    patch_size = 2

    fig, axes = plt.subplots(num_row, num_column, figsize=(1.5*num_column,2*num_row))
    fig.suptitle('Results of running Occlusion with a trained {}'.format(model_name), fontsize=16)
    for i in range(len(indices)):
        ax = axes[i//num_column, i%num_column]
        data = ([x_test[indices[i]]], None)
        grid = o_explainer.explain(data, model, y_test[indices[i]], patch_size)
        ax.imshow(grid, vmin=0, vmax=1)
        if config.dataset == 'fmnist':
            class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
            ax.set_title('{}'.format(class_names[y_test[indices[i]]]))
        else:
            ax.set_title('Label: {}'.format(y_test[indices[i]]))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    #plt.show()
    plt.savefig('Results of running Occlusion with a trained {}'.format(model_name))
    plt.close()

def write_test_results(config, model, test_acc, test_loss, Y_test, Y_pred):
    test_results = classification_report(Y_test, Y_pred)

    with open('test_results({}_{}).txt'.format(model, config.dataset), 'w') as f:
        f.write('########### Test Results of {} Model in {} Dataset ###########\n\n'.format(model, config.dataset))
        f.write('***************************************************************************\n')
        f.write('·Test Accuracy: {:.4f}\n'.format(test_acc))
        f.write('·Test Loss: {:.4f}\n'.format(test_loss))
        f.write('·Micro Average of Precision Score: {:.4f}\n'.format(precision_score(Y_test, Y_pred, average='micro')))
        f.write('·Micro Average of Recall Score: {:.4f}\n'.format(recall_score(Y_test, Y_pred, average='micro')))
        f.write('·Micro Average of F1 Score: {:.4f}\n'.format(f1_score(Y_test, Y_pred, average='micro')))
        f.write('·Macro Average of Precision Score: {:.4f}\n'.format(precision_score(Y_test, Y_pred, average='macro')))
        f.write('·Macro Average of Recall Score: {:.4f}\n'.format(recall_score(Y_test, Y_pred, average='macro')))
        f.write('·Macro Average of F1 Score: {:.4f}\n'.format(f1_score(Y_test, Y_pred, average='macro')))
        f.write('·Weighted Average of Precision Score: {:.4f}\n'.format(precision_score(Y_test, Y_pred, average='weighted')))
        f.write('·Weighted Average of Recall Score: {:.4f}\n'.format(recall_score(Y_test, Y_pred, average='weighted')))
        f.write('·Weighted Average of F1 Score: {:.4f}\n'.format(f1_score(Y_test, Y_pred, average='weighted')))
        f.write('***************************************************************************\n\n')
        f.write('########## Classification Report of {} Model in {} Dataset ###########\n\n'.format(model, config.dataset))
        f.write(test_results)

def main(config):
    
    BATCH_SIZE = config.batch_size
    EPOCHS = config.epochs

    if config.dataset == 'byclass': # unbalanced classes
        x_train, y_train = extract_training_samples('byclass')
        x_test, y_test = extract_test_samples('byclass')
        UNITS = 256
        NUM_CLASSES = 62
    elif config.dataset == 'bymerge': # unbalanced classes
        x_train, y_train = extract_training_samples('bymerge')
        x_test, y_test = extract_test_samples('bymerge')
        UNITS = 128
        NUM_CLASSES = 47
    elif config.dataset == 'balanced': # balanced classes
        x_train, y_train = extract_training_samples('balanced')
        x_test, y_test = extract_test_samples('balanced')
        UNITS = 128
        NUM_CLASSES = 47
    elif config.dataset == 'letters': # balanced classes
        x_train, y_train = extract_training_samples('letters')
        x_test, y_test = extract_test_samples('letters')
        UNITS = 64
        NUM_CLASSES = 37
    elif config.dataset == 'digits': # balanced classes
        x_train, y_train = extract_training_samples('digits')
        x_test, y_test = extract_test_samples('digits')
        UNITS = 32
        NUM_CLASSES = 10
    elif config.dataset == 'fmnist': # balanced classes
        fashion_mnist = tf.keras.datasets.fashion_mnist
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        UNITS = 32
        NUM_CLASSES = 10
    else: # balanced classes
        x_train, y_train = extract_training_samples('mnist')
        x_test, y_test = extract_test_samples('mnist')
        UNITS = 32
        NUM_CLASSES = 10

    # Rescale the images from [0,255] to the [0.0,1.0] range.
    x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

    print('***************************************************************************')
    print("·Number of original training examples:", len(x_train))
    print("·Number of original test examples:", len(x_test))
    print('***************************************************************************')
    print('\n')

    plot_example_images(x_train, y_train)

    x_train = tf.image.resize(x_train[:], (10,10)).numpy()
    x_test = tf.image.resize(x_test[:], (10,10)).numpy()

    y_train = y_train[:]
    y_test = y_test[:]

    width = np.shape(x_train)[1]
    height = np.shape(x_train)[2]
    channels = np.shape(x_train)[3]
    IMG_SHAPE = (width, height, channels)

    # Fully-connected Layer
    fc_model = FC(IMG_SHAPE,UNITS,NUM_CLASSES)
    fc_history = fc_model.fit(x_train,
                              y_train,
                              steps_per_epoch=int(len(x_train)/BATCH_SIZE),
                              validation_data=(x_test, y_test),
                              epochs=EPOCHS,
                              batch_size=BATCH_SIZE)
    
    Y_pred = fc_model.predict(x_test, verbose=1)
    Y_pred = np.argmax(Y_pred, 1) # Decode predicted labels
    Y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
    Y_test = np.argmax(Y_test, 1) # # Decode labels
    
    exploring_dataset(Y_test, Y_pred)
    plot_confusion_matrix(config, 'FC', Y_test, Y_pred)
    
    test_loss, test_acc = fc_model.evaluate(x_test, y_test, verbose=1)
    write_test_results(config, 'FC', test_acc, test_loss, Y_test, Y_pred)

    write_FC_loss = pd.DataFrame(fc_history.history['val_loss'], columns=['FC'])
    write_FC_loss.to_csv('FC_loss({}).csv'.format(config.dataset), index=False)

    write_FC_acc = pd.DataFrame(fc_history.history['val_accuracy'], columns=['FC'])
    write_FC_acc.to_csv('FC_acc({}).csv'.format(config.dataset), index=False)

    save_model(model=fc_model, filepath='FC_{}.h5'.format(config.dataset), overwrite=True, include_optimizer=True)
    
    # Basic Convolutional Neural Network
    cnn_model = CNN(IMG_SHAPE,UNITS,NUM_CLASSES)
    cnn_history = cnn_model.fit(x_train,
                                y_train,
                                steps_per_epoch=int(len(x_train)/BATCH_SIZE),
                                validation_data=(x_test, y_test),
                                epochs=EPOCHS,
                                batch_size=BATCH_SIZE)
    
    run_explainer(config, cnn_model, 'CNN', x_test, y_test)
    Y_pred = cnn_model.predict(x_test, verbose=1)
    Y_pred = np.argmax(Y_pred, 1)
    plot_confusion_matrix(config, 'CNN', Y_test, Y_pred)

    test_loss, test_acc = cnn_model.evaluate(x_test, y_test, verbose=1)
    write_test_results(config, 'CNN', test_acc, test_loss, Y_test, Y_pred)

    write_CNN_loss = pd.DataFrame(cnn_history.history['val_loss'], columns=['CNN'])
    write_CNN_loss.to_csv('CNN_loss({}).csv'.format(config.dataset), index=False)

    write_CNN_acc = pd.DataFrame(cnn_history.history['val_accuracy'], columns=['CNN'])
    write_CNN_acc.to_csv('CNN_acc({}).csv'.format(config.dataset), index=False)

    save_model(model=cnn_model, filepath='CNN_{}.h5'.format(config.dataset), overwrite=True, include_optimizer=True)
    
    # Vainilla Quantum Convolutional Neural Network
    qcnn_model = QCNN(IMG_SHAPE,UNITS,NUM_CLASSES)
    qcnn_history = qcnn_model.fit(x_train,
                                  y_train,
                                  steps_per_epoch=int(len(x_train)/BATCH_SIZE),
                                  validation_data=(x_test, y_test),
                                  epochs=EPOCHS,
                                  batch_size=BATCH_SIZE)
    
    run_explainer(config, qcnn_model, 'QCNN', x_test, y_test)
    Y_pred = qcnn_model.predict(x_test, verbose=1)
    Y_pred = np.argmax(Y_pred, 1)
    plot_confusion_matrix(config, 'QCNN', Y_test, Y_pred)

    test_loss, test_acc = qcnn_model.evaluate(x_test, y_test, verbose=1)
    write_test_results(config, 'QCNN', test_acc, test_loss, Y_test, Y_pred)

    write_QCNN_loss = pd.DataFrame(qcnn_history.history['val_loss'], columns=['QCNN'])
    write_QCNN_loss.to_csv('QCNN_loss({}).csv'.format(config.dataset), index=False)

    write_QCNN_acc = pd.DataFrame(qcnn_history.history['val_accuracy'], columns=['QCNN'])
    write_QCNN_acc.to_csv('QCNN_acc({}).csv'.format(config.dataset), index=False)

    # Quantum Inception Network - Naive
    naive_model = Naive(IMG_SHAPE,UNITS,NUM_CLASSES)
    naive_history = naive_model.fit(x_train,
                                    y_train,
                                    steps_per_epoch=int(len(x_train)/BATCH_SIZE),
                                    validation_data=(x_test, y_test),
                                    epochs=EPOCHS,
                                    batch_size=BATCH_SIZE)
    
    run_explainer(config, naive_model, 'Proposed(Naive)', x_test, y_test)
    Y_pred = naive_model.predict(x_test, verbose=1)
    Y_pred = np.argmax(Y_pred, 1)
    plot_confusion_matrix(config, 'Naive', Y_test, Y_pred)
    
    test_loss, test_acc = naive_model.evaluate(x_test, y_test, verbose=1)
    write_test_results(config, 'Naive', test_acc, test_loss, Y_test, Y_pred)

    write_Naive_loss = pd.DataFrame(naive_history.history['val_loss'], columns=['Naive'])
    write_Naive_loss.to_csv('Naive_loss({}).csv'.format(config.dataset), index=False)

    write_Naive_acc = pd.DataFrame(naive_history.history['val_accuracy'], columns=['Naive'])
    write_Naive_acc.to_csv('Naive_acc({}).csv'.format(config.dataset), index=False)

    # Quantum Inception Network - Dimension_Reductions
    dr_model = Dimension_Reductions(IMG_SHAPE,UNITS,NUM_CLASSES)
    dr_history = dr_model.fit(x_train,
                              y_train,
                              steps_per_epoch=int(len(x_train)/BATCH_SIZE),
                              validation_data=(x_test, y_test),
                              epochs=EPOCHS,
                              batch_size=BATCH_SIZE)
    
    run_explainer(config, dr_model, 'Proposed(DR)', x_test, y_test)
    Y_pred = dr_model.predict(x_test, verbose=1)
    Y_pred = np.argmax(Y_pred, 1)
    plot_confusion_matrix(config, 'Dimension_Reductions', Y_test, Y_pred)
    
    test_loss, test_acc = dr_model.evaluate(x_test, y_test, verbose=1)
    write_test_results(config, 'Dimension_Reductions', test_acc, test_loss, Y_test, Y_pred)

    write_Dimension_Reductions_loss = pd.DataFrame(dr_history.history['val_loss'], columns=['Dimension_Reductions'])
    write_Dimension_Reductions_loss.to_csv('Dimension_Reductions_loss({}).csv'.format(config.dataset), index=False)

    write_Dimension_Reductions_acc = pd.DataFrame(dr_history.history['val_accuracy'], columns=['Dimension_Reductions'])
    write_Dimension_Reductions_acc.to_csv('Dimension_Reductions_acc({}).csv'.format(config.dataset), index=False)

    # ResNet
    resnet_model = Resnet(IMG_SHAPE,UNITS,NUM_CLASSES)
    resnet_history = resnet_model.fit(x_train,
                              y_train,
                              steps_per_epoch=int(len(x_train)/BATCH_SIZE),
                              validation_data=(x_test, y_test),
                              epochs=EPOCHS,
                              batch_size=BATCH_SIZE)
    
    Y_pred = resnet_model.predict(x_test, verbose=1)
    Y_pred = np.argmax(Y_pred, 1)
    Y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
    Y_test = np.argmax(Y_test, 1) # # Decode labels
    plot_confusion_matrix(config, 'ResNet', Y_test, Y_pred)
    
    test_loss, test_acc = resnet_model.evaluate(x_test, y_test, verbose=1)
    write_test_results(config, 'ResNet', test_acc, test_loss, Y_test, Y_pred)

    write_ResNet_loss = pd.DataFrame(resnet_history.history['val_loss'], columns=['ResNet'])
    write_ResNet_loss.to_csv('ResNet_loss({}).csv'.format(config.dataset), index=False)

    write_ResNet_acc = pd.DataFrame(resnet_history.history['val_accuracy'], columns=['ResNet'])
    write_ResNet_acc.to_csv('ResNet_acc({}).csv'.format(config.dataset), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quantum CNN')

    parser.add_argument('--batch_size', type=int, default=32, help='size of batch')
    parser.add_argument('--dataset', type=str, default='mnist', help='name of dataset')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    
    config = parser.parse_args()
    main(config)