#!/usr/bin/env python
# coding: utf-8

# In[1]:


# convlstm model
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
import time


# In[2]:


# Configure GPU options
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


# In[3]:

start_time = time.time()


# In[4]:


def load_and_process_data_10(number):

    base_path = '/50_ms'
    number_str = str(number)
    path_data_imu = os.path.join(base_path, number_str, 'IMU_Windows', f'{number}_Windows')
    path_data_ik = os.path.join(base_path, number_str, 'IK_Windows', f'{number}_Windows')
    # path_data_gon = os.path.join(base_path, number_str, 'GON_Windows', f'{number}_Windows')
    path_labels_ik = os.path.join(base_path, number_str, 'IK_Windows', f'{number}_y_labels')


#     files_data_gon = [
#                     GON csv files here
#                      ]
#
    files_data_imu = [
                      # IMU csv files here
                    ]
    
    files_data_ik = [
     #              IK joint angle csv files here
                                                                    ]
    

#     data_matrices_gon = [pd.read_csv(f"{path_data_gon}/{file}") for file in files_data_gon]
    data_matrices_imu = [pd.read_csv(f"{path_data_imu}/{file}") for file in files_data_imu]
    data_matrices_ik = [pd.read_csv(f"{path_data_ik}/{file}") for file in files_data_ik]
    

#     data_matrix_gon = np.dstack([matrix.values for matrix in data_matrices_gon])
    data_matrix_imu = np.dstack([matrix.values for matrix in data_matrices_imu])
    data_matrix_ik = np.dstack([matrix.values for matrix in data_matrices_ik])
    data_matrix = np.dstack([data_matrix_ik, data_matrix_imu])


    label_matrix = pd.read_csv(f"{path_labels_ik}/{number}_labels.csv")


    trainX, testX, trainy, testy = train_test_split(data_matrix, label_matrix, test_size=0.3, random_state=42, stratify=label_matrix)

    # zero-offset class values
    trainy = trainy - 1
    testy = testy - 1

    # one hot encode y
    trainy = to_categorical(trainy)
    testy = to_categorical(testy)

    print("trainX shape:", trainX.shape)
    print("testX shape:", testX.shape)
    print("trainy shape:", trainy.shape)
    print("testy shape:", testy.shape)
    print("trainX type is: ", type(trainX))
    print("trainy type is: ", type(trainy))
    print("testX type is: ", type(testX))
    print("testy type is: ", type(testy))
    print("trainX float type:", trainX.dtype)
    print("trainy float type:", trainy.dtype)
    print("testX float type:", testX.dtype)
    print("testy float type:", testy.dtype)

    return trainX, testX, trainy, testy



# In[10]:
# fit and evaluate a model
def evaluate_model(trainX, trainy, testX, testy):
    # define model
    verbose, epochs, batch_size =
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    # reshape into subsequences (samples, time steps, rows, cols, channels)
#     n_steps, n_length = 4, 50
    n_steps, n_length = 1, 10
    trainX = trainX.reshape((trainX.shape[0], n_steps, n_length, n_features))
    testX = testX.reshape((testX.shape[0], n_steps, n_length, n_features))
    # define model
    model = Sequential()
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'), input_shape=(None,n_length,n_features)))
    model.add(TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(100))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
#     model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    history = model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # calculate train loss
    train_loss = history.history['loss']
    # print training accuracy
    train_accuracy = history.history['accuracy'][-1]
    print("Training accuracy: ", history.history['accuracy'][-1])  # Print the training accuracy of the last epoch
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)

    # predict class for test set
    y_pred = model.predict(testX)
    # convert predictions from probabilities to class labels
    y_pred = np.argmax(y_pred, axis=1)
    # convert test set from one-hot encoding to class labels
    y_true = np.argmax(testy, axis=1)
    
    return model, accuracy, train_accuracy, train_loss, y_pred, y_true

# In[]:
def display_trainloss(train_loss, subject, r, base_save_path):
    plt.ion()
    fig = plt.figure()
    plt.plot(train_loss, label="train_loss")
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Cross-entropy loss')
    save_path_fig = os.path.join(base_save_path,
                                 f"{subject}/trainloss/",
                                 f'{subject}_50ms_CNNLSTM_IK_Hip_Flexion_WithAngvel_trainloss_{r+1}.png')
    # check if the path exists
    os.makedirs(os.path.dirname(save_path_fig), exist_ok=True)
    # save_path_fig = f"D:\\OneDrive - NJIT\\Research\\Dataset_Test\\Results\\50 ms\\IK_Test\\TripleIK\\AngleOnly\\Feb2024\\MLP\\{subject}_50ms_MLP_IK_Hip+Knee+Ankle_trainloss_{r+1}.png"
    plt.savefig(save_path_fig, format='png')
    plt.show()
    plt.close(fig)


# In[]:
def save_cm(cm, subject, r, base_save_path):
    labels = ['stand', 'walk', 'rampascend', 'rampdescend', 'stairascend', 'stairdescend']

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    save_path_csv = os.path.join(base_save_path,
                                 f"{subject}/cm_csv/",
                                 f'{subject}_cm_{r+1}.csv')

    os.makedirs(os.path.dirname(save_path_csv), exist_ok=True)

    cm_df.to_csv(save_path_csv, index=True)


# In[11]:
def display_confusion_matrix(cm, subject, accuracy, r, base_save_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = mpl.cm.get_cmap("rainbow")
    im = ax.imshow(cm, cmap)
    ax.set_xticks(np.arange(6))
    ax.set_yticks(np.arange(6))
    ax.set_xticklabels(('stand','walk','rampascend','rampdescend','stairascend','stairdescend'))
    ax.set_yticklabels(('stand','walk','rampascend','rampdescend','stairascend','stairdescend'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    for i in range(6):
        for j in range(6):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", color="w")
            
    ax.set_title("Subject: %s Accuracy: %.2f" % (subject, 100*accuracy))
    fig.tight_layout()

    # Save the figure to the specified path and format
    save_path_fig = os.path.join(base_save_path,
                                 f"{subject}/cm_fig/",
                                 f'{subject}_results_{r+1}.png')
    # check if the path exists
    os.makedirs(os.path.dirname(save_path_fig), exist_ok=True)
    plt.savefig(save_path_fig, format='png')
    plt.show()
    plt.close(fig)

# In[12]:


def display_percentage_confusion_matrix(cm, subject, accuracy, r, base_save_path):
    # draw percentage confusion matrix
    percentage_matrix = np.zeros(cm.shape)
    for i in range(cm.shape[0]):
        row_sum = np.sum(cm[i, :])
        if row_sum > 0:
            percentage_matrix[i, :] = cm[i, :] / row_sum * 100
            
    # Format the percentage matrix with 2 decimal places
    percentage_matrix = np.around(percentage_matrix, 2)

    # Plotting the matrix with a style similar to the one you uploaded
    with plt.style.context('classic'):
        plt.figure(figsize=(10, 8))
        # Set the background color to white
        fig = plt.gcf()
        fig.patch.set_facecolor('white')

        ax = sns.heatmap(percentage_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=False, linewidths=1, linecolor='white', annot_kws={"size": 14})
        plt.title(f'Subject: {subject} Accuracy: {100*accuracy:.2f}\nModel: CNNLSTM\nIK_Joints: Hip\nAngular Velocity Added', color='black', size=16)
        plt.ylabel('True', color='black', size=14)
        plt.xlabel('Predicted', color='black', size=14)
        ax.xaxis.set_tick_params(labelcolor='black', labelsize=12)
        ax.yaxis.set_tick_params(labelcolor='black', labelsize=12)
        ax.set_xticklabels(('stand','walk','rampascent','rampdescent','stairascent','stairdescent'))
        ax.set_yticklabels(('stand','walk','rampascent','rampdescent','stairascent','stairdescent'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        save_path_fig = os.path.join(base_save_path,
                                     f"{subject}/cm_p_fig/",
                                     f'{subject}_results_{r+1}.png')
        # check if the path exists
        os.makedirs(os.path.dirname(save_path_fig), exist_ok=True)
        plt.savefig(save_path_fig, format='png')
        plt.show()   
        plt.close(fig)  # Close the figure to free up memory


# In[13]:

def run_experiment(repeats=20):
    # define random seed
    seed = 42
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # load data
    subjects = ['AB06', 'AB08', 'AB09', 'AB10', 'AB11', 'AB12', 'AB13', 'AB14', 'AB15', 'AB16', 'AB17', 'AB18', 'AB19', 'AB20', 'AB21', 'AB23', 'AB24', 'AB25', 'AB27', 'AB28', 'AB30']

    base_save_path = ('/CNNLSTM')

    for subject in subjects:
        results = {}
        trainX, testX, trainy, testy = load_and_process_data_10(subject)
        
        accuracy_result = pd.DataFrame(columns=['Round', 'Training_Accuracy', 'Test_Accuracy'])
        print("Current Subject: ", subject)
        print("trainX shape:", trainX.shape)
        print("testX shape:", testX.shape)
        print("trainy shape:", trainy.shape)
        print("testy shape:", testy.shape)
        
        # repeat experiment
        for r in range(repeats):
            # reset random seed
            np.random.seed(seed + r)
            tf.random.set_seed(seed + r)
            
            print("Round %d:" % (r+1))
            model, accuracy, train_accuracy, train_loss, y_pred, y_true = evaluate_model(trainX, trainy, testX, testy)
            round_result = pd.DataFrame(
                {'Round': [r + 1], 'Training_Accuracy': [train_accuracy], 'Test_Accuracy': [accuracy]})
            accuracy_result = pd.concat([accuracy_result, round_result], ignore_index=True)
            
            # calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            print("Confusion Matrix:")
            print(cm)

            # Save the confusion matrix for each round
            save_cm(cm, subject, r, base_save_path)
            display_trainloss(train_loss, subject, r, base_save_path)
            display_confusion_matrix(cm, subject, accuracy, r, base_save_path)
            display_percentage_confusion_matrix(cm, subject, accuracy, r, base_save_path)
            
            print('Test Accuracy: ', accuracy*100)
    
        save_path_csv = os.path.join(base_save_path,
                                     f"{subject}/Accuracy_csv/",
                                     f'{subject}_50ms_CNNLSTM_IK_Hip_Flexion_WithAngvel_results.csv')
        os.makedirs(os.path.dirname(save_path_csv), exist_ok=True)
        accuracy_result.to_csv(save_path_csv)
        print(f'{subject} is done.')


# In[14]:


run_experiment()


# In[ ]:


end_time = time.time()

total_time = end_time - start_time

hours, remainder = divmod(total_time, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"Program total running time：{int(hours)}Hours{int(minutes)}minutes{seconds}Seconds")


# In[ ]:




