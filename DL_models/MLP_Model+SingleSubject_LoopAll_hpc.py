#!/usr/bin/env python
# coding: utf-8

# In[1]:


# MLP
import numpy as np
from numpy import mean
from numpy import std
from numpy import dstack
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas import read_csv


import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# In[3]:

start_time = time.time()


# In[5]:

def load_and_process_data(number):
    base_path = '/EPIC_Lower_Limb_Dataset_Windows_09072023/50_ms'
    number_str = str(number)
    path_data_imu = os.path.join(base_path, number_str, 'IMU_Windows', f'{number}_Windows')
    # path_data_ik = os.path.join(base_path, number_str, 'IK_Windows', f'{number}_Windows')
    # path_data_gon = os.path.join(base_path, number_str, 'GON_Windows', f'{number}_Windows')
    path_labels_ik = os.path.join(base_path, number_str, 'IK_Windows', f'{number}_y_labels')

    # files_data_ik_angvel = [
    #                   IK angular velocity csv data here

    #                                                                 ]


    files_data_ik_angles = [
    #               IK joint angles csv data here

     ]

    files_data_imu = [
    #               IMU csv data here

                        ]

#     files_data_gon = [
#                       GON csv data here

#                      ]

    

    # data_matrices_ik = [pd.read_csv(f"{path_data_ik}/{file}") for file in files_data_ik]
    data_matrices_imu = [pd.read_csv(f"{path_data_imu}/{file}") for file in files_data_imu]
#     data_matrices_gon = [pd.read_csv(f"{path_data_gon}/{file}") for file in files_data_gon]
    

    # data_matrix_ik = np.dstack([matrix.values for matrix in data_matrices_ik])
    data_matrix_imu = np.dstack([matrix.values for matrix in data_matrices_imu])
#     data_matrix_gon = np.dstack([matrix.values for matrix in data_matrices_gon])
#     data_matrix = np.dstack([data_matrix_imu, data_matrix_ik])


    label_matrix = pd.read_csv(f"{path_labels_ik}/{number}_ik_ankle_angle_l_y_labels.csv")



    trainX, testX, trainy, testy = train_test_split(data_matrix_imu, label_matrix, test_size=0.3, random_state=42, stratify=label_matrix)


# load the data and convert to tensors
    trainX = trainX.reshape(trainX.shape[0], -1)
    testX = testX.reshape(testX.shape[0], -1)

    trainX = torch.tensor(trainX, dtype=torch.float32).to(device)
    testX = torch.tensor(testX, dtype=torch.float32).to(device)
    

    # load the data and convert to tensors
#     trainX = trainX.reshape(trainX.shape[0], -1)
#     testX = testX.reshape(testX.shape[0], -1)

#     trainX = torch.tensor(trainX, dtype=torch.float32).to(device)
#     testX = torch.tensor(testX, dtype=torch.float32).to(device)
    
    # encode labels
    encoder = LabelEncoder()
    trainy = encoder.fit_transform(trainy)
    testy = encoder.transform(testy)

    trainy = torch.tensor(trainy, dtype=torch.long).to(device)
    testy = torch.tensor(testy, dtype=torch.long).to(device)

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


# In[6]:


# set the hyperparameters
# Time series setups
input_size =
hidden_size =
num_classes =
num_epochs =
batch_size =
learning_rate =


# In[7]:


# define the network architecture
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        #self.relu = nn.ReLU()
        self.out = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


# In[8]:
# set the device to run on
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Current device:", device)

if device.type == 'cuda':
    print("GPU name:", torch.cuda.get_device_name(0))
    print("Number of available GPUs:", torch.cuda.device_count())
else:
    print("GPU is not available. Using CPU.")


# In[9]:
# create data loaders for training and testing
def create_data_loaders(trainX, trainy, testX, testy, batch_size):
    train_data = torch.utils.data.TensorDataset(trainX, trainy)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torch.utils.data.TensorDataset(testX, testy)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    return train_data, train_loader, test_data, test_loader


# In[10]:
# initialize the network and move to device
def initialize_network():
    model = MLP(input_size, hidden_size, num_classes).to(device)
    return model


# In[11]:
# define the loss function and optimizer
def lossfunction_optimizer(model):
    criterion = nn.CrossEntropyLoss()
#     criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    return criterion, optimizer


# In[12]:
# Function for training the model
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    train_loss = []
    train_loss_avg = []
    for epoch in range(num_epochs):
        train_loss.append(0)
        train_loss_avg.append(0)
        num_batches = 0
        
        for i, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(data)
            loss = criterion(outputs, labels)

            # backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # --- training ends
            train_loss[-1] += loss.item()
            train_loss_avg[-1] += loss.item()
            num_batches += 1
            
#             train_loss.append(loss.item())
        train_loss_avg[-1] /= num_batches
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss[-1]:.4f}, Loss_Average: {train_loss_avg[-1]}')
    
    
    return train_loss


# In[ ]:
def display_trainloss(train_loss, subject, r, base_save_path):
    plt.ion()
    fig = plt.figure()
    plt.plot(train_loss, label="train_loss")
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Cross-entropy loss')
    save_path_fig = os.path.join(base_save_path,
                                 f"{subject}/trainloss/",
                                 f'{subject}_trainloss_{r+1}.png')
    # check if the path exists
    os.makedirs(os.path.dirname(save_path_fig), exist_ok=True)
    plt.savefig(save_path_fig, format='png')
    plt.show()
    plt.close(fig)


# In[13]:
# Function for testing the model
def test_model(model, test_loader):
    with torch.no_grad():
        correct = 0
        total = 0
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)

            # forward pass and get predictions
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)

            # update accuracy metrics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Test Accuracy: {100 * correct / total}%')

    return accuracy


# In[14]:
# set to evaluation mode
def evaluation_mode(model, test_loader, criterion):
    model.eval()

    num_incorrect = 0
    test_loss_avg = 0
    num_batches = 0
    num_instances = 0
    
    test_labels = []
    pred_labels = []

    for data, labels in test_loader:
        # move data and labels to device
        data = data.to(device)
        labels = labels.to(device)   
        
        # forward pass and get predictions
        outputs = model(data)
        
        
        _, predicted = torch.max(outputs.data, 1)
        num_incorrect += torch.ne(predicted, labels).sum().item()
        test_labels.append(labels)
        pred_labels.append(predicted)
        
        # cross-entropy loss
#         loss = F.nll_loss(outputs, labels)
        loss = criterion(outputs, labels)
        
        test_loss_avg += loss.item()
        num_batches += 1
        num_instances += data.size(0)
        
    test_loss_avg /= num_batches
    print('average loss: %f' % (test_loss_avg))
    print('classification error: %f%%' % ((num_incorrect / num_instances)*100))
    
    return test_labels, pred_labels, test_loss_avg

# In[]:
def save_cm(cm, subject, r, base_save_path):
    # 定义类别标签
    labels = ['stand', 'walk', 'rampascend', 'rampdescend', 'stairascend', 'stairdescend']
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    save_path_csv = os.path.join(base_save_path,
                                 f"{subject}/cm_csv/",
                                 f'{subject}_cm_{r+1}.csv')
    os.makedirs(os.path.dirname(save_path_csv), exist_ok=True)
    cm_df.to_csv(save_path_csv, index=True)


# In[15]:
def display_confusion_matrix(cm, subject, accuracy, r, base_save_path):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    cm = metrics.ConfusionMatrixDisplay(cm, display_labels=range(6))
    cm.plot(values_format='d', ax=ax)
    plt.title('subject: %s Accuracy = %.2f ' % (subject, 100*accuracy))
    save_path_fig = os.path.join(base_save_path,
                                 f"{subject}/cm_fig/",
                                 f'{subject}_results_{r+1}.png')
    # check if the path exists
    os.makedirs(os.path.dirname(save_path_fig), exist_ok=True)
    # save_path_fig = f"D:\\OneDrive - NJIT\\Research\\Dataset_Test\\Results\\50 ms\\IK_Test\\TripleIK\\AngleOnly\\Feb2024\\MLP\\{subject}_40ms_MLP_IK+IMU_Hip+Knee_results_1.png"
    plt.savefig(save_path_fig, format='png')
    plt.show()
    plt.close(fig)


# In[ ]:


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
        plt.figure(figsize=(10, 10))
        # Set the background color to white
        fig = plt.gcf()
        fig.patch.set_facecolor('white')

        ax = sns.heatmap(percentage_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=False, linewidths=1, linecolor='white', annot_kws={"size": 14})
        # plt.title(f'Subject: {subject} Accuracy: {100*accuracy:.2f}\nModel: MLP\nIK_Joints: Hip+Knee+Ankle(Bilateral)\nAngular Velocity Added', color='black', size=16)
        plt.title(
            f'Subject: {subject} Accuracy: {100 * accuracy:.2f}\nModel: MLP\nIMU: Shank',
            color='black', size=16)
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
                                     f'{subject}_p_results_{r+1}.png')
        # check if the path exists
        os.makedirs(os.path.dirname(save_path_fig), exist_ok=True)
        # save_path_fig = f"D:\\OneDrive - NJIT\\Research\\Dataset_Test\\Results\\50 ms\\IK_Test\\TripleIK\\AngleOnly\\Feb2024\\MLP\\{subject}_50ms_LSTM_IK_Ankle_Flexion_WithAngvel_p_results_{r}.png"
        plt.savefig(save_path_fig, format='png')
        plt.show()   
        plt.close(fig)  # Close the figure to free up memory


# In[16]:


# Training and testing the model
def run_experiment(repeats=20):
    seed = 42
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Define the numbers list
    subjects = ['AB06', 'AB08', 'AB09', 'AB10', 'AB11', 'AB12', 'AB13', 'AB14', 'AB15', 'AB16', 'AB17', 'AB18', 'AB19', 'AB20', 'AB21', 'AB23', 'AB24', 'AB25', 'AB27', 'AB28', 'AB30']
    # subjects = ['AB06']
    base_save_path = ('/MLP/')

    # 初始化一个dataframe来储存average loss和Test accuracy
    columns = ["Subject", "Average Loss", "Test Accuracy"]
    results_df = pd.DataFrame(columns=columns)

    for subject in subjects:
        print("Current subject:", subject)
        trainX, testX, trainy, testy = load_and_process_data(subject)

        train_data, train_loader, test_data, test_loader = create_data_loaders(trainX, trainy, testX, testy, batch_size)

        for r in range(repeats):
            torch.manual_seed(seed + r)
            np.random.seed(seed + r)
            random.seed(seed + r)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed + r)

            model = initialize_network()
            criterion, optimizer = lossfunction_optimizer(model)
            train_loss = train_model(model, train_loader, criterion, optimizer, num_epochs)
            accuracy = test_model(model, test_loader)
            test_labels, pred_labels, test_loss_avg = evaluation_mode(model, test_loader, criterion)

            new_row_df = pd.DataFrame({
                "Subject": [subject],
                "Average Loss": [test_loss_avg],
                "Test Accuracy": [accuracy]})

            results_df = pd.concat([results_df, new_row_df], ignore_index=True)

            pred_labels = torch.cat(pred_labels, dim=0).cpu().numpy()
            test_labels = torch.cat(test_labels, dim=0).cpu().numpy()
            cm = metrics.confusion_matrix(test_labels, pred_labels)

            print("Confusion Matrix:")
            print(cm)

            # Save the confusion matrix for each round
            save_cm(cm, subject, r, base_save_path)
            display_trainloss(train_loss, subject, r, base_save_path)
            display_confusion_matrix(cm, subject, accuracy, r, base_save_path)
            display_percentage_confusion_matrix(cm, subject, accuracy, r, base_save_path)


        save_path_csv = os.path.join(base_save_path,
                                     f"{subject}/Accuracy_csv/",
                                     f'{subject}_results.csv')
        os.makedirs(os.path.dirname(save_path_csv), exist_ok=True)
        results_df.to_csv(save_path_csv, index=False)

        print(f"{subject} is done.")
        print("   ")


# In[14]:


run_experiment()


# In[ ]:

end_time = time.time()


total_time = end_time - start_time

hours, remainder = divmod(total_time, 3600)
minutes, seconds = divmod(remainder, 60)

print(f"Program total running time：{int(hours)}Hours{int(minutes)}minutes{seconds}Seconds")

