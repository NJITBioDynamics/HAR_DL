#!/usr/bin/env python
# coding: utf-8

import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from tensorflow.keras.models import load_model


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(len(gpus), "Physical GPUs detected. GPU acceleration enabled.")
    except RuntimeError as e:
        print(e)

tf.config.optimizer.set_jit(True)
tf.config.run_functions_eagerly(False)


base_save_path = "/LSTM"

splitted_dir = "/Splitted"

comb_keys = ["all_joints", "hip_ankle", "hip_knee", "ankle_knee", "hip", "ankle", "knee"]

CM_CSV_SUBDIR = "cm_csv"
CM_FIG_SUBDIR = "cm_fig"
CM_PERCENT_FIG_SUBDIR = "cm_p_fig"


def save_cm(cm, subject, comb_key, r, base_save_path):

    labels = ['stand', 'walk', 'rampascend', 'rampdescend', 'stairascend', 'stairdescend']

    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    save_path_csv = os.path.join(base_save_path,
                                 f"{subject}/{CM_CSV_SUBDIR}/",
                                 f'{subject}_{comb_key}_LSTM_cm_{r + 1}.csv')
    os.makedirs(os.path.dirname(save_path_csv), exist_ok=True)
    cm_df.to_csv(save_path_csv, index=True)


def display_confusion_matrix(cm, subject, comb_key, accuracy, r, base_save_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = mpl.cm.get_cmap("rainbow")
    im = ax.imshow(cm, cmap=cmap)
    ax.set_xticks(np.arange(6))
    ax.set_yticks(np.arange(6))
    ax.set_xticklabels(('stand', 'walk', 'rampascend', 'rampdescend', 'stairascend', 'stairdescend'))
    ax.set_yticklabels(('stand', 'walk', 'rampascend', 'rampdescend', 'stairascend', 'stairdescend'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="w")
    ax.set_title("Subject: %s, comb: %s, Accuracy: %.2f" % (subject, comb_key, 100 * accuracy))
    fig.tight_layout()
    save_path_fig = os.path.join(base_save_path,
                                 f"{subject}/{CM_FIG_SUBDIR}/",
                                 f'{subject}_{comb_key}_LSTM_results_{r + 1}.png')
    os.makedirs(os.path.dirname(save_path_fig), exist_ok=True)
    plt.savefig(save_path_fig, format='png')


def display_percentage_confusion_matrix(cm, subject, comb_key, accuracy, r, base_save_path):
    percentage_matrix = np.zeros(cm.shape)
    for i in range(cm.shape[0]):
        row_sum = np.sum(cm[i, :])
        if row_sum > 0:
            percentage_matrix[i, :] = cm[i, :] / row_sum * 100
    percentage_matrix = np.around(percentage_matrix, 2)
    with plt.style.context('classic'):
        plt.figure(figsize=(10, 8))
        fig = plt.gcf()
        fig.patch.set_facecolor('white')
        ax = sns.heatmap(percentage_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=False,
                         linewidths=1, linecolor='white', annot_kws={"size": 14})
        plt.title(f'Subject: {subject}, comb: {comb_key}, Accuracy: {100 * accuracy:.2f}\nModel: LSTM', color='black',
                  size=16)
        plt.ylabel('True', color='black', size=14)
        plt.xlabel('Predicted', color='black', size=14)
        ax.xaxis.set_tick_params(labelcolor='black', labelsize=12)
        ax.yaxis.set_tick_params(labelcolor='black', labelsize=12)
        ax.set_xticklabels(('stand', 'walk', 'rampascend', 'rampdescend', 'stairascend', 'stairdescend'))
        ax.set_yticklabels(('stand', 'walk', 'rampascend', 'rampdescend', 'stairascend', 'stairdescend'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        save_path_fig = os.path.join(base_save_path,
                                     f"{subject}/{CM_PERCENT_FIG_SUBDIR}/",
                                     f'{subject}_{comb_key}_LSTM_p_results_{r + 1}.png')
        os.makedirs(os.path.dirname(save_path_fig), exist_ok=True)
        plt.savefig(save_path_fig, format='png')


def load_test_data(subject, comb_key):
    testX_path = os.path.join(splitted_dir, subject, f"{subject}_{comb_key}_testX.npy")
    testy_path = os.path.join(splitted_dir, subject, f"{subject}_{comb_key}_testy.npy")
    if os.path.exists(testX_path) and os.path.exists(testy_path):
        testX = np.load(testX_path)
        testy = np.load(testy_path)
        return testX, testy
    else:
        print(f"Test data for subject {subject}, comb {comb_key} not found.")
        return None, None


def run_testing_experiment(repeats=1):
    subjects = ['AB06', 'AB08', 'AB09', 'AB10', 'AB11', 'AB12', 'AB13', 'AB14', 'AB15', 'AB16',
                'AB17', 'AB18', 'AB19', 'AB20', 'AB21', 'AB23', 'AB24', 'AB25', 'AB27', 'AB28', 'AB30']

    pred_time_stats = {comb_key: [] for comb_key in comb_keys}

    pred_time_stats_per_sample = {comb_key: [] for comb_key in comb_keys}

    for subject in subjects:
        for comb_key in comb_keys:
            print("Testing Subject:", subject, "Combination:", comb_key)
            testX, testy = load_test_data(subject, comb_key)
            if testX is None or testy is None:
                continue

            print("testX size is: ", testX.shape)
            print(" ")

            accuracy_result = pd.DataFrame(columns=['Round', 'Test_Accuracy', 'Prediction_Time'])

            for r in range(repeats):
                model_path = os.path.join(
                                          subject,
                                          f"{subject}_{comb_key}_model_{r + 1}.h5"
                )
                if not os.path.exists(model_path):
                    print(f"Model for {subject} comb {comb_key} round {r + 1} not found, skipping.")
                    continue

                print("Loading model: ")
                model = load_model(model_path)
                print("Model loaded.")

                batch_size = 1

                test_accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)[1]
                # For CNNLSTM, use the following 2 lines:
                # testX_reshaped = testX.reshape((testX.shape[0], 1, 10, testX.shape[2]))
                # test_accuracy = model.evaluate(testX_reshaped, testy, batch_size=batch_size, verbose=0)[1]

                predictions_list = []
                inference_times = []

                # For LSTM, use the following:
                input_buffer = tf.constant(
                    np.zeros((1, *testX.shape[1:]), dtype=np.float32),
                    dtype=tf.float32
                )
                # For CNN-LSTM, use the following:
                # input_buffer = tf.constant(
                #     np.zeros((1, 1, 10, testX.shape[-1]),  # 4D 形状
                #              dtype=np.float32)
                #     )
                # input_buffer = tf.cast(input_buffer, dtype=model.input.dtype)  # 与模型输入类型对齐

                @tf.function
                def static_inference(x):
                    return model(x, training=False)


                _ = static_inference(input_buffer)
                print("Start inferring: ")

                for i in range(testX.shape[0]):

                    start_single = time.perf_counter()

                    # For LSTM, use the following:
                    np.copyto(input_buffer.numpy(), testX[i:i + 1, :, :])
                    # For CNN-LSTM, use the following:
                    # np.copyto(input_buffer.numpy(), testX_reshaped[i:i + 1, :, :, :])
                    # x_single = testX[i:i + 1, :, :]
                    # model.predict(input_tensor, batch_size=1)
                    # model(input_tensor)


                    single_pred = static_inference(input_buffer)
                    end_single = time.perf_counter()

                    predictions_list.append(single_pred.numpy().copy())
                    iter_time = end_single - start_single
                    # total_time += iter_time

                    print(f"Prediction time: {end_single - start_single}")
                    inference_times.append(end_single - start_single)

                prediction_time = np.sum(inference_times)
                avg_time_per_sample = prediction_time / testX.shape[0]


                print(f"Subject: {subject}, Comb: {comb_key}, Round {r + 1}")
                print("Prediction Time:", prediction_time, "seconds")
                print("Avg Time per Sample:", avg_time_per_sample*1000, "miliseconds")


                pred_time_stats[comb_key].append(prediction_time)
                pred_time_stats_per_sample[comb_key].append(avg_time_per_sample)

                round_result = pd.DataFrame({'Round': [r + 1],
                                             'Test_Accuracy': [test_accuracy],
                                             'Prediction_Time': [prediction_time],
                                            'Avg_Time_Per_Sample': [avg_time_per_sample]})
                accuracy_result = pd.concat([accuracy_result, round_result], ignore_index=True)

            save_path_csv = os.path.join(base_save_path,
                                         f"{subject}/Accuracy_csv/",
                                         f'{subject}_{comb_key}_LSTM_results.csv')
            os.makedirs(os.path.dirname(save_path_csv), exist_ok=True)
            accuracy_result.to_csv(save_path_csv, index=False)
            print(f"{subject} Combination {comb_key} testing is done.")

    stats_list = []
    for comb_key in comb_keys:
        times = pred_time_stats[comb_key]
        per_sample_times = pred_time_stats_per_sample[comb_key]
        if len(times) > 0:
            avg_total_time = np.mean(times)
            std_total_time = np.std(times)
            avg_time_per_sample = np.mean(per_sample_times)
            std_time_per_sample = np.std(per_sample_times)
        else:
            avg_total_time = None
            std_total_time = None
            avg_time_per_sample = None
            std_time_per_sample = None
        stats_list.append({
            'comb_key': comb_key,
            'Average_Total_Prediction_Time': avg_total_time,
            'Std_Total_Prediction_Time': std_total_time,
            'Average_Time_Per_Sample': avg_time_per_sample,
            'Std_Time_Per_Sample': std_time_per_sample
            })
        print(f"Combination: {comb_key}, Average Prediction Time: {avg_total_time}, Std: {std_total_time}")
        print(f"Combination: {comb_key}, Average Time per Sample: {avg_time_per_sample}, Std: {std_time_per_sample}")

    stats_df = pd.DataFrame(stats_list)
    stats_csv_path = os.path.join(base_save_path, "LSTM_Prediction_Time_Stats.csv")
    stats_df.to_csv(stats_csv_path, index=False)
    print("Overall prediction time statistics saved to:", stats_csv_path)


if __name__ == '__main__':
    print("Starting LSTM testing experiment...")
    start_time = time.time()
    run_testing_experiment(repeats=1)
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Testing total running time: {int(hours)} Hours {int(minutes)} minutes {seconds} Seconds")
