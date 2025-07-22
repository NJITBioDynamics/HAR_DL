#!/usr/bin/env python
# coding: utf-8

import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F


model_save_root = '/MLP'
splitted_dir = '/Splitted/MLP'
comb_keys = ["all_joints", "hip_ankle", "hip_knee", "ankle_knee", "hip", "ankle", "knee"]

stats_csv_path = os.path.join(model_save_root, "MLP_Prediction_Time_Stats.csv")

device = torch.device(f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
print("Current device:", device)
if device.type == 'cuda':
    print("GPU name:", torch.cuda.get_device_name(device.index))

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


def create_data_loaders(testX, testy, batch_size):
    test_dataset = TensorDataset(torch.tensor(testX, dtype=torch.float32), torch.tensor(testy, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.out = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

hidden_size =
num_classes =
batch_size =


def run_testing_experiment(repeats=5):
    subjects = ['AB06', 'AB08', 'AB09', 'AB10', 'AB11', 'AB12', 'AB13', 'AB14', 'AB15', 'AB16',
                'AB17', 'AB18', 'AB19', 'AB20', 'AB21', 'AB23', 'AB24', 'AB25', 'AB27', 'AB28', 'AB30']


    pred_time_stats = {ck: [] for ck in comb_keys}
    pred_time_stats_per_sample = {ck: [] for ck in comb_keys}

    for subject in subjects:
        for comb_key in comb_keys:
            print(f"\nTesting Subject: {subject}, comb: {comb_key}")
            testX, testy = load_test_data(subject, comb_key)
            if testX is None or testy is None:
                continue
            test_loader = create_data_loaders(testX, testy, batch_size)

            testX_tensor = torch.FloatTensor(testX).to(device)
            test_samples = testX_tensor.unbind(0)


            input_size = testX.shape[1]
            input_buffer = torch.empty(
                (1, input_size),
                dtype=torch.float32,
                device=device
            )
            if device.type == 'cpu':
                input_buffer = input_buffer.pin_memory()

            assert input_buffer.device.type == device.type, \
                f"Type of device is not correct！It should be：{device.type}，It's now：{input_buffer.device.type}"

            results = []
            for r in range(repeats):

                model_path = os.path.join(model_save_root, subject, f"{subject}_{comb_key}_model_{r + 1}.pt")
                if not os.path.exists(model_path):
                    print(f"Model for {subject} comb {comb_key} round {r + 1} not found, skipping.")
                    continue
                model = MLP(input_size, hidden_size, num_classes).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()


                try:

                    example_input = torch.randn(1, input_size, device=device)

                    scripted_model = torch.jit.trace(model, example_input)

                    scripted_model = torch.jit.optimize_for_inference(scripted_model)
                except Exception as e:
                    print(f"JIT compiling failed: {e}, use original model.")
                    scripted_model = model

                    example_input = torch.randn(1, input_size, device=device)


                with torch.no_grad():
                    input_buffer.copy_(example_input)
                    _ = scripted_model(input_buffer)

                all_preds = []
                total_time = 0.0

                for sample in test_samples:

                    input_buffer.copy_(sample.view(1, -1))

                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    start_time = time.perf_counter()

                    with torch.no_grad():
                        outputs = scripted_model(input_buffer)
                        _, preds = torch.max(outputs, 1)

                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    end_time = time.perf_counter()

                    all_preds.append(preds.cpu().numpy()[0])
                    total_time += (end_time - start_time)

                avg_time_per_sample = total_time / len(test_samples)
                pred_time_stats[comb_key].append(total_time)
                pred_time_stats_per_sample[comb_key].append(avg_time_per_sample)

                correct = 0
                total = 0
                with torch.no_grad():
                    for data, labels in test_loader:
                        data = data.to(device)
                        labels = labels.to(device)
                        outputs = model(data)
                        _, preds = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (preds == labels).sum().item()
                test_accuracy = correct / total

                print(f"Subject: {subject}, comb: {comb_key}, Round {r + 1}")
                print(
                    f"Test Accuracy: {100 * test_accuracy:.2f}%, Total Prediction Time: {total_time:.4f}s, Avg per Sample: {avg_time_per_sample:.6f}s")
                results.append({"Round": r + 1, "Test_Accuracy": test_accuracy,
                                "Total_Prediction_Time": total_time,
                                "Avg_Time_Per_Sample": avg_time_per_sample})

            if results:
                results_df = pd.DataFrame(results)
                res_dir = os.path.join(model_save_root, subject, "Accuracy_csv")
                os.makedirs(res_dir, exist_ok=True)
                results_df.to_csv(os.path.join(res_dir, f"{subject}_{comb_key}_MLP_results.csv"), index=False)
                print(f"Subject {subject}, comb {comb_key} testing done.")


    stats_list = []
    for ck in comb_keys:
        times = pred_time_stats[ck]
        per_sample_times = pred_time_stats_per_sample[ck]
        if times:
            avg_total = np.mean(times)
            std_total = np.std(times)
            avg_sample = np.mean(per_sample_times)
            std_sample = np.std(per_sample_times)
        else:
            avg_total = std_total = avg_sample = std_sample = None
        stats_list.append({
            "comb_key": ck,
            "Average_Total_Prediction_Time": avg_total,
            "Std_Total_Prediction_Time": std_total,
            "Average_Time_Per_Sample": avg_sample,
            "Std_Time_Per_Sample": std_sample
        })
        print(f"Comb: {ck}, Avg Total Time: {avg_total}, Std Total: {std_total}")
        print(f"Comb: {ck}, Avg per Sample: {avg_sample}, Std per Sample: {std_sample}")

    stats_df = pd.DataFrame(stats_list)
    os.makedirs(os.path.dirname(stats_csv_path), exist_ok=True)
    stats_df.to_csv(stats_csv_path, index=False)
    print("Overall prediction time statistics saved to:", stats_csv_path)


if __name__ == '__main__':
    print("Starting MLP testing experiment...")
    start_time = time.time()
    run_testing_experiment(repeats=1)
    end_time = time.time()
    total_time = end_time - start_time
    hrs, rem = divmod(total_time, 3600)
    mins, secs = divmod(rem, 60)
    print(f"Testing total running time: {int(hrs)} Hours {int(mins)} minutes {secs} Seconds")
