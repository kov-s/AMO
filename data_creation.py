import numpy as np
import os
import pandas as pd


def generate_data(size=100, anomaly_prob=0.05):
    temperatures = np.linspace(-30, 30, size)
    # Добавим шум
    noise = np.random.normal(-2, 2, size)
    temperatures += noise
    for i in range(size):
        if np.random.rand() < anomaly_prob:
            temperatures[i] += np.random.randint(-15, 15)
    return temperatures

train_dir = 'train'
test_dir = 'test'
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for i in range(3):
    data = generate_data(size=100)
    df = pd.DataFrame(data, columns=['Temperature'])
    df.to_csv(f'{train_dir}/train_data_{i}.csv', index=False)

for i in range(3):
    data = generate_data(size=50)
    df = pd.DataFrame(data, columns=['Temperature'])
    df.to_csv(f'{test_dir}/test_data_{i}.csv', index=False)

print("Данные успешно созданы и сохранены в папки 'train' и 'test'.")