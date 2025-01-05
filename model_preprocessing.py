import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Функция для предобработки данных
def preprocess_data(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            data = pd.read_csv(os.path.join(input_dir, file))
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data[['Temperature']])
            df_scaled = pd.DataFrame(scaled_data, columns=['Temperature'])
            df_scaled.to_csv(os.path.join(output_dir, file), index=False)

    print(f"Предобработка завершена для папки {input_dir}, результаты сохранены в {output_dir}")

# Предобработка тренировочных и тестовых данных
preprocess_data(train_dir, 'train_processed')
preprocess_data(test_dir, 'test_processed')
