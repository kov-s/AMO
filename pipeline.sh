#!/bin/bash

# Создание данных
echo "файл 1"
python3 data_creation.py

# Предобработка данных
echo "файл2"
python3 model_preprocessing.py

# Обучение модели
echo "файл 3"
python3 model_preparation.py

# Тестирование модели
echo "файл 4"
python3 model_testing.py

echo "финиш"