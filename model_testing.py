model = joblib.load('model.pkl')
test_dir = 'test_processed'

def test_model(input_dir):
    X_test = []
    y_test = []
    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            data = pd.read_csv(os.path.join(input_dir, file))
            X_test.append(data[['Temperature']].values)
            y_test.append(data[['Temperature']].shift(-1).fillna(method='ffill').values)
    X_test = np.concatenate(X_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    # Прогнозирование
    y_pred = model.predict(X_test)

    # Оценка
    mse = mean_squared_error(y_test, y_pred)
    print(f"Среднеквадратичная ошибка на тестовых данных: {mse:.4f}")

# Тестируем
test_model(test_dir)