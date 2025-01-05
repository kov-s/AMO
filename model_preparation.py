from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Загрузка предобработанных данных
train_dir = 'train_processed'

# Функция для обучения модели
def train_model(input_dir):
    X = []
    y = []

    # Загружаем все данные из папки
    for file in os.listdir(input_dir):
        if file.endswith('.csv'):
            data = pd.read_csv(os.path.join(input_dir, file))
            X.append(data[['Temperature']].values)
            y.append(data[['Temperature']].shift(-1).fillna(method='ffill').values)  # Прогнозируем следующее значение

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)

    # Создаем и обучаем модель
    model = LinearRegression()
    model.fit(X, y)

    # Сохраняем модель
    joblib.dump(model, 'model.pkl')
    print("Модель успешно обучена и сохранена как 'model.pkl'.")

# Обучаем модель на тренировочных данных
train_model(train_dir)