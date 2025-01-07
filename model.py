import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pickle
from imblearn.over_sampling import SMOTE

# Загрузка данных из CSV файла
data = pd.read_csv('train.csv', sep=';')

# Извлечение признаков (цветовые компоненты HSL)
colors = data[['h', 's', 'l']]

# Извлечение целевой переменной (эмоции)
emotions = data['emotion']

# Вывод количества каждой эмоции для анализа распределения классов
emotion_counts = emotions.value_counts()
print(emotion_counts)

# Преобразование строковых значений эмоций в числовые метки
label_encoder = LabelEncoder()
encoded_emotions = label_encoder.fit_transform(emotions)

# Разделение данных на обучающую и тестовую выборки
x_train, x_test, y_train, y_test = train_test_split(
    colors, encoded_emotions, test_size=0.2, random_state=42, shuffle=True
)

# Масштабирование признаков
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Применение метода SMOTE для балансировки классов
smote = SMOTE(random_state=42)
x_train_smoted, y_train_smoted = smote.fit_resample(x_train_scaled, y_train)

# Определение моделей для обучения
models = {
    'logistic_regression': LogisticRegression(),
    'k_nearest_neighbors': KNeighborsClassifier(),
    'decision_tree': DecisionTreeClassifier(),
    'random_forest': RandomForestClassifier(),
    'support_vector_machine': SVC()
}


# Функция сохранения модели
def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


# Функция загрузки модели
def load_model(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# Обучение всех моделей и сохранение их
for model_name, model in models.items():
    model.fit(x_train_smoted, y_train_smoted)
    save_model(model, f'model_{model_name}.pkl')

# Сохранение скалера и кодировщика меток
save_model(scaler, 'scaler.pkl')
save_model(label_encoder, 'label_encoder.pkl')

# Прогнозирование на тестовой выборке
test_predictions = {}
for model_name, model in models.items():
    loaded_model = load_model(f'model_{model_name}.pkl')
    predictions = loaded_model.predict(x_test_scaled)
    test_predictions[model_name] = predictions

# Оценка точности моделей
for model_name, predictions in test_predictions.items():
    accuracy = accuracy_score(y_test, predictions)
    print(f'{model_name}: Accuracy = {accuracy:.2f}')
