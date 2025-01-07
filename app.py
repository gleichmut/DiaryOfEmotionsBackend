from fastapi import FastAPI, Request, HTTPException
import uvicorn
import pickle
import numpy as np

app = FastAPI()  # Создание экземпляра приложения FastAPI

loaded_model = None  # Переменная для хранения загруженной модели
loaded_scaler = None  # Переменная для хранения загруженного скалера
loaded_label_encoder = None  # Переменная для хранения загруженного кодировщика меток


# Запускается при старте сервера
@app.on_event("startup")
async def startup_event():
    """
    Загружает модель, скалера и кодировщик меток при запуске приложения.
    """
    global loaded_model
    global loaded_scaler
    global loaded_label_encoder
    loaded_model = load_model('model_k_nearest_neighbors.pkl')
    loaded_scaler = load_model('scaler.pkl')
    loaded_label_encoder = load_model('label_encoder.pkl')


def load_model(filename):
    """
    Загружает объект из файла, используя модуль pickle.

    :param filename: Имя файла, из которого нужно загрузить объект.
    :return: Загруженный объект.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)


@app.post("/predict_emotion")
async def predict_emotion(request: Request):
    """
    Обрабатывает POST-запросы на путь /predict_emotion.
    Получает JSON-данные, проверяет корректность входных данных,
    преобразует их, делает прогноз и возвращает предсказанную эмоцию.

    :param request: Объект запроса.
    :raises HTTPException: Если данные некорректны или отсутствуют необходимые ключи.
    :return: Словарь с ключом "predicted_emotion" и значением предсказанной эмоции.
    """
    request_data = await request.json()
    if not all(key in request_data for key in ['h', 's', 'l']):
        raise HTTPException(status_code=400,
                            detail="Недостаточно данных. Пожалуйста, предоставьте значения для 'h', 's' и 'l'.")
    # Значение оттенка цвета (Hue)
    h = float(request_data["h"])
    # Значение насыщенности цвета (Saturation)
    s = float(request_data["s"])
    # Значение яркости цвета (Lightness)
    l = float(request_data["l"])
    if not (0 <= h <= 360 and 0 <= s <= 100 and 0 <= l <= 100):
        raise HTTPException(status_code=400,
                            detail="Данные вне границ. Пожалуйста, предоставьте значения для 'h' от 0 до 360, для 's' и 'l' от 0 до 100.")
    # Форматируем входные данные в виде NumPy массива
    input_vector = np.array([[h, s, l]])
    # Масштабируем данные с помощью загруженного скалера
    scaled_input = loaded_scaler.transform(input_vector)
    # Делаем прогноз с помощью загруженной модели
    prediction = loaded_model.predict(scaled_input)[0]
    # Декодируем предсказанный класс обратно в строку
    predicted_emotion = loaded_label_encoder.inverse_transform([prediction])[0]
    # Возвращаем результат в формате JSON
    return {"predicted_emotion": predicted_emotion}


if __name__ == "__main__":
    # Запуск сервера
    uvicorn.run(app, host="0.0.0.0", port=8000)
