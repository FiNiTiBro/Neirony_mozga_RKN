import csv
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm


def load_and_resize_images(dataset_path, img_size):
    """
    Загружает и изменяет размер изображений в памяти.

    :param dataset_path: Путь к датасету.
    :param img_size: Новый размер изображений.
    :return: Массив изображений и соответствующие метки классов.
    """
    images = []
    labels = []
    image_names = []
    class_names = sorted(os.listdir(dataset_path))
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    for class_name in tqdm(class_names, desc="Загрузка и изменение размера изображений"):
        class_path = os.path.join(dataset_path, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue  # Пропустить изображения, которые не удалось загрузить
            img_resized = cv2.resize(img, img_size)
            images.append(img_resized)
            labels.append(class_to_idx[class_name])
            image_names.append(img_name)

    return np.array(images), np.array(labels), image_names, class_to_idx


def load_and_resize_test_images(dataset_path, img_size):
    """
    Загружает и изменяет размер изображений для тестового датасета.

    :param dataset_path: Путь к тестовому датасету.
    :param img_size: Новый размер изображений.
    :return: Массив изображений и имена файлов.
    """
    images = []
    image_names = []

    # Проходим по всем файлам в директории dataset_path
    for img_name in tqdm(os.listdir(dataset_path), desc="Загрузка и изменение размера изображений"):
        img_path = os.path.join(dataset_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue  # Пропускаем изображения, которые не удалось загрузить
        img_resized = cv2.resize(img, img_size)
        images.append(img_resized)
        image_names.append(img_name)

    return np.array(images), image_names

def split_dataset(images, labels, validation_size=0.2, test_size=0.2, random_state=42):
    """
    Разделяет датасет на тренировочные, валидационные и тестовые данные.

    :param images: Массив изображений.
    :param labels: Массив меток.
    :param validation_size: Доля валидационных данных от общего количества.
    :param test_size: Доля тестовых данных от общего количества.
    :param random_state: Начальное состояние генератора случайных чисел для воспроизводимости.
    :return: Кортеж (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # Сначала выделяем тестовые данные
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    # Далее выделяем валидационные данные из тренировочных
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=validation_size / (1 - test_size), random_state=random_state, stratify=y_train
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def build_model(input_shape=(320, 320, 3), num_classes=105):

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # Для классификации
    ])
    return model



def extract_features_batched(images, model, batch_size):
    """
    Извлечение признаков для изображений пакетами, чтобы снизить нагрузку на память.
    """
    features = []
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size].astype("float32") / 255.0
        batch_features = model.predict(batch_images, verbose=0)
        features.append(batch_features)
    return np.concatenate(features, axis=0)
    
def generate_submission(images, dataset_features, dataset_image_names, model, output_path="submission.csv"):
    """
    Генерация CSV с рекомендациями.

    :param images: Тестовые изображения.
    :param dataset_features: Векторы признаков обучающего датасета.
    :param dataset_image_names: Имена изображений обучающего датасета.
    :param model: Модель для извлечения признаков.
    :param output_path: Путь для сохранения CSV.
    """
    test_features = extract_features_batched(images, model, batch_size=128)

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image', 'recs'])  # Заголовки

        for idx, query_feature in enumerate(test_features):
            recs = find_similar_images(query_feature, dataset_features, dataset_image_names, top_k=10)

            writer.writerow([dataset_image_names[idx], f'"{",".join(recs)}"'])






def find_similar_images(query_features, dataset_features, image_names, top_k=10):
    distances = cdist(query_features[np.newaxis, :], dataset_features, metric='euclidean')[0]
    indices = np.argsort(distances)[:top_k]
    similar_results = [image_names[idx] for idx in indices]
    return similar_results

def load_model_from_project(X_test,y_test):
    # Пересоздание той же модели, учитывая веса и оптимайзер
    new_model = tf.keras.models.load_model('checkpoints/RKN_43-3.3887.keras')

    # Вывод архитектуры модели
    new_model.summary()

    '''# проверка новой модели
    #new_loss, new_acc = new_model.evaluate(X_test / 255.0, y_test, verbose=1)
    #print('Restored model, accuracy: {:5.2f}%'.format(100 * new_acc))'''
    return new_model

def preprocess_image(img_path, img_size):
    """
    Загрузка и предварительная обработка изображения пользователя.

    :param img_path: Путь к изображению, загруженному пользователем.
    :param img_size: Размер для изменения изображения.
    :return: Изображение в формате numpy массива с нормализацией.
    """
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Добавление batch dimension
    img_array = img_array / 255.0  # Нормализация
    return img_array

def find_similar_images_for_user_upload(images, user_img_path, dataset_features, image_names, model, top_k=10):
    """
    Поиск топ-K похожих изображений из датасета для загруженного пользователем изображения.

    :param user_img_path: Путь к изображению пользователя.
    :param dataset_features: Признаки изображений датасета.
    :param image_names: Имена изображений в датасете.
    :param model: Модель для извлечения признаков.
    :param top_k: Количество ближайших изображений для возврата.
    :return: Список имен файлов наиболее релевантных изображений.
    """
    user_img = preprocess_image(user_img_path, img_size=(320, 320))
    user_features = extract_features_batched(user_img, model, batch_size=128)  # Извлечение признаков для загруженного изображения

    # Найти похожие изображения
    similar_images = find_similar_images(user_features[0], dataset_features, image_names, top_k)

    similar_images_with_classes = []
    for img in similar_images:
        # Здесь предполагаем, что img_name имеет формат "class_name/image.jpg"
        class_name = os.path.basename(os.path.dirname(img))  # Извлекаем только имя папки-класса
        image_name = os.path.basename(img)  # Извлекаем имя файла
        similar_images_with_classes.append(f"{class_name}/{image_name}")



    return similar_images_with_classes

def compute_map_matrix(feature_db, labels_db, top_k=10):
    """
    Рассчет Mean Average Precision (mAP) с использованием матричных операций.

    :param feature_db: База данных признаков (массив размера [N, D], где N - число изображений, D - размер признаков).
    :param labels_db: Метки классов (массив размера [N]).
    :param top_k: Количество ближайших соседей, которые используются для вычислений.
    :return: Значение mAP.
    """
    # Нормализуем признаки для ускорения расстояний (опционально)
    normalized_features = feature_db / np.linalg.norm(feature_db, axis=1, keepdims=True)

    # Вычисляем расстояния между всеми парами изображений
    distances = cdist(normalized_features, normalized_features, metric='euclidean')

    # Получаем индексы ближайших соседей для каждого изображения
    # Убираем первое совпадение (само изображение, расстояние 0)
    indices = np.argsort(distances, axis=1)[:, 1:top_k + 1]

    # Формируем матрицу релевантности
    relevance_matrix = np.zeros((labels_db.shape[0], top_k), dtype=np.float32)
    for i in range(labels_db.shape[0]):
        # Для каждого изображения проверяем, какие из ближайших соседей имеют одинаковую метку
        relevance_matrix[i] = (labels_db[indices[i]] == labels_db[i]).astype(np.float32)

    # Вычисляем Average Precision для каждого изображения
    precisions = np.cumsum(relevance_matrix, axis=1) / np.arange(1, top_k + 1)
    relevant_counts = np.sum(relevance_matrix, axis=1)  # Количество релевантных соседей
    average_precisions = np.sum(precisions * relevance_matrix, axis=1) / np.maximum(relevant_counts, 1)  # Деление на максимум из count и 1

    # Вычисляем mAP как среднее значение AP только для изображений с релевантными соседями
    mean_ap = np.mean(average_precisions[relevant_counts > 0])
    return mean_ap



image_path = 'rkn_test_image.jpg' # вот тута, никита ввод данных

test_dataset_path = "neironi_mozga_rkn/test_data_rkn/test_dataset"  # Путь к исходному датасету
dataset_path= "neironi_mozga_rkn/train_data_rkn/dataset"
img_size = (320, 320)  # Новый размер
batch_size = 128

#вызов метода ввода данных
user_image = preprocess_image(image_path, img_size)

#загрузка и ресайз изображений
images, labels, image_names, class_to_idx = load_and_resize_images(dataset_path, img_size)
test_images, test_image_names = load_and_resize_test_images(test_dataset_path, img_size)

# Разделяем на тренировочные, валидационные и тестовые данные
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(images, labels, validation_size=0.2, test_size=0.2)

print(f"Тренировочные данные: {X_train.shape}, {y_train.shape}")
print(f"Валидационные данные: {X_val.shape}, {y_val.shape}")
print(f"Тестовые данные: {X_test.shape}, {y_test.shape}")

#Создание и обучение модели
#model = build_model(input_shape=img_size + (3,), num_classes=len(class_to_idx))
#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#обьявление новой модели
new_model = load_model_from_project(X_test,y_test)

'''datagen = ImageDataGenerator(rescale=1./255)  # Нормализация данных

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,  # Размер изображений
    batch_size=700,  # Размер пакета
    class_mode='sparse'  # Подходит для целых меток
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=img_size,
    batch_size=700,
    class_mode='sparse'
)

# Создаем папку, если ее еще нет
os.makedirs("checkpoints", exist_ok=True)



# Создаем коллбэк для сохранения модели
checkpoint_callback = ModelCheckpoint(
    filepath="checkpoints/RKN_{epoch:02d}-{val_loss:.4f}.keras",  # Путь для сохранения
    save_weights_only=False,
    save_best_only=False,  # Сохранять только лучшие модели
    monitor='val_loss',  # Метрика для отслеживания
    mode='min',  # Минимизация ошибки
    verbose=1
)
callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=25,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=0
    )
# Обучение модели
model.fit(
    train_generator,
    epochs=500,
    validation_data=val_generator,
    callbacks=[checkpoint_callback, callback]
)'''

# Извлечение признаков для тренировочной выборки
train_features = extract_features_batched(images, new_model, batch_size)

# Извлечение признаков для валидационной выборки
val_features = extract_features_batched(images, new_model, batch_size)

# Извлечение признаков для тестовой выборки
test_features = extract_features_batched(test_images, new_model, batch_size)

# Находим похожие изображения для юзера (Некит это для вывода)
similar_images = find_similar_images_for_user_upload(test_images, image_path, test_features, test_image_names, new_model, top_k=10)
print("10 наиболее релевантных изображений:")
for img_name in similar_images:
    print(img_name)

# Генерация CSV
generate_submission(test_images, test_features, test_image_names, new_model)
print("Файл submission.csv успешно создан!")

'''#вычисление mAP
train_map = compute_map_matrix(train_features, y_train)
print(f"Mean Average Precision (mAP) на тренировочных данных: {train_map:.4f}")

val_map = compute_map_matrix(val_features, y_val)
print(f"Mean Average Precision (mAP) на валидационной выборке: {val_map:.4f}")'''
