# Простая полносвязная нейронная сеть
# по набору данных ирисов из модуля sklearn
# Алгоритм - стохастический градиентный спуск (sgd)

import random
import numpy as np
import matplotlib.pyplot as plt

# Значения исходных данных и гиперпараметров
INPUT_DIM = 4
OUT_DIM = 3
H_DIM = 10
NUM_EPOCHS = 400
BATCH_SIZE = 50
LEARNING_RATE = 0.0004

# Инициализация весов сети случайными числами
# равномерного распределения от 0 до 1 для двух скрытых слоев
W1 = np.random.rand(INPUT_DIM, H_DIM)
b1 = np.random.rand(1, H_DIM)
W2 = np.random.rand(H_DIM, OUT_DIM)
b2 = np.random.rand(1, OUT_DIM)

# Поднастройка инициализации весов
W1 = (W1 - 0.5) * 2 * np.sqrt(1/INPUT_DIM)
b1 = (b1 - 0.5) * 2 * np.sqrt(1/INPUT_DIM)
W2 = (W2 - 0.5) * 2 * np.sqrt(1/H_DIM)
b2 = (b2 - 0.5) * 2 * np.sqrt(1/H_DIM)


# функция измерения расстояния между вероятностными распределениями для пакета образцов
def sparse_cross_entropy_batch(z, y):
    return -np.log(np.array([z[j, y[j]] for j in range(len(y))]))


# Функция активации
def relu(t):
    return np.maximum(t, 0)


# Нормирующее преобразование вероятностей
def softmax(t):
    out = np.exp(t)
    return out / np.sum(out)


# Нормирующее преобразование для пакета образцов
def softmax_batch(t):
    out = np.exp(t)
    return out / np.sum(out, axis=1, keepdims=True)


# Вектора с ожидаемым результатом для каждого пакета образцов
def to_full_batch(y, num_classes):
    y_full = np.zeros((len(y), num_classes))
    for j, yj in enumerate(y):
        y_full[j, yj] = 1
    return y_full


# Производная функции активации
def relu_deriv(t):
    return (t >= 0).astype(float)


# Функция вероятностей
def predict(x):
    t1 = x @ W1 + b1
    h1 = relu(t1)
    t2 = h1 @ W2 + b2
    z = softmax(t2)
    return z


# Вычисление точности сети
def calc_accuracy():
    correct = 0
    for x, y in dataset:
        z = predict(x)
        y_pred = np.argmax(z)
        if y_pred == y:
            correct += 1
    acc = correct / len(dataset)
    return acc


# Импорт датасета
from sklearn import datasets
iris = datasets.load_iris()
dataset = [(iris.data[i][None, ...], iris.target[i]) for i in range(len(iris.target))]

loss_arr = []

# Цикл итераций
for ep in range(NUM_EPOCHS):
    random.shuffle(dataset)
    for i in range(len(dataset) // BATCH_SIZE):
        batch_x, batch_y = zip(*dataset[i*BATCH_SIZE : i*BATCH_SIZE+BATCH_SIZE])
        x = np.concatenate(batch_x)
        y = np.array(batch_y)

        # forward Прямое распространение
        t1 = x @ W1 + b1
        h1 = relu(t1)
        t2 = h1 @ W2 + b2
        z = softmax_batch(t2)  # Вектор из вероятностей
        E = np.sum(sparse_cross_entropy_batch(z, y))  # Суммарная ошибка по группе образцов

        # Backward Обратное распространение, вычисление градиента
        y_full = to_full_batch(y, OUT_DIM)
        dE_dt2 = z - y_full
        dE_dW2 = h1.T @ dE_dt2
        dE_db2 = np.sum(dE_dt2, axis=0, keepdims=True)
        dE_dh1 = dE_dt2 @ W2.T
        dE_dt1 = dE_dh1 * relu_deriv(t1)
        dE_dW1 = x.T @ dE_dt1
        dE_db1 = np.sum(dE_dt1, axis=0, keepdims=True)

        # Update Обновление весов
        W1 = W1 - LEARNING_RATE * dE_dW1
        b1 = b1 - LEARNING_RATE * dE_db1
        W2 = W2 - LEARNING_RATE * dE_dW2
        b2 = b2 - LEARNING_RATE * dE_db2

        loss_arr.append(E)


accuracy = calc_accuracy()
print("Accuracy: ", accuracy)

# График зависимости ошибки от итераций
plt.plot(loss_arr)
plt.show()

