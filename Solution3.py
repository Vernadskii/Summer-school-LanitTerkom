import numpy as np  # подключаем библиотеку
from numpy import array, dot, random

# X - массив входных значений
X = np.arange(100)

# Y - массив эталонных значений
Y = np.square(X)

# проверим размерность массива
x_shape = X.shape
print(x_shape)

# создадим массив со значениями смещений
arr_bias = np.ones((100), dtype=int)
print(arr_bias)

X = np.append(X, arr_bias, axis=1)
print(X)

# W - рандомные начальные значения весов для входных значений
W = random.rand(3)
print(W)

# функция активации
activate = lambda sum: 0 if sum < 0 else 1

# скорость обучения
lr = 100e-1
print(lr)
# количество итераций обучения
N = 10000

for i in range(N):
    # рандомное число от 0 до 3 для определения индекса примера из массива Х
    rndm = random.randint(4)
    # присвоение значения переменной по сгенерированному индексу
    element_x = X[rndm]
    expected = Y[rndm]
    # произведение входного значения (Х) на вес (W)
    result = dot(W, element_x)
    # вычисление ошибки
    error = expected - activate(result)
    # изменение весов
    for i in range(len(W)):
        W[i] += lr * error * element_x[i]
print(W)

# вывод результата тренировки
for element_x in X:
    result = dot(element_x, W)
    print("{}: {} -> {}".format(element_x[:2], result, activate(result)))
