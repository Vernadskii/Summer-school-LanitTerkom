import numpy as np  # подключаем библиотеку
from numpy import array, dot, random
import matplotlib
import matplotlib.pyplot as plt
# X - массив входных значений
X = np.array([[i] for i in range(20)])
# Y - массив эталонных значений
Y = np.square(X)

# проверим размерность массива
x_shape = X.shape
print(x_shape)

# создадим массив со значениями смещений
arr_bias = np.ones((x_shape[0], 1), dtype=int)
#print(arr_bias)
print(arr_bias.shape)

X = np.append(X, arr_bias, axis=1)
print(X)

# W - рандомные начальные значения весов для входных значений
W = random.rand(2)
print(W)

# скорость обучения
lr = 0.00001
print(lr)
# количество итераций обучения
N = 8000

for i in range(N):
    # рандомное число от 0 до 3 для определения индекса примера из массива Х
    rndm = random.randint(20)
    # присвоение значения переменной по сгенерированному индексу
    element_x = X[rndm]
    expected = Y[rndm]
    # произведение входного значения (Х) на вес (W)
    result = dot(W, element_x)
    # вычисление ошибки
    error = expected - result
    # изменение весов
    for i in range(len(W)):
        W[i] += lr * error * element_x[i]
print(W)

x = []
# вывод результата тренировки
for element_x in X:
    result = dot(element_x, W)
    print("{}: {}".format(element_x[:1], result))
    x.append(result)

matplotlib.rcParams['figure.figsize']=(8.0, 5.0)
plt.plot(x,marker='o', color='green')
plt.plot(Y, marker='o', color='red')
plt.show()

plt.savefig("Test1")
