import numpy as np

def f(x):
    return 1/(1 + np.exp(-x))

def df(x):
    return f(x)*(1 - f(x))


W1 = np.array([[-0.2, 0.3, -0.4], [0.1, -0.3, -0.4]])
W2 = np.array([[0.2, 0.3], [0.2, -0.3]])


def go_forward(inp):
    sum = np.dot(W1, inp)
    out = np.array([f(x) for x in sum])

    sum = np.dot(W2, out)
    y = f(sum)
    return (y, out)

def train(epoch):
    global W2, W1
    lmd = 0.01          # шаг обучения
    N = 10          # число итераций при обучении
    count = len(epoch)
    for k in range(N):
        x = epoch[np.random.randint(0, count)]  # случайных выбор входного сигнала из обучающей выборки
        y, out = go_forward(x[0])             # прямой проход по НС и вычисление выходных значений нейронов
        e = y - x[1]                           # ошибка
        delta = e * df(y)
        for j in range(W2.shape[0]):
            W2[j, :] = W2[j, :] - lmd * delta[j] * out

        for j in range(W1.shape[0]):
            sigma = sum(W2[:, j] * delta)
            delta2 = sigma * df(out[j])
            W1[j, :] = W1[j, :] - np.array(x[0]) * delta2 * lmd

# обучающая выборка (она же полная выборка)

epoch = [[[-1, -1, -1], [1, 0]], #-1
         [[-1, -1, 1], [0, 1]], #1
         [[-1, 1, -1], [1, 0]], #-1
         [[-1, 1, 1], [0, 1]], #1
         [[1, -1, -1], [1, 0]], #-1
         [[1, -1, 1], [0, 1]], #1
         [[1, 1, -1], [1, 0]], #-1
         [[1, 1, 1], [1, 0]]] #-1

# epoch = [[[0, 0, 0], [1, 0]], #-1
#          [[0, 0, 1], [1, 0]], #1
#          [[0, 1, 0], [1, 0]], #-1
#          [[0, 1, 1], [0, 1]], #1
#          [[1, 0, 0], [1, 0]], #-1
#          [[1, 1, 0], [0, 1]], #1
#          [[1, 0, 1], [0, 1]], #-1
#          [[1, 1, 1], [0, 1]]] #-1

train(epoch)        # запуск обучения сети

# проверка полученных результатов
for x in epoch:
    y, out = go_forward(x[0])
    print(f"Выходное значение НС: {y} => {x[-1]}")