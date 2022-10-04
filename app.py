import numpy as np
import sys

from PIL import Image
from os import listdir
from os.path import isfile, join


def f_activation(x):
    return 2 / (1 + np.exp(-x)) - 1


def der_f_activation(x):
    return 0.5 * (1 + f_activation(x)) * (1 - f_activation(x))


def go_forward(inp):
    s = np.dot(W1, inp)
    out = np.array([f_activation(x) for x in s])

    s = np.dot(W2, out)
    y = f_activation(s)

    return y, out


def train(epoch, true):
    global W1, W2
    lmbd = 0.01
    N = 1000
    count = len(epoch)
    for i in range(N):
        index = np.random.randint(0, count)
        x = epoch[index]
        x_true = true[index]
        y, out = go_forward(x)
        e = y - x_true
        delta = e * der_f_activation(y)
        W2[0] = W2[0] - lmbd * delta * out[0]
        W2[1] = W2[1] - lmbd * delta * out[1]

        delta2 = W2 * delta * der_f_activation(out)

        W1[0, :] = W1[0, :] - np.array(x) * delta2[0] * lmbd
        W1[1, :] = W1[1, :] - np.array(x) * delta2[1] * lmbd









if __name__ == '__main__':
    epoch = []
    true_output = []
    directories = ['Latin/' + chr(i) + '/' for i in range(65, 70)]
    for dir in directories:
        print(dir)
        files = [f for f in listdir(dir) if isfile(join(dir, f))][:30]
        i = 0
        for file in files:
            print(f'{i}/{len(files)}')
            image = Image.open(f'{dir}{file}')
            image_array = np.asarray(image).tolist()
            res = []
            for x in image_array:
                res.extend(x if isinstance(x, list) else [x])
            res1 = []
            for x in res:
                res1.extend(x if isinstance(x, list) else [x])
            epoch.append(res1)
            true_output.append(-1 + (ord(dir[len(dir) - 2]) - 65) / 12.5)
            i += 1
    print(true_output)
    # W1 = np.array([list(np.random.rand(len(epoch[0]))) for i in range(10)])
    # print(W1.shape)
    W1 = np.random.randint(-7, 7, (10, 309136)) / 10
    W2 = np.random.randint(-7, 7, (10, )) / 10
    print(W1, W2)
    # print(W2.shape)


    # epoch = [(-1, -1, -1, [1, 0]), (-1, -1, 1, [2, 2]), (-1, 1, -1, [0, 0]), (-1, 1, 1, [3, 3]),
    #          (1, -1, -1, [0, 1]), (1, -1, 1, [1, 0]), (1, 1, -1, [0, 0]), (1, 1, 1, [0, 0])]

    train(epoch, true_output)

    print('Обучение завершилось')

    example = Image.open('5a31f049f39bc.png')
    # example = Image.open('5a6735484b979.png')
    example = np.asarray(example).tolist()
    res = []
    for x in example:
        res.extend(x if isinstance(x, list) else [x])
    example = []
    for x in res:
        example.extend(x if isinstance(x, list) else [x])

    print(len(example))

    print(go_forward(example))



