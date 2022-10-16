import json

import numpy as np
import sys

from PIL import Image
from os import listdir
from os.path import isfile, join


# W1 = json.load(open("weights.json", "r"))['W1']
# W1 = np.array(W1)
# # W2 = json.load(open("weights.json", "r"))['W2']
# W2 = np.array(W2)
W1 = np.random.randint(0, 10, (20, 400)) / 10
for w in W1:
    for w1 in w:
        if w1 == 0:
            w1 = 0.5
W2 = np.random.randint(0, 10, (26, 20)) / 10
for w in W2:
    for w1 in w:
        if w1 == 0:
            w1 = 0.5
N = 0

def f_activation(x):
    return 1 / (1 + np.exp(-x))


def der_f_activation(x):
    return f_activation(x) * (1 - f_activation(x))


def go_forward(inp):
    s = np.dot(W1, inp)
    out = np.array([f_activation(x) for x in s])

    s = np.dot(W2, out)
    y = f_activation(s)
    # print(y.shape)
    # print(out.shape)
    return y, out


def train(epoch, true):
    global W1, W2
    lmbd = 0.01
    N = 500
    count = len(epoch)
    i = 0
    for i in range(N):
        for k in range(len(epoch)):
            index = k
            if index % 1000 == 0:
                print(i, ' ', index)
            x = epoch[index]
            x_true = true[index]
            y, out = go_forward(x)
            e = y - x_true
            delta = e * der_f_activation(y)
            for j in range(W2.shape[0]):
                W2[j] -= lmbd * delta[j] * out

            for j in range(W1.shape[0]):
                sigma = sum(W2[:, j] * delta)
                delta2 = sigma * der_f_activation(out[j])
                W1[j] -= lmbd * delta2 * np.array(x)


if __name__ == '__main__':
    epoch = []
    true_output = []
    directories = ['Latin/' + chr(i) + '/' for i in range(65, 91)]
    for dir in directories:
        print(dir)
        files = [f for f in listdir(dir) if isfile(join(dir, f))]
        N += len(files)
        i = 0
        for file in files:
            if i % 100 == 0:
                print(f'{i}/{len(files)}')
            image = Image.open(f'{dir}{file}').resize((20, 20))
            image_array = np.asarray(image).tolist()


            res = []
            for x in image_array:
                res.extend(x if isinstance(x, list) else [x])
            res1 = []
            for x in res:
                res1.append(0 if x == [0, 0, 0, 0] else 1)


            epoch.append(res1)
            # true_output.append((ord(dir[len(dir) - 2]) - 65) / 25)
            # true_output.append(ord(dir[len(dir) - 2]) / 100)
            true_output.append(np.zeros((26, ), dtype=float))
            true_output[-1][ord(dir[len(dir) - 2]) - 65] = 1.
            i += 1

    # print(true_output)
    # print(W1.shape[0])
    # print(W2.shape[0])
    # print('нач')
    # print(W1, W2)


    # epoch = [(-1, -1, -1, [1, 0]), (-1, -1, 1, [2, 2]), (-1, 1, -1, [0, 0]), (-1, 1, 1, [3, 3]),
    #          (1, -1, -1, [0, 1]), (1, -1, 1, [1, 0]), (1, 1, -1, [0, 0]), (1, 1, 1, [0, 0])]
    print(len(epoch))
    train(epoch, true_output)
    print('после обучения')
    print(W1, W2)
    print('Обучение завершилось')

    example = Image.open('test/a.png').resize((20, 20))
    # example = Image.open('5a6735484b979.png')
    example = np.asarray(example).tolist()
    res = []
    for x in example:
        res.extend(x if isinstance(x, list) else [x])
    example = []
    for x in res:
        example.append(0 if x == [0, 0, 0, 0] else 1)



    example1 = Image.open('test/z.png').resize((20, 20))
    # example = Image.open('5a6735484b979.png')
    example1 = np.asarray(example1).tolist()
    res = []
    for x in example1:
        res.extend(x if isinstance(x, list) else [x])
    example1 = []
    for x in res:
        example1.append(0 if x == [0, 0, 0, 0] else 1)

    print('Пример:')

    print(go_forward(example)[0])
    print(go_forward(example1)[0])




    weights = {
        'W1': W1.tolist(),
        'W2': W2.tolist(),
    }

    with open('weights.json', 'w') as outfile:
        json.dump(weights, outfile, indent=4)



