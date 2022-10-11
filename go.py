import json

import numpy as np
from PIL import Image

from app import go_forward
W1 = json.load(open("weights.json", "r"))['W1']
W2 = json.load(open("weights.json", "r"))['W2']


def f_activation(x):
    return 2 / (1 + np.exp(-x)) - 1


def der_f_activation(x):
    return 0.5 * (1 - x) * (1 + x)


def go_forward(inp):
    s = np.dot(W1, inp)
    out = np.array([f_activation(x) for x in s])

    s = np.dot(W2, out)
    y = f_activation(s)

    return y, out



if __name__ == "__main__":
    example = Image.open('o.png').resize((20, 20))
    example = np.asarray(example).tolist()
    res = []
    for x in example:
        res.extend(x if isinstance(x, list) else [x])
    example = []
    for x in res:
        example.append(0 if x == [0, 0, 0, 0] else 1)

    print('Пример:')

    print(go_forward(example)[0])
